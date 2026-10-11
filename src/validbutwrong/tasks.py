"""Tasks: where items, prompts, gold answers and per-arm answer extraction live.

A task supplies everything that is task-specific. The arms, engines, metrics and
report never look inside an item; they only call the task's methods. Adding your
own task means writing one of these classes (see ``Task``).
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .schema import bfcl_to_jsonschema


@dataclass
class Item:
    id: str
    prompt: str            # the user-facing problem text
    gold: Any              # task-specific gold
    meta: dict = field(default_factory=dict)


class Task:
    """Interface every task implements.

    ``schemas``          name -> JSON Schema for the contract arms (prompt_json / constrained).
    ``answer_schema``    which schema name reason_then_constrain packages into.
    ``free_prompt``      instruction for the unconstrained arm (free text).
    ``json_prompt``      instruction for the contract arms, given a schema.
    ``extract_free``     free text -> answer (or None).
    ``extract_json``     parsed JSON object -> answer (or None).
    ``is_correct``       (answer, item) -> bool.
    """

    name = "task"
    correctness_label = "answer accuracy"
    schemas: dict[str, dict] = {}
    answer_schema = "answer"

    def items(self, n: int | None = None) -> list[Item]:
        raise NotImplementedError

    def schemas_for(self, item: Item) -> dict[str, dict]:
        return self.schemas

    def free_prompt(self, item: Item) -> str:
        raise NotImplementedError

    def json_prompt(self, item: Item, schema: dict) -> str:
        raise NotImplementedError

    def package_prompt(self, item: Item, schema: dict) -> str:
        return ("Now give your final answer as ONLY a JSON object matching this JSON Schema, "
                "with no other text:\n" + json.dumps(schema))

    def extract_free(self, text: str, item: Item) -> Any:
        raise NotImplementedError

    def extract_json(self, obj: Any, item: Item) -> Any:
        raise NotImplementedError

    def is_correct(self, answer: Any, item: Item) -> bool:
        raise NotImplementedError


# --------------------------------------------------------------------------- GSM8K

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _to_number(x: Any) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        m = _NUM.findall(x.replace("$", ""))
        if m:
            try:
                return float(m[-1].replace(",", ""))
            except ValueError:
                return None
    return None


class GSM8K(Task):
    """GSM8K test split (openai/grade-school-math, MIT). Gold = the number after '####'."""

    name = "gsm8k"
    correctness_label = "answer accuracy"
    schemas = {
        "answer": {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
        "reasoning": {
            "type": "object",
            "properties": {"reasoning": {"type": "string"}, "answer": {"type": "integer"}},
            "required": ["reasoning", "answer"],
            "additionalProperties": False,
        },
    }
    answer_schema = "answer"

    def __init__(self, path: str):
        self.path = path

    def items(self, n: int | None = None) -> list[Item]:
        out = []
        with open(self.path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if n is not None and i >= n:
                    break
                row = json.loads(line)
                gold = float(row["answer"].split("####")[-1].strip().replace(",", ""))
                out.append(Item(id=f"gsm8k_test_{i}", prompt=row["question"], gold=gold))
        return out

    def free_prompt(self, item: Item) -> str:
        return (f"{item.prompt}\n\nSolve the problem. Think step by step, then finish with a "
                "final line of the form 'Answer: <number>'.")

    def json_prompt(self, item: Item, schema: dict) -> str:
        return (f"{item.prompt}\n\nSolve the problem. Respond with ONLY a JSON object matching "
                f"this JSON Schema, with no other text:\n{json.dumps(schema)}")

    def extract_free(self, text: str, item: Item) -> Any:
        # Priority: an explicit "Answer: N" line, then LaTeX \boxed{N} (Qwen's habit), then the
        # last number in the text. \boxed{} support was added after inspecting real outputs:
        # without it, "\boxed{1400} bananas to last for 2 months" was read as 2.
        m = re.findall(r"[Aa]nswer\s*[:=]\s*\**\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)", text)
        if m:
            return _to_number(m[-1])
        boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
        if boxed and _to_number(boxed[-1]) is not None:
            return _to_number(boxed[-1])
        return _to_number(text)  # fall back to the last number in the text

    def extract_json(self, obj: Any, item: Item) -> Any:
        if isinstance(obj, dict) and "answer" in obj:
            return _to_number(obj["answer"])
        return None

    def is_correct(self, answer: Any, item: Item) -> bool:
        return answer is not None and abs(answer - item.gold) < 1e-6


# --------------------------------------------------------------------------- BFCL

def _norm_str(s: str) -> str:
    # BFCL-style string normalisation: case-insensitive, ignore spaces and common punctuation.
    return re.sub(r"[ ,./\-_*^]", "", s.lower())


def _value_matches(value: Any, allowed: Any) -> bool:
    if isinstance(allowed, str):
        return isinstance(value, str) and _norm_str(value) == _norm_str(allowed)
    if isinstance(allowed, bool) or isinstance(value, bool):
        return value is allowed
    if isinstance(allowed, (int, float)):
        return isinstance(value, (int, float)) and float(value) == float(allowed)
    if isinstance(allowed, list):
        return (isinstance(value, (list, tuple)) and len(value) == len(allowed)
                and all(_value_matches(v, a) for v, a in zip(value, allowed)))
    if isinstance(allowed, dict):
        # nested dict in possible_answer: each key maps to a list of allowed values
        return isinstance(value, dict) and ast_match(value, allowed, {"properties": dict.fromkeys(allowed)})
    return value == allowed


def ast_match(args: dict | None, gold: dict, params: dict) -> bool:
    """Simplified BFCL-style AST match of one call's arguments against possible_answer.

    Rules: every gold parameter must be present unless its allowed list contains "" (optional);
    a present value must match one allowed value; no parameter outside the function's schema.
    This mirrors the official checker's structure; it is NOT the official checker.
    """
    if not isinstance(args, dict):
        return False
    known = set(params.get("properties", {}))
    if any(k not in known for k in args):
        return False
    for k, allowed in gold.items():
        if k not in args:
            if "" in allowed:
                continue
            return False
        if not any(_value_matches(args[k], a) for a in allowed if a != ""):
            return False
    return True


class BFCLSimple(Task):
    """BFCL v3 'simple' category (gorilla-llm/Berkeley-Function-Calling-Leaderboard, apache-2.0).

    One function per item. The contract arms emit the call's ARGUMENTS as a JSON object; the
    schema is the function's parameter schema converted to JSON Schema. The unconstrained arm
    writes a Python-syntax call, parsed with ``ast``. Scored by AST match, never by execution.
    """

    name = "bfcl-simple"
    correctness_label = "AST-match accuracy"
    answer_schema = "arguments"

    def __init__(self, questions_path: str, answers_path: str):
        self.questions_path = questions_path
        self.answers_path = answers_path

    def items(self, n: int | None = None) -> list[Item]:
        with open(self.answers_path, encoding="utf-8") as f:
            gold = {r["id"]: r["ground_truth"][0] for r in map(json.loads, f)}
        out = []
        with open(self.questions_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if n is not None and i >= n:
                    break
                row = json.loads(line)
                fn = row["function"][0]
                (gname, gargs), = gold[row["id"]].items()
                out.append(Item(id=row["id"], prompt=row["question"][0][-1]["content"],
                                gold=gargs, meta={"function": fn, "gold_name": gname}))
        return out

    def schemas_for(self, item: Item) -> dict[str, dict]:
        return {"arguments": bfcl_to_jsonschema(item.meta["function"]["parameters"])}

    def _fn_doc(self, item: Item) -> str:
        return json.dumps(item.meta["function"])

    def free_prompt(self, item: Item) -> str:
        fn = item.meta["function"]["name"]
        return (f"You can call this function:\n{self._fn_doc(item)}\n\nRequest: {item.prompt}\n\n"
                f"Write the single call to `{fn}` that fulfils the request, in Python syntax, "
                f"e.g. {fn}(arg=value). Write only the call.")

    def json_prompt(self, item: Item, schema: dict) -> str:
        fn = item.meta["function"]["name"]
        return (f"You can call this function:\n{self._fn_doc(item)}\n\nRequest: {item.prompt}\n\n"
                f"Give the arguments for the call to `{fn}` as ONLY a JSON object matching this "
                f"JSON Schema, with no other text:\n{json.dumps(schema)}")

    def package_prompt(self, item: Item, schema: dict) -> str:
        return ("Now give the arguments of that call as ONLY a JSON object matching this JSON "
                "Schema, with no other text:\n" + json.dumps(schema))

    def extract_free(self, text: str, item: Item) -> Any:
        m = re.search(r"[\w.]+\s*\(.*\)", text.replace("\n", " "))
        if not m:
            return None
        try:
            call = ast.parse(m.group(0).strip(), mode="eval").body
        except SyntaxError:
            return None
        if not isinstance(call, ast.Call):
            return None
        # Positional arguments are mapped by the function's declared parameter order. That is
        # MORE lenient than BFCL's checker, deliberately: being charitable to the free-text
        # baseline keeps any "the constraint helps" result conservative.
        order = list(item.meta["function"]["parameters"].get("properties", {}))
        try:
            args = {order[i]: ast.literal_eval(a) for i, a in enumerate(call.args) if i < len(order)}
            args.update({kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg})
            return args
        except (ValueError, SyntaxError):
            return None

    def extract_json(self, obj: Any, item: Item) -> Any:
        return obj if isinstance(obj, dict) else None

    def is_correct(self, answer: Any, item: Item) -> bool:
        return ast_match(answer, item.gold, self.schemas_for(item)["arguments"])
