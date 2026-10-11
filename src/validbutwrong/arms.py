"""The four decoding arms. Generation only; scoring happens later from the stored text.

For each item and each schema S the task offers:

* ``unconstrained``                free text, answer extracted by the task's rule
* ``prompt_json[S]``               asked for JSON matching S in the prompt, NO grammar
* ``constrained[S]``               the SAME prompt as prompt_json[S], grammar-enforced
* ``reason_then_constrain[S*]``    free reasoning, then a constrained packaging turn into the
                                   task's answer schema S* (CRANE-style "reason free, constrain
                                   late"). Its free phase IS the unconstrained generation (same
                                   prompt, greedy decoding), so it is reused rather than re-run.

prompt_json[S] vs constrained[S] differ ONLY in whether the grammar is enforced: that pair is
the headline comparison. unconstrained vs an answer-only constrained arm also removes the
reasoning text, so it mixes two effects and is never the headline.
"""

from __future__ import annotations

from .engines import Engine, UnsupportedSchema
from .tasks import Item, Task

MAX_TOKENS = {"free": 384, "json": 384, "package": 384}  # package was 48: truncated BFCL argument lists


def arm_keys(task: Task, item: Item) -> list[str]:
    keys = ["unconstrained"]
    for s in task.schemas_for(item):
        keys += [f"prompt_json[{s}]", f"constrained[{s}]"]
    keys.append(f"reason_then_constrain[{task.answer_schema}]")
    return keys


def _rec(g, **extra) -> dict:
    return {"text": g.text, "seconds": round(g.seconds, 3),
            "compile_seconds": round(g.compile_seconds, 5), "n_tokens": g.n_tokens, **extra}


def run_item(engine: Engine, task: Task, item: Item, max_tokens: dict | None = None) -> dict:
    mt = {**MAX_TOKENS, **(max_tokens or {})}
    schemas = task.schemas_for(item)
    arms: dict[str, dict] = {}

    free_msgs = [{"role": "user", "content": task.free_prompt(item)}]
    free = engine.generate(free_msgs, None, mt["free"])
    arms["unconstrained"] = _rec(free)

    for s, schema in schemas.items():
        msgs = [{"role": "user", "content": task.json_prompt(item, schema)}]
        arms[f"prompt_json[{s}]"] = _rec(engine.generate(msgs, None, mt["json"]))
        arms[f"constrained[{s}]"] = _constrained(engine, msgs, schema, mt["json"])

    s_ans = task.answer_schema
    pkg_msgs = free_msgs + [{"role": "assistant", "content": free.text},
                            {"role": "user", "content": task.package_prompt(item, schemas[s_ans])}]
    arms[f"reason_then_constrain[{s_ans}]"] = _constrained(engine, pkg_msgs, schemas[s_ans], mt["package"],
                                                           reuses="unconstrained")
    return {"id": item.id, "arms": arms}


def _constrained(engine: Engine, msgs: list[dict], schema: dict, max_tokens: int, **extra) -> dict:
    """A schema the BACKEND cannot compile is a backend limitation, not a model error: record it,
    so scoring can exclude the item (and report how many) instead of scoring the model as wrong."""
    try:
        return _rec(engine.generate(msgs, schema, max_tokens), **extra)
    except UnsupportedSchema as e:
        return {"text": "", "unsupported_schema": str(e).splitlines()[0][:200], **extra}
