"""Scoring from stored text: per-item states, metrics, transitions, paired bootstrap.

Every metric is a rate over ALL N items of the run (never over the valid subset), so the
arms are compared on the same denominator:

* schema validity        output contains a JSON object that validates against the arm's schema
* answer accuracy        the extracted answer is right (task-defined; AST match for BFCL)
* executable accuracy    only for tasks that supply an execution check; otherwise "not computed"
* wrong-but-valid rate   valid AND wrong: the share that "100% valid" hides

``unconstrained`` has no output contract, so its validity is "n/a", not 0.
"""

from __future__ import annotations

import random
from typing import Any

from .schema import first_json_object, is_valid
from .tasks import Item, Task

N_BOOT = 2000
SEED = 42


def score_arm(task: Task, item: Item, key: str, text: str) -> dict:
    if key == "unconstrained":
        ans = task.extract_free(text, item)
        return {"valid": None, "correct": task.is_correct(ans, item), "answer": ans}
    schema_name = key[key.index("[") + 1:-1]
    schema = task.schemas_for(item)[schema_name]
    obj = first_json_object(text)
    ans = task.extract_json(obj, item)
    return {"valid": is_valid(obj, schema), "correct": task.is_correct(ans, item), "answer": ans}


def excluded_ids(records: list[dict]) -> list[str]:
    """Items where the backend could not compile some arm's schema. They are dropped from EVERY
    arm (keeping comparisons paired) and their count is reported, never silently."""
    return [r["id"] for r in records if any("unsupported_schema" in a for a in r["arms"].values())]


def score_run(task: Task, items: list[Item], records: list[dict]) -> dict[str, list[dict]]:
    """-> {arm_key: [state per scored item, in item order]} (excluded items are skipped)"""
    by_id = {r["id"]: r for r in records}
    skip = set(excluded_ids(records))
    states: dict[str, list[dict]] = {}
    for it in items:
        if it.id in skip:
            continue
        for key, arm in by_id[it.id]["arms"].items():
            states.setdefault(key, []).append(score_arm(task, it, key, arm["text"]))
    return states


def _rate(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _indicators(states: list[dict]) -> dict[str, list[float] | None]:
    valid = None if states[0]["valid"] is None else [float(s["valid"]) for s in states]
    correct = [float(s["correct"]) for s in states]
    wbv = None if valid is None else [float(s["valid"] and not s["correct"]) for s in states]
    return {"schema_validity": valid, "answer_accuracy": correct, "wrong_but_valid": wbv}


def bootstrap_ci(xs: list[float], ys: list[float] | None = None, n_boot: int = N_BOOT,
                 seed: int = SEED) -> tuple[float, float]:
    """95% percentile CI of mean(xs), or of the PAIRED difference mean(ys - xs)."""
    rng = random.Random(seed)
    n = len(xs)
    diffs = xs if ys is None else [y - x for x, y in zip(xs, ys)]
    stats = sorted(_rate([diffs[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    return stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot) - 1]


def quadrant(states: list[dict]) -> dict[str, int] | None:
    if states[0]["valid"] is None:
        return None
    q = {"valid_right": 0, "valid_wrong": 0, "invalid_right": 0, "invalid_wrong": 0}
    for s in states:
        q[f"{'valid' if s['valid'] else 'invalid'}_{'right' if s['correct'] else 'wrong'}"] += 1
    return q


def arm_metrics(states: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(states)}
    for name, xs in _indicators(states).items():
        out[name] = None if xs is None else {"rate": _rate(xs), "ci95": bootstrap_ci(xs)}
    out["executable_accuracy"] = None  # filled only by tasks with an execution check
    out["quadrant"] = quadrant(states)
    return out


def _state(s: dict) -> str:
    v = "n/a" if s["valid"] is None else ("valid" if s["valid"] else "invalid")
    return f"{v}/{'right' if s['correct'] else 'wrong'}"


def compare(a: list[dict], b: list[dict], a_key: str, b_key: str) -> dict[str, Any]:
    """Paired comparison a -> b on the same items: transitions + CI on each metric difference."""
    ia, ib = _indicators(a), _indicators(b)
    diffs = {}
    for m in ("schema_validity", "answer_accuracy", "wrong_but_valid"):
        if ia[m] is None or ib[m] is None:
            continue
        lo, hi = bootstrap_ci(ia[m], ib[m])
        diffs[m] = {"delta": _rate(ib[m]) - _rate(ia[m]), "ci95": (lo, hi), "excludes_0": lo > 0 or hi < 0}
    trans: dict[str, int] = {}
    for sa, sb in zip(a, b):
        k = f"{_state(sa)} -> {_state(sb)}"
        trans[k] = trans.get(k, 0) + 1
    right_to_wrong = sum(1 for sa, sb in zip(a, b) if sa["correct"] and not sb["correct"])
    wrong_to_right = sum(1 for sa, sb in zip(a, b) if not sa["correct"] and sb["correct"])
    return {"from": a_key, "to": b_key, "n": len(a), "deltas": diffs, "transitions": trans,
            "flips": {"right_to_wrong": right_to_wrong, "wrong_to_right": wrong_to_right}}


def ledger(items: list[Item], states: dict[str, list[dict]]) -> list[dict]:
    """Per-item transition ledger: every item's (valid?, right?) state in every arm.
    ``items`` must be the scored items (excluded ones removed), in the same order as ``states``."""
    return [{"id": it.id, **{k: _state(v[i]) for k, v in states.items()}} for i, it in enumerate(items)]
