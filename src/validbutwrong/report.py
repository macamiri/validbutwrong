"""Build the report: JSON (machine), markdown (human), SVG quadrant chart (README/CI artifact)."""

from __future__ import annotations

from html import escape
from typing import Any

from . import metrics as M
from .tasks import Item, Task


def comparisons(keys: list[str]) -> list[tuple[str, str, str]]:
    """(from, to, role). The headline pairs toggle ONLY the grammar at a fixed output contract."""
    out = []
    for k in keys:
        if k.startswith("constrained["):
            out.append((k.replace("constrained[", "prompt_json[", 1), k, "headline: grammar off -> on, same contract"))
    for k in keys:
        if k.startswith("constrained[") or k.startswith("reason_then_constrain["):
            out.append(("unconstrained", k, "secondary: also changes whether the model reasons in text (confounded)"))
    return out


def build(task: Task, items: list[Item], records: list[dict], meta: dict, mode: str) -> dict[str, Any]:
    excluded = M.excluded_ids(records)
    scored = [it for it in items if it.id not in set(excluded)]
    states = M.score_run(task, items, records)
    keys = list(states)
    return {
        "task": task.name, "correctness_label": task.correctness_label, "n": len(scored), "mode": mode,
        "excluded": {"n": len(excluded), "ids": excluded,
                     "reason": "the decoding backend could not compile the schema"},
        "meta": meta,
        "arms": {k: M.arm_metrics(states[k]) for k in keys},
        "comparisons": [{**M.compare(states[a], states[b], a, b), "role": role}
                        for a, b, role in comparisons(keys) if a in states and b in states],
        "ledger": M.ledger(scored, states),
    }


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _cell(m: dict | None) -> str:
    if m is None:
        return "n/a"
    lo, hi = m["ci95"]
    return f"{_pct(m['rate'])} [{_pct(lo)}, {_pct(hi)}]"


def verdict_phrase(d: dict, n: int) -> str:
    lo, hi = d["ci95"]
    rng = f"95% CI [{100 * lo:+.1f}, {100 * hi:+.1f}] pts"
    if not d["excludes_0"]:
        return f"no detectable effect at N={n} ({rng})"
    return f"{100 * d['delta']:+.1f} pts ({rng})"


def markdown(r: dict) -> str:
    n = r["n"]
    meta = r["meta"]
    lines = [f"# validbutwrong report: {r['task']} (N={n}, {r['mode']})", ""]
    if r["mode"] == "replay":
        lines.append(f"Scored from a committed replay generated {meta.get('started', '?')} in "
                     f"{meta.get('wall_seconds', '?')} s on {meta.get('hardware', {}).get('cpu', '?')} "
                     f"({meta.get('engine', {}).get('engine')}, {meta.get('engine', {}).get('model_file')}). "
                     "Numbers are REPLAYED, not regenerated; use `--live` to regenerate.")
        lines.append("")
    if r["excluded"]["n"]:
        lines += [f"**{r['excluded']['n']} item(s) excluded** from every arm ({r['excluded']['reason']}): "
                  + ", ".join(r["excluded"]["ids"]), ""]
    lines += ["## Metrics per arm (rate over all N items, 95% bootstrap CI)", "",
              f"| arm | schema validity | {r['correctness_label']} | executable accuracy | wrong-but-valid |",
              "|---|---|---|---|---|"]
    for k, a in r["arms"].items():
        lines.append(f"| `{k}` | {_cell(a['schema_validity'])} | {_cell(a['answer_accuracy'])} | "
                     f"not computed in this run (task has no execution check) | {_cell(a['wrong_but_valid'])} |")
    lines += ["", "## Paired comparisons (same items)", ""]
    for c in r["comparisons"]:
        f = c["flips"]
        lines.append(f"**`{c['from']}` → `{c['to']}`** ({c['role']})")
        for m, d in c["deltas"].items():
            lines.append(f"- {m.replace('_', ' ')}: {verdict_phrase(d, n)}")
        lines.append(f"- per-item flips: {f['right_to_wrong']} right→wrong, {f['wrong_to_right']} wrong→right")
        lines.append("")
    widths = [(k, a["answer_accuracy"]["ci95"][1] - a["answer_accuracy"]["ci95"][0]) for k, a in r["arms"].items()]
    lines.append("CI widths (answer accuracy): " + ", ".join(f"`{k}` ±{50 * w:.1f} pts" for k, w in widths))
    lines.append("")
    return "\n".join(lines)


def quadrant_svg(r: dict) -> str:
    """One 2x2 panel per contract arm: rows valid/invalid, columns right/wrong. Zero dependencies."""
    arms = [(k, a["quadrant"]) for k, a in r["arms"].items() if a["quadrant"]]
    cell, gap, pad_top, pad_left = 64, 34, 64, 70
    panel_w = 2 * cell + 20
    width = pad_left + len(arms) * (panel_w + gap)
    height = pad_top + 2 * cell + 64
    n = r["n"]
    ink, muted, grid, accent = "#1f2328", "#59636e", "#d0d7de", "#cf222e"
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif">',
           f'<rect width="100%" height="100%" fill="#ffffff"/>',
           f'<text x="{pad_left}" y="22" font-size="14" font-weight="600" fill="{ink}">'
           f'{escape(r["task"])}: validity × correctness per arm (N={n})</text>',
           f'<text x="{pad_left}" y="40" font-size="11" fill="{muted}">red cell = valid but wrong: '
           f'the answers a 100%-valid dashboard hides</text>',
           f'<text x="{pad_left - 8}" y="{pad_top + cell / 2 + 4}" font-size="11" fill="{muted}" text-anchor="end">valid</text>',
           f'<text x="{pad_left - 8}" y="{pad_top + 1.5 * cell + 4}" font-size="11" fill="{muted}" text-anchor="end">invalid</text>']
    for i, (k, q) in enumerate(arms):
        x0 = pad_left + i * (panel_w + gap)
        for row, v in enumerate(("valid", "invalid")):
            for col, c in enumerate(("right", "wrong")):
                cnt = q[f"{v}_{c}"]
                share = cnt / n if n else 0
                hot = v == "valid" and c == "wrong"
                fill = accent if hot else "#57606a"
                x, y = x0 + col * cell, pad_top + row * cell
                out.append(f'<rect x="{x}" y="{y}" width="{cell - 2}" height="{cell - 2}" fill="{fill}" '
                           f'fill-opacity="{0.08 + 0.82 * share:.3f}" stroke="{grid}"/>')
                tc = "#ffffff" if share > 0.45 else ink
                out.append(f'<text x="{x + cell / 2 - 1}" y="{y + cell / 2 + 5}" font-size="14" '
                           f'font-weight="{600 if hot else 400}" text-anchor="middle" fill="{tc}">{cnt}</text>')
        out.append(f'<text x="{x0 + cell / 2}" y="{pad_top - 8}" font-size="10" fill="{muted}" text-anchor="middle">right</text>')
        out.append(f'<text x="{x0 + 1.5 * cell}" y="{pad_top - 8}" font-size="10" fill="{muted}" text-anchor="middle">wrong</text>')
        out.append(f'<text x="{x0 + cell}" y="{pad_top + 2 * cell + 18}" font-size="11" fill="{ink}" '
                   f'text-anchor="middle">{escape(k)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def gate(r: dict, threshold: float | None) -> list[str]:
    """Arms whose wrong-but-valid rate exceeds the threshold (the CI exit-code gate)."""
    if threshold is None:
        return []
    return [k for k, a in r["arms"].items()
            if a["wrong_but_valid"] is not None and k.startswith("constrained[")
            and a["wrong_but_valid"]["rate"] > threshold]
