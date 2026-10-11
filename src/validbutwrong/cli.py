"""validbutwrong CLI.

    validbutwrong run                         # re-scores BOTH committed replays (bfcl-simple, gsm8k) in seconds
    validbutwrong run --task bfcl-simple      # one task
    validbutwrong run --live --n 20           # opt-in regeneration (needs [live]; downloads the ~491 MB GGUF)
    validbutwrong run --fail-if-wrong-valid-above 0.2   # exit 1 if a constrained arm exceeds it

Exit codes: 0 ok · 1 gate tripped · 2 usage/setup error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import urllib.request
from importlib import resources

from . import report as R
from .run import generate, read_replay
from .tasks import BFCLSimple, GSM8K, Task

MODEL = {
    "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    "revision": "9217f5db79a29953eb74d5343926648285ec7e67",
    "file": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "sha256": "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db",
}


# Default = REPLAY for every task (operator ruling 2026-10-11); --live is opt-in. The pre-registered
# N_live measurements (10-min CPU budget, Apple M1, 4 threads) are kept as documentation of what a
# live run costs: gsm8k N_live = 37, bfcl-simple N_live = 104. See README "Regenerate the numbers".
TASKS = ["bfcl-simple", "gsm8k"]  # order of `--task all` (the default)


def _data(name: str) -> str:
    return str(resources.files("validbutwrong") / "data" / name)


def make_task(name: str) -> Task:
    if name == "gsm8k":
        return GSM8K(_data("gsm8k_test_first200.jsonl"))
    if name == "bfcl-simple":
        return BFCLSimple(_data("bfcl_v3_simple.jsonl"), _data("bfcl_v3_simple_possible_answer.jsonl"))
    raise SystemExit(f"unknown task {name!r}")


def bundled_replay(task: str) -> str | None:
    p = resources.files("validbutwrong") / "replays" / f"{task}.jsonl"
    return str(p) if p.is_file() else None


def fetch_model(cache_dir: str) -> str:
    path = os.path.join(cache_dir, MODEL["file"])
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        url = (f"https://huggingface.co/{MODEL['repo']}/resolve/{MODEL['revision']}/{MODEL['file']}")
        print(f"downloading {url} (~491 MB) ...", file=sys.stderr)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != MODEL["sha256"]:
        raise SystemExit(f"sha256 mismatch for {path}; delete it and retry")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="validbutwrong", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="re-score the committed replays (default) or regenerate with --live")
    r.add_argument("--task", default="all", choices=["all", *TASKS],
                   help="default 'all' = bfcl-simple then gsm8k")
    r.add_argument("--live", action="store_true", help="regenerate (needs the [live] extra + a GGUF)")
    r.add_argument("--n", type=int, default=None, help="items (live default 20; replay default: all 200)")
    r.add_argument("--model", help="path to a chat GGUF (default: the pinned Qwen2.5-0.5B Q4_K_M)")
    r.add_argument("--engine", default="outlines", choices=["outlines", "llamacpp"])
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--replay", help="score this replay file instead of the bundled one (single --task)")
    r.add_argument("--out", default="validbutwrong-out", help="output directory (one subdir per task for 'all')")
    r.add_argument("--fail-if-wrong-valid-above", type=float, default=None, metavar="RATE",
                   help="exit 1 if any constrained arm's wrong-but-valid rate exceeds RATE (0-1)")
    a = ap.parse_args(argv)

    if a.replay and a.task == "all":
        print("--replay needs a single --task", file=sys.stderr)
        return 2
    if a.live:
        try:
            import llama_cpp, outlines  # noqa: F401
        except ImportError:
            print("--live needs: pip install 'validbutwrong[live]'", file=sys.stderr)
            return 2
    names = TASKS if a.task == "all" else [a.task]
    tripped: list[str] = []
    for name in names:
        out = os.path.join(a.out, name) if a.task == "all" else a.out
        rc = _run_one(a, name, out)
        if rc == 2:
            return 2
        tripped += rc
    if tripped:
        print(f"GATE: wrong-but-valid above {a.fail_if_wrong_valid_above} in: {', '.join(tripped)}",
              file=sys.stderr)
        return 1
    return 0


def _run_one(a: argparse.Namespace, name: str, out: str) -> list[str] | int:
    task = make_task(name)
    os.makedirs(out, exist_ok=True)
    if a.live:
        model = a.model or fetch_model(os.path.expanduser("~/.cache/validbutwrong"))
        replay = os.path.join(out, f"{name}-live.jsonl")
        generate(task, model, replay, a.n or 20, engine=a.engine, n_threads=a.threads,
                 model_source=None if a.model else f"{MODEL['repo']}@{MODEL['revision']}/{MODEL['file']}")
        mode = "live"
    else:
        replay = a.replay or bundled_replay(name)
        if replay is None:
            print(f"no replay bundled for {name}; run with --live", file=sys.stderr)
            return 2
        mode = "replay"

    meta, records = read_replay(replay)
    if a.n and not a.live:
        records = records[: a.n]
    by_id = {rec["id"] for rec in records}
    items = [it for it in task.items() if it.id in by_id]
    rep = R.build(task, items, records, meta, mode)
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    md = R.markdown(rep)
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(out, "quadrants.svg"), "w", encoding="utf-8") as f:
        f.write(R.quadrant_svg(rep))
    with open(os.path.join(out, "ledger.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rep["ledger"][0]))
        w.writeheader()
        w.writerows(rep["ledger"])
    print(md)
    print(f"wrote {out}/report.json, report.md, quadrants.svg, ledger.csv\n")
    return [f"{name}:{k}" for k in R.gate(rep, a.fail_if_wrong_valid_above)]


if __name__ == "__main__":
    sys.exit(main())
