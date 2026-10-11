"""Live generation: run every item through every arm and write a replay file (JSONL).

Line 1 is ``{"meta": {...}}`` (model file + sha256, backend versions, hardware, timings);
each following line is one item's generations. Scoring never needs the model: ``score``
re-reads this file, so a committed replay re-scores in seconds.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from importlib import metadata

from .arms import MAX_TOKENS, run_item
from .engines import ENGINES
from .tasks import Task


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hardware(n_threads: int) -> dict:
    cpu = platform.processor()
    if platform.system() == "Darwin":
        try:
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True).stdout.strip() or cpu
        except OSError:
            pass
    return {"cpu": cpu, "machine": platform.machine(), "os": platform.platform(),
            "logical_cpus": os.cpu_count(), "n_threads": n_threads}


def generate(task: Task, model_path: str, out_path: str, n: int, engine: str = "outlines",
             n_threads: int = 4, model_source: str | None = None, progress: bool = True) -> dict:
    t_start = time.perf_counter()
    wall_start = time.time()  # perf_counter pauses while the machine sleeps; time.time() does not
    eng = ENGINES[engine](model_path, n_threads=n_threads)
    t_st = time.perf_counter()
    eng.selftest()  # refuses to run if the constraint does not bind
    selftest_s = time.perf_counter() - t_st
    items = task.items(n)
    meta = {
        "task": task.name, "n": len(items), "engine": eng.info(),
        "model_sha256": _sha256(model_path), "model_source": model_source,
        "versions": {p: _version(p) for p in ("validbutwrong", "outlines", "outlines-core",
                                              "llama-cpp-python", "numba", "jsonschema")},
        "hardware": _hardware(n_threads), "decoding": "greedy (temperature 0)",
        "max_tokens": MAX_TOKENS, "selftest_seconds": round(selftest_s, 3),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path + ".partial", "w", encoding="utf-8") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for i, it in enumerate(items):
            t0 = time.perf_counter()
            rec = run_item(eng, task, it)
            rec["item_seconds"] = round(time.perf_counter() - t0, 3)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if progress:
                print(f"[{i + 1}/{len(items)}] {it.id} {rec['item_seconds']:.1f}s", flush=True)
    meta["wall_seconds"] = round(time.perf_counter() - t_start, 1)            # compute time
    meta["wall_clock_seconds"] = round(time.time() - wall_start, 1)          # includes any sleep
    meta["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    # rewrite the header with the final timings, then publish atomically
    with open(out_path + ".partial", encoding="utf-8") as f:
        lines = f.readlines()
    lines[0] = json.dumps({"meta": meta}) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    os.remove(out_path + ".partial")
    return meta


def read_replay(path: str) -> tuple[dict, list[dict]]:
    with open(path, encoding="utf-8") as f:
        meta = json.loads(f.readline())["meta"]
        records = [json.loads(line) for line in f if line.strip()]
    return meta, records
