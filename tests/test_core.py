"""Tests that need no model: parsing, scoring, statistics, the CLI gate, and the canary."""

import json

import pytest

from validbutwrong import cli
from validbutwrong import metrics as M
from validbutwrong.engines import Engine, Generation
from validbutwrong.schema import bfcl_to_jsonschema, first_json_object, is_valid
from validbutwrong.tasks import BFCLSimple, GSM8K, Item, ast_match

GSM = GSM8K.schemas


# --------------------------------------------------------------------------- schema / parsing

def test_bfcl_types_are_converted_to_json_schema():
    s = bfcl_to_jsonschema({"type": "dict", "properties": {
        "x": {"type": "float"}, "p": {"type": "tuple", "items": {"type": "integer"}}, "o": {"type": "any"}},
        "required": ["x"]})
    assert s["type"] == "object" and s["properties"]["x"]["type"] == "number"
    assert s["properties"]["p"] == {"type": "array", "items": {"type": "integer"}}
    assert s["properties"]["o"]["type"] == ["string", "number", "integer", "boolean", "array", "object", "null"]
    assert s["additionalProperties"] is False and s["required"] == ["x"]


def test_first_json_object_keeps_string_values_intact():
    # the vLLM benchmark stripped every space before parsing; we must not
    obj = first_json_object('Sure! {"name": "John Smith", "n": 2} done')
    assert obj == {"name": "John Smith", "n": 2}
    assert first_json_object("no json here") is None
    assert first_json_object('{"a": 1') is None


def test_empty_object_is_not_valid_against_a_required_schema():
    # the exact case where vLLM's correct_rate scores 100: '{}' parses but must not validate
    assert first_json_object("{}") == {}
    assert not is_valid({}, GSM["answer"])
    assert is_valid({"answer": 3}, GSM["answer"])
    assert not is_valid({"answer": "3"}, GSM["answer"])


# --------------------------------------------------------------------------- tasks

def _gsm_item(gold=18.0):
    return Item(id="g", prompt="q", gold=gold)


def test_gsm8k_free_text_extraction():
    t = GSM8K("unused")
    assert t.extract_free("work...\nAnswer: 1,234", _gsm_item()) == 1234.0
    assert t.extract_free("so she makes $18 per day.", _gsm_item()) == 18.0
    assert t.extract_free("no number", _gsm_item()) is None


def test_gsm8k_json_answer_is_scored_even_if_schema_invalid():
    t = GSM8K("unused")
    it = _gsm_item()
    assert t.is_correct(t.extract_json({"answer": "18"}, it), it)  # right but schema-invalid
    assert not t.is_correct(t.extract_json({"answer": 16}, it), it)


FN = {"name": "calculate_triangle_area", "parameters": {"type": "dict", "properties": {
    "base": {"type": "integer"}, "height": {"type": "integer"}, "unit": {"type": "string"}},
    "required": ["base", "height"]}}
GOLD = {"base": [10], "height": [5], "unit": ["units", ""]}


def _bfcl_item():
    return Item(id="b", prompt="area", gold=GOLD, meta={"function": FN, "gold_name": FN["name"]})


def test_ast_match_rules():
    params = bfcl_to_jsonschema(FN["parameters"])
    assert ast_match({"base": 10, "height": 5}, GOLD, params)            # optional omitted
    assert ast_match({"base": 10, "height": 5, "unit": "Units"}, GOLD, params)  # string normalised
    assert not ast_match({"base": 10}, GOLD, params)                       # required missing
    assert not ast_match({"base": 10, "height": 6}, GOLD, params)          # wrong value
    assert not ast_match({"base": 10, "height": 5, "colour": "red"}, GOLD, params)  # unknown param
    assert not ast_match(None, GOLD, params)


def test_bfcl_free_text_maps_positional_args_charitably():
    t = BFCLSimple("unused", "unused")
    it = _bfcl_item()
    assert t.extract_free("calculate_triangle_area(10, 5)", it) == {"base": 10, "height": 5}
    assert t.is_correct(t.extract_free("calculate_triangle_area(base=10, height=5)", it), it)
    assert t.extract_free("I cannot do that", it) is None


# --------------------------------------------------------------------------- metrics

def _states(pairs):
    return [{"valid": v, "correct": c, "answer": None} for v, c in pairs]


def test_quadrant_and_wrong_but_valid_use_all_items_as_denominator():
    st = _states([(True, True), (True, False), (True, False), (False, True)])
    m = M.arm_metrics(st)
    assert m["quadrant"] == {"valid_right": 1, "valid_wrong": 2, "invalid_right": 1, "invalid_wrong": 0}
    assert m["wrong_but_valid"]["rate"] == 0.5
    assert m["schema_validity"]["rate"] == 0.75


def test_unconstrained_validity_is_na_not_zero():
    m = M.arm_metrics(_states([(None, True), (None, False)]))
    assert m["schema_validity"] is None and m["wrong_but_valid"] is None and m["quadrant"] is None


def test_paired_bootstrap_detects_a_large_effect_and_not_a_null():
    a = _states([(True, True)] * 50 + [(True, False)] * 50)
    b_big = _states([(True, False)] * 100)
    c = M.compare(a, b_big, "a", "b")
    assert c["deltas"]["answer_accuracy"]["excludes_0"]
    assert c["flips"] == {"right_to_wrong": 50, "wrong_to_right": 0}
    c0 = M.compare(a, a, "a", "a")
    assert not c0["deltas"]["answer_accuracy"]["excludes_0"]


def test_bootstrap_is_deterministic():
    xs = [1.0, 0.0] * 30
    assert M.bootstrap_ci(xs) == M.bootstrap_ci(xs)


# --------------------------------------------------------------------------- canary (positive AND negative control)

class _FakeEngine(Engine):
    def __init__(self, obeys: bool):  # no model
        self.name, self.obeys = "fake", obeys

    def generate(self, messages, schema, max_tokens):
        text = json.dumps({"canary": "vbw-canary-7f3q"}) if (schema and self.obeys) else '{"greeting": "hello"}'
        return Generation(text, 0.0, 0.0, 1)


def test_canary_passes_when_constraint_binds():
    _FakeEngine(obeys=True).selftest()


def test_canary_refuses_when_constraint_is_silently_ignored():
    with pytest.raises(RuntimeError, match="did NOT bind"):
        _FakeEngine(obeys=False).selftest()


# --------------------------------------------------------------------------- CLI on a synthetic replay

def _write_replay(path, n_wrong_valid):
    t = GSM8K(cli._data("gsm8k_test_first200.jsonl"))
    items = t.items(4)
    with open(path, "w") as f:
        f.write(json.dumps({"meta": {"task": "gsm8k", "n": 4}}) + "\n")
        for i, it in enumerate(items):
            good = json.dumps({"answer": int(it.gold)})
            bad = json.dumps({"answer": int(it.gold) + 1})
            con = bad if i < n_wrong_valid else good
            arms = {"unconstrained": {"text": f"Answer: {int(it.gold)}"},
                    "prompt_json[answer]": {"text": good}, "constrained[answer]": {"text": con},
                    "prompt_json[reasoning]": {"text": "oops"},
                    "constrained[reasoning]": {"text": json.dumps({"reasoning": "r", "answer": int(it.gold)})},
                    "reason_then_constrain[answer]": {"text": good}}
            f.write(json.dumps({"id": it.id, "arms": arms}) + "\n")


def test_cli_scores_replay_and_writes_artifacts(tmp_path):
    rp = tmp_path / "r.jsonl"
    _write_replay(rp, n_wrong_valid=1)
    out = tmp_path / "out"
    assert cli.main(["run", "--task", "gsm8k", "--replay", str(rp), "--out", str(out)]) == 0
    rep = json.loads((out / "report.json").read_text())
    assert rep["arms"]["constrained[answer]"]["quadrant"]["valid_wrong"] == 1
    assert rep["arms"]["prompt_json[reasoning]"]["schema_validity"]["rate"] == 0.0
    assert (out / "quadrants.svg").read_text().startswith("<svg")
    head = next(c for c in rep["comparisons"] if c["to"] == "constrained[answer]" and c["from"].startswith("prompt_json"))
    assert head["flips"]["right_to_wrong"] == 1


def test_cli_gate_exit_code(tmp_path):
    rp = tmp_path / "r.jsonl"
    _write_replay(rp, n_wrong_valid=2)  # 50% wrong-but-valid on constrained[answer]
    args = ["run", "--task", "gsm8k", "--replay", str(rp), "--out", str(tmp_path / "o")]
    assert cli.main(args + ["--fail-if-wrong-valid-above", "0.4"]) == 1
    assert cli.main(args + ["--fail-if-wrong-valid-above", "0.6"]) == 0


# --------------------------------------------------------------------------- custom task, end to end with a fake engine

def test_custom_task_runs_through_all_arms_and_scores(tmp_path, monkeypatch):
    import importlib.util
    import os

    spec = importlib.util.spec_from_file_location(
        "custom_task", os.path.join(os.path.dirname(__file__), "..", "examples", "custom_task.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    data = tmp_path / "q.jsonl"
    data.write_text('{"question": "Capital of France?", "answer": "Paris"}\n'
                    '{"question": "2+2 in words?", "answer": "four"}\n')

    class Scripted(_FakeEngine):
        """Free text right; prompt_json right; the grammar arm valid but wrong."""
        def __init__(self, model_path, n_threads=4):
            self.name, self.obeys = "scripted", True
        def info(self):
            return {"engine": "scripted"}
        def generate(self, messages, schema, max_tokens):
            q = messages[0]["content"]
            gold = "Paris" if "France" in q else "four"
            if schema and "canary" in json.dumps(schema):
                return Generation(json.dumps({"canary": "vbw-canary-7f3q"}), 0, 0, 1)
            if schema:
                return Generation(json.dumps({"answer": "London"}), 0, 0, 1)
            if "JSON" in q:
                return Generation(json.dumps({"answer": gold}), 0, 0, 1)
            return Generation(f"thinking...\nAnswer: {gold}", 0, 0, 1)

    from validbutwrong import run as run_mod
    monkeypatch.setitem(run_mod.ENGINES, "scripted", Scripted)
    monkeypatch.setattr(run_mod, "_sha256", lambda p: "test")
    task = mod.ExactMatchQA(str(data))
    out = tmp_path / "replay.jsonl"
    run_mod.generate(task, "unused.gguf", str(out), n=2, engine="scripted", progress=False)
    meta, records = run_mod.read_replay(str(out))
    from validbutwrong.report import build
    rep = build(task, task.items(2), records, meta, "live")
    assert rep["arms"]["unconstrained"]["answer_accuracy"]["rate"] == 1.0
    assert rep["arms"]["prompt_json[answer]"]["answer_accuracy"]["rate"] == 1.0
    assert rep["arms"]["constrained[answer]"]["quadrant"]["valid_wrong"] == 2
    head = next(c for c in rep["comparisons"] if c["from"] == "prompt_json[answer]")
    assert head["flips"]["right_to_wrong"] == 2


def test_gsm8k_reads_latex_boxed_before_last_number():
    t = GSM8K("unused")
    it = _gsm_item(1400.0)
    text = "So the zookeeper needs \\boxed{1400} bananas to last for 2 months."
    assert t.extract_free(text, it) == 1400.0
    assert t.extract_free("Answer: 7 (not \\boxed{9})", it) == 7.0  # explicit line wins


def test_unsupported_schema_is_excluded_and_reported_not_scored_wrong(tmp_path):
    rp = tmp_path / "r.jsonl"
    _write_replay(rp, n_wrong_valid=0)
    lines = rp.read_text().splitlines()
    rec = json.loads(lines[2])
    rec["arms"]["constrained[answer]"] = {"text": "", "unsupported_schema": "outlines: Unsupported JSON Schema"}
    lines[2] = json.dumps(rec)
    rp.write_text("\n".join(lines) + "\n")
    out = tmp_path / "o"
    assert cli.main(["run", "--task", "gsm8k", "--replay", str(rp), "--out", str(out)]) == 0
    rep = json.loads((out / "report.json").read_text())
    assert rep["n"] == 3 and rep["excluded"]["n"] == 1 and rep["excluded"]["ids"] == [rec["id"]]
    assert all(a["n"] == 3 for a in rep["arms"].values())          # dropped from EVERY arm (paired)
    assert len(rep["ledger"]) == 3
    assert "excluded" in (out / "report.md").read_text()


def test_only_schema_compile_errors_are_caught():
    from validbutwrong.arms import _constrained
    from validbutwrong.engines import UnsupportedSchema

    class Boom(_FakeEngine):
        def __init__(self, exc):
            self.name, self.exc = "boom", exc
        def generate(self, messages, schema, max_tokens):
            raise self.exc

    assert "unsupported_schema" in _constrained(Boom(UnsupportedSchema("x")), [], {}, 8)
    with pytest.raises(ValueError):  # e.g. llama.cpp "Failed to create llama_context" must NOT be swallowed
        _constrained(Boom(ValueError("Failed to create llama_context")), [], {}, 8)


def test_default_run_replays_both_tasks_without_model_libs(tmp_path):
    out = tmp_path / "o"
    assert cli.main(["run", "--out", str(out)]) == 0
    for name in ("bfcl-simple", "gsm8k"):
        rep = json.loads((out / name / "report.json").read_text())
        assert rep["mode"] == "replay" and rep["n"] == 200
    assert cli.main(["run", "--replay", "x.jsonl", "--out", str(out)]) == 2  # ambiguous without --task
