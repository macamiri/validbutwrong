"""Point validbutwrong at YOUR task: a JSONL of {"question", "answer"} rows and your own schema.

    pip install 'validbutwrong[live]'
    python examples/custom_task.py my_questions.jsonl path/to/chat-model.gguf

A task is ~30 lines: items, two prompts, a schema, and how to read + check an answer.
Everything else (the four arms, the canary self-test, metrics, report) is reused.
"""

import json
import sys

from validbutwrong import Item, Task, build_report, generate, read_replay
from validbutwrong.report import markdown


class ExactMatchQA(Task):
    name = "my-qa"
    correctness_label = "exact-match accuracy"
    answer_schema = "answer"
    schemas = {"answer": {"type": "object", "properties": {"answer": {"type": "string"}},
                          "required": ["answer"], "additionalProperties": False}}

    def __init__(self, path):
        self.path = path

    def items(self, n=None):
        rows = [json.loads(line) for line in open(self.path, encoding="utf-8")][:n]
        return [Item(id=f"q{i}", prompt=r["question"], gold=r["answer"]) for i, r in enumerate(rows)]

    def free_prompt(self, item):
        return f"{item.prompt}\n\nThink it through, then end with a line 'Answer: <answer>'."

    def json_prompt(self, item, schema):
        return f"{item.prompt}\n\nRespond with ONLY a JSON object matching:\n{json.dumps(schema)}"

    def extract_free(self, text, item):
        lines = [ln for ln in text.splitlines() if ln.lower().startswith("answer:")]
        return lines[-1].split(":", 1)[1].strip() if lines else None

    def extract_json(self, obj, item):
        return obj.get("answer") if isinstance(obj, dict) else None

    def is_correct(self, answer, item):
        return isinstance(answer, str) and answer.strip().lower() == str(item.gold).strip().lower()


if __name__ == "__main__":
    data, model = sys.argv[1], sys.argv[2]
    task = ExactMatchQA(data)
    generate(task, model, "my-qa-replay.jsonl", n=20)          # live generation, all four arms
    meta, records = read_replay("my-qa-replay.jsonl")           # re-score any time, no model needed
    items = task.items(len(records))
    print(markdown(build_report(task, items, records, meta, mode="live")))
