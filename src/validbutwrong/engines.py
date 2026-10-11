"""Decoding backends. Each engine generates from chat messages, optionally under a JSON Schema.

Both engines run the SAME llama.cpp runtime and the same GGUF weights; they differ only in
how the schema becomes a constraint:

* ``llamacpp``  — llama.cpp's native GBNF grammar, compiled from the JSON Schema
                  (``LlamaGrammar.from_json_schema``).
* ``outlines``  — outlines' JSON-Schema -> regex -> token-FSM logits processor, driving the
                  same ``llama_cpp.Llama`` object.

So a speed difference between them is the constraint machinery, not the model runtime.
Greedy decoding (temperature 0) throughout, so a run is reproducible on the same build.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass


class UnsupportedSchema(ValueError):
    """The backend cannot compile this JSON Schema into a constraint (a backend limitation)."""


@dataclass
class Generation:
    text: str
    seconds: float              # wall time of this generation (excludes grammar compile)
    compile_seconds: float      # grammar/FSM compile time paid by THIS call (0 if cached)
    n_tokens: int


class Engine:
    name = "engine"

    def __init__(self, model_path: str, n_threads: int = 4, n_ctx: int = 4096, seed: int = 42):
        import llama_cpp

        t0 = time.perf_counter()
        self.llm = llama_cpp.Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads,
                                   n_gpu_layers=0, seed=seed, verbose=False)
        self.load_seconds = time.perf_counter() - t0
        self.model_path = model_path
        self.llama_cpp_version = llama_cpp.__version__
        self._template = self.llm.metadata.get("tokenizer.chat_template")
        self._cache: dict[str, object] = {}

    # -- prompt formatting: the model's own chat template from the GGUF metadata
    def format(self, messages: list[dict]) -> str:
        if not self._template:
            raise ValueError("GGUF has no tokenizer.chat_template; pass a chat model")
        import jinja2

        env = jinja2.Environment(loader=jinja2.BaseLoader())
        return env.from_string(self._template).render(messages=messages, add_generation_prompt=True)

    def generate(self, messages: list[dict], schema: dict | None, max_tokens: int) -> Generation:
        raise NotImplementedError

    # A constraint that fails can fail SILENTLY: outlines' logits processor raising inside
    # llama.cpp's ctypes sampler callback is logged as "Exception ignored" and generation
    # continues unconstrained. Every engine therefore proves, before a run, that its
    # constraint actually binds: it must force a constant the model would never emit.
    CANARY = {"type": "object", "properties": {"canary": {"const": "vbw-canary-7f3q"}},
              "required": ["canary"], "additionalProperties": False}

    def selftest(self) -> None:
        g = self.generate([{"role": "user", "content": "Reply with the word hello."}], self.CANARY, 24)
        try:
            ok = json.loads(g.text) == {"canary": "vbw-canary-7f3q"}
        except json.JSONDecodeError:
            ok = False
        if not ok:
            raise RuntimeError(f"{self.name}: constraint did NOT bind (canary output {g.text!r}); "
                               "refusing to run, since the 'constrained' arms would be unconstrained")

    def info(self) -> dict:
        return {"engine": self.name, "model_file": self.model_path.rsplit("/", 1)[-1],
                "llama_cpp_python": self.llama_cpp_version, "load_seconds": round(self.load_seconds, 3)}


class LlamaCppGrammarEngine(Engine):
    name = "llamacpp"

    def _grammar(self, schema: dict):
        import llama_cpp

        # property ORDER is part of the contract (reasoning-before-answer), so never sort keys
        key = json.dumps(schema)
        if key in self._cache:
            return self._cache[key], 0.0
        t0 = time.perf_counter()
        try:
            g = llama_cpp.LlamaGrammar.from_json_schema(key, verbose=False)
        except Exception as e:  # noqa: BLE001 - any compile failure is a backend limitation
            raise UnsupportedSchema(f"llamacpp: {e}") from e
        dt = time.perf_counter() - t0
        self._cache[key] = g
        return g, dt

    def generate(self, messages, schema, max_tokens):
        grammar, compile_s = (None, 0.0) if schema is None else self._grammar(schema)
        t0 = time.perf_counter()
        out = self.llm.create_completion(self.format(messages), max_tokens=max_tokens,
                                         temperature=0.0, grammar=grammar, stop=["<|im_end|>"])
        dt = time.perf_counter() - t0
        return Generation(out["choices"][0]["text"], dt, compile_s, out["usage"]["completion_tokens"])


class OutlinesEngine(Engine):
    name = "outlines"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import outlines

        self._model = outlines.from_llamacpp(self.llm, chat_mode=False)
        self._outlines = outlines

    def _generator(self, schema: dict):
        key = json.dumps(schema)  # keep property order (see LlamaCppGrammarEngine._grammar)
        if key in self._cache:
            return self._cache[key], 0.0
        t0 = time.perf_counter()
        try:
            g = self._outlines.Generator(self._model, self._outlines.types.JsonSchema(key))
        except Exception as e:  # noqa: BLE001 - any compile failure is a backend limitation
            raise UnsupportedSchema(f"outlines: {e}") from e
        dt = time.perf_counter() - t0
        self._cache[key] = g
        return g, dt

    def generate(self, messages, schema, max_tokens):
        prompt = self.format(messages)
        if schema is None:
            t0 = time.perf_counter()
            out = self.llm.create_completion(prompt, max_tokens=max_tokens, temperature=0.0,
                                             stop=["<|im_end|>"])
            return Generation(out["choices"][0]["text"], time.perf_counter() - t0, 0.0,
                              out["usage"]["completion_tokens"])
        gen, compile_s = self._generator(schema)
        t0 = time.perf_counter()
        text = gen(prompt, max_tokens=max_tokens, temperature=0.0)
        dt = time.perf_counter() - t0
        return Generation(text, dt, compile_s, len(self.llm.tokenize(text.encode(), add_bos=False)))


ENGINES = {"llamacpp": LlamaCppGrammarEngine, "outlines": OutlinesEngine}
