"""JSON helpers: BFCL->JSON Schema conversion, first-object extraction, and validation."""

from __future__ import annotations

import json
from typing import Any

import jsonschema

_BFCL_TYPES = {"dict": "object", "float": "number", "tuple": "array",
               # "any" as an explicit union: some backends (outlines) cannot compile a typeless leaf
               "any": ["string", "number", "integer", "boolean", "array", "object", "null"]}


def bfcl_to_jsonschema(params: dict) -> dict:
    """Convert BFCL's parameter dialect (``dict``/``float``/``tuple``/``any``) to JSON Schema.

    BFCL schemas are not valid JSON Schema as published; a grammar compiler or ``jsonschema``
    needs this conversion first. Descriptions and ``enum`` are kept; ``default`` is dropped
    (it is documentation, not a constraint). Objects are closed (``additionalProperties: false``).
    """
    out: dict[str, Any] = {}
    t = params.get("type")
    if t is not None:
        out["type"] = _BFCL_TYPES.get(t, t)
    for key in ("description", "enum"):
        if key in params:
            out[key] = params[key]
    if "properties" in params:
        out["properties"] = {k: bfcl_to_jsonschema(v) for k, v in params["properties"].items()}
        out["type"] = "object"
        out["additionalProperties"] = False
        if params.get("required"):
            out["required"] = list(params["required"])
    if "items" in params:
        out["items"] = bfcl_to_jsonschema(params["items"])
    return out


def first_json_object(text: str) -> Any:
    """Return the first parseable top-level JSON object in ``text``, or None.

    Unlike a ``{.*}`` regex, this scans balanced braces and never rewrites the payload
    (no whitespace stripping), so string values are graded as the model produced them.
    """
    dec = json.JSONDecoder()
    i = text.find("{")
    while i != -1:
        try:
            obj, _ = dec.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        i = text.find("{", i + 1)
    return None


def is_valid(obj: Any, schema: dict) -> bool:
    if obj is None:
        return False
    try:
        jsonschema.validate(obj, schema)
        return True
    except jsonschema.ValidationError:
        return False
