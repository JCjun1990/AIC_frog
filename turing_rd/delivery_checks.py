"""Read-only delivery format checks with immutable legacy JSON exceptions."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

# Frozen original bytes; a package-controlled registry cannot authorize new NaNs.
LEGACY_JSON = {
    "task3_inversion/summary.json": ("d975a6094a8859c1b58602bfa17d371cb03a92478017446a20dea2338853469b", {"NaN": 61}),
    "task1_sphere/ai_model/dataset.json": ("4157f250928e0c0ee7ad8976ef76f2f4a915b84b36b15f5a1a0146aa9da9a4f6", {"Infinity": 14}),
    "task2_frog/ai_model/dataset.json": ("ee19eb9e949faa7cdf62883e2c609ec4464b4b836ed19c11eb6d0d4ffa1165bf", {"Infinity": 10}),
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key: " + key)
        result[key] = value
    return result


def _finite_number(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("overflowing JSON number")
    return number


def audit_json(root, paths):
    """Reject malformed, duplicate-key, overflowing and unknown nonstandard JSON."""
    root = Path(root).resolve(); strict, legacy = [], []
    for relative in sorted(set(paths)):
        if Path(relative).suffix.lower() != ".json":
            continue
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("JSON path escapes package")
        data = path.read_bytes(); tokens = Counter()
        def constant(value):
            tokens[value] += 1
            return None  # Only inspect syntax, never reinterpret historical measurements.
        json.loads(data, parse_constant=constant, parse_float=_finite_number, object_pairs_hook=_unique_object)
        if tokens:
            digest = hashlib.sha256(data).hexdigest()
            if LEGACY_JSON.get(relative) != (digest, dict(tokens)):
                raise ValueError("unapproved nonstandard JSON: " + relative)
            legacy.append({"path": relative, "sha256": digest, "nonfinite_tokens": dict(tokens),
                           "status": "preserved_legacy_python_json_extension"})
        else:
            strict.append(relative)
    return {"protocol": "delivery-json-acceptance-v1", "strict_json_files": strict,
            "legacy_nonstandard_json": legacy, "strict_count": len(strict), "legacy_count": len(legacy),
            "scope": "syntax and immutable legacy compatibility; no alteration of input values"}
