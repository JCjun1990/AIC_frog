"""Strict, complete-before-write JSON serialization for I02 experiment evidence.

Why this module exists
----------------------
The first I02 perturbation runs finished their numerical work and then died in

    File "scripts/i02_stability.py", line 219, in perturb
    write_json(... "passed": all(...) and rate_diff < max(1e-9, abs(lam.real)*.03) ...)
    TypeError: Object of type bool is not JSON serializable

``abs(lam.real) * .03`` is a ``numpy.float64``, so ``max(...)`` returns a NumPy
scalar and the comparison yields ``numpy.bool_`` rather than ``bool``.  ``json``
has no encoder for that type.  Because the old ``write_json`` streamed straight
into the destination file, the failure happened *after* ~37 kB had been written:
both directories kept a truncated ``perturbation.json`` ending in ``"passed": ``.

This module removes both halves of the defect:

* :func:`jsonable` converts NumPy scalars/arrays to native JSON values and
  rejects anything else instead of silently stringifying it;
* :func:`write_json_exclusive` serializes the entire document in memory first,
  so a serialization error can never create or truncate an output file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["jsonable", "dumps_json", "write_json_exclusive"]


def jsonable(value):
    """Return a strictly JSON-serializable copy of ``value``.

    NumPy scalars become the corresponding Python builtin (``numpy.bool_`` ->
    ``bool``, ``numpy.float64`` -> ``float``, ...), arrays become lists, and
    mappings/sequences are converted recursively.  Unknown types raise
    ``TypeError``: silently turning a live object into ``str(obj)`` would put
    unverifiable text into an evidence file.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    # NumPy scalars first: numpy.float64 is also a float subclass, but converting
    # through .item() guarantees a pure Python value of the documented type.
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        converted = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object keys must be strings, got {type(key).__name__}")
            converted[key] = jsonable(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    raise TypeError(f"Unsupported JSON value of type {type(value).__name__}: {value!r}")


def dumps_json(value, **kwargs):
    """Serialize ``value`` completely in memory (``allow_nan`` is always False)."""
    kwargs.pop("allow_nan", None)
    kwargs.pop("indent", None)
    return json.dumps(jsonable(value), indent=2, allow_nan=False, **kwargs) + "\n"


def write_json_exclusive(path, value):
    """Serialize fully, then create ``path`` in exclusive mode and write it once.

    A serialization failure therefore leaves no file behind, and an existing file
    is never replaced.  Returns the exact text that was written.
    """
    text = dumps_json(value)
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
    return text
