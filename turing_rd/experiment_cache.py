"""Versioned local experiment cache. Legacy plain pickles are never trusted here.

An entry is a canonical JSON header followed by a SHA256-bound pickle payload.
These are trusted local files, not a safe interchange format for untrusted input.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import pickle
import platform
import tempfile
import time

import numpy as np

PROTOCOL = "aic-experiment-v1"
MAGIC = b"AIC-EXPERIMENT-CACHE-1\n"


class CacheRejected(ValueError):
    pass


def canonical_bytes(value):
    """JSON primitives only: round-trip float precision, sorted keys, no repr()."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def fingerprint(identity):
    return hashlib.sha256(canonical_bytes(identity)).hexdigest()


def file_identity(path, *, required=True):
    path = Path(path).resolve()
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"required experiment input missing: {path}")
        return {"path": path.as_posix(), "status": "missing"}
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return {"path": path.as_posix(), "sha256": h.hexdigest(), "size": path.stat().st_size}


def array_identity(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(canonical_bytes({"shape": list(a.shape), "dtype": a.dtype.str}))
        h.update(a.tobytes())
    return h.hexdigest()


def dependency_versions():
    versions = {"python": platform.python_version(), "platform": platform.platform(),
                "machine": platform.machine()}
    for package in ("numpy", "scipy", "torch", "scikit-learn", "robust-laplacian", "joblib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    versions["threads"] = {k: os.environ.get(k) for k in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
    return versions


def code_identity(root):
    root = Path(root)
    paths = sorted((root / "turing_rd").glob("*.py")) + [root / "scripts/ai_generation.py"]
    return {p.relative_to(root).as_posix(): file_identity(p)["sha256"] for p in paths}


def defaults(function):
    """Snapshot effective defaults, including explicitly recognised torch dtypes."""
    import torch
    result = {}
    for name, parameter in inspect.signature(function).parameters.items():
        v = parameter.default
        if v is inspect.Parameter.empty:
            continue
        if isinstance(v, torch.dtype):
            v = str(v)
        result[name] = v
    canonical_bytes(result)  # reject unsupported defaults instead of object repr()
    return result


def model_identity(path):
    path = Path(path)
    return {"weights": file_identity(path),
            "training_data": file_identity(path.parent / "dataset.json", required=False),
            "training_provenance": file_identity(path.with_suffix(".provenance.json"), required=False),
            "historical_provenance": "absent sidecar means original training environment is not established"}


def cache_path(directory, identity):
    return Path(directory) / PROTOCOL / (fingerprint(identity) + ".cache")


def atomic_write(path, data):
    """Same-directory unique temporary file + atomic replace, including concurrent writers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # Windows readers can briefly deny delete-sharing on the old file. Retry
        # only that transient sharing error; never truncate the existing entry.
        for attempt in range(10):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(min(.005 * 2 ** attempt, .1))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_cache(path, identity, result):
    if result.get("error"):
        raise CacheRejected("failed computations are not reusable cache entries")
    payload = pickle.dumps(result, protocol=5)
    header = {"protocol": PROTOCOL, "fingerprint": fingerprint(identity), "identity": identity,
              "payload_sha256": hashlib.sha256(payload).hexdigest(), "payload_size": len(payload)}
    atomic_write(path, MAGIC + canonical_bytes(header) + b"\n" + payload)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CacheRejected(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_cache(path, expected=None):
    """Verify protocol, complete identity, filename and payload before deserialising."""
    path = Path(path)
    try:
        with path.open("rb") as stream:
            if stream.readline() != MAGIC:
                raise CacheRejected("legacy or invalid cache format")
            line = stream.readline(1024 * 1024 + 1)
            if not line.endswith(b"\n") or len(line) > 1024 * 1024:
                raise CacheRejected("invalid cache header length")
            header = json.loads(line, object_pairs_hook=_unique_object)
            identity = header["identity"]
            digest = fingerprint(identity)
            if header["protocol"] != PROTOCOL or identity["protocol"] != PROTOCOL:
                raise CacheRejected("unsupported cache protocol")
            if header["fingerprint"] != digest or path.name != digest + ".cache":
                raise CacheRejected("fingerprint/filename mismatch")
            if expected is not None and canonical_bytes(identity) != canonical_bytes(expected):
                raise CacheRejected("experiment identity mismatch")
            payload = stream.read()
            if len(payload) != header["payload_size"] or hashlib.sha256(payload).hexdigest() != header["payload_sha256"]:
                raise CacheRejected("cache payload damaged")
        result = pickle.loads(payload)
        if not isinstance(result, dict) or result.get("error"):
            raise CacheRejected("invalid or failed cached result")
        return result, identity
    except Exception as exc:
        raise CacheRejected(f"cache rejected: {exc}") from exc


def write_training_provenance(model_path, data, protocol):
    """Only explicit training stages call this, after writing NEW weights."""
    record = {"schema": "aic-training-provenance-v1", "model": file_identity(model_path),
              "training_inputs": data, "training_protocol": protocol,
              "environment": dependency_versions(), "code": code_identity(Path(__file__).resolve().parents[1])}
    atomic_write(Path(model_path).with_suffix(".provenance.json"), canonical_bytes(record) + b"\n")
    return record
