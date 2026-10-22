"""Float64 MAT persistence and read-back certificates for formal generated fields.

Acceptance here covers structure, precision and the complete stationary PDE residual.
It is deliberately independent of linear stability and does not discard rotation modes.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import scipy
import scipy.io as sio

from .diagnostics import (FULL_STEADY_RESIDUAL_FORMULA, FULL_STEADY_RESIDUAL_VERSION,
                          full_steady_residual)


CERTIFICATE_VERSION = "aic-frog-mat-v1"
ACCEPTANCE_SCOPE = "float64_structure_and_full_steady_residual_only"
DEFAULT_RESIDUAL_TOL = 1e-11


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _array_hash(*arrays) -> str:
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(str(a.shape).encode("ascii") + b"\0" + a.dtype.str.encode("ascii") + b"\0")
        h.update(a.tobytes())
    return h.hexdigest()


def geometry_identity(geom) -> dict:
    """Fingerprint the actual mesh and L/M used in the residual calculation."""
    L = geom.L.tocsr(copy=True)
    L.sum_duplicates()
    L.eliminate_zeros()
    L.sort_indices()
    return {
        "name": str(geom.name),
        "n_vertices": int(geom.n),
        "n_faces": int(len(geom.F)),
        "laplacian": str(geom.laplacian),
        "mesh_sha256": _array_hash(np.asarray(geom.V, dtype="<f8"), np.asarray(geom.F, dtype="<i8")),
        "actual_L_M_sha256": _array_hash(np.asarray(L.shape, dtype="<i8"),
                                          np.asarray(L.indptr, dtype="<i8"),
                                          np.asarray(L.indices, dtype="<i8"),
                                          np.asarray(L.data, dtype="<f8"),
                                          np.asarray(geom.M, dtype="<f8")),
        "mass_sum": float(np.asarray(geom.M).sum()),
    }


def _parameters(p) -> dict:
    out = {name: float(p[name] if isinstance(p, Mapping) else getattr(p, name))
           for name in ("F", "k", "Du", "Dv")}
    if (not all(np.isfinite(x) for x in out.values()) or out["F"] <= 0 or out["k"] < 0 or
            out["Du"] <= 0 or out["Dv"] <= 0):
        raise ValueError("Finite positive F/Du/Dv and nonnegative k are required")
    return out


def _seed(seed) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("A concrete integer random seed is required")
    return int(seed)


def _tolerance(tol) -> float:
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("Residual tolerance must be finite and positive")
    return float(tol)


def _version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _vector(array, n, name):
    a = np.asarray(array)
    if a.shape == (n, 1):
        a = a[:, 0]
    if a.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},) or ({n}, 1); got {a.shape}")
    if a.dtype.kind != "f" or a.dtype.itemsize != 8:
        raise TypeError(f"Formal {name} requires float64 input, got {a.dtype}; no implicit promotion is allowed")
    return a


def _blank_certificate(path, geom, p, seed, tol, provenance):
    # Validate provenance is durable JSON data, without permitting it to replace certificate fields.
    provenance = {} if provenance is None else json.loads(json.dumps(provenance, allow_nan=False))
    return {
        "schema_version": CERTIFICATE_VERSION,
        "diagnostic_version": FULL_STEADY_RESIDUAL_VERSION,
        "acceptance_scope": ACCEPTANCE_SCOPE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mat_file": Path(path).name,
        "mat_sha256": file_sha256(path),
        "parameters": _parameters(p),
        "seed": _seed(seed),
        "geometry": geometry_identity(geom),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "robust_laplacian": _version("robust-laplacian")},
        "provenance": provenance,
        "structure": {"passed": False},
        "precision": {"passed": False},
        "physical_range": {"passed": False},
        "residual": {"formula": FULL_STEADY_RESIDUAL_FORMULA, "normalization": "F",
                     "rotation_projection": False, "value": None, "tolerance": _tolerance(tol),
                     "passed": False},
        "stability_status": "not_evaluated_by_i01",
        "accepted": False,
        "failure_reasons": [],
    }


def read_mat_fields(path, n: int):
    """Read the exact required MATLAB struct/column-vector layout without squeezing it."""
    data = sio.loadmat(path, struct_as_record=True, squeeze_me=False)
    solution = data.get("Solution")
    if not isinstance(solution, np.ndarray) or solution.shape != (1, 1):
        raise ValueError("Solution must be a 1x1 MATLAB struct")
    if solution.dtype.names is None or not {"u", "v"}.issubset(solution.dtype.names):
        raise ValueError("Solution must have fields u and v")
    u, v = solution["u"][0, 0], solution["v"][0, 0]
    if not isinstance(u, np.ndarray) or not isinstance(v, np.ndarray):
        raise ValueError("Solution.u/v must be numeric arrays")
    if u.shape != (n, 1) or v.shape != (n, 1):
        raise ValueError(f"Solution.u/v must each have shape ({n}, 1); got {u.shape}, {v.shape}")
    return u[:, 0], v[:, 0]


def verify_mat(path, geom, p, *, seed, tol=DEFAULT_RESIDUAL_TOL, provenance=None) -> dict:
    """Recompute a certificate from the saved bytes using this exact Geometry's L and M.

    A float64 container whose entire nonuniform field is exactly float32-representable
    is flagged instead of being advertised as a recovered high-precision solution.
    Exact constant fields are separately identified: their precision origin cannot be
    inferred from their values, but no digits are needed to store an exact constant.
    """
    cert = _blank_certificate(path, geom, p, seed, tol, provenance)
    failures = cert["failure_reasons"]
    try:
        u, v = read_mat_fields(path, geom.n)
    except Exception as error:
        failures.append("invalid_mat_structure")
        cert["structure"]["error"] = f"{type(error).__name__}: {error}"
        return cert

    dtype_ok = all(x.dtype.kind == "f" and x.dtype.itemsize == 8 for x in (u, v))
    finite = all(np.isfinite(x).all() for x in (u, v))
    cert["structure"] = {"passed": bool(dtype_ok and finite), "solution_shape": [1, 1],
                         "u_shape": [geom.n, 1], "v_shape": [geom.n, 1],
                         "u_dtype": str(u.dtype), "v_dtype": str(v.dtype), "finite": bool(finite)}
    if not dtype_ok:
        failures.append("not_float64")
    if not finite:
        failures.append("nonfinite_values")
    if not dtype_ok or not finite:
        return cert

    with np.errstate(over="ignore", invalid="ignore"):
        u32 = bool(np.array_equal(u, u.astype(np.float32).astype(np.float64)))
        v32 = bool(np.array_equal(v, v.astype(np.float32).astype(np.float64)))
    exact_constant = bool(np.ptp(u) == 0 and np.ptp(v) == 0)
    quantized_nonuniform = bool((u32 and np.ptp(u) > 0) or (v32 and np.ptp(v) > 0))
    cert["precision"] = {"passed": not quantized_nonuniform, "storage_dtype": "float64",
                         "u_float32_roundtrip_exact": u32, "v_float32_roundtrip_exact": v32,
                         "exact_constant_field": exact_constant,
                         "value_precision_origin": "not_inferable_for_exact_constants" if exact_constant else
                         ("float32_quantized_nonuniform" if quantized_nonuniform else "not_entirely_float32_representable")}
    if quantized_nonuniform:
        failures.append("float32_quantized_nonuniform_field")

    range_ok = bool(np.all(u >= 0) and np.all(u <= 1) and np.all(v >= 0))
    cert["physical_range"] = {"passed": range_ok, "bounds": {"u": [0.0, 1.0], "v_min": 0.0},
                              "u_min": float(u.min()), "u_max": float(u.max()),
                              "v_min": float(v.min()), "v_max": float(v.max())}
    if not range_ok:
        failures.append("outside_physical_range")

    with np.errstate(over="ignore", invalid="ignore"):
        residual = full_steady_residual(geom, SimpleNamespace(**cert["parameters"]), u, v)
    value = residual["full_joint_rms_over_F"]
    residual_ok = bool(np.isfinite(value) and value <= tol)
    cert["residual"].update(value=float(value) if np.isfinite(value) else None,
                            Ru_rms=float(residual["Ru_rms"]) if np.isfinite(residual["Ru_rms"]) else None,
                            Rv_rms=float(residual["Rv_rms"]) if np.isfinite(residual["Rv_rms"]) else None,
                            passed=residual_ok)
    if not residual_ok:
        failures.append("full_residual_above_tolerance" if np.isfinite(value) else "nonfinite_residual")
    cert["accepted"] = bool(cert["structure"]["passed"] and cert["precision"]["passed"] and
                            cert["physical_range"]["passed"] and residual_ok)
    return cert


def save_certified_mat(path, geom, p, u, v, *, seed, tol=DEFAULT_RESIDUAL_TOL,
                       provenance=None, overwrite=False) -> dict:
    """Save float64 fields, reread them, and certify the actual saved values.

    Invalid residuals/ranges remain available as explicitly failed candidates. Input
    promotion is forbidden, and an existing file is never replaced unless requested.
    """
    path = Path(path)
    u, v = _vector(u, geom.n, "u"), _vector(v, geom.n, "v")
    _seed(seed)
    _parameters(p)
    _tolerance(tol)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb" if overwrite else "xb") as stream:
        sio.savemat(stream, {"Solution": {"u": u[:, None], "v": v[:, None]}})
    saved_u, saved_v = read_mat_fields(path, geom.n)
    roundtrip_equal = bool(np.array_equal(u, saved_u) and np.array_equal(v, saved_v))
    cert = verify_mat(path, geom, p, seed=seed, tol=tol, provenance=provenance)
    cert["precision"]["source_dtype"] = {"u": str(u.dtype), "v": str(v.dtype)}
    cert["precision"]["source_values_preserved"] = roundtrip_equal
    if not roundtrip_equal:
        cert["precision"]["passed"] = False
        cert["accepted"] = False
        cert["failure_reasons"].append("mat_roundtrip_changed_values")
    return cert


def validate_certificate(path, certificate, *, expected_params, expected_seed,
                         expected_geometry, tol=DEFAULT_RESIDUAL_TOL) -> dict:
    """Validate the binding and independently repeat the read-back PDE acceptance.

    ``expected_geometry`` must be the real Geometry object, not merely a label or
    supplied hash. This prevents a cotan check being accepted as robust validation.
    Stability is outside this function's acceptance scope.
    """
    cert = json.loads(Path(certificate).read_text(encoding="utf-8")) if isinstance(certificate, (str, Path)) else certificate
    if not isinstance(cert, Mapping):
        raise ValueError("MAT certificate must be a mapping")
    checks = {"schema_version": CERTIFICATE_VERSION, "diagnostic_version": FULL_STEADY_RESIDUAL_VERSION,
              "acceptance_scope": ACCEPTANCE_SCOPE, "mat_sha256": file_sha256(path),
              "parameters": _parameters(expected_params), "seed": _seed(expected_seed),
              "geometry": geometry_identity(expected_geometry)}
    for field, value in checks.items():
        if cert.get(field) != value:
            raise ValueError(f"MAT certificate {field} mismatch")
    res = cert.get("residual", {})
    if (res.get("formula") != FULL_STEADY_RESIDUAL_FORMULA or res.get("normalization") != "F" or
            res.get("rotation_projection") is not False):
        raise ValueError("MAT certificate residual definition mismatch")
    recorded_tol = _tolerance(res.get("tolerance", float("nan")))
    if recorded_tol > _tolerance(tol):
        raise ValueError("MAT certificate tolerance is looser than the required tolerance")
    fresh = verify_mat(path, expected_geometry, expected_params, seed=expected_seed, tol=recorded_tol,
                       provenance=cert.get("provenance"))
    if (not cert.get("accepted") or not fresh["accepted"] or
            any(cert.get(field, {}).get("passed") is not True
                for field in ("structure", "precision", "physical_range", "residual"))):
        raise ValueError("MAT certificate or independent read-back acceptance failed: " +
                         ", ".join(fresh["failure_reasons"]))
    for section in ("structure", "precision", "physical_range"):
        for field, value in fresh[section].items():
            if cert.get(section, {}).get(field) != value:
                raise ValueError(f"MAT certificate {section}.{field} mismatch")
    value = res.get("value")
    if value is None or not np.isfinite(value) or not np.isclose(value, fresh["residual"]["value"], rtol=1e-8, atol=1e-15):
        raise ValueError("MAT certificate residual does not match read-back recomputation")
    if cert.get("stability_status") != "not_evaluated_by_i01":
        raise ValueError("I01 certificate cannot certify linear stability")
    return fresh
