"""Versioned endpoint-label and reference-library quality; no integration/training.

Legacy AI scalar residuals describe the ORIGINAL endpoint, not a recertified
float32 cache. Spectral pass requires a declared, validated finite protocol;
old scalar maxima / stable flags never supply missing eigenpair evidence.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math

from .diagnostics import FULL_STEADY_RESIDUAL_FORMULA, FULL_STEADY_RESIDUAL_VERSION, full_steady_residual
from .spectral_policy import (STEADY_TOL, INVERSION_SCREEN_CONFIGS, evaluate_spectra)

QUALITY_VERSION = "endpoint-label-quality-v1"
TRAINING_POLICIES = ("legacy_all_endpoints", "all_endpoints", "complete_steady", "steady_and_spectrum")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def full_residual_record(geom, params, u, v):
    """Measure BOTH equations on the actual returned float64 fields and actual L/M."""
    import numpy as np
    from .experiment_cache import array_identity
    from .result_io import geometry_identity
    diagnostic = full_steady_residual(geom, params, u, v)
    return {"diagnostic_version": FULL_STEADY_RESIDUAL_VERSION, "formula": FULL_STEADY_RESIDUAL_FORMULA,
            "normalization": "F", "rotation_projection": False, "parameters": params.as_dict(),
            "value": diagnostic["full_joint_rms_over_F"] if finite(diagnostic["full_joint_rms_over_F"]) else None,
            "Ru_rms": diagnostic["Ru_rms"] if finite(diagnostic["Ru_rms"]) else None,
            "Rv_rms": diagnostic["Rv_rms"] if finite(diagnostic["Rv_rms"]) else None,
            "geometry": geometry_identity(geom), "state_sha256": array_identity(np.asarray(u), np.asarray(v)),
            "state_dtype": [str(np.asarray(x).dtype) for x in (u, v)],
            "scope": "original_returned_fields_not_quantized_cache"}


def strict_full_steady(record):
    """Missing/ambiguous residuals or 'polished' alone are insufficient."""
    d = record.get("full_steady_diagnostic")
    if not isinstance(d, dict):
        return False
    if (d.get("diagnostic_version") != FULL_STEADY_RESIDUAL_VERSION or
            d.get("formula") != FULL_STEADY_RESIDUAL_FORMULA or d.get("normalization") != "F" or
            d.get("rotation_projection") is not False or d.get("state_dtype") != ["float64", "float64"]):
        return False
    a, b, value = d.get("Ru_rms"), d.get("Rv_rms"), d.get("value")
    parameters = d.get("parameters")
    if not isinstance(parameters, dict):
        return False
    F = parameters.get("F")
    if any(key in record and record[key] != parameters.get(key) for key in ("F", "k", "Du", "Dv")):
        return False
    if "u" in record or "v" in record:
        import numpy as np
        from .experiment_cache import array_identity
        if "u" not in record or "v" not in record or any(np.asarray(record[key]).dtype != np.float64 for key in ("u", "v")):
            return False
        if d.get("state_sha256") != array_identity(np.asarray(record["u"]), np.asarray(record["v"])):
            return False
    if not all(finite(x) and x >= 0 for x in (a, b, value)) or not finite(F) or F <= 0:
        return False
    expected = math.hypot(a, b) / math.sqrt(2.) / F
    return bool(0 <= value <= STEADY_TOL and math.isclose(value, expected, rel_tol=16 * math.ulp(1.), abs_tol=0.))


def label_quality(row):
    valid = not row.get("error") and row.get("type") in ("uniform", "spots", "holes", "stripes_or_maze")
    for key in ("v_min", "v_max"):
        if key in row and not finite(row[key]):
            valid = False
    d = row.get("full_steady_diagnostic")
    value = d.get("value") if isinstance(d, dict) else row.get("rel_residual")
    # Explicit compatibility adapter for historical ai_generation label files.
    complete = strict_full_steady(row) if isinstance(d, dict) else finite(value) and 0 <= value <= STEADY_TOL
    sample = row.get("linear_stability_sample", {})
    report = sample.get("spectral_report", sample) if isinstance(sample, dict) else {}
    decision = evaluate_spectra([report] if report else [], value,
        configs=INVERSION_SCREEN_CONFIGS, scope="label_single_shift_k8_finite_check")
    passed = bool(valid and complete and decision["numerical_check_passed"])
    return {"quality_protocol": QUALITY_VERSION, "finite_time_state": bool(valid),
            "complete_steady": bool(valid and complete),
            "spectrum_checked_pass": passed,
            "spectrum_positive": bool(decision["found_reliable_positive"]),
            "spectrum_insufficient": not passed and not decision["found_reliable_positive"],
            "spectral_decision": decision,
            "residual_evidence": "full_dual_equation_diagnostic" if isinstance(d, dict) else
                                 "historical_saved_scalar_not_new_field_readback",
            "spectrum_scope": "finite_sample_not_full_spectrum_proof"}


def select_training_rows(rows, policy="legacy_all_endpoints"):
    if policy not in TRAINING_POLICIES:
        raise ValueError(f"unknown quality policy: {policy}")
    rows = list(rows)
    if policy == "legacy_all_endpoints":
        return rows  # Deliberately identical original v1/v2 protocol.
    key = {"all_endpoints": "finite_time_state", "complete_steady": "complete_steady",
           "steady_and_spectrum": "spectrum_checked_pass"}[policy]
    return [r for r in rows if label_quality(r)[key]]


def quality_summary(rows):
    rows = list(rows); q = [label_quality(r) for r in rows]
    keys = ("finite_time_state", "complete_steady", "spectrum_checked_pass", "spectrum_positive", "spectrum_insufficient")
    groups = defaultdict(list)
    for r, flags in zip(rows, q):
        groups[(r.get("surface"), r["F"], r["k"])].append((r, flags))
    mixed, steady_mixed = [], []
    for (surface, F, k), values in groups.items():
        types = {r["type"] for r, _ in values}
        steady_types = {r["type"] for r, flags in values if flags["complete_steady"]}
        if len(types) > 1:
            mixed.append({"surface": surface, "F": F, "k": k, "types": sorted(types),
                          "seeds": [r.get("seed") for r, _ in values]})
        if len(steady_types) > 1:
            steady_mixed.append({"surface": surface, "F": F, "k": k, "types": sorted(steady_types)})
    return {"quality_protocol": QUALITY_VERSION, "total": len(rows),
            **{key: sum(v[key] for v in q) for key in keys},
            "not_complete_steady": sum(not v["complete_steady"] for v in q),
            "types": dict(Counter(r.get("type") for r in rows)),
            "steady_types": dict(Counter(r.get("type") for r, flags in zip(rows, q) if flags["complete_steady"])),
            "mixed_terminal_type_points": mixed, "mixed_complete_steady_type_points": steady_mixed,
            "multistability_limitation": "different finite-time types alone do not prove steady multistability",
            "spectral_limitation": "summary-only historical samples, including unknown requested counts, are insufficient under this protocol"}
