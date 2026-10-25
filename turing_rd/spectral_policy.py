"""Trusted finite-sample acceptance policy, independent of certificate tolerances.

Threshold provenance (unchanged): stability.eigenpair_diagnostics / aggregate_spectra
and i03_frog_stability.decide_frog at f643a04. Sphere formal: two shifts + direct
LR; frog formal: three shifts, with its additional 1e-9 near-zero exclusion.
Candidate screens have explicitly smaller scopes and never certify a whole spectrum.
Residuals are backward errors, not rigorous forward eigenvalue error bounds.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import NamedTuple

POLICY_VERSION = "finite-spectrum-acceptance-v1"
PAIR_ABSOLUTE_TOL = 1e-9
PAIR_RELATIVE_TOL = 1e-7
STEADY_TOL = 1e-11
POSITIVE_FLOOR = 1e-12
FROG_SIGN_FLOOR = 1e-9
SIGN_ERROR_FACTOR = 10
MAXIMA_ABSOLUTE_TOL = 1e-10
MAXIMA_ERROR_FACTOR = 100


class SpectrumConfig(NamedTuple):
    sigma: float | None
    which: str
    k: int


SPHERE_CONFIGS = (SpectrumConfig(1e-3, "LM", 12), SpectrumConfig(0., "LM", 20),
                  SpectrumConfig(None, "LR", 8))
FROG_CONFIGS = (SpectrumConfig(1e-3, "LM", 12), SpectrumConfig(0., "LM", 20),
                SpectrumConfig(1e-5, "LM", 8))
CANDIDATE_CONFIGS = (SpectrumConfig(0., "LM", 12), SpectrumConfig(1e-3, "LM", 12))
INVERSION_SCREEN_CONFIGS = (SpectrumConfig(1e-3, "LM", 8),)


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def failed_spectrum(config, error):
    """Keep an exception visible as missing evidence for a predeclared configuration."""
    return {"sigma": config.sigma, "which": config.which, "requested_k": config.k,
            "returned_k": 0, "pairs": [], "solver_converged": False,
            "all_pairs_validated": False, "full_max_real": None,
            "solver_error": str(error), "found_reliable_positive": False}


def evaluate_spectra(reports, full_residual, *, configs, scope,
                     tol=STEADY_TOL, sign_floor=POSITIVE_FLOOR):
    """Fail closed using caller-declared configs and version-controlled thresholds.

    Recompute residual/sign tests from numbers, never from editable pass flags.
    Even a partial solve can disprove stability with a validated positive pair;
    it cannot supply a passing configuration. Historical summaries without pairs
    remain insufficient evidence. This function does not rerun eigenvectors.
    """
    if not finite_number(tol) or not 0 < tol <= STEADY_TOL:
        raise ValueError("full-residual tolerance may not exceed the trusted policy")
    if sign_floor not in (POSITIVE_FLOOR, FROG_SIGN_FLOOR):
        raise ValueError("unknown sign policy")
    configs = tuple(configs)
    if not configs or len(set(configs)) != len(configs):
        raise ValueError("a nonempty unique predeclared configuration protocol is required")
    reasons, positives, unresolved, maxima, errors, valid_configs = [], [], [], [], [], []
    def fail(reason):
        if reason not in reasons:
            reasons.append(reason)
    steady = bool(finite_number(full_residual) and 0 <= full_residual <= tol)
    if not steady:
        fail("full_stationary_residual_failed")
    if not isinstance(reports, (list, tuple)):
        reports = []
    identities, pair_count, all_negative = [], 0, True
    for i, run in enumerate(reports):
        prefix = f"configuration_{i}:"
        before = len(reasons)
        if not isinstance(run, dict):
            fail(prefix + "malformed_record")
            continue
        sigma, which, k = run.get("sigma"), run.get("which"), run.get("requested_k")
        identity = None
        if ((sigma is None or finite_number(sigma)) and isinstance(which, str)
                and isinstance(k, int) and not isinstance(k, bool) and k > 0):
            identity = SpectrumConfig(sigma, which, k)
            identities.append(identity)
        else:
            fail(prefix + "missing_configuration_identity")
        if run.get("solver_converged") is not True or run.get("solver_error"):
            fail(prefix + "solver_not_converged")
        if run.get("all_pairs_validated") is not True:
            fail(prefix + "pair_validation_incomplete")
        for key, cap in (("pair_absolute_tolerance", PAIR_ABSOLUTE_TOL),
                         ("pair_relative_tolerance", PAIR_RELATIVE_TOL)):
            value = run.get(key)
            if not finite_number(value) or not 0 < value <= cap:
                fail(prefix + "untrusted_" + key)
        pairs = run.get("pairs", [])
        if not isinstance(pairs, list):
            pairs = []
        returned = run.get("returned_k")
        if (identity not in configs or len(pairs) != k or not pairs
                or not isinstance(returned, int) or isinstance(returned, bool)
                or returned != len(pairs)):
            fail(prefix + "eigenpair_count_or_protocol_mismatch")
        if not pairs:
            all_negative = False
            fail(prefix + "missing_eigenpair_residuals")
        real_parts = []
        for pair in pairs:
            pair_count += 1
            if not isinstance(pair, dict):
                all_negative = False
                fail(prefix + "malformed_eigenpair")
                continue
            ev = pair.get("eigenvalue")
            a, r = pair.get("residual_absolute_M"), pair.get("residual_relative_M")
            if (not isinstance(ev, (list, tuple)) or len(ev) != 2
                    or not all(finite_number(x) for x in (*ev, a, r)) or a < 0 or r < 0):
                all_negative = False
                fail(prefix + "nonfinite_or_missing_pair_metrics")
                continue
            real_parts.append(ev[0])
            valid = a <= PAIR_ABSOLUTE_TOL and r <= PAIR_RELATIVE_TOL
            if not valid or pair.get("residual_passed") is not True:
                fail(prefix + "eigenpair_residual_failed")
            margin = max(SIGN_ERROR_FACTOR * a, POSITIVE_FLOOR)
            positive = valid and ev[0] > margin
            negative = valid and ev[0] < -max(margin, sign_floor)
            if positive:
                positives.append(list(ev))
            if valid:
                errors.append(a)
                if not positive and not negative:
                    unresolved.append(list(ev))
            if not negative:
                all_negative = False
            # Flags can reject a record, but cannot overrule numerical metrics.
            if (pair.get("positive_reliable") is not (valid and ev[0] > margin)
                    or pair.get("negative_resolved") is not (valid and ev[0] < -margin)):
                fail(prefix + "pair_sign_flags_contradict_metrics")
        if real_parts:
            maximum = max(real_parts)
            maxima.append(maximum)
            if (not finite_number(run.get("full_max_real")) or not math.isclose(
                    maximum, run["full_max_real"], rel_tol=1e-8, abs_tol=1e-12)):
                fail(prefix + "sampled_maximum_contradicts_pairs")
        if len(reasons) == before:
            valid_configs.append(identity)
    if Counter(identities) != Counter(configs):
        fail("required_configuration_set_incomplete_or_duplicated")
    complete = len(valid_configs) == len(configs) and Counter(valid_configs) == Counter(configs)
    agreement_tol = max(MAXIMA_ABSOLUTE_TOL, MAXIMA_ERROR_FACTOR * max(errors, default=0.))
    agree = bool(len(maxima) == len(configs) and max(maxima) - min(maxima) <= agreement_tol)
    if not agree:
        fail("sampled_largest_real_parts_disagree")
    if positives:
        fail("reliable_positive_full_jacobian_eigenvalue")
    if not all_negative or pair_count == 0:
        fail("not_all_checked_eigenvalue_signs_resolved_negative")
    passed = not reasons and steady and complete and agree and all_negative
    status = ("unstable_positive_mode" if positives else "full_steady_residual_failed" if not steady
              else "numerical_check_passed" if passed else "insufficient_evidence")
    return {"acceptance_policy_version": POLICY_VERSION, "scope": scope,
            "required_configurations": [c._asdict() for c in configs],
            "full_steady_state": steady,
            "full_residual": full_residual if finite_number(full_residual) else None,
            "full_residual_tolerance": tol, "found_reliable_positive": bool(positives),
            "reliable_positive_eigenvalues": positives,
            "unresolved_near_zero_eigenvalues": unresolved, "checked_eigenpairs": pair_count,
            "full_max_real_checked": max(maxima) if maxima else None,
            "sampled_maxima": maxima, "configurations": len(reports),
            "distinct_valid_shifts": sorted({c.sigma for c in valid_configs if c.sigma is not None}),
            "valid_direct_LR_runs": sum(c.sigma is None and c.which == "LR" for c in valid_configs),
            "all_requested_solves_and_pairs_validated": complete,
            "sampled_maxima_agree": agree, "sampled_maxima_agreement_tolerance": agreement_tol,
            "all_checked_signs_resolved_negative": bool(pair_count and all_negative),
            "numerical_check_passed": passed, "status": status,
            "outcome": "passed" if passed else "failed" if positives or not steady else "insufficient",
            "failure_reasons": reasons, "relative_equilibrium_accepted_as_steady": False,
            "rotation_projection": False, "rotations_excluded_from_full_decision": False,
            "mathematical_full_spectrum_proof": False}


def candidate_decision(row):
    """Recheck old rows without upgrading their summary-only evidence."""
    return evaluate_spectra(row.get("spectra", []), row.get("full_residual"),
                            configs=CANDIDATE_CONFIGS, scope="two_shift_candidate_screen")


def screen_candidate(geom, params, u, v, full_residual, solve):
    """Bounded two-configuration screen shared by both sphere candidate scripts."""
    reports = []
    if finite_number(full_residual) and 0 <= full_residual <= STEADY_TOL:
        for cfg in CANDIDATE_CONFIGS:
            try:
                report, _ = solve(geom, params, u, v, k=cfg.k, sigma=cfg.sigma, which=cfg.which)
            except Exception as exc:
                report = failed_spectrum(cfg, repr(exc))
            reports.append(report)
    decision = candidate_decision({"spectra": reports, "full_residual": full_residual})
    return {"spectra": reports, "spectral_decision": decision,
            "stability_status": decision["status"], "failure_reasons": decision["failure_reasons"],
            "max_real": decision["full_max_real_checked"],
            "screen_passed": decision["numerical_check_passed"],
            "stable": decision["numerical_check_passed"],  # compatibility alias, finite screen only
            "full_discrete_stability_certified": False}
