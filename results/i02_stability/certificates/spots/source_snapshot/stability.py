"""Unprojected, mass-weighted numerical linear-stability checks.

Rotational overlap labels eigenvectors; it never removes a positive eigenvalue
from the full-system decision. Finite ARPACK samples are numerical evidence,
not a proof that the whole spectrum lies in the left half plane.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .diagnostics import jacobian


STABILITY_DIAGNOSTIC_VERSION = "unprojected-mass-spectrum-v1"
PAIR_ABSOLUTE_TOL = 1e-9
PAIR_RELATIVE_TOL = 1e-7
POSITIVE_FLOOR = 1e-12
SPECTRUM_LIMITATION = (
    "Finite eigenpair sampling with multiple shifts and direct largest-real-part "
    "ARPACK is a numerical check, not a mathematical full-spectrum proof. "
    "Rotational overlap is descriptive and never suppresses a positive mode."
)


def rotation_subspace(geom, u, v, rotations=None):
    """Return orthonormal orbit tangent in sqrt(M)-weighted coordinates."""
    from .steady import rotation_generators

    u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
    if u.shape != (geom.n,) or v.shape != (geom.n,):
        raise ValueError("Rotation subspace requires flat vertex fields")
    rotations = rotation_generators(geom) if rotations is None else rotations
    mass = np.concatenate([geom.M, geom.M])
    root = np.sqrt(mass)
    if not rotations:
        return np.zeros((2 * geom.n, 0)), {
            "rank": 0, "singular_values": [], "rank_relative_threshold": 1e-3,
            "basis_coordinates": "sqrt(Mfull)*[T_i*u;T_i*v]",
            "used_to_project_jacobian": False,
        }
    tangent = np.stack([np.concatenate([T @ u, T @ v]) for T in rotations], axis=1)
    Q, singular, _ = np.linalg.svd(root[:, None] * tangent, full_matrices=False)
    scale = max(float(singular[0]), 1e-6 * np.linalg.norm(root * np.concatenate([u, v])))
    rank = int(np.count_nonzero(singular > 1e-3 * scale))
    return Q[:, :rank], {
        "rank": rank, "singular_values": singular.tolist(),
        "rank_relative_threshold": 1e-3,
        "basis_coordinates": "sqrt(Mfull)*[T_i*u;T_i*v]",
        "used_to_project_jacobian": False,
        "overlap_label_threshold": 0.5,
        "interpretation": "near-rotation tangent; finite mesh need not have exact SO(3) zero modes",
    }


def eigenpair_diagnostics(J, lam, W, mass, rotation_basis=None):
    """Independent original-operator residuals for eigenvectors in field coordinates.

    absolute = ||Jw-lambda*w||_M / ||w||_M [1/time]
    relative = ||Jw-lambda*w||_M / (||Jw||_M + |lambda|*||w||_M)
    The absolute test prevents the large high-frequency diffusion scale from
    concealing errors of the weak near-zero modes.
    """
    mass = np.asarray(mass, dtype=np.float64)
    lam, W = np.asarray(lam), np.asarray(W)
    if mass.shape != (J.shape[0],) or np.any(mass <= 0) or not np.isfinite(mass).all():
        raise ValueError("Expected positive finite mass per Jacobian row")
    if W.shape != (J.shape[0], len(lam)):
        raise ValueError("Eigenvector dimensions disagree with eigenvalues or Jacobian")
    root = np.sqrt(mass)
    Q = rotation_basis
    if Q is not None and Q.shape[0] != J.shape[0]:
        raise ValueError("Rotation basis must use sqrt(Mfull)-weighted coordinates")
    rows = []
    for i, value in enumerate(lam):
        w = W[:, i]
        Jw = J @ w
        norm = float(np.linalg.norm(root * w))
        raw = float(np.linalg.norm(root * (Jw - value * w)))
        denom = float(np.linalg.norm(root * Jw)) + abs(value) * norm
        absolute = raw / norm if norm > 0 else float("inf")
        relative = raw / max(denom, np.finfo(float).tiny)
        finite = bool(np.isfinite(value) and np.isfinite(w).all() and
                      np.isfinite(absolute) and np.isfinite(relative) and norm > 0)
        valid = bool(finite and absolute <= PAIR_ABSOLUTE_TOL and relative <= PAIR_RELATIVE_TOL)
        overlap = float(np.linalg.norm(Q.conj().T @ (root * w)) / norm) if Q is not None and norm > 0 else 0.0
        # Numerical roundoff can give 1+epsilon for a unit projection.
        overlap = min(overlap, 1.0)
        positive = bool(valid and value.real > max(10 * absolute, POSITIVE_FLOOR))
        negative = bool(valid and value.real < -max(10 * absolute, POSITIVE_FLOOR))
        rows.append({
            "eigenvalue": [float(value.real), float(value.imag)],
            "residual_absolute_M": absolute,
            "residual_relative_M": relative,
            "residual_passed": valid,
            "rotation_overlap": overlap,
            "rotation_like": bool(overlap >= 0.5),
            "positive_reliable": positive,
            "negative_resolved": negative,
            "sign_resolution_margin": max(10 * absolute, POSITIVE_FLOOR),
        })
    return rows


def solve_spectrum(geom, p, u, v, k=12, sigma=1e-3, which="LM",
                   tol=1e-10, maxiter=3000, ncv=None, seed=2026):
    """Compute unprojected eigenpairs; return report and field-coordinate W.

    B = sqrt(Mfull) J invsqrt(Mfull) is similar to the strong Jacobian.
    This is not an extra mass normalization of the eigenproblem.
    ``sigma=None, which='LR'`` requests a direct largest-real-part search.
    An ARPACK partial result is retained but never marked solver-converged.
    """
    started = time.perf_counter()
    J = jacobian(geom, p, u, v)
    dimension = J.shape[0]
    if not 1 <= int(k) < dimension - 1:
        raise ValueError("ARPACK requires 1 <= k < dimension-1")
    if tol <= 0 or maxiter < 1:
        raise ValueError("Positive tol/maxiter are required")
    if sigma is not None and not np.isfinite(sigma):
        raise ValueError("Shift must be finite")
    if ncv is None:
        ncv = min(dimension, max(2 * int(k) + 1, 40))
    if not int(k) + 1 < int(ncv) <= dimension:
        raise ValueError("ncv must be greater than k+1 and <= dimension")
    mass = np.concatenate([geom.M, geom.M])
    if np.any(mass <= 0) or not np.isfinite(mass).all():
        raise ValueError("Invalid PDE mass matrix")
    root = np.sqrt(mass)
    B = (sp.diags(root) @ J @ sp.diags(1.0 / root)).tocsc()
    v0 = np.random.default_rng(seed).standard_normal(dimension)
    v0 /= np.linalg.norm(v0)
    converged, error = True, None
    try:
        lam, weighted_vectors = spla.eigs(
            B, k=int(k), sigma=sigma, which=which, tol=float(tol),
            maxiter=int(maxiter), ncv=int(ncv), v0=v0,
        )
    except spla.ArpackNoConvergence as exc:
        converged, error = False, str(exc)
        lam = np.asarray(exc.eigenvalues) if exc.eigenvalues is not None else np.empty(0, complex)
        weighted_vectors = (np.asarray(exc.eigenvectors) if exc.eigenvectors is not None
                            else np.empty((dimension, 0), complex))
    order = np.argsort(-lam.real)
    lam, weighted_vectors = lam[order], weighted_vectors[:, order]
    W = weighted_vectors / root[:, None]
    Q, rotation = rotation_subspace(geom, u, v)
    pairs = eigenpair_diagnostics(J, lam, W, mass, rotation_basis=Q)
    # For comparison only: never used as the full-system stability decision.
    nonrotation = [row["eigenvalue"][0] for row in pairs if not row["rotation_like"]]
    report = {
        "diagnostic_version": STABILITY_DIAGNOSTIC_VERSION,
        "requested_k": int(k), "returned_k": len(lam),
        "solver_converged": bool(converged and len(lam) == int(k)),
        "solver_error": error,
        "method": "mass_similarity_arpack_shift_invert" if sigma is not None else "mass_similarity_arpack_direct",
        "similarity": "B=sqrt(Mfull)*J*invsqrt(Mfull); J=Mfull^-1*weak_J",
        "which": which, "sigma": float(sigma) if sigma is not None else None,
        "arpack_tolerance": float(tol), "arpack_maxiter": int(maxiter), "ncv": int(ncv), "v0_seed": int(seed),
        "pairs": pairs,
        "full_max_real": float(lam.real.max()) if len(lam) else None,
        "nonrotation_max_real": max(nonrotation) if nonrotation else None,
        "rotation": rotation,
        "all_pairs_validated": bool(pairs and all(row["residual_passed"] for row in pairs)),
        "found_reliable_positive": any(row["positive_reliable"] for row in pairs),
        "pair_absolute_tolerance": PAIR_ABSOLUTE_TOL,
        "pair_relative_tolerance": PAIR_RELATIVE_TOL,
        "absolute_residual_definition": "norm_M(J*w-lambda*w)/norm_M(w)",
        "relative_residual_definition": "norm_M(J*w-lambda*w)/(norm_M(J*w)+abs(lambda)*norm_M(w))",
        "wall_seconds": time.perf_counter() - started,
        "limitation": SPECTRUM_LIMITATION,
    }
    return report, W


def aggregate_spectra(reports, full_residual, tol=1e-11):
    """Conservative decision from independent shifts AND direct LR samples.

    A reliable positive pair disproves numerical stability even if another solve
    failed. Success requires full stationarity, all requested solves converged
    and validated, at least two distinct shifts, a direct LR solve, agreement
    of the sampled maxima, and resolved negative signs for all checked pairs.
    """
    if tol <= 0:
        raise ValueError("Positive full-residual tolerance is required")
    reports = list(reports)
    reasons = []
    steady = bool(np.isfinite(full_residual) and 0 <= full_residual <= tol)
    if not steady:
        reasons.append("full_stationary_residual_failed")
    positives = [row for report in reports for row in report.get("pairs", []) if row.get("positive_reliable")]
    complete = bool(reports and all(report.get("solver_converged") and
                                    report.get("all_pairs_validated") for report in reports))
    if not complete:
        reasons.append("solver_or_pair_validation_incomplete")
    shifts = {report.get("sigma") for report in reports if report.get("sigma") is not None and
              report.get("solver_converged") and report.get("all_pairs_validated")}
    direct = [report for report in reports if report.get("sigma") is None and report.get("which") == "LR"
              and report.get("solver_converged") and report.get("all_pairs_validated")]
    if len(shifts) < 2:
        reasons.append("fewer_than_two_valid_distinct_shifts")
    if not direct:
        reasons.append("no_valid_direct_largest_real_part_search")
    maxima = [report["full_max_real"] for report in reports if report.get("full_max_real") is not None]
    absolute_errors = [row["residual_absolute_M"] for report in reports for row in report.get("pairs", [])
                       if row.get("residual_passed")]
    agreement_tolerance = max(1e-10, 100 * max(absolute_errors, default=0.0))
    maxima_agree = bool(maxima and np.ptp(maxima) <= agreement_tolerance)
    if not maxima_agree:
        reasons.append("sampled_largest_real_parts_disagree")
    pairs = [row for report in reports for row in report.get("pairs", [])]
    negative = bool(pairs and all(row.get("negative_resolved", False) for row in pairs))
    if positives:
        reasons.append("reliable_positive_full_jacobian_eigenvalue")
    elif not negative:
        reasons.append("not_all_checked_eigenvalue_signs_resolved_negative")
    passed = bool(steady and complete and len(shifts) >= 2 and direct and maxima_agree and negative and not positives)
    return {
        "diagnostic_version": STABILITY_DIAGNOSTIC_VERSION,
        "full_steady_state": steady,
        "relative_equilibrium_accepted_as_steady": False,
        "full_residual": float(full_residual), "full_residual_tolerance": float(tol),
        "found_reliable_positive": bool(positives),
        "reliable_positive_eigenvalues": [row["eigenvalue"] for row in positives],
        "full_max_real_checked": max(maxima) if maxima else None,
        "distinct_valid_shifts": sorted(shifts),
        "valid_direct_LR_runs": len(direct),
        "all_requested_solves_and_pairs_validated": complete,
        "sampled_maxima_agree": maxima_agree,
        "sampled_maxima_agreement_tolerance": agreement_tolerance,
        "all_checked_signs_resolved_negative": negative,
        "numerical_check_passed": passed,
        "status": "unstable_positive_mode" if positives else ("numerical_check_passed" if passed else "stability_not_closed"),
        "failure_reasons": reasons,
        "rotations_excluded_from_full_decision": False,
        "mathematical_full_spectrum_proof": False,
        "limitation": SPECTRUM_LIMITATION,
    }
