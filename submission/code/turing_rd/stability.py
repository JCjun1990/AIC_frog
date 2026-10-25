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


from .spectral_policy import (PAIR_ABSOLUTE_TOL, PAIR_RELATIVE_TOL, POSITIVE_FLOOR,
                              POLICY_VERSION, SIGN_ERROR_FACTOR, SPHERE_CONFIGS,
                              evaluate_spectra)

STABILITY_DIAGNOSTIC_VERSION = "unprojected-mass-spectrum-v1"
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
        positive = bool(valid and value.real > max(SIGN_ERROR_FACTOR * absolute, POSITIVE_FLOOR))
        negative = bool(valid and value.real < -max(SIGN_ERROR_FACTOR * absolute, POSITIVE_FLOOR))
        rows.append({
            "eigenvalue": [float(value.real), float(value.imag)],
            "residual_absolute_M": absolute,
            "residual_relative_M": relative,
            "residual_passed": valid,
            "rotation_overlap": overlap,
            "rotation_like": bool(overlap >= 0.5),
            "positive_reliable": positive,
            "negative_resolved": negative,
            "sign_resolution_margin": max(SIGN_ERROR_FACTOR * absolute, POSITIVE_FLOOR),
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
        "acceptance_policy_version": POLICY_VERSION,
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


def aggregate_spectra(reports, full_residual, tol=1e-11, *, configs=SPHERE_CONFIGS):
    """Apply the trusted sphere protocol; experiments must predeclare other counts.

    Two shifts and direct LR are mandatory for formal sphere acceptance. Merely
    passing a subset, or changing the requested counts in a record, cannot pass.
    """
    decision = evaluate_spectra(reports, full_residual, configs=configs,
                                scope="sphere_finite_numerical_check", tol=tol)
    decision.update(diagnostic_version=STABILITY_DIAGNOSTIC_VERSION,
                    limitation=SPECTRUM_LIMITATION)
    return decision
