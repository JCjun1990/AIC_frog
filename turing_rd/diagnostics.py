"""Residuals, stability, pattern statistics and scale/units checks (all float64 NumPy)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components

from .geometry import Geometry
from .solver import GSParams


def _np(x):
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().double().numpy()
    except ImportError:
        pass
    return np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------------------------------------
# PDE residual
# ---------------------------------------------------------------------------------------------
def pde_residual(geom: Geometry, p: GSParams, u, v, u_prev=None, v_prev=None, dt: float | None = None) -> dict:
    """Continuous-in-time residual of both equations at state (u, v).

    R_u = Du Delta u - u v^2 + F (1 - u) - du/dt   (du/dt from the last step, 0 if not given)
    Reported as area-weighted RMS, and relative to the RMS of the reaction term so that
    numbers are comparable between parameter sets. At a steady state R -> 0 and du/dt -> 0.
    """
    u, v = _np(u), _np(v)
    r = u * v * v
    fu, fv = -r + p.F * (1 - u), r - (p.F + p.k) * v
    Ru = p.Du * geom.lap(u) + fu
    Rv = p.Dv * geom.lap(v) + fv
    out = {}
    if u_prev is not None and dt:
        dudt, dvdt = (u - _np(u_prev)) / dt, (v - _np(v_prev)) / dt
        out.update(dudt_rms=geom.rms(dudt), dvdt_rms=geom.rms(dvdt))
        Ru_t, Rv_t = Ru - dudt, Rv - dvdt
        out.update(Ru_rms=geom.rms(Ru_t), Rv_rms=geom.rms(Rv_t))
    else:
        out.update(Ru_rms=geom.rms(Ru), Rv_rms=geom.rms(Rv))
    # steady residual = du/dt at a fixed point (no time derivative subtracted)
    out.update(steady_Ru_rms=geom.rms(Ru), steady_Rv_rms=geom.rms(Rv),
               reaction_scale=geom.rms(np.abs(fu)) + geom.rms(np.abs(fv)) + 1e-30)
    # relative to the reaction terms, floored at the feed rate F so the trivial state (1, 0) is not 0/0
    out["steady_rel"] = (out["steady_Ru_rms"] + out["steady_Rv_rms"]) / max(out["reaction_scale"], p.F)
    return out


# ---------------------------------------------------------------------------------------------
# linear stability of a (near) steady state
# ---------------------------------------------------------------------------------------------
def jacobian(geom: Geometry, p: GSParams, u, v):
    """Strong-form Jacobian J of the semi-discrete system d[u;v]/dt = J [u;v] around (u, v)."""
    u, v = _np(u), _np(v)
    Minv = sp.diags(1.0 / geom.M)
    D = Minv @ geom.L
    fu, fv = -v * v - p.F, -2 * u * v
    gu, gv = v * v, 2 * u * v - (p.F + p.k)
    return sp.bmat([[p.Du * D + sp.diags(fu), sp.diags(fv)],
                    [sp.diags(gu), p.Dv * D + sp.diags(gv)]]).tocsc()


def linear_stability(geom: Geometry, p: GSParams, u, v, k: int = 8, sigma: float = 1e-3,
                     rotations=None) -> dict:
    """Eigenvalues of J closest to `sigma` (shift-invert ARPACK).

    A stable steady pattern has max Re(lambda) <= 0. On the sphere, three eigenvalues ~ 0 are
    expected: rigid rotations of the pattern (a check that the solver respects the symmetry).
    With `rotations` = [T_1, T_2, T_3] (steady.rotation_generators), k + r eigenpairs are computed
    (r = number of rotations that move the pattern: 3, 2 if axisymmetric, 0 if uniform); the eigenvectors
    closest to span{T_i U} (M-weighted projection >= 0.5, at most r) are reported separately as the
    rotation modes, and max_real is the largest real part of the remaining eigenvalues.
    """
    J = jacobian(geom, p, u, v)
    if rotations is None:
        lam = spla.eigs(J, k=k, sigma=sigma, which="LM", return_eigenvectors=False, maxiter=5000, tol=1e-10)
        lam = lam[np.argsort(-lam.real)]
        return {"eigenvalues": [[float(z.real), float(z.imag)] for z in lam],
                "max_real": float(lam.real.max()), "sigma": sigma}
    u, v = _np(u), _np(v)
    Mf = np.concatenate([geom.M, geom.M])
    B = np.sqrt(Mf)[:, None] * np.stack([np.concatenate([Ti @ u, Ti @ v]) for Ti in rotations], axis=1)
    Q, sv, _ = np.linalg.svd(B, full_matrices=False)               # M-orthonormal basis of the orbit tangent
    U = np.sqrt(Mf) * np.concatenate([u, v])
    r = int((sv > 1e-3 * max(sv[0], 1e-6 * np.linalg.norm(U))).sum())   # 0 (uniform), 2 (axisymmetric) or 3
    lam, W = spla.eigs(J, k=k + r, sigma=sigma, which="LM", maxiter=5000, tol=1e-10)
    Ws = np.sqrt(Mf)[:, None] * W
    overlap = np.linalg.norm(Q[:, :r].T @ Ws, axis=0) / np.linalg.norm(Ws, axis=0)
    rot = [i for i in np.argsort(-overlap)[:r] if overlap[i] >= 0.5]  # e.g. 2 for an axisymmetric pattern
    rest = np.setdiff1d(np.arange(len(lam)), rot)
    order = rest[np.argsort(-lam[rest].real)]
    return {"eigenvalues": [[float(lam[i].real), float(lam[i].imag)] for i in order],
            "rotation_eigenvalues": [[float(lam[i].real), float(lam[i].imag)] for i in rot],
            "rotation_overlap": [float(overlap[i]) for i in rot],
            "max_overlap_other": float(overlap[rest].max()),
            "max_real": float(lam[rest].real.max()), "max_real_all": float(lam.real.max()), "sigma": sigma}


# ---------------------------------------------------------------------------------------------
# pattern statistics
# ---------------------------------------------------------------------------------------------
def contour_segments(geom: Geometry, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero level set of the piecewise-linear field s (marching triangles): face indices, segment lengths."""
    V, F = geom.V, geom.F
    sf = s[F]
    mixed = np.flatnonzero((sf.max(1) > 0) & (sf.min(1) <= 0))
    if not len(mixed):
        return mixed, np.zeros(0)
    Fm, sm = F[mixed], sf[mixed]
    pts = []
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        sa, sb = sm[:, a], sm[:, b]
        cross = (sa > 0) != (sb > 0)
        t = np.where(cross, sa / np.where(cross, sa - sb, 1.0), np.nan)
        pts.append(V[Fm[:, a]] + t[:, None] * (V[Fm[:, b]] - V[Fm[:, a]]))
    P = np.stack(pts, 1)                                   # (m, 3 edges, 3)
    ok = ~np.isnan(P[..., 0])
    two = ok.sum(1) == 2
    P2 = P[two][ok[two]].reshape(-1, 2, 3)
    return mixed[two], np.linalg.norm(P2[:, 0] - P2[:, 1], axis=1)


def _contour_length(geom: Geometry, s: np.ndarray) -> float:
    """Total length of the zero level set of the piecewise-linear field s (marching triangles)."""
    return float(contour_segments(geom, s)[1].sum())


def pattern_stats(geom: Geometry, v, threshold: float | None = None, min_size: int = 3) -> dict:
    """Morphology of the high-v phase: count, area fraction, Euler characteristic, compactness, scale.

    compactness Q = 4 pi A n / P^2 (A, P total area / perimeter of the high phase, n components):
    1 for round spots, << 1 for stripes or labyrinths. Wavelength from the Dirichlet ratio:
    lambda = 2 pi / sqrt(int |grad v|^2 / int (v - mean)^2).
    """
    v = _np(v)
    q05, q95 = np.quantile(v, [0.05, 0.95])
    out = {"v_min": float(v.min()), "v_max": float(v.max()), "contrast": float(q95 - q05)}
    g = v - geom.mean(v)
    N = float(geom.M @ (g * g))
    out["wavelength"] = float(2 * np.pi / np.sqrt(max(-(g @ (geom.L @ g)) / N, 1e-30))) if N > 1e-20 else np.inf
    if q95 - q05 < 1e-3:
        out.update(type="uniform", n_components=0, area_fraction=0.0, euler=0, compactness=0.0, threshold=None)
        return out
    thr = 0.5 * (q05 + q95) if threshold is None else threshold
    hi = v > thr
    sub = geom.adjacency[hi][:, hi]
    nc, lab = connected_components(sub, directed=False)
    sizes = np.bincount(lab, minlength=nc) if nc else np.zeros(0, int)
    n_comp = int((sizes >= min_size).sum())
    lo_sub = geom.adjacency[~hi][:, ~hi]
    nc_lo, lab_lo = connected_components(lo_sub, directed=False)
    n_lo = int((np.bincount(lab_lo, minlength=nc_lo) >= min_size).sum()) if nc_lo else 0
    E_in = int((hi[geom.edges[:, 0]] & hi[geom.edges[:, 1]]).sum())
    F_in = int(hi[geom.F].all(1).sum())
    euler = int(hi.sum() - E_in + F_in)
    A = float(geom.M[hi].sum())
    P = _contour_length(geom, v - thr)
    Q = 4 * np.pi * A * max(n_comp, 1) / P ** 2 if P > 0 else 0.0
    frac = A / geom.area
    if n_comp >= 3 and Q > 0.45 and frac < 0.5:
        kind = "spots"
    elif n_lo >= 3 and frac > 0.5 and euler < 0:
        kind = "holes"
    else:
        kind = "stripes_or_maze"
    out.update(type=kind, threshold=float(thr), n_components=n_comp, n_low_components=n_lo,
               area_fraction=frac, euler=euler, perimeter=P, compactness=float(Q),
               mean_component_area=A / max(n_comp, 1))
    return out


FEATURE_TOL = {"wavelength": 0.10, "n_components": 0.25, "area_fraction": 0.05, "compactness": 0.15,
               "v_max": 0.20}


def feature_distance(s: dict, target: dict, tol: dict = FEATURE_TOL) -> tuple[float, float]:
    """Statistical distance between two patterns, in units of the tolerances.

    shape = RMS of (relative wavelength error / 10 %, relative count error / 25 %,
                    area-fraction error / 0.05, compactness error / 0.15), +inf if the type differs.
    peak  = relative error of max v / 20 % (reported separately: amplitudes may carry solver artefacts).
    shape <= 1 means every shape feature is roughly within tolerance.
    """
    if s.get("type") != target.get("type") or not s.get("n_components"):
        return float("inf"), float("inf")
    d = [np.log(s["wavelength"] / target["wavelength"]) / tol["wavelength"],
         (s["n_components"] - target["n_components"]) / (tol["n_components"] * target["n_components"]),
         (s["area_fraction"] - target["area_fraction"]) / tol["area_fraction"],
         (s["compactness"] - target["compactness"]) / tol["compactness"]]
    peak = abs(s["v_max"] / target["v_max"] - 1) / tol["v_max"]
    return float(np.sqrt(np.mean(np.square(d)))), float(peak)


def lb_power_spectrum(geom: Geometry, f, k: int = 200, bin_width: float = 1.0):
    """Power of f in Laplace-Beltrami eigen-modes, binned by wavenumber kappa = sqrt(lambda)."""
    lam, phi = geom.eigenbasis(k)
    f = _np(f)
    c = phi.T @ (geom.M * (f - geom.mean(f)))
    kappa = np.sqrt(np.maximum(lam, 0))
    bins = np.arange(0, kappa.max() + bin_width, bin_width)
    power = np.bincount(np.digitize(kappa, bins) - 1, c * c, minlength=len(bins))[:len(bins)]
    return bins + 0.5 * bin_width, power


# ---------------------------------------------------------------------------------------------
# scales, units and step-size limits
# ---------------------------------------------------------------------------------------------
def scales(geom: Geometry, p: GSParams, dt: float) -> dict:
    """Non-dimensional groups for the units/dimension check.

    Units: length = model length unit (sphere radius 1), time = reaction time unit;
    [Du] = [Dv] = L^2/T, [F] = [k] = 1/T, u and v dimensionless. Every term is 1/T.
    """
    ell = np.sqrt(p.Du / p.F)
    R_eq = np.sqrt(geom.area / (4 * np.pi))
    return {
        "diffusion_length_ell": float(ell), "reaction_time_tau": float(1 / p.F),
        "equivalent_radius": float(R_eq), "radius_over_ell": float(R_eq / ell),
        "h_median_over_ell": float(geom.h_med / ell),
        "explicit_dt_limit": float(geom.h_min ** 2 / (4 * max(p.Du, p.Dv))),
        "dt_over_tau": float(dt * p.F), "diffusion_number_median": float(p.Du * dt / geom.h_med ** 2),
        "uniform_equilibrium_exists": bool(p.F >= 4 * (p.F + p.k) ** 2),
    }
