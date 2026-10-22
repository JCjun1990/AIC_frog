"""AI initial guesses for steady Turing patterns: pattern statistics, field synthesis, learned map.

A steady Gray-Scott pattern is summarised by a few mesh-independent statistics:

    kappa0, gamma   centre and relative width of the ring that holds most of the power of v in the
                    Laplace-Beltrami spectrum (kappa = sqrt(eigenvalue))
    q_v             area-weighted quantiles of v at the probabilities PROBS (amplitude distribution;
                    the high-v area fraction separates spots from stripes)
    g_u             mean of u at those quantiles of v (the pointwise u-v relation)

Synthesis draws a Gaussian random field whose LB spectrum is a ring at (kappa0, gamma), maps it through
a Gaussian copula onto q_v and sets u = g_u(v). A small neural network (ai_generation.py) learns
(F, k) -> statistics from self-generated steady states on the given surface; the synthesized state is
then driven to an exact steady state by pseudo-transient continuation / Newton (steady.ptc_polish) and
checked for linear stability, so the PDE, not the network, guarantees the result.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from .geometry import Geometry

PROBS = ndtr(np.linspace(-3.0, 3.0, 31))       # quantile probabilities, dense in the tails
WINDOW = 0.02                                   # probability half-window for the u-v relation


def _weighted_sort(geom: Geometry, v):
    order = np.argsort(v)
    w = geom.M[order] / geom.area
    return order, np.cumsum(w) - 0.5 * w        # mid-point cumulative probability of each vertex


def spectrum(geom: Geometry, v, n_modes: int = 600):
    """Ring energy of v - mean(v) per LB mode: kappa_i and c_i^2 (M-orthonormal basis)."""
    lam, phi = geom.eigenbasis(n_modes)
    c = phi.T @ (geom.M * (v - geom.mean(v)))
    return np.sqrt(np.maximum(lam, 0.0)), c * c


def pattern_statistics(geom: Geometry, u, v, n_modes: int = 600) -> dict:
    kap, pw = spectrum(geom, v, n_modes)
    kap, pw = kap[1:], pw[1:]                                     # drop the constant mode
    grid = np.linspace(0.5, kap.max(), 400)
    width = 0.6
    ring = np.array([(pw * np.exp(-0.5 * ((kap - x) / width) ** 2)).sum() for x in grid])
    kappa0 = float(grid[np.argmax(ring)])
    near = np.abs(kap - kappa0) < 0.5 * kappa0
    sigma = float(np.sqrt((pw[near] * (kap[near] - kappa0) ** 2).sum() / max(pw[near].sum(), 1e-300)))
    order, cp = _weighted_sort(geom, v)
    vs, us, w = v[order], u[order], geom.M[order]
    qv = np.interp(PROBS, cp, vs)
    cw, cu = np.concatenate([[0], np.cumsum(w)]), np.concatenate([[0], np.cumsum(w * us)])
    cpe = np.concatenate([[0], np.cumsum(w)]) / w.sum()
    lo = np.interp(np.clip(PROBS - WINDOW, 0, 1), cpe, cw)
    hi = np.interp(np.clip(PROBS + WINDOW, 0, 1), cpe, cw)
    lo_u = np.interp(np.clip(PROBS - WINDOW, 0, 1), cpe, cu)
    hi_u = np.interp(np.clip(PROBS + WINDOW, 0, 1), cpe, cu)
    gu = (hi_u - lo_u) / np.maximum(hi - lo, 1e-300)
    return {"kappa0": kappa0, "gamma": sigma / kappa0, "qv": qv, "gu": gu,
            "captured_power": float(pw[near].sum() / max(pw.sum(), 1e-300))}


def to_vector(s: dict) -> np.ndarray:
    return np.concatenate([[np.log(s["kappa0"]), s["gamma"]], s["qv"], s["gu"]])


def from_vector(x: np.ndarray) -> dict:
    n = len(PROBS)
    qv = np.maximum.accumulate(x[2:2 + n])                       # quantiles must not decrease
    return {"kappa0": float(np.exp(x[0])), "gamma": float(max(x[1], 0.02)), "qv": qv, "gu": x[2 + n:2 + 2 * n]}


def ring_field(geom: Geometry, kappa0: float, gamma: float, seed: int, n_modes: int = 600):
    """Gaussian random field whose LB power sits on a ring at kappa0 with relative width gamma."""
    lam, phi = geom.eigenbasis(n_modes)
    kap = np.sqrt(np.maximum(lam, 0.0))
    var = np.exp(-0.5 * ((kap - kappa0) / (gamma * kappa0)) ** 2) / np.maximum(kap, 1e-9)
    var[0] = 0.0
    return phi @ (np.random.default_rng(seed).standard_normal(len(kap)) * np.sqrt(var))


def synthesize(geom: Geometry, s: dict, seed: int, mode: str = "spots", n_modes: int = 600):
    """Random state with the statistics s.

    mode "spots" (also holes): super-level sets of a ring field at kappa0 -> isolated blobs.
    mode "stripes": neighbourhoods of the nodal lines of a ring field at kappa0 / 2. Nodal lines of a
    random wave are about pi / kappa apart and never end, so this gives a connected maze with period
    2 pi / kappa0 even when the high-v phase covers less than half the area (where super-level sets
    of a Gaussian field would break into blobs).
    The field is then mapped onto the quantiles q_v by its area-weighted rank and u = g_u(v)."""
    if mode == "stripes":
        z = -np.abs(ring_field(geom, 0.5 * s["kappa0"], s["gamma"], seed, n_modes))
    else:
        z = ring_field(geom, s["kappa0"], s["gamma"], seed, n_modes)
    order, cp = _weighted_sort(geom, z)
    prob = np.empty(geom.n)
    prob[order] = cp
    v = np.interp(prob, PROBS, s["qv"])
    u = np.interp(v, s["qv"], s["gu"])
    return np.clip(u, 0.0, 1.0), np.maximum(v, 0.0)


def bands(geom: Geometry, s: dict, phase: float = 0.0, axis=(0.0, 0.0, 1.0)):
    """Stripes as latitude bands about `axis` with the statistics s (period 2 pi / kappa0, amplitudes
    mapped onto q_v, u = g_u(v)), the defect-free stripe arrangement a sphere allows.

    Used for stripes on the sphere: random mazes there keep rearranging (non-rigid motion with a
    residual ~1e-4 for > 10^5 time units), while the bands settle to a steady state."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    polar = np.arccos(np.clip(geom.V @ a / np.linalg.norm(geom.V, axis=1), -1.0, 1.0))
    z = -np.abs(np.cos(0.5 * s["kappa0"] * polar + phase))
    order, cp = _weighted_sort(geom, z)
    prob = np.empty(geom.n)
    prob[order] = cp
    v = np.interp(prob, PROBS, s["qv"])
    u = np.interp(v, s["qv"], s["gu"])
    return np.clip(u, 0.0, 1.0), np.maximum(v, 0.0)


# ---------------------------------------------------------------------------------------------
# spot patterns: learned radial profile on evenly spaced centres
# ---------------------------------------------------------------------------------------------
R_GRID = np.linspace(0.0, 0.6, 31)             # radial grid of the spot profile (surface units)


def spot_profile(geom: Geometry, u, v, threshold: float | None = None) -> dict:
    """Spot count and mean radial profiles u(r), v(r) around the spot centres (maxima of v in each
    connected high-v component); r = Euclidean distance to the nearest centre."""
    from scipy.spatial import cKDTree
    from .diagnostics import pattern_stats  # noqa: F401  (same thresholding convention)
    import scipy.sparse.csgraph as csg
    thr = threshold if threshold is not None else 0.5 * (v.max() + np.median(v))
    hi = v > thr
    E = geom.edges
    keep = hi[E[:, 0]] & hi[E[:, 1]]
    import scipy.sparse as sp
    A = sp.coo_matrix((np.ones(keep.sum()), (E[keep, 0], E[keep, 1])), shape=(geom.n, geom.n))
    ncomp, lab = csg.connected_components(A, directed=False)
    centres = []
    for c in np.unique(lab[hi]):
        idx = np.where((lab == c) & hi)[0]
        if len(idx) >= 3:
            centres.append(idx[np.argmax(v[idx])])
    centres = np.array(centres)
    d, _ = cKDTree(geom.V[centres]).query(geom.V)
    bins = np.clip(np.searchsorted(R_GRID, d) - 1, 0, len(R_GRID) - 1)
    w = geom.M
    pu = np.array([np.average(u[bins == b], weights=w[bins == b]) if (bins == b).any() else np.nan for b in range(len(R_GRID))])
    pv = np.array([np.average(v[bins == b], weights=w[bins == b]) if (bins == b).any() else np.nan for b in range(len(R_GRID))])
    for arr in (pu, pv):                               # fill empty outer bins with the last value
        good = np.where(np.isfinite(arr))[0]
        arr[good[-1] + 1:] = arr[good[-1]]
    return {"n_spots": int(len(centres)), "density": len(centres) / geom.area, "pu": pu, "pv": pv}


def cvt_centres(geom: Geometry, n: int, seed: int, iters: int = 40) -> np.ndarray:
    """n evenly spaced vertices: area-weighted Lloyd iterations (centroidal Voronoi), snapped to the mesh."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    tree_all = cKDTree(geom.V)
    c = geom.V[rng.choice(geom.n, size=n, replace=False, p=geom.M / geom.M.sum())]
    for _ in range(iters):
        _, lab = cKDTree(c).query(geom.V)
        w = np.bincount(lab, weights=geom.M, minlength=n)
        cen = np.stack([np.bincount(lab, weights=geom.M * geom.V[:, j], minlength=n) for j in range(3)], 1)
        c = cen / np.maximum(w, 1e-300)[:, None]
        c = geom.V[tree_all.query(c)[1]]                # project back onto the surface
    return tree_all.query(c)[1]


def place_spots(geom: Geometry, centres: np.ndarray, prof: dict):
    from scipy.spatial import cKDTree
    d, _ = cKDTree(geom.V[centres]).query(geom.V)
    return np.clip(np.interp(d, R_GRID, prof["pu"]), 0, 1), np.maximum(np.interp(d, R_GRID, prof["pv"]), 0)
