"""Initial conditions: the organisers' suggestion plus reproducible, mesh-independent perturbations."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .geometry import Geometry


def suggested_ic(geom: Geometry, normalize_z: bool = False):
    """u0 = 1 - G, v0 = G / 2 with G = exp(-6 (z - 0.05 cos 6 theta)^2), (theta, z) cylindrical.

    The band width was set for the unit sphere (z in [-1, 1]). On the frog z only spans +-0.48, so
    G >= 0.22 everywhere and the whole body starts excited. normalize_z=True rescales z by the
    surface's half-height so the band covers the same fraction of the body as on the sphere.
    """
    z = geom.z / np.abs(geom.z).max() if normalize_z else geom.z
    G = np.exp(-6.0 * (z - 0.05 * np.cos(6.0 * geom.theta)) ** 2)
    return 1.0 - G, 0.5 * G


def smooth_noise(geom: Geometry, seed: int, length: float = 0.05) -> np.ndarray:
    """White noise smoothed by one implicit heat step of duration length^2 / 4.

    Smoothing over a physical length (not a number of vertices) makes the perturbation
    statistically the same on different meshes of one surface. Output: zero mean, unit RMS.
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(geom.n) / np.sqrt(geom.M / geom.M.mean())  # white noise per unit area
    A = (sp.diags(geom.M) - (length ** 2 / 4.0) * geom.L).tocsc()
    xi = spla.spsolve(A, geom.M * w)
    xi -= geom.mean(xi)
    return xi / geom.rms(xi)


def perturbed_ic(geom: Geometry, eps: float = 0.05, seed: int = 0, length: float = 0.05,
                 mode: str = "additive", normalize_z: bool = False):
    """Suggested IC + smooth noise, clipped to the physical range [0, 1].

    mode="additive": u0 + eps*xi_u, v0 + eps*xi_v.
    mode="seeded":   additionally plants the classical (u, v) = (1/2, 1/4) state (Pearson 1993)
                     where the smooth noise exceeds one standard deviation, i.e. localised seeds of
                     size ~`length`. Needed in the spot regime, where F < 4 (F + k)^2 means there is no
                     non-trivial uniform equilibrium and a smooth initial state decays to (1, 0).
    mode="pearson":  the same seeds on the trivial background (u, v) = (1, 0), ignoring the band.
    """
    if mode == "pearson":
        u0, v0 = np.ones(geom.n), np.zeros(geom.n)
    else:
        u0, v0 = suggested_ic(geom, normalize_z=normalize_z)
    xi_u, xi_v = smooth_noise(geom, seed, length), smooth_noise(geom, seed + 1, length)
    u = u0 + eps * xi_u
    v = v0 + eps * xi_v
    if mode in ("seeded", "pearson"):
        seed_mask = smooth_noise(geom, seed + 2, length) > 1.0
        u = np.where(seed_mask, 0.5 + eps * xi_u, u)
        v = np.where(seed_mask, 0.25 + eps * xi_v, v)
    elif mode != "additive":
        raise ValueError(mode)
    return np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0)
