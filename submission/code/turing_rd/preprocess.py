"""Denoising of the target Solution.mat and the closed-form (integral-identity) estimate of F and k.

Noise model (README, "Solution.mat 来源排查" and "按「原始浓度 + 稳态解」方向的系统研究"):
    * a small-scale component shared by every discretisation, strongly anti-correlated between u
      and v, that keeps 2.6 % of the variance of v beyond 600 Laplace-Beltrami modes (a converged
      steady state keeps 0.4 %);
    * smooth "negative plateaus" where v < 0 (11.7 % of the area, minimum -0.17) and a few vertices
      with u > 1, both impossible for a steady state (maximum principle).

Preprocessing = low-pass, then projection on the physical range:
    1. heat-kernel low-pass: n implicit heat steps (M - (tau/n) L) f_{j+1} = M f_j. Transfer function
       (1 + tau lambda / n)^-n ~ exp(-tau kappa^2) for the LB mode of wavenumber kappa = sqrt(lambda);
       tau is set so that the transfer is 1/2 at the cut-off kappa_half. With the robust Laplacian
       (non-negative edge weights) M - (tau/n) L is an M-matrix, so the filter preserves the mean and
       creates no new extrema: it cannot push admissible values out of [0, 1].
    2. clip u to [0, 1] and v to [0, inf).

Choice of kappa_half (`select_cutoff`): for each candidate cut-off the filtered, clipped target gets
closed-form F, k from the integral identities, and the relative strong-form steady residual
rms(Du Delta u + f) + rms(Dv Delta v + g), divided by F, is evaluated with the given Du, Dv. Weak
filtering leaves the noise, which the Laplacian amplifies by kappa^2; strong filtering erodes the
spots. The residual therefore falls steeply, then flattens; the chosen cut-off is the weakest
filtering whose residual is within `rel_tol` (2 %) of the minimum. No ground-truth F, k enter.

Integral identities (exact for any steady state on a closed surface, any discretisation of the
Laplacian with zero column sums): integrating both equations kills the diffusion terms,
    F     = int u v^2 / int (1 - u),
    F + k = int u v^2 / int v.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .geometry import Geometry

DEFAULT_KAPPAS = tuple(range(10, 41)) + (45, 50, 60, 80)


class HeatLowPass:
    """f -> heat-kernel low-pass of f with transfer 1/2 at wavenumber kappa_half (inf: identity)."""

    def __init__(self, geom: Geometry, kappa_half: float, n_steps: int = 8):
        self.g, self.kappa_half, self.n = geom, float(kappa_half), n_steps
        if np.isinf(self.kappa_half):
            self.tau, self._lu = 0.0, None
            return
        self.tau = n_steps * (2 ** (1 / n_steps) - 1) / self.kappa_half ** 2
        self._lu = spla.splu((sp.diags(geom.M) - (self.tau / n_steps) * geom.L).tocsc())

    @property
    def sigma(self) -> float:
        """Width of the equivalent Gaussian smoothing kernel (geodesic length)."""
        return float(np.sqrt(2 * self.tau))

    def transfer(self, kappa):
        return (1 + self.tau * np.asarray(kappa) ** 2 / self.n) ** (-self.n)

    def __call__(self, f):
        f = np.asarray(f, dtype=float)
        if self._lu is None:
            return f.copy()
        Mf = self.g.M if f.ndim == 1 else self.g.M[:, None]
        for _ in range(self.n):
            f = self._lu.solve(Mf * f)
        return f


def admissible(u, v):
    """Project on the range every steady state satisfies: 0 <= u <= 1, v >= 0."""
    return np.clip(u, 0.0, 1.0), np.maximum(v, 0.0)


def integral_identities(geom: Geometry, u, v) -> tuple[float, float]:
    """Closed-form (F, k) of a steady state from the two integral identities."""
    r = geom.M @ (u * v * v)
    F = r / (geom.M @ (1 - u))
    return float(F), float(r / (geom.M @ v) - F)


def steady_residual(geom: Geometry, u, v, F, k, Du, Dv) -> dict:
    """Relative strong-form steady residual (rms over the surface, divided by F)."""
    r = u * v * v
    Ru = Du * geom.lap(u) - r + F * (1 - u)
    Rv = Dv * geom.lap(v) + r - (F + k) * v
    return {"rel": (geom.rms(Ru) + geom.rms(Rv)) / F, "rel_u": geom.rms(Ru) / F, "rel_v": geom.rms(Rv) / F}


def select_cutoff(geom: Geometry, u, v, Du, Dv, kappas=DEFAULT_KAPPAS, rel_tol: float = 0.02):
    """Residual-versus-cut-off curve and the chosen kappa_half (see module docstring)."""
    rows = []
    for kh in list(kappas) + [np.inf]:
        lp = HeatLowPass(geom, kh)
        us, vs = lp(u), lp(v)
        uc, vc = admissible(us, vs)
        F, k = integral_identities(geom, uc, vc)
        F_raw, k_raw = integral_identities(geom, us, vs)
        res = steady_residual(geom, uc, vc, F, k, Du, Dv)
        rows.append({"kappa_half": float(kh), "sigma": lp.sigma, "F": F, "k": k, "F_unclipped": F_raw,
                     "k_unclipped": k_raw, "residual": res["rel"], "residual_u": res["rel_u"],
                     "residual_v": res["rel_v"], "neg_area": float(geom.M[vs < 0].sum() / geom.area),
                     "v_max": float(vs.max()), "change_v": geom.rms(vs - v) / geom.rms(v - geom.mean(v))})
    best = min(r["residual"] for r in rows)
    ok = [r for r in rows if r["residual"] <= (1 + rel_tol) * best]
    chosen = max(ok, key=lambda r: r["kappa_half"])
    return chosen["kappa_half"], rows


def preprocess(geom: Geometry, u, v, Du, Dv, kappa_half: float | None = None, **kw) -> dict:
    """Denoised, admissible target plus the closed-form estimate; kappa_half chosen if not given."""
    rows = None
    if kappa_half is None:
        kappa_half, rows = select_cutoff(geom, u, v, Du, Dv, **kw)
    lp = HeatLowPass(geom, kappa_half)
    uc, vc = admissible(lp(u), lp(v))
    F, k = integral_identities(geom, uc, vc)
    return {"u": uc, "v": vc, "kappa_half": float(kappa_half), "lowpass": lp, "F": F, "k": k, "curve": rows}
