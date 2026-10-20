"""Steady-state polishing: Newton with line search, relax-then-polish, and pseudo-transient continuation.

The semi-discrete system is M dU/dt = R(U) with U = [u; v] and the weak-form residual

    R_u = Du L u + M (-u v^2 + F (1 - u)),   R_v = Dv L v + M (u v^2 - (F + k) v).

PTC takes implicit-Euler steps (M / dtau - J) dU = R(U) with a pseudo-time step that grows as the
residual falls (switched evolution relaxation, dtau_{k+1} = dtau_k |R_k| / |R_{k+1}|), so it follows
the dynamics while far from equilibrium and becomes Newton's method near it. This is the "solver
polish" stage: any candidate pattern (a long time integration, or later an AI prediction) is driven
to a steady state of the discrete PDE whose residual is at round-off level.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .geometry import Geometry
from .solver import GSParams


def weak_residual(geom: Geometry, p: GSParams, u, v):
    r = u * v * v
    Ru = p.Du * (geom.L @ u) + geom.M * (-r + p.F * (1 - u))
    Rv = p.Dv * (geom.L @ v) + geom.M * (r - (p.F + p.k) * v)
    return np.concatenate([Ru, Rv])


def weak_jacobian(geom: Geometry, p: GSParams, u, v):
    M = geom.M
    return sp.bmat([[p.Du * geom.L + sp.diags(M * (-v * v - p.F)), sp.diags(M * (-2 * u * v))],
                    [sp.diags(M * (v * v)), p.Dv * geom.L + sp.diags(M * (2 * u * v - (p.F + p.k)))]]).tocsc()


def rel_residual(geom: Geometry, p: GSParams, u, v) -> float:
    """rms of the strong residual (R / M) relative to F, the convergence measure of ptc_polish."""
    Mfull = np.concatenate([geom.M, geom.M])
    s = weak_residual(geom, p, u, v) / Mfull
    return float(np.sqrt(Mfull @ (s * s) / (2 * geom.area))) / p.F


def newton_polish(geom: Geometry, p: GSParams, u, v, tol: float = 1e-11, max_iter: int = 15,
                  min_step: float = 1e-4):
    """Newton's method with backtracking on the relative residual. Converges quadratically once the
    state is in the basin (on the frog from a relative residual of about 1e-5); outside it the line
    search shrinks the steps and the iteration stalls, which `relax_polish` detects."""
    n, t0 = geom.n, time.time()
    u, v = np.array(u, dtype=np.float64), np.array(v, dtype=np.float64)
    res = rel_residual(geom, p, u, v)
    hist = [{"iter": 0, "step": 0.0, "rel_residual": res, "wall_s": 0.0}]
    for it in range(1, max_iter + 1):
        if res < tol:
            break
        dU = spla.splu((-weak_jacobian(geom, p, u, v)).tocsc()).solve(weak_residual(geom, p, u, v))
        a = 1.0
        while a >= min_step:
            ut, vt = u + a * dU[:n], v + a * dU[n:]
            new = rel_residual(geom, p, ut, vt)
            if np.isfinite(new) and new < res:
                break
            a /= 2
        if a < min_step:
            break
        u, v, res = ut, vt, new
        hist.append({"iter": it, "step": a, "rel_residual": res, "wall_s": time.time() - t0})
    return u, v, hist


def relax_polish(geom: Geometry, p: GSParams, u0, v0, schedule=((1.0, 2000.0), (4.0, 400000.0)),
                 check_every: float = 500.0, try_below: float = 1e-4, tol: float = 1e-11,
                 newton_iter: int = 15, verbose: bool = False):
    """Time-integrate (SBDF2, LU) in chunks; whenever the relative steady residual has dropped below
    `try_below` (then 3x further after each failed try), attempt `newton_polish`. Stop at the first
    Newton run that reaches `tol`; a failed try is discarded and the integration continues.
    Used identically for AI initial guesses and for ordinary initial conditions, so their wall times
    are directly comparable. Returns u, v, info."""
    import torch
    from .solver import GrayScottSolver

    t_wall, t = time.time(), 0.0
    u, v = np.asarray(u0, float), np.asarray(v0, float)
    info = {"newton": [], "relax_wall_s": 0.0, "relax_steps": 0}
    res, next_try, done = rel_residual(geom, p, u, v), try_below, False
    for dt, t_end in schedule:
        solver = GrayScottSolver(geom, dt=dt)          # its LU factors are reused across chunks
        while not done and t < t_end:
            if res < next_try:
                un, vn, hist = newton_polish(geom, p, u, v, tol=tol, max_iter=newton_iter)
                info["newton"].append({"t": t, "from": res, "to": hist[-1]["rel_residual"],
                                       "iters": len(hist) - 1, "wall_s": hist[-1]["wall_s"]})
                if verbose:
                    print(f"  t={t:.0f}: Newton {res:.2e} -> {hist[-1]['rel_residual']:.2e}", flush=True)
                if hist[-1]["rel_residual"] < tol:
                    u, v, res, done = un, vn, hist[-1]["rel_residual"], True
                    break
                next_try = res / 3.0
            n = max(1, int(round(min(check_every, t_end - t) / dt)))
            tr = time.time()
            with torch.no_grad():
                uu, vv = solver.rollout(u, v, p, n)
            u, v = uu.numpy(), vv.numpy()
            info["relax_wall_s"] += time.time() - tr
            info["relax_steps"] += n
            t += n * dt
            res = rel_residual(geom, p, u, v)
        if done:
            break
    info.update(relax_time=t, rel_residual=res, converged=done, wall_s=time.time() - t_wall)
    return u, v, info


def ptc_polish(geom: Geometry, p: GSParams, u, v, tol: float = 1e-12, dtau0: float = 50.0,
               dtau_max: float = 1e12, max_iter: int = 60, verbose: bool = False):
    """Drive (u, v) to a steady state. Returns u, v and a per-iteration history.

    Convergence measure: rms of the strong residual (R / M), relative to the feed rate F.
    """
    u, v = np.array(u, dtype=np.float64), np.array(v, dtype=np.float64)
    n, Mfull = geom.n, np.concatenate([geom.M, geom.M])

    def rel(R):
        s = R / Mfull
        return float(np.sqrt(Mfull @ (s * s) / (2 * geom.area))) / p.F

    R = weak_residual(geom, p, u, v)
    res, dtau, hist, t0 = rel(R), dtau0, [], time.time()
    hist.append({"iter": 0, "dtau": 0.0, "rel_residual": res, "wall_s": 0.0, "accepted": True})
    for it in range(1, max_iter + 1):
        if res < tol:
            break
        A = (sp.diags(Mfull / dtau) - weak_jacobian(geom, p, u, v)).tocsc()
        dU = spla.splu(A).solve(R)
        u_try, v_try = u + dU[:n], v + dU[n:]
        R_try = weak_residual(geom, p, u_try, v_try)
        new = rel(R_try)
        accepted = new < 2.0 * res and np.isfinite(new)
        if accepted:                       # SER: grow the pseudo-step as the residual falls
            u, v, R = u_try, v_try, R_try
            dtau = min(dtau * res / max(new, 1e-300), dtau_max)
            res = new
        else:                              # overshoot: back off towards time stepping
            dtau = max(dtau / 10.0, 1e-2)
        hist.append({"iter": it, "dtau": dtau, "rel_residual": res, "wall_s": time.time() - t0,
                     "accepted": bool(accepted)})
        if verbose:
            print(f"  PTC {it:2d}: dtau={dtau:.2e} rel residual={res:.3e}{'' if accepted else ' (rejected)'}",
                  flush=True)
    return u, v, hist
