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


# ---------------------------------------------------------------------------------------------
# rotation-aware Newton for (nearly) SO(3)-symmetric surfaces: relative equilibria on the sphere
# ---------------------------------------------------------------------------------------------
def rotation_generators(geom: Geometry):
    """Sparse T_1, T_2, T_3 (n x n) with T_i u = (e_i x x) . grad_Gamma u at the vertices.

    grad u is the P1 gradient of each face, averaged to the vertices with face-area weights. On the
    continuous sphere T_i U spans the tangent of the orbit of a pattern under rigid rotations, i.e. the
    3 null vectors of the linearization; on the icosphere they are near-null (the mesh breaks the
    symmetry only weakly)."""
    V, F, n = geom.V, geom.F, geom.n
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    A2 = np.linalg.norm(fn, axis=1)                                   # twice the face area
    nf = fn / A2[:, None]
    rows, cols, vals = [], [], []
    for a, b, c in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        g = np.cross(nf, V[F[:, c]] - V[F[:, b]]) / A2[:, None]       # grad phi_a on each face
        for j in range(3):                                            # to every vertex of the face
            rows.append(F[:, j])
            cols.append(F[:, a])
            vals.append(0.5 * A2[:, None] * g)                        # area-weighted
    rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    w = np.bincount(F.ravel(), np.repeat(0.5 * A2, 3), minlength=n)   # sum of incident face areas
    G = [sp.csr_matrix((vals[:, k] / w[rows], (rows, cols)), shape=(n, n)) for k in range(3)]
    T = []
    for i in range(3):
        r = np.cross(np.eye(3)[i], V)                                 # e_i x x at the vertices
        T.append((sp.diags(r[:, 0]) @ G[0] + sp.diags(r[:, 1]) @ G[1] + sp.diags(r[:, 2]) @ G[2]).tocsr())
    return T


def _orbit_tangent(T, U):
    """(2n, r) matrix [T_i u; T_i v] of the rigid-rotation directions at U = [u; v]."""
    n = T[0].shape[0]
    return np.stack([np.concatenate([Ti @ U[:n], Ti @ U[n:]]) for Ti in T], axis=1)


def _active_generators(T, U, Mfull, rel: float = 1e-3):
    """Combinations of T_1..3 along which U actually moves (M-weighted SVD of the orbit tangent): 3 for a
    generic pattern, 2 for an axisymmetric one, none for a uniform state. Returns the r generators and
    the 3 x r matrix V that maps their coefficients to omega in the e_1, e_2, e_3 basis."""
    W = np.sqrt(Mfull)
    _, sv, Vt = np.linalg.svd(W[:, None] * _orbit_tangent(T, U), full_matrices=False)
    keep = sv > rel * max(sv[0], 1e-6 * np.linalg.norm(W * U))
    V = Vt[keep].T
    return [(V[0, j] * T[0] + V[1, j] * T[1] + V[2, j] * T[2]).tocsr() for j in range(V.shape[1])], V


def _rel_norm(geom: Geometry, p: GSParams, Mfull, x):
    """The relative residual measure of `rel_residual`, for a weak-form residual vector x."""
    s = x / Mfull
    return float(np.sqrt(Mfull @ (s * s) / (2 * geom.area))) / p.F


def rotation_split(geom: Geometry, p: GSParams, u, v, T=None) -> dict:
    """Split the steady residual into a rigid rotation and the rest.

    The strong residual s = R / M is fitted, M-weighted least squares, by -sum_i c_i T_i U; c (angular
    velocity components, in the units of T) is the drift that a rigid rotation explains. Returns the full
    relative residual, the residual after removing that part, and c."""
    T = rotation_generators(geom) if T is None else T
    U = np.concatenate([u, v])
    Mfull = np.concatenate([geom.M, geom.M])
    R = weak_residual(geom, p, u, v)
    Ta, V = _active_generators(T, U, Mfull)
    c, rest = np.zeros(3), R
    if Ta:
        B = Mfull[:, None] * _orbit_tangent(Ta, U)
        W = 1.0 / np.sqrt(Mfull)[:, None]
        cr = np.linalg.lstsq(W * B, -(W[:, 0] * R), rcond=None)[0]
        c, rest = V @ cr, R + B @ cr
    full, corr = _rel_norm(geom, p, Mfull, R), _rel_norm(geom, p, Mfull, rest)
    return {"rel_residual": full, "rel_residual_rot": corr, "omega": c.tolist(),
            "rotation_share": float(1 - (corr / max(full, 1e-300)) ** 2)}


class _BorderedLU:
    """Solver for K = [[Jw, Bw], [P^T, 0]] (sparse block, r <= 3 dense border columns and rows).

    Block elimination with the sparse LU of Jw and the r x r Schur complement P^T Jw^{-1} Bw, plus
    iterative refinement on K. (A direct splu of K is 10-70x slower: the dense border rows ruin the
    fill-reducing ordering.) Jw itself is only nearly singular (the rotation modes), so two refinement
    steps restore full accuracy. Right-hand sides may be vectors or matrices."""

    def __init__(self, Jw, Bw, P, refine: int = 2):
        self.Jw, self.Bw, self.P, self.refine = Jw.tocsc(), Bw, P, refine
        self.lu = spla.splu(self.Jw)
        self.X = self.lu.solve(Bw)
        self.S = P.T @ self.X
        self.m = Jw.shape[0]

    def _once(self, r1, r2):
        y = self.lu.solve(r1)
        z2 = np.linalg.solve(self.S, self.P.T @ y - r2)
        return y - self.X @ z2, z2

    def solve(self, rhs):
        r1, r2 = rhs[:self.m], rhs[self.m:]
        z1, z2 = self._once(r1, r2)
        for _ in range(self.refine):
            d1, d2 = self._once(r1 - self.Jw @ z1 - self.Bw @ z2, r2 - self.P.T @ z1)
            z1, z2 = z1 + d1, z2 + d2
        return np.concatenate([z1, z2])


def _corotating_jacobian(geom: Geometry, p: GSParams, U, omega, TT, Mfull):
    """Jacobian of R(U) + sum_i omega_i M T_i U with respect to U (weak form)."""
    n = geom.n
    Jw = weak_jacobian(geom, p, U[:n], U[n:])
    for om, Ti in zip(omega, TT):
        Jw = Jw + om * (sp.diags(Mfull) @ Ti)
    return Jw.tocsc()


def rotation_newton(geom: Geometry, p: GSParams, u, v, tol: float = 1e-11, max_iter: int = 15,
                    min_step: float = 1e-4, T=None, anchor=None, shift=None, omega=None, reduce: bool = True,
                    verbose: bool = False):
    """Newton's method for a relative equilibrium of the rotation group (bordered system).

    Unknowns U = [u; v] and omega (3 numbers):

        R(U) + sum_i omega_i M T_i U = 0,     <M T_i U0, U - U0> = shift_i     (i = 1..3)

    The 3 phase conditions pin the position on the (near-)orbit of rotated patterns (anchor U0 = the
    start state unless given; `shift` = 0 unless given), so the bordered Jacobian [[J + sum omega_i M T_i,
    M T U], [(M T U0)^T, 0]] is well conditioned although J itself has 3 near-zero eigenvalues. At the
    solution the residual R(U) is exactly a rigid rotation, M dU/dt = -sum omega_i M T_i U: the pattern
    is steady up to a rigid rotation. Backtracking on the rotation-corrected relative residual
    |R + omega M T U|. Only the rotations that move U are used (`reduce`; an axisymmetric pattern has 2,
    a uniform state none, and then this is `newton_polish`); with reduce=False, T is used as given and
    shift / omega refer to it. Returns u, v, omega (e_1..e_3 basis), history (rel_residual = full,
    rel_residual_rot = corrected)."""
    n, t0 = geom.n, time.time()
    T = rotation_generators(geom) if T is None else T
    Mfull = np.concatenate([geom.M, geom.M])
    U = np.concatenate([np.asarray(u, np.float64), np.asarray(v, np.float64)])
    U0 = U.copy() if anchor is None else np.concatenate([anchor[0], anchor[1]])
    if reduce:
        T, V = _active_generators(T, U0, Mfull)
        if not T:
            u, v, hist = newton_polish(geom, p, U[:n], U[n:], tol=tol, max_iter=max_iter, min_step=min_step)
            for h in hist:
                h["rel_residual_rot"] = h["rel_residual"]
            return u, v, np.zeros(3), hist
    else:
        V = np.eye(len(T))
    r = len(T)
    P = Mfull[:, None] * _orbit_tangent(T, U0)                        # phase-condition rows
    shift = np.zeros(r) if shift is None else np.asarray(shift, float)
    if omega is None:
        B = Mfull[:, None] * _orbit_tangent(T, U)
        W = 1.0 / np.sqrt(Mfull)[:, None]
        omega = np.linalg.lstsq(W * B, -(W[:, 0] * weak_residual(geom, p, U[:n], U[n:])), rcond=None)[0]
    else:
        omega = V.T @ np.asarray(omega, float) if reduce else np.asarray(omega, float)
    TT = [sp.block_diag([Ti, Ti]).tocsr() for Ti in T]
    norm = lambda x: _rel_norm(geom, p, Mfull, x)                     # noqa: E731

    def G(U, om):
        B = Mfull[:, None] * _orbit_tangent(T, U)
        R = weak_residual(geom, p, U[:n], U[n:])
        return R + B @ om, P.T @ (U - U0) - shift, B, R

    def record(it, a, G1, G2, R, om):
        hist.append({"iter": it, "step": a, "rel_residual_rot": norm(G1), "phase": float(np.abs(G2).max()),
                     "rel_residual": norm(R), "omega": (V @ om).tolist(), "wall_s": time.time() - t0})

    G1, G2, B, R = G(U, omega)
    res, hist = norm(G1), []
    record(0, 0.0, G1, G2, R, omega)
    phase_tol = 1e-12 * np.abs(P).sum(0).max()
    for it in range(1, max_iter + 1):
        if res < tol and np.abs(G2).max() < phase_tol:
            break
        lu = _BorderedLU(_corotating_jacobian(geom, p, U, omega, TT, Mfull), B, P)
        z = lu.solve(-np.concatenate([G1, G2]))
        dU, dom = z[:2 * n], z[2 * n:]
        a = 1.0
        while a >= min_step:
            Ut, omt = U + a * dU, omega + a * dom
            G1t, G2t, Bt, Rt = G(Ut, omt)
            new = norm(G1t)
            if np.isfinite(new) and new < res:
                break
            a /= 2
        if a < min_step:
            break
        U, omega, G1, G2, B, R, res = Ut, omt, G1t, G2t, Bt, Rt, new
        record(it, a, G1, G2, R, omega)
        if verbose:
            h = hist[-1]
            print(f"  rot-Newton {it:2d}: step {a:.3g} corrected {h['rel_residual_rot']:.2e} "
                  f"full {h['rel_residual']:.2e} |omega| {np.linalg.norm(omega):.2e}", flush=True)
    return U[:n], U[n:], V @ omega, hist


def orient_polish(geom: Geometry, p: GSParams, u, v, omega, tol: float = 1e-11, rot_tol: float = 1e-12,
                  theta_max: float = 0.05, max_steps: int = 40, T=None, verbose: bool = False):
    """Follow the family of relative equilibria to an orientation where the drift omega vanishes, i.e.
    a true steady state of the discrete system (an orientation the mesh singles out).

    The phase shift s (s_i = <M T_i U0, U - U0>, anchored at the current state) parametrizes the family
    near a relative equilibrium (U, omega); the bordered Jacobian gives dU/ds and Y = domega/ds. On the
    family the pattern drifts as ds/dt = -Gram omega (Gram_ij = <M T_i U, T_j U>). Each step is an
    implicit-Euler step of this 3-dimensional slow dynamics, (I + dt Gram Y) ds = -dt Gram omega, capped
    at a rigid rotation of theta_max rad, with dt grown by switched evolution relaxation as |omega|
    falls (pseudo-transient continuation on the orientation; dt -> infinity is Newton on omega(s) = 0).
    Predictor U + dU/ds ds, corrector `rotation_newton` with the shifted phase conditions, so every
    state on the way is a relative equilibrium. Stops when the full relative residual (= the drift
    alone) is below tol. Returns the final steady state, or else the state with the smallest drift seen:
    u, v, omega, history; the last entry carries the eigenvalues of the reduced Jacobian -Gram Y (the
    rigid-rotation block of the linearization)."""
    n, t0 = geom.n, time.time()
    T = rotation_generators(geom) if T is None else T
    Mfull = np.concatenate([geom.M, geom.M])
    U = np.concatenate([u, v]).astype(np.float64)
    full = lambda U: rel_residual(geom, p, U[:n], U[n:])           # noqa: E731
    hist = [{"step": 0, "dt": 0.0, "rel_residual": full(U), "omega": np.asarray(omega, float).tolist(),
             "theta": 0.0, "wall_s": 0.0}]
    T, V = _active_generators(T, U, Mfull)
    if not T:
        return U[:n], U[n:], np.zeros(3), hist
    TT = [sp.block_diag([Ti, Ti]).tocsr() for Ti in T]
    r, omega = len(T), V.T @ np.asarray(omega, float)
    best = (hist[0]["rel_residual"], U, omega)
    dt = theta_max / max(np.linalg.norm(omega), 1e-300)              # first step ~ theta_max of drift
    for k in range(1, max_steps + 1):
        TU = _orbit_tangent(T, U)
        B = Mfull[:, None] * TU
        lu = _BorderedLU(_corotating_jacobian(geom, p, U, omega, TT, Mfull), B, B)
        Z = lu.solve(np.vstack([np.zeros((2 * n, r)), np.eye(r)]))
        X, Y = Z[:2 * n], Z[2 * n:]                                   # dU/ds, domega/ds
        Gram = TU.T @ B                                               # ds = -Gram theta for a rotation theta
        if hist[-1]["rel_residual"] < tol:
            hist[-1]["reduced_eigenvalues"] = np.linalg.eigvals(-Gram @ Y).real.tolist()
            break
        Un = None
        for _ in range(6):
            ds = np.linalg.solve(np.eye(r) + dt * Gram @ Y, -dt * Gram @ omega)
            th = float(np.linalg.norm(np.linalg.solve(Gram, ds)))
            if th > theta_max:
                ds, th = ds * theta_max / th, theta_max
            un, vn, omn, h = rotation_newton(geom, p, U[:n] + X[:n] @ ds, U[n:] + X[n:] @ ds, tol=rot_tol,
                                             T=T, anchor=(U[:n], U[n:]), shift=ds, omega=omega + Y @ ds,
                                             reduce=False)
            if h[-1]["rel_residual_rot"] < 10 * rot_tol:
                Un = np.concatenate([un, vn])
                break
            dt /= 4
        if Un is None:
            break
        dt *= min(10.0, np.linalg.norm(omega) / max(np.linalg.norm(omn), 1e-300))   # SER
        U, omega = Un, omn
        hist.append({"step": k, "dt": dt, "rel_residual": full(U), "omega": (V @ omega).tolist(),
                     "theta": th, "wall_s": time.time() - t0})
        if hist[-1]["rel_residual"] < best[0]:
            best = (hist[-1]["rel_residual"], U, omega)
        if verbose:
            print(f"  orient {k:2d}: dt {dt:.2e}, rotated {th:.3e} rad, |omega| {np.linalg.norm(omega):.3e}, "
                  f"full residual {hist[-1]['rel_residual']:.3e}", flush=True)
    if hist[-1]["rel_residual"] >= tol:
        _, U, omega = best
    return U[:n], U[n:], V @ omega, hist


def relax_polish(geom: Geometry, p: GSParams, u0, v0, schedule=((1.0, 2000.0), (4.0, 400000.0)),
                 check_every: float = 500.0, try_below: float = 1e-4, tol: float = 1e-11,
                 newton_iter: int = 15, rotations=None, orient: bool = False, verbose: bool = False):
    """Time-integrate (SBDF2, LU) in chunks; whenever the relative steady residual has dropped below
    `try_below` (then 3x further after each failed try), attempt `newton_polish`. Stop at the first
    Newton run that reaches `tol`; a failed try is discarded and the integration continues.
    Used identically for AI initial guesses and for ordinary initial conditions, so their wall times
    are directly comparable. Returns u, v, info.

    With `rotations` = rotation_generators(geom) (the sphere, whose rotation symmetry the mesh breaks
    only weakly), the trigger and the test use the rotation-corrected residual and each try is
    `rotation_newton`: converged means a relative equilibrium, steady up to a rigid rotation
    (info: rel_residual = full residual, rel_residual_rot = corrected, omega = the drift). With
    `orient` as well, a converged run is then taken along the family by `orient_polish` and finished by
    `newton_polish`; that result replaces it only if its full residual reaches `tol` (info["steady"])."""
    import torch
    from .solver import GrayScottSolver

    t_wall, t = time.time(), 0.0
    u, v = np.asarray(u0, float), np.asarray(v0, float)
    info = {"newton": [], "relax_wall_s": 0.0, "relax_steps": 0}
    if rotations is None:
        measure, key = (lambda u, v: rel_residual(geom, p, u, v)), "rel_residual"
    else:
        measure, key = (lambda u, v: rotation_split(geom, p, u, v, rotations)["rel_residual_rot"]), "rel_residual_rot"
    res, next_try, done, omega = measure(u, v), try_below, False, None
    for dt, t_end in schedule:
        solver = GrayScottSolver(geom, dt=dt)          # its LU factors are reused across chunks
        while not done and t < t_end:
            if res < next_try:
                if rotations is None:
                    un, vn, hist = newton_polish(geom, p, u, v, tol=tol, max_iter=newton_iter)
                else:
                    un, vn, om, hist = rotation_newton(geom, p, u, v, tol=tol, max_iter=newton_iter, T=rotations)
                info["newton"].append({"t": t, "from": res, "to": hist[-1][key],
                                       "iters": len(hist) - 1, "wall_s": hist[-1]["wall_s"]})
                if verbose:
                    print(f"  t={t:.0f}: Newton {res:.2e} -> {hist[-1][key]:.2e}", flush=True)
                if hist[-1][key] < tol:
                    u, v, res, done = un, vn, hist[-1][key], True
                    if rotations is not None:
                        omega = om
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
            res = measure(u, v)
        if done:
            break
    info.update(relax_time=t, rel_residual=res, converged=done, wall_s=time.time() - t_wall)
    if rotations is not None:
        rs = rotation_split(geom, p, u, v, rotations)
        info.update(rel_residual=rs["rel_residual"], rel_residual_rot=res,
                    omega=(rs["omega"] if omega is None else np.asarray(omega).tolist()), steady=False)
        if done and orient:
            uo, vo, omo, ho = orient_polish(geom, p, u, v, omega, tol=tol, T=rotations, verbose=verbose)
            un, vn, hn = newton_polish(geom, p, uo, vo, tol=tol)
            info["orient"] = {"steps": len(ho) - 1, "rotated_rad": float(sum(h["theta"] for h in ho)),
                              "from": ho[0]["rel_residual"], "to": min(h["rel_residual"] for h in ho),
                              "newton_to": hn[-1]["rel_residual"], "wall_s": ho[-1]["wall_s"] + hn[-1]["wall_s"]}
            if hn[-1]["rel_residual"] < tol:                          # a true steady state
                uo, vo = un, vn
            rs = rotation_split(geom, p, uo, vo, rotations)           # else the relative equilibrium
            u, v = uo, vo                                             # with the smallest drift
            info.update(rel_residual=rs["rel_residual"], rel_residual_rot=rs["rel_residual_rot"],
                        omega=rs["omega"], steady=bool(rs["rel_residual"] < tol))
        info["wall_s"] = time.time() - t_wall
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
