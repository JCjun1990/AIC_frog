"""Differentiable IMEX solver for the surface Gray-Scott system.

    du/dt = Du Delta u - u v^2 + F (1 - u)
    dv/dt = Dv Delta v + u v^2 - (F + k) v

Diffusion is implicit, reaction explicit. With lumped mass M and weak Laplacian L, each step solves
two SPD systems (c M - alpha L) x = b with alpha = dt * D:

    SBDF1: c = 1,   b = M (u^n + dt f^n)
    SBDF2: c = 3/2, b = M (2 u^n - u^{n-1}/2 + dt (2 f^n - f^{n-1}))

Two interchangeable backends, both differentiable w.r.t. the right-hand side and alpha (hence
w.r.t. F, k, Du, Dv and the initial state):

    "lu" -- SciPy sparse LU factorised once per (c, alpha), float64, CPU. Exact; used for checks.
    "cg" -- Jacobi-preconditioned conjugate gradients in pure PyTorch. Runs on CUDA, supports a
            batch of different alphas (one per column), float32 for throughput.

Backward pass: A is symmetric, so dL/db = A^{-1} g and dL/dalpha = (A^{-1} g)^T L x.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, asdict
import math
from typing import Callable

import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from torch.utils.checkpoint import checkpoint

from .geometry import Geometry

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")


@dataclass(frozen=True)
class GSParams:
    F: float
    k: float
    Du: float = 2.78e-4
    Dv: float = 1.39e-4

    def as_dict(self) -> dict:
        return asdict(self)


SPOTS = GSParams(F=0.028, k=0.062)
STRIPES = GSParams(F=0.054, k=0.063)


# ---------------------------------------------------------------------------------------------
# linear algebra
# ---------------------------------------------------------------------------------------------
class CGConvergenceError(RuntimeError):
    """No usable solution/gradient: inspect ``info`` for per-RHS true residuals."""

    def __init__(self, info):
        self.info = info
        super().__init__(f"CG {info.get('phase', 'raw')} failed: {info['status']}; "
                         f"iterations={info['iterations']}/{info['maxiter']}, "
                         f"max true relative residual={info['max_relative_residual']}, rtol={info['rtol']}")

    def __reduce__(self):
        return type(self), (self.info,)


class SPDOperator:
    """Solve (c M - alpha L) x = b; CG failures raise CGConvergenceError.

    Successful calls retain the tensor API. ``last_cg_info`` records the latest
    raw/forward/adjoint solve, not a trajectory-wide convergence certificate.
    """

    def __init__(self, geom: Geometry, backend: str = "lu", device="cpu", dtype=torch.float64,
                 cg_rtol: float = 1e-10, cg_maxiter: int = 2000, lu_cache: int = 8):
        if backend not in ("lu", "cg"):
            raise ValueError(backend)
        self.backend, self.device, self.dtype = backend, torch.device(device), dtype
        self.L_sp = geom.L.tocsc()
        self.M_np = geom.M
        L = geom.L.tocsr()
        self.L = torch.sparse_csr_tensor(torch.as_tensor(L.indptr, dtype=torch.int64),
                                         torch.as_tensor(L.indices, dtype=torch.int64),
                                         torch.as_tensor(L.data), size=L.shape,
                                         dtype=dtype, device=self.device, check_invariants=True)
        self.M = torch.as_tensor(geom.M, dtype=dtype, device=self.device)
        self.Ldiag = torch.as_tensor(geom.L.diagonal(), dtype=dtype, device=self.device)
        self.cg_rtol, self.cg_maxiter = cg_rtol, cg_maxiter
        if backend == "cg" and (not math.isfinite(cg_rtol) or cg_rtol < 0 or
                                not isinstance(cg_maxiter, int) or cg_maxiter < 0):
            raise ValueError("CG requires finite rtol >= 0 and integer maxiter >= 0")
        self._lu: OrderedDict = OrderedDict()
        self._lu_cache = lu_cache
        self.cg_iterations = 0
        self.last_cg_info = None

    # L x for (n,) or (n, B)
    def Lmv(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return (self.L @ x.unsqueeze(1)).squeeze(1)
        return self.L @ x

    def _lu_factor(self, c: float, alpha: float):
        key = (float(c), float(alpha))
        if key in self._lu:
            self._lu.move_to_end(key)
            return self._lu[key]
        A = (c * sp.diags(self.M_np) - alpha * self.L_sp).tocsc()
        lu = spla.splu(A)
        self._lu[key] = lu
        if len(self._lu) > self._lu_cache:
            self._lu.popitem(last=False)
        return lu

    def _lu_solve(self, c: float, alpha: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if alpha.numel() != 1:
            raise ValueError("the LU backend needs one diffusion coefficient per solve; use backend='cg' "
                             "for a batch with different Du/Dv")
        lu = self._lu_factor(c, float(alpha))
        x = lu.solve(np.ascontiguousarray(b.detach().cpu().numpy(), dtype=np.float64))
        return torch.as_tensor(x, dtype=b.dtype, device=b.device)

    def _cg_solve(self, c: float, alpha: torch.Tensor, b: torch.Tensor, x0=None) -> torch.Tensor:
        """Jacobi-PCG, accepted only if ||b-Ax||_2 <= cg_rtol ||b||_2 per RHS.

        The recurrence residual is a stopping hint, never the acceptance test.
        No absolute floor or automatic LU fallback relaxes the requested tolerance.
        """
        self.cg_iterations, self.last_cg_info = 0, None
        if b.dim() not in (1, 2) or alpha.numel() not in (1, 1 if b.dim() == 1 else b.shape[1]):
            raise ValueError("CG expects b=(n,) or (n,B), alpha scalar or one value per RHS")
        squeeze = b.dim() == 1
        B = b.unsqueeze(1) if squeeze else b
        a = alpha.reshape(1, -1) if alpha.numel() > 1 else alpha.reshape(1, 1)
        Mc = self.M.unsqueeze(1)

        def A(x):
            return c * Mc * x - a * (self.L @ x)

        diagonal = c * Mc - a * self.Ldiag.unsqueeze(1)
        dinv = 1.0 / diagonal
        x = B * dinv if x0 is None else (x0.unsqueeze(1) if squeeze else x0).clone()
        bnorm = B.norm(dim=0)
        limit = self.cg_rtol * bnorm
        iterations, restarts = 0, 0

        def finish(reason, residual=None):
            true_r = B - A(x) if residual is None else residual
            absolute = true_r.norm(dim=0)
            finite_inputs = (math.isfinite(c) and bool(torch.isfinite(B).all()) and
                             bool(torch.isfinite(a).all()) and bool(torch.isfinite(x).all()))
            finite = (finite_inputs and bool(torch.isfinite(true_r).all()) and
                      bool(torch.isfinite(bnorm).all()) and bool(torch.isfinite(absolute).all()))
            residual_nonzero = (true_r != 0).any(dim=0)
            norm_underflow = bool((((bnorm == 0) & (B != 0).any(dim=0)) |
                                   ((absolute == 0) & residual_nonzero)).any())
            passed = torch.isfinite(absolute) & torch.isfinite(bnorm) & (absolute <= limit)
            passed = passed & ~((absolute == 0) & residual_nonzero)
            relative = torch.where(bnorm > 0, absolute / torch.where(bnorm > 0, bnorm, torch.ones_like(bnorm)),
                                   torch.where(~residual_nonzero, torch.zeros_like(absolute),
                                               torch.full_like(absolute, float("inf"))))
            fatal = reason in ("nonfinite_input_or_operator", "nonfinite_iteration", "nonpositive_preconditioner", "breakdown")
            converged = bool(finite and passed.all() and not fatal and not norm_underflow)

            def numbers(tensor):
                return [float(v) if math.isfinite(float(v)) else None for v in tensor.detach().cpu().tolist()]

            self.cg_iterations = iterations  # number of updates, including the last permitted update
            self.last_cg_info = {
                "backend": "cg", "phase": "raw", "status": "converged" if converged else reason,
                "converged": converged, "iterations": iterations, "maxiter": self.cg_maxiter,
                "rtol": self.cg_rtol, "dtype": str(b.dtype), "finite": finite,
                "rhs_norm": numbers(bnorm), "absolute_residual": numbers(absolute),
                "relative_residual": numbers(relative), "residual_limit": numbers(limit),
                "max_relative_residual": numbers(relative.max().reshape(1))[0],
                "converged_per_rhs": passed.detach().cpu().tolist(), "residual_restarts": restarts,
                "true_residual_verified": finite and not norm_underflow, "norm_underflow": norm_underflow, "fallback": None,
                "residual_definition": "Euclidean ||b-(c M-alpha L)x||_2 / ||b||_2 per RHS; zero RHS requires exact zero residual"}
            if not converged:
                if not finite:
                    self.last_cg_info["status"] = "nonfinite_input_or_operator" if iterations == 0 else "nonfinite_iteration"
                elif norm_underflow:
                    self.last_cg_info["status"] = "residual_norm_underflow"
                elif reason == "converged":
                    self.last_cg_info["status"] = "true_residual_failed"
                raise CGConvergenceError(self.last_cg_info)
            return x.squeeze(1) if squeeze else x

        if not math.isfinite(c) or not bool(torch.isfinite(B).all()) or not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(x).all()):
            return finish("nonfinite_input_or_operator")
        if not bool(torch.isfinite(diagonal).all()) or not bool((diagonal > 0).all()):
            return finish("nonpositive_preconditioner")
        r = B - A(x)
        if not bool(torch.isfinite(r).all()):
            return finish("nonfinite_input_or_operator", r)
        z = dinv * r
        p = z.clone()
        rz = (r * z).sum(0)
        if not bool(torch.isfinite(z).all()) or not bool(torch.isfinite(rz).all()):
            return finish("nonfinite_iteration")
        for _ in range(self.cg_maxiter):
            done = r.norm(dim=0) <= limit
            if bool(done.all()):
                true_r = B - A(x)
                if bool((true_r.norm(dim=0) <= limit).all()):
                    return finish("converged", true_r)
                # Recurrence drift: restart only the previously invalid stopping
                # path, using the true residual and the unchanged tolerance.
                r = true_r
                z = dinv * r
                p, rz = z.clone(), (r * z).sum(0)
                restarts += 1
                done = r.norm(dim=0) <= limit
            Ap = A(p)
            pAp = (p * Ap).sum(0)
            if not bool(torch.isfinite(Ap).all()) or not bool(torch.isfinite(rz).all()):
                return finish("nonfinite_iteration")
            if not bool(torch.isfinite(pAp).all()):
                return finish("nonfinite_iteration")
            if bool(((~done) & ((pAp <= 0) | (rz <= 0))).any()):
                return finish("breakdown")
            step = torch.where(done, torch.zeros_like(pAp), rz / torch.where(done, torch.ones_like(pAp), pAp))
            x = x + step * p
            r = r - step * Ap
            iterations += 1
            if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(r).all()):
                return finish("nonfinite_iteration")
            z = dinv * r
            rz_new = (r * z).sum(0)
            if not bool(torch.isfinite(z).all()) or not bool(torch.isfinite(rz_new).all()):
                return finish("nonfinite_iteration")
            beta = torch.where(done, torch.zeros_like(rz), rz_new / torch.where(done, torch.ones_like(rz), rz))
            p = z + beta * p
            if not bool(torch.isfinite(p).all()):
                return finish("nonfinite_iteration")
            rz = rz_new
        # A final update may converge exactly at maxiter; only its TRUE residual decides.
        return finish("max_iterations")

    def solve_raw(self, c: float, alpha: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.backend == "lu":
            return self._lu_solve(c, alpha, b)
        return self._cg_solve(c, alpha, b)

    def solve(self, c: float, alpha: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Differentiable solve of (c M - alpha L) x = b."""
        return _SPDSolve.apply(b, alpha, c, self)


class _SPDSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, b, alpha, c, op):
        try:
            x = op.solve_raw(c, alpha, b)
        except CGConvergenceError as exc:
            exc.info["phase"] = "forward"
            raise CGConvergenceError(exc.info) from exc
        if op.backend == "cg":
            op.last_cg_info["phase"] = "forward"
        ctx.save_for_backward(x, alpha)
        ctx.c, ctx.op = c, op
        return x

    @staticmethod
    def backward(ctx, g):
        x, alpha = ctx.saved_tensors
        op, c = ctx.op, ctx.c
        try:
            lam = op.solve_raw(c, alpha, g.contiguous())
        except CGConvergenceError as exc:
            exc.info["phase"] = "backward"
            raise CGConvergenceError(exc.info) from exc
        if op.backend == "cg":
            op.last_cg_info["phase"] = "backward"
        grad_alpha = None
        if ctx.needs_input_grad[1]:
            prod = lam * op.Lmv(x)                      # dA/dalpha = -L  ->  dL/dalpha = lam^T L x
            if alpha.numel() > 1:
                grad_alpha = prod.sum(0).reshape(alpha.shape)
            else:
                grad_alpha = prod.sum().reshape(alpha.shape)
        return lam, grad_alpha, None, None


# ---------------------------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------------------------
def _as_param_tensors(p, dtype, device):
    """GSParams or dict -> dict of tensors (scalars, or (B,) for a batch)."""
    d = p.as_dict() if isinstance(p, GSParams) else dict(p)
    return {k: torch.as_tensor(d[k], dtype=dtype, device=device) for k in ("F", "k", "Du", "Dv")}


class GrayScottSolver:
    """IMEX time stepper. Fields are (n,) or (n, B) tensors; parameters broadcast over the batch."""

    def __init__(self, geom: Geometry, dt: float = 1.0, scheme: str = "sbdf2", backend: str = "lu",
                 device="cpu", dtype=torch.float64, **op_kwargs):
        if scheme not in ("sbdf1", "sbdf2"):
            raise ValueError(scheme)
        self.geom, self.dt, self.scheme = geom, float(dt), scheme
        self.dtype, self.device = dtype, torch.device(device)
        self.op = SPDOperator(geom, backend=backend, device=device, dtype=dtype, **op_kwargs)

    def tensor(self, x) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x,
                               dtype=self.dtype, device=self.device)

    @staticmethod
    def reaction(u, v, P):
        r = u * v * v
        return -r + P["F"] * (1 - u), r - (P["F"] + P["k"]) * v

    def _mass(self, x):
        return self.op.M * x if x.dim() == 1 else self.op.M.unsqueeze(1) * x

    def _sbdf1(self, u, v, P):
        fu, fv = self.reaction(u, v, P)
        dt = self.dt
        un = self.op.solve(1.0, dt * P["Du"], self._mass(u + dt * fu))
        vn = self.op.solve(1.0, dt * P["Dv"], self._mass(v + dt * fv))
        return un, vn, fu, fv

    def _sbdf2(self, u, v, up, vp, fup, fvp, P):
        fu, fv = self.reaction(u, v, P)
        dt = self.dt
        un = self.op.solve(1.5, dt * P["Du"], self._mass(2 * u - 0.5 * up + dt * (2 * fu - fup)))
        vn = self.op.solve(1.5, dt * P["Dv"], self._mass(2 * v - 0.5 * vp + dt * (2 * fv - fvp)))
        return un, vn, fu, fv

    def _advance(self, n_steps, P, u, v, up, vp, fup, fvp):
        for _ in range(n_steps):
            if self.scheme == "sbdf1":
                u, v, fup, fvp = self._sbdf1(u, v, P)
            else:
                un, vn, fup, fvp = self._sbdf2(u, v, up, vp, fup, fvp, P)
                up, vp, u, v = u, v, un, vn
        return u, v, up, vp, fup, fvp

    def rollout(self, u0, v0, params, n_steps: int, record_every: int = 0,
                callback: Callable | None = None, checkpoint_every: int = 0):
        """Integrate n_steps from (u0, v0). Returns final (u, v) tensors (differentiable).

        callback(step, t, u, v, u_prev, v_prev) is called every `record_every` steps with detached
        tensors; returning True stops the integration early. `checkpoint_every` > 0 recomputes
        chunks in the backward pass, trading compute for memory on long differentiable horizons.
        """
        P = params if isinstance(params, dict) and torch.is_tensor(next(iter(params.values()))) \
            else _as_param_tensors(params, self.dtype, self.device)
        P = {key: val.to(device=self.device, dtype=self.dtype) for key, val in P.items()}   # keeps autograd
        u, v = self.tensor(u0), self.tensor(v0)
        # first step is always SBDF1 (it also seeds the SBDF2 history)
        un, vn, fup, fvp = self._sbdf1(u, v, P)
        up, vp, u, v = u, v, un, vn
        step = 1
        if callback is not None and record_every and callback(step, step * self.dt, u.detach(), v.detach(),
                                                              up.detach(), vp.detach()):
            return u, v
        chunk = record_every or checkpoint_every or n_steps
        if checkpoint_every:
            chunk = min(chunk, checkpoint_every)
        while step < n_steps:
            m = min(chunk - step % chunk, n_steps - step)       # land on multiples of the log interval
            if record_every:
                m = min(m, record_every - step % record_every)  # checkpoint chunks must not skip callbacks
            # SBDF2 history (up, vp) is u^{n-1}; for logging du/dt we keep the last two levels
            if checkpoint_every and torch.is_grad_enabled():
                # Bind this segment's length: backward recomputation happens
                # after the loop has reassigned m for the later segments.
                u, v, up, vp, fup, fvp = checkpoint(lambda *s, n_steps=m: self._advance(n_steps, P, *s),
                                                    u, v, up, vp, fup, fvp, use_reentrant=False)
            else:
                u, v, up, vp, fup, fvp = self._advance(m, P, u, v, up, vp, fup, fvp)
            step += m
            if callback is not None and record_every and step % record_every == 0:
                if self.scheme == "sbdf1":
                    up_log, vp_log = None, None
                else:
                    up_log, vp_log = up.detach(), vp.detach()
                if callback(step, step * self.dt, u.detach(), v.detach(), up_log, vp_log):
                    break
        return u, v
