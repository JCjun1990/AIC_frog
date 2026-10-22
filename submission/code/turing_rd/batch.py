"""Batched NumPy integrator: many (F, k, initial state) columns sharing one (Du, Dv).

The diffusion matrices depend only on (Du, Dv, dt), so all columns share four sparse LU factors and
each solve handles every column at once (about 0.7 ms per column per solve on the frog, against
~14 ms for a single right-hand side). Columns can be retired (dead, blown up, steady) to save work.
Scheme: SBDF2 with an SBDF1 first step, as in solver.GrayScottSolver.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .geometry import Geometry


class BatchIntegrator:
    def __init__(self, geom: Geometry, Du: float, Dv: float, dt: float = 1.0):
        self.g, self.Du, self.Dv, self.dt = geom, Du, Dv, dt
        Md = sp.diags(geom.M)
        self.lu1u, self.lu1v = (spla.splu((Md - dt * D * geom.L).tocsc()) for D in (Du, Dv))
        self.lu2u, self.lu2v = (spla.splu((1.5 * Md - dt * D * geom.L).tocsc()) for D in (Du, Dv))
        self.Mc = geom.M[:, None]

    @staticmethod
    def react(u, v, F, k):
        r = u * v * v
        return -r + F * (1 - u), r - (F + k) * v

    def start(self, U0, V0, F, k, ids=None):
        """U0, V0: (n, B) initial states; F, k: (B,) parameters; ids: column labels."""
        self.F, self.k = np.asarray(F, float).copy(), np.asarray(k, float).copy()
        self.ids = np.arange(len(self.F)) if ids is None else np.asarray(ids)
        dt = self.dt
        fu, fv = self.react(U0, V0, self.F, self.k)
        self.up, self.vp, self.fup, self.fvp = U0, V0, fu, fv
        self.u = self.lu1u.solve(self.Mc * (U0 + dt * fu))
        self.v = self.lu1v.solve(self.Mc * (V0 + dt * fv))
        self.step = 1

    @property
    def t(self):
        return self.step * self.dt

    def advance(self, n_steps: int):
        dt, Mc = self.dt, self.Mc
        for _ in range(n_steps):
            fu, fv = self.react(self.u, self.v, self.F, self.k)
            un = self.lu2u.solve(Mc * (2 * self.u - 0.5 * self.up + dt * (2 * fu - self.fup)))
            vn = self.lu2v.solve(Mc * (2 * self.v - 0.5 * self.vp + dt * (2 * fv - self.fvp)))
            self.up, self.vp, self.u, self.v, self.fup, self.fvp = self.u, self.v, un, vn, fu, fv
        self.step += n_steps

    def rates(self):
        """rms |dv/dt| per column, from the last step."""
        dv = (self.v - self.vp) / self.dt
        return np.sqrt(self.g.M @ (dv * dv) / self.g.area)

    def keep(self, mask):
        """Retire every column where mask is False."""
        mask = np.asarray(mask, bool)
        for name in ("u", "v", "up", "vp", "fup", "fvp"):
            setattr(self, name, getattr(self, name)[:, mask])
        self.F, self.k, self.ids = self.F[mask], self.k[mask], self.ids[mask]

    @property
    def size(self):
        return len(self.ids)
