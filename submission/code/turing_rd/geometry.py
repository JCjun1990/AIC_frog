"""Discrete differential geometry on triangle meshes.

Conventions
-----------
* `L` is the weak (stiffness) Laplacian, negative semidefinite: (L u)_i = sum_j w_ij (u_j - u_i).
* `M` is the lumped (barycentric) mass, a vector of vertex areas.
* The strong Laplace-Beltrami operator is Delta u ~= M^{-1} L u.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .mesh import Mesh


def _tri_areas(V, F):
    return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)


def cotan_laplacian(V: np.ndarray, F: np.ndarray):
    """Classical cotangent Laplacian (Pinkall-Polthier / Meyer et al. 2003) + barycentric mass."""
    n = len(V)
    rows, cols, vals = [], [], []
    for a, b, c in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        e1 = V[F[:, a]] - V[F[:, c]]
        e2 = V[F[:, b]] - V[F[:, c]]
        cot = np.einsum("ij,ij->i", e1, e2) / np.linalg.norm(np.cross(e1, e2), axis=1)
        rows += [F[:, a], F[:, b]]
        cols += [F[:, b], F[:, a]]
        vals += [0.5 * cot, 0.5 * cot]
    W = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n)).tocsr()
    L = (W - sp.diags(np.asarray(W.sum(axis=1)).ravel())).tocsr()
    M = np.bincount(F.ravel(), np.repeat(_tri_areas(V, F), 3) / 3.0, minlength=n)
    return L, M


def robust_laplacian(V: np.ndarray, F: np.ndarray, mollify_factor: float = 1e-5):
    """Intrinsic-Delaunay cotan Laplacian (Sharp & Crane 2020): non-negative edge weights."""
    import robust_laplacian as rl
    Lp, Mp = rl.mesh_laplacian(np.ascontiguousarray(V), np.ascontiguousarray(F),
                               mollify_factor=mollify_factor)
    return (-Lp).tocsr(), np.asarray(Mp.diagonal()).ravel()


class Geometry:
    """Everything the PDE needs from a surface, computed once and cached.

    Attributes: V, F, n, L, M, area, normals, H (mean curvature), K (Gaussian curvature),
    theta / z (cylindrical coordinates), edges, adjacency, edge length statistics.
    """

    def __init__(self, mesh: Mesh, laplacian: str = "robust", cache_dir: str | Path | None = "results/cache"):
        V, F = mesh.V.astype(np.float64), mesh.F.astype(np.int64)
        # orient faces outward (positive signed volume) so that H > 0 on convex parts
        vol = np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6.0
        if vol < 0:
            F = F[:, [0, 2, 1]]
        self.mesh, self.V, self.F, self.n = mesh, V, F, len(V)
        self.name, self.laplacian = mesh.name, laplacian
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if laplacian == "cotan":
            self.L, self.M = cotan_laplacian(V, F)
        elif laplacian == "robust":
            self.L, self.M = robust_laplacian(V, F)
        else:
            raise ValueError(f"unknown laplacian '{laplacian}'")
        self.area = float(self.M.sum())

        E = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
        self.edges = np.unique(E, axis=0)
        el = np.linalg.norm(V[self.edges[:, 0]] - V[self.edges[:, 1]], axis=1)
        self.h_min, self.h_med, self.h_max = float(el.min()), float(np.median(el)), float(el.max())
        A = sp.coo_matrix((np.ones(len(self.edges)), (self.edges[:, 0], self.edges[:, 1])),
                          shape=(self.n, self.n))
        self.adjacency = (A + A.T).tocsr()

        self.theta = np.arctan2(V[:, 1], V[:, 0])
        self.z = V[:, 2].copy()
        self.normals = self._vertex_normals()
        self.H = self._mean_curvature()
        self.K = self._gaussian_curvature()
        self._eig = None

    # ---- operators -------------------------------------------------------------
    def lap(self, u: np.ndarray) -> np.ndarray:
        """Strong Laplace-Beltrami M^{-1} L u (u may be (n,) or (n, B))."""
        Lu = self.L @ u
        return Lu / (self.M if Lu.ndim == 1 else self.M[:, None])

    def integrate(self, f: np.ndarray) -> float:
        return float(self.M @ f)

    def mean(self, f: np.ndarray) -> float:
        return self.integrate(f) / self.area

    def rms(self, f: np.ndarray) -> float:
        """Area-weighted root mean square, the norm used for all residuals."""
        return float(np.sqrt(self.M @ (f * f) / self.area))

    def negative_weight_fraction(self) -> float:
        off = self.L.copy()
        off.setdiag(0)
        off.eliminate_zeros()
        return float((off.data < 0).mean())

    # ---- curvature -------------------------------------------------------------
    def _vertex_normals(self):
        V, F = self.V, self.F
        fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])  # area-weighted
        N = np.zeros_like(V)
        for i in range(3):
            np.add.at(N, F[:, i], fn)
        return N / np.linalg.norm(N, axis=1, keepdims=True)

    def _mean_curvature(self):
        # Delta_Gamma x = -2 H n  (H = mean of principal curvatures, outward normal)
        return -0.5 * np.einsum("ij,ij->i", self.lap(self.V), self.normals)

    def _gaussian_curvature(self):
        V, F = self.V, self.F
        defect = np.full(self.n, 2 * np.pi)
        for a, b, c in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
            e1 = V[F[:, b]] - V[F[:, a]]
            e2 = V[F[:, c]] - V[F[:, a]]
            ang = np.arctan2(np.linalg.norm(np.cross(e1, e2), axis=1), np.einsum("ij,ij->i", e1, e2))
            np.subtract.at(defect, F[:, a], ang)
        return defect / self.M

    # ---- spectral --------------------------------------------------------------
    def _cache_key(self, tag: str) -> Path | None:
        if self.cache_dir is None:
            return None
        h = hashlib.sha1(self.V.tobytes() + self.F.tobytes() + self.laplacian.encode()).hexdigest()[:12]
        return self.cache_dir / f"{self.name}_{self.laplacian}_{tag}_{h}.npz"

    def eigenbasis(self, k: int = 200):
        """First k Laplace-Beltrami eigenpairs: -L phi = lam M phi, phi^T M phi = I (lam ascending)."""
        if self._eig is not None and self._eig[0].size >= k:
            return self._eig[0][:k], self._eig[1][:, :k]
        path = self._cache_key(f"eig{k}")
        if path is not None and path.exists():
            d = np.load(path)
            self._eig = (d["lam"], d["phi"])
            return self._eig
        lam, phi = spla.eigsh(-self.L.tocsc(), k=k, M=sp.diags(self.M).tocsc(), sigma=-1e-3, which="LM")
        order = np.argsort(lam)
        lam, phi = lam[order], phi[:, order]
        lam[0] = max(lam[0], 0.0)
        self._eig = (lam, phi)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, lam=lam, phi=phi)
        return self._eig

    def summary(self) -> dict:
        return {
            "name": self.name, "laplacian": self.laplacian, "vertices": self.n, "triangles": len(self.F),
            "area": self.area, "euler_characteristic": int(self.n - len(self.edges) + len(self.F)),
            "edge_min": self.h_min, "edge_median": self.h_med, "edge_max": self.h_max,
            "negative_weight_fraction": self.negative_weight_fraction(),
            "bbox_extent": (self.V.max(0) - self.V.min(0)).tolist(),
        }
