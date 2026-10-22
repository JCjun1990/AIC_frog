"""Surface meshes: the competition frog, unit icospheres, and .ply export."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.io as sio


@dataclass
class Mesh:
    """Triangle mesh. `quads` keeps the original polygon faces when the source had them."""
    V: np.ndarray                      # (n, 3) float64 vertex positions
    F: np.ndarray                      # (m, 3) int64 triangles, 0-based
    name: str = "mesh"
    quads: np.ndarray | None = field(default=None, repr=False)

    @property
    def n_vertices(self) -> int:
        return len(self.V)


def triangulate_quads(V: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Split each quad along its shorter diagonal (keeps triangles closer to equilateral)."""
    d02 = np.linalg.norm(V[Q[:, 0]] - V[Q[:, 2]], axis=1)
    d13 = np.linalg.norm(V[Q[:, 1]] - V[Q[:, 3]], axis=1)
    use02 = d02 <= d13
    T = np.empty((2 * len(Q), 3), dtype=np.int64)
    T[0::2] = np.where(use02[:, None], Q[:, [0, 1, 2]], Q[:, [0, 1, 3]])
    T[1::2] = np.where(use02[:, None], Q[:, [0, 2, 3]], Q[:, [1, 2, 3]])
    return T


def load_frog(path: str | Path = "frog.mat") -> Mesh:
    """Load the organisers' frog surface (MATLAB struct `fv` with 1-based quad `faces`)."""
    fv = sio.loadmat(str(path))["fv"]
    V = np.asarray(fv["vertices"][0, 0], dtype=np.float64)
    Q = np.asarray(fv["faces"][0, 0], dtype=np.int64) - 1
    if Q.shape[1] == 3:
        return Mesh(V, Q, name="frog")
    return Mesh(V, triangulate_quads(V, Q), name="frog", quads=Q)


def icosphere(level: int, radius: float = 1.0) -> Mesh:
    """Unit icosphere by repeated 4-way subdivision; level L has 10*4^L + 2 vertices."""
    t = (1.0 + 5 ** 0.5) / 2.0
    V = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
         [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    F = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
         [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
         [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    V = np.asarray(V, dtype=np.float64)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    F = np.asarray(F, dtype=np.int64)
    for _ in range(level):
        E = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
        E, inv = np.unique(E, axis=0, return_inverse=True)
        mid = V[E[:, 0]] + V[E[:, 1]]
        mid /= np.linalg.norm(mid, axis=1, keepdims=True)
        m = len(F)
        a, b, c = (len(V) + inv.reshape(-1)[i * m:(i + 1) * m] for i in range(3))
        V = np.vstack([V, mid])
        F = np.vstack([np.c_[F[:, 0], a, c], np.c_[F[:, 1], b, a],
                       np.c_[F[:, 2], c, b], np.c_[a, b, c]])
    return Mesh(radius * V, F, name=f"sphere_L{level}")


def save_ply(path: str | Path, mesh: Mesh, scalar: np.ndarray, vmin=None, vmax=None,
             cmap: str = "viridis") -> None:
    """Write a binary PLY with per-vertex colours (opens in MeshLab / ParaView / Blender)."""
    import matplotlib
    vmin = np.quantile(scalar, 0.01) if vmin is None else vmin
    vmax = np.quantile(scalar, 0.99) if vmax is None else vmax
    rgb = (matplotlib.colormaps[cmap](np.clip((scalar - vmin) / (vmax - vmin + 1e-30), 0, 1))[:, :3]
           * 255).astype(np.uint8)
    faces = mesh.quads if mesh.quads is not None else mesh.F
    k = faces.shape[1]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(mesh.V)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "property float value\n"
              f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
    vdt = np.dtype([("p", "<f4", 3), ("c", "u1", 3), ("s", "<f4")])
    vert = np.empty(len(mesh.V), dtype=vdt)
    vert["p"], vert["c"], vert["s"] = mesh.V, rgb, scalar
    fdt = np.dtype([("k", "u1"), ("i", "<i4", k)])
    face = np.empty(len(faces), dtype=fdt)
    face["k"], face["i"] = k, faces
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(vert.tobytes())
        fh.write(face.tobytes())
