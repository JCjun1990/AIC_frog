"""Matplotlib rendering of vertex fields on surfaces and convergence histories."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from .geometry import Geometry  # noqa: E402

VIEWS = {"top": (90, -90), "bottom": (-90, -90), "front": (0, -90), "side": (0, 0),
         "iso": (25, -55), "iso_back": (25, 125)}


def _shade(geom: Geometry, faces, elev, azim):
    """Lambert shading factor per face for a light at the camera."""
    e, a = np.deg2rad(elev), np.deg2rad(azim)
    light = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    V = geom.V
    n = np.cross(V[faces[:, 1]] - V[faces[:, 0]], V[faces[:, 2]] - V[faces[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
    return 0.55 + 0.45 * np.abs(n @ light)


def render_field(geom: Geometry, f, path, title: str = "", views=("top", "iso", "iso_back"),
                 vmin=None, vmax=None, cmap: str = "viridis", label: str = "v", dpi: int = 110):
    f = np.asarray(f, dtype=float)
    faces = geom.mesh.quads if geom.mesh.quads is not None else geom.F
    vmin = np.quantile(f, 0.005) if vmin is None else vmin
    vmax = np.quantile(f, 0.995) if vmax is None else vmax
    norm = plt.Normalize(vmin, vmax)
    cm = plt.get_cmap(cmap)
    fig = plt.figure(figsize=(4.2 * len(views), 4.4))
    lo, hi = geom.V.min(0), geom.V.max(0)
    for i, name in enumerate(views):
        el, az = VIEWS[name]
        ax = fig.add_subplot(1, len(views), i + 1, projection="3d")
        col = cm(norm(f[faces].mean(1)))
        col[:, :3] *= _shade(geom, faces, el, az)[:, None]
        ax.add_collection3d(Poly3DCollection(geom.V[faces], facecolors=col, edgecolor="none"))
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo)
        ax.view_init(el, az)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
    fig.colorbar(sm, ax=fig.axes, shrink=0.6, label=label)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_history(rows: list[dict], path, title: str = "", polish: list[dict] | None = None):
    """Convergence record: |du/dt|, steady residual, pattern count and wavelength vs time,
    plus the steady-state polish (relative residual per Newton/PTC iteration) when given."""
    t = np.array([r["t"] for r in rows])
    fig, ax = plt.subplots(1, 4 if polish else 3, figsize=(18 if polish else 14, 3.6))
    if polish:
        it = [h["iter"] for h in polish]
        ax[3].semilogy(it, [h["rel_residual"] for h in polish], "C2o-", ms=4)
        ax[3].set_xlabel("PTC / Newton iteration")
        ax[3].set_title("steady polish: residual / F")
    for key, lab in [("dudt_rms", "|du/dt| rms"), ("dvdt_rms", "|dv/dt| rms"),
                     ("steady_Ru_rms", "steady R_u rms"), ("steady_Rv_rms", "steady R_v rms")]:
        y = np.array([r.get(key, np.nan) for r in rows], dtype=float)
        ax[0].semilogy(t, np.maximum(y, 1e-18), label=lab)
    ax[0].set_xlabel("t"); ax[0].legend(fontsize=8); ax[0].set_title("residual / rate")
    ax[1].plot(t, [r.get("n_components", 0) for r in rows], "k.-", ms=3)
    ax[1].set_xlabel("t"); ax[1].set_title("high-v components")
    ax[2].plot(t, [r.get("wavelength", np.nan) for r in rows], "C3.-", ms=3)
    ax[2].set_xlabel("t"); ax[2].set_title("wavelength (Dirichlet ratio)")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
