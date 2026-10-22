"""I03: render the four formal candidates from the submitted MAT bytes.

Produces one multi-view figure per candidate (used as the package visualization)
and a compact 2x2 panel for the two-page report.  Nothing is inferred from old
images: every pixel comes from the MAT that is actually submitted.

    python scripts/i03_render_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import matplotlib                                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
import numpy as np                                                  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection             # noqa: E402

from turing_rd import Geometry, icosphere, load_frog                # noqa: E402
from turing_rd.result_io import read_mat_fields                     # noqa: E402
from turing_rd.viz import VIEWS, _shade, render_field               # noqa: E402

DELIVERY = ROOT / "results" / "i03_delivery"
FIGURES = DELIVERY / "figures"
TITLES = {
    "task1_sphere_spots": "Unit sphere · spots (45)",
    "task1_sphere_stripes": "Unit sphere · stripes (7 bands)",
    "task2_frog_spots": "Frog surface · spots (22)",
    "task2_frog_stripes": "Frog surface · maze (5 segments)",
}
SHORT = {
    "task1_sphere_spots": "(a) sphere spots",
    "task1_sphere_stripes": "(b) sphere stripes",
    "task2_frog_spots": "(c) frog spots",
    "task2_frog_stripes": "(d) frog maze",
}


def load_candidates():
    document = json.loads((DELIVERY / "candidates.json").read_text(encoding="utf-8"))
    return document["candidates"]


def geometry(surface):
    mesh = load_frog(ROOT / "frog.mat") if surface == "frog" else icosphere(6)
    return Geometry(mesh, laplacian="robust", cache_dir=None)


def panel(axis, geom, field, title, elev=22, azim=-58):
    faces = geom.mesh.quads if geom.mesh.quads is not None else geom.F
    vmin, vmax = np.quantile(field, 0.005), np.quantile(field, 0.995)
    norm = plt.Normalize(vmin, vmax)
    colors = plt.get_cmap("viridis")(norm(field[faces].mean(1)))
    colors[:, :3] *= _shade(geom, faces, elev, azim)[:, None]
    axis.add_collection3d(Poly3DCollection(geom.V[faces], facecolors=colors, edgecolor="none"))
    lo, hi = geom.V.min(0), geom.V.max(0)
    axis.set_xlim(lo[0], hi[0]); axis.set_ylim(lo[1], hi[1]); axis.set_zlim(lo[2], hi[2])
    axis.set_box_aspect(hi - lo)
    axis.view_init(elev, azim)
    axis.set_title(title, fontsize=8.5, pad=1)
    axis.axis("off")
    return norm


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates()
    geometries = {}
    fig = plt.figure(figsize=(7.6, 2.35))
    norm = None
    for index, (key, candidate) in enumerate(candidates.items()):
        surface = candidate["surface"]
        if surface not in geometries:
            geometries[surface] = geometry(surface)
        geom = geometries[surface]
        u, v = read_mat_fields(ROOT / candidate["mat"], geom.n)
        title = TITLES.get(key, key)
        render_field(geom, v, FIGURES / f"{key}.png", title=title,
                     views=("top", "iso", "iso_back"), dpi=120)
        axis = fig.add_subplot(1, 4, index + 1, projection="3d")
        norm = panel(axis, geom, v, SHORT.get(key, title), elev=24, azim=-60)
        print(f"{key}: {geom.n} vertices, v in [{v.min():.4f}, {v.max():.4f}] -> {key}.png")
    fig.subplots_adjust(left=0.0, right=0.9, bottom=0.02, top=0.9, wspace=0.02)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    fig.colorbar(scalar, ax=fig.axes, shrink=0.75, pad=0.01, label="v")
    fig.savefig(FIGURES / "report_panel.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    from PIL import Image, ImageChops
    image = Image.open(FIGURES / "report_panel.png").convert("RGB")
    background = Image.new("RGB", image.size, (255, 255, 255))
    box = ImageChops.difference(image, background).getbbox()
    if box:
        margin = 8
        image.crop((max(0, box[0] - margin), max(0, box[1] - margin),
                    min(image.width, box[2] + margin), min(image.height, box[3] + margin))
                   ).save(FIGURES / "report_panel.png")
    (FIGURES / "README.txt").write_text(
        "All images are rendered from the submitted MAT files by scripts/i03_render_figures.py.\n"
        "report_panel.png is the 4-panel strip used in the two-page technical report.\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
