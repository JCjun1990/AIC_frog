"""Summarise results/scan/*: ranking, combined heatmaps, best match rendered beside the target.

Writes results/scan/ranking.csv, heatmaps.png and best_match.png.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.io as sio  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from turing_rd import Geometry, load_frog  # noqa: E402
from turing_rd.diagnostics import pattern_stats  # noqa: E402
from turing_rd.viz import render_field  # noqa: E402


def main(top_n: int = 15):
    base = ROOT / "results" / "scan"
    groups = sorted(d for d in base.iterdir() if (d / "scan.csv").exists())
    rows = []
    for d in groups:
        for r in csv.DictReader(open(d / "scan.csv")):
            r = {k: (float(v) if k != "status" else v) for k, v in r.items()}
            r["group"] = d.name
            rows.append(r)
    rows.sort(key=lambda r: -r["best_corr_v"])
    with open(base / "ranking.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} trajectories in {len(groups)} groups; top {top_n} by pointwise corr(v):")
    for r in rows[:top_n]:
        print(f"  corr_v={r['best_corr_v']:.3f} corr_u={r['best_corr_u']:.3f} relL2={r['rel_l2_v']:.2f} "
              f"F={r['F']:.4f} k={r['k']:.4f} Du={r['Du']:.3g} Dv={r['Dv']:.3g} t={r['t_best']:.0f} [{r['status']}]")

    n = len(groups)
    cols = min(4, n)
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(4.2 * cols, 3.6 * int(np.ceil(n / cols))),
                             squeeze=False)
    vmax = max(0.3, rows[0]["best_corr_v"])
    for ax, d in zip(axes.flat, groups):
        rs = [r for r in rows if r["group"] == d.name]
        Fs, ks = sorted({r["F"] for r in rs}), sorted({r["k"] for r in rs})
        Z = np.full((len(ks), len(Fs)), np.nan)
        for r in rs:
            Z[ks.index(r["k"]), Fs.index(r["F"])] = r["best_corr_v"]
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=vmax,
                       extent=[Fs[0], Fs[-1], ks[0], ks[-1]])
        ax.set_title(d.name, fontsize=9)
        ax.set_xlabel("F", fontsize=8); ax.set_ylabel("k", fontsize=8)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.8, label="best pointwise corr(v, target)")
    fig.savefig(base / "heatmaps.png", dpi=100, bbox_inches="tight")

    best = rows[0]
    z = np.load(base / best["group"] / "best_states.npz")
    j = int(np.argmax(z["corr_v"]))
    g = Geometry(load_frog(ROOT / "frog.mat"), cache_dir=None)
    S = sio.loadmat(str(ROOT / "Solution.mat"))["Solution"]
    vt = S["v"][0, 0].ravel()
    vb = z["v"][:, j]
    st, sb = pattern_stats(g, vt), pattern_stats(g, vb)
    tmp_t, tmp_b = base / "_target.png", base / "_best.png"
    render_field(g, vt, tmp_t, title=f"target Solution.mat · n={st.get('n_components')}, λ={st['wavelength']:.3f}",
                 vmin=0, vmax=0.5)
    render_field(g, vb, tmp_b, title=(f"best scan match · F={best['F']:.4f}, k={best['k']:.4f}, Du={best['Du']:.3g}, "
                                      f"Dv={best['Dv']:.3g}, t={best['t_best']:.0f} · corr={best['best_corr_v']:.3f} · "
                                      f"n={sb.get('n_components')}, λ={sb['wavelength']:.3f}"), vmin=0, vmax=0.5)
    fig, ax = plt.subplots(2, 1, figsize=(12, 7.5))
    for a, p in zip(ax, (tmp_t, tmp_b)):
        a.imshow(plt.imread(p)); a.axis("off")
    fig.tight_layout(); fig.savefig(base / "best_match.png", dpi=110)
    tmp_t.unlink(); tmp_b.unlink()


if __name__ == "__main__":
    main()
