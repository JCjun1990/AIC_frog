"""Geometry verification: convergence of the discrete Laplace-Beltrami operator and curvatures.

Unit sphere, icosphere levels 2-6: LB eigenvalues vs l(l+1), mean / Gaussian curvature vs 1,
area vs 4 pi, Gauss-Bonnet. Frog: mesh statistics for both Laplacians.
Writes results/verification/geometry.json and geometry_convergence.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from turing_rd import Geometry, icosphere, load_frog  # noqa: E402


def order(h, e):
    return float(np.polyfit(np.log(h), np.log(e), 1)[0])


def main():
    out = ROOT / "results" / "verification"
    out.mkdir(parents=True, exist_ok=True)
    l_max = 6
    exact = np.concatenate([[l * (l + 1)] * (2 * l + 1) for l in range(l_max + 1)])
    report = {"sphere": {}, "frog": {}}
    for lap in ("robust", "cotan"):
        rows = []
        for level in range(2, 7):
            g = Geometry(icosphere(level), laplacian=lap, cache_dir=None)
            lam, _ = g.eigenbasis(len(exact))
            rel = np.abs(lam[1:] - exact[1:]) / exact[1:]
            rows.append({"level": level, "n": g.n, "h": g.h_med,
                         "eig_rel_err_max": float(rel.max()),
                         "H_rms_err": g.rms(g.H - 1), "K_rms_err": g.rms(g.K - 1),
                         "area_rel_err": abs(g.area - 4 * np.pi) / (4 * np.pi),
                         "gauss_bonnet_chi": g.integrate(g.K) / (2 * np.pi)})
            print(lap, rows[-1], flush=True)
        h = np.array([r["h"] for r in rows])
        report["sphere"][lap] = {
            "levels": rows,
            "order_eig": order(h, [r["eig_rel_err_max"] for r in rows]),
            "order_H": order(h, [r["H_rms_err"] for r in rows]),
            "order_K": order(h, [r["K_rms_err"] for r in rows]),
            "order_area": order(h, [r["area_rel_err"] for r in rows]),
        }
        print(lap, {k: round(v, 2) for k, v in report["sphere"][lap].items() if k.startswith("order")})
    for lap in ("robust", "cotan"):
        g = Geometry(load_frog(ROOT / "frog.mat"), laplacian=lap, cache_dir=None)
        s = g.summary()
        s.update(H_quantiles=np.quantile(g.H, [0.01, 0.5, 0.99]).tolist(),
                 gauss_bonnet_chi=g.integrate(g.K) / (2 * np.pi))
        report["frog"][lap] = s
    (out / "geometry.json").write_text(json.dumps(report, indent=2))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    rows = report["sphere"]["robust"]["levels"]
    h = [r["h"] for r in rows]
    for key, lab in [("eig_rel_err_max", "LB eigenvalues (l ≤ 6)"), ("H_rms_err", "mean curvature H"),
                     ("K_rms_err", "Gaussian curvature K"), ("area_rel_err", "area")]:
        ax.loglog(h, [r[key] for r in rows], "o-", label=f"{lab}, slope {order(h, [r[key] for r in rows]):.2f}")
    ax.set_xlabel("median edge length h")
    ax.set_ylabel("relative / rms error")
    ax.set_title("Unit sphere, robust Laplacian")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "geometry_convergence.png", dpi=120)


if __name__ == "__main__":
    main()
