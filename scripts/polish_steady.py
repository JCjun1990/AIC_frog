"""Polish a saved state to a steady state of the discrete PDE and redo the diagnostics.

Works on any results directory that holds final.npz and summary.json (a baseline now, an AI
prediction later). The polished state is kept only if the residual drops and the number of
high-v components is unchanged; then solution.mat, final.npz, summary.json and field_v.png are
rewritten and the original state is kept as final_unpolished.npz.

    python scripts/polish_steady.py results/baseline/sphere_stripes
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from turing_rd import Geometry, GSParams, icosphere, load_frog  # noqa: E402
from turing_rd.diagnostics import linear_stability, pattern_stats, pde_residual  # noqa: E402
from turing_rd.steady import ptc_polish  # noqa: E402
from turing_rd.viz import render_field  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--tol", type=float, default=1e-11)
    ap.add_argument("--dtau0", type=float, default=1e3)
    ap.add_argument("--max-iter", type=int, default=40)
    args = ap.parse_args()
    d = Path(args.run_dir)
    s = json.loads((d / "summary.json").read_text())
    a = s["args"]
    mesh = icosphere(a["level"]) if s["surface"] == "sphere" else load_frog(a["frog"])
    geom = Geometry(mesh, laplacian=a["laplacian"], cache_dir=ROOT / "results" / "cache")
    p = GSParams(**s["params"])
    z = np.load(d / "final.npz")
    u, v = z["u"], z["v"]
    t0 = time.time()
    up, vp, hist = ptc_polish(geom, p, u, v, tol=args.tol, dtau0=args.dtau0, max_iter=args.max_iter, verbose=True)
    same = pattern_stats(geom, vp).get("n_components") == pattern_stats(geom, v).get("n_components")
    accepted = bool(hist[-1]["rel_residual"] < 0.1 * hist[0]["rel_residual"] and same)
    print(f"residual {hist[0]['rel_residual']:.2e} -> {hist[-1]['rel_residual']:.2e}, accepted={accepted}")
    s["polish"] = {"history": hist, "accepted": accepted, "seconds": time.time() - t0,
                   "max_abs_change_v": float(np.abs(vp - v).max()), "source": "scripts/polish_steady.py"}
    if accepted:
        np.savez(d / "final_unpolished.npz", **{k: z[k] for k in z.files})
        u, v = up, vp
        np.savez(d / "final.npz", u=u, v=v, vertices=geom.V, faces=geom.F,
                 **{k: z[k] for k in ("u0", "v0") if k in z.files})
        sio.savemat(d / "solution.mat", {"Solution": {"u": u[:, None], "v": v[:, None]}})
        s["final_pattern"] = pattern_stats(geom, v)
        s["final_residual"] = pde_residual(geom, p, u, v)
        st = linear_stability(geom, p, u, v)
        s["linear_stability"] = st
        print("eigenvalues nearest 0:", [tuple(round(x, 7) for x in e) for e in st["eigenvalues"][:6]])
        fp = s["final_pattern"]
        views = ("top", "iso", "iso_back") if s["surface"] == "frog" else ("top", "front", "iso")
        title = (f"{s['surface']} · {s['pattern']} (F={p.F}, k={p.k}) · t={s['time']['t_final']:.0f} + polish · "
                 f"{fp['type']}, n={fp.get('n_components', 0)}, λ={fp['wavelength']:.3f}")
        render_field(geom, v, d / "field_v.png", title=title, views=views, vmin=0.0, vmax=max(0.3, v.max()))
    (d / "summary.json").write_text(json.dumps(s, indent=2, default=float))


if __name__ == "__main__":
    main()
