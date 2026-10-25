"""Sphere spots: AI spot layout from a repulsion-energy minimum instead of a centroidal Voronoi layout.

The archived multi-seed audit of the sphere spot task (23 seeds, AI initial state = network spot
profile on CVT centres) reached a complete stable steady state in 14/23 runs: 6 were still slowly
rearranging their 45 spots at t = 2e5 and 3 ended with a weak positive mode. Steady spot patterns are
close to minimum-energy arrangements of mutually repelling spots (cf. the Thomson problem; Smale's 7th
problem), so here the n spot centres come from a Riesz-energy minimum on the sphere (projected gradient
descent from a seeded random start, E = sum 1/|x_i - x_j|), then the same network profile, the same
rotation-aware Newton + orientation, and the full Jacobian spectrum (shift-invert at 0 and 1e-3, all
modes incl. rotation-like). Only the layout differs from the audited pipeline. Formal results untouched.

    python scripts/sphere_spots_layout.py --seeds $(seq 1 23) --workers 8
    python scripts/sphere_spots_layout.py --summary
"""
from __future__ import annotations

import os

for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import ai_generation as A  # noqa: E402
from turing_rd import SPOTS, GSParams, ai_init  # noqa: E402
from turing_rd.diagnostics import full_steady_residual, pattern_stats  # noqa: E402
from turing_rd.stability import solve_spectrum  # noqa: E402
from turing_rd.spectral_policy import candidate_decision, screen_candidate  # noqa: E402
from turing_rd.steady import relax_polish  # noqa: E402

OUT = ROOT / "results" / "sphere_spots_layout"
MODEL = next((q for q in (ROOT / "submission/task1_sphere/ai_model/model.joblib",
                          ROOT.parent / "task1_sphere/ai_model/model.joblib") if q.exists()),
             ROOT / "submission/task1_sphere/ai_model/model.joblib")
AUDIT = {"seeds": 23, "complete_and_no_positive_mode": 14, "not_converged": 6, "positive_mode": 3}


def riesz_points(n, seed, iters=3000, step=0.02):
    """n unit vectors at a (local) minimum of sum 1/|x_i - x_j| by projected gradient descent."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    for it in range(iters):
        d = x[:, None, :] - x[None, :, :]
        r = np.linalg.norm(d, axis=2) + np.eye(n)
        force = (d / r[..., None] ** 3).sum(1)                      # -grad of the energy
        force -= (force * x).sum(1, keepdims=True) * x               # tangential part
        x += step * force / max(np.abs(force).max(), 1e-12) * (1 - it / iters)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def riesz_energy(x):
    d = np.linalg.norm(x[:, None] - x[None], axis=2)
    return float((1 / d[np.triu_indices(len(x), 1)]).sum())


def run(surface, seed, t_max=200000.0):
    import joblib
    from scipy.spatial import cKDTree
    g, p = A.geom(surface), GSParams(SPOTS.F, SPOTS.k)
    rot = A.rotations(surface)
    pred = A.predict(joblib.load(MODEL), p.F, p.k)
    n = max(1, int(round(pred["density"] * g.area)))
    t0 = time.time()
    x = riesz_points(n, seed)
    centres = cKDTree(g.V / np.linalg.norm(g.V, axis=1, keepdims=True)).query(x)[1]
    u0, v0 = ai_init.place_spots(g, centres, pred)
    setup = time.time() - t0
    u, v, info = relax_polish(g, p, u0, v0, schedule=A.AI_SCHEDULE(t_max), check_every=500.0, tol=A.TOL,
                              rotations=rot, orient=True)
    full = full_steady_residual(g, p, u, v)["full_joint_rms_over_F"]
    st = pattern_stats(g, v)
    row = {"seed": seed, "n_centres": n, "riesz_energy": riesz_energy(x), "setup_s": setup, "type": st.get("type"),
           "n_spots": st.get("n_components"), "full_residual": float(full), "relax_time": info["relax_time"],
           "relax_steps": info["relax_steps"], "wall_s": time.time() - t0, "spectra": []}
    row.update(screen_candidate(g, p, u, v, full, solve_spectrum))
    np.savez_compressed(OUT / f"state_seed{seed}.npz", u=u, v=v)
    return row


def summary():
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("row_seed*.json"))]
    decisions = [candidate_decision(r) for r in rows]
    ok = [r for r, d in zip(rows, decisions) if d["numerical_check_passed"]]
    out = {"runs": len(rows), "complete_and_stable": len(ok),
           "not_converged": sum(r["full_residual"] > A.TOL for r in rows),
           "positive_mode": sum(d["status"] == "unstable_positive_mode" for d in decisions),
           "insufficient_evidence": sum(d["outcome"] == "insufficient" for d in decisions),
           "strict_reassessment": decisions,
           "median_relax_steps": float(np.median([r["relax_steps"] for r in rows])) if rows else None,
           "spot_counts": sorted({r["n_spots"] for r in rows}), "audit_cvt_layout": AUDIT, "rows": rows}
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k != "rows"}, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--surface", default="sphere6")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.seeds:
        A.geom(args.surface).eigenbasis(600)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        todo = [s for s in args.seeds if not (OUT / f"row_seed{s}.json").exists()]
        with ProcessPoolExecutor(args.workers) as ex:
            futs = {ex.submit(run, args.surface, s): s for s in todo}
            for f in as_completed(futs):
                r = f.result()
                (OUT / f"row_seed{r['seed']}.json").write_text(json.dumps(r, indent=1))
                print(f"seed {r['seed']}: {r['n_spots']} spots, residual {r['full_residual']:.1e}, max Re {r.get('max_real')}, "
                      f"stable {r['stable']}, steps {r['relax_steps']} ({r['wall_s']:.0f}s)", flush=True)
    summary()


if __name__ == "__main__":
    main()
