"""Paired comparison on 64 random initial states: refined optimum vs the organisers' spot parameters
with Dv/Du = 1/3. Writes results/refine/paired_test.json.

    python scripts/refine_paired_test.py
"""
import sys, json, multiprocessing as mp, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import refine_cmaes as R
if __name__ == "__main__":
    seeds = list(80_000 + 11 * np.arange(32)) + list(95_000 + 11 * np.arange(32))   # 64 seeds
    A = np.array([0.03368, 0.06083, -0.1363, 0.3416]); B = np.array([0.028, 0.062, 0.0, 1/3])
    halves = [seeds[:16], seeds[16:32], seeds[32:48], seeds[48:]]
    with mp.get_context("spawn").Pool(8, R._init_worker, (20, 400, 10)) as pool:
        ra = pool.map(R.evaluate, [(A, h, False) for h in halves]); rb = pool.map(R.evaluate, [(B, h, False) for h in halves])
    da = np.concatenate([r[0] for r in ra]); db = np.concatenate([r[0] for r in rb])
    diff = db - da
    rng = np.random.default_rng(0)
    bs = [np.median(rng.choice(diff, len(diff))) for _ in range(4000)]
    print(f"64 seeds: refined median {np.median(da):.3f}, given-spots+Dv/Du=1/3 median {np.median(db):.3f}")
    print(f"paired difference (B-A): median {np.median(diff):.3f}, 95% bootstrap CI [{np.quantile(bs,0.025):.3f}, {np.quantile(bs,0.975):.3f}], B better on {np.mean(diff<0):.0%} of seeds")
    json.dump({"refined": da.tolist(), "given_spots_ratio_1_3": db.tolist()}, open(ROOT / "results" / "refine" / "paired_test.json", "w"))
