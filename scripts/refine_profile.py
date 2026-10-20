"""One-dimensional profiles of the refinement objective around the optimum.

For each parameter, J is evaluated on the independent uncertainty seeds of refine_cmaes.py at
offsets of -6 ... +6 search-scale units with the other parameters fixed at the optimum. The interval
where J stays within the acceptance threshold (optimum + 2 bootstrap standard errors) is the
conditional uncertainty of that parameter. Correlated directions (F and k along the saddle-node
band) are wider than these conditional intervals; see uncertainty.png from refine_cmaes.py.

Writes results/refine/profiles.png and profiles.json.

    python scripts/refine_profile.py
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import refine_cmaes as R  # noqa: E402

OFFSETS = [-6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6]
SCALE = np.array([0.0016, 0.0014, 0.096, 0.018])


def main(workers: int = 8, n_seeds: int = 16):
    final = json.loads((R.OUT / "final.json").read_text())
    opt = np.array([final["theta"][n] for n in R.NAMES])
    seeds = list(90_000 + 11 * np.arange(n_seeds))          # the uncertainty seeds of refine_cmaes.py
    jobs, keys = [], []
    for i in range(4):
        for o in OFFSETS:
            th = opt.copy()
            th[i] = np.clip(opt[i] + o * SCALE[i], R.LO[i], R.HI[i])
            jobs.append((th, seeds, False))
            keys.append((i, o, float(th[i])))
    with mp.get_context("spawn").Pool(workers, R._init_worker, (20, 400, 10)) as pool:
        res = pool.map(R.evaluate, jobs)
    J = np.array([np.median(r[0]) for r in res])
    j0 = np.median([J[k] for k, (i, o, _) in enumerate(keys) if o == 0])
    se = R.boot_se(res[keys.index(next(k for k in keys if k[0] == 0 and k[1] == 0))][0])
    thr = j0 + 2 * se
    out = {"J_opt": float(j0), "se": se, "threshold": float(thr), "profiles": {}}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.8))
    for i, name in enumerate(R.NAMES):
        pts = [(v, J[k]) for k, (ii, o, v) in enumerate(keys) if ii == i]
        xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        ok = ys <= thr
        # contiguous accepted run around the optimum
        c = OFFSETS.index(0)
        lo, hi = c, c
        while lo > 0 and ok[lo - 1]:
            lo -= 1
        while hi < len(ok) - 1 and ok[hi + 1]:
            hi += 1
        out["profiles"][name] = {"values": xs.tolist(), "J": ys.tolist(),
                                 "interval": [float(xs[lo]), float(xs[hi])],
                                 "bounded": [bool(lo > 0), bool(hi < len(ok) - 1)]}
        a = axs[i]
        a.plot(xs, ys, "o-")
        a.axhline(thr, color="C3", ls="--", lw=1, label="acceptance threshold")
        a.axvspan(xs[lo], xs[hi], color="C2", alpha=0.15, label="accepted interval")
        a.axvline(opt[i], color="k", lw=1)
        a.set_xlabel(name); a.set_ylabel("J (independent seeds)")
        a.legend(fontsize=7)
        print(f"{name:12s} interval [{xs[lo]:.4g}, {xs[hi]:.4g}] (opt {opt[i]:.4g}) "
              f"{'' if lo > 0 and hi < len(ok) - 1 else '— not bounded inside the probed range'}", flush=True)
    fig.suptitle("Conditional profiles of the objective around the refined optimum")
    fig.tight_layout(); fig.savefig(R.OUT / "profiles.png", dpi=110)
    (R.OUT / "profiles.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "profiles"}))


if __name__ == "__main__":
    main()
