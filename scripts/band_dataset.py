"""Snapshot dataset inside the phase 2 parameter band, for the identifiability study.

One (Du, Dv) group per run. (F, k) are sampled around the Gray-Scott saddle-node line,
k = sqrt(F)/2 - F + delta with delta in [-0.003, 0.010], F in [0.015, 0.04]; every (F, k) is run
from several random initial states (baseline protocol: z-normalised suggested IC + seeds + noise,
a different seed per trajectory). Pattern fingerprints (turing_rd.features) are recorded at fixed
snapshot times. Dead or blown-up trajectories stop; steady ones are recorded once and stop.

Writes results/band/<tag>.npz with X (rows x features), the feature names and per-row F, k, Du, Dv,
ic_seed, trajectory id, t, kind (spots / other / uniform) and status.

    python scripts/band_dataset.py --group 0 --du-scale 0.7 --ratio 0.25
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from turing_rd import Geometry, load_frog  # noqa: E402
from turing_rd.batch import BatchIntegrator  # noqa: E402
from turing_rd.features import FeatureExtractor  # noqa: E402
from turing_rd.initial import perturbed_ic  # noqa: E402

DU0 = 2.78e-4
SNAPS = [50, 100, 150, 200, 300, 400, 500, 650, 800, 1000, 1250, 1500, 2000, 2500, 3000]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", type=int, required=True, help="group index (sets seeds and trajectory ids)")
    ap.add_argument("--du-scale", type=float, required=True)
    ap.add_argument("--ratio", type=float, required=True, help="Dv / Du")
    ap.add_argument("--n-fk", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=2, help="random initial states per (F, k)")
    ap.add_argument("--F-range", type=float, nargs=2, default=[0.015, 0.04])
    ap.add_argument("--delta-range", type=float, nargs=2, default=[-0.003, 0.010])
    ap.add_argument("--snaps", type=float, nargs="*", default=SNAPS)
    ap.add_argument("--out", default=str(ROOT / "results" / "band"))
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"g{args.group:02d}_Du{args.du_scale:g}_r{args.ratio:g}"
    Du, Dv = args.du_scale * DU0, args.du_scale * DU0 * args.ratio
    rng = np.random.default_rng(1000 + args.group)
    Fs = rng.uniform(*args.F_range, args.n_fk)
    ks = np.sqrt(Fs) / 2 - Fs + rng.uniform(*args.delta_range, args.n_fk)

    g = Geometry(load_frog(ROOT / "frog.mat"), cache_dir=ROOT / "results" / "cache")
    fx = FeatureExtractor(g)
    F = np.repeat(Fs, args.seeds)
    k = np.repeat(ks, args.seeds)
    ic_seed = 10_000 * (args.group + 1) + np.arange(len(F)) * 7
    U0 = np.empty((g.n, len(F)))
    V0 = np.empty((g.n, len(F)))
    for j, s in enumerate(ic_seed):
        U0[:, j], V0[:, j] = perturbed_ic(g, seed=int(s), mode="seeded", normalize_z=True)
    traj = args.group * 1000 + np.arange(len(F))

    bi = BatchIntegrator(g, Du, Dv)
    bi.start(U0, V0, F, k, ids=np.arange(len(F)))
    rows, X, t0 = [], [], time.time()

    def record(cols, status):
        for j in cols:
            c = bi.ids[j]
            f, kind = fx(bi.u[:, j], bi.v[:, j])
            X.append(f)
            rows.append((F[c], k[c], Du, Dv, ic_seed[c], traj[c], bi.t, kind, status))

    for ts in sorted(args.snaps):
        bi.advance(int(round(ts / bi.dt)) - bi.step)
        vmax = np.abs(bi.v).max(axis=0)
        dead = vmax < 1e-4
        blown = ~np.isfinite(vmax) | (vmax > 10)
        steady = (bi.rates() < 1e-7) & ~dead & ~blown
        alive = ~(dead | blown | steady)
        record(np.where(alive)[0], "running")
        record(np.where(steady)[0], "steady")
        record(np.where(dead)[0], "dead")
        bi.keep(alive)
        print(f"[{tag}] t={bi.t:5.0f} alive={bi.size:3d}/{len(F)} rows={len(rows)} ({time.time() - t0:.0f}s)",
              flush=True)
        if bi.size == 0:
            break

    cols = list(zip(*rows))
    np.savez(out / f"{tag}.npz", X=np.array(X), names=np.array(fx.names),
             F=np.array(cols[0]), k=np.array(cols[1]), Du=np.array(cols[2]), Dv=np.array(cols[3]),
             ic_seed=np.array(cols[4]), traj=np.array(cols[5]), t=np.array(cols[6]),
             kind=np.array(cols[7]), status=np.array(cols[8]))
    print(f"[{tag}] DONE {len(rows)} rows from {len(F)} trajectories ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
