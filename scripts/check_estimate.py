"""Close the loop: simulate a parameter estimate with the PDE solver and compare with the target.

For each parameter set (the inversion estimate and, as controls, the organisers' spot and stripe
parameters), integrate several random initial states (baseline protocol) and every --every time
units compute (a) the statistical distance used in the scan (diagnostics.feature_distance) and (b)
the distance in the full standardised feature space used by the identifiability study. Reports the
best match per run and how it compares with the target's nearest neighbour in the band dataset.

Writes results/estimate_check/: summary.md, distance_vs_t.png, best_vs_target.png.

    python scripts/check_estimate.py --F 0.0311 --k 0.0597 --du-scale 0.80 --ratio 0.336
"""
from __future__ import annotations

import argparse
import json
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
from turing_rd.batch import BatchIntegrator  # noqa: E402
from turing_rd.diagnostics import feature_distance, pattern_stats  # noqa: E402
from turing_rd.features import FeatureExtractor  # noqa: E402
from turing_rd.initial import perturbed_ic  # noqa: E402
from turing_rd.viz import render_field  # noqa: E402

DU0 = 2.78e-4
OUT = ROOT / "results" / "estimate_check"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--F", type=float, required=True)
    ap.add_argument("--k", type=float, required=True)
    ap.add_argument("--du-scale", type=float, required=True)
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--t-max", type=float, default=800)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--no-controls", action="store_true")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    g = Geometry(load_frog(ROOT / "frog.mat"), cache_dir=ROOT / "results" / "cache")
    fx = FeatureExtractor(g)
    S = sio.loadmat(str(ROOT / "Solution.mat"))["Solution"]
    ut, vt = S["u"][0, 0].ravel(), S["v"][0, 0].ravel()
    tstats = pattern_stats(g, vt)
    ft, _ = fx(ut, vt)

    # feature scaling and the target's nearest-neighbour distance, from the band dataset
    parts = [np.load(f) for f in sorted((ROOT / "results" / "band").glob("g*.npz"))]
    X = np.concatenate([p["X"] for p in parts])
    kind = np.concatenate([p["kind"] for p in parts])
    X = X[kind == "spots"]
    med = np.nanmedian(X, axis=0)
    scale = np.nanpercentile(X, 75, axis=0) - np.nanpercentile(X, 25, axis=0)
    scale[scale == 0] = 1

    def zdist(f):
        a = (np.where(np.isnan(f), med, f) - med) / scale
        b = (np.where(np.isnan(ft), med, ft) - med) / scale
        return float(np.sqrt(((a - b) ** 2).mean()))

    Z = (np.where(np.isnan(X), med, X) - med) / scale
    zt = (np.where(np.isnan(ft), med, ft) - med) / scale
    nn_dataset = float(np.sqrt(((Z - zt) ** 2).mean(axis=1)).min())

    cases = {"estimate": (args.F, args.k, args.du_scale, args.ratio)}
    if not args.no_controls:
        cases["given spots"] = (0.028, 0.062, 1.0, 0.5)
        cases["given stripes"] = (0.054, 0.063, 1.0, 0.5)
    seeds = 50_000 + 13 * np.arange(args.seeds)
    U0 = np.empty((g.n, len(seeds)))
    V0 = np.empty((g.n, len(seeds)))
    for j, s in enumerate(seeds):
        U0[:, j], V0[:, j] = perturbed_ic(g, seed=int(s), mode="seeded", normalize_z=True)

    results, curves, best_state = {}, {}, {}
    for name, (F, k, ds, r) in cases.items():
        bi = BatchIntegrator(g, ds * DU0, ds * DU0 * r)
        B = len(seeds)
        bi.start(U0, V0, np.full(B, F), np.full(B, k))
        ts, sd, zd = [], [], []
        best = (np.inf, None, None, None)
        while bi.t < args.t_max:
            bi.advance(args.every - (bi.step % args.every) if bi.step % args.every else args.every)
            row_s, row_z = [], []
            for j in range(B):
                st = pattern_stats(g, bi.v[:, j])
                d_shape, d_peak = feature_distance(st, tstats)
                f, _ = fx(bi.u[:, j], bi.v[:, j])
                dz = zdist(f)
                row_s.append(d_shape)
                row_z.append(dz)
                if dz < best[0]:
                    best = (dz, bi.t, j, (bi.u[:, j].copy(), bi.v[:, j].copy(), st, d_shape, d_peak))
            ts.append(bi.t); sd.append(row_s); zd.append(row_z)
        sd, zd = np.array(sd), np.array(zd)
        curves[name] = (np.array(ts), sd, zd)
        frac_match = float((sd <= 1).any(axis=0).mean())
        results[name] = {"F": F, "k": k, "Du": ds * DU0, "Dv": ds * DU0 * r,
                         "best_feature_distance": best[0], "best_t": best[1], "best_seed_index": best[2],
                         "best_shape_d": best[3][3], "best_peak_d": best[3][4],
                         "best_n": best[3][2].get("n_components"), "best_wavelength": best[3][2]["wavelength"],
                         "runs_with_shape_match": frac_match,
                         "median_min_feature_distance": float(np.median(zd.min(axis=0)))}
        best_state[name] = best[3]
        print(name, json.dumps({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in results[name].items()}),
              flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for i, (name, (ts, sd, zd)) in enumerate(curves.items()):
        ax[0].plot(ts, np.median(zd, axis=1), color=f"C{i}", label=f"{name} (median of {zd.shape[1]} runs)")
        ax[0].fill_between(ts, zd.min(axis=1), np.quantile(zd, 0.75, axis=1), color=f"C{i}", alpha=0.15)
        ax[1].plot(ts, np.median(np.minimum(sd, 20), axis=1), color=f"C{i}", label=name)
    ax[0].axhline(nn_dataset, color="k", ls="--", lw=1, label="target → nearest snapshot in band dataset")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("full feature distance to target"); ax[0].legend(fontsize=8)
    ax[1].axhline(1, color="k", ls="--", lw=1)
    ax[1].set_yscale("log"); ax[1].set_xlabel("t"); ax[1].set_ylabel("scan shape distance (≤ 1 = match)")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "distance_vs_t.png", dpi=120)

    tmp = OUT / "_tmp.png"
    imgs = []
    render_field(g, vt, tmp, title=f"target · n={tstats['n_components']}, λ={tstats['wavelength']:.3f}",
                 vmin=0, vmax=0.6)
    imgs.append(plt.imread(tmp))
    for name in cases:
        u, v, st, d_shape, d_peak = best_state[name]
        r = results[name]
        render_field(g, v, tmp, vmin=0, vmax=0.6, title=(
            f"{name}: F={r['F']:.4f}, k={r['k']:.4f}, Du={r['Du']:.3g}, Dv={r['Dv']:.3g}, t={r['best_t']:.0f} · "
            f"n={st.get('n_components')}, λ={st['wavelength']:.3f} · feature dist {r['best_feature_distance']:.2f}"))
        imgs.append(plt.imread(tmp))
    tmp.unlink()
    fig, axs = plt.subplots(len(imgs), 1, figsize=(12, 3.6 * len(imgs)))
    for a, im in zip(axs, imgs):
        a.imshow(im); a.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "best_vs_target.png", dpi=100)
    np.savez(OUT / "best_estimate_state.npz", u=best_state["estimate"][0], v=best_state["estimate"][1])

    lines = ["# 反演估计的求解器校验\n",
             f"每组参数 {len(seeds)} 个随机初值，t ≤ {args.t_max:g}，每 {args.every} 个时间单位比较一次。"
             f"参照：目标到带状数据集里最近快照的特征距离 = {nn_dataset:.2f}。\n",
             "| 参数组 | F | k | Du | Dv | 最佳特征距离 | 最佳时刻 | 各初值最小距离中位数 | 形态匹配的初值比例 | 最佳形态距离 / 峰值距离 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['F']:.4f} | {r['k']:.4f} | {r['Du']:.3g} | {r['Dv']:.3g} | "
                     f"{r['best_feature_distance']:.2f} | {r['best_t']:.0f} | {r['median_min_feature_distance']:.2f} | "
                     f"{r['runs_with_shape_match']:.0%} | {r['best_shape_d']:.2f} / {r['best_peak_d']:.2f} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
