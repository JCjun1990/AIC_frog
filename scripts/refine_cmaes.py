"""Local refinement of the inversion estimate with CMA-ES, the PDE solver in the loop.

Objective for theta = (F, k, log2 Du/Du0, Dv/Du):
    J(theta) = median over random initial states of  min_{t in window} d(features(t), target)
where d is the RMS distance in the standardised 37-feature space of the identifiability study
(robust scaling from the band dataset) and the snapshot time t is profiled out. The same training
initial states are reused for every candidate (common random numbers), so the landscape is smooth
enough for CMA-ES. Afterwards:
    validation   the optimum, the regression estimate and the organisers' parameters are re-run
                 with fresh initial states
    uncertainty  candidates sampled around the optimum are re-scored on an independent seed set;
                 those whose J is within 2 bootstrap standard errors of the optimum's are accepted,
                 and their spread is the reported uncertainty

Writes results/refine/: evals.csv, summary.md, final.json, convergence.png, uncertainty.png,
best_vs_target.png.

    python scripts/refine_cmaes.py --F 0.0311 --k 0.0597 --log2du -0.330 --ratio 0.336
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DU0 = 2.78e-4
OUT = ROOT / "results" / "refine"
NAMES = ["F", "k", "log2 Du/Du0", "Dv/Du"]
LO = np.array([0.010, 0.040, -1.5, 0.10])
HI = np.array([0.060, 0.075, 0.5, 0.60])
W = {}                                   # per-worker state


def _init_worker(t_lo, t_hi, every):
    import scipy.io as sio
    from turing_rd import Geometry, load_frog
    from turing_rd.features import FeatureExtractor
    g = Geometry(load_frog(ROOT / "frog.mat"), cache_dir=ROOT / "results" / "cache")
    fx = FeatureExtractor(g)
    S = sio.loadmat(str(ROOT / "Solution.mat"))["Solution"]
    ft, _ = fx(S["u"][0, 0].ravel(), S["v"][0, 0].ravel())
    parts = [np.load(f) for f in sorted((ROOT / "results" / "band").glob("g*.npz"))]
    X = np.concatenate([p["X"] for p in parts])[np.concatenate([p["kind"] for p in parts]) == "spots"]
    med = np.nanmedian(X, axis=0)
    scale = np.nanpercentile(X, 75, axis=0) - np.nanpercentile(X, 25, axis=0)
    scale[scale == 0] = 1
    W.update(g=g, fx=fx, zt=(np.where(np.isnan(ft), med, ft) - med) / scale, med=med, scale=scale,
             t_lo=t_lo, t_hi=t_hi, every=every, ics={})


def _ic(seed):
    from turing_rd.initial import perturbed_ic
    if seed not in W["ics"]:
        W["ics"][seed] = perturbed_ic(W["g"], seed=int(seed), mode="seeded", normalize_z=True)
    return W["ics"][seed]


def evaluate(args):
    """theta (physical), seeds -> per-seed min distance, time of the min, and the best state."""
    from turing_rd.batch import BatchIntegrator
    theta, seeds, keep_state = args
    F, k, l2, r = theta
    g, fx = W["g"], W["fx"]
    Du = DU0 * 2.0 ** l2
    U0 = np.stack([_ic(s)[0] for s in seeds], 1)
    V0 = np.stack([_ic(s)[1] for s in seeds], 1)
    bi = BatchIntegrator(g, Du, Du * r)
    B = len(seeds)
    bi.start(U0, V0, np.full(B, F), np.full(B, k))
    best_d = np.full(B, np.inf)
    best_t = np.zeros(B)
    state = None
    while bi.t < W["t_hi"]:
        bi.advance(W["every"])
        if bi.t < W["t_lo"]:
            continue
        for j in range(B):
            f, _ = fx(bi.u[:, j], bi.v[:, j])
            z = (np.where(np.isnan(f), W["med"], f) - W["med"]) / W["scale"]
            d = float(np.sqrt(((z - W["zt"]) ** 2).mean()))
            if d < best_d[j]:
                best_d[j], best_t[j] = d, bi.t
                if keep_state and d <= best_d.min():
                    state = (bi.u[:, j].copy(), bi.v[:, j].copy(), bi.t, int(seeds[j]))
    return best_d, best_t, state


def to_phys(x, x0, s):
    return np.clip(x0 + x * s, LO, HI)


def boot_se(d, n=2000, rng=np.random.default_rng(0)):
    return float(np.std([np.median(rng.choice(d, len(d))) for _ in range(n)]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--F", type=float, required=True)
    ap.add_argument("--k", type=float, required=True)
    ap.add_argument("--log2du", type=float, required=True, help="log2(Du / 2.78e-4)")
    ap.add_argument("--ratio", type=float, required=True, help="Dv / Du")
    ap.add_argument("--scale", type=float, nargs=4, default=[0.0016, 0.0014, 0.096, 0.018],
                    help="search scale per parameter (CV RMSE of the regression)")
    ap.add_argument("--sigma0", type=float, default=1.5)
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--maxiter", type=int, default=25)
    ap.add_argument("--train-seeds", type=int, default=8)
    ap.add_argument("--val-seeds", type=int, default=32)
    ap.add_argument("--unc-samples", type=int, default=64)
    ap.add_argument("--unc-seeds", type=int, default=16)
    ap.add_argument("--t-window", type=float, nargs=2, default=[20, 400])
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    import cma

    OUT.mkdir(parents=True, exist_ok=True)
    x0 = np.array([args.F, args.k, args.log2du, args.ratio])
    s = np.array(args.scale)
    train = list(70_000 + 11 * np.arange(args.train_seeds))
    val = list(80_000 + 11 * np.arange(args.val_seeds))
    unc = list(90_000 + 11 * np.arange(args.unc_seeds))
    pool = mp.get_context("spawn").Pool(args.workers, _init_worker, (*args.t_window, args.every))
    t0 = time.time()

    es = cma.CMAEvolutionStrategy(np.zeros(4), args.sigma0, {
        "popsize": args.popsize, "maxiter": args.maxiter, "seed": 1, "verbose": -9,
        "bounds": [list((LO - x0) / s), list((HI - x0) / s)]})
    evals, hist = [], []
    gen = 0
    while not es.stop():
        X = es.ask()
        res = pool.map(evaluate, [(to_phys(x, x0, s), train, False) for x in X])
        J = [float(np.median(r[0])) for r in res]
        es.tell(X, J)
        gen += 1
        for x, j, r in zip(X, J, res):
            evals.append({"gen": gen, **dict(zip(NAMES, to_phys(x, x0, s))), "J": j,
                          "J_mean": float(np.mean(r[0])), "t_median": float(np.median(r[1]))})
        xb = to_phys(es.result.xbest, x0, s)
        xm = to_phys(es.result.xfavorite, x0, s)
        hist.append({"gen": gen, "J_best": float(es.result.fbest), "J_gen_min": min(J), "sigma": float(es.sigma),
                     **{f"mean {n}": v for n, v in zip(NAMES, xm)}})
        print(f"gen {gen:2d}  J_gen_min={min(J):.4f}  J_best={es.result.fbest:.4f}  sigma={es.sigma:.2f}  "
              f"best F={xb[0]:.4f} k={xb[1]:.4f} Du={DU0 * 2 ** xb[2]:.3g} Dv/Du={xb[3]:.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    with open(OUT / "evals.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(evals[0]))
        w.writeheader()
        w.writerows(evals)

    # validation with fresh initial states
    theta_best = to_phys(es.result.xbest, x0, s)
    theta_mean = to_phys(es.result.xfavorite, x0, s)
    cands = {"CMA-ES best": theta_best, "CMA-ES mean": theta_mean, "regression estimate": x0,
             "given spots": np.array([0.028, 0.062, 0.0, 0.5])}
    val_res = dict(zip(cands, pool.map(evaluate, [(th, val, name == "CMA-ES mean")
                                                  for name, th in cands.items()])))
    print("validation (fresh seeds):", {n: round(float(np.median(r[0])), 4) for n, r in val_res.items()}, flush=True)
    final_name = min(("CMA-ES best", "CMA-ES mean"), key=lambda n: np.median(val_res[n][0]))
    theta_opt = cands[final_name]

    # uncertainty: sample around the optimum, re-score on an independent seed set
    rng = np.random.default_rng(7)
    C = es.sm.C if hasattr(es, "sm") else np.eye(4)
    L = np.linalg.cholesky(C + 1e-12 * np.eye(4))
    spread = max(es.sigma, 0.5) * 2.0
    xs = [np.zeros(4)] + [((theta_opt - x0) / s) + spread * L @ rng.standard_normal(4)
                          for _ in range(args.unc_samples)]
    thetas = [to_phys(((theta_opt - x0) / s) if i == 0 else x, x0, s) for i, x in enumerate(xs)]
    thetas[0] = theta_opt
    ures = pool.map(evaluate, [(th, unc, False) for th in thetas])
    pool.close()
    Ju = np.array([np.median(r[0]) for r in ures])
    se = boot_se(ures[0][0])
    accept = Ju <= Ju[0] + 2 * se
    acc = np.array(thetas)[accept]
    q = {n: (float(np.quantile(acc[:, i], 0.1)), float(np.quantile(acc[:, i], 0.9)),
             float(acc[:, i].min()), float(acc[:, i].max())) for i, n in enumerate(NAMES)}
    t_opt = val_res[final_name][1]
    final = {
        "theta": dict(zip(NAMES, map(float, theta_opt))), "Du": float(DU0 * 2 ** theta_opt[2]),
        "Dv": float(DU0 * 2 ** theta_opt[2] * theta_opt[3]), "chosen": final_name,
        "J_validation": {n: float(np.median(r[0])) for n, r in val_res.items()},
        "J_validation_se": {n: boot_se(r[0]) for n, r in val_res.items()},
        "t_snapshot_median": float(np.median(t_opt)), "t_snapshot_iqr": [float(np.quantile(t_opt, 0.25)),
                                                                          float(np.quantile(t_opt, 0.75))],
        "uncertainty_accept_threshold": float(Ju[0] + 2 * se), "accepted": int(accept.sum()),
        "sampled": len(thetas), "ranges_10_90_min_max": q, "cma_generations": gen,
        "wall_seconds": time.time() - t0}
    (OUT / "final.json").write_text(json.dumps(final, indent=2))
    print(json.dumps(final, indent=2), flush=True)
    np.savez(OUT / "uncertainty_samples.npz", theta=np.array(thetas), J=Ju, accept=accept)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from turing_rd.viz import render_field
    import scipy.io as sio

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    gens = [h["gen"] for h in hist]
    ax[0].plot(gens, [h["J_best"] for h in hist], "o-", label="best so far")
    ax[0].plot(gens, [h["J_gen_min"] for h in hist], ".--", label="best in generation")
    ax[0].axhline(final["J_validation"]["regression estimate"], color="C3", ls=":", label="regression estimate (val)")
    ax[0].set_xlabel("generation"); ax[0].set_ylabel("J (median min feature distance)"); ax[0].legend(fontsize=8)
    for i, n in enumerate(NAMES):
        v = np.array([h[f"mean {n}"] for h in hist])
        ax[1].plot(gens, (v - x0[i]) / s[i], label=n)
    ax[1].set_xlabel("generation"); ax[1].set_ylabel("CMA mean, in regression-RMSE units from start")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "convergence.png", dpi=120)

    T = np.array(thetas)
    pairs = [(0, 1), (2, 3), (0, 3), (1, 2)]
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.8))
    for a, (i, j) in zip(axs, pairs):
        sc = a.scatter(T[:, i], T[:, j], c=Ju, cmap="viridis_r", s=22, edgecolors=np.where(accept, "k", "none"))
        a.plot(theta_opt[i], theta_opt[j], "r*", ms=14)
        a.plot(x0[i], x0[j], "wx", ms=9, mew=2)
        a.set_xlabel(NAMES[i]); a.set_ylabel(NAMES[j])
    fig.colorbar(sc, ax=axs, label="J on independent seeds (black edge = accepted)")
    fig.suptitle("Uncertainty: candidates around the optimum (red star = optimum, white x = regression estimate)")
    fig.savefig(OUT / "uncertainty.png", dpi=110, bbox_inches="tight")

    _init_worker(*args.t_window, args.every)
    g = W["g"]
    st = val_res["CMA-ES mean"][2]
    vt = sio.loadmat(str(ROOT / "Solution.mat"))["Solution"]["v"][0, 0].ravel()
    tmp = OUT / "_tmp.png"
    imgs = []
    render_field(g, vt, tmp, title="target Solution.mat", vmin=0, vmax=0.6)
    imgs.append(plt.imread(tmp))
    if st is not None:
        render_field(g, st[1], tmp, vmin=0, vmax=0.6, title=(
            f"refined (CMA-ES mean): F={theta_mean[0]:.4f}, k={theta_mean[1]:.4f}, Du={DU0 * 2 ** theta_mean[2]:.3g}, "
            f"Dv/Du={theta_mean[3]:.3f}, t={st[2]:.0f} (validation seed {st[3]})"))
        imgs.append(plt.imread(tmp))
    tmp.unlink()
    fig, axs = plt.subplots(len(imgs), 1, figsize=(12, 3.6 * len(imgs)))
    for a, im in zip(np.atleast_1d(axs), imgs):
        a.imshow(im); a.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "best_vs_target.png", dpi=100)
    if st is not None:
        np.savez(OUT / "refined_state.npz", u=st[0], v=st[1], t=st[2], seed=st[3], theta=theta_mean)

    lines = ["# CMA-ES 局部精修结果\n",
             f"目标函数：{args.train_seeds} 个固定随机初值上「窗口 t ∈ [{args.t_window[0]:g}, {args.t_window[1]:g}] 内最小特征距离」的中位数；"
             f"{gen} 代 × {args.popsize} 个候选，用时 {final['wall_seconds'] / 60:.0f} 分钟。\n",
             "## 验证（{0} 个新随机初值）\n".format(args.val_seeds),
             "| 参数组 | F | k | Du | Dv/Du | J（中位数 ± 自助标准误） |", "| --- | --- | --- | --- | --- | --- |"]
    for n, th in cands.items():
        lines.append(f"| {n} | {th[0]:.4f} | {th[1]:.4f} | {DU0 * 2 ** th[2]:.3g} | {th[3]:.3f} | "
                     f"{final['J_validation'][n]:.3f} ± {final['J_validation_se'][n]:.3f} |")
    lines += ["", f"## 最终参数（{final_name}）与不确定性\n",
              f"在独立的 {args.unc_seeds} 个初值上，周围 {len(thetas) - 1} 个候选中有 {int(accept.sum()) - 1} 个与最优解统计上不可区分"
              f"（J ≤ 最优 + 2 个自助标准误 = {final['uncertainty_accept_threshold']:.3f}）。\n",
              "| 参数 | 最终值 | 不可区分集合 10–90% | 最小–最大 |", "| --- | --- | --- | --- |"]
    for i, n in enumerate(NAMES):
        lo, hi, mn, mx = q[n]
        lines.append(f"| {n} | {theta_opt[i]:.4g} | [{lo:.4g}, {hi:.4g}] | [{mn:.4g}, {mx:.4g}] |")
    lines.append(f"| 快照时刻 t | {final['t_snapshot_median']:.0f} | 四分位 [{final['t_snapshot_iqr'][0]:.0f}, "
                 f"{final['t_snapshot_iqr'][1]:.0f}] | — |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
