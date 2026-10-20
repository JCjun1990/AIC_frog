"""Identifiability study: how well do snapshot features pin down (F, k, Du, Dv, t)?

Uses the band dataset (results/band/*.npz, scripts/band_dataset.py). Only spot-type snapshots are
used, since the target is a spot pattern. For each feature subset, gradient-boosted trees predict each
parameter; scores are 5-fold cross-validated with folds grouped by trajectory, so snapshots of one
trajectory never sit on both sides. Also: a learning curve (does more data help?), permutation
importances, and an estimate for Solution.mat (model prediction with CV error, nearest neighbours in
feature space, and a check that the target lies inside the data distribution).

Writes results/identifiability/: summary.md, r2.csv and figures.
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
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from turing_rd import Geometry, load_frog  # noqa: E402
from turing_rd.features import GROUPS, FeatureExtractor  # noqa: E402

DU0 = 2.78e-4
OUT = ROOT / "results" / "identifiability"
SUBSETS = {
    "scan 5 stats": ["wavelength", "n_spots", "area_fraction", "compactness", "v_max"],
    "+ morphology": GROUPS["morphology"] + ["v_max"],
    "+ spot sizes": GROUPS["morphology"] + ["v_max"] + GROUPS["spot_sizes"],
    "+ amplitude": GROUPS["morphology"] + GROUPS["spot_sizes"] + GROUPS["amplitude"],
    "all (+ spectrum)": GROUPS["morphology"] + GROUPS["spot_sizes"] + GROUPS["amplitude"] + GROUPS["spectrum"],
}
TARGETS = {
    "F": lambda d: d["F"],
    "k": lambda d: d["k"],
    "delta (k above saddle-node)": lambda d: d["k"] - (np.sqrt(d["F"]) / 2 - d["F"]),
    "log2 Du/Du0": lambda d: np.log2(d["Du"] / DU0),
    "Dv/Du": lambda d: d["Dv"] / d["Du"],
    "log10 t": lambda d: np.log10(d["t"]),
}


def load():
    files = sorted((ROOT / "results" / "band").glob("g*.npz"))
    parts = [np.load(f, allow_pickle=False) for f in files]
    names = list(parts[0]["names"])
    d = {k: np.concatenate([p[k] for p in parts]) for k in ("X", "F", "k", "Du", "Dv", "t", "traj", "kind", "status")}
    return d, names, len(files)


QUICK = "--quick" in sys.argv          # small, fast run for debugging the pipeline


def model():
    return HistGradientBoostingRegressor(max_iter=40 if QUICK else 300, learning_rate=0.06, max_leaf_nodes=31,
                                         random_state=0)


def main():
    global SUBSETS
    if QUICK:
        SUBSETS = {k: SUBSETS[k] for k in ("scan 5 stats", "all (+ spectrum)")}
    OUT.mkdir(parents=True, exist_ok=True)
    d, names, n_files = load()
    keep = (d["kind"] == "spots") & (d["status"] == "running")
    d = {k: v[keep] for k, v in d.items()}
    X_all, groups = d["X"], d["traj"]
    col = {n: i for i, n in enumerate(names)}
    Y = {t: f(d) for t, f in TARGETS.items()}
    n_traj = len(np.unique(groups))
    print(f"{n_files} groups, {len(groups)} spot snapshots from {n_traj} trajectories", flush=True)
    gkf = GroupKFold(n_splits=3 if QUICK else 5)
    folds = list(gkf.split(X_all, groups=groups))

    # cross-validated R2 per subset and parameter
    r2, rmse, cv_pred = {}, {}, {}
    for sname, feats in SUBSETS.items():
        X = X_all[:, [col[f] for f in feats]]
        for tname, y in Y.items():
            pred = np.empty_like(y)
            for tr, te in folds:
                pred[te] = model().fit(X[tr], y[tr]).predict(X[te])
            r2[sname, tname] = r2_score(y, pred)
            rmse[sname, tname] = float(np.sqrt(np.mean((pred - y) ** 2)))
            if sname == "all (+ spectrum)":
                cv_pred[tname] = pred
        print(f"  {sname:18s} " + "  ".join(f"{t.split(' ')[0]}={r2[sname, t]:.2f}" for t in Y), flush=True)
    with open(OUT / "r2.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subset"] + list(Y) + [f"rmse {t}" for t in Y])
        for s in SUBSETS:
            w.writerow([s] + [f"{r2[s, t]:.3f}" for t in Y] + [f"{rmse[s, t]:.4g}" for t in Y])

    fig, ax = plt.subplots(figsize=(10, 4.2))
    width = 0.8 / len(SUBSETS)
    for i, s in enumerate(SUBSETS):
        ax.bar(np.arange(len(Y)) + i * width, [max(r2[s, t], 0) for t in Y], width, label=s)
    ax.set_xticks(np.arange(len(Y)) + 0.4 - width / 2)
    ax.set_xticklabels(list(Y), fontsize=8)
    ax.set_ylabel("cross-validated R² (grouped by trajectory)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, ncol=3)
    ax.set_title("How well single-snapshot features identify each parameter")
    fig.tight_layout(); fig.savefig(OUT / "r2_by_subset.png", dpi=120)

    # learning curve with all features
    feats = SUBSETS["all (+ spectrum)"]
    X = X_all[:, [col[f] for f in feats]]
    rng = np.random.default_rng(0)
    fracs = [0.3, 1.0] if QUICK else [0.15, 0.3, 0.5, 0.75, 1.0]
    lc = {t: [] for t in ("F", "k", "Dv/Du", "log2 Du/Du0")}
    for fr in fracs:
        scores = {t: [] for t in lc}
        for tr, te in folds:
            trajs = np.unique(groups[tr])
            sub = rng.choice(trajs, max(5, int(fr * len(trajs))), replace=False)
            m = np.isin(groups, sub) & np.isin(np.arange(len(groups)), tr)
            for t in lc:
                scores[t].append(r2_score(Y[t][te], model().fit(X[m], Y[t][m]).predict(X[te])))
        for t in lc:
            lc[t].append(float(np.mean(scores[t])))
    n_train = [int(fr * n_traj * (1 - 1 / len(folds))) for fr in fracs]
    fig, ax = plt.subplots(figsize=(6, 4))
    for t, v in lc.items():
        ax.plot(n_train, v, "o-", label=t)
    ax.set_xscale("log"); ax.set_xlabel("training trajectories"); ax.set_ylabel("CV R²")
    ax.set_ylim(0, 1); ax.legend(); ax.set_title("Learning curve (all features)")
    fig.tight_layout(); fig.savefig(OUT / "learning_curve.png", dpi=120)

    # permutation importance on one held-out fold
    tr, te = folds[0]
    imp = {}
    for t in ("F", "k", "Dv/Du", "log10 t"):
        m = model().fit(X[tr], Y[t][tr])
        pi = permutation_importance(m, X[te], Y[t][te], n_repeats=2 if QUICK else 5, random_state=0, scoring="r2")
        imp[t] = pi.importances_mean
    order = np.argsort(-sum(imp.values()))[:14]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, t in enumerate(imp):
        ax.barh(np.arange(len(order)) + i * 0.2, imp[t][order], 0.2, label=t)
    ax.set_yticks(np.arange(len(order)) + 0.3)
    ax.set_yticklabels([feats[j] for j in order], fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("permutation importance (drop in R²)")
    ax.legend(fontsize=8); ax.set_title("Most informative features")
    fig.tight_layout(); fig.savefig(OUT / "importance.png", dpi=120)

    # the target
    g = Geometry(load_frog(ROOT / "frog.mat"), cache_dir=ROOT / "results" / "cache")
    fx = FeatureExtractor(g)
    S = sio.loadmat(str(ROOT / "Solution.mat"))["Solution"]
    ft, kind = fx(S["u"][0, 0].ravel(), S["v"][0, 0].ravel())
    xt = ft[[names.index(f) for f in feats]]
    est = {}
    for t, y in Y.items():
        preds = [model().fit(X[tr], y[tr]).predict(xt[None])[0] for tr, _ in folds]
        est[t] = (float(np.mean(preds)), float(np.std(preds)), rmse["all (+ spectrum)", t])
    # nearest neighbours in standardised feature space
    med = np.nanmedian(X, axis=0)
    scale = np.nanpercentile(X, 75, axis=0) - np.nanpercentile(X, 25, axis=0)
    scale[scale == 0] = 1
    Z = (np.where(np.isnan(X), med, X) - med) / scale
    zt = (np.where(np.isnan(xt), med, xt) - med) / scale
    dist = np.sqrt(((Z - zt) ** 2).mean(axis=1))
    nn = np.argsort(dist)[:30]
    # typical nearest-neighbour distance between dataset snapshots of different trajectories
    probe = rng.choice(len(Z), 300, replace=False)
    nnd = []
    for i in probe:
        di = np.sqrt(((Z - Z[i]) ** 2).mean(axis=1))
        di[groups == groups[i]] = np.inf
        nnd.append(di.min())
    nnd = np.array(nnd)
    pct = float((nnd < dist[nn[0]]).mean())

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    Fl = np.linspace(0.012, 0.042, 100)
    sc = ax[0].scatter(d["F"][nn], d["k"][nn], c=(d["Dv"] / d["Du"])[nn], s=40, cmap="plasma", vmin=0.15, vmax=0.5)
    ax[0].plot(Fl, np.sqrt(Fl) / 2 - Fl, "k--", lw=1, label="saddle-node F = 4(F+k)²")
    ax[0].errorbar(est["F"][0], est["k"][0], xerr=est["F"][2], yerr=est["k"][2], fmt="*", ms=14, color="C2",
                   label="model estimate ± CV RMSE")
    fig.colorbar(sc, ax=ax[0], label="Dv/Du of neighbour")
    ax[0].set_xlabel("F"); ax[0].set_ylabel("k"); ax[0].legend(fontsize=8)
    ax[0].set_title("30 nearest snapshots to the target in feature space")
    ax[1].hist(nnd, bins=30, color="0.7", label="snapshot → nearest other trajectory")
    ax[1].axvline(dist[nn[0]], color="C3", lw=2, label="target → nearest snapshot")
    ax[1].set_xlabel("feature-space distance"); ax[1].legend(fontsize=8)
    ax[1].set_title("Is the target inside the data distribution?")
    fig.tight_layout(); fig.savefig(OUT / "target_estimate.png", dpi=120)

    lines = ["# 单张快照的参数可辨识性\n",
             f"数据：{n_files} 组 (Du, Dv)，{n_traj} 条轨迹（各自随机初值），{len(groups)} 张斑点类快照。"
             f"{len(folds)} 折交叉验证按轨迹分组。\n",
             "## 交叉验证 R²（1 = 完全确定，0 = 与猜均值无异）\n",
             "| 特征集 | " + " | ".join(Y) + " |", "| --- |" + " --- |" * len(Y)]
    for s in SUBSETS:
        lines.append(f"| {s} | " + " | ".join(f"{r2[s, t]:.2f}" for t in Y) + " |")
    lines += ["", "## 学习曲线（全部特征）\n", "| 训练轨迹数 | " + " | ".join(lc) + " |", "| --- |" + " --- |" * len(lc)]
    for i, n in enumerate(n_train):
        lines.append(f"| {n} | " + " | ".join(f"{lc[t][i]:.2f}" for t in lc) + " |")
    lines += ["", "## 目标 Solution.mat 的估计（全部特征）\n",
              f"目标类型：{kind}。到最近快照的特征距离位于数据集内部最近邻距离的第 {100 * pct:.0f} 百分位"
              "（越高越像分布外）。\n",
              "| 参数 | 模型估计 | 折间标准差 | 交叉验证 RMSE | 30 个最近邻的 10–90% 区间 |", "| --- | --- | --- | --- | --- |"]
    for t, y in Y.items():
        q = np.quantile(y[nn], [0.1, 0.9])
        lines.append(f"| {t} | {est[t][0]:.4g} | {est[t][1]:.2g} | {est[t][2]:.2g} | [{q[0]:.4g}, {q[1]:.4g}] |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
