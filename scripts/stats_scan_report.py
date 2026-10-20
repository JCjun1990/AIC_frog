"""Summarise the statistical trajectory scan (results/scan_stats/*): which (F, k, Du, Dv) and which
time windows reproduce the target's pattern statistics.

Writes to results/scan_stats/:
    ranking.csv       all trajectories, best statistical distance first
    summary.md        top candidates, stationary vs transient matches, parameter box for phase 2
    heatmaps.png      best distance per (F, k) for every (Du, Dv, IC) group
    features_top.png  feature-vs-time curves of the top candidates with target and tolerance bands
    top_matches.png   target and best-matching states rendered side by side
"""
from __future__ import annotations

import csv
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
from turing_rd.diagnostics import FEATURE_TOL, pattern_stats  # noqa: E402
from turing_rd.viz import render_field  # noqa: E402

BASE = ROOT / "results" / "scan_stats"
FEATS = [("wavelength", "wavelength λ"), ("n_components", "spot count"), ("area_fraction", "area fraction"),
         ("compactness", "compactness Q"), ("v_max", "v peak")]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return x


def main(top_n: int = 20):
    groups = sorted(d for d in BASE.iterdir() if (d / "stats_scan.csv").exists())
    rows = []
    for d in groups:
        for r in csv.DictReader(open(d / "stats_scan.csv")):
            r = {k: num(v) for k, v in r.items()}
            r["group"] = d.name
            rows.append(r)
    rows.sort(key=lambda r: (r["best_shape_d"], -r["match_time"]))
    with open(BASE / "ranking.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    g = Geometry(load_frog(ROOT / "frog.mat"), cache_dir=None)
    vt = sio.loadmat(str(ROOT / "Solution.mat"))["Solution"]["v"][0, 0].ravel()
    tgt = pattern_stats(g, vt)

    matched = [r for r in rows if r["best_shape_d"] <= 1]
    stationary = [r for r in rows if isinstance(r["final_shape_d"], float) and r["final_shape_d"] <= 1]
    lines = [f"# 统计匹配扫描汇总\n",
             f"目标：{tgt['type']}，{tgt['n_components']} 个，λ = {tgt['wavelength']:.3f}，面积分数 "
             f"{tgt['area_fraction']:.3f}，紧致度 {tgt['compactness']:.3f}，v 峰值 {tgt['v_max']:.3f}。",
             f"容差：λ ±{FEATURE_TOL['wavelength']:.0%}，数量 ±{FEATURE_TOL['n_components']:.0%}，面积分数 "
             f"±{FEATURE_TOL['area_fraction']}，紧致度 ±{FEATURE_TOL['compactness']}；峰值 ±{FEATURE_TOL['v_max']:.0%} 单独计。\n",
             f"共 {len(rows)} 条轨迹（{len(groups)} 组）。某段时间形态匹配（shape_d ≤ 1）：{len(matched)} 条；"
             f"终态（稳态或 t_max 时）仍匹配：{len(stationary)} 条。\n"]
    if matched:
        P = {k: np.array([r[k] for r in matched]) for k in ("F", "k", "Du", "Dv")}
        lines.append("匹配轨迹的参数范围（阶段 2 初始框）：" + "，".join(
            f"{k} ∈ [{P[k].min():.4g}, {P[k].max():.4g}]" for k in P) + "\n")
        by_group = {}
        for r in matched:
            by_group.setdefault(r["group"], []).append(r)
        lines.append("| 组 | 匹配条数 | 其中终态匹配 | 最佳 shape_d |\n| --- | --- | --- | --- |")
        for grp in sorted(by_group, key=lambda x: min(r["best_shape_d"] for r in by_group[x])):
            rs = by_group[grp]
            lines.append(f"| {grp} | {len(rs)} | {sum(1 for r in rs if r in stationary)} | "
                         f"{min(r['best_shape_d'] for r in rs):.2f} |")
        lines.append("")
    lines.append(f"| 排名 | 组 | F | k | shape_d | 最佳时刻 | 数量 | λ | 面积分数 | Q | v 峰值 | 匹配时长 | 匹配窗口 | 终态 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for i, r in enumerate(rows[:top_n]):
        win = json.loads(r["match_windows"])
        wtxt = "; ".join(f"{a:.0f}–{b:.0f}" for a, b in win[:3]) + (" …" if len(win) > 3 else "")
        lines.append(f"| {i + 1} | {r['group']} | {r['F']:.4f} | {r['k']:.4f} | {r['best_shape_d']:.2f} | "
                     f"{r['t_best']:.0f} | {r['best_n_components']:.0f} | {r['best_wavelength']:.3f} | "
                     f"{r['best_area_fraction']:.3f} | {r['best_compactness']:.2f} | {r['best_v_max']:.3f} | "
                     f"{r['match_time']:.0f} | {wtxt or '—'} | {r['status']}, {r['final_type']} "
                     f"n={r['final_n']} |")
    (BASE / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # heatmaps of best distance
    n = len(groups)
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(4.2 * cols, 3.5 * int(np.ceil(n / cols))),
                             squeeze=False)
    for ax, d in zip(axes.flat, groups):
        rs = [r for r in rows if r["group"] == d.name]
        Fs, ks = sorted({r["F"] for r in rs}), sorted({r["k"] for r in rs})
        Z = np.full((len(ks), len(Fs)), np.nan)
        for r in rs:
            Z[ks.index(r["k"]), Fs.index(r["F"])] = np.log10(np.clip(r["best_shape_d"], 0.1, 100))
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis_r", vmin=-1, vmax=2,
                       extent=[Fs[0], Fs[-1], ks[0], ks[-1]])
        for r in rs:
            if r["best_shape_d"] <= 1:
                ax.plot(r["F"], r["k"], "w+", ms=6)
        ax.set_title(d.name, fontsize=9)
        ax.set_xlabel("F", fontsize=8); ax.set_ylabel("k", fontsize=8)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.8, label="log10 best statistical distance (+ = match ≤ 1)")
    fig.savefig(BASE / "heatmaps.png", dpi=100, bbox_inches="tight")

    # match map: every trajectory whose shape statistics match at some time
    ratios = sorted({round(r["Dv"] / r["Du"], 3) for r in rows})
    fig, axes = plt.subplots(1, len(ratios), figsize=(6.2 * len(ratios), 5), squeeze=False)
    Fl = np.linspace(0.005, 0.085, 200)
    for ax, q in zip(axes[0], ratios):
        rs = [r for r in matched if round(r["Dv"] / r["Du"], 3) == q]
        for r in rs:
            peak = r["match_time_with_peak"] > 0
            ax.scatter(r["F"] + 0.0012 * (np.log(r["Du"] / 2.78e-4) / np.log(2)), r["k"],
                       s=12 + r["match_time"] / 12, marker="o" if r["ic"] == "seeded" else "s",
                       c=[plt.cm.viridis(min(r["best_shape_d"], 1))], edgecolors="k" if peak else "none",
                       linewidths=1.0, alpha=0.85)
        ax.plot(Fl, np.sqrt(Fl) / 2 - Fl, "k--", lw=1)
        ax.text(0.06, 0.0745, "F = 4(F+k)²", fontsize=8)
        ax.set_xlim(0.005, 0.085); ax.set_ylim(0.038, 0.077)
        ax.set_xlabel("F (small offset by Du: left 0.5×, centre 0.7×, right 1×)"); ax.set_ylabel("k")
        ax.set_title(f"Dv/Du = {q:g}: {len(rs)} matches, {sum(r['match_time_with_peak'] > 0 for r in rs)} with peak")
    fig.suptitle("Trajectories reproducing the target statistics (size = time in tolerance, colour = best distance, "
                 "black edge = v peak also matches, circle = seeded IC, square = suggested IC)", fontsize=9)
    fig.tight_layout(); fig.savefig(BASE / "match_map.png", dpi=110)

    # feature curves of the top candidates
    top = rows[:3]
    fig, axes = plt.subplots(len(FEATS), 1, figsize=(10, 2.1 * len(FEATS)), sharex=True)
    for ci, r in enumerate(top):
        z = np.load(BASE / r["group"] / "timeseries.npz")
        i = int(np.where((np.isclose(z["F"], r["F"])) & (np.isclose(z["k"], r["k"])))[0][0])
        for ax, (key, lab) in zip(axes, FEATS):
            ax.plot(z["t"], z[key][i], color=f"C{ci}", lw=1.2,
                    label=f"#{ci + 1} {r['group']} F={r['F']:.4f} k={r['k']:.4f}")
    for ax, (key, lab) in zip(axes, FEATS):
        tv = tgt[key]
        tol = FEATURE_TOL[key]
        lo, hi = ((tv * np.exp(-tol), tv * np.exp(tol)) if key == "wavelength" else
                  (tv * (1 - tol), tv * (1 + tol)) if key in ("n_components", "v_max") else (tv - tol, tv + tol))
        ax.axhspan(lo, hi, color="0.85", zorder=0)
        ax.axhline(tv, color="k", lw=1, ls="--")
        ax.set_ylabel(lab, fontsize=9)
    axes[0].legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("t")
    fig.suptitle("Top candidates: features vs time (dashed = target, grey = tolerance)")
    fig.tight_layout(); fig.savefig(BASE / "features_top.png", dpi=110)

    # renders
    imgs = []
    tmp = BASE / "_t.png"
    render_field(g, vt, tmp, title=f"target · n={tgt['n_components']}, λ={tgt['wavelength']:.3f}, "
                                   f"frac={tgt['area_fraction']:.2f}", vmin=0, vmax=0.5)
    imgs.append(plt.imread(tmp)); tmp.unlink()
    for ci, r in enumerate(top):
        z = np.load(BASE / r["group"] / "stats_best_states.npz")
        j = int(np.where((np.isclose(z["F"], r["F"])) & (np.isclose(z["k"], r["k"])))[0][0])
        s = pattern_stats(g, z["v"][:, j])
        render_field(g, z["v"][:, j], tmp, title=(f"#{ci + 1} {r['group']} F={r['F']:.4f} k={r['k']:.4f} t={r['t_best']:.0f}"
                                                  f" · n={s['n_components']}, λ={s['wavelength']:.3f}, "
                                                  f"frac={s['area_fraction']:.2f}, d={r['best_shape_d']:.2f}"),
                     vmin=0, vmax=0.5)
        imgs.append(plt.imread(tmp)); tmp.unlink()
    fig, axes = plt.subplots(len(imgs), 1, figsize=(12, 3.6 * len(imgs)))
    for ax, im in zip(axes, imgs):
        ax.imshow(im); ax.axis("off")
    fig.tight_layout(); fig.savefig(BASE / "top_matches.png", dpi=100)


if __name__ == "__main__":
    main()
