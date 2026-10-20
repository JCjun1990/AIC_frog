"""Check the orientation and defect features on the four baselines and on the target.

Expected: spots -> low coherence, skeleton reduced to points (endpoint, junction and skeleton
densities ~ 0), few spot defects; stripes / maze -> high coherence, skeleton density ~ 1, free ends
and junctions counted. The target is shown raw and after the inversion's preprocessing.

Writes results/features_validation/summary.md and features.json.

    python scripts/validate_features.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from turing_rd import SPOTS, Geometry, icosphere, load_frog  # noqa: E402
from turing_rd.features import EXTENDED_GROUPS, FeatureExtractor  # noqa: E402
from turing_rd.preprocess import HeatLowPass, admissible, select_cutoff  # noqa: E402

OUT = ROOT / "results" / "features_validation"
COLS = ["type_detail", "n_spots", "wavelength", "compactness", "coherence", "axial_order", "stripe_angle_deg",
        "spot_irregular_frac", "spot_size_outlier_frac", "spot_nondisk_frac", "spot_defect_frac",
        "n_endpoints", "n_junctions", "endpoint_density", "junction_density", "skeleton_density"]
HEAD = ["类型", "数量", "波长 λ", "紧致度", "相干度", "轴向序", "条纹与体轴夹角", "不规则斑", "大小离群", "非圆盘",
        "缺陷斑合计", "端点", "分叉点", "端点密度", "分叉密度", "骨架密度"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "results" / "cache"
    geoms = {"frog": Geometry(load_frog(ROOT / "frog.mat"), cache_dir=cache),
             "sphere": Geometry(icosphere(6), cache_dir=cache)}
    fxs = {name: FeatureExtractor(g, extended=True) for name, g in geoms.items()}
    cases = []
    for surf in ("sphere", "frog"):
        for pat in ("spots", "stripes"):
            z = np.load(ROOT / "results" / "baseline" / f"{surf}_{pat}" / "final.npz")
            cases.append((f"{'球面' if surf == 'sphere' else '青蛙'} · {'斑点' if pat == 'spots' else '条纹'}基线",
                          surf, z["u"], z["v"]))
    S = sio.loadmat(str(ROOT / "Solution.mat"))["Solution"]
    ut, vt = S["u"][0, 0].ravel(), S["v"][0, 0].ravel()
    g = geoms["frog"]
    kh, _ = select_cutoff(g, ut, vt, SPOTS.Du, SPOTS.Dv)
    lp = HeatLowPass(g, kh)
    cases.append(("目标 Solution.mat（原始）", "frog", ut, vt))
    cases.append((f"目标（预处理 κ½ = {kh:g}）", "frog", *admissible(lp(ut), lp(vt))))
    zb = np.load(ROOT / "results" / "baseline" / "frog_spots" / "final.npz")
    cases.append((f"青蛙 · 斑点基线（同一预处理）", "frog", *admissible(lp(zb["u"]), lp(zb["v"]))))

    rows, out = [], {}
    for name, surf, u, v in cases:
        t0 = time.time()
        d = fxs[surf].describe(u, v)
        d["seconds"] = time.time() - t0
        out[name] = {c: d.get(c) for c in COLS + ["seconds", "n_spots", "skeleton_length"]}
        cells = []
        for c in COLS:
            x = d.get(c)
            cells.append(x if isinstance(x, str) else (f"{x:.0f}" if c in ("n_spots", "n_endpoints", "n_junctions")
                                                       else f"{x:.3g}" if c != "stripe_angle_deg" else f"{x:.0f}°"))
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
        print(name, {c: out[name][c] for c in COLS}, f"{d['seconds']:.2f}s", flush=True)
    axis = {s: np.round(fx.axis, 3).tolist() for s, fx in fxs.items()}
    lines = ["# 方向性与缺陷特征：基线校验\n",
             f"体轴：青蛙 {axis['frog']}（面积二阶矩主轴），球面 {axis['sphere']}（各向同性，取 z 轴）。"
             f"结构张量窗口 σ = {fxs['frog'].orient_sigma}，骨架剪枝长度 {fxs['frog'].prune_length}，"
             f"密度按 λ_ref = {fxs['frog'].lambda_ref} 归一化。\n",
             "| 状态 | " + " | ".join(HEAD) + " |", "| --- " * (len(HEAD) + 1) + "|", *rows, "",
             "特征分组：" + "；".join(f"{k}：{', '.join(v)}" for k, v in EXTENDED_GROUPS.items())]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    (OUT / "features.json").write_text(json.dumps({"axis": axis, "cases": out}, indent=2, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
