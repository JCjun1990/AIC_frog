"""Compare the converged steady states of results/steady_runs/* with Solution.mat.

Rows: steady states reached from the target itself (made admissible) and from random initial
states, for the parameter sets suggested by the steady-state hypothesis. Columns: convergence,
pattern statistics, pointwise correlation with the target, steady residual, leading eigenvalue.

Writes results/steady_runs/summary.md, from_target.png and from_random.png.
"""
from __future__ import annotations

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
from turing_rd.diagnostics import pattern_stats  # noqa: E402
from turing_rd.viz import render_field  # noqa: E402

BASE = ROOT / "results" / "steady_runs"
LABEL = {"stationarity_opt": "稳态反演最优", "stated": "题给斑点参数", "identities": "积分恒等式 + 题给 D"}


def main():
    g = Geometry(load_frog(ROOT / "frog.mat"), cache_dir=ROOT / "results" / "cache")
    S = sio.loadmat(str(ROOT / "Solution.mat"))["Solution"]
    vt = S["v"][0, 0].ravel()
    ts = pattern_stats(g, vt)

    def wcorr(a, b):
        a = a - g.mean(a); b = b - g.mean(b)
        return float((g.M @ (a * b)) / np.sqrt((g.M @ (a * a)) * (g.M @ (b * b))))

    runs = sorted(d for d in BASE.iterdir() if (d / "summary.json").exists())
    extra = [ROOT / "results" / "baseline" / "frog_spots"]          # seed 0 at the stated parameters
    rows, states = [], {}
    for d in runs + extra:
        s = json.loads((d / "summary.json").read_text())
        z = np.load(d / "final.npz")
        name = d.name if d.parent == BASE else "random0_stated"
        kind = name.split("_", 1)[1] if "_" in name else name
        start = "目标" if name.startswith("target") else f"随机初值 {name[6]}"
        st = pattern_stats(g, z["v"])
        p = s["params"]
        rows.append({"name": name, "set": LABEL.get(kind.replace("from_", ""), kind), "start": start,
                     "F": p["F"], "k": p["k"], "Du": p["Du"], "Dv": p["Dv"],
                     "t": s["time"]["t_final"], "converged": s["converged_time_integration"],
                     "residual": s["final_residual"]["steady_rel"],
                     "max_eig": (s.get("linear_stability") or {}).get("max_real", float("nan")),
                     "type": st["type"], "n": st.get("n_components", 0), "wavelength": st["wavelength"],
                     "area_fraction": st.get("area_fraction", 0), "compactness": st.get("compactness", 0),
                     "v_max": st["v_max"], "corr": wcorr(z["v"], vt)})
        states[name] = z["v"]
    rows.sort(key=lambda r: (r["start"] != "目标", r["set"], r["name"]))
    lines = ["# 稳态假设下的稳态解与目标对比\n",
             f"目标：{ts['type']}，{ts['n_components']} 个，λ = {ts['wavelength']:.3f}，面积分数 {ts['area_fraction']:.3f}，"
             f"紧致度 {ts['compactness']:.2f}，v 峰值 {ts['v_max']:.3f}。\n",
             "| 起点 | 参数组 | F | k | Du | Dv | 终止 t | 收敛 | 残差/F | max Re λ | 类型 | 数量 | λ | 面积分数 | 紧致度 | v 峰值 | 与目标逐点相关 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['start']} | {r['set']} | {r['F']:.4f} | {r['k']:.4f} | {r['Du']:.3g} | {r['Dv']:.3g} | {r['t']:.0f} | "
                     f"{'是' if r['converged'] else '否'} | {r['residual']:.1e} | {r['max_eig']:.1e} | {r['type']} | {r['n']} | "
                     f"{r['wavelength']:.3f} | {r['area_fraction']:.3f} | {r['compactness']:.2f} | {r['v_max']:.3f} | {r['corr']:.3f} |")
    (BASE / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    def sheet(names, path):
        tmp = BASE / "_tmp.png"
        imgs = []
        render_field(g, vt, tmp, title=f"target · n={ts['n_components']}, λ={ts['wavelength']:.3f}, v_max={ts['v_max']:.2f}",
                     vmin=0, vmax=0.6)
        imgs.append(plt.imread(tmp))
        for nm in names:
            r = next(x for x in rows if x["name"] == nm)
            render_field(g, states[nm], tmp, vmin=0, vmax=0.6, title=(
                f"{nm}: steady state, n={r['n']}, λ={r['wavelength']:.3f}, v_max={r['v_max']:.2f}, corr with target {r['corr']:.2f}"))
            imgs.append(plt.imread(tmp))
        tmp.unlink()
        fig, axs = plt.subplots(len(imgs), 1, figsize=(12, 3.5 * len(imgs)), squeeze=False)
        for a, im in zip(axs[:, 0], imgs):
            a.imshow(im); a.axis("off")
        fig.tight_layout(); fig.savefig(path, dpi=95)

    sheet([n for n in ("target_from_stationarity_opt", "target_from_stated", "target_from_identities") if n in states],
          BASE / "from_target.png")
    sheet([n for n in ("random0_stationarity_opt", "random0_identities", "random0_stated") if n in states],
          BASE / "from_random.png")


if __name__ == "__main__":
    main()
