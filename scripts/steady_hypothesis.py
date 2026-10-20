"""Is Solution.mat a (noisy) steady state? Tests behind the organisers' answer "steady state".

1. Strong-form steady fit (PDE-FIND style) on the target after light heat smoothing of increasing
   strength: noise in the discrete Laplacian biases the diffusion coefficients towards zero.
2. Weak-form steady fit with Laplace-Beltrami eigenfunctions as test functions (no derivatives of
   the data): phi_j^T L f = -lambda_j phi_j^T M f.
3. For candidate parameter sets, start from the lightly denoised target (clipped to the physical
   range) and (a) drive it to the nearest steady state with pseudo-transient continuation,
   (b) integrate it in time. If the target is a steady state of those parameters, both stay close to
   it (pointwise correlation near 1).

Writes results/steady_hypothesis/summary.md and steady_vs_target.png.
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
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from turing_rd import Geometry, GrayScottSolver, GSParams, load_frog  # noqa: E402
from turing_rd.diagnostics import linear_stability, pattern_stats  # noqa: E402
from turing_rd.steady import ptc_polish  # noqa: E402
from turing_rd.viz import render_field  # noqa: E402

OUT = ROOT / "results" / "steady_hypothesis"
CANDIDATES = {
    "CMA-ES refined": GSParams(0.03368, 0.06083, 2.53e-4, 2.53e-4 * 0.3416),
    "given spots, Dv = Du/3": GSParams(0.028, 0.062, 2.78e-4, 2.78e-4 / 3),
    "given spots, Dv = Du/4": GSParams(0.028, 0.062, 2.78e-4, 2.78e-4 / 4),
    "weak-form fit (K=200)": GSParams(0.0301, 0.0618, 2.49e-4, 3.49e-5),
    "given spots (Dv = Du/2)": GSParams(0.028, 0.062, 2.78e-4, 1.39e-4),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g = Geometry(load_frog(ROOT / "frog.mat"), cache_dir=ROOT / "results" / "cache")
    S = sio.loadmat(str(ROOT / "Solution.mat"))["Solution"]
    ut, vt = S["u"][0, 0].ravel(), S["v"][0, 0].ravel()
    M, L = g.M, g.L

    def smooth(f, c):
        return f if c == 0 else spla.spsolve((sp.diags(M) - c * g.h_med ** 2 * L).tocsc(), M * f)

    def wcorr(a, b):
        a = a - g.mean(a); b = b - g.mean(b)
        return float((M @ (a * b)) / np.sqrt((M @ (a * a)) * (M @ (b * b))))

    lines = ["# Solution.mat 是否为（带噪声的）稳态\n"]
    # 1. strong form vs smoothing
    lines += ["## 1. 强形式稳态拟合 vs 平滑强度\n",
              "| 平滑 c（τ = c·h²） | v 的改变量 / rms(v) | Du | F | Dv | k | Dv/Du | R²(u) | R²(v) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    w = np.sqrt(M)
    for c in (0, 0.5, 1, 2, 4, 8):
        us, vs = smooth(ut, c), smooth(vt, c)
        r = us * vs * vs
        X1, y1 = np.c_[g.lap(us), 1 - us] * w[:, None], r * w
        X2, y2 = np.c_[g.lap(vs), -vs] * w[:, None], -r * w
        p1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
        p2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
        R2 = [1 - np.sum((y - X @ p) ** 2) / np.sum((y - y.mean()) ** 2) for X, y, p in ((X1, y1, p1), (X2, y2, p2))]
        lines.append(f"| {c} | {g.rms(vs - vt) / g.rms(vt - g.mean(vt)):.3f} | {p1[0]:.3g} | {p1[1]:.4f} | {p2[0]:.3g} | "
                     f"{p2[1] - p1[1]:.4f} | {p2[0] / p1[0]:.3f} | {R2[0]:.3f} | {R2[1]:.3f} |")
    # 2. weak form
    lam, phi = g.eigenbasis(600)
    P = (phi * M[:, None]).T
    cu, cv, cr, c1 = P @ ut, P @ vt, P @ (ut * vt * vt), P @ np.ones(g.n)
    lines += ["", "## 2. 弱形式稳态拟合（LB 特征函数为检验函数）\n",
              "| 检验函数数 K | Du | F | Dv | k | Dv/Du | R²(u) | R²(v) |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for K in (100, 200, 400, 600):
        s = slice(0, K)
        X1, y1 = np.c_[-lam[s] * cu[s], c1[s] - cu[s]], cr[s]
        X2, y2 = np.c_[-lam[s] * cv[s], -cv[s]], -cr[s]
        p1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
        p2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
        R2 = [1 - np.sum((y - X @ p) ** 2) / np.sum((y - y.mean()) ** 2) for X, y, p in ((X1, y1, p1), (X2, y2, p2))]
        lines.append(f"| {K} | {p1[0]:.3g} | {p1[1]:.4f} | {p2[0]:.3g} | {p2[1] - p1[1]:.4f} | {p2[0] / p1[0]:.3f} | "
                     f"{R2[0]:.3f} | {R2[1]:.3f} |")

    # 3. nearest steady state and time evolution from the denoised target
    u0 = np.clip(smooth(ut, 1), 0, 1)
    v0 = np.clip(smooth(vt, 1), 0, 1)
    ts = pattern_stats(g, vt)
    lines += ["", "## 3. 从去噪目标出发：最近的稳态与时间演化\n",
              "初值：目标经 c = 1 平滑并截断到 [0, 1]（与原目标逐点相关 "
              f"{wcorr(v0, vt):.3f}）。目标：{ts['n_components']} 个斑，λ = {ts['wavelength']:.3f}。\n",
              "| 参数组 | Newton 是否收敛 | 稳态与目标的相关 | 稳态 v 改变量 / rms(v) | 稳态斑数 / λ | max Re λ(J) | "
              "演化 t = 500 / 2000 时与目标的相关 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    states, summary = {}, {}
    for name, p in CANDIDATES.items():
        us, vs, hist = ptc_polish(g, p, u0, v0, tol=1e-10, dtau0=20.0, max_iter=80)
        conv = hist[-1]["rel_residual"] < 1e-8
        st = pattern_stats(g, vs)
        stab = linear_stability(g, p, us, vs, k=6) if conv else {"max_real": float("nan")}
        sol = GrayScottSolver(g, dt=1.0)
        corr_t = {}

        def cb(step, t, u, v, up, vp):
            if int(t) in (500, 2000):
                corr_t[int(t)] = wcorr(v.numpy(), vt)
            return False
        with torch.no_grad():
            sol.rollout(u0, v0, p, 2000, record_every=500, callback=cb)
        states[name] = vs
        summary[name] = {"converged": conv, "corr_steady": wcorr(vs, vt),
                         "change": g.rms(vs - vt) / g.rms(vt - g.mean(vt)), "n": st.get("n_components"),
                         "wavelength": st["wavelength"], "max_real": stab["max_real"], "corr_t": corr_t,
                         "iterations": len(hist) - 1, "final_residual": hist[-1]["rel_residual"]}
        r = summary[name]
        lines.append(f"| {name} | {'是' if conv else '否'}（{r['iterations']} 次，残差 {r['final_residual']:.1e}） | "
                     f"{r['corr_steady']:.3f} | {r['change']:.2f} | {r['n']} / {r['wavelength']:.3f} | "
                     f"{r['max_real']:.1e} | {corr_t.get(500, float('nan')):.3f} / {corr_t.get(2000, float('nan')):.3f} |")
        print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    tmp = OUT / "_tmp.png"
    imgs = []
    render_field(g, vt, tmp, title="target Solution.mat", vmin=0, vmax=0.6)
    imgs.append(plt.imread(tmp))
    for name in ("CMA-ES refined", "given spots, Dv = Du/3", "given spots (Dv = Du/2)"):
        r = summary[name]
        render_field(g, states[name], tmp, vmin=0, vmax=0.6, title=(
            f"nearest steady state, {name}: corr {r['corr_steady']:.3f}, n={r['n']}, λ={r['wavelength']:.3f}"))
        imgs.append(plt.imread(tmp))
    tmp.unlink()
    fig, axs = plt.subplots(len(imgs), 1, figsize=(12, 3.6 * len(imgs)))
    for a, im in zip(axs, imgs):
        a.imshow(im); a.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "steady_vs_target.png", dpi=100)


if __name__ == "__main__":
    main()
