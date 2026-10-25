"""Sphere stripes: AI proposes the two band counts next to its predicted wavenumber, the PDE decides.

On the unit sphere (L6 mesh) the stripe parameters have two steady band states that differ only in the
band count: 6 bands carry a weak positive mode (+1.67e-6, a tilt of the band axis that the mesh no longer
holds neutral), 7 bands are stable (max Re lambda -1.48e-6). In the archived multi-seed audit every one
of the 13 positive-mode seeds was the 6-band state and every stable seed the 7-band state: the random
band phase of the AI initial state picked the count by chance. Here the AI initial state uses exactly
n symmetric latitude bands for n = floor and ceil of kappa0 / 2 (kappa0 from the network), each is
driven to a complete steady state (rotation-aware Newton + orientation, full residual <= 1e-11), and
the full unprojected Jacobian spectrum is sampled near zero (shift-invert at sigma = 0 and 1e-3); the
stable one is kept. Seeds set a random band axis. Formal results and certificates are untouched.

    python scripts/sphere_bandcount.py --seeds 1 2 3 4 5 6 7 8 9 10 --workers 8     both candidates per axis
    python scripts/sphere_bandcount.py --install                                     evidence folder
"""
from __future__ import annotations

import os

for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import ai_generation as A  # noqa: E402
from turing_rd import STRIPES, GSParams, ai_init  # noqa: E402
from turing_rd.diagnostics import full_steady_residual, pattern_stats  # noqa: E402
from turing_rd.stability import solve_spectrum  # noqa: E402
from turing_rd.spectral_policy import candidate_decision, screen_candidate  # noqa: E402
from turing_rd.steady import relax_polish  # noqa: E402

OUT = ROOT / "results" / "sphere_bandcount"
MODEL = next((q for q in (ROOT / "submission/task1_sphere/ai_model/model.joblib",
                          ROOT.parent / "task1_sphere/ai_model/model.joblib") if q.exists()),
             ROOT / "submission/task1_sphere/ai_model/model.joblib")


def axis(seed):
    if seed == 0:
        return np.array([0.0, 0.0, 1.0])
    a = np.random.default_rng(seed).standard_normal(3)
    return a / np.linalg.norm(a)


def candidate_counts(kappa0):
    """The two integer band counts around the network's wavenumber (n bands ~ kappa0 / 2)."""
    return sorted({int(np.floor(kappa0 / 2)), int(np.ceil(kappa0 / 2))})


def run(surface, n, seed, t_max=200000.0):
    import joblib
    g, p = A.geom(surface), GSParams(STRIPES.F, STRIPES.k)
    rot = A.rotations(surface)
    pred = A.predict(joblib.load(MODEL), p.F, p.k)
    s = dict(pred["stats"], kappa0=2.0 * n)            # exactly n bands, symmetric about the equator
    t0 = time.time()
    u0, v0 = ai_init.bands(g, s, phase=0.0, axis=axis(seed))
    u, v, info = relax_polish(g, p, u0, v0, schedule=A.AI_SCHEDULE(t_max), check_every=500.0, tol=A.TOL,
                              rotations=rot, orient=True)
    full = full_steady_residual(g, p, u, v)["full_joint_rms_over_F"]
    st = pattern_stats(g, v)
    row = {"n_init": n, "seed": seed, "axis": axis(seed).tolist(), "kappa0_pred": pred["stats"]["kappa0"],
           "type": st.get("type"), "n_bands": st.get("n_components"), "full_residual": float(full),
           "relax_time": info["relax_time"], "wall_s": time.time() - t0, "spectra": []}
    row.update(screen_candidate(g, p, u, v, full, solve_spectrum))
    np.savez_compressed(OUT / f"state_n{n}_seed{seed}.npz", u=u, v=v)
    return row


def install():
    """Summary table + figure into submission/evidence/sphere_bandcount/ (packed by i03_package.py)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from json_safe import write_json_exclusive
    from turing_rd.viz import render_field
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("row_*.json"))]
    dest = ROOT / "submission" / "evidence" / "sphere_bandcount"
    if dest.exists():
        raise SystemExit(f"{dest} exists; nothing overwritten")
    dest.mkdir(parents=True)
    seeds = sorted({r["seed"] for r in rows})
    chosen = {}
    for sd in seeds:
        stable = [r for r in rows if r["seed"] == sd and candidate_decision(r)["numerical_check_passed"]]
        chosen[sd] = min(stable, key=lambda r: r["max_real"])["n_init"] if stable else None
    write_json_exclusive(dest / "summary.json", {"kappa0_pred": rows[0]["kappa0_pred"], "rows": rows,
                                                 "chosen_band_count": {str(k): v for k, v in chosen.items()},
                                                 "strict_reassessment": [candidate_decision(r) for r in rows]})
    L = ["# 球面条纹：AI 提出候选条带数，PDE 谱检查决定\n",
         f"网络预测 κ₀ = {rows[0]['kappa0_pred']:.2f}，候选条带数 {candidate_counts(rows[0]['kappa0_pred'])}"
         "（对称纬向条带，随机轴向由种子决定）。每个候选都由转动修正 Newton + 定向收敛到完整稳态（R ≤ 1e-11），"
         "再对完整 Jacobian 做 σ = 0、1e-3 两组 shift-invert 采样。\n",
         "| 种子（轴向） | 6 条带：残差 / 最大 Re λ | 7 条带：残差 / 最大 Re λ | 选中 |", "| --- | --- | --- | --- |"]
    for sd in seeds:
        cell = {}
        for n in (6, 7):
            r = next((x for x in rows if x["seed"] == sd and x["n_init"] == n), None)
            cell[n] = "—" if r is None else f"{r['full_residual']:.1e} / {r.get('max_real') if r.get('max_real') is None else format(r['max_real'], '+.2e')}"
        L.append(f"| {sd} | {cell[6]} | {cell[7]} | {chosen[sd] if chosen[sd] else '无'} |")
    ok = sum(v is not None for v in chosen.values())
    L += ["", f"有稳定候选的轴向：{ok}/{len(seeds)}。对照：随机相位的旧初值在 23 个种子中 9 个无正模态（归档审计）。",
          "谱检查是有限采样，不是全谱证明；正式提交的七条带 seed2 结果与证书不变。"]
    (dest / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    g = A.geom("sphere6")
    tmp = dest / "_tmp.png"
    imgs = []
    for n in (6, 7):
        z = np.load(OUT / f"state_n{n}_seed{seeds[0]}.npz")
        r = next(x for x in rows if x["seed"] == seeds[0] and x["n_init"] == n)
        render_field(g, z["v"], tmp, title=f"{n} bands (seed {seeds[0]}): max Re λ = {r['max_real']:+.2e}",
                     vmin=0, vmax=0.4, dpi=90)
        imgs.append(plt.imread(tmp))
    tmp.unlink()
    fig, axs = plt.subplots(2, 1, figsize=(10, 7))
    for a, im in zip(axs, imgs):
        a.imshow(im)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(dest / "bands_6_vs_7.png", dpi=90)
    plt.close(fig)
    print(f"installed into {dest.relative_to(ROOT)}; stable candidate for {ok}/{len(seeds)} axes")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--install", action="store_true", help="write the evidence folder from finished rows")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--counts", type=int, nargs="*", default=None, help="band counts (default: around kappa0/2)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--surface", default="sphere6")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.install:
        install()
        return
    import joblib
    kappa0 = A.predict(joblib.load(MODEL), STRIPES.F, STRIPES.k)["stats"]["kappa0"]
    counts = args.counts or candidate_counts(kappa0)
    print(f"network kappa0 = {kappa0:.3f} -> band counts {candidate_counts(kappa0)}; running {counts} x seeds {args.seeds}",
          flush=True)
    A.geom(args.surface).eigenbasis(600)
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(args.surface, n, sd) for sd in args.seeds for n in counts
            if not (OUT / f"row_n{n}_seed{sd}.json").exists()]
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run, *j): j for j in jobs}
        for f in futs:
            r = f.result()
            (OUT / f"row_n{r['n_init']}_seed{r['seed']}.json").write_text(json.dumps(r, indent=1))
            print(f"n={r['n_init']} seed={r['seed']}: {r['type']} {r['n_bands']} bands, residual {r['full_residual']:.1e}, "
                  f"max Re {r.get('max_real')}, stable {r['stable']} ({r['wall_s']:.0f}s)", flush=True)
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("row_*.json"))]
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).append(r)
    chosen = {sd: next((r["n_init"] for r in sorted(rs, key=lambda r: r["n_init"], reverse=True) if candidate_decision(r)["numerical_check_passed"]), None)
              for sd, rs in by_seed.items()}
    (OUT / "summary.json").write_text(json.dumps({"kappa0_pred": kappa0, "rows": rows, "chosen_band_count": chosen,
                                                 "strict_reassessment": [candidate_decision(r) for r in rows]},
                                                 indent=1))
    print("stable per (n, seed):", {(r["n_init"], r["seed"]): r["stable"] for r in rows}, flush=True)


if __name__ == "__main__":
    main()
