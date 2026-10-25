"""Task 3: the generated pattern closest to the target at the inverted (F, k) (AI multistart + selection).

At the inverted parameters the Gray-Scott model is multistable: random initial states end in steady
spot patterns with 18-26 spots on the frog, and which one appears depends on the initial state. The
earlier task-3 pattern was one random seed. Following the single-snapshot likelihood of Kazarnikov et
al. (2025, arXiv:2504.02530), which scores a parameter set by its best-matching ensemble member
(f = min_k f_k: "accepted if some simulated patterns resemble the data"), and the IC-invariant pattern
features of Schnoerr & Schnoerr (2023, Mach. Learn. 112:3151, incl. the number of connected
components), the pattern reported here is the steady state of the ensemble that is closest to the
target. F and k are not changed.

    1 target layout     spot centres of the preprocessed target (low-pass of invert_target.py)
    2 AI initial states  the frog network's steady spot profile u(r), v(r) at (F, k), placed on
                        a) the target's spot centres plus extra centres in the largest gaps, a few
                           Lloyd steps (anchored)            b) evenly spaced CVT centres (seeded)
                        for spot counts from the target's count to the network's predicted count
    3 PDE               relax_polish (SBDF2 + Newton) to the full residual <= 1e-11, one-shift
                        spectral screen
    4 selection         among complete, screened-stable states: minimum feature distance J (the same
                        objective and measurement operator as invert_target.py)
    5 certificate       float64 MAT with read-back certificate; three shift-invert spectra (the
                        frog-task protocol of i03_frog_stability.py)

    python scripts/inversion_best_match.py                  66 AI initial states; ~20 min on 8 Mac processes
    python scripts/inversion_best_match.py --quick          plumbing check (4 candidates)
"""
from __future__ import annotations

import os

for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy.io as sio  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from turing_rd import SPOTS, Geometry, GSParams, ai_init, load_frog  # noqa: E402
from turing_rd.diagnostics import full_steady_residual, jacobian, linear_stability  # noqa: E402
from turing_rd.spectral_policy import (INVERSION_SCREEN_CONFIGS, evaluate_spectra,
                                       failed_spectrum)  # noqa: E402
from turing_rd.inverse import feature_distance  # noqa: E402
from turing_rd.preprocess import HeatLowPass, admissible  # noqa: E402
from turing_rd.steady import relax_polish  # noqa: E402

def _first(*paths):
    """Repository layout first, then the unpacked answer package (scripts live in code/scripts there)."""
    return next((q for q in paths if q.is_file()), paths[0])


OUT = ROOT / "results" / "inversion_best_match"
INVERSION = _first(ROOT / "results" / "inversion" / "summary.json", ROOT.parent / "task3_inversion" / "summary.json")
MODEL = _first(ROOT / "submission" / "task2_frog" / "ai_model" / "model.joblib",
               ROOT.parent / "task2_frog" / "ai_model" / "model.joblib")
PREVIOUS = _first(ROOT / "submission" / "task3_inversion" / "solution.mat", ROOT.parent / "task3_inversion" / "solution.mat")
TOL = 1e-11
SCHEDULE = ((1.0, 2000.0), (4.0, 100000.0))
REPORT_KEYS = ("wavelength", "kappa_v", "coherence", "axial_order", "spot_defect_frac", "endpoint_density",
               "junction_density", "skeleton_density", "n_spots", "spot_irregular_frac", "spot_size_outlier_frac")
W: dict = {}                                          # per-worker state


# ---------------------------------------------------------------------------------------------
# measurement (identical operator for target and candidates)
# ---------------------------------------------------------------------------------------------
def _init_worker(kh):
    from turing_rd.features import FeatureExtractor
    g = Geometry(load_frog(ROOT / "frog.mat"), cache_dir=ROOT / "results" / "cache")
    S = sio.loadmat(str(ROOT / "Solution.mat"))["Solution"]
    ut, vt = S["u"][0, 0].ravel().astype(float), S["v"][0, 0].ravel().astype(float)
    lp = HeatLowPass(g, kh)
    W.update(g=g, lp=lp, fx=FeatureExtractor(g, extended=True), ut=ut, vt=vt)
    W["up"], W["vp"] = admissible(lp(ut), lp(vt))
    W["target"] = features(ut, vt)


def features(u, v):
    d = W["fx"].describe(*admissible(W["lp"](u), W["lp"](v)))
    return {k: (x if isinstance(x, str) else float(x)) for k, x in d.items() if k not in ("spot_Q", "spot_area")}


def spectral_peak_wavelength(v, width=0.6):
    """2 pi / kappa at the maximum of the Gaussian-smoothed Laplace-Beltrami power spectrum of v
    (the usual 'dominant wavelength' of a pattern; the 'wavelength' feature is the mean-gradient one)."""
    g = W["g"]
    lam, phi = g.eigenbasis(600)
    kap = np.sqrt(np.maximum(lam, 0.0))[1:]
    c = phi.T @ (g.M * (v - g.mean(v)))
    p = (c * c)[1:]
    grid = np.linspace(2.0, kap.max(), 600)
    ring = np.array([(p * np.exp(-0.5 * ((kap - x) / width) ** 2)).sum() for x in grid])
    return float(2 * np.pi / grid[np.argmax(ring)])


def correlation(a, b):
    M = W["g"].M
    a, b = a - np.average(a, weights=M), b - np.average(b, weights=M)
    return float(np.sum(M * a * b) / np.sqrt(np.sum(M * a * a) * np.sum(M * b * b)))


def spot_centres(v):
    """Maxima of v in the connected components above the mid level (the feature extractor's threshold)."""
    import scipy.sparse as sp
    import scipy.sparse.csgraph as csg
    g = W["g"]
    hi = v > 0.5 * (np.quantile(v, 0.05) + np.quantile(v, 0.95))
    E = g.edges[hi[g.edges[:, 0]] & hi[g.edges[:, 1]]]
    A = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(g.n, g.n))
    _, lab = csg.connected_components(A, directed=False)
    comps = [np.flatnonzero((lab == c) & hi) for c in np.unique(lab[hi])]
    return np.array([i[np.argmax(v[i])] for i in comps if len(i) >= 3])


def measure(u, v):
    f = features(u, v)
    J, per = feature_distance(f, W["target"])
    vl = W["lp"](v)
    return {"J": J, "per_criterion": per, "features": f, "peak_wavelength": spectral_peak_wavelength(vl),
            "corr_target": correlation(vl, W["vp"])}


# ---------------------------------------------------------------------------------------------
# AI initial states
# ---------------------------------------------------------------------------------------------
def anchored_centres(n, iters):
    """The target's spot centres, extra centres at the farthest points from the current set, then a
    few area-weighted Lloyd steps (keeps the target's layout where it has spots)."""
    from scipy.spatial import cKDTree
    g = W["g"]
    c = list(spot_centres(W["vp"]))[:n]
    while len(c) < n:
        d, _ = cKDTree(g.V[c]).query(g.V)
        c.append(int(np.argmax(d)))
    P = g.V[np.array(c)]
    tree = cKDTree(g.V)
    for _ in range(iters):
        _, lab = cKDTree(P).query(g.V)
        w = np.bincount(lab, weights=g.M, minlength=n)
        cen = np.stack([np.bincount(lab, weights=g.M * g.V[:, j], minlength=n) for j in range(3)], 1)
        P = g.V[tree.query(cen / np.maximum(w, 1e-300)[:, None])[1]]
    return tree.query(P)[1]


def initial_state(pred, layout, n, seed):
    g = W["g"]
    if layout.startswith("anchored"):
        centres = anchored_centres(n, iters=int(layout.split("_")[1]))
    else:
        centres = ai_init.cvt_centres(g, n, seed)
    return ai_init.place_spots(g, centres, pred)


def job(args):
    """One AI initial state -> steady state, screen, measurement."""
    idx, F, k, layout, n, seed = args
    import joblib
    from ai_generation import predict
    g, p = W["g"], GSParams(F, k, SPOTS.Du, SPOTS.Dv)
    t0 = time.time()
    pred = predict(joblib.load(MODEL), F, k)
    u0, v0 = initial_state(pred, layout, n, seed)
    u, v, info = relax_polish(g, p, u0, v0, schedule=SCHEDULE, check_every=250.0, try_below=1e-4, tol=TOL)
    res = full_steady_residual(g, p, u, v)["full_joint_rms_over_F"]
    row = {"idx": idx, "layout": layout, "n_init": n, "seed": seed, "residual": float(res),
           "relax_time": info["relax_time"], "wall_s": None, "max_re_screen": None}
    reports = []
    if np.isfinite(res) and 0 <= res <= TOL:
        try:
            sample = linear_stability(g, p, u, v, k=8)
            row["max_re_screen"] = sample["max_real"]
            # The legacy scalar maximum alone is never passing evidence.
            reports.append(sample.get("spectral_report", {}))
        except Exception as exc:
            row["screen_error"] = repr(exc)[:200]
            reports.append(failed_spectrum(INVERSION_SCREEN_CONFIGS[0], repr(exc)))
    row["screen_spectra"] = reports
    row["screen_decision"] = evaluate_spectra(reports, float(res),
        configs=INVERSION_SCREEN_CONFIGS, scope="task3_single_shift_preliminary_screen", tol=TOL)
    row.update(measure(u, v))
    row["n_final"] = int(len(spot_centres(v)))
    row["wall_s"] = time.time() - t0
    row["state"] = (u, v)
    return row


# ---------------------------------------------------------------------------------------------
# certificate for the selected state
# ---------------------------------------------------------------------------------------------
def certify(g, p, u, v, seed, out):
    from i03_frog_stability import solve_config
    from turing_rd.result_io import save_certified_mat
    from turing_rd.spectrum_evidence import save_evidence
    cert = save_certified_mat(out / "solution_best_match.mat", g, p, u, v, seed=seed,
                              provenance={"script": "scripts/inversion_best_match.py",
                                          "method": "AI multistart + minimum feature distance"})
    evidence = save_evidence(out / "solution_best_match.mat", cert, g, p, out / "spectral_evidence", solve_config)
    if not evidence["decision"]["numerical_check_passed"]:
        raise RuntimeError("best-match spectral evidence did not pass; failed evidence preserved")
    evidence["evidence_file"] = "spectral_evidence/certificate.json"
    return cert, evidence


def render(g, panels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from turing_rd.viz import render_field
    tmp = path.with_name("_tmp.png")
    imgs = []
    for f, title in panels:
        render_field(g, f, tmp, title=title, vmin=0, vmax=0.5)
        imgs.append(plt.imread(tmp))
    tmp.unlink()
    fig, axs = plt.subplots(len(imgs), 1, figsize=(12, 3.6 * len(imgs)))
    for a, im in zip(axs, imgs):
        a.imshow(im)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def report_panel(run_dir):
    """Compact target-vs-best figure for the two-page report (top and back views side by side)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from turing_rd.result_io import read_mat_fields
    from turing_rd.viz import render_field
    s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    _init_worker(s["kappa_half"])
    g = W["g"]
    _, v = read_mat_fields(run_dir / "solution_best_match.mat", g.n)
    tmp = run_dir / "_tmp.png"
    imgs = []
    for f, title in ((W["vt"], f"target Solution.mat ({s['n_target']} spots)"),
                     (v, f"generated, F={s['F']:.4f} k={s['k']:.4f} ({s['best']['n_final']} spots)")):
        render_field(g, f, tmp, title=title, views=("top", "iso_back"), vmin=0, vmax=0.5, dpi=90)
        imgs.append(plt.imread(tmp))
    tmp.unlink()
    fig, axs = plt.subplots(1, 2, figsize=(16, 4.2))
    for a, im in zip(axs, imgs):
        a.imshow(im)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(run_dir / "report_panel.png", dpi=90)
    plt.close(fig)


INSTALL_FILES = {"solution_best_match.mat": "任务 3 生成斑图：反演参数下最接近目标的完整稳态（float64）",
                 "summary.json": "最接近目标稳态的选择记录、特征、MAT 证书与三组谱检查",
                 "summary.md": "目标 / 新生成 / 旧代表解的特征对照表",
                 "best_match_vs_target.png": "目标与生成斑图对比（多视角）",
                 "report_panel.png": "目标与生成斑图对比（报告用）",
                 "candidates.csv": "全部 AI 初值的收敛、稳定与特征距离"}


def install(run_dir, submission=ROOT / "submission"):
    """Copy a finished run into submission/task3_inversion/best_match/ and register the files in
    submission/PACKAGE_MANIFEST.json (so i03_package.py carries them). Never overwrites."""
    import hashlib
    import shutil

    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    dest = submission / "task3_inversion" / "best_match"
    if dest.exists():
        raise SystemExit(f"{dest} exists; nothing overwritten")
    manifest_path = submission / "PACKAGE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    known = {r["path"] for r in manifest["files"]}
    dest.mkdir(parents=True)
    files = dict(INSTALL_FILES)
    saved = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")).get("stability", {})
    if saved.get("evidence_file"):
        evidence_path = run_dir / saved["evidence_file"]
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        files[evidence_path.relative_to(run_dir).as_posix()] = "MAT-bound finite-spectrum evidence index"
        for entry in evidence["artifacts"]:
            for key in ("report_file", "vectors_file"):
                name = (evidence_path.parent / entry[key]).relative_to(run_dir).as_posix()
                files[name] = "saved eigenvectors and independent pair residuals"
    for name, reason in files.items():
        src, dst = run_dir / name, dest / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        rel = f"task3_inversion/best_match/{name}"
        if rel in known:
            raise SystemExit(f"{rel} already in the manifest")
        manifest["files"].append({"path": rel, "source": f"submission/{rel}", "source_sha256": digest(src),
                                  "sha256": digest(dst), "bytes": dst.stat().st_size, "reason": reason,
                                  "sanitized": False, "transformations": [],
                                  "historical_provenance": {"source": f"{run_dir.relative_to(ROOT).as_posix()}/{name}",
                                                            "source_sha256": digest(src)}})
    newline = "\r\n" if b"\r\n" in manifest_path.read_bytes()[:200] else "\n"     # keep the file's own line endings
    text = json.dumps(manifest, ensure_ascii=False, indent=1, allow_nan=False) + "\n"
    manifest_path.write_bytes(text.replace("\n", newline).encode("utf-8"))
    print(f"installed {len(files)} files into {dest.relative_to(ROOT)} and the package manifest")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--layouts", nargs="+", default=["anchored_0", "anchored_1", "anchored_2", "anchored_3", "anchored_4", "anchored_6", "anchored_8", "cvt_3", "cvt_4", "cvt_5", "cvt_6"])
    ap.add_argument("--quick", action="store_true", help="4 candidates, no certificate")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--panel-only", action="store_true", help="only (re)draw report_panel.png of an existing --out")
    ap.add_argument("--install", action="store_true",
                    help="copy an existing --out into submission/task3_inversion/best_match and the manifest")
    args = ap.parse_args()
    if args.panel_only:
        report_panel(args.out)
        return
    if args.install:
        install(args.out.resolve())
        return
    if args.out.exists():
        ap.error(f"{args.out} exists; nothing overwritten")
    args.out.mkdir(parents=True)
    T0 = time.time()
    inv = json.loads(INVERSION.read_text(encoding="utf-8"))
    F, k, kh = inv["estimate"]["F"], inv["estimate"]["k"], inv["preprocess"]["kappa_half"]
    _init_worker(kh)
    import joblib
    from ai_generation import predict
    pred = predict(joblib.load(MODEL), F, k)
    n_target = len(spot_centres(W["vp"]))
    n_model = int(round(pred["density"] * W["g"].area))
    counts = list(range(n_target, max(n_target, n_model) + 1))
    # the seed depends only on (n, layout name): cvt_X uses seed 100 n + X, so any layout list reproduces it
    jobs = [(i, F, k, layout, n, 100 * n + int(layout.split("_")[1])) for i, (n, layout) in
            enumerate((n, lay) for n in counts for lay in args.layouts)]
    if args.quick:
        jobs = jobs[:4]
    print(f"F={F:.5f} k={k:.5f} kappa_half={kh:g}; target {n_target} spots, network {n_model}; "
          f"{len(jobs)} AI initial states", flush=True)
    rows = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, _init_worker, (kh,)) as pool:
        for r in pool.imap_unordered(job, jobs):
            rows.append(r)
            print(f"  [{len(rows)}/{len(jobs)}] {r['layout']:10s} n {r['n_init']:2d} -> {r['n_final']:2d}  "
                  f"res {r['residual']:.1e}  maxRe {r['max_re_screen']}  J {r['J']:.3f}  "
                  f"lambda_peak {r['peak_wavelength']:.3f}  {r['wall_s']:.0f}s", flush=True)
    rows.sort(key=lambda r: r["idx"])
    from json_safe import write_json_exclusive
    write_json_exclusive(args.out / "screening_records.json", [
        {key: r[key] for key in ("idx", "layout", "n_init", "seed", "residual",
                                "max_re_screen", "screen_spectra", "screen_decision")} for r in rows])
    ok = [r for r in rows if r["screen_decision"]["numerical_check_passed"]]
    if not ok:
        raise SystemExit("no complete, screened-stable candidate")
    best = min(ok, key=lambda r: r["J"])

    fields = ["idx", "layout", "n_init", "seed", "n_final", "residual", "max_re_screen", "J", "peak_wavelength",
              "corr_target", "relax_time", "wall_s"]
    with open(args.out / "candidates.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(fields + list(REPORT_KEYS))
        for r in rows:
            wr.writerow([r[f] for f in fields] + [r["features"].get(key) for key in REPORT_KEYS])

    g, p = W["g"], GSParams(F, k, SPOTS.Du, SPOTS.Dv)
    from turing_rd.result_io import read_mat_fields
    u_prev, v_prev = read_mat_fields(PREVIOUS, g.n)
    prev = measure(u_prev, v_prev)
    prev["n_final"] = int(len(spot_centres(v_prev)))
    target = {"features": W["target"], "peak_wavelength": spectral_peak_wavelength(W["vp"]), "n_final": n_target}
    summary = {"F": F, "k": k, "Du": SPOTS.Du, "Dv": SPOTS.Dv, "kappa_half": kh, "n_target": n_target,
               "n_network": n_model, "candidates": len(rows), "complete_and_screened_stable": len(ok),
               "selection": "minimum feature distance J among complete, screened-stable steady states",
               "best": {key: best[key] for key in fields + ["per_criterion", "features"]},
               "previous_representative": {key: prev[key] for key in ("J", "per_criterion", "features",
                                                                      "peak_wavelength", "corr_target", "n_final")},
               "target": target,
               "J_distribution": {"min": min(r["J"] for r in ok), "median": float(np.median([r["J"] for r in ok])),
                                  "max": max(r["J"] for r in ok)}}
    if not args.quick:
        u, v = best["state"]
        cert, stab = certify(g, p, u, v, seed=best["seed"], out=args.out)
        summary["mat_certificate"] = cert
        summary["stability"] = stab
        render(g, [(W["vt"], "target Solution.mat (raw)"),
                   (W["vp"], f"target preprocessed (low-pass κ½ = {kh:g}, clipped), {n_target} spots"),
                   (v, f"best match F={F:.4f}, k={k:.4f}: steady state, residual {cert['residual']['value']:.0e}, "
                       f"{best['n_final']} spots, J = {best['J']:.2f}"),
                   (v_prev, f"previous representative (random seed): {prev['n_final']} spots, J = {prev['J']:.2f}")],
               args.out / "best_match_vs_target.png")
    summary["wall_seconds"] = time.time() - T0
    summary["workers"] = args.workers
    from json_safe import write_json_exclusive
    write_json_exclusive(args.out / "summary.json", summary)
    write_markdown(args.out, summary)
    if not args.quick:
        report_panel(args.out)
    print(f"best: {best['layout']} n {best['n_init']} -> {best['n_final']}, J {best['J']:.3f} "
          f"(previous {prev['J']:.3f}); total {summary['wall_seconds'] / 60:.1f} min", flush=True)


def write_markdown(out, s):
    b, pr, t = s["best"], s["previous_representative"], s["target"]
    L = ["# 任务 3：反演参数下最接近目标的生成斑图\n",
         f"F = {s['F']:.5f}，k = {s['k']:.5f}（反演结果不变），Du、Dv 取题给值。",
         f"AI 初值 {s['candidates']} 个（斑点数 {s['n_target']}–{max(s['n_target'], s['n_network'])}，"
         f"目标布局锚定与均匀布局），完全稳态且单移位谱为负的 {s['complete_and_screened_stable']} 个；"
         f"按特征距离 J 取最小（J 中位数 {s['J_distribution']['median']:.2f}，最小 {s['J_distribution']['min']:.2f}）。\n",
         "| 量 | 目标 | 最接近目标的稳态（新） | 随机初值代表解（旧） |",
         "| --- | --- | --- | --- |",
         f"| 斑点数 | {t['n_final']} | {b['n_final']} | {pr['n_final']} |",
         f"| 谱峰波长 | {t['peak_wavelength']:.3f} | {b['peak_wavelength']:.3f} | {pr['peak_wavelength']:.3f} |"]
    for key in REPORT_KEYS:
        if key == "n_spots":
            continue
        L.append(f"| {key} | {t['features'].get(key, float('nan')):.3f} | {b['features'].get(key, float('nan')):.3f} | "
                 f"{pr['features'].get(key, float('nan')):.3f} |")
    L += [f"| 特征距离 J | 0 | {b['J']:.3f} | {pr['J']:.3f} |",
          f"| 与目标的逐点相关 | 1 | {b['corr_target']:.3f} | {pr['corr_target']:.3f} |"]
    if "mat_certificate" in s:
        d = s["stability"]["decision"]
        L += ["", f"完整残差 {s['mat_certificate']['residual']['value']:.2e}（≤ 1e-11）；三组 shift-invert 谱："
                  f"{d['status']}，采样最大实部 {max(d['sampled_maxima']):.3e}。有限谱采样，不是全谱证明。"]
    L += ["", "这是固定 F/k 下的多初值搜索。AI 提供初值剖面，PDE/Newton 与候选筛选决定最终状态。",
          "没有非 AI 等预算多初值对照，不能据此证明 AI 的独立因果贡献或参数反演精度提升。"]
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
