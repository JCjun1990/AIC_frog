"""AI-PDE generation for tasks 1-2: a small neural network gives the initial state, Newton makes it exact.

    python scripts/ai_generation.py all            dataset -> train -> evaluate -> generate (one click)
    python scripts/ai_generation.py dataset        self-generated steady states on an (F, k) grid
    python scripts/ai_generation.py train          networks + leave-one-point-out generalization error
    python scripts/ai_generation.py evaluate       AI + Newton vs ordinary IC + time integration
    python scripts/ai_generation.py generate       task results (given spot / stripe parameters)

Pipeline (turing_rd/ai_init.py, turing_rd/steady.py):
  1. data   For each surface (frog; unit sphere, icosphere L5) and each point of a 9 x 5 grid in
            F in [0.020, 0.060], k in [0.057, 0.065] (Du, Dv as given), run the ordinary pipeline
            (seeded IC, SBDF2, Newton polish) and record the pattern type and statistics.
  2. AI     Ensembles of small MLPs (scikit-learn, CPU) learn (F, k) -> pattern type, and
            (F, k) -> spot density + radial spot profile u(r), v(r), or -> spectral ring + amplitude
            quantiles + u-v relation for non-spot patterns.
  3. guess  Spots: N = density x area centres placed by Lloyd (centroidal Voronoi) iterations, each
            carrying the predicted profile. Stripes: nodal-line maze with the predicted statistics;
            on the sphere latitude bands with the predicted period and amplitudes instead (a maze
            there keeps rearranging for > 10^5 time units, the bands settle).
  4. PDE    The same relax + Newton routine as the ordinary pipeline (steady.relax_polish) drives the
            guess to a steady state with relative residual <= 1e-11; linear stability is checked.
            On the sphere the Newton step is rotation-aware (steady.rotation_newton: relative equilibrium,
            residual <= 1e-11 after removing the rigid rotation; stability without the 3 rotation modes).
The given task parameter points are never in the training grid, so every task result is a
generalization test. Outputs: results/ai_generation/.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from turing_rd import Geometry, GSParams, SPOTS, STRIPES, icosphere, load_frog  # noqa: E402
from turing_rd import ai_init  # noqa: E402
from turing_rd.diagnostics import linear_stability, pattern_stats  # noqa: E402
from turing_rd.initial import perturbed_ic  # noqa: E402
from turing_rd.steady import relax_polish, rotation_generators  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "results" / "ai_generation"
GRID_F = np.round(np.arange(0.020, 0.06001, 0.005), 4)            # 9 x 5 "coarse" training grid
GRID_K = np.array([0.057, 0.059, 0.061, 0.063, 0.065])
GRIDS = {"coarse": (GRID_F, GRID_K),                                 # 17 x 9 "fine" grid: --grid fine
         "fine": (np.round(np.arange(0.020, 0.06001, 0.0025), 4), np.round(np.arange(0.057, 0.06501, 0.001), 3))}
TASKS = {"spots": SPOTS, "stripes": STRIPES}
TOL = 1e-11
PDE_SCHEDULE = lambda t_max: ((1.0, 20000.0), (4.0, t_max))        # noqa: E731  baseline protocol
AI_SCHEDULE = lambda t_max: ((1.0, 2000.0), (4.0, t_max))          # noqa: E731
_GEOM: dict = {}
_ROT: dict = {}


def geom(surface: str) -> Geometry:
    if surface not in _GEOM:
        mesh = load_frog(ROOT / "frog.mat") if surface == "frog" else icosphere(int(surface[-1]))
        _GEOM[surface] = Geometry(mesh, cache_dir=ROOT / "results" / "cache")
    return _GEOM[surface]


def rotations(surface: str):
    """Rotation generators T_1..3 on the sphere (rotation-aware Newton, steady.rotation_newton), else None."""
    if not surface.startswith("sphere"):
        return None
    if surface not in _ROT:
        _ROT[surface] = rotation_generators(geom(surface))
    return _ROT[surface]


def held_out_points():
    rng = np.random.default_rng(2026)
    pts = [(SPOTS.F, SPOTS.k, "task_spots"), (STRIPES.F, STRIPES.k, "task_stripes")]
    while len(pts) < 8:
        F, k = rng.uniform(0.022, 0.058), rng.uniform(0.0575, 0.0645)
        if np.abs(GRID_F - F).min() > 0.001 and np.abs(GRID_K - k).min() > 0.0004:
            pts.append((round(F, 4), round(k, 4), f"random{len(pts) - 1}"))
    return pts


def analyse(g: Geometry, p: GSParams, u, v, info: dict | None = None, rot=None) -> dict:
    """Pattern statistics; with `info` also convergence and linear stability. On the sphere (`rot` given)
    converged means a relative equilibrium (rotation-corrected residual <= TOL), and stability is judged
    with the 3 near-zero rigid-rotation modes excluded (they are reported separately)."""
    st = pattern_stats(g, v)
    out = {"type": st.get("type"), "n_components": st.get("n_components"), "wavelength": st.get("wavelength"),
           "v_max": float(v.max()), "v_min": float(v.min())}
    if info is not None:
        out.update(converged=info["converged"], rel_residual=info["rel_residual"], relax_time=info["relax_time"],
                   relax_steps=info["relax_steps"], newton_tries=len(info["newton"]), wall_s=info["wall_s"])
        if rot is not None:
            out.update(rel_residual_rot=info["rel_residual_rot"], omega=info["omega"], steady=info["steady"])
            if "orient" in info:
                out["orient"] = info["orient"]
        out["max_re_lambda"] = None
        if info["converged"]:
            try:
                ls = linear_stability(g, p, u, v, rotations=rot)
                out["max_re_lambda"] = ls["max_real"]
                if rot is not None:
                    out.update(rotation_eigenvalues=[z[0] for z in ls["rotation_eigenvalues"]],
                               rotation_overlap=ls["rotation_overlap"])
            except Exception as e:          # e.g. ARPACK not converging: record it, do not crash
                out["stability_error"] = repr(e)[:200]
        out["stable"] = bool(out["max_re_lambda"] is not None and out["max_re_lambda"] < 1e-6)
    return out


def labels(g: Geometry, u, v, kind: str) -> dict:
    """Training targets of one steady state."""
    lab = {}
    if kind != "uniform":
        lab["stats"] = ai_init.to_vector(ai_init.pattern_statistics(g, u, v)).tolist()
    if kind == "spots":
        prof = ai_init.spot_profile(g, u, v)
        lab["spot"] = [float(np.log(prof["density"]))] + prof["pu"].tolist() + prof["pv"].tolist()
    return lab


# ---------------------------------------------------------------------------------------------
# workers (run in separate processes)
# ---------------------------------------------------------------------------------------------
def run_pde(surface, F, k, seed, t_max, keep_state=False):
    g, p = geom(surface), GSParams(F, k)
    t0 = time.time()
    u0, v0 = perturbed_ic(g, eps=0.05, seed=seed, mode="seeded", normalize_z=True)
    rot = rotations(surface)
    u, v, info = relax_polish(g, p, u0, v0, schedule=PDE_SCHEDULE(t_max), check_every=1000.0, tol=TOL, rotations=rot)
    res = {"pipeline": "pde", "surface": surface, "F": F, "k": k, "seed": seed, "setup_s": 0.0}
    res.update(analyse(g, p, u, v, info, rot))
    res["solve_s"] = res["setup_s"] + res["wall_s"]
    res["total_s"] = time.time() - t0
    if keep_state:
        res["labels"] = labels(g, u, v, res["type"])
        res["state"] = (u.astype(np.float32), v.astype(np.float32))
    return res


def ai_initial_state(g: Geometry, model: dict, F: float, k: float, seed: int):
    pred = predict(model, F, k)
    kind = pred["type"]
    if kind == "uniform":
        return np.ones(g.n), np.zeros(g.n), pred
    if kind == "spots" and "spot" in pred:
        n = max(1, int(round(pred["density"] * g.area)))
        u0, v0 = ai_init.place_spots(g, ai_init.cvt_centres(g, n, seed), pred)
        pred["n_centres"] = n
    elif kind == "stripes_or_maze" and g.name.startswith("sphere"):
        # latitude bands (seed -> phase of the bands): a random maze on the sphere keeps rearranging
        u0, v0 = ai_init.bands(g, pred["stats"], phase=0.0 if seed == 0 else float(np.random.default_rng(seed).uniform(0, np.pi)))
        pred["layout"] = "bands"
    else:
        u0, v0 = ai_init.synthesize(g, pred["stats"], seed, mode="stripes" if kind == "stripes_or_maze" else "spots")
    return u0, v0, pred


def run_ai(surface, F, k, seed, t_max, model_path, keep_state=False, orient=False):
    import joblib
    g, p = geom(surface), GSParams(F, k)
    rot = rotations(surface)
    model = joblib.load(model_path)
    t0 = time.time()
    u0, v0, pred = ai_initial_state(g, model, F, k, seed)
    setup = time.time() - t0
    u, v, info = relax_polish(g, p, u0, v0, schedule=AI_SCHEDULE(t_max), check_every=500.0, tol=TOL,
                              rotations=rot, orient=orient)
    res = {"pipeline": "ai", "surface": surface, "F": F, "k": k, "seed": seed, "setup_s": setup,
           "predicted_type": pred["type"], "n_centres": pred.get("n_centres")}
    res.update(analyse(g, p, u, v, info, rot))
    res["solve_s"] = res["setup_s"] + res["wall_s"]
    res["total_s"] = time.time() - t0
    if keep_state:
        res["state"] = (u.astype(np.float32), v.astype(np.float32))
        res["initial"] = (u0.astype(np.float32), v0.astype(np.float32))
        res["history"] = info["newton"]
    return res


def _host():
    import platform
    return platform.node()


def _safe(fn, args):
    """Run a job in a worker; any exception becomes an error row (some exceptions, e.g. ARPACK's,
    cannot be pickled back to the parent and would otherwise break the whole pool)."""
    try:
        r = fn(*args)
        r["host"] = _host()
        return r
    except Exception as e:
        return {"pipeline": "ai" if fn is run_ai else "pde", "surface": args[0], "F": args[1], "k": args[2],
                "seed": args[3], "error": repr(e)[:300], "type": None, "n_components": None, "converged": False,
                "stable": False, "total_s": 0.0, "solve_s": float("nan"), "relax_time": float("nan")}


def pool_map(jobs, workers, log, checkpoint: Path | None = None):
    """Run jobs in parallel. With `checkpoint`, every finished job is pickled there at once and jobs
    already present are loaded instead of rerun, so an interrupted stage resumes where it stopped."""
    import pickle
    out, todo = [], []
    for fn, a in jobs:
        key = f"{'ai' if fn is run_ai else 'pde'}_{a[0]}_{a[1]:.4f}_{a[2]:.4f}_{a[3]}.pkl"
        path = checkpoint / key if checkpoint else None
        if path is not None and path.exists():
            out.append(pickle.loads(path.read_bytes()))
        else:
            todo.append((fn, a, path))
    if out:
        log(f"resuming: {len(out)} finished jobs loaded from {checkpoint}, {len(todo)} to run")
    if checkpoint:
        checkpoint.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_safe, fn, a): path for fn, a, path in todo}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if futs[f] is not None:
                futs[f].write_bytes(pickle.dumps(r))
            out.append(r)
            if r.get("error"):
                log(f"[{i}/{len(todo)}] ERROR {r['pipeline']} {r['surface']} F={r['F']:.4f} k={r['k']:.4f}: {r['error']}")
                continue
            log(f"[{i}/{len(jobs)}] {r['pipeline']} {r['surface']} F={r['F']:.4f} k={r['k']:.4f} seed={r['seed']}: "
                f"{r['type']} n={r['n_components']} converged={r.get('converged')} t={r.get('relax_time', 0):.0f} "
                f"{r['total_s']:.0f}s")
    return out


# ---------------------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------------------
def _mlp_regressor(seed):
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    net = MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-3, max_iter=20000, tol=1e-7, random_state=seed)
    return TransformedTargetRegressor(make_pipeline(StandardScaler(), net), transformer=StandardScaler())


def _mlp_classifier(seed):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 32), alpha=1e-3,
                                                         max_iter=20000, random_state=seed))


def fit_model(X, types, stats_y, stats_mask, spot_y, spot_mask, n_members=5):
    model = {"classes": sorted(set(types)), "members": n_members}
    model["cls"] = [_mlp_classifier(s).fit(X, types) for s in range(n_members)] if len(set(types)) > 1 else None
    model["stats"] = [_mlp_regressor(s).fit(X[stats_mask], stats_y[stats_mask]) for s in range(n_members)]
    model["spot"] = ([_mlp_regressor(s).fit(X[spot_mask], spot_y[spot_mask]) for s in range(n_members)]
                     if spot_mask.sum() >= 4 else None)
    return model


def predict(model: dict, F: float, k: float) -> dict:
    x = np.array([[F, k]])
    if model["cls"] is None:
        kind = model["classes"][0]
    else:
        proba = np.mean([c.predict_proba(x)[0] for c in model["cls"]], axis=0)
        kind = model["cls"][0].classes_[int(np.argmax(proba))]
    out = {"type": str(kind)}
    sv = np.mean([r.predict(x)[0] for r in model["stats"]], axis=0)
    out["stats"] = ai_init.from_vector(sv)
    if model["spot"] is not None:
        y = np.mean([r.predict(x)[0] for r in model["spot"]], axis=0)
        nr = len(ai_init.R_GRID)
        out.update(spot=True, density=float(np.exp(y[0])), pu=y[1:1 + nr], pv=y[1 + nr:1 + 2 * nr])
    return out


# ---------------------------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------------------------
def stage_dataset(surfaces, workers, log, t_max=100000.0, grid="coarse", seeds_per_point=1):
    for surface in surfaces:
        if (OUT / surface / "dataset.json").exists():
            log(f"{surface}: dataset already present, skipped")
            continue
        g = geom(surface)
        g.eigenbasis(600)                                # cache before the workers start
        jobs, idx = [], 0
        grid_f, grid_k = GRIDS[grid]
        for F in grid_f:
            for k in grid_k:
                for s in range(seeds_per_point):            # s = 0 reproduces the original seeds
                    jobs.append((run_pde, (surface, float(F), float(k), 1000 + idx + 100000 * s, t_max, True)))
                idx += 1
        t0 = time.time()
        rows = [r for r in pool_map(jobs, workers, log, checkpoint=OUT / surface / "jobs") if not r.get("error")]
        rows.sort(key=lambda r: (r["F"], r["k"]))
        d = OUT / surface
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "dataset_states.npz", u=np.stack([r["state"][0] for r in rows]),
                            v=np.stack([r["state"][1] for r in rows]), F=[r["F"] for r in rows], k=[r["k"] for r in rows])
        for r in rows:
            r.pop("state")
        (d / "dataset.json").write_text(json.dumps({"rows": rows, "wall_s": time.time() - t0}, indent=1))
        log(f"{surface}: dataset of {len(rows)} steady states in {time.time() - t0:.0f}s, types "
            f"{ {t: sum(r['type'] == t for r in rows) for t in set(r['type'] for r in rows)} }")


def _arrays(rows):
    X = np.array([[r["F"], r["k"]] for r in rows])
    types = np.array([r["type"] for r in rows])
    nstat = 2 + 2 * len(ai_init.PROBS)
    nspot = 1 + 2 * len(ai_init.R_GRID)
    stats_y = np.array([r["labels"].get("stats", [np.nan] * nstat) for r in rows])
    spot_y = np.array([r["labels"].get("spot", [np.nan] * nspot) for r in rows])
    return X, types, stats_y, np.isfinite(stats_y).all(1), spot_y, np.isfinite(spot_y).all(1)


def _cv_add(cv, pr, i, types, sm, stats_y, pm, spot_y, g):
    """Errors of prediction `pr` against held-out sample i."""
    cv["type_correct"].append(bool(pr["type"] == types[i]))
    if sm[i]:
        true = ai_init.from_vector(stats_y[i])
        cv["kappa0_rel"].append(abs(pr["stats"]["kappa0"] / true["kappa0"] - 1))
        cv["qv_rmse"].append(float(np.sqrt(np.mean((pr["stats"]["qv"] - true["qv"]) ** 2))))
    if pm[i] and "spot" in pr:
        dens = np.exp(spot_y[i][0])
        cv["density_rel"].append(abs(pr["density"] / dens - 1))
        cv["n_spots_rel"].append(abs(round(pr["density"] * g.area) / round(dens * g.area) - 1))
        cv["pv_rmse"].append(float(np.sqrt(np.mean((pr["pv"] - spot_y[i][1 + len(ai_init.R_GRID):]) ** 2))))


def stage_train(surfaces, log):
    import joblib
    for surface in surfaces:
        g = geom(surface)
        d = OUT / surface
        rows = json.loads((d / "dataset.json").read_text())["rows"]
        X, types, stats_y, sm, spot_y, pm = _arrays(rows)
        t0 = time.time()
        model = fit_model(X, types, stats_y, sm, spot_y, pm)
        joblib.dump(model, d / "model.joblib")
        fit_s = time.time() - t0
        # leave-one-grid-point-out generalization error
        cv = {"type_correct": [], "kappa0_rel": [], "qv_rmse": [], "density_rel": [], "n_spots_rel": [], "pv_rmse": []}
        keys = [(r["F"], r["k"]) for r in rows]
        for key in sorted(set(keys)):
            test = np.array([kk == key for kk in keys])
            tr = ~test
            m = fit_model(X[tr], types[tr], stats_y[tr], sm[tr], spot_y[tr], pm[tr], n_members=3)
            pr = predict(m, *key)
            for i in np.where(test)[0]:
                _cv_add(cv, pr, i, types, sm, stats_y, pm, spot_y, g)
        summary = {"n_samples": len(rows), "fit_s": fit_s,
                   "type_accuracy": float(np.mean(cv["type_correct"])),
                   **{f"{k}_median": float(np.median(v)) for k, v in cv.items() if k != "type_correct" and v},
                   **{f"{k}_p90": float(np.quantile(v, 0.9)) for k, v in cv.items() if k != "type_correct" and v}}
        (d / "cv.json").write_text(json.dumps({"summary": summary, "per_point": cv}, indent=1))
        log(f"{surface}: model fitted in {fit_s:.1f}s; leave-one-out {json.dumps(summary)}")


def stage_evaluate(surfaces, workers, log, seeds=(1, 2, 3), t_max=200000.0, pipelines=("ai", "pde")):
    jobs = []
    for surface in surfaces:
        train_surface = "sphere5" if surface.startswith("sphere") else surface
        model_path = str(OUT / train_surface / "model.joblib")
        if "ai" in pipelines:
            geom(surface).eigenbasis(600)
        pts = held_out_points() if surface == train_surface else held_out_points()[:2]
        for F, k, name in pts:
            for s in seeds:
                if "ai" in pipelines:
                    jobs.append((run_ai, (surface, F, k, s, t_max, model_path)))
                if "pde" in pipelines:
                    jobs.append((run_pde, (surface, F, k, s, t_max)))
    pool_map(jobs, workers, log, checkpoint=OUT / "evaluation_jobs")


def stage_summarize(log):
    """Merge every finished evaluation job (from this or another machine's evaluation_jobs/)."""
    import pickle
    rows = [pickle.loads(p.read_bytes()) for p in sorted((OUT / "evaluation_jobs").glob("*.pkl"))]
    rows = [r for r in rows if not r.get("error")]
    names = {(F, k): n for F, k, n in held_out_points()}
    for r in rows:
        r["point"] = names.get((r["F"], r["k"]), f"F{r['F']}_k{r['k']}")
    (OUT / "evaluation.json").write_text(json.dumps(rows, indent=1, default=float))
    write_summary(rows, log)


def write_summary(rows, log):
    import csv
    keys = sorted({k for r in rows for k in r})
    with open(OUT / "evaluation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def med(xs):
        xs = [x for x in xs if x is not None]
        return float(np.median(xs)) if xs else float("nan")

    def group(*keys_):
        out = {}
        for r in rows:
            out.setdefault(tuple(r[k] for k in keys_), []).append(r)
        return dict(sorted(out.items()))

    ok = lambda rs: [r for r in rs if r.get("stable")]  # noqa: E731
    corr = lambda r: r.get("rel_residual_rot", r.get("rel_residual")) or 1  # noqa: E731
    lines = ["# AI 初值 + Newton 精修 vs 普通初值 + 时间积分", "",
             "- 成功：相对残差 ≤ 1e-11，且线性稳定（max Re λ < 1e-6）。",
             "- 球面：图案会整体缓慢转动（网格对转动的约束很弱），所以用转动修正 Newton（steady.rotation_newton）"
             "求相对平衡态。成功指扣除刚体转动后的残差 ≤ 1e-11，且排除 3 个转动模后线性稳定；"
             "「末残差」列是完整残差，其中就是转动漂移本身。",
             "- 求解时间：AI 推断与合成 + 积分与 Newton 的墙钟，不含稳定性分析。两条流程交错并行运行。",
             "- 未成功的运行按上限 t = 200 000 计时，计入中位数。", "",
             "- 「≤1e-6」列：（球面上为扣除转动后的）相对残差 ≤ 1e-6 的运行数。",
             "",
             "| 曲面 | 参数点 | 流程 | 成功 | ≤1e-6 | 末残差中位数 | 扣除转动后中位数 | 求解时间中位数 s | 积分时间中位数 | 类型 | 斑点数 | 波长 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for (surf, point, pipe), rs in group("surface", "point", "pipeline").items():
        good = ok(rs)
        loose = [r for r in rs if corr(r) <= 1e-6]
        lines.append(f"| {surf} | {point} (F={rs[0]['F']}, k={rs[0]['k']}) | {pipe} | {len(good)}/{len(rs)} | "
                     f"{len(loose)}/{len(rs)} | {med([r.get('rel_residual') for r in rs]):.1e} | "
                     f"{med([corr(r) for r in rs]):.1e} | "
                     f"{med([r['solve_s'] for r in rs]):.0f} | {med([r['relax_time'] for r in rs]):.0f} | "
                     f"{'/'.join(sorted({r['type'] for r in good})) or '—'} | "
                     f"{med([r['n_components'] for r in good]):.0f} | {med([r['wavelength'] for r in good]):.3f} |")
    lines += ["", "## 每个曲面的汇总", "",
              "| 曲面 | 机器 | AI 成功率 | 普通成功率 | AI ≤1e-6 | 普通 ≤1e-6 | AI 求解时间中位数 s | 普通求解时间中位数 s | 加速比 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for (surf,), rs in group("surface").items():
        a = [r for r in rs if r["pipeline"] == "ai"]
        b = [r for r in rs if r["pipeline"] == "pde"]
        la = sum(corr(r) <= 1e-6 for r in a)
        lb = sum(corr(r) <= 1e-6 for r in b)
        ta, tb = med([r["solve_s"] for r in a]), med([r["solve_s"] for r in b])
        hosts = "/".join(sorted({str(r.get("host", "?")) for r in rs}))
        lines.append(f"| {surf} | {hosts} | {len(ok(a))}/{len(a)} | {len(ok(b))}/{len(b)} | {la}/{len(a)} | {lb}/{len(b)} | "
                     f"{ta:.0f} | {tb:.0f} | {tb / ta:.1f}x |")
    lines += ["", "## 端到端泛化误差（留出参数点上，AI 结果相对普通流程结果的特征差，各取成功运行的中位数）", "",
              "| 曲面 | 参数点 | 类型一致 | 波长相对差 | 斑点数相对差 | v 峰值相对差 |", "| --- | --- | --- | --- | --- | --- |"]
    for (surf, point), rs in group("surface", "point").items():
        a = ok([r for r in rs if r["pipeline"] == "ai"])
        b = ok([r for r in rs if r["pipeline"] == "pde"])
        if not a or not b:
            lines.append(f"| {surf} | {point} | — | — | — | — |")
            continue
        ta = max({r["type"] for r in a}, key=lambda t: sum(r["type"] == t for r in a))
        tb = max({r["type"] for r in b}, key=lambda t: sum(r["type"] == t for r in b))
        rel = lambda c: abs(med([r[c] for r in a]) / med([r[c] for r in b]) - 1) if med([r[c] for r in b]) else float("nan")  # noqa: E731
        lines.append(f"| {surf} | {point} | {'是' if ta == tb else '否'} | {rel('wavelength'):.1%} | "
                     f"{rel('n_components'):.1%} | {rel('v_max'):.1%} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))


def stage_generate(log, surfaces=("frog", "sphere6"), t_max=200000.0):
    """Task results: given spot / stripe parameters on the frog and the unit sphere (L6). On the sphere a
    converged relative equilibrium is then followed to the mesh-preferred orientation (orient=True)."""
    from turing_rd.viz import render_field
    import scipy.io as sio
    for surface in surfaces:
        train = "sphere5" if surface.startswith("sphere") else surface
        g = geom(surface)
        g.eigenbasis(600)
        for name, p in TASKS.items():
            r = run_ai(surface, p.F, p.k, 0, t_max, str(OUT / train / "model.joblib"), keep_state=True,
                       orient=surface.startswith("sphere"))
            d = OUT / "tasks" / f"{surface}_{name}"
            d.mkdir(parents=True, exist_ok=True)
            u, v = (x.astype(float) for x in r.pop("state"))
            u0, v0 = (x.astype(float) for x in r.pop("initial"))
            np.savez_compressed(d / "final.npz", u=u, v=v, u0=u0, v0=v0)
            sio.savemat(d / "solution.mat", {"Solution": {"u": u[:, None], "v": v[:, None]}})
            render_field(g, v0, d / "ai_initial_v.png", title=f"{surface} {name}: AI initial state (v)")
            rot_note = f", rotation-corrected {r['rel_residual_rot']:.1e}" if "rel_residual_rot" in r else ""
            render_field(g, v, d / "final_v.png",
                         title=f"{surface} {name}: after Newton, residual {r['rel_residual']:.1e}{rot_note}")
            (d / "summary.json").write_text(json.dumps(r, indent=1, default=float))
            log(f"task {surface}/{name}: {r['type']} n={r['n_components']} residual {r['rel_residual']:.1e}{rot_note} "
                f"max Re lambda {r['max_re_lambda']} total {r['total_s']:.0f}s")


def stage_robustness(log, surfaces, seeds, workers, t_max=200000.0):
    """Task pipeline (as in generate) at the given parameters for extra seeds; with seed 0 from generate it
    gives the spread of each task result. Finished runs are kept in tasks/robustness_jobs, so adding seeds later
    only runs the new ones. Writes tasks/robustness.md and .json."""
    jobs = []
    for surface in sorted(surfaces, key=lambda s: not s.startswith("sphere")):          # slow sphere runs first
        train = "sphere5" if surface.startswith("sphere") else surface
        geom(surface).eigenbasis(600)
        for name, p in sorted(TASKS.items(), key=lambda kv: kv[0] != "spots"):
            for sd in seeds:
                jobs.append((run_ai, (surface, p.F, p.k, sd, t_max, str(OUT / train / "model.joblib"), False,
                                      surface.startswith("sphere"))))
    pool_map(jobs, workers, log, checkpoint=OUT / "tasks" / "robustness_jobs")
    import pickle
    rows = [pickle.loads(f.read_bytes()) for f in sorted((OUT / "tasks" / "robustness_jobs").glob("*.pkl"))]
    lines = ["# 任务结果的多种子稳健性", "",
             "- 题给参数下，用同一条 AI 初值 + Newton 流程跑多个随机种子；种子 0 即 `tasks/` 里的任务结果。",
             "- 成功：相对残差 ≤ 1e-11（球面为扣除转动后）且线性稳定。", "",
             "| 任务 | 种子数 | 成功 | 类型 | 斑点数 / 段数 | 波长 λ | 最大相对残差 | max Re λ 范围 | 求解时间中位数 s |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    out = {}
    for surface in surfaces:
        for name, p in TASKS.items():
            rs = [r for r in rows if r.get("surface") == surface and abs(r["F"] - p.F) < 1e-9 and abs(r["k"] - p.k) < 1e-9
                  and not r.get("error")]
            f0 = OUT / "tasks" / f"{surface}_{name}" / "summary.json"
            if f0.exists():
                r0 = json.loads(f0.read_text()); r0.setdefault("seed", 0); rs = [r0] + [r for r in rs if r["seed"] != 0]
            if not rs:
                continue
            ok = [r for r in rs if r.get("stable")]
            lam = [r["max_re_lambda"] for r in ok if r.get("max_re_lambda") is not None]
            nc = [r["n_components"] for r in rs]
            wl = [r["wavelength"] for r in rs]
            res = [r.get("rel_residual_rot", r["rel_residual"]) if surface.startswith("sphere") else r["rel_residual"] for r in rs]
            types = sorted({r["type"] for r in rs})
            key = f"{surface}_{name}"
            out[key] = {"seeds": [r["seed"] for r in rs], "stable": len(ok), "n_components": nc, "wavelength": wl,
                        "residual": res, "max_re_lambda": lam, "types": types}
            lines.append(f"| {surface} · {name} | {len(rs)} | {len(ok)}/{len(rs)} | {'/'.join(types)} | {min(nc)}–{max(nc)} | "
                         f"{min(wl):.3f}–{max(wl):.3f} | {max(res):.1e} | "
                         f"{(f'{min(lam):.1e} … {max(lam):.1e}') if lam else '—'} | "
                         f"{float(np.median([r.get('solve_s', r.get('total_s', 0)) for r in rs])):.0f} |")
    (OUT / "tasks" / "robustness.json").write_text(json.dumps(out, indent=1, default=float))
    (OUT / "tasks" / "robustness.md").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["all", "dataset", "train", "evaluate", "summarize", "generate", "robustness"])
    ap.add_argument("--surfaces", nargs="+", default=["frog", "sphere5"])
    ap.add_argument("--eval-surfaces", nargs="+", default=None,
                    help="surfaces to evaluate (default: --surfaces, plus sphere6 task points if sphere5 is there); "
                         "lets two machines split the evaluation, merged later with `summarize`")
    ap.add_argument("--generate-surfaces", nargs="+", default=["frog", "sphere6"])
    ap.add_argument("--generate-t-max", type=float, default=200000.0, help="time limit of each task run")
    ap.add_argument("--robust-seeds", type=int, nargs="+", default=[1, 2, 3, 4], help="extra seeds for `robustness`")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-workers", type=int, default=4)
    ap.add_argument("--t-max", type=float, default=200000.0, help="time limit of each evaluation run")
    ap.add_argument("--pipelines", nargs="+", default=["ai", "pde"], choices=["ai", "pde"])
    ap.add_argument("--out", default=str(OUT), help="output folder (use a new one for a new training grid)")
    ap.add_argument("--grid", choices=sorted(GRIDS), default="coarse", help="training grid: coarse 9x5 or fine 17x9")
    ap.add_argument("--seeds-per-point", type=int, default=1, help="random initial states per training grid point")
    args = ap.parse_args()
    if sys.platform == "win32":     # keep Windows from sleeping while this job runs; released when it exits
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    logf = open(OUT / "run.log", "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    if args.stage in ("all", "dataset"):
        stage_dataset(args.surfaces, args.workers, log, grid=args.grid, seeds_per_point=args.seeds_per_point)
    if args.stage in ("all", "train"):
        stage_train(args.surfaces, log)
    if args.stage in ("all", "evaluate"):
        surfaces = args.eval_surfaces or (args.surfaces + ["sphere6"] if "sphere5" in args.surfaces else args.surfaces)
        stage_evaluate(surfaces, args.eval_workers, log, t_max=args.t_max, pipelines=args.pipelines)
    if args.stage in ("all", "evaluate", "summarize"):
        stage_summarize(log)
    if args.stage in ("all", "generate"):
        stage_generate(log, args.generate_surfaces, t_max=args.generate_t_max)
    if args.stage == "robustness":
        stage_robustness(log, args.generate_surfaces, args.robust_seeds, args.workers, t_max=args.generate_t_max)


if __name__ == "__main__":
    main()
