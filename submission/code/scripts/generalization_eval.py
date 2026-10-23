"""Generalization error of the AI component on an independent test set (formal results untouched).

The network that produced the formal results (v1, 9 x 5 training grid, 45 states per surface) is
compared on the SAME independent test set with
    v2   the same MLP ensemble on the dense self-generated data (17 x 9 grid, 2 initial states each,
         306 states per surface; results/ai_generation_fine)
    v2p  v2 with physics-informed inputs: besides (F, k), quantities of the homogeneous Gray-Scott
         states that follow from the equations (existence of the nontrivial uniform state, its trace,
         determinant and Turing dispersion relation: largest growth rate and fastest wavenumber).
         Pearson's (1993) pattern boundaries follow these curves (saddle-node, Hopf, Turing), so they
         are a physical inductive bias for unseen (F, k).
Test set: Sobol points in the training box (F 0.020-0.060, k 0.057-0.065), seeds disjoint from the
training seeds, the same PDE protocol and labels as the training data (ai_generation.run_pde, t <= 1e5).
Metrics: type accuracy per state and per point (prediction equals one of the types the PDE produced
there, which accounts for multistability), relative error of the dominant wavenumber kappa0 and of the
spot density. Nothing here changes the formal MATs, models or certificates.

    python scripts/generalization_eval.py testset --workers 20      PDE test set (resumable)
    python scripts/generalization_eval.py train                     fit v2p (and the LOO of v2p)
    python scripts/generalization_eval.py evaluate                  metrics table + figure
    python scripts/generalization_eval.py install                   copy into submission/evidence/generalization
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
from turing_rd import SPOTS, ai_init  # noqa: E402

OUT = ROOT / "results" / "generalization"
SURFACES = ("frog", "sphere5")
PACKAGE = ROOT.parent                                   # unpacked answer package: scripts live in code/scripts


def _first(*paths):
    """Repository layout first, then the unpacked answer package."""
    return next((q for q in paths if q.exists()), paths[0])


EVIDENCE = _first(ROOT / "submission" / "evidence" / "generalization", PACKAGE / "evidence" / "generalization")

MODELS_V1 = {"frog": _first(ROOT / "submission/task2_frog/ai_model", PACKAGE / "task2_frog/ai_model"),
             "sphere5": _first(ROOT / "submission/task1_sphere/ai_model", PACKAGE / "task1_sphere/ai_model")}
DENSE = ROOT / "results" / "ai_generation_fine"
N_POINTS, SEEDS = 32, (0, 1)
T_MAX = 100000.0


# ---------------------------------------------------------------------------------------------
# physics-informed inputs
# ---------------------------------------------------------------------------------------------
def physics_features(F, k, Du=SPOTS.Du, Dv=SPOTS.Dv):
    """[F, k, s, exists, tr, det, sigma0, sigma_max, kappa_star] of the upper nontrivial uniform state.

    Uniform states: v = F (1 +- sqrt(s)) / (2 (F + k)), u = (F + k) / v with s = 1 - 4 (F + k)^2 / F;
    they exist for s >= 0 (saddle-node line). At the upper state u v = F + k, so the reaction Jacobian is
    J = [[-v^2 - F, -2 (F + k)], [v^2, F + k]]; with diffusion the growth rate of wavenumber kappa is the
    largest real eigenvalue of J - kappa^2 diag(Du, Dv) (Turing analysis)."""
    s = 1.0 - 4.0 * (F + k) ** 2 / F
    if s < 0:
        return np.array([F, k, s, 0.0, 0.0, 0.0, -1e-2, -1e-2, 0.0])
    v = F * (1 + np.sqrt(s)) / (2 * (F + k))
    J = np.array([[-v * v - F, -2 * (F + k)], [v * v, F + k]])
    q = np.linspace(0.0, 2500.0, 2501)                              # kappa^2
    tr = J[0, 0] + J[1, 1] - q * (Du + Dv)
    det = (J[0, 0] - q * Du) * (J[1, 1] - q * Dv) - J[0, 1] * J[1, 0]
    disc = tr * tr - 4 * det
    growth = np.where(disc >= 0, 0.5 * (tr + np.sqrt(np.abs(disc))), 0.5 * tr)
    i = int(np.argmax(growth))
    return np.array([F, k, s, 1.0, J[0, 0] + J[1, 1], np.linalg.det(J), growth[0], growth[i], np.sqrt(q[i])])


def inputs(X, kind):
    return np.array([physics_features(F, k) for F, k in X]) if kind == "physics" else np.asarray(X, float)


def fit(rows, kind, n_members=5):
    X, types, stats_y, sm, spot_y, pm = A._arrays(rows)
    model = A.fit_model(inputs(X, kind), types, stats_y, sm, spot_y, pm, n_members=n_members)
    model["inputs"] = kind
    return model


def predict(model, F, k):
    """ai_generation.predict for either input kind (v1/v2 models carry no 'inputs' key: plain (F, k))."""
    x = inputs([[F, k]], model.get("inputs", "plain"))
    out = {}
    if model["cls"] is None:
        out["type"] = model["classes"][0]
    else:
        proba = np.mean([c.predict_proba(x)[0] for c in model["cls"]], axis=0)
        out["type"] = str(model["cls"][0].classes_[int(np.argmax(proba))])
    out["stats"] = ai_init.from_vector(np.mean([r.predict(x)[0] for r in model["stats"]], axis=0))
    if model["spot"] is not None:
        y = np.mean([r.predict(x)[0] for r in model["spot"]], axis=0)
        out["density"] = float(np.exp(y[0]))
    return out


def predict_types(model, X):
    """Batch type prediction (ensemble-mean probabilities), for maps."""
    if model["cls"] is None:
        return [model["classes"][0]] * len(X)
    x = inputs(X, model.get("inputs", "plain"))
    proba = np.mean([c.predict_proba(x) for c in model["cls"]], axis=0)
    return [str(t) for t in model["cls"][0].classes_[np.argmax(proba, axis=1)]]


# ---------------------------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------------------------
def test_points():
    from scipy.stats import qmc
    pts = qmc.Sobol(2, scramble=True, seed=2026).random(N_POINTS)
    return [(round(0.020 + 0.040 * a, 5), round(0.057 + 0.008 * b, 5)) for a, b in pts]


def stage_testset(workers, log):
    jobs = []
    for surface in SURFACES:
        A.geom(surface).eigenbasis(600)
        for i, (F, k) in enumerate(test_points()):
            for s in SEEDS:                                     # disjoint from training seeds 1000 + idx + 1e5 s
                jobs.append((A.run_pde, (surface, F, k, 700000 + 1000 * i + s, T_MAX, True, "float32")))
    rows = A.pool_map(jobs, workers, log, checkpoint=OUT / "testset_jobs")
    rows = [r for r in rows if not r.get("error")]
    for r in rows:
        r.pop("state", None)
    (OUT / "testset.json").write_text(json.dumps({"rows": rows}, indent=1, default=float))
    log(f"test set: {len(rows)} steady states")


def dense_rows(surface):
    path = _first(DENSE / surface / "dataset.json", EVIDENCE / f"dense_dataset_{surface}.json")
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def model_v2(surface):
    """The dense-data network of ai_generation.py --grid fine; refitted from its dataset when the model
    file is not at hand (same fit_model and seeds; MLP training may differ in the last digits by platform)."""
    import joblib
    path = DENSE / surface / "model.joblib"
    return joblib.load(path) if path.exists() else fit(dense_rows(surface), "plain")


def model_v2p(surface):
    import joblib
    path = OUT / f"model_v2p_{surface}.joblib"
    return joblib.load(path) if path.exists() else fit(dense_rows(surface), "physics")


def testset_rows():
    return json.loads(_first(OUT / "testset.json", EVIDENCE / "testset.json").read_text(encoding="utf-8"))["rows"]


def stage_train(log, workers):
    import joblib
    from concurrent.futures import ProcessPoolExecutor
    for surface in SURFACES:
        rows = dense_rows(surface)
        t0 = time.time()
        joblib.dump(fit(rows, "physics"), OUT / f"model_v2p_{surface}.joblib")
        log(f"{surface}: v2p fitted on {len(rows)} states in {time.time() - t0:.0f}s")
        # grouped leave-one-point-out of v2p (same protocol as ai_generation.stage_train)
        keys = sorted({(r["F"], r["k"]) for r in rows})
        with ProcessPoolExecutor(workers) as ex:
            folds = list(ex.map(_loo_fold, [(surface, key) for key in keys]))
        acc = [c for f in folds for c in f["type_correct"]]
        k0 = [x for f in folds for x in f["kappa0_rel"]]
        (OUT / f"loo_v2p_{surface}.json").write_text(json.dumps(
            {"type_accuracy": float(np.mean(acc)), "kappa0_rel_median": float(np.median(k0)), "n": len(acc)}, indent=1))
        log(f"{surface}: v2p leave-one-point-out type accuracy {np.mean(acc):.3f}, kappa0 rel {np.median(k0):.3f}")


def _loo_fold(args):
    surface, key = args
    rows = dense_rows(surface)
    tr = [r for r in rows if (r["F"], r["k"]) != key]
    te = [r for r in rows if (r["F"], r["k"]) == key]
    m = fit(tr, "physics", n_members=3)
    pr = predict(m, *key)
    out = {"type_correct": [pr["type"] == r["type"] for r in te], "kappa0_rel": []}
    for r in te:
        if "stats" in r["labels"]:
            out["kappa0_rel"].append(abs(pr["stats"]["kappa0"] / ai_init.from_vector(np.array(r["labels"]["stats"]))["kappa0"] - 1))
    return out


def metrics(model, rows, g):
    by_point = {}
    for r in rows:
        by_point.setdefault((r["F"], r["k"]), []).append(r)
    per_state, per_point, k0, dens = [], [], [], []
    for key, rs in by_point.items():
        pr = predict(model, *key)
        per_state += [pr["type"] == r["type"] for r in rs]
        per_point.append(pr["type"] in {r["type"] for r in rs})
        for r in rs:
            if "stats" in r["labels"]:
                k0.append(abs(pr["stats"]["kappa0"] / ai_init.from_vector(np.array(r["labels"]["stats"]))["kappa0"] - 1))
            if "spot" in r["labels"] and "density" in pr:
                dens.append(abs(pr["density"] / np.exp(r["labels"]["spot"][0]) - 1))
    return {"type_accuracy_states": float(np.mean(per_state)), "type_accuracy_points": float(np.mean(per_point)),
            "kappa0_rel_median": float(np.median(k0)) if k0 else None,
            "spot_density_rel_median": float(np.median(dens)) if dens else None,
            "n_states": len(per_state), "n_points": len(per_point)}


def stage_evaluate(log):
    import joblib
    rows_all = testset_rows()
    table = {}
    for surface in SURFACES:
        rows = [r for r in rows_all if r["surface"] == surface and "labels" in r]
        g = A.geom(surface)
        models = {"v1": joblib.load(MODELS_V1[surface] / "model.joblib"), "v2": model_v2(surface),
                  "v2p": model_v2p(surface)}
        table[surface] = {name: metrics(m, rows, g) for name, m in models.items()}
        types = {}
        for r in rows:
            types[r["type"]] = types.get(r["type"], 0) + 1
        table[surface]["test_types"] = types
        loo = OUT / f"loo_v2p_{surface}.json"
        if loo.exists():
            table[surface]["loo_v2p"] = json.loads(loo.read_text())
        for name in ("v1", "v2", "v2p"):
            m = table[surface][name]
            log(f"{surface} {name}: type {m['type_accuracy_states']:.1%} (points {m['type_accuracy_points']:.1%}), "
                f"kappa0 {m['kappa0_rel_median']:.1%}, density {m['spot_density_rel_median']}")
    (OUT / "summary.json").write_text(json.dumps(table, indent=1))
    write_markdown(table)
    figure(table, rows_all)


def write_markdown(t):
    L = ["# AI 组件泛化误差：独立测试集\n",
         f"测试集：训练区域内 {N_POINTS} 个 Sobol 参数点 × {len(SEEDS)} 个随机初值，与训练种子不重合，"
         "PDE 协议与训练数据相同（t ≤ 1e5，类型由稳态判定）。正式结果仍由 v1 生成，证书不变。"
         "证据目录中的 JSON 为严格格式，非有限数值写为 null。\n",
         "| 曲面 | 模型 | 训练数据 | 类型准确率（逐稳态） | 类型准确率（逐参数点，含多稳） | κ₀ 相对误差中位数 | 斑点密度相对误差中位数 |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    data = {"v1": "9×5，45 态（正式）", "v2": "17×9×2，306 态", "v2p": "306 态 + 物理特征"}
    for surface, rows in t.items():
        for name in ("v1", "v2", "v2p"):
            m = rows[name]
            dens = "—" if m["spot_density_rel_median"] is None else f"{m['spot_density_rel_median']:.1%}"
            L.append(f"| {surface} | {name} | {data[name]} | {m['type_accuracy_states']:.1%} | "
                     f"{m['type_accuracy_points']:.1%} | {m['kappa0_rel_median']:.1%} | {dens} |")
    L.append("")
    for surface, rows in t.items():
        L.append(f"- {surface} 测试集类型分布：{rows['test_types']}"
                 + (f"；v2p 加密网格留一法类型准确率 {rows['loo_v2p']['type_accuracy']:.1%}" if "loo_v2p" in rows else ""))
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def figure(t, rows_all):
    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"spots": "tab:orange", "stripes_or_maze": "tab:blue", "uniform": "lightgray", "holes": "tab:green"}
    fig, axs = plt.subplots(len(SURFACES), 3, figsize=(13, 4.2 * len(SURFACES)))
    Fg, kg = np.meshgrid(np.linspace(0.02, 0.06, 81), np.linspace(0.057, 0.065, 41))
    for i, surface in enumerate(SURFACES):
        models = {"v1": joblib.load(MODELS_V1[surface] / "model.joblib"), "v2": model_v2(surface),
                  "v2p": model_v2p(surface)}
        rows = [r for r in rows_all if r["surface"] == surface]
        for j, (name, m) in enumerate(models.items()):
            a = axs[i, j]
            types = predict_types(m, np.column_stack([Fg.ravel(), kg.ravel()]))
            lab = np.array([list(colors).index(t) if t in colors else 0 for t in types]).reshape(Fg.shape)
            a.contourf(Fg, kg, lab, levels=np.arange(-0.5, len(colors)), colors=list(colors.values()), alpha=0.35)
            for r in rows:
                ok = predict(m, r["F"], r["k"])["type"] == r["type"]
                a.scatter(r["F"], r["k"], c=colors.get(r["type"], "k"), marker="o" if ok else "x", s=28,
                          edgecolors="k", linewidths=0.4)
            mm = t[surface][name]
            a.set_title(f"{surface} {name}: type {mm['type_accuracy_states']:.0%}, κ₀ err {mm['kappa0_rel_median']:.1%}",
                        fontsize=10)
            a.set_xlabel("F")
            a.set_ylabel("k")
    from matplotlib.patches import Patch
    axs[0, 0].legend([Patch(color=c) for c in colors.values()] + [plt.Line2D([], [], marker="o", ls="", c="k"),
                      plt.Line2D([], [], marker="x", ls="", c="k")],
                     list(colors) + ["test state, predicted right", "test state, predicted wrong"],
                     fontsize=7, loc="lower right", framealpha=0.8)
    fig.suptitle("Predicted pattern type (shading) vs PDE type on the independent test set (markers)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "type_maps.png", dpi=100)
    plt.close(fig)


def _finite(value):
    """Strict JSON: non-finite floats become null (recorded in summary.md)."""
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):                          # machine names are not evidence
        return {k: _finite(v) for k, v in value.items() if k != "host"}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def stage_install(log):
    """Copy the evaluation into submission/evidence/generalization/ (i03_package.py packs evidence/
    automatically). Formal MATs, models and certificates are not touched. Never overwrites."""
    import shutil
    from json_safe import write_json_exclusive
    dest = ROOT / "submission" / "evidence" / "generalization"
    if dest.exists():
        raise SystemExit(f"{dest} exists; nothing overwritten")
    dest.mkdir(parents=True)
    write_json_exclusive(dest / "summary.json", _finite(json.loads((OUT / "summary.json").read_text())))
    write_json_exclusive(dest / "testset.json", {"rows": _finite(testset_rows())})
    for surface in SURFACES:
        write_json_exclusive(dest / f"dense_dataset_{surface}.json", {"rows": _finite(dense_rows(surface))})
    for name in ("summary.md", "type_maps.png"):
        shutil.copy2(OUT / name, dest / name)
    log(f"installed into {dest.relative_to(ROOT)}: {sorted(p.name for p in dest.iterdir())}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("testset", "train", "evaluate", "install"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    A.OUT = OUT                                        # pool_map checkpoints and geometry caches stay here
    fh = open(OUT / f"{args.stage}.log", "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()
    if args.stage == "testset":
        stage_testset(args.workers, log)
    elif args.stage == "train":
        stage_train(log, args.workers)
    elif args.stage == "evaluate":
        stage_evaluate(log)
    else:
        stage_install(log)


if __name__ == "__main__":
    main()
