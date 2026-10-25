"""Active learning for the AI component: label the (F, k) points where the network is least sure.

The dense-data network v2 (306 states per surface) still errs almost only next to pattern-type
boundaries. Uncertainty sampling (Settles 2009; for PDE surrogates e.g. Pestourie et al. 2020, npj
Comput. Mater. 6:164) spends new PDE solves exactly there:
    select   score a Sobol candidate pool by the ensemble's type uncertainty (1 - margin between the two
             most probable types, plus the spread of the members), keep the top points under a minimum
             spacing; no test-set information is used
    run      PDE steady states at the selected points (same protocol, labels and t <= 1e5 as the
             training data; seeds disjoint from training and test seeds), resumable
    train    v3 = the same MLP ensemble on dense + new states; metrics on the SAME independent test set as
             generalization_eval.py (which is never used for selection or training here)

    python scripts/active_learning.py select
    python scripts/active_learning.py run --surfaces frog --workers 8
    python scripts/active_learning.py train
    python scripts/active_learning.py validate               dense-grid LOO with / without the new states
    python scripts/active_learning.py e2e --model v2         AI initial state + PDE at the 8 held-out frog points x 3 seeds
    python scripts/active_learning.py install                evidence folder
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
import generalization_eval as G  # noqa: E402

OUT = ROOT / "results" / "active_learning"
N_SELECT, SEEDS, POOL, MIN_SPACING = 48, (0, 1), 4096, 0.03
BOX = ((0.020, 0.060), (0.057, 0.065))


def _unit(X):
    return np.column_stack([(X[:, 0] - BOX[0][0]) / (BOX[0][1] - BOX[0][0]), (X[:, 1] - BOX[1][0]) / (BOX[1][1] - BOX[1][0])])


def uncertainty(model, X):
    """1 - (p1 - p2) of the ensemble-mean type probabilities, plus the members' spread of p1's class."""
    x = G.inputs(X, model.get("inputs", "plain"))
    P = np.stack([c.predict_proba(x) for c in model["cls"]])         # members x points x classes
    mean = P.mean(0)
    top = np.sort(mean, axis=1)
    best = np.argmax(mean, axis=1)
    spread = P[:, np.arange(len(X)), best].std(0)
    return 1.0 - (top[:, -1] - top[:, -2]) + spread


def stage_select(log):
    from scipy.stats import qmc
    sel = {}
    for surface in G.SURFACES:
        model = G.model_v2(surface)
        pool = qmc.Sobol(2, scramble=True, seed=777).random(POOL)
        X = np.column_stack([BOX[0][0] + (BOX[0][1] - BOX[0][0]) * pool[:, 0], BOX[1][0] + (BOX[1][1] - BOX[1][0]) * pool[:, 1]])
        score = uncertainty(model, X)
        train = _unit(np.array([[r["F"], r["k"]] for r in G.dense_rows(surface)]))
        chosen = []
        for i in np.argsort(-score):
            u = _unit(X[i:i + 1])[0]
            near = np.vstack([train] + ([_unit(X[chosen])] if chosen else []))
            if np.min(np.linalg.norm(near - u, axis=1)) >= MIN_SPACING:
                chosen.append(i)
            if len(chosen) == N_SELECT:
                break
        sel[surface] = [{"F": round(float(X[i, 0]), 5), "k": round(float(X[i, 1]), 5), "score": float(score[i])}
                        for i in chosen]
        log(f"{surface}: {len(chosen)} points, uncertainty {score[chosen].min():.2f}-{score[chosen].max():.2f} "
            f"(pool median {np.median(score):.2f})")
    (OUT / "selected.json").write_text(json.dumps(sel, indent=1))


def stage_run(surfaces, workers, log):
    sel = json.loads((OUT / "selected.json").read_text())
    jobs = []
    for surface in surfaces:
        A.geom(surface).eigenbasis(600)
        for i, p in enumerate(sel[surface]):
            for s in SEEDS:                                  # disjoint from training (1000+) and test (700000+) seeds
                jobs.append((A.run_pde, (surface, p["F"], p["k"], 900000 + 1000 * i + s, G.T_MAX, True, "float32")))
    rows = [r for r in A.pool_map(jobs, workers, log, checkpoint=OUT / "jobs") if not r.get("error")]
    log(f"{len(rows)} labelled states for {surfaces}")


def new_rows(surface):
    # Reconstruct the requested jobs; never sweep old rounded-key pickle caches.
    sel = json.loads((OUT / "selected.json").read_text(encoding="utf-8"))
    jobs = [(A.run_pde, (surface, p["F"], p["k"], 900000 + 1000 * i + s,
                        G.T_MAX, True, "float32"))
            for i, p in enumerate(sel[surface]) for s in SEEDS]
    rows = []
    for r in A.cached_results(jobs, OUT / "jobs"):
        if not r.get("error") and "labels" in r:
            r.pop("state", None)
            rows.append(r)
    return rows


def stage_train(log):
    import joblib
    table = {}
    for surface in G.SURFACES:
        rows_new = new_rows(surface)
        rows = G.dense_rows(surface) + rows_new
        t0 = time.time()
        model = G.fit(rows, "plain")
        joblib.dump(model, OUT / f"model_v3_{surface}.joblib")
        A.EC.write_training_provenance(OUT / f"model_v3_{surface}.joblib",
            {"dense_data": A.EC.file_identity(G._first(G.DENSE / surface / "dataset.json",
                                                     G.EVIDENCE / f"dense_dataset_{surface}.json")),
             "selection": A.EC.file_identity(OUT / "selected.json"),
             "new_experiments": [r["experiment_cache"] for r in rows_new]},
            {"stage": "explicit active_learning train", "rows": len(rows)})
        test = [r for r in G.testset_rows() if r["surface"] == surface and "labels" in r]
        g = A.geom(surface)
        table[surface] = {"v2": G.metrics(G.model_v2(surface), test, g), "v3": G.metrics(model, test, g),
                          "n_train_v2": len(G.dense_rows(surface)), "n_train_v3": len(rows), "n_new": len(rows_new),
                          "fit_s": time.time() - t0}
        for name in ("v2", "v3"):
            m = table[surface][name]
            log(f"{surface} {name}: type {m['type_accuracy_states']:.1%} (points {m['type_accuracy_points']:.1%}), "
                f"kappa0 {m['kappa0_rel_median']:.1%}, density {m['spot_density_rel_median']}")
    (OUT / "summary.json").write_text(json.dumps(table, indent=1))


def _extra(surface, variant):
    new = new_rows(surface)
    if variant == "none":
        return []
    if variant == "all":
        return new
    types = {}
    for r in new:
        types.setdefault((r["F"], r["k"]), set()).add(r["type"])
    return [r for r in new if len(types[(r["F"], r["k"])]) == 1]       # both seeds agree on the type


def _val_fold(args):
    surface, key, variant = args
    dense = G.dense_rows(surface)
    tr = [r for r in dense if (r["F"], r["k"]) != key] + _extra(surface, variant)
    te = [r for r in dense if (r["F"], r["k"]) == key]
    X = np.array([[r["F"], r["k"]] for r in tr])
    y = np.array([r["type"] for r in tr])
    ms = [A._mlp_classifier(s).fit(X, y) for s in range(3)]
    p = np.mean([m.predict_proba([key])[0] for m in ms], axis=0)
    t = ms[0].classes_[int(np.argmax(p))]
    return [bool(t == r["type"]) for r in te]


def stage_validate(workers, log):
    """Validation without the test set: leave-one-grid-point-out on the dense grid, training on the rest
    of the grid plus none / all / seed-consistent active-learning states."""
    from concurrent.futures import ProcessPoolExecutor
    out = {}
    for surface in G.SURFACES:
        keys = sorted({(r["F"], r["k"]) for r in G.dense_rows(surface)})
        out[surface] = {}
        for variant in ("none", "all", "consistent"):
            with ProcessPoolExecutor(workers) as ex:
                res = list(ex.map(_val_fold, [(surface, key, variant) for key in keys]))
            acc = float(np.mean([o for r in res for o in r]))
            out[surface][variant] = {"dense_grid_loo_type_accuracy": acc, "n_extra": len(_extra(surface, variant))}
            log(f"{surface} +AL {variant}: dense-grid LOO type accuracy {acc:.3f} ({len(_extra(surface, variant))} extra states)")
    (OUT / "validation.json").write_text(json.dumps(out, indent=1))


def stage_install(log):
    """Evidence folder submission/evidence/active_learning/ (packed by i03_package.py). Never overwrites."""
    from json_safe import write_json_exclusive
    dest = ROOT / "submission" / "evidence" / "active_learning"
    if dest.exists():
        raise SystemExit(f"{dest} exists; nothing overwritten")
    dest.mkdir(parents=True)
    summary = json.loads((OUT / "summary.json").read_text())
    val = json.loads((OUT / "validation.json").read_text())
    e2e = {p.stem: json.loads(p.read_text()) for p in sorted(OUT.glob("e2e_*_frog.json"))}
    sel = json.loads((OUT / "selected.json").read_text())
    write_json_exclusive(dest / "summary.json", {"test_set": summary, "validation": val, "selected_points": sel,
                                                 "end_to_end": e2e})
    disagree = {}
    for surface in G.SURFACES:
        types = {}
        for r in new_rows(surface):
            types.setdefault((r["F"], r["k"]), set()).add(r["type"])
        disagree[surface] = (sum(len(v) > 1 for v in types.values()), len(types))
    record = {"test_set": summary, "validation": val, "selected_points": sel, "end_to_end": e2e}
    (dest / "summary.md").write_text(render_markdown(record, disagree), encoding="utf-8")
    log(f"installed {dest.relative_to(ROOT)}")


def render_markdown(record, disagreement=None):
    """Pure renderer; never select points, train, or solve."""
    from scripts.evidence_stats import end_to_end_statistics
    lines = ["# 主动学习与中位积分步数", "",
             "主动学习补充的是 PDE 数值终态，不能全部视为完整稳态。选点未使用测试集。",
             "正式模型为 v1；研究中保留的改进模型为 v2，v3 为未采用的对照。", "",
             "| 曲面 | 测试集终态类型准确率 v2 → v3 | 加密网格留一：不加 / 全加 / 一致子集 | 新终态数 | 类型差异信息 |",
             "|---|---|---|---:|---|"]
    for surface in G.SURFACES:
        t, v = record["test_set"][surface], record["validation"][surface]
        n = len(record["selected_points"][surface]); total = v["all"]["n_extra"]; consistent = v["consistent"]["n_extra"]
        if disagreement is not None and surface in disagreement:
            different, points = disagreement[surface]
            detail = f"原始标签统计 {different}/{points} 个参数点类型不同"
        elif total == 2 * n and 0 <= consistent <= total and (total - consistent) % 2 == 0:
            detail = f"按一致筛选数量推算 {(total-consistent)//2}/{n}；未重验逐新样本稳态"
        else:
            detail = "缺少逐新样本记录，无法判定"
        lines.append(f"| {surface} | {t['v2']['type_accuracy_states']:.1%} → {t['v3']['type_accuracy_states']:.1%} | "
                     f"{v['none']['dense_grid_loo_type_accuracy']:.1%} / {v['all']['dense_grid_loo_type_accuracy']:.1%} / "
                     f"{v['consistent']['dense_grid_loo_type_accuracy']:.1%} | {total} | {detail} |")
    lines += ["", "不同初值的终态标签存在差异；缺少严格稳态证据时，不能据此证明差异一定来自动力学多稳态。",
              "本批主动学习记录未带来一致提升；这不证明其他补样策略必然无效。"]
    for name, e in end_to_end_statistics(record).items():
        lines += ["", f"## 中位积分步数（{name}）", "",
                  "完成计数沿用历史残差和单配置谱标志，不代表当前三配置正式验收。",
                  "| 流程 | 历史完成记录 | 中位积分步数 |", "|---|---:|---:|"]
        for label, value in e["summary"].items():
            lines.append(f"| {label} | {value['complete']}/{value['runs']} | {value['median_steps']:g} |")
        lines += ["", f"纯 PDE / 新 AI 的中位积分步数比为 **{e['median_integration_step_ratio']:.2f}×**。",
                  "该指标不是完整墙钟加速；没有包含完整初始化、Newton、求谱等流程耗时的可比逐运行记录。",
                  "pde/v1 仅保存历史汇总，不能重算它们的逐运行耗时或把旧完成数升级为新严格证书。"]
    return "\n".join(lines) + "\n"


def model_path(model_name, surface):
    """Model names resolve to saved weights, never to an implicit training fallback."""
    paths = {"v1": G.MODELS_V1[surface] / "model.joblib",
             "v2": G.DENSE / surface / "model.joblib",
             "v2p": G.OUT / f"model_v2p_{surface}.joblib",
             "v3": OUT / f"model_v3_{surface}.joblib"}
    if model_name not in paths:
        raise ValueError(f"unknown model: {model_name}")
    path = paths[model_name]
    if not path.is_file():
        raise FileNotFoundError(f"saved {model_name} weights missing; train explicitly: {path}")
    return path


def stage_e2e(workers, log, model_name="v3", surface="frog"):
    """AI initial state from the new network + the unchanged PDE pipeline at the 8 held-out points x 3
    seeds of the legacy evaluation; compared with that evaluation's ordinary-PDE and v1 rows by the
    machine-independent number of time steps and by completion (full residual <= 1e-11, stable)."""
    import csv
    weights = str(model_path(model_name, surface))
    A.geom(surface).eigenbasis(600)
    jobs = [(A.run_ai, (surface, F, k, s, 200000.0, weights)) for F, k, _ in A.held_out_points() for s in (1, 2, 3)]
    rows = [r for r in A.pool_map(jobs, workers, log, checkpoint=OUT / f"e2e_{model_name}_jobs") if not r.get("error")]
    legacy = [r for r in csv.DictReader(open(ROOT / "submission/evidence/evaluation.csv", encoding="utf-8"))
              if r["surface"] == surface]
    names = {(F, k): n for F, k, n in A.held_out_points()}

    def summarize(rs, conv, steps):
        ok = [conv(r) for r in rs]
        return {"runs": len(rs), "complete": int(sum(ok)), "median_steps": float(np.median([steps(r) for r in rs])),
                "median_steps_complete": float(np.median([steps(r) for r, o in zip(rs, ok) if o])) if any(ok) else None}
    out = {"pde (legacy)": summarize([r for r in legacy if r["pipeline"] == "pde"],
                                     lambda r: r["converged"] == "True" and r["stable"] == "True", lambda r: float(r["relax_steps"])),
           "ai v1 (legacy)": summarize([r for r in legacy if r["pipeline"] == "ai"],
                                       lambda r: r["converged"] == "True" and r["stable"] == "True", lambda r: float(r["relax_steps"])),
           f"ai {model_name}": summarize(rows, lambda r: bool(r["converged"] and r["stable"]), lambda r: float(r["relax_steps"]))}
    per_point = {}
    for r in rows:
        per_point.setdefault(names.get((r["F"], r["k"]), str((r["F"], r["k"]))), []).append(
            {"seed": r["seed"], "type": r["type"], "predicted": r["predicted_type"], "steps": r["relax_steps"],
             "residual": r["rel_residual"], "stable": r["stable"]})
    (OUT / f"e2e_{model_name}_{surface}.json").write_text(json.dumps({"summary": out, "per_point": per_point}, indent=1,
                                                                     default=float))
    for k_, v in out.items():
        log(f"{surface} {k_}: complete {v['complete']}/{v['runs']}, median steps {v['median_steps']:.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("select", "run", "train", "validate", "e2e", "install"))
    ap.add_argument("--model", default="v3", choices=("v1", "v2", "v2p", "v3"), help="e2e: saved model version")
    ap.add_argument("--surfaces", nargs="+", default=list(G.SURFACES))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fh = open(OUT / f"{args.stage}.log", "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()
    if args.stage == "select":
        stage_select(log)
    elif args.stage == "run":
        stage_run(args.surfaces, args.workers, log)
    elif args.stage == "train":
        stage_train(log)
    elif args.stage == "validate":
        stage_validate(args.workers, log)
    elif args.stage == "e2e":
        stage_e2e(args.workers, log, args.model)
    else:
        stage_install(log)


if __name__ == "__main__":
    main()
