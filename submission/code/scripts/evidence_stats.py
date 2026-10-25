"""Scientific endpoint statistics and pure renderers; no training or PDE solves.

python scripts/evidence_stats.py --out <fresh-directory>
Only the saved v1/v2 weights are used. Missing models fail explicitly.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from turing_rd.spectral_policy import (PAIR_ABSOLUTE_TOL, PAIR_RELATIVE_TOL,
                                       POSITIVE_FLOOR, SIGN_ERROR_FACTOR, STEADY_TOL)

SURFACES = ("frog", "sphere5")
SPECTRAL_STATUSES = ("not_attempted", "solver_not_converged", "invalid_eigenpairs",
                     "sign_unresolved", "numerical_sample_passed", "reliable_positive",
                     "attempted_insufficient_detail", "not_recorded")
STATUS_LABELS = ("未尝试", "已尝试但不收敛", "特征对无效", "符号未判定", "单配置数值谱检查通过",
                 "可靠正模态", "已尝试但细节不足", "未记录，无法判定")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def full_steady(row):
    value = row.get("rel_residual")
    return finite(value) and 0 <= value <= STEADY_TOL


def spectral_assessment(row):
    """Classify saved single-shift evidence without inventing absent solver fields.

    analyse() skips spectroscopy for nonsteady endpoints. Older training rows
    without that explicit scope/gate are 'not recorded', not 'not attempted'.
    A passing sample is not a formal three-configuration certificate.
    """
    sample = row.get("linear_stability_sample")
    if not isinstance(sample, dict) or not sample:
        if row.get("stability_error"):
            return "attempted_insufficient_detail"
        if (row.get("stability_scope") == "single_shift_unprojected_sample"
                and not full_steady(row) and row.get("max_re_lambda") is None):
            return "not_attempted"
        return "not_recorded"
    pairs = sample.get("pairs")
    metrics, incomplete = [], False
    for pair in pairs if isinstance(pairs, list) else []:
        ev = pair.get("eigenvalue") if isinstance(pair, dict) else None
        a = pair.get("residual_absolute_M") if isinstance(pair, dict) else None
        r = pair.get("residual_relative_M") if isinstance(pair, dict) else None
        if (not isinstance(ev, list) or len(ev) != 2 or not all(finite(x) for x in (*ev, a, r))
                or a < 0 or r < 0):
            incomplete = True
            continue
        valid = a <= PAIR_ABSOLUTE_TOL and r <= PAIR_RELATIVE_TOL
        if pair.get("residual_passed") is not True:
            incomplete = True
        margin = max(SIGN_ERROR_FACTOR * a, POSITIVE_FLOOR)
        metrics.append((valid, ev[0], margin))
    if any(valid and real > margin for valid, real, margin in metrics):
        return "reliable_positive"
    if sample.get("solver_converged") is False:
        return "solver_not_converged"
    if any(not valid for valid, _, _ in metrics):
        return "invalid_eigenpairs"
    values = sample.get("eigenvalues")
    if (incomplete or not metrics or sample.get("solver_converged") is not True
            or sample.get("all_pairs_validated") is not True or not isinstance(values, list)
            or len(pairs) != len(values) or any(p["eigenvalue"] != v for p, v in zip(pairs, values))):
        return "attempted_insufficient_detail"
    if not all(real < -margin for _, real, margin in metrics):
        return "sign_unresolved"
    if sample.get("sampled_signs_resolved_negative") is not True:
        return "sign_unresolved"
    return "numerical_sample_passed"


def counts(rows, surface):
    # Both terms use the same surface, including when the caller supplies all 128 rows.
    rows = [r for r in rows if r["surface"] == surface]
    keys = [(r["surface"], r["F"], r["k"], r["seed"]) for r in rows]
    statuses = Counter(spectral_assessment(r) for r in rows)
    steady = [r for r in rows if full_steady(r)]
    return {"surface": surface, "n": len(rows), "unique_rows": len(set(keys)),
            "duplicate_rows": len(rows) - len(set(keys)), "steady": len(steady),
            "nonsteady": len(rows) - len(steady), "types": dict(Counter(r["type"] for r in rows)),
            "steady_types": dict(Counter(r["type"] for r in steady)),
            "spectral_status": {s: statuses[s] for s in SPECTRAL_STATUSES},
            "spectrum_attempted": sum(v for s, v in statuses.items() if s not in ("not_attempted", "not_recorded")),
            "steady_and_spec": sum(full_steady(r) and spectral_assessment(r) == "numerical_sample_passed" for r in rows),
            "spectrum_scope": "saved single-shift sample, not formal multi-configuration acceptance"}


def accuracy(rows, predictions, predicate=lambda r: True):
    if len(rows) != len(predictions):
        raise ValueError("one prediction is required for each endpoint")
    selected = [(r, p) for r, p in zip(rows, predictions) if predicate(r)]
    correct = sum(r["type"] == p for r, p in selected)
    return {"n": len(selected), "correct": correct,
            "accuracy": correct / len(selected) if selected else None}


def independence(train, test, surface):
    train = [r for r in train if r["surface"] == surface]
    test = [r for r in test if r["surface"] == surface]
    tr = {(r["F"], r["k"]) for r in train}; te = {(r["F"], r["k"]) for r in test}
    ranges = {n: [min(r[n] for r in train), max(r[n] for r in train)] for n in ("F", "k")}
    inside = all(all(ranges[n][0] <= r[n] <= ranges[n][1] for n in ranges) for r in test)
    return {"train_points": len(tr), "test_points": len(te), "overlapping_points": len(tr & te),
            "overlapping_seeds": len({r["seed"] for r in train} & {r["seed"] for r in test}),
            "duplicate_test_rows": counts(test, surface)["duplicate_rows"],
            **{f"train_{n}_range": ranges[n] for n in ranges},
            **{f"test_{n}_range": [min(r[n] for r in test), max(r[n] for r in test)] for n in ranges},
            "scope": "training_box_interpolation" if inside else "contains_out_of_box_points"}


def build_quality(test, dense, predictions):
    q = {"schema_version": "scientific-quality-v2", "threshold": STEADY_TOL,
         "dense": {}, "test": {}, "independence": {}, "accuracy": {},
         "model_roles": {"formal": "v1", "research_retained": "v2"}}
    for surface in SURFACES:
        rows = [r for r in test if r["surface"] == surface]
        q["dense"][surface] = counts(dense[surface], surface)
        q["test"][surface] = counts(rows, surface)
        q["independence"][surface] = independence(dense[surface], rows, surface)
        q["accuracy"][surface] = {}
        for model, pred in predictions[surface].items():
            q["accuracy"][surface][model] = {
                "all": accuracy(rows, pred), "steady": accuracy(rows, pred, full_steady),
                "steady_spec": accuracy(rows, pred, lambda r: full_steady(r) and spectral_assessment(r) == "numerical_sample_passed"),
                "steady_nontrivial": accuracy(rows, pred, lambda r: full_steady(r) and r["type"] != "uniform"),
                "nonsteady": accuracy(rows, pred, lambda r: not full_steady(r)),
                "by_class": {t: accuracy(rows, pred, lambda r, t=t: r["type"] == t) for t in sorted({r["type"] for r in rows})}}
    q["totals"] = {"endpoints": len(test), "steady": sum(full_steady(r) for r in test)}
    q["limitations"] = [
        "Strata use stored endpoint residuals, not a new float32-field MAT certification.",
        "Single-shift sample acceptance is not formal three-configuration acceptance or a full-spectrum proof.",
        "Original requested_k and eigenvectors are not saved in these legacy single-shift summaries.",
        "Missing training spectral diagnostics do not establish either an attempt or a pass.",
        "Uniform steady samples cannot demonstrate nontrivial patterned-steady generalization."]
    return q


def recompute_quality(root=ROOT):
    """Infer with immutable saved models, never fit or solve."""
    from scripts import generalization_eval as G
    import joblib
    root = Path(root); evidence = root / "submission/evidence/generalization"
    test_file = evidence / "testset.json"; test = load(test_file)["rows"]
    dense, predictions, sources, model_info, formal_independence = {}, {}, {}, {}, {}
    sources[test_file.relative_to(root).as_posix()] = sha(test_file)
    for surface in SURFACES:
        path = evidence / f"dense_dataset_{surface}.json"
        dense[surface] = load(path)["rows"]; sources[path.relative_to(root).as_posix()] = sha(path)
        rows = [r for r in test if r["surface"] == surface]
        paths = {"v1": root / ("submission/task2_frog/ai_model" if surface == "frog" else "submission/task1_sphere/ai_model") / "model.joblib",
                 "v2": root / "results/ai_generation_fine" / surface / "model.joblib"}
        predictions[surface], model_info[surface] = {}, {}
        for name, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"saved {surface}/{name} model required; automatic refitting is disabled: {path}")
            model = joblib.load(path)
            predictions[surface][name] = G.predict_types(model, [[r["F"], r["k"]] for r in rows])
            source = path.relative_to(root).as_posix(); sources[source] = sha(path)
            model_info[surface][name] = {"path": source, "sha256": sources[source]}
        formal_data = paths["v1"].with_name("dataset.json")
        formal_rows = load(formal_data)["rows"]
        sources[formal_data.relative_to(root).as_posix()] = sha(formal_data)
        formal_independence[surface] = independence(formal_rows, rows, surface)
    q = build_quality(test, dense, predictions)
    from importlib.metadata import version
    q.update(provenance={"files": sources, "method": "saved model inference, no training/PDE",
                         "environment": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib")}},
             models=model_info, formal_train_independence=formal_independence)
    prediction_rows = []
    for surface in SURFACES:
        for i, row in enumerate(r for r in test if r["surface"] == surface):
            prediction_rows.append({**{k: row[k] for k in ("surface", "F", "k", "seed", "type")},
                                    "full_steady": full_steady(row), "spectral_status": spectral_assessment(row),
                                    **{name: pred[i] for name, pred in predictions[surface].items()}})
    return q, prediction_rows


def fraction(value):
    return "无样本" if not value["n"] else f"{value['correct']}/{value['n']} ({value['accuracy']:.1%})"


def quality_markdown(q):
    L = ["# 泛化数据质量分层（从原始记录和已保存模型重算）", "",
         f"共 {q['totals']['endpoints']} 个 PDE 数值终态，其中 {q['totals']['steady']} 个满足记录的完整稳态残差 ≤ {q['threshold']:g}。",
         "单配置谱筛选与正式三配置证书分开；训练记录缺少谱细节时记为未记录，无法判定。", "",
         "| 数据集 | 数值终态 | 完整稳态 | 重复键 | " + " | ".join(STATUS_LABELS) + " |",
         "|---|---:|---:|---:|" + "---:|" * len(SPECTRAL_STATUSES)]
    for dataset, title in (("dense", "训练"), ("test", "测试")):
        for surface in SURFACES:
            r = q[dataset][surface]
            L.append(f"| {surface} {title} | {r['n']} | {r['steady']} | {r['duplicate_rows']} | "
                     + " | ".join(str(r["spectral_status"][s]) for s in SPECTRAL_STATUSES) + " |")
    L += ["", "| 曲面 | 模型 | 全部数值终态类型准确率 | 完整稳态子集 | 稳态且单配置谱通过 | 非平凡稳态子集 |",
          "|---|---|---:|---:|---:|---:|"]
    for surface in SURFACES:
        for model, r in q["accuracy"][surface].items():
            L.append(f"| {surface} | {model} | {fraction(r['all'])} | {fraction(r['steady'])} | "
                     f"{fraction(r['steady_spec'])} | {fraction(r['steady_nontrivial'])} |")
    for surface in SURFACES:
        r, ind = q["test"][surface], q["independence"][surface]
        L += ["", f"- {surface} 完整稳态 {r['steady']}/{r['n']}，类型分布：{r['steady_types']}。",
              f"- {surface}：{ind['test_points']} 个测试参数点，训练点重叠 {ind['overlapping_points']}，"
              f"种子重叠 {ind['overlapping_seeds']}，重复测试记录 {ind['duplicate_test_rows']}；范围口径 `{ind['scope']}`。"]
    L += ["", "球面完整稳态子集全部为均匀态时，不能证明非平凡斑点或条纹稳态预测能力。",
          "训练范围内的新参数点属于插值测试；混合终态分类准确率不等于生成成功率。",
          "正式生成模型为 v1，研究中保留的改进模型为 v2；训练终态中包含未达完整稳态的标签。", ""]
    return "\n".join(L)


def end_to_end_statistics(record):
    output = {}
    for name, data in record.get("end_to_end", {}).items():
        summary = {k: dict(v) for k, v in data["summary"].items()}
        rows = [r for rs in data.get("per_point", {}).values() for r in rs]
        ai = next((k for k in summary if k.startswith("ai ") and "legacy" not in k), None)
        if rows and ai:
            derived = {"runs": len(rows), "complete": sum(finite(r.get("residual")) and 0 <= r["residual"] <= STEADY_TOL and r.get("stable") is True for r in rows),
                       "median_steps": statistics.median(r["steps"] for r in rows)}
            for key, value in derived.items():
                if summary[ai][key] != value:
                    raise ValueError(f"end-to-end summary contradicts per-run records: {name}/{key}")
            summary[ai].update(derived)
        pde = summary.get("pde (legacy)", {})
        ratio = pde.get("median_steps", 0) / summary[ai]["median_steps"] if ai and summary[ai]["median_steps"] else None
        output[name] = {"summary": summary, "median_integration_step_ratio": ratio,
                        "new_per_run_records": len(rows), "point_count": len(data.get("per_point", {})),
                        "wall_clock_speedup_verified": False,
                        "scope": "legacy completion flags plus stored residual, not formal spectral recertification"}
    return output


def context(root, quality):
    root = Path(root); sub = root / "submission"
    active = load(sub / "evidence/active_learning/summary.json")
    bands = load(sub / "evidence/sphere_bandcount/summary.json")
    bm = load(sub / "task3_inversion/best_match/summary.json")
    inv = load(sub / "task3_inversion/summary.json")
    chosen = bands["chosen_band_count"]
    stripes = load(sub / "task1_sphere/stripes/summary.json")
    pattern = stripes.get("pattern_after") or stripes.get("pattern") or {}
    # Read only saved evidence here: formatting must never start a PDE/eigensolve.
    spectral_path = sub / "evidence/methodology/best_match_spectrum/certificate.json"
    saved_spectrum = None
    if spectral_path.is_file():
        from turing_rd.spectral_policy import FROG_CONFIGS, FROG_SIGN_FLOOR, evaluate_spectra
        saved_spectrum = load(spectral_path)
        if saved_spectrum["input"]["mat_sha256"] != sha(sub / "task3_inversion/best_match/solution_best_match.mat"):
            raise ValueError("best-match supplemental evidence belongs to another MAT")
        decision = evaluate_spectra(saved_spectrum["configs"], bm["mat_certificate"]["residual"]["value"],
            configs=FROG_CONFIGS, sign_floor=FROG_SIGN_FLOOR, scope="task3_three_shift_saved_vector_finite_check")
        if decision != saved_spectrum["decision"] or not decision["numerical_check_passed"]:
            raise ValueError("supplemental spectrum does not satisfy the fixed policy")
        for artifact in saved_spectrum["artifacts"]:
            for name, digest in (("report_file", "report_sha256"), ("vectors_file", "vectors_sha256")):
                relative = artifact[name]
                if Path(relative).name != relative or sha(spectral_path.parent / relative) != artifact[digest]:
                    raise ValueError("supplemental spectrum artifact mismatch")
    methodology_path = sub / "evidence/methodology/methodology.json"
    return {"quality": quality, "active": active,
            "active_source_sha256": sha(sub / "evidence/active_learning/summary.json"),
            "e2e": end_to_end_statistics(active),
            "bands": {"axes": len(chosen), "selected": sum(v is not None for v in chosen.values()),
                      "selected_counts": dict(Counter(str(v) for v in chosen.values() if v is not None)),
                      "strict_evidence": "summary_only_not_recertified"},
            "best_match": bm, "inversion": inv, "best_match_spectrum": saved_spectrum,
            "methodology": load(methodology_path) if methodology_path.is_file() else None,
            "formal_stripes": {"bands": pattern["n_components"], "seed": stripes["seed"]}}


def claims(c):
    """Data-derived claims shared by Markdown and PDF; no historical count literals."""
    q, b, inv = c["quality"], c["best_match"], c["inversion"]
    parts = [f"独立测试集共 {q['totals']['endpoints']} 个 PDE 数值终态，其中 {q['totals']['steady']} 个完整稳态。"]
    for surface in SURFACES:
        r = q["test"][surface]; a = q["accuracy"][surface]
        parts.append(f"{surface} 完整稳态 {r['steady']}/{r['n']}；v1 全部终态 {fraction(a['v1']['all'])}、"
                     f"稳态子集 {fraction(a['v1']['steady'])}；v2 分别为 {fraction(a['v2']['all'])}、{fraction(a['v2']['steady'])}。")
    parts += ["球面稳态子集只有均匀态，不证明非平凡斑点或条纹稳态预测能力。",
              "测试参数在训练范围内，为新参数点插值测试；正式模型为 v1，研究中保留的改进模型为 v2。"]
    if any(q["independence"][s]["scope"] != "training_box_interpolation" for s in SURFACES):
        parts[-1] = "测试集中存在训练范围外参数，范围见质量分层表；正式模型 v1，研究保留模型 v2。"
    efficiency = []
    for e in c["e2e"].values():
        s = e["summary"]; ai = next(k for k in s if k.startswith("ai ") and "legacy" not in k)
        efficiency.append(f"青蛙 {ai} 完成记录 {s[ai]['complete']}/{s[ai]['runs']}；纯 PDE 中位积分步数 "
                          f"{s['pde (legacy)']['median_steps']:g}，{ai} 为 {s[ai]['median_steps']:g}，"
                          f"比值 {e['median_integration_step_ratio']:.2f}×。这是中位积分步数比，不是完整墙钟加速。"
                          "旧 PDE/v1 只有汇总，完成标志沿用历史单配置口径，未重新做正式谱验收。")
    bands = c["bands"]; values = bands["selected_counts"]
    selected = (f"选中均为 {next(iter(values))} 条带" if len(values) == 1 else f"条带数分布 {values}")
    formal = c["formal_stripes"]
    band_text = (f"新条带实验历史选择 {bands['selected']}/{bands['axes']}，{selected}。正式 {formal['bands']} 条带 seed{formal['seed']} "
                 "为独立协议结果。历史候选只保存谱汇总，不能升级为新严格政策通过，也不能与正式候选混算成功率。")
    inverse = (f"任务 3 在固定 F={b['F']:.7f}、k={b['k']:.7f} 下搜索 {b['candidates']} 个初值，"
               f"选中 {b['best']['layout']}：初始 {b['best']['n_init']} 斑，最终 {b['best']['n_final']} 斑，目标 {b['target']['n_final']} 斑。"
               f"同一特征距离 J 为 {b['previous_representative']['J']:.3f} → {b['best']['J']:.3f}。"
               f"反演估计的另一目标指标 {inv['estimate']['J']:.3f} 与此不同，不能混用。"
               "F/k 不因此改变；未执行非 AI 等预算多初值对照，不能证明 AI 的独立因果贡献。"
               "最佳匹配历史谱仅保留三组配置摘要。")
    spectrum = c.get("best_match_spectrum")
    if spectrum:
        decision = spectrum["decision"]
        inverse += (f"另有独立补充证据，实际保存三组共 {decision['checked_eigenpairs']} 个特征对及向量，绑定同一 MAT；"
                    f"最大采样实部 {decision['full_max_real_checked']:+.3e}，验收可重算保存向量的残差。"
                    "这不是全谱证明，也不表示本轮重新运行多初值搜索。")
    else:
        inverse += "缺少逐对残差和向量，不能凭摘要升级为严格谱通过。"
    return {"generalization": "".join(parts), "efficiency": "".join(efficiency),
            "bands": band_text, "inverse": inverse}


def source_fingerprint(root=ROOT):
    root = Path(root)
    entries = [(p.relative_to(root).as_posix(), sha(p)) for folder in ("scripts", "turing_rd", "tests")
               for p in sorted((root / folder).glob("*.py"))]
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()


def validate_test_record(record, root=ROOT):
    if (record.get("schema_version") != "executed-unittest-v1" or record.get("successful") is not True
            or record.get("failures") != 0 or record.get("errors") != 0 or record.get("skipped") != 0
            or not isinstance(record.get("tests_run"), int) or record["tests_run"] <= 0
            or record.get("source_fingerprint") != source_fingerprint(root)):
        raise ValueError("fresh successful executed test evidence matching current sources is required")
    return record["tests_run"]


def run_quality(out, root=ROOT):
    out = Path(out)
    if out.exists():
        raise FileExistsError("statistics output exists; historical results are preserved")
    q, predictions = recompute_quality(root)
    out.mkdir(parents=True)
    (out / "QUALITY_BREAKDOWN.json").write_text(json.dumps(q, ensure_ascii=False, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    (out / "QUALITY_BREAKDOWN.md").write_text(quality_markdown(q), encoding="utf-8")
    with (out / "PREDICTIONS.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    q = run_quality(args.out)
    print(json.dumps({"totals": q["totals"], "test": q["test"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
