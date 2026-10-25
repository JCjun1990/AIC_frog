"""Recompute retained F8/F9 evidence; missing grid fields stay unknown. No training/PDE."""
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
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from turing_rd.quality import quality_summary
from evidence_stats import spectral_assessment


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source(path, root):
    return {"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def inversion_summary(record):
    grid = record["grid"]; rows = grid["per_seed"]
    n = sum(len(row) for row in rows)
    if len(rows) != len(grid["points"]) or any(len(row) != len(grid["seeds"]) for row in rows):
        raise ValueError("inconsistent retained inverse loss dimensions")
    counts = {}
    for name, key in (("polished", "polished_fraction"), ("legacy_accepted", "steady_fraction")):
        product = n * grid[key]; count = round(product)
        if not math.isclose(product, count, rel_tol=16 * math.ulp(1.), abs_tol=0.):
            raise ValueError("aggregate fraction does not correspond to an integer count")
        counts[name] = count
    if sum(grid["status_counts"].values()) != n:
        raise ValueError("status aggregate count mismatch")
    landscape = []
    for point, values, stored_J, stored_use in zip(grid["points"], rows, grid["J"], grid["usable"]):
        median = statistics.median(values)
        usable = sum(v < record["type_penalty"] for v in values) / len(values)
        if not math.isclose(median, stored_J, rel_tol=1e-14, abs_tol=1e-14) or not math.isclose(usable, stored_use, rel_tol=1e-14, abs_tol=1e-14):
            raise ValueError("saved loss surface disagrees with retained per-seed distances")
        valid = [v for v in values if v < record["type_penalty"]]
        landscape.append({"F": point[0], "k": point[1], "legacy_median_with_penalties": median,
                          "legacy_conditional_spot_median": statistics.median(valid) if valid else None,
                          "legacy_usable_spot_fraction": usable, "strict_loss": None})
    return {"total": n, "counts_from_aggregate_only": counts,
            "extra_legacy_accepted": counts["legacy_accepted"] - counts["polished"],
            "legacy_accepted_fraction": counts["legacy_accepted"] / n,
            "raw_per_endpoint_states_rates_residuals_present": False,
            "strict_confirmed_from_retained_grid_evidence": 0, "actual_strict_steady_count": None,
            "strict_loss_surface": None, "strict_parameter_shift": None,
            "counts_scope": "integer counts cross-checked from aggregate fractions/statuses; not a new row-by-row is_steady replay",
            "missing": "grid_cache.npz with per-record status/rate/full residual and original fields is absent",
            "interpretation": "zero strict evidence coverage is not evidence of zero physical steady states",
            "status_counts": grid["status_counts"], "formal_estimate_unchanged": record["estimate"],
            "landscape": landscape}


def build(root=ROOT):
    root = Path(root); datasets, inputs = {}, []
    paths = {"v1_frog": "submission/task2_frog/ai_model/dataset.json",
             "v1_sphere5": "submission/task1_sphere/ai_model/dataset.json",
             "v2_frog": "submission/evidence/generalization/dense_dataset_frog.json",
             "v2_sphere5": "submission/evidence/generalization/dense_dataset_sphere5.json"}
    for name, rel in paths.items():
        path = root / rel; rows = load(path)["rows"]
        detail = quality_summary(rows); statuses = Counter(spectral_assessment(r) for r in rows)
        detail.update(source=source(path, root), legacy_spectral_status=dict(statuses),
                      legacy_stable_flag_count=sum(r.get("stable") is True for r in rows),
                      spectrum_known_failed=sum(statuses[s] for s in ("solver_not_converged", "invalid_eigenpairs")),
                      spectrum_failure_unknown=sum(statuses[s] for s in ("not_recorded", "attempted_insufficient_detail")))
        datasets[name] = detail; inputs.append(source(path, root))
    test_path = root / "submission/evidence/generalization/testset.json"
    test = load(test_path)["rows"]; inputs.append(source(test_path, root))
    tests = {s: quality_summary([r for r in test if r["surface"] == s]) for s in ("frog", "sphere5")}
    inv_path = root / "submission/task3_inversion/summary.json"; inverse = inversion_summary(load(inv_path)); inputs.append(source(inv_path, root))
    best_path = root / "submission/task3_inversion/best_match/summary.json"; best = load(best_path); inputs.append(source(best_path, root))
    if (best["F"], best["k"]) != (inverse["formal_estimate_unchanged"]["F"], inverse["formal_estimate_unchanged"]["k"]):
        raise ValueError("best match must retain the formal inversion parameters")
    return {"protocol": "methodology-evidence-v1", "sources": inputs, "datasets": datasets, "test": tests,
            "inversion": inverse, "best_match": {"layout": best["best"]["layout"], "seed": best["best"]["seed"],
                "initial_spots": best["best"]["n_init"], "final_spots": best["best"]["n_final"],
                "J_previous": best["previous_representative"]["J"], "J_best": best["best"]["J"],
                "F": best["F"], "k": best["k"], "mat_sha256": best["mat_certificate"]["mat_sha256"],
                "historical_spectra_have_pair_residuals": all("pairs" in r for r in best["stability"]["configs"]),
                "historical_spectra_have_vectors": False, "equal_budget_non_AI_ablation_executed": False},
            "limitations": ["Sphere steady test subset contains only uniform states; 16/16 is not nontrivial-pattern generalization.",
                "Finite validated samples are not full-spectrum proofs; unknown requested counts prevent strict protocol pass.",
                "Training quality uses saved original endpoint residuals, not newly certified float32 caches.",
                "Missing inverse grid fields prevent strict reclassification or quantification of F/k bias."]}


def render_report(data):
    lines = ["# F8/F9 方法学与证据范围", "", "本轮只复算已有记录，不训练、不重新执行反演网格，不修改正式参数。", "",
             "| 数据 | 总数 | 完整残差记录达标 | 未达标 | 严格有限谱通过 | 谱证据不足 | 混合终态类型参数点 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in data["datasets"].items():
        lines.append(f"| {name} | {row['total']} | {row['complete_steady']} | {row['not_complete_steady']} | "
                     f"{row['spectrum_checked_pass']} | {row['spectrum_insufficient']} | {len(row['mixed_terminal_type_points'])} |")
    inv, best = data["inversion"], data["best_match"]
    lines += ["", "谱证据不足不等于谱失败；旧 stable=True 不能替代特征对及配置验证。不同终态类型不单独证明稳态多稳态。",
              "球面完整稳态测试子集全部是均匀态，16/16 不证明非平凡斑点或条纹泛化能力。",
              f"反演保留 {inv['total']} 条逐种子损失；汇总交叉核对 polished={inv['counts_from_aggregate_only']['polished']}、"
              f"旧规则接受={inv['counts_from_aggregate_only']['legacy_accepted']}、差额={inv['extra_legacy_accepted']}。",
              "缺少逐条原始状态、rate 与完整残差，以上不是新执行的逐条稳态判定；严格证据覆盖为零，真实严格稳态数未知。",
              "已重算旧损失曲面的中位数和可用斑点比例；不能由此推导严格损失曲面、严格最优参数变化或 k 偏差的因果来源。",
              f"最佳匹配 {best['layout']}，初始 {best['initial_spots']} 斑、最终 {best['final_spots']} 斑；固定 F/k，"
              f"特征距离 J={best['J_previous']:.6f}→{best['J_best']:.6f}。",
              "AI 初值引导 + PDE/Newton + 多候选筛选是已有事实；没有非 AI 等预算对照，AI 独立因果贡献未证明。",
              "历史最佳匹配三组谱只有摘要；新增向量证据必须单独实际求谱、绑定原 MAT 并独立验收，不能补写到原记录。"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--out", type=Path, required=True); args = ap.parse_args()
    if args.out.exists():
        ap.error("output exists; keep historical records")
    result = build(); args.out.mkdir(parents=True)
    (args.out / "methodology.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (args.out / "methodology.md").write_text(render_report(result), encoding="utf-8")
    with (args.out / "legacy_loss_landscape.csv").open("x", newline="", encoding="utf-8") as stream:
        rows = result["inversion"]["landscape"]; writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"datasets": {k: {n:v[n] for n in ("total", "complete_steady", "not_complete_steady", "spectrum_checked_pass")} for k,v in result["datasets"].items()},
                      "inverse_counts": result["inversion"]["counts_from_aggregate_only"]}, indent=2))


if __name__ == "__main__":
    main()
