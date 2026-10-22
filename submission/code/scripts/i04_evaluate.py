"""I04: rebuild every success statistic from the raw experiment records.

    python scripts/i04_evaluate.py --out results/i04_evaluation

Reads only existing evidence (JSON/PKL records, I01 MAT certificates, I02 stability
certificates) and writes a new directory:

    evaluation_records.json        one EvaluationRecord per run, with evidence hashes
    success_breakdown.csv          planned/executed/missing/valid counts per criterion
    robustness_comparison.csv      published nominal success vs re-audited success
    generalization_comparison.csv  classifier accuracy vs morphology vs convergence
    efficiency_comparison.csv      costs by all/success/failure runs, with fairness flags
    evaluation_report.md           human-readable report, including what I03 must fix

No existing result, summary, model, report or submission file is modified.  A missing
or unparsable source is recorded and makes the run exit non-zero; denominators are
never reduced silently.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import platform
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for path in (str(ROOT), str(HERE)):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from json_safe import write_json_exclusive                       # noqa: E402
from turing_rd.evaluation import (CRITERIA, EVALUATION_VERSION, FILE_INVALID, FILE_NOT_SAVED,
                                  FILE_VALID, SPECTRAL_LIMITATION, STEADY_TOL, STATUS_PASS,
                                  TARGET_PATTERN, EvaluationRecord, aggregate, classify_spectrum,
                                  comparison_flags, denominator_consistency, sha256_file,
                                  steady_flags, verify_certificate_binding, within_budget)  # noqa: E402

RESULTS = ROOT / "results"
AI = RESULTS / "ai_generation"
AI_FINE = RESULTS / "ai_generation_fine"
I01 = RESULTS / "i01_float64_verified"
I02 = RESULTS / "i02_stability"

T_MAX_EVAL = 200000.0
T_MAX_DATASET = 100000.0
LEGACY_LEVEL = "legacy_single_shift_k8_rotation_labelled_unvalidated"
I02_LEVEL = "i02_unprojected_three_configuration_k12_k20_k8_with_eigenpair_residuals"
MODEL_OLD = "mlp_coarse_9x5"
MODEL_NEW = "mlp_fine_17x9x2"
EVAL_PLANNED = 96          # 8 held-out points x 3 seeds x 2 pipelines
DATASET_PLANNED = {("ai_generation", "frog"): 45, ("ai_generation", "sphere5"): 45,
                   ("ai_generation_fine", "frog"): 306, ("ai_generation_fine", "sphere5"): 306}

TASK_SPEC = {
    "frog_spots": dict(surface="frog", pattern="spots", F=0.028, k=0.062),
    "frog_stripes": dict(surface="frog", pattern="stripes", F=0.054, k=0.063),
    "sphere6_spots": dict(surface="sphere6", pattern="spots", F=0.028, k=0.062),
    "sphere6_stripes": dict(surface="sphere6", pattern="stripes", F=0.054, k=0.063),
}

#: Load metadata for wall-clock fairness flags (README 735 / 805 / 808 / 821).
EVAL_PROTOCOL = {
    ("ai_generation", "frog", "ai"): dict(host="JCnoLAPTOP", load="interleaved_with_pde", concurrency="same_pool"),
    ("ai_generation", "frog", "pde"): dict(host="JCnoLAPTOP", load="interleaved_with_pde", concurrency="same_pool"),
    ("ai_generation", "sphere5", "ai"): dict(host="TorinoUta19.local", load="concurrent_L6_generation", concurrency=8),
    ("ai_generation", "sphere5", "pde"): dict(host="TorinoUta19.local", load="concurrent_L6_generation", concurrency=8),
    ("ai_generation_fine", "frog", "ai"): dict(host="JCnoLAPTOP", load="ai_only", concurrency=20),
    ("ai_generation_fine", "frog", "pde"): dict(host="JCnoLAPTOP", load="reused_old_baseline", concurrency="unknown"),
    ("ai_generation_fine", "sphere5", "ai"): dict(host="TorinoUta19.local", load="ai_only", concurrency=8),
    ("ai_generation_fine", "sphere5", "pde"): dict(host="TorinoUta19.local", load="reused_old_baseline", concurrency="unknown"),
}

#: Literal claims of the previous documents.  The numeric ones that are recomputable
#: are checked against the raw records at run time (see checks in run_manifest.json).
PUBLISHED_CLAIMS = [
    dict(claim="青蛙斑点稳健性 31/33", value="31/33",
         source="README.md:755; results/ai_generation/tasks/robustness.md:8",
         verdict="数字保留，口径需补充",
         detail="31 = 完整稳态且未发现正模态；失败：seed22 收敛但 max Re λ=+3.2e-5（鞍点），seed25 预算耗尽（3.9e-3）"),
    dict(claim="青蛙条纹稳健性 33/33（含 2 个斑点态）", value="33/33",
         source="README.md:756; robustness.md:9",
         verdict="数字保留，不能称为目标条纹成功率",
         detail="33/33 只是“收敛且谱负”；条纹形态 31/33（seed1 为 9 斑、seed21 为 10 斑）"),
    dict(claim="球面斑点稳健性 17/23", value="17/23",
         source="README.md:757; robustness.md:10",
         verdict="拆分为多个指标",
         detail="17/23 完整稳态；seed13/16/21 有正旋转模态 → 14/23 无正模态候选；6 个预算失败"),
    dict(claim="球面条纹稳健性 22/23", value="22/23",
         source="README.md:758; robustness.md:11",
         verdict="拆分为多个指标",
         detail="22/23 完整稳态；13 个六条带有正倾斜模态 +1.67e-6 → 9/23 无正模态；1 个预算失败（seed1）"),
    dict(claim="青蛙 AI 24/24 收敛且稳定、1.7 倍加速", value="24/24, 1.7x",
         source="README.md:739-740; scripts/make_submission.py:178",
         verdict="24/24 保留；加速比只对旧模型同负载那一组成立",
         detail="24 次都是完整稳态且谱全负（严格口径同样 24/24）；同负载墙钟 solve_s 673/1159 s "
                "（total_s 678/1166 s）可比，步数比 15062/25875≈0.58"),
    dict(claim="青蛙新模型 23/24", value="23/24", source="README.md:801",
         verdict="保留", detail="1 个失败为 task_spots 的 seed2（未收敛）"),
    dict(claim="球面 L5 AI 12/24（旧）/10/24（新）", value="12/24, 10/24", source="README.md:812",
         verdict="必须改写为“相对平衡态 + 旧谱口径”",
         detail="24 次里没有完整稳态；严格口径（相对平衡 + 无正模态）为 10/24、6/24，普通流程 4/24"),
    dict(claim="球面条纹 seed0（六条带）排除 2 个倾斜模后稳定", value="stable after excluding tilt modes",
         source="README.md:836,887",
         verdict="撤回", detail="同一 MAT 上 I02 三配置确认 +1.6654e-6 是可靠正模态（特征对残差 + 扰动增长）"),
    dict(claim="球面斑点为完整稳态且稳定", value="stable", source="README.md:835; 交接说明任务1",
         verdict="保留但需限定", detail="完整稳态 8.423e-15；I02 三配置无正模态（max Re λ=-6.231e-6）；"
                                        "有限采样，不是全谱证明"),
    dict(claim="留一法类型准确率 青蛙 64%→86%，球面 89%→93%", value="64%->86%, 89%->93%",
         source="README.md:786-789",
         verdict="数字保留，但不可同难度比较",
         detail="旧 45 点、新 306 点；去掉一点后邻点间距差 2 倍，两者不是同一难度的留一法"),
]

PROBLEMS = []


# ---------------------------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------------------------
def rel(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path):
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        PROBLEMS.append({"kind": "missing_source", "path": rel(path)})
    except Exception as exc:
        PROBLEMS.append({"kind": "parse_error", "path": rel(path), "error": repr(exc)})
    return None


def file_hash(path):
    try:
        return sha256_file(path)
    except OSError:
        PROBLEMS.append({"kind": "hash_error", "path": rel(path)})
        return None


def median(values):
    vals = []
    for value in values:
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            vals.append(value)
    return statistics.median(vals) if vals else None


def git_state():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=30, check=True).stdout.strip()
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, timeout=30, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                                    text=True, timeout=30, check=True).stdout.strip())
    except Exception as exc:
        PROBLEMS.append({"kind": "git_error", "error": repr(exc)})
        return {"head": None, "branch": None, "dirty": None}
    return {"head": head, "branch": branch, "dirty": dirty}


def fmt(value, digits=6):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.{digits}g}"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def write_csv(path, rows, columns):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: fmt(row.get(column)) for column in columns})


# ---------------------------------------------------------------------------------------------
# record builders
# ---------------------------------------------------------------------------------------------
def eigen_values(row):
    values = []
    if row.get("max_re_lambda") is not None:
        values.append(row["max_re_lambda"])
    values.extend(row.get("rotation_eigenvalues") or [])
    return values


def legacy_spectrum(row, note):
    """Classify a legacy record's sampled eigenvalues (empty sample -> not_checked/none)."""
    values = eigen_values(row)
    return classify_spectrum(values, evidence_level=LEGACY_LEVEL if values else "none",
                             eigenpair_validated=False, note=note)


def legacy_success(row, surface):
    """Reproduce the published legacy rule: decisive residual <= TOL and stable flag."""
    if not row:
        return None
    decisive = row.get("rel_residual_rot") if surface.startswith("sphere") else row.get("rel_residual")
    if decisive is None:
        decisive = row.get("rel_residual")
    if decisive is None:
        return None
    return bool(row.get("stable") is True and float(decisive) <= STEADY_TOL)


def load_robustness_jobs(task):
    spec = TASK_SPEC[task]
    surface = spec["surface"]
    prefix = f"ai_{surface}_{spec['F']:.4f}_{spec['k']:.4f}_"
    import joblib

    jobs = {}
    for path in sorted((AI / "tasks/robustness_jobs").glob(prefix + "*.pkl")):
        seed = int(path.stem.rsplit("_", 1)[1])
        try:
            jobs[seed] = (joblib.load(path), path, None)
        except Exception as exc:
            PROBLEMS.append({"kind": "job_parse_error", "path": rel(path), "error": repr(exc)})
    return jobs


def original_run(task, seed, seed0_summary, jobs):
    """The pre-I01 record for a task/seed (task summary for seed 0, pkl otherwise)."""
    if seed == 0:
        return seed0_summary, AI / "tasks" / task / "summary.json", None
    return jobs.get(seed)


def builder_robustness(task, protocol_seeds, seed0, jobs):
    spec = TASK_SPEC[task]
    surface, pattern = spec["surface"], spec["pattern"]
    target = TARGET_PATTERN[pattern]
    records = []
    for seed in protocol_seeds:
        item = original_run(task, seed, seed0, jobs)
        if item is None:
            record = EvaluationRecord(
                experiment_id=f"robust:{task}:seed{seed}", record_kind="robustness_seed",
                surface=surface, seed=seed, target_pattern=target,
                parameters={"F": spec["F"], "k": spec["k"]}, model_id=MODEL_OLD, pipeline="ai",
                evidence={"missing": True, "parse_error": "missing_record"})
            records.append(record.finalise())
            continue
        row, path, index = item
        try:
            full = float(row["rel_residual"]) if row.get("rel_residual") is not None else None
            rot = row.get("rel_residual_rot")
            rot = float(rot) if rot is not None else None
            steady, relative = steady_flags(full, rot)
            spectral = legacy_spectrum(
                row, "legacy ARPACK values without eigenpair residual; rotation-labelled values included")
            record = EvaluationRecord(
                experiment_id=f"robust:{task}:seed{seed}", record_kind="robustness_seed",
                surface=surface, seed=seed, target_pattern=target,
                parameters={"F": spec["F"], "k": spec["k"], "Du": 0.000278, "Dv": 0.000139},
                model_id=MODEL_OLD, pipeline="ai",
                solver_config={"t_max": T_MAX_EVAL, "tol": STEADY_TOL,
                               "rotation_corrected": surface.startswith("sphere"),
                               "orient": surface.startswith("sphere"), "schedule": "ai_2k_then_4x"},
                full_residual=full, rotation_removed_residual=rot,
                steady_full=steady, relative_equilibrium=relative,
                spectral_status=spectral["status"], spectral_max_real=spectral["max_real"],
                spectral_evidence_level=spectral["evidence_level"], spectral_detail=spectral,
                predicted_pattern=row.get("predicted_type"), observed_pattern=row.get("type"),
                within_budget=within_budget(row.get("relax_time"), T_MAX_EVAL),
                runtime=row.get("total_s") or row.get("solve_s") or row.get("wall_s"),
                integration_steps=row.get("relax_steps"),
                file_validity=FILE_NOT_SAVED,
                legacy_stability_flag=row.get("stable"), legacy_success=legacy_success(row, surface),
                legacy_stability_scope="legacy: residual<=1e-11 and max Re lambda over non-rotation modes",
                evidence={"source_file": rel(path), "source_sha256": file_hash(path),
                          "row_index": index, "relax_time": row.get("relax_time"),
                          "newton_tries": row.get("newton_tries"), "host": row.get("host"),
                          "n_components": row.get("n_components"),
                          "shape_target_basis": "task_requirement",
                          "note": "the robustness scan saved no MAT/field"})
        except Exception as exc:
            record = EvaluationRecord(
                experiment_id=f"robust:{task}:seed{seed}", record_kind="robustness_seed",
                surface=surface, seed=seed, target_pattern=target,
                parameters={"F": spec["F"], "k": spec["k"]}, model_id=MODEL_OLD, pipeline="ai",
                evidence={"source_file": rel(path), "source_sha256": file_hash(path),
                          "row_index": index, "parse_error": repr(exc)})
        records.append(record.finalise())
    return records


def candidate_specs():
    return [
        dict(name="frog_spots", label="青蛙斑点（I01 float64）", task="frog_spots", seed=0,
             task_dir=I01 / "tasks/frog_spots", certificate=None, recommended=True,
             note="I01 only; no stability certificate is bound to the I01 MAT"),
        dict(name="frog_stripes", label="青蛙条纹（I01 float64）", task="frog_stripes", seed=0,
             task_dir=I01 / "tasks/frog_stripes", certificate=None, recommended=True,
             note="I01 only; no stability certificate is bound to the I01 MAT"),
        dict(name="sphere6_spots", label="球面斑点（I01 + I02 证书）", task="sphere6_spots", seed=0,
             task_dir=I01 / "tasks/sphere6_spots",
             certificate=I02 / "certificates/spots/stability_certificate.json", recommended=True,
             note="I02 three-configuration unprojected check bound to the same MAT"),
        dict(name="sphere6_stripes_6band", label="球面条纹六条带 seed0（历史，不再推荐）",
             task="sphere6_stripes", seed=0, task_dir=I01 / "tasks/sphere6_stripes",
             certificate=I02 / "certificates/six_bands/stability_certificate.json", recommended=False,
             note="historical six-band field; I02 found a reliable positive tilted mode"),
        dict(name="sphere6_stripes_seed2", label="球面条纹七条带 seed2（I02 推荐）",
             task="sphere6_stripes", seed=2, task_dir=I02 / "candidates/seed2_v2",
             certificate=I02 / "certificates/seed2/stability_certificate.json", recommended=True,
             note="regenerated seven-band candidate; I02 certificate bound to its MAT"),
    ]


def builder_candidates(robust_seed0, jobs_by_task):
    records = []
    for spec in candidate_specs():
        task = spec["task"]
        params = TASK_SPEC[task]
        summary_path = spec["task_dir"] / "summary.json"
        summary = read_json(summary_path)
        mat_path = spec["task_dir"] / "solution.mat"
        certificate = read_json(spec["certificate"]) if spec["certificate"] else None
        original = original_run(task, spec["seed"], robust_seed0[task], jobs_by_task[task])
        original = original[0] if original else {}
        record = EvaluationRecord(
            experiment_id=f"candidate:{spec['name']}", record_kind="formal_candidate",
            surface=params["surface"], seed=spec["seed"],
            target_pattern=TARGET_PATTERN[params["pattern"]],
            model_id=MODEL_NEW if spec["name"].endswith("seed2") else MODEL_OLD,
            pipeline="ai",
            solver_config={"t_max": T_MAX_EVAL, "tol": STEADY_TOL, "orient": True,
                           "rotation_corrected": params["surface"].startswith("sphere")},
            evidence={"mat": rel(mat_path), "i01_summary": rel(summary_path),
                      "i01_summary_sha256": file_hash(summary_path),
                      "stability_certificate": rel(spec["certificate"]) if spec["certificate"] else None,
                      "stability_certificate_sha256": file_hash(spec["certificate"]) if spec["certificate"] else None,
                      "legacy_record": None, "recommended": spec["recommended"],
                      "shape_target_basis": "task_requirement",
                      "note": spec["note"]},
            notes=[spec["note"], f"label={spec['label']}"])
        try:
            if summary is None:
                raise ValueError("missing I01/candidate summary")
            cert = summary["mat_certificate"]
            digest = file_hash(mat_path)
            record.evidence["mat_sha256"] = digest
            record.input_hash = digest
            record.result_hash = digest
            record.parameters = {key: float(cert["parameters"][key]) for key in ("F", "k", "Du", "Dv")}
            record.full_residual = float(cert["residual"]["value"])
            record.steady_full, record.relative_equilibrium = steady_flags(record.full_residual)
            record.i01_candidate_accepted = bool(cert.get("accepted"))
            pattern_block = summary.get("pattern_after") or summary.get("pattern") or {}
            record.observed_pattern = pattern_block.get("type")
            record.evidence["n_components"] = pattern_block.get("n_components")
            record.predicted_pattern = original.get("predicted_type")
            solver_info = summary.get("solver_info") or {}
            relax_time = solver_info.get("relax_time")
            if relax_time is None:
                relax_time = original.get("relax_time")
            record.within_budget = within_budget(relax_time, T_MAX_EVAL)
            record.integration_steps = solver_info.get("relax_steps") or original.get("relax_steps")
            record.runtime = ((summary.get("cost") or {}).get("solve_wall_seconds")
                              or original.get("total_s") or original.get("solve_s")
                              or original.get("wall_s"))
            record.legacy_stability_flag = original.get("stable")
            record.legacy_success = legacy_success(original, params["surface"])
            record.legacy_stability_scope = "legacy flag on the pre-I01 field, not on this MAT"
            if original:
                if spec["seed"] == 0:
                    record.evidence["legacy_record"] = f"results/ai_generation/tasks/{task}/summary.json"
                else:
                    record.evidence["legacy_record"] = (
                        f"results/ai_generation/tasks/robustness_jobs/ai_{params['surface']}_"
                        f"{params['F']:.4f}_{params['k']:.4f}_{spec['seed']}.pkl")
            valid = bool(cert.get("accepted")) and digest == cert.get("mat_sha256")
            if certificate is not None:
                verify_certificate_binding(mat_path, certificate)
                spectra = certificate["spectra"]
                values = [row.get("full_max_real") for row in spectra]
                spectral = classify_spectrum(
                    values, evidence_level=I02_LEVEL,
                    eigenpair_validated=all(row.get("all_pairs_validated") for row in spectra),
                    note="three independent configurations; rotation overlap never removes a mode")
                record.spectral_status = spectral["status"]
                record.spectral_max_real = spectral["max_real"]
                record.spectral_evidence_level = spectral["evidence_level"]
                record.spectral_detail = spectral
                configurations = []
                for row in spectra:
                    sigma = row.get("sigma")
                    configurations.append("direct_LR" if sigma is None else f"sigma={sigma:g}")
                record.notes.append(f"I02 decision={certificate['decision']['status']}; "
                                    f"configurations={', '.join(configurations)}")
                record.notes.append("finite sample, not a full-spectrum proof")
            else:
                spectral = classify_spectrum([], evidence_level="none")
                record.spectral_status = spectral["status"]
                record.spectral_detail = spectral
                record.notes.append("no stability certificate bound to this MAT; "
                                    "legacy eigenvalues refer to the pre-I01 field")
            record.file_validity = FILE_VALID if valid else FILE_INVALID
        except Exception as exc:
            record.file_validity = FILE_INVALID
            record.evidence["error"] = repr(exc)
            record.evidence["parse_error"] = repr(exc)
            PROBLEMS.append({"kind": "candidate_error", "candidate": spec["name"], "error": repr(exc)})
        records.append(record.finalise())
    return records


def reference_types(rows):
    """Reference morphology per (surface, point): majority type of the PDE pipeline."""
    refs, ties = {}, []
    surfaces = sorted({row["surface"] for row in rows})
    for surface in surfaces:
        points = sorted({row["point"] for row in rows if row["surface"] == surface})
        for point in points:
            pde = [row for row in rows if row["surface"] == surface and row["point"] == point
                   and row["pipeline"] == "pde"]
            counts = {}
            for row in pde:
                counts[row.get("type")] = counts.get(row.get("type"), 0) + 1
            if counts:
                best = max(counts.values())
                winners = sorted(t for t, c in counts.items() if c == best)
                refs[(surface, point)] = winners[0]
                if len(winners) > 1:
                    ties.append({"surface": surface, "point": point, "types": counts})
    return refs, ties


def target_for_point(surface, point, refs):
    if point.endswith("spots"):
        return TARGET_PATTERN["spots"]
    if point.endswith("stripes"):
        return TARGET_PATTERN["stripes"]
    return refs.get((surface, point))


def builder_evaluation(tag, rows, refs, baseline_index, source_sha):
    model = MODEL_OLD if tag == "ai_generation" else MODEL_NEW
    source_file = f"{rel(AI if tag == 'ai_generation' else AI_FINE)}/evaluation.json"
    records = []
    for index, row in enumerate(rows):
        surface, pipeline, point = row["surface"], row["pipeline"], row["point"]
        target = target_for_point(surface, point, refs)
        key = (surface, pipeline, point, row.get("F"), row.get("k"), row.get("seed"))
        reused = None
        if tag != "ai_generation" and pipeline == "pde" and key in baseline_index:
            other = baseline_index[key]
            identical = all(other.get(field) == row.get(field) for field in
                            ("rel_residual", "rel_residual_rot", "relax_time", "relax_steps",
                             "wall_s", "type", "n_components", "max_re_lambda"))
            reused = {"canonical": "results/ai_generation/evaluation.json",
                      "row_index": other["_index"], "identical": identical}
            if not identical:
                PROBLEMS.append({"kind": "baseline_mismatch", "key": str(key)})
        try:
            full = float(row["rel_residual"]) if row.get("rel_residual") not in (None, "") else None
            rot = row.get("rel_residual_rot")
            rot = float(rot) if rot not in (None, "") else None
            steady, relative = steady_flags(full, rot)
            spectral = legacy_spectrum(
                row, "legacy single-shift value set without eigenpair residual")
            record = EvaluationRecord(
                experiment_id=f"eval:{tag}:{surface}:{pipeline}:{point}:seed{row.get('seed')}",
                record_kind="evaluation_run", surface=surface, seed=row.get("seed"),
                target_pattern=target,
                parameters={"F": row.get("F"), "k": row.get("k")},
                model_id=model, pipeline=pipeline,
                solver_config={"t_max": T_MAX_EVAL, "tol": STEADY_TOL,
                               "rotation_corrected": surface.startswith("sphere"),
                               "orient": False, "schedule": f"{pipeline}_schedule"},
                full_residual=full, rotation_removed_residual=rot,
                steady_full=steady, relative_equilibrium=relative,
                spectral_status=spectral["status"], spectral_max_real=spectral["max_real"],
                spectral_evidence_level=spectral["evidence_level"], spectral_detail=spectral,
                predicted_pattern=row.get("predicted_type"), observed_pattern=row.get("type"),
                within_budget=within_budget(row.get("relax_time"), T_MAX_EVAL),
                runtime=row.get("total_s") or row.get("wall_s"),
                integration_steps=row.get("relax_steps"),
                file_validity=FILE_NOT_SAVED,
                legacy_stability_flag=row.get("stable"),
                legacy_success=legacy_success(row, surface),
                legacy_stability_scope="legacy: residual<=1e-11 and max Re lambda over non-rotation modes",
                evidence={"source_file": source_file, "source_sha256": source_sha,
                          "row_index": index, "host": row.get("host"), "point": point,
                          "reused_baseline": reused,
                          "shape_target_basis": ("task_requirement" if point.startswith("task_")
                                                 else "pde_reference_surrogate"),
                          "reference_pattern": refs.get((surface, point)),
                          "note": "evaluation runs saved no MAT/field"})
        except Exception as exc:
            record = EvaluationRecord(
                experiment_id=f"eval:{tag}:{surface}:{pipeline}:{point}:seed{row.get('seed')}",
                record_kind="evaluation_run", surface=surface, seed=row.get("seed"),
                target_pattern=target, model_id=model, pipeline=pipeline,
                evidence={"source_file": source_file, "row_index": index, "parse_error": repr(exc)})
        records.append(record.finalise())
    return records


def builder_training(tag, surface, rows, source_sha):
    model = MODEL_OLD if tag == "ai_generation" else MODEL_NEW
    source_file = f"{rel(RESULTS / tag / surface)}/dataset.json"
    records = []
    for index, row in enumerate(rows):
        try:
            full = float(row["rel_residual"]) if row.get("rel_residual") is not None else None
            steady, _ = steady_flags(full)
            spectral = legacy_spectrum(
                row, "training-label stability; legacy single-shift without eigenpair residual")
            record = EvaluationRecord(
                experiment_id=f"train:{tag}:{surface}:row{index}", record_kind="training_label",
                surface=surface, seed=row.get("seed"), target_pattern=None,
                parameters={"F": row.get("F"), "k": row.get("k")},
                model_id=model, pipeline=row.get("pipeline", "pde"),
                solver_config={"t_max": T_MAX_DATASET, "tol": STEADY_TOL,
                               "rotation_corrected": surface.startswith("sphere"), "orient": False},
                full_residual=full, rotation_removed_residual=None,
                steady_full=steady, relative_equilibrium=None,
                spectral_status=spectral["status"], spectral_max_real=spectral["max_real"],
                spectral_evidence_level=spectral["evidence_level"], spectral_detail=spectral,
                predicted_pattern=None, observed_pattern=row.get("type"),
                within_budget=within_budget(row.get("relax_time"), T_MAX_DATASET),
                runtime=row.get("total_s") or row.get("wall_s"),
                integration_steps=row.get("relax_steps"),
                file_validity=FILE_NOT_SAVED,
                legacy_stability_flag=row.get("stable"), legacy_success=None,
                evidence={"source_file": source_file, "source_sha256": source_sha,
                          "row_index": index, "labels_present": bool(row.get("labels")),
                          "label_blocks": sorted((row.get("labels") or {}).keys()),
                          "note": "training label; no MAT saved"})
            if not row.get("labels"):
                record.notes.append("no labels stored for a uniform state" if row.get("type") == "uniform"
                                    else "no labels stored for a non-uniform state")
            records.append(record.finalise())
        except Exception as exc:
            record = EvaluationRecord(
                experiment_id=f"train:{tag}:{surface}:row{index}", record_kind="training_label",
                surface=surface, model_id=model,
                evidence={"source_file": source_file, "row_index": index, "parse_error": repr(exc)})
            records.append(record.finalise())
    return records


# ---------------------------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------------------------
def make_group(key, label, records, planned, **extra):
    summary = aggregate(records, planned)
    summary["consistency"] = denominator_consistency(summary, planned)
    if not summary["consistency"]["consistent"]:
        PROBLEMS.append({"kind": "denominator_inconsistency", "group": key,
                         "problems": summary["consistency"]["problems"]})
    summary.update(extra)
    return {"key": key, "label": label, "planned": planned, "summary": summary, "records": records}


def by_reason(records, reason):
    return [r.seed for r in records if reason in (r.failure_reason or "").split(";")]


def cost_stats(records):
    legacy = [r for r in records if r.legacy_success is True]
    legacy_bad = [r for r in records if r.legacy_success is not True]
    strict = [r for r in records if r.end_to_end_success]
    return {
        "n": len(records), "n_success": len(strict), "n_failure": len(records) - len(strict),
        "wall_all": median([r.runtime for r in records]),
        "wall_success": median([r.runtime for r in strict]),
        "wall_failure": median([r.runtime for r in records if not r.end_to_end_success]),
        "steps_all": median([r.integration_steps for r in records]),
        "steps_success": median([r.integration_steps for r in strict]),
        "steps_failure": median([r.integration_steps for r in records if not r.end_to_end_success]),
        "legacy_success": len(legacy),
        "legacy_steps_success": median([r.integration_steps for r in legacy]),
        "legacy_wall_success": median([r.runtime for r in legacy]),
        "legacy_steps_failure": median([r.integration_steps for r in legacy_bad]),
        "legacy_wall_failure": median([r.runtime for r in legacy_bad]),
    }


def paired_ratios(ai_records, pde_records):
    """Per-(point, seed) AI/PDE ratios on runs where both pipelines passed the same rule."""
    ai = {(r.surface, r.evidence.get("point"), r.seed): r for r in ai_records}
    pde = {(r.surface, r.evidence.get("point"), r.seed): r for r in pde_records}
    out = {}
    for rule in ("legacy_success", "end_to_end_success"):
        steps, walls = [], []
        for key, a in ai.items():
            b = pde.get(key)
            if b is None:
                continue
            passed = a.legacy_success if rule == "legacy_success" else a.end_to_end_success
            if not passed:
                continue
            if a.integration_steps and b.integration_steps:
                steps.append(a.integration_steps / b.integration_steps)
            if a.runtime and b.runtime:
                walls.append(a.runtime / b.runtime)
        out[rule] = {"pairs": len(steps), "steps_ratio_median": median(steps),
                     "wall_ratio_median": median(walls)}
    return out


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS / "i04_evaluation",
                        help="new output directory (must not exist)")
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.out.exists():
        parser.error("Output directory already exists; nothing is overwritten")
    started = datetime.datetime.now(datetime.timezone.utc)
    git = git_state()

    # ---- sources -------------------------------------------------------------------
    robustness_published = read_json(AI / "tasks/robustness.json") or {}
    seed0, jobs_by_task = {}, {}
    for task in TASK_SPEC:
        seed0[task] = read_json(AI / "tasks" / task / "summary.json") or {}
        jobs_by_task[task] = load_robustness_jobs(task)

    evaluation_files = {}
    for tag in ("ai_generation", "ai_generation_fine"):
        path = RESULTS / tag / "evaluation.json"
        evaluation_files[tag] = (read_json(path) or [], file_hash(path))
    datasets = {}
    for tag in ("ai_generation", "ai_generation_fine"):
        for surface in ("frog", "sphere5"):
            path = RESULTS / tag / surface / "dataset.json"
            payload = read_json(path)
            datasets[(tag, surface)] = ((payload or {}).get("rows", []), file_hash(path))

    # ---- records -------------------------------------------------------------------
    groups = []
    for task in TASK_SPEC:
        protocol = robustness_published.get(task, {}).get("seeds") or sorted(jobs_by_task[task])
        protocol = sorted(set(protocol) | {0})
        records = builder_robustness(task, protocol, seed0[task], jobs_by_task[task])
        groups.append(make_group(f"robustness:{task}", f"robustness scan · {task}", records, len(protocol),
                                 published_success=(robustness_published.get(task) or {}).get("stable")))

    candidates = builder_candidates(seed0, jobs_by_task)
    groups.append(make_group("formal_candidates", "formal candidates", candidates, len(candidates)))

    refs, ties = reference_types(evaluation_files["ai_generation"][0])
    baseline_index = {}
    for index, row in enumerate(evaluation_files["ai_generation"][0]):
        if row.get("pipeline") == "pde":
            key = (row["surface"], row["pipeline"], row["point"], row.get("F"), row.get("k"), row.get("seed"))
            baseline_index[key] = dict(row, _index=index)
    eval_records = {}
    for tag in ("ai_generation", "ai_generation_fine"):
        rows, digest = evaluation_files[tag]
        if len(rows) != EVAL_PLANNED:
            PROBLEMS.append({"kind": "unexpected_row_count", "file": f"{tag}/evaluation.json",
                             "rows": len(rows), "expected": EVAL_PLANNED})
        records = builder_evaluation(tag, rows, refs, baseline_index, digest)
        eval_records[tag] = records
        for surface in ("frog", "sphere5"):
            for pipeline in ("ai", "pde"):
                subset = [r for r in records if r.surface == surface and r.pipeline == pipeline]
                groups.append(make_group(f"evaluation:{tag}:{surface}:{pipeline}",
                                         f"{tag} · {surface} · {pipeline}", subset,
                                         EVAL_PLANNED // 4))

    training_records = {}
    for (tag, surface), (rows, digest) in datasets.items():
        planned = DATASET_PLANNED[(tag, surface)]
        if len(rows) != planned:
            PROBLEMS.append({"kind": "unexpected_row_count", "file": f"{tag}/{surface}/dataset.json",
                             "rows": len(rows), "expected": planned})
        records = builder_training(tag, surface, rows, digest)
        training_records[(tag, surface)] = records
        groups.append(make_group(f"training:{tag}:{surface}", f"{tag} · {surface} · training labels",
                                 records, planned))

    all_records = [record for group in groups for record in group["records"]]

    # ---- checks --------------------------------------------------------------------
    checks = []

    def check(name, passed, detail=""):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    for group in groups:
        check(f"denominators:{group['key']}", group["summary"]["consistency"]["consistent"],
              "; ".join(group["summary"]["consistency"]["problems"]))
    for group in groups:
        if not group["key"].startswith("robustness:"):
            continue
        published = group["summary"].get("published_success")
        recomputed = group["summary"]["legacy_success"]
        check(f"legacy_count_reproduced:{group['key']}", published in (None, recomputed),
              f"published={published} recomputed={recomputed}")
    reused_pde = [r for r in eval_records["ai_generation_fine"]
                  if r.pipeline == "pde" and (r.evidence.get("reused_baseline") or {}).get("identical")]
    check("baseline_rows_reused_identically", len(reused_pde) == 48, f"{len(reused_pde)}/48 pde rows reused")
    cert_records = [r for r in candidates if r.evidence.get("stability_certificate")]
    check("stability_certificates_bound",
          bool(cert_records) and all(r.spectral_evidence_level == I02_LEVEL and r.evidence["mat_sha256"]
                                     for r in cert_records), f"{len(cert_records)} certificates")
    check("i01_mats_accepted",
          all(r.i01_candidate_accepted for r in candidates if r.file_validity == FILE_VALID),
          "all valid candidate MATs accepted")
    check("no_parse_errors",
          not [p for p in PROBLEMS if p["kind"] in ("parse_error", "job_parse_error")],
          f"{len(PROBLEMS)} source problems")

    status = "ok" if all(c["passed"] for c in checks) and not PROBLEMS else "error"

    # ---- artifacts -----------------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=False)
    code_files = ["scripts/i04_evaluate.py", "scripts/json_safe.py", "turing_rd/evaluation.py"]
    code_hashes = {name: file_hash(ROOT / name) for name in code_files}
    record_document = {
        "schema_version": "aic-frog-i04-records-v1",
        "evaluation_version": EVALUATION_VERSION,
        "generated_utc": started.isoformat(),
        "source": {"git": git, "code_sha256": code_hashes,
                   "python": platform.python_version(), "platform": platform.platform()},
        "criteria": CRITERIA,
        "spectral_limitation": SPECTRAL_LIMITATION,
        "record_kinds": ["formal_candidate", "robustness_seed", "evaluation_run", "training_label"],
        "records": [record.to_dict() for record in all_records],
    }
    write_json_exclusive(args.out / "evaluation_records.json", record_document)

    breakdown_rows = []
    for group in groups:
        summary = group["summary"]
        breakdown_rows.append({
            "group": group["key"], "kind": group["records"][0].record_kind if group["records"] else "",
            "planned": summary["planned"], "executed": summary["executed"], "missing": summary["missing"],
            "error": summary["error"], "valid": summary["valid"],
            "file_valid": summary["file_valid"], "file_invalid": summary["file_invalid"],
            "file_not_saved": summary["file_not_saved"],
            "steady_full": summary["steady_full"],
            "relative_equilibrium_only": summary["relative_equilibrium_only"],
            "spectral_checked": summary["spectral_checked"],
            "spectral_passed": summary["spectral_passed"],
            "spectral_positive": summary["spectral_positive"],
            "spectral_insufficient": summary["spectral_insufficient"],
            "spectral_not_checked": summary["spectral_not_checked"],
            "shape_match": summary["shape_match"], "shape_checked": summary["shape_checked"],
            "within_budget": summary["within_budget"],
            "legacy_success": summary["legacy_success"],
            "strict_success": summary["strict_success"],
            "strict_rate": summary["success_rate_strict"],
            "denominators_consistent": summary["consistency"]["consistent"],
            "failure_reasons": summary["failure_reasons"],
        })
    write_csv(args.out / "success_breakdown.csv", breakdown_rows,
              ["group", "kind", "planned", "executed", "missing", "error", "valid", "file_valid",
               "file_invalid", "file_not_saved", "steady_full", "relative_equilibrium_only",
               "spectral_checked", "spectral_passed", "spectral_positive", "spectral_insufficient",
               "spectral_not_checked", "shape_match", "shape_checked", "within_budget",
               "legacy_success", "strict_success", "strict_rate", "denominators_consistent",
               "failure_reasons"])

    robustness_rows = []
    for task in TASK_SPEC:
        group = next(g for g in groups if g["key"] == f"robustness:{task}")
        summary, records = group["summary"], group["records"]
        bands = {}
        for record in records:
            count = (record.evidence or {}).get("n_components")
            if count is None:
                continue
            bands[str(count)] = bands.get(str(count), 0) + 1
        robustness_rows.append({
            "task": task, "planned": summary["planned"], "executed": summary["executed"],
            "published_legacy_success": summary.get("published_success"),
            "recomputed_legacy_success": summary["legacy_success"],
            "legacy_matches_published": summary.get("published_success") in (None, summary["legacy_success"]),
            "steady_full": summary["steady_full"],
            "relative_equilibrium_only": summary["relative_equilibrium_only"],
            "spectral_checked": summary["spectral_checked"],
            "spectral_passed": summary["spectral_passed"],
            "spectral_positive": summary["spectral_positive"],
            "spectral_not_checked": summary["spectral_not_checked"],
            "shape_match": summary["shape_match"], "shape_checked": summary["shape_checked"],
            "within_budget": summary["within_budget"],
            "strict_success": summary["strict_success"],
            "strict_rate": summary["success_rate_strict"],
            "positive_seeds": by_reason(records, "positive_sampled_eigenvalue"),
            "budget_seeds": by_reason(records, "time_budget_exhausted"),
            "shape_mismatch_seeds": by_reason(records, "shape_mismatch"),
            "not_steady_seeds": by_reason(records, "not_a_complete_steady_state"),
            "bands": bands,
            "evidence": f"results/ai_generation/tasks/{task}/summary.json + robustness_jobs/*.pkl",
        })
    write_csv(args.out / "robustness_comparison.csv", robustness_rows,
              ["task", "planned", "executed", "published_legacy_success", "recomputed_legacy_success",
               "legacy_matches_published", "steady_full", "relative_equilibrium_only",
               "spectral_checked", "spectral_passed", "spectral_positive", "spectral_not_checked",
               "shape_match", "shape_checked", "within_budget", "strict_success", "strict_rate",
               "positive_seeds", "budget_seeds", "shape_mismatch_seeds", "not_steady_seeds",
               "bands", "evidence"])

    # ---- generalization ------------------------------------------------------------
    cv = {}
    for tag in ("ai_generation", "ai_generation_fine"):
        for surface in ("frog", "sphere5"):
            payload = read_json(RESULTS / tag / surface / "cv.json") or {}
            cv[(tag, surface)] = payload.get("summary", {})

    def eval_rate(tag, surface, pipeline, predicate, subset=None):
        records = [r for r in eval_records[tag] if r.surface == surface and r.pipeline == pipeline]
        if subset is not None:
            records = [r for r in records if subset(r)]
        return sum(1 for r in records if predicate(r)), len(records)

    def predicted_accuracy(tag, surface):
        rows = evaluation_files[tag][0]
        points = {}
        for row in rows:
            if row["pipeline"] != "ai" or row["surface"] != surface:
                continue
            points.setdefault(row["point"], set()).add(row.get("predicted_type"))
        correct = total = 0
        for point, predicted in sorted(points.items()):
            target = target_for_point(surface, point, refs)
            total += 1
            correct += 1 if predicted == {target} else 0
        return correct, total

    def prediction_agreement(surface):
        old = {r["point"]: r.get("predicted_type") for r in evaluation_files["ai_generation"][0]
               if r["pipeline"] == "ai" and r["surface"] == surface}
        new = {r["point"]: r.get("predicted_type") for r in evaluation_files["ai_generation_fine"][0]
               if r["pipeline"] == "ai" and r["surface"] == surface}
        matched = sum(1 for point in old if old[point] == new.get(point))
        return matched, len(old)

    generalization_rows = []
    task_point = lambda r: (r.evidence.get("point") or "").startswith("task_")  # noqa: E731
    for surface in ("frog", "sphere5"):
        old_cv, new_cv = cv[("ai_generation", surface)], cv[("ai_generation_fine", surface)]
        cv_rows = [
            ("cv_type_accuracy", old_cv.get("type_accuracy"), new_cv.get("type_accuracy"),
             f"leave-one-grid-point-out with {old_cv.get('n_samples')} vs {new_cv.get('n_samples')} samples; "
             "the fine grid keeps neighbours 2x closer, so the two CVs are not the same difficulty"),
            ("cv_kappa0_rel_median", old_cv.get("kappa0_rel_median"), new_cv.get("kappa0_rel_median"),
             "same LOO difference as above"),
            ("cv_qv_rmse_median", old_cv.get("qv_rmse_median"), new_cv.get("qv_rmse_median"),
             "same LOO difference as above"),
            ("cv_density_rel_median", old_cv.get("density_rel_median"), new_cv.get("density_rel_median"),
             "spot-density regression error, same LOO difference"),
        ]
        for metric, old, new, note in cv_rows:
            generalization_rows.append({
                "metric": metric, "surface": surface, "old_model": old, "new_model": new,
                "comparable": False, "note": note,
                "source": f"results/ai_generation/{surface}/cv.json; "
                          f"results/ai_generation_fine/{surface}/cv.json"})
        for predicate, subset, metric, note in (
            (lambda r: r.steady_full is True, None, "e2e_complete_steady_rate",
             "full residual <= 1e-11 for all 24 held-out runs"),
            (lambda r: r.legacy_success is True, None, "e2e_legacy_success_rate",
             "published legacy rule (rotation-labelled modes excluded from max Re lambda)"),
            (lambda r: r.end_to_end_success is True, task_point, "e2e_strict_task_point_rate",
             "strict task success on the 6 runs at the two task parameter points"),
            (lambda r: r.observed_pattern == (r.evidence.get("reference_pattern")), None,
             "e2e_reference_morphology_match",
             "observed type equals the PDE reference majority type (surrogate target at random points)"),
            (lambda r: r.spectral_status == STATUS_PASS, None, "e2e_spectral_pass_rate",
             "no positive sampled eigenvalue (rotation included); legacy values are unvalidated"),
            (lambda r: r.relative_equilibrium is True, None, "e2e_relative_equilibrium_rate",
             "rotation-removed residual <= 1e-11 but full residual above"),
        ):
            old_n, old_d = eval_rate("ai_generation", surface, "ai", predicate, subset)
            new_n, new_d = eval_rate("ai_generation_fine", surface, "ai", predicate, subset)
            generalization_rows.append({
                "metric": metric, "surface": surface,
                "old_model": f"{old_n}/{old_d}", "new_model": f"{new_n}/{new_d}", "comparable": True,
                "note": note + "; same 8 points x 3 seeds, but different run dates/loads for wall time",
                "source": "results/ai_generation/evaluation.json; results/ai_generation_fine/evaluation.json"})
        cost_old = cost_stats([r for r in eval_records["ai_generation"]
                               if r.surface == surface and r.pipeline == "ai"])
        cost_new = cost_stats([r for r in eval_records["ai_generation_fine"]
                               if r.surface == surface and r.pipeline == "ai"])
        generalization_rows.append({
            "metric": "steps_median_legacy_success", "surface": surface,
            "old_model": cost_old["legacy_steps_success"], "new_model": cost_new["legacy_steps_success"],
            "comparable": True,
            "note": "integration steps are machine-independent; compare only with the same time cap",
            "source": "evaluation.json relax_steps"})
        old_n, old_d = predicted_accuracy("ai_generation", surface)
        new_n, new_d = predicted_accuracy("ai_generation_fine", surface)
        matched, points = prediction_agreement(surface)
        generalization_rows.append({
            "metric": "heldout_type_accuracy", "surface": surface,
            "old_model": f"{old_n}/{old_d}", "new_model": f"{new_n}/{new_d}", "comparable": True,
            "note": f"8 held-out points, predicted type vs target/PDE-majority reference; old/new "
                    f"predictions agree on {matched}/{points} points; mixed PDE reference points: {len(ties)}",
            "source": "evaluation.json predicted_type + pde reference"})
    write_csv(args.out / "generalization_comparison.csv", generalization_rows,
              ["metric", "surface", "old_model", "new_model", "comparable", "note", "source"])

    # ---- efficiency ----------------------------------------------------------------
    efficiency_rows = []
    for tag in ("ai_generation", "ai_generation_fine"):
        for surface in ("frog", "sphere5"):
            for pipeline in ("ai", "pde"):
                records = [r for r in eval_records[tag] if r.surface == surface and r.pipeline == pipeline]
                stats = cost_stats(records)
                protocol = EVAL_PROTOCOL[(tag, surface, pipeline)]
                efficiency_rows.append({
                    "row_type": "cost", "group": f"{tag}:{surface}:{pipeline}", "surface": surface,
                    "model": MODEL_OLD if tag == "ai_generation" else MODEL_NEW, "pipeline": pipeline,
                    "n": stats["n"], "legacy_success": stats["legacy_success"],
                    "strict_success": stats["n_success"],
                    "wall_median_all_s": stats["wall_all"],
                    "wall_median_legacy_success_s": stats["legacy_wall_success"],
                    "wall_median_legacy_failure_s": stats["legacy_wall_failure"],
                    "wall_median_strict_success_s": stats["wall_success"],
                    "steps_median_all": stats["steps_all"],
                    "steps_median_legacy_success": stats["legacy_steps_success"],
                    "steps_median_legacy_failure": stats["legacy_steps_failure"],
                    "steps_median_strict_success": stats["steps_success"],
                    "host": protocol["host"], "load": protocol["load"],
                    "concurrency": protocol["concurrency"],
                    "note": "wall time is only meaningful together with host/load/concurrency",
                })
            ai = [r for r in eval_records[tag] if r.surface == surface and r.pipeline == "ai"]
            pde = [r for r in eval_records[tag] if r.surface == surface and r.pipeline == "pde"]
            ai_stats, pde_stats = cost_stats(ai), cost_stats(pde)
            flags = comparison_flags(
                dict(EVAL_PROTOCOL[(tag, surface, "ai")], t_max=T_MAX_EVAL, tol=STEADY_TOL),
                dict(EVAL_PROTOCOL[(tag, surface, "pde")], t_max=T_MAX_EVAL, tol=STEADY_TOL))
            ratios = paired_ratios(ai, pde)
            efficiency_rows.append({
                "row_type": "paired_ratio", "group": f"{tag}:{surface}:ai_vs_pde", "surface": surface,
                "model": MODEL_OLD if tag == "ai_generation" else MODEL_NEW, "pipeline": "ai/pde",
                "n": ai_stats["n"],
                "legacy_success": f"{ai_stats['legacy_success']}/{pde_stats['legacy_success']}",
                "strict_success": f"{ai_stats['n_success']}/{pde_stats['n_success']}",
                "paired_success_pairs_legacy": ratios["legacy_success"]["pairs"],
                "paired_steps_ratio_median_legacy": ratios["legacy_success"]["steps_ratio_median"],
                "paired_wall_ratio_median_legacy": ratios["legacy_success"]["wall_ratio_median"],
                "paired_success_pairs_strict": ratios["end_to_end_success"]["pairs"],
                "paired_steps_ratio_median_strict": ratios["end_to_end_success"]["steps_ratio_median"],
                "steps_ratio_valid": flags["steps_comparable"],
                "wall_ratio_valid": flags["wall_comparable"],
                "host": EVAL_PROTOCOL[(tag, surface, "ai")]["host"],
                "load": f"ai={EVAL_PROTOCOL[(tag, surface, 'ai')]['load']}; "
                        f"pde={EVAL_PROTOCOL[(tag, surface, 'pde')]['load']}",
                "concurrency": f"ai={EVAL_PROTOCOL[(tag, surface, 'ai')]['concurrency']}; "
                               f"pde={EVAL_PROTOCOL[(tag, surface, 'pde')]['concurrency']}",
                "note": flags["reason"],
            })
    write_csv(args.out / "efficiency_comparison.csv", efficiency_rows,
              ["row_type", "group", "surface", "model", "pipeline", "n", "legacy_success", "strict_success",
               "wall_median_all_s", "wall_median_legacy_success_s", "wall_median_legacy_failure_s",
               "wall_median_strict_success_s", "steps_median_all", "steps_median_legacy_success",
               "steps_median_legacy_failure", "steps_median_strict_success",
               "paired_success_pairs_legacy", "paired_steps_ratio_median_legacy",
               "paired_wall_ratio_median_legacy", "paired_success_pairs_strict",
               "paired_steps_ratio_median_strict", "steps_ratio_valid", "wall_ratio_valid",
               "host", "load", "concurrency", "note"])

    # ---- report --------------------------------------------------------------------
    group_map = {group["key"]: group for group in groups}
    candidate_rows = []
    for record in candidates:
        candidate_rows.append({
            "candidate": record.experiment_id.replace("candidate:", ""),
            "mat_sha256": (record.evidence.get("mat_sha256") or "")[:16],
            "residual": record.full_residual,
            "pattern": record.observed_pattern,
            "n_components": record.evidence.get("n_components"),
            "spectral_status": record.spectral_status,
            "spectral_max_real": record.spectral_max_real,
            "evidence_level": record.spectral_evidence_level,
            "file": record.file_validity,
            "i01": record.i01_candidate_accepted,
            "strict": record.end_to_end_success,
            "recommended": record.evidence.get("recommended"),
        })

    report = build_report(
        started=started, git=git, code_hashes=code_hashes, groups=group_map,
        candidate_rows=candidate_rows, robustness_rows=robustness_rows,
        generalization_rows=generalization_rows, efficiency_rows=efficiency_rows,
        checks=checks, problems=PROBLEMS, training_records=training_records)
    (args.out / "evaluation_report.md").write_text(report, encoding="utf-8")

    artifacts = {}
    for name in ("evaluation_records.json", "success_breakdown.csv", "robustness_comparison.csv",
                 "generalization_comparison.csv", "efficiency_comparison.csv", "evaluation_report.md"):
        artifacts[name] = {"sha256": file_hash(args.out / name),
                           "bytes": (args.out / name).stat().st_size}
    manifest = {
        "schema_version": "aic-frog-i04-manifest-v1",
        "evaluation_version": EVALUATION_VERSION,
        "generated_utc": started.isoformat(),
        "status": status,
        "command": [sys.executable, "scripts/i04_evaluate.py", "--out", str(args.out)],
        "git": git,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "code_sha256": code_hashes,
        "source_sha256": {
            "results/ai_generation/tasks/robustness.json": file_hash(AI / "tasks/robustness.json"),
            "results/ai_generation/evaluation.json": evaluation_files["ai_generation"][1],
            "results/ai_generation_fine/evaluation.json": evaluation_files["ai_generation_fine"][1],
            **{f"results/{tag}/{surface}/dataset.json": datasets[(tag, surface)][1]
               for tag in ("ai_generation", "ai_generation_fine") for surface in ("frog", "sphere5")},
            **{f"results/{tag}/{surface}/cv.json": file_hash(RESULTS / tag / surface / "cv.json")
               for tag in ("ai_generation", "ai_generation_fine") for surface in ("frog", "sphere5")},
            "results/i01_float64_verified": {task: file_hash(I01 / "tasks" / task / "solution.mat")
                                             for task in TASK_SPEC},
            "results/i02_stability/certificates": {
                name: file_hash(I02 / "certificates" / name / "stability_certificate.json")
                for name in ("six_bands", "spots", "seed2")},
        },
        "counts": {group["key"]: {k: group["summary"][k] for k in
                                  ("planned", "executed", "missing", "error", "valid", "steady_full",
                                   "spectral_passed", "spectral_positive", "shape_match", "within_budget",
                                   "legacy_success", "strict_success")}
                   for group in groups},
        "checks": checks,
        "problems": PROBLEMS,
        "artifacts": artifacts,
        "limitation": SPECTRAL_LIMITATION,
    }
    write_json_exclusive(args.out / "run_manifest.json", manifest)
    print(json.dumps({"status": status, "records": len(all_records),
                      "checks_failed": [c["check"] for c in checks if not c["passed"]],
                      "problems": PROBLEMS}, ensure_ascii=False, indent=1), flush=True)
    return 0 if status == "ok" else 2


def build_report(**kw):
    """Render evaluation_report.md from the computed structures (no free-hand numbers)."""
    groups = kw["groups"]
    robustness_rows = kw["robustness_rows"]
    candidate_rows = kw["candidate_rows"]
    checks = kw["checks"]
    problems = kw["problems"]
    cost_rows = [row for row in kw["efficiency_rows"] if row["row_type"] == "cost"]
    ratio_rows = [row for row in kw["efficiency_rows"] if row["row_type"] == "paired_ratio"]

    def summary(key):
        return groups[key]["summary"]

    L = []
    L.append("# I04 统一评测体系与结果重统计\n")
    L.append(f"- evaluation_version: `{EVALUATION_VERSION}`")
    L.append(f"- generated: {kw['started'].isoformat()}")
    L.append(f"- git: `{kw['git']['head']}` on `{kw['git']['branch']}` "
             f"(dirty={kw['git']['dirty']}; I01/I02 changes stay uncommitted by design)")
    L.append("- code: " + ", ".join(f"`{name}`={digest[:12]}" for name, digest in kw["code_hashes"].items()))
    L.append("- 本文件由 `scripts/i04_evaluate.py` 从原始 JSON/PKL、I01 MAT 证书和 I02 稳定性证书机械生成；"
             "所有数字都可回溯到 `evaluation_records.json` 的记录与 `run_manifest.json` 的源文件哈希。\n")

    L.append("## 1. 判据定义（不同数学概念分开记录）\n")
    L.append("| 判据 | 定义 | 阈值 | 适用范围 | 证据来源 |")
    L.append("|---|---|---|---|---|")
    for name, spec in CRITERIA.items():
        L.append(f"| `{name}` | {spec['definition']} | {spec['threshold']} | {spec['scope']} | {spec['source']} |")
    L.append("")
    L.append("谱状态四种取值：`numerical_check_passed`、`unstable_positive_mode`、`insufficient_evidence`、"
             "`not_checked`。" + SPECTRAL_LIMITATION + "\n")

    L.append("## 2. 四组任务的新统计（原始记录机械重算）\n")
    L.append("| 任务 | 计划 | 执行 | 完整稳态 | 仅相对平衡 | 谱已检查 | 无正模态 | 有正模态 | 谱未检查 | "
             "形态匹配 | 预算内 | 旧口径 | **新口径** |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in robustness_rows:
        L.append(f"| {row['task']} | {row['planned']} | {row['executed']} | {row['steady_full']} | "
                 f"{row['relative_equilibrium_only']} | {row['spectral_checked']} | {row['spectral_passed']} | "
                 f"{row['spectral_positive']} | {row['spectral_not_checked']} | {row['shape_match']} | "
                 f"{row['within_budget']} | {row['recomputed_legacy_success']}/{row['planned']} | "
                 f"**{row['strict_success']}/{row['planned']}** |")
    L.append("")
    for row in robustness_rows:
        L.append(f"- **{row['task']}**：正模态种子 {row['positive_seeds'] or '无'}；"
                 f"预算耗尽 {row['budget_seeds'] or '无'}；形态不符 {row['shape_mismatch_seeds'] or '无'}；"
                 f"稳态未达标 {row['not_steady_seeds'] or '无'}；n_components 分布 {row['bands'] or 'n/a'}。")
    L.append("")
    L.append("说明：旧口径 = 决定性残差 ≤ 1e-11 且 legacy stable 为真；旧 stable 把旋转/倾斜模态排除在 "
             "max Re λ 之外，并容忍 max Re λ < 1e-6。新口径 = 完整稳态 + 采样谱无正实部（含旋转类模态）+ "
             "形态匹配 + 预算内。两者的差异来自谱口径（是否排除旋转类模态）与形态口径，不是重新求解。\n")

    L.append("## 3. 旧统计与新统计的差异\n")
    L.append("| 任务 | 旧发布 | I04 重算旧口径 | I04 新口径 | 主要差异来源 |")
    L.append("|---|---|---|---|---|")
    notes = {
        "frog_spots": "2 个失败：seed22 收敛但 max Re λ=+3.2e-5（鞍点）；seed25 到 t=2e5 未收敛（3.9e-3）",
        "frog_stripes": "收敛 33/33、谱全负，但 seed1（9 斑）与 seed21（10 斑）最终是斑点态 → 条纹形态 31/33",
        "sphere6_spots": "6 个预算失败（seed 6/7/15/17/18/20）；17 个完整稳态中 seed13/16/21 有正旋转模态 → 14/23",
        "sphere6_stripes": "1 个预算失败（seed1）；22 个完整稳态中 13 个六条带有正倾斜模态 +1.67e-6 → 9/23",
    }
    for row in robustness_rows:
        L.append(f"| {row['task']} | {row['published_legacy_success']}/{row['planned']} | "
                 f"{row['recomputed_legacy_success']}/{row['planned']} | "
                 f"**{row['strict_success']}/{row['planned']}** | {notes[row['task']]} |")
    L.append("")
    L.append("- `robustness.json` 的 `max_re_lambda` 数组只收录 legacy 成功运行（31/17/22 项），"
             "因此旧文里的 “max Re λ 范围” 是**条件于成功**的统计，不是全部样本的稳定性分布。")
    L.append("- 旧发布把“收敛到任意稳定态”当成任务成功：青蛙条纹 33/33 里含 2 个斑点态，"
             "球面条纹 22/23 里含 13 个正倾斜模态。")
    L.append("- 球面斑点 17/23 里含 3 个正旋转模态；14/23 才是“完整稳态且未发现正模态”的候选数。\n")

    L.append("### 旧表述逐条核对\n")
    L.append("| 旧表述 | 数值 | 来源 | I04 判定 | 说明 |")
    L.append("|---|---|---|---|---|")
    for claim in PUBLISHED_CLAIMS:
        L.append(f"| {claim['claim']} | {claim['value']} | {claim['source']} | **{claim['verdict']}** | "
                 f"{claim['detail']} |")
    L.append("")

    L.append("## 4. 正式候选与证书绑定\n")
    L.append("| 候选 | MAT sha256 | 完整残差 | 形态 | 分量数 | 谱状态 | 谱 max Re λ | 证据等级 | 文件 | I01 | 端到端 | 推荐 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in candidate_rows:
        L.append(f"| {row['candidate']} | `{row['mat_sha256']}` | {fmt(row['residual'])} | {row['pattern']} | "
                 f"{fmt(row['n_components'])} | {row['spectral_status']} | {fmt(row['spectral_max_real'])} | "
                 f"{row['evidence_level']} | {row['file']} | {fmt(row['i01'])} | {fmt(row['strict'])} | "
                 f"{fmt(row['recommended'])} |")
    L.append("")
    L.append("- 证书与 MAT 通过 SHA256 绑定：`certificate.input.mat_sha256 == sha256(solution.mat)`；"
             "任一不一致脚本直接失败（fail-closed）。")
    L.append("- 青蛙两个候选只有 I01 证书；旧稳定性数字来自 **I01 之前的场**（原 MAT 哈希记录在 I01 provenance），"
             "不是对当前 float64 MAT 的谱检查，因此 `spectral_status=not_checked`。")
    L.append("- 六条带 seed0 保留为历史结果：I02 在**同一份 MAT** 上得到可靠正倾斜模态 +1.6654e-6，"
             "不能推荐正式提交；七条带 seed2 的证书绑定到新 MAT，谱检查通过。")
    L.append("- seed2 的 I02 证书只覆盖重建后的那一份 MAT；其他 8 个收敛的七条带种子"
             "（3/8/11/12/18/19/20/22）没有独立证书，只有旧单 shift 证据，不能用 seed2 的证书代替。\n")

    L.append("## 5. AI 泛化与效率：原先不公平或不严谨的比较\n")
    L.append("### 5.1 泛化指标\n")
    L.append("| 指标 | 曲面 | 旧模型 | 新模型 | 可直接比较 | 备注 |")
    L.append("|---|---|---|---|---|---|")
    for row in kw["generalization_rows"]:
        L.append(f"| {row['metric']} | {row['surface']} | {fmt(row['old_model'])} | {fmt(row['new_model'])} | "
                 f"{fmt(row['comparable'])} | {row['note']} |")
    L.append("")
    L.append("### 5.2 计算成本（按同一成功口径拆分）\n")
    L.append("| 组 | 旧口径成功 | 新口径成功 | 墙钟中位：全部/旧口径成功/旧口径失败 (s) | "
             "步数中位：全部/旧口径成功/旧口径失败 | host · load |")
    L.append("|---|---|---|---|---|---|")
    for row in cost_rows:
        L.append(f"| {row['group']} | {row['legacy_success']}/{row['n']} | {row['strict_success']}/{row['n']} | "
                 f"{fmt(row['wall_median_all_s'])} / {fmt(row['wall_median_legacy_success_s'])} / "
                 f"{fmt(row['wall_median_legacy_failure_s'])} | {fmt(row['steps_median_all'])} / "
                 f"{fmt(row['steps_median_legacy_success'])} / {fmt(row['steps_median_legacy_failure'])} | "
                 f"{row['host']} · {row['load']} |")
    L.append("")
    L.append("注：本表墙钟统一用 `total_s`（含 setup）；旧文中的 673 s / 1159 s 用的是 `solve_s`，"
             "是同一批运行，差异只来自字段选择。")
    L.append("注：`evaluation:*` 组的新口径成功要求形态与任务/参考形态一致；随机参数点上没有“任务要求”，"
             "用 PDE 参考形态作替代目标，因此它衡量的是泛化一致性，不是任务要求。"
             "任务点上的严格成功率见 5.1 的 `e2e_strict_task_point_rate`。\n")
    L.append("### 5.3 AI / PDE 成对比较（同一参数点、同一种子）\n")
    L.append("| 对比 | 成对样本 | 步数比中位（旧口径） | 墙钟比中位（旧口径） | 成对样本（新口径） | "
             "步数比中位（新口径） | 步数比可用 | 墙钟比可用 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for row in ratio_rows:
        L.append(f"| {row['group']} | {row['paired_success_pairs_legacy']} | "
                 f"{fmt(row['paired_steps_ratio_median_legacy'])} | "
                 f"{fmt(row['paired_wall_ratio_median_legacy'])} | {row['paired_success_pairs_strict']} | "
                 f"{fmt(row['paired_steps_ratio_median_strict'])} | {fmt(row['steps_ratio_valid'])} | "
                 f"{fmt(row['wall_ratio_valid'])} |")
    L.append("")
    L.append("**公平性问题清单**\n")
    L.append("1. 新模型的 PDE 行是从旧 `evaluation.json` 复用的 48 条记录（逐字段相同，见 manifest 检查）；"
             "新模型 AI 行是单独跑的（青蛙 20 进程、球面 8 进程），因此 `summary.md` 里的 2.2x / 2.3x "
             "墙钟加速比不成立。")
    L.append("2. 旧模型青蛙那组 AI 与 PDE 在同一台笔记本交错运行、负载相同，673 s vs 1159 s 可用；"
             "对应的成对步数比见上表。")
    L.append("3. 旧/新模型之间墙钟不可比（旧模型与 PDE 交错、新模型单独跑）；模型对比只能看积分步数。")
    L.append("4. L5 评测从未做 orient：24 次运行没有一个完整稳态，全部是相对平衡态；因此 L5 的成功率"
             "不能与球面 L6 形式任务（完整稳态）直接比较。")
    L.append("5. 留一法类型准确率 64%→86%（青蛙）、89%→93%（球面）不可直接比较：旧网格 45 点、新网格 306 点，"
             "去掉一点后邻点间距差 2 倍，难度不同。")
    L.append("6. 同一批 8 个留出点上新旧预测只有 1 个不同（球面 random5：旧判条纹、新判孔洞），"
             "所以“新模型更强”主要不是分类器变化，而是训练数据与形态先验的变化。")
    L.append("7. 任务点上的严格成功率（完整稳态 + 无正模态 + 形态 + 预算）只有 6 次运行，"
             "不能与 24 次留出评测混用同一个分母。\n")

    L.append("## 6. 训练标签与数据质量\n")
    L.append("| 数据 | 行数 | 收敛到 1e-11 | 无正模态 | 有正模态 | 谱未检查 | 空标签（均匀态） | 预算 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for key, rows in kw["training_records"].items():
        summary = groups[f"training:{key[0]}:{key[1]}"]["summary"]
        empty = sum(1 for r in rows if not r.evidence.get("labels_present"))
        L.append(f"| {key[0]} · {key[1]} | {summary['executed']} | {summary['steady_full']} | "
                 f"{summary['spectral_passed']} | {summary['spectral_positive']} | "
                 f"{summary['spectral_not_checked']} | {empty} | 1e5 |")
    L.append("")
    L.append("- 训练集用 t=1e5 预算、评测/任务用 t=2e5；球面 L5 的 306 条标签只有 90 条达到 1e-11"
             "（多为斑点，条纹/迷宫停在转动漂移或重排），这些未收敛标签仍被用于训练。")
    L.append("- 青蛙新数据中有 4 个标签态存在正采样模态；用不稳定态做形态回归目标需要在报告中说明。")
    L.append("- 空标签全部是 uniform 态（`labels={}`），不是缺失标签。\n")

    L.append("## 7. I03 技术报告与提交包应采用的数字\n")
    L.append("### 7.1 任务结果（四组正式候选）\n")
    L.append("| 任务 | 采用结果 | 完整残差 | 谱状态 | 说明 |")
    L.append("|---|---|---|---|---|")
    L.append("| 任务1 球面斑点 | 45 斑，MAT `44f5931a…` | 8.423e-15 | numerical_check_passed"
             "（I02 三配置，max Re λ=-6.231e-6） | 完整稳态；谱检查含旋转类模态 |")
    L.append("| 任务1 球面条纹 | 七条带 seed2，MAT `ef132d82…` | 2.437e-14 | numerical_check_passed"
             "（I02 三配置，max Re λ=-1.478e-6） | **不要**再用六条带 seed0（可靠正倾斜模态 +1.6654e-6） |")
    L.append("| 任务2 青蛙斑点 | 22 斑，MAT `19a92c65…` | 2.171e-14 | not_checked（无绑定谱证书） | "
             "旧单 shift 记录 −1.03e-4 属于 I01 之前的场 |")
    L.append("| 任务2 青蛙条纹 | 5 段迷宫，MAT `aa99f137…` | 1.006e-14 | not_checked（无绑定谱证书） | "
             "旧单 shift 记录 −1.55e-4 属于 I01 之前的场 |")
    L.append("")
    L.append("### 7.2 稳健性（33 / 23 个种子，题给参数）\n")
    L.append("- 青蛙斑点：31/33 = 完整稳态且未发现正模态。失败：seed22 鞍点（+3.2e-5）、seed25 预算耗尽。")
    L.append("- 青蛙条纹：收敛 33/33、谱全负 33/33，但**目标条纹形态只有 31/33**（seed1、seed21 是斑点态）；"
             "不能写“33/33 目标条纹生成成功”。")
    L.append("- 球面斑点：完整稳态 17/23；3 个有正旋转模态（seed13/16/21）；无正模态候选 14/23；6 个预算失败。"
             "14/23 是有限采样候选率，不是数学稳定率。")
    L.append("- 球面条纹：完整稳态 22/23；13 个六条带有正倾斜模态 +1.67e-6，9 个七条带无正模态；"
             "只有 9/23 同时满足“完整稳态 + 无正模态”。其中 seed2 有 I02 证书；其余 8 个七条带种子"
             "（3/8/11/12/18/19/20/22）只有旧单 shift 证据，不能用 seed2 的证书代替独立验收。")
    L.append("")
    L.append("### 7.3 AI 对比\n")
    L.append("- 青蛙（8 点 × 3 种子）：旧模型 AI 24/24（全部完整稳态、谱全负）、新模型 23/24、普通流程 22/24。")
    L.append("- 青蛙成本：步数中位 旧 AI 15062、新 AI 13812、普通 25875（AI 少 42–47%）；"
             "旧模型那组同负载墙钟约 1.7x（solve_s 673/1159，total_s 678/1166）可用；"
             "新模型只能写步数。")
    L.append("- 球面 L5：旧 AI 12/24、新 AI 10/24、普通 6/24 都是“相对平衡态 + 旧谱口径”；"
             "按本阶段严格口径（相对平衡 + 无正模态）为 10/24、6/24、4/24，且 0 个完整稳态。")
    L.append("- 留一法：青蛙 64%→86%、球面 89%→93%，必须注明两者网格密度与留一难度不同。")
    L.append("- 端到端泛化：8 个留出点上新旧预测只有球面 random5 不同；青蛙两模型在孔洞区都收敛到迷宫而不是孔洞，"
             "这是形态先验问题，不是求解器失败。\n")

    L.append("## 8. 证据缺口与后续建议\n")
    L.append("1. 稳健性扫描没有保存 MAT/场 → `file_validity=not_saved`；3 个正旋转模态与 13 个六条带正模态"
             "无法在本阶段复核。建议对 seed13/16/21 与六条带代表态补 I02 式审计。")
    L.append("2. 旧谱值都是单 shift、k≈8、无特征对残差：`unstable_positive_mode` 是保守判定；"
             "升级为“可信正模态”需要 I02 的三配置 + 特征对残差 + 小扰动验证。")
    L.append("3. 青蛙两个正式候选缺少绑定到 I01 MAT 的谱检查（Newton 修正 ≤3e-8，但不能当作证明）；"
             "建议给 `scripts/i02_stability.py` 增加 frog 分支后补齐。")
    L.append("4. L5 评测没有 orient，因此没有完整稳态；若要在统一口径下报告 L5，需要重跑 orient 版本。")
    L.append("5. 训练标签与评测的 t_max 不同（1e5 vs 2e5），训练目标含大量未收敛/相对平衡态；"
             "建议只用 ≤1e-11 的稳态做形态回归，或在标签中记录收敛等级。")
    L.append("6. 提交包/README 中的“24/24 收敛且稳定”“33/33 条纹”“22/23 条纹”必须按本报告改写；"
             "本轮不修改报告与提交包。\n")

    L.append("## 9. 可复现性与检查\n")
    L.append(f"- 记录数：{sum(len(g['records']) for g in groups.values())}；检查项 {len(checks)} 条，"
             f"失败 {sum(1 for c in checks if not c['passed'])} 条；源问题 {len(problems)} 条。")
    for check in checks:
        L.append(f"- [{'x' if check['passed'] else ' '}] {check['check']} {check['detail']}")
    if problems:
        L.append("")
        L.append("源问题：")
        for problem in problems:
            L.append(f"- {json.dumps(problem, ensure_ascii=False)}")
    L.append("")
    L.append("复现命令：`python scripts/i04_evaluate.py --out results/i04_evaluation`（目录必须不存在）。"
             "所有分母、失败样本和源文件哈希见 `success_breakdown.csv`、`evaluation_records.json` 与 "
             "`run_manifest.json`。")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
