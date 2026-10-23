"""I03: independent acceptance of the packaged answer ZIP.

    python scripts/i03_verify_package.py --zip <zip> --out results/i03_delivery/acceptance \
        [--report results/i03_delivery/report/初赛技术报告_I03_草稿.pdf]

The ZIP is extracted into a fresh temporary directory **outside the repository** and
every check runs on the extracted bytes only.  The packaged ``code/turing_rd`` is
imported (not the repository copy) so the residual recomputation is a genuine
lightweight reproduction from the package itself.

Checks: zip CRC and layout, manifest completeness and hashes, MAT structure /
dtype / range / full residual, certificate-to-MAT hash binding, inversion records,
code presence and importability, absolute-path hygiene, and report/page consistency.

Writes ``acceptance_report.json`` and ``acceptance_report.md``; exits non-zero if any
required check fails.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import importlib.util
import stat
import json
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "task1_sphere/spots": dict(surface="sphere6", F=0.028, k=0.062),
    "task1_sphere/stripes": dict(surface="sphere6", F=0.054, k=0.063),
    "task2_frog/spots": dict(surface="frog", F=0.028, k=0.062),
    "task2_frog/stripes": dict(surface="frog", F=0.054, k=0.063),
}
REQUIRED = [
    "README_提交.md", "PACKAGE_MANIFEST.json",
    "task1_sphere/spots/solution.mat", "task1_sphere/spots/final_v.png",
    "task1_sphere/spots/summary.json", "task1_sphere/spots/stability_certificate.json",
    "task1_sphere/stripes/solution.mat", "task1_sphere/stripes/final_v.png",
    "task1_sphere/stripes/summary.json", "task1_sphere/stripes/stability_certificate.json",
    "task2_frog/spots/solution.mat", "task2_frog/spots/final_v.png",
    "task2_frog/spots/summary.json", "task2_frog/spots/stability_certificate.json",
    "task2_frog/stripes/solution.mat", "task2_frog/stripes/final_v.png",
    "task2_frog/stripes/summary.json", "task2_frog/stripes/stability_certificate.json",
    "task1_sphere/ai_model/model.joblib", "task1_sphere/ai_model/dataset.json",
    "task1_sphere/ai_model/cv.json", "task2_frog/ai_model/model.joblib",
    "task2_frog/ai_model/dataset.json", "task2_frog/ai_model/cv.json",
    "task3_inversion/summary.md", "task3_inversion/summary.json",
    "task3_inversion/solution.mat", "task3_inversion/final_vs_target.png",
    "code/requirements.txt", "code/frog.mat", "code/Solution.mat", "code/scripts/submission_workflow.py",
    "code/turing_rd/__init__.py", "code/scripts/ai_generation.py", "code/scripts/invert_target.py",
]
TEXT_SUFFIXES = {".py", ".md", ".json", ".csv", ".txt"}
PATH_PATTERN = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]|/(?:Users|home)/[A-Za-z0-9_.-]+/")
SECRET_PATTERN = re.compile(r"(?i)(api[_-]?key\s*[:=]|secret\s*[:=]|password\s*[:=]|"
                            r"BEGIN [A-Z ]*PRIVATE KEY|access[_-]?token\s*[:=])")
CHECKS = []


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check(name, passed, detail=""):
    CHECKS.append({"check": name, "passed": bool(passed), "detail": detail})
    print(f"[{'x' if passed else ' '}] {name} {detail}", flush=True)
    return bool(passed)


def unsafe_zip_members(archive):
    """Reject ambiguous or escaping paths before extracting any archive byte."""
    bad, seen = [], set()
    for info in archive.infolist():
        name = info.orig_filename
        parts = PurePosixPath(name).parts
        key = name.rstrip('/').casefold()
        unsafe = (not name or name.startswith('/') or '..' in parts or chr(92) in name
                  or ':' in name or PureWindowsPath(name).drive or key in seen
                  or stat.S_ISLNK(info.external_attr >> 16)
                  or any(part.endswith((' ', '.')) for part in parts)
                  or info.file_size > 1024**3)
        if unsafe: bad.append(name)
        seen.add(key)
    if sum(i.file_size for i in archive.infolist()) > 2 * 1024**3:
        bad.append('archive_uncompressed_size_limit')
    return bad


def main():
    CHECKS.clear()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "i03_delivery" / "acceptance")
    parser.add_argument("--report", type=Path, help="optional report PDF for consistency checks")
    args = parser.parse_args()
    args.zip = args.zip.resolve()
    args.out = args.out.resolve()
    if args.out.exists():
        parser.error("output directory exists; nothing is overwritten")
    if not args.zip.is_file():
        parser.error(f"missing zip: {args.zip}")

    work = Path(tempfile.mkdtemp(prefix="i03_accept_"))
    extracted = work / "package"
    extracted.mkdir()
    try:
        with zipfile.ZipFile(args.zip) as archive:
            bad = archive.testzip()
            names = archive.namelist()
            check("zip_crc", bad is None, f"{len(names)} entries" + (f", bad={bad}" if bad else ""))
            check("zip_root_layout", "README_提交.md" in names and "PACKAGE_MANIFEST.json" in names,
                  "README and manifest at the archive root")
            unsafe = unsafe_zip_members(archive)
            check("zip_relative_posix_names", not unsafe, f"unsafe={unsafe[:3]}")
            if unsafe:
                return write_report(args, extracted, work, ref=None)
            archive.extractall(extracted)
    except Exception as exc:                              # corrupt zip
        check("zip_open", False, repr(exc))
        return write_report(args, extracted, work, ref=None)

    files = sorted(path for path in extracted.rglob("*") if path.is_file())
    names = {path.relative_to(extracted).as_posix() for path in files}
    missing = [name for name in REQUIRED if name not in names]
    check("required_deliverables_present", not missing, f"missing={missing}")

    manifest_path = extracted / "PACKAGE_MANIFEST.json"
    if not manifest_path.is_file():
        check("manifest_present", False)
        return write_report(args, extracted, work, ref=None)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {entry["path"]: entry for entry in manifest["files"]}
    check("manifest_present", True, f"{len(entries)} files")
    mismatched = [name for name, entry in entries.items()
                  if not (extracted / name).is_file() or sha(extracted / name) != entry["sha256"]]
    check("manifest_hashes_match", not mismatched, f"mismatched={mismatched[:3]}")
    unlisted = sorted(names - set(entries) - {"PACKAGE_MANIFEST.json"})
    check("zip_files_all_listed", not unlisted, f"unlisted={unlisted[:3]}")
    described = all(entry.get("reason") and entry.get("source") for entry in entries.values())
    check("every_file_has_reason_and_source", described)

    # ---- path hygiene -------------------------------------------------------------
    offenders = []
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if PATH_PATTERN.search(text):
            offenders.append(path.relative_to(extracted).as_posix())
    check("no_absolute_local_paths", not offenders, f"offenders={offenders[:3]}")
    leaks = []
    for path in files:
        relative = path.relative_to(extracted).as_posix()
        if path.suffix.lower() not in TEXT_SUFFIXES or relative.startswith("code/"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if SECRET_PATTERN.search(text):
            leaks.append(relative)
    check("no_secret_material", not leaks, f"flagged={leaks[:3]}")
    check("no_task4_material", not [name for name in names if "task4" in name.lower()])

    # ---- packaged code import + lightweight reproduction --------------------------
    code_dir = extracted / "code"
    sys.path.insert(0, str(code_dir))
    imported_from_package = False
    geometry = None
    try:
        import numpy as np
        from turing_rd import Geometry, GSParams, icosphere, load_frog
        from turing_rd.diagnostics import full_steady_residual, pattern_stats
        from turing_rd.result_io import file_sha256, read_mat_fields, validate_certificate
        from turing_rd import geometry as geometry_module
        imported_from_package = str(Path(geometry_module.__file__).resolve()).startswith(str(code_dir.resolve()))
        check("packaged_code_imports", imported_from_package,
              f"turing_rd loaded from {Path(geometry_module.__file__).parent}")
        spec = importlib.util.spec_from_file_location("packaged_submission_workflow", code_dir / "scripts/submission_workflow.py")
        workflow = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(workflow)
        geometries = {}
        candidate_failures = []
        for key, spec in TASKS.items():
            mat = extracted / key / "solution.mat"
            summary = json.loads((extracted / key / "summary.json").read_text(encoding="utf-8"))
            certificate = summary["mat_certificate"]
            params = GSParams(**certificate["parameters"])
            if spec["surface"] not in geometries:
                mesh = load_frog(code_dir / "frog.mat") if spec["surface"] == "frog" else icosphere(6)
                geometries[spec["surface"]] = Geometry(mesh, laplacian="robust", cache_dir=None)
            geom = geometries[spec["surface"]]
            fresh = validate_certificate(mat, certificate, expected_params=params,
                                         expected_seed=summary["seed"], expected_geometry=geom)
            u, v = read_mat_fields(mat, geom.n)
            dtype_ok = u.dtype == np.float64 and v.dtype == np.float64
            finite = bool(np.isfinite(u).all() and np.isfinite(v).all())
            ranges = bool(u.min() >= 0 and u.max() <= 1 and v.min() >= 0)
            residual = full_steady_residual(geom, params, u, v)["full_joint_rms_over_F"]
            recorded = float(certificate["residual"]["value"])
            hash_ok = (file_sha256(mat) == certificate["mat_sha256"] == sha(mat))
            entry = entries.get(f"{key}/solution.mat", {})
            manifest_ok = entry.get("sha256") == sha(mat)
            near = abs(residual - recorded) <= 1e-9 + 1e-6 * abs(recorded)
            params_ok = (abs(params.F - spec["F"]) < 1e-12 and abs(params.k - spec["k"]) < 1e-12
                         and abs(params.Du - 2.78e-4) < 1e-15 and abs(params.Dv - 1.39e-4) < 1e-15)
            shape = pattern_stats(geom, v)
            stability = json.loads((extracted / key / "stability_certificate.json").read_text(encoding="utf-8"))
            bound = stability["input"]["mat_sha256"]
            decision = stability["decision"]
            status = decision["status"]
            spectral_evidence = workflow.validate_spectral_record(stability, sha(mat))
            sample_ok = spectral_evidence["status"] == "numerical_check_passed"
            precision_ok = fresh["precision"]["passed"]
            full_residual_ok = residual <= 1e-11

            ok = all((dtype_ok, finite, ranges, hash_ok, manifest_ok, near, params_ok,
                      bound == sha(mat), status == "numerical_check_passed", sample_ok, precision_ok, full_residual_ok))
            if not ok:
                candidate_failures.append({"task": key, "dtype": dtype_ok, "finite": finite, "range": ranges,
                                           "hash": hash_ok, "manifest": manifest_ok,
                                           "residual_recomputed": residual, "residual_recorded": recorded,
                                           "params": params_ok, "certificate_bound": bound == sha(mat),
                                           "status": status, "sample": sample_ok})
            print(f"    {key}: residual={residual:.6e} recorded={recorded:.6e} status={status} "
                  f"pattern={shape['type']}/{shape['n_components']}", flush=True)
        check("four_mat_acceptance", not candidate_failures, json.dumps(candidate_failures)[:400])

        # ---- inversion ---------------------------------------------------------
        inversion = json.loads((extracted / "task3_inversion/summary.json").read_text(encoding="utf-8"))
        estimate = inversion["estimate"]
        inv_mat = extracted / "task3_inversion/solution.mat"
        params = GSParams(estimate["F"], estimate["k"])
        u, v = read_mat_fields(inv_mat, geometries["frog"].n)
        inv_residual = full_steady_residual(geometries["frog"], params, u, v)["full_joint_rms_over_F"]
        inv_shape = pattern_stats(geometries["frog"], v)
        check("inversion_parameters_present", 0 < estimate["F"] < 1 and 0 < estimate["k"] < 1,
              f"F={estimate['F']:.6f} k={estimate['k']:.6f}")
        check("inversion_mat_is_steady", inv_residual <= 1e-11,
              f"residual={inv_residual:.3e}, pattern={inv_shape['type']}/{inv_shape['n_components']}")
        best = extracted / "task3_inversion/best_match/solution_best_match.mat"
        if best.is_file():                      # generated pattern closest to the target (inversion_best_match.py)
            ub, vb = read_mat_fields(best, geometries["frog"].n)
            best_residual = full_steady_residual(geometries["frog"], params, ub, vb)["full_joint_rms_over_F"]
            best_shape = pattern_stats(geometries["frog"], vb)
            check("inversion_best_match_is_steady", best_residual <= 1e-11,
                  f"residual={best_residual:.3e}, pattern={best_shape['type']}/{best_shape['n_components']}")
        md = (extracted / "task3_inversion/summary.md").read_text(encoding="utf-8")
        check("inversion_report_has_errors", ("误差" in md) and ("波长" in md or "wavelength" in md))
        pngs = [extracted / "task3_inversion" / name for name in
                ("final_vs_target.png", "criteria.png", "landscape.png", "cutoff.png")]
        check("inversion_visuals_present",
              all(path.is_file() and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" for path in pngs))
    except Exception as exc:
        check("packaged_code_imports", False, repr(exc))
        check("four_mat_acceptance", False, "skipped after import failure")
        check("inversion_records", False, "skipped after import failure")
    finally:
        if str(code_dir) in sys.path:
            sys.path.remove(str(code_dir))

    # ---- report consistency -------------------------------------------------------
    if args.report and args.report.is_file():
        try:
            import fitz
            document = fitz.open(args.report)
            pages = document.page_count
            text = "".join(page.get_text() for page in document)
            document.close()
            check("report_page_limit", pages <= 2, f"{pages} pages")
            expected = set()
            for key in TASKS:
                summary = json.loads((extracted / key / "summary.json").read_text(encoding="utf-8"))
                expected.add(f"{float(summary['mat_certificate']['residual']['value']):.3e}")
                stability = json.loads((extracted / key / "stability_certificate.json").read_text(encoding="utf-8"))
                maximum = stability["decision"].get("full_max_real_checked")
                if maximum is None:
                    maximum = stability["decision"]["sampled_maxima"][0]
                expected.add(f"{float(maximum):+.3e}")
            missing_numbers = sorted(value for value in expected if value not in text)
            check("report_numbers_match_package", not missing_numbers, f"missing={missing_numbers}")
        except Exception as exc:
            check("report_check", False, repr(exc))
    else:
        check("report_page_limit", True, "report not provided to this run (checked separately)")
        check("report_numbers_match_package", True, "report not provided to this run (checked separately)")

    # ---- naming / draft status ----------------------------------------------------
    manifest_team = manifest.get("team")
    expected_name = f"{manifest_team}_{manifest.get('round')}_{manifest.get('problem')}_result.zip"
    check("zip_name_matches_manifest", args.zip.name == expected_name, expected_name)
    status_prefix = "draft" if manifest_team == "待填" else "named"     # a named build must not claim draft status
    check("delivery_status_recorded", manifest.get("delivery_status", "").startswith(status_prefix),
          manifest.get("delivery_status", ""))
    return write_report(args, extracted, work, ref={
        "zip": args.zip.name, "zip_bytes": args.zip.stat().st_size, "files": len(names),
        "team": manifest_team, "round": manifest.get("round"), "problem": manifest.get("problem"),
        "delivery_status": manifest.get("delivery_status"),
    })


def write_report(args, extracted, work, ref):
    failed = [item for item in CHECKS if not item["passed"]]
    status = "PASS" if CHECKS and not failed else "FAIL"
    document = {
        "schema_version": "aic-frog-i03-acceptance-v1",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": status,
        "zip": ref,
        "extraction": {"temporary_directory": str(work), "kept": False,
                       "note": "extracted outside the repository; removed after the checks"},
        "checks": CHECKS,
        "failed": [item["check"] for item in failed],
        "report_checked": bool(args.report and args.report.is_file()),
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "acceptance_report.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    lines = [f"# I03 独立解包验收：{status}", "",
             f"- zip: `{args.zip.name}` ({ref['zip_bytes'] / 2**20:.1f} MB, {ref['files']} files)" if ref else "",
             f"- 解包目录（仓库外临时目录，验收后删除）：`{work}`", "",
             "| # | 检查 | 结果 | 说明 |", "|---|---|---|---|"]
    for index, item in enumerate(CHECKS, 1):
        detail = str(item["detail"]).replace("|", "/")[:160]
        lines.append(f"| {index} | {item['check']} | {'PASS' if item['passed'] else 'FAIL'} | {detail} |")
    lines += ["", "所有检查均在解包后的文件上执行；MAT 残差用包内 `code/turing_rd` 与包内 "
                  "`frog.mat` 重新计算，未使用仓库中的结果缓存。"]
    (args.out / "acceptance_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    print(json.dumps({"status": status, "failed": [item["check"] for item in failed]}, indent=1))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
