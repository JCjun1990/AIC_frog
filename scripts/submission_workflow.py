"""Portable reproduction helpers. No historical results directory is required."""
from __future__ import annotations
import argparse, hashlib, json, math, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TASKS = ("task1_sphere/spots", "task1_sphere/stripes",
         "task2_frog/spots", "task2_frog/stripes")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


PAYLOAD_IGNORED_DIRS = ("drafts/", "report/")


def manifest_path(relative):
    """Require one portable, canonical package path."""
    from pathlib import PurePosixPath
    if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
            or any(ord(c) < 32 for c in relative)):
        raise ValueError("unsafe manifest path: " + repr(relative))
    parts = relative.split("/")
    if (any(p in ("", ".", "..") or p.endswith((" ", ".")) for p in parts)
            or PurePosixPath(relative).is_absolute()):
        raise ValueError("unsafe manifest path: " + relative)
    return relative


def manifest_entries(document):
    """Validate before indexing: duplicates must never collapse into a dict."""
    entries = document.get("files")
    if not isinstance(entries, list):
        raise ValueError("manifest files must be a list")
    seen = set()
    for entry in entries:
        relative = manifest_path(entry["path"])
        if relative.casefold() in seen:
            raise ValueError("duplicate manifest paths: " + relative)
        seen.add(relative.casefold())
        digest = entry.get("sha256")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("invalid manifest SHA256: " + relative)
        if type(entry.get("bytes")) is not int or entry["bytes"] < 0:
            raise ValueError("invalid manifest byte count: " + relative)
        if not entry.get("source") or not entry.get("reason"):
            raise ValueError("manifest source/reason missing: " + relative)
    count = document.get("payload_file_count", len(entries))
    if type(count) is not int or count != len(entries):
        raise ValueError("manifest payload count mismatch")
    return entries


def payload_files(submission):
    """Enumerate actual payload, excluding only separate reports/archives and bytecode."""
    submission = Path(submission).resolve()
    files = []
    for path in sorted(submission.rglob("*")):
        relative = path.relative_to(submission).as_posix()
        if (relative == "PACKAGE_MANIFEST.json" or relative.startswith(PAYLOAD_IGNORED_DIRS)
                or (path.parent == submission and path.suffix.lower() == ".zip")
                or "__pycache__" in path.parts or path.suffix.lower() == ".pyc"):
            continue
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError("linked payload path: " + relative)
        if path.is_file():
            manifest_path(relative)
            if not path.resolve().is_relative_to(submission):
                raise ValueError("escaping payload path: " + relative)
            files.append(relative)
    return files


def check_manifest(submission):
    """Fail on missing, modified, duplicated, excluded or unregistered payload."""
    submission = Path(submission).resolve()
    document = load(submission / "PACKAGE_MANIFEST.json")
    entries = manifest_entries(document)
    actual = set(payload_files(submission))
    listed = {entry["path"] for entry in entries}
    for entry in entries:
        relative = entry["path"]
        target = (submission / relative).resolve()
        if not target.is_relative_to(submission) or not target.is_file():
            raise ValueError("manifest path missing or escaping: " + relative)
        if sha(target) != entry["sha256"]:
            raise ValueError("manifest hash/path mismatch: " + relative)
        if target.stat().st_size != entry["bytes"]:
            raise ValueError("manifest byte count mismatch: " + relative)
    if listed - actual:
        raise ValueError("manifest lists excluded payload: " + ", ".join(sorted(listed - actual)))
    if actual - listed:
        raise ValueError("unregistered payload file: " + ", ".join(sorted(actual - listed)))
    return {"manifest_files": len(entries), "payload_files": len(actual)}


def zip_manifest_hashes(zip_path):
    """Check unique paths, exact membership, sizes, CRC and actual ZIP hashes."""
    import zipfile
    with zipfile.ZipFile(zip_path) as archive:
        names, infos = set(), {}
        for info in archive.infolist():
            relative = info.orig_filename
            if info.is_dir():
                manifest_path(relative.rstrip("/"))
                continue
            manifest_path(relative)
            if relative.casefold() in names:
                raise ValueError("duplicate ZIP paths: " + relative)
            names.add(relative.casefold())
            infos[relative] = info
        if "PACKAGE_MANIFEST.json" not in infos:
            raise ValueError("packed manifest missing")
        document = json.loads(archive.read("PACKAGE_MANIFEST.json"))
        entries = manifest_entries(document)
        hashes = {entry["path"]: entry["sha256"] for entry in entries}
        packed = set(infos) - {"PACKAGE_MANIFEST.json"}
        if packed - set(hashes):
            raise ValueError("packed file missing from packed manifest: " + sorted(packed - set(hashes))[0])
        if set(hashes) - packed:
            raise ValueError("packed manifest lists a missing file: " + sorted(set(hashes) - packed)[0])
        for entry in entries:
            data = archive.read(entry["path"])
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError("packed hash mismatch: " + entry["path"])
            if len(data) != entry["bytes"]:
                raise ValueError("packed byte count mismatch: " + entry["path"])
        bad = archive.testzip()
        if bad is not None:
            raise ValueError("ZIP CRC mismatch: " + bad)
    return hashes


def compare_zip_to_workspace(submission, zip_path):
    """Strict byte/version comparison; a historical ZIP is never implicitly current."""
    submission = Path(submission).resolve()
    packed = zip_manifest_hashes(zip_path)
    different = {}
    for relative, expected in packed.items():
        target = submission / relative
        actual = sha(target) if target.is_file() else None
        if actual != expected:
            different[relative] = {"workspace_sha256": actual, "zip_sha256": expected}
    files = set(payload_files(submission))
    only_workspace = sorted(files - set(packed))
    only_zip = sorted(set(packed) - files)
    try:
        check_manifest(submission)
        manifest_error = None
    except (ValueError, OSError, KeyError) as exc:
        manifest_error = str(exc)
    status = "PASS" if not different and not only_workspace and not only_zip and not manifest_error else "FAIL"
    return {"status": status, "different": different,
            "only_workspace": only_workspace, "only_zip": only_zip,
            "workspace_manifest_error": manifest_error}


def validate_spectral_record(certificate, mat_sha256, mat_certificate=None, *, surface=None):
    """Recheck saved metadata under a fixed policy, without rerunning eigenvectors.

    The verifier chooses the protocol from the task (or a supported legacy schema
    for standalone calls). A certificate cannot select weaker counts or tolerances.
    """
    from turing_rd.spectral_policy import (SPHERE_CONFIGS, FROG_CONFIGS, FROG_SIGN_FLOOR,
                                          POSITIVE_FLOOR, STEADY_TOL, evaluate_spectra,
                                          finite_number)
    if not isinstance(certificate, dict):
        raise ValueError("spectral certificate must be an object")
    schema = certificate.get("schema_version")
    schemas = {"aic-frog-stability-v1": "sphere", "aic-frog-i03-frog-stability-v2": "frog"}
    if schema not in schemas or (surface is not None and schemas[schema] != surface):
        raise ValueError("unsupported or mismatched spectral protocol")
    frog = schemas[schema] == "frog"
    meta = certificate.get("input", {})
    if meta.get("mat_sha256") != mat_sha256:
        raise ValueError("spectral certificate MAT SHA256 mismatch")
    if certificate.get("mathematical_full_spectrum_proof") is not False:
        raise ValueError("finite sampling must not claim a full-spectrum proof")
    decision = certificate.get("decision", {})
    runs = certificate.get("configurations" if frog else "spectra", [])
    residual = meta.get("full_steady_residual", {})
    if (not isinstance(decision, dict) or not isinstance(residual, dict)
            or not isinstance(runs, list) or not all(isinstance(r, dict) for r in runs)):
        raise ValueError("malformed spectral evidence")
    from turing_rd.diagnostics import FULL_STEADY_RESIDUAL_FORMULA
    if (residual.get("formula") != FULL_STEADY_RESIDUAL_FORMULA
            or residual.get("normalization") != "F" or residual.get("rotation_projection") is not False
            or residual.get("value") != decision.get("full_residual")):
        raise ValueError("full stationary residual definition/binding mismatch")
    if mat_certificate is not None:
        for key in ("mat_sha256", "parameters", "seed", "geometry"):
            if meta.get(key) != mat_certificate.get(key):
                raise ValueError("spectral/MAT certificate " + key + " mismatch")
        recorded = mat_certificate.get("residual", {}).get("value")
        # Historical records differ by one float64 rounding unit. This is a
        # binding comparison, not a change to the fixed 1e-11 acceptance limit.
        value = residual.get("value")
        if (not finite_number(value) or not finite_number(recorded)
                or not math.isclose(value, recorded, rel_tol=16 * math.ulp(1.), abs_tol=0.)):
            raise ValueError("spectral/MAT certificate full residual mismatch")
    if (decision.get("status") != "numerical_check_passed"
            or decision.get("numerical_check_passed") is not True
            or decision.get("full_steady_state") is not True
            or decision.get("all_checked_signs_resolved_negative") is not True):
        raise ValueError("unprojected full steady/negative sample required")
    for record in (meta, decision, *runs):
        if (record.get("rotations_excluded_from_full_decision", False)
                or record.get("rotation_projection", False)
                or record.get("rotation", {}).get("used_to_project_jacobian", False)):
            raise ValueError("positive rotation-like modes cannot be excluded")
    declared_tol = decision.get("full_residual_tolerance", STEADY_TOL)
    if not finite_number(declared_tol) or not 0 < declared_tol <= STEADY_TOL:
        raise ValueError("untrusted full-residual tolerance")
    checked = evaluate_spectra(runs, residual.get("value"),
        configs=FROG_CONFIGS if frog else SPHERE_CONFIGS,
        scope="frog_formal_saved_metadata" if frog else "sphere_formal_saved_metadata",
        sign_floor=FROG_SIGN_FLOOR if frog else POSITIVE_FLOOR)
    if not checked["numerical_check_passed"]:
        raise ValueError(checked["status"] + ": " + ", ".join(checked["failure_reasons"]))
    for run in runs:
        if run.get("input", {}).get("mat_sha256", mat_sha256) != mat_sha256:
            raise ValueError("configuration MAT binding mismatch")
    if frog:
        attempt = certificate.get("unshifted_lr_attempt", {})
        if (attempt.get("attempted") is not True or attempt.get("solver_converged") is not False
                or attempt.get("used_as_pass_evidence") is not False):
            raise ValueError("failed frog direct-LR attempt must remain explicit")
    pairs = [pair for run in runs for pair in run["pairs"]]
    return {"status": checked["status"], "checked_pairs": checked["checked_eigenpairs"],
            "sampled_max_real": checked["full_max_real_checked"],
            "max_absolute_pair_residual": max(p["residual_absolute_M"] for p in pairs),
            "max_relative_pair_residual": max(p["residual_relative_M"] for p in pairs),
            "acceptance_policy_version": checked["acceptance_policy_version"],
            "evidence": "saved metadata checked; eigenvectors are not bundled",
            "mathematical_full_spectrum_proof": False}


def portable_acceptance(mat, cert, geom, exc, tol=1e-11, abs_agree=1e-12):
    """Fallback when ONLY the bit-level fingerprint of the assembled L/M differs (another CPU / BLAS /
    compiler rounds the cotangent weights differently in the last bits). Everything else is still
    checked strictly: the MAT SHA256, parameters, seed, the input mesh hash, vertex/face counts,
    the operator type and the total mass; the full residual is recomputed here with this machine's
    robust operator and must be <= tol and agree with the recorded value to abs_agree."""
    from turing_rd import GSParams
    from turing_rd.result_io import geometry_identity, validate_certificate_portable
    if str(exc) != "MAT certificate geometry mismatch":
        raise exc
    mine, rec = geometry_identity(geom), cert.get("geometry", {})
    for field in ("name", "n_vertices", "n_faces", "laplacian", "mesh_sha256"):
        if rec.get(field) != mine[field]:
            raise ValueError(f"MAT certificate geometry.{field} mismatch") from exc
    if not math.isclose(rec.get("mass_sum", float("nan")), mine["mass_sum"], rel_tol=1e-12):
        raise ValueError("MAT certificate geometry.mass_sum mismatch") from exc
    if cert.get("mat_sha256") != sha(mat):
        raise ValueError("MAT certificate mat_sha256 mismatch") from exc
    params = GSParams(**cert["parameters"])
    # Reuse every strict certificate check after the permitted operator mismatch.
    # A local residual alone cannot validate a certificate's formula, pass flags,
    # tolerance or claimed stability scope.
    try:
        fresh = validate_certificate_portable(mat, cert, expected_geometry=geom,
                                              expected_params=params, expected_seed=cert["seed"], tol=tol)
    except ValueError as error:
        raise ValueError("portable read-back acceptance failed: " + str(error)) from error
    recorded = cert.get("residual", {}).get("value")
    value = fresh["residual"]["value"]
    if not (fresh["accepted"] and cert.get("accepted") and recorded is not None and recorded <= tol
            and abs(value - recorded) <= abs_agree):
        raise ValueError("portable read-back acceptance failed: " + ", ".join(fresh["failure_reasons"])) from exc
    fresh["verification_mode"] = "portable: L/M bit fingerprint differs (floating-point rounding); mesh, mass and full residual rechecked"
    return fresh


def verify_best_match_steady(submission, geom=None):
    """Check A: actual robust MAT, parameters, hash and complete PDE residual."""
    from turing_rd import Geometry, GSParams, load_frog
    from turing_rd.result_io import validate_certificate_portable
    from turing_rd.spectral_policy import FROG_CONFIGS, FROG_SIGN_FLOOR, evaluate_spectra
    submission = Path(submission)
    directory = submission / "task3_inversion/best_match"
    summary = load(directory / "summary.json")
    estimate = load(submission / "task3_inversion/summary.json")["estimate"]
    params = GSParams(estimate["F"], estimate["k"])
    if summary["F"] != params.F or summary["k"] != params.k:
        raise ValueError("best-match parameters differ from formal inversion estimate")
    if geom is None:
        geom = Geometry(load_frog(submission / "code/frog.mat"), laplacian="robust", cache_dir=None)
    if geom.laplacian != "robust":
        raise ValueError("best-match acceptance requires the robust operator")
    cert = summary["mat_certificate"]
    fresh = validate_certificate_portable(directory / "solution_best_match.mat", cert, expected_params=params,
                                          expected_seed=summary["best"]["seed"], expected_geometry=geom)
    historical = evaluate_spectra(summary.get("stability", {}).get("configs", []),
        fresh["residual"]["value"], configs=FROG_CONFIGS, sign_floor=FROG_SIGN_FLOOR,
        scope="historical_best_match_summary_only")
    return {"available": True, "pde_residual_recomputed": True, "full_residual": fresh["residual"]["value"],
              "verification_mode": fresh.get("verification_mode", "strict: bit-identical operator"),
              "mat_sha256": fresh["mat_sha256"], "historical_spectrum_status": historical["status"],
              "historical_spectrum_failure_reasons": historical["failure_reasons"],
              "saved_spectral_metadata_verified": False, "eigenpair_residuals_recomputed_from_saved_vectors": False,
              "current_saved_vector_spectrum_status": "missing_evidence",
              "fresh_eigensolve_in_this_verification": False, "mathematical_full_spectrum_proof": False}


def verify_best_match(submission, geom=None, evidence=None, *, historical_compatibility=False):
    """Check B is mandatory by default; explicit legacy mode never yields formal PASS."""
    from turing_rd import Geometry, GSParams, load_frog
    submission = Path(submission); directory = submission / "task3_inversion/best_match"
    if historical_compatibility and not (directory / "solution_best_match.mat").is_file():
        return {"available": False, "status": "LIMITED_EVIDENCE",
                "historical_spectrum_status": "not_available",
                "current_saved_vector_spectrum_status": "missing_evidence"}
    if geom is None:
        geom = Geometry(load_frog(submission / "code/frog.mat"), laplacian="robust", cache_dir=None)
    result = verify_best_match_steady(submission, geom)
    summary = load(directory / "summary.json")
    estimate = load(submission / "task3_inversion/summary.json")["estimate"]
    params = GSParams(estimate["F"], estimate["k"])
    declared = summary.get("stability", {}).get("evidence_file")
    if evidence is not None:
        path = Path(evidence)
    elif declared:
        path = directory / declared
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("best-match evidence escapes its result directory")
    else:
        path = submission / "evidence/methodology/best_match_spectrum/certificate.json"
    if evidence is not None or declared or path.is_file() or not historical_compatibility:
        from turing_rd.spectrum_evidence import validate_evidence
        result.update(validate_evidence(path, directory / "solution_best_match.mat", geom, params,
                                         summary["best"]["seed"]))
    else:
        result["limitation"] = "historical spectral summaries lack per-pair residuals/vectors; not upgraded to a strict pass"
    result["status"] = "LIMITED_EVIDENCE" if historical_compatibility else "PASS"
    result["acceptance_protocol"] = "explicit_historical_compatibility" if historical_compatibility else "current_saved_vectors_v1"
    if not historical_compatibility and not current_saved_vectors_passed(result):
        raise ValueError("current best-match saved-vector evidence did not pass")
    return result


def current_saved_vectors_passed(result):
    """A historical insufficient-evidence status is never positive pass evidence."""
    return (result.get("pde_residual_recomputed") is True
            and result.get("current_saved_vector_spectrum_status") == "numerical_check_passed"
            and result.get("saved_spectral_metadata_verified") is True
            and result.get("eigenpair_residuals_recomputed_from_saved_vectors") is True
            and result.get("decision", {}).get("numerical_check_passed") is True)


def verify(submission, best_match_evidence=None, *, historical_compatibility=False):
    from turing_rd import Geometry, GSParams, icosphere, load_frog
    from turing_rd.result_io import validate_certificate
    from turing_rd.diagnostics import pattern_stats
    submission = Path(submission).resolve()
    checked = check_manifest(submission)
    geometries, results = {}, {}
    for task in TASKS:
        directory = submission / task
        summary = load(directory / "summary.json"); cert = summary["mat_certificate"]
        surface = task.split("/")[0]
        if surface not in geometries:
            mesh = (load_frog(submission / "code/frog.mat") if surface == "task2_frog" else icosphere(6))
            geometries[surface] = Geometry(mesh, laplacian="robust", cache_dir=None)
        geom = geometries[surface]
        try:
            fresh = validate_certificate(directory / "solution.mat", cert, expected_params=GSParams(**cert["parameters"]),
                                         expected_seed=cert["seed"], expected_geometry=geom)
            mode = "strict: bit-identical operator"
        except ValueError as exc:
            fresh = portable_acceptance(directory / "solution.mat", cert, geom, exc)
            mode = fresh["verification_mode"]
        spectral = validate_spectral_record(load(directory / "stability_certificate.json"),
            sha(directory / "solution.mat"), cert, surface="frog" if surface == "task2_frog" else "sphere")
        results[task] = {"mat_sha256": fresh["mat_sha256"], "full_residual": fresh["residual"]["value"],
                         "float64_accepted": fresh["accepted"], "verification_mode": mode, "spectrum": spectral}
    return {"status": "LIMITED_EVIDENCE" if historical_compatibility else "PASS",
            "tasks": results, "manifest_files": checked["manifest_files"],
            "best_match": verify_best_match(submission, geometries.get("task2_frog"), best_match_evidence,
                                             historical_compatibility=historical_compatibility)}


def prepare_models(submission, out):
    submission, out = Path(submission).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError("model output exists; nothing overwritten")
    sources = {"frog": submission / "task2_frog/ai_model",
               "sphere5": submission / "task1_sphere/ai_model"}
    for source in sources.values():
        for name in ("model.joblib", "dataset.json", "cv.json"):
            if not (source / name).is_file():
                raise FileNotFoundError(source / name)
    for surface, source in sources.items():
        shutil.copytree(source, out / surface)
    return {"status": "PASS", "models": {k: sha(out / k / "model.joblib") for k in sources}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "verify-zip", "prepare-models", "spectrum"))
    parser.add_argument("--submission", type=Path, required=True, help="directory containing PACKAGE_MANIFEST.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--zip", type=Path, help="built package ZIP, for verify-zip")
    parser.add_argument("--best-match-evidence", type=Path, help="independent task3 saved-vector certificate, read-only")
    parser.add_argument("--historical-compatibility", action="store_true", help="explicit legacy review; reports LIMITED_EVIDENCE, never PASS")
    args = parser.parse_args()
    if args.out and args.out.exists():
        parser.error("output exists; use a fresh path")
    if args.command == "prepare-models":
        if not args.out: parser.error("--out required")
        result = prepare_models(args.submission, args.out)
    elif args.command == "verify":
        result = verify(args.submission, args.best_match_evidence, historical_compatibility=args.historical_compatibility)
    elif args.command == "verify-zip":
        if not args.zip:
            parser.error("--zip required")
        result = compare_zip_to_workspace(args.submission, args.zip)
    else:
        if not args.task or not args.out: parser.error("--task and --out required")
        import numpy as np
        from turing_rd import Geometry, GSParams, icosphere, load_frog
        from turing_rd.result_io import read_mat_fields, validate_certificate_portable, geometry_identity
        from turing_rd.stability import solve_spectrum
        directory = args.submission / args.task; cert = load(directory / "summary.json")["mat_certificate"]
        mesh = load_frog(args.submission / "code/frog.mat") if args.task.startswith("task2") else icosphere(6)
        geom = Geometry(mesh, laplacian="robust", cache_dir=None); params = GSParams(**cert["parameters"])
        fresh = validate_certificate_portable(directory / "solution.mat", cert, expected_params=params, expected_seed=cert["seed"], expected_geometry=geom)
        u, v = read_mat_fields(directory / "solution.mat", geom.n)
        report, vectors = solve_spectrum(geom, params, u, v, k=8, sigma=0, which="LM")
        args.out.mkdir(parents=True, exist_ok=False)
        with (args.out / "eigenpairs.npz").open("xb") as stream:
            np.savez_compressed(stream, vectors=vectors)
        result = {"mat_sha256": sha(directory / "solution.mat"), "geometry": geometry_identity(geom),
                  "full_residual": fresh["residual"]["value"], "spectrum": report,
                  "eigenvectors_sha256": sha(args.out / "eigenpairs.npz"),
                  "scope": "new single-shift sample, does not replace the saved three-configuration certificate"}
        (args.out / "spectrum.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 1 if isinstance(result, dict) and result.get("status") in ("FAIL", "LIMITED_EVIDENCE") else 0


if __name__ == "__main__":
    raise SystemExit(main())
