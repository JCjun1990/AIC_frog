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


def validate_spectral_record(certificate, mat_sha256):
    """Validate saved evidence, without claiming to have rerun ARPACK."""
    if certificate["input"]["mat_sha256"] != mat_sha256:
        raise ValueError("spectral certificate MAT SHA256 mismatch")
    if certificate.get("mathematical_full_spectrum_proof") is not False:
        raise ValueError("finite sampling must not claim a full-spectrum proof")
    decision = certificate["decision"]
    runs = certificate.get("configurations", certificate.get("spectra", []))
    if len(runs) != 3 or decision.get("status") != "numerical_check_passed":
        raise ValueError("three passing numerical configurations required")
    if not decision.get("full_steady_state") or not decision.get("all_checked_signs_resolved_negative"):
        raise ValueError("unprojected full steady/negative sample required")
    if decision.get("rotations_excluded_from_full_decision", False) or decision.get("rotation_projection", False):
        raise ValueError("positive rotation-like modes cannot be excluded")
    maxima, pair_count, abs_res, rel_res = [], 0, [], []
    for run in runs:
        pairs = run["pairs"]
        if not run.get("solver_converged") or len(pairs) != run["requested_k"] or run["returned_k"] != len(pairs):
            raise ValueError("incomplete eigensolve")
        for pair in pairs:
            ev = pair["eigenvalue"]
            a, r = pair["residual_absolute_M"], pair["residual_relative_M"]
            if (not all(math.isfinite(x) for x in (*ev, a, r)) or a < 0 or r < 0
                    or a > run["pair_absolute_tolerance"] or r > run["pair_relative_tolerance"]
                    or not pair["residual_passed"] or not pair["negative_resolved"]
                    or pair["positive_reliable"] or ev[0] >= 0):
                raise ValueError("invalid or nonnegative eigenpair")
            abs_res.append(a); rel_res.append(r)
        maximum = max(x["eigenvalue"][0] for x in pairs)
        if not math.isclose(maximum, run["full_max_real"], rel_tol=1e-8, abs_tol=1e-12):
            raise ValueError("sampled maximum contradicts eigenpairs")
        if run.get("input", {}).get("mat_sha256", mat_sha256) != mat_sha256:
            raise ValueError("configuration MAT binding mismatch")
        maxima.append(maximum); pair_count += len(pairs)
    tolerance = decision.get("sampled_maxima_agreement_tolerance", 1e-10)
    if max(maxima) - min(maxima) > tolerance:
        raise ValueError("sampled maxima disagree")
    frog = "configurations" in certificate
    if frog:
        if len({run["sigma"] for run in runs}) != 3:
            raise ValueError("frog requires three distinct shifts")
        attempt = certificate.get("unshifted_lr_attempt", {})
        if not attempt.get("attempted") or attempt.get("solver_converged") is not False or attempt.get("used_as_pass_evidence") is not False:
            raise ValueError("failed frog direct-LR attempt must remain explicit")
    elif not any(run["which"] == "LR" and run["sigma"] is None for run in runs):
        raise ValueError("sphere requires saved converged direct LR evidence")
    return {"status": "numerical_check_passed", "checked_pairs": pair_count,
            "sampled_max_real": max(maxima), "max_absolute_pair_residual": max(abs_res),
            "max_relative_pair_residual": max(rel_res),
            "evidence": "saved metadata checked; eigenvectors are not bundled",
            "mathematical_full_spectrum_proof": False}


def portable_acceptance(mat, cert, geom, exc, tol=1e-11, abs_agree=1e-12):
    """Fallback when ONLY the bit-level fingerprint of the assembled L/M differs (another CPU / BLAS /
    compiler rounds the cotangent weights differently in the last bits). Everything else is still
    checked strictly: the MAT SHA256, parameters, seed, the input mesh hash, vertex/face counts,
    the operator type and the total mass; the full residual is recomputed here with this machine's
    robust operator and must be <= tol and agree with the recorded value to abs_agree."""
    from turing_rd import GSParams
    from turing_rd.result_io import geometry_identity, verify_mat
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
    fresh = verify_mat(mat, geom, params, seed=cert["seed"], tol=tol, provenance=cert.get("provenance"))
    recorded = cert.get("residual", {}).get("value")
    value = fresh["residual"]["value"]
    if not (fresh["accepted"] and cert.get("accepted") and recorded is not None and recorded <= tol
            and abs(value - recorded) <= abs_agree):
        raise ValueError("portable read-back acceptance failed: " + ", ".join(fresh["failure_reasons"])) from exc
    fresh["verification_mode"] = "portable: L/M bit fingerprint differs (floating-point rounding); mesh, mass and full residual rechecked"
    return fresh


def verify(submission):
    from turing_rd import Geometry, GSParams, icosphere, load_frog
    from turing_rd.result_io import validate_certificate
    from turing_rd.diagnostics import pattern_stats
    submission = Path(submission).resolve()
    manifest = load(submission / "PACKAGE_MANIFEST.json")
    listed = manifest["files"]
    if len({r["path"] for r in listed}) != len(listed):
        raise ValueError("duplicate manifest paths")
    for entry in listed:
        target = (submission / entry["path"]).resolve()
        if not target.is_relative_to(submission) or not target.is_file() or sha(target) != entry["sha256"]:
            raise ValueError("manifest hash/path mismatch: " + entry["path"])
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
        spectral = validate_spectral_record(load(directory / "stability_certificate.json"), sha(directory / "solution.mat"))
        results[task] = {"mat_sha256": fresh["mat_sha256"], "full_residual": fresh["residual"]["value"],
                         "float64_accepted": fresh["accepted"], "verification_mode": mode, "spectrum": spectral}
    return {"status": "PASS", "tasks": results, "manifest_files": len(listed)}


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
    parser.add_argument("command", choices=("verify", "prepare-models", "spectrum"))
    parser.add_argument("--submission", type=Path, required=True, help="directory containing PACKAGE_MANIFEST.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--task", choices=TASKS)
    args = parser.parse_args()
    if args.out and args.out.exists():
        parser.error("output exists; use a fresh path")
    if args.command == "prepare-models":
        if not args.out: parser.error("--out required")
        result = prepare_models(args.submission, args.out)
    elif args.command == "verify":
        result = verify(args.submission)
    else:
        if not args.task or not args.out: parser.error("--task and --out required")
        import numpy as np
        from turing_rd import Geometry, GSParams, icosphere, load_frog
        from turing_rd.result_io import read_mat_fields, validate_certificate, geometry_identity
        from turing_rd.stability import solve_spectrum
        directory = args.submission / args.task; cert = load(directory / "summary.json")["mat_certificate"]
        mesh = load_frog(args.submission / "code/frog.mat") if args.task.startswith("task2") else icosphere(6)
        geom = Geometry(mesh, laplacian="robust", cache_dir=None); params = GSParams(**cert["parameters"])
        fresh = validate_certificate(directory / "solution.mat", cert, expected_params=params, expected_seed=cert["seed"], expected_geometry=geom)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
