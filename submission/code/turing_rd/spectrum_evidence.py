"""MAT-bound finite spectral evidence with saved vectors; never a full-spectrum proof."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

from .diagnostics import jacobian
from .result_io import file_sha256, read_mat_fields, validate_certificate, validate_certificate_portable
from .spectral_policy import (FROG_CONFIGS, FROG_SIGN_FLOOR, PAIR_ABSOLUTE_TOL, PAIR_RELATIVE_TOL,
                              POLICY_VERSION, evaluate_spectra, failed_spectrum)
from .stability import eigenpair_diagnostics
from .experiment_cache import code_identity, dependency_versions

EVIDENCE_VERSION = "mat-bound-eigenvectors-v1"
# Eigenpair residuals sit at the round-off floor (~1e-15), so another numpy / BLAS / L-M rounding recomputes
# them as other round-off-floor values (measured: 6e-15 relative on x86; up to 13% relative, 4e-16 absolute
# on ARM). Only differences far below the pass thresholds are accepted; the stability decision itself is
# re-derived from the recomputed numbers below, so a pair that fails here-and-now still fails.
PAIR_RECOMPUTE_RTOL = 1e-9
PAIR_RECOMPUTE_ATOL = {"residual_absolute_M": 1e-3 * PAIR_ABSOLUTE_TOL, "residual_relative_M": 1e-3 * PAIR_RELATIVE_TOL}


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(_clean(value), stream, indent=2, allow_nan=False)


def _decide(reports, residual):
    return evaluate_spectra(reports, residual, configs=FROG_CONFIGS,
                            scope="task3_three_shift_saved_vector_finite_check", sign_floor=FROG_SIGN_FLOOR)


def save_evidence(mat, mat_certificate, geom, params, out, solve):
    """Only a real solver call can create vectors. Partial failures stay failures."""
    out, mat = Path(out), Path(mat)
    if out.exists():
        raise FileExistsError("spectral evidence output exists")
    fresh = validate_certificate(mat, mat_certificate, expected_params=params,
                                 expected_seed=mat_certificate["seed"], expected_geometry=geom)
    if not fresh["accepted"]:
        raise ValueError("full stationary MAT residual failed")
    u, v = read_mat_fields(mat, geom.n)
    J = jacobian(geom, params, u, v)
    mass = np.concatenate([geom.M, geom.M]); root = np.sqrt(mass)
    out.mkdir(parents=True, exist_ok=False)
    reports, artifacts = [], []
    for i, cfg in enumerate(FROG_CONFIGS):
        config = dict(name=f"configuration_{i}", sigma=cfg.sigma, which=cfg.which, k=cfg.k, ncv=48)
        try:
            report, vectors = solve(J, mass, root, config)
            values = np.asarray([complex(*p["eigenvalue"]) for p in report["pairs"]], dtype=np.complex128)
            vectors = np.asarray(vectors)
            if vectors.shape != (J.shape[0], len(values)):
                raise ValueError("returned eigenvector dimensions disagree")
            report = copy.deepcopy(report)
            report["pairs"] = eigenpair_diagnostics(J, values, vectors, mass)
            report["all_pairs_validated"] = bool(report["pairs"] and all(p["residual_passed"] for p in report["pairs"]))
            report["found_reliable_positive"] = any(p["positive_reliable"] for p in report["pairs"])
            report["metrics_recomputed_from_returned_vectors"] = True
        except Exception as exc:
            report = failed_spectrum(cfg, repr(exc))
            values, vectors = np.empty(0, complex), np.empty((J.shape[0], 0), complex)
        report["input"] = {"mat_sha256": fresh["mat_sha256"]}
        npz = out / f"eigenpairs_{i}.npz"
        with npz.open("xb") as stream:
            np.savez_compressed(stream, eigenvalues=values, eigenvectors=vectors)
        report_file = out / f"spectrum_{i}.json"
        write_json(report_file, report)
        report = _clean(report); reports.append(report)
        artifacts.append({"report_file": report_file.name, "report_sha256": file_sha256(report_file),
                          "vectors_file": npz.name, "vectors_sha256": file_sha256(npz)})
        print(f"spectrum {i}: {report.get('returned_k')}/{cfg.k}, max_real={report.get('full_max_real')}, "
              f"converged={report.get('solver_converged')}", flush=True)
    result = {"schema_version": EVIDENCE_VERSION, "acceptance_policy_version": POLICY_VERSION,
              "provenance": {"code": code_identity(Path(__file__).resolve().parents[1]),
                             "environment": dependency_versions()},
              "input": {key: fresh[key] for key in ("mat_sha256", "parameters", "seed", "geometry")},
              "mat_certificate": fresh, "configs": reports, "artifacts": artifacts,
              "decision": _decide(reports, fresh["residual"]["value"]),
              "eigenvectors_saved": True, "fresh_eigensolves_executed": True,
              "mathematical_full_spectrum_proof": False,
              "limitation": "Three shift-invert samples; no independent unshifted rightmost-spectrum proof."}
    write_json(out / "certificate.json", result)
    return result


def validate_evidence(certificate_path, mat, geom, params, seed, *, recompute_pairs=True):
    """Independent MAT readback and optional Jw-lambda*w recomputation, not a new eigensolve."""
    path = Path(certificate_path); certificate = json.loads(path.read_text(encoding="utf-8"))
    if certificate.get("schema_version") != EVIDENCE_VERSION or certificate.get("acceptance_policy_version") != POLICY_VERSION:
        raise ValueError("unknown eigenvector evidence protocol")
    if certificate.get("mathematical_full_spectrum_proof") is not False:
        raise ValueError("finite evidence cannot claim a full-spectrum proof")
    fresh = validate_certificate_portable(mat, certificate["mat_certificate"], expected_params=params,
                                          expected_seed=seed, expected_geometry=geom)
    # the recorded geometry was just bound to this mesh (bit-identical, or up to L/M round-off on another machine)
    bound = dict(fresh, geometry=certificate["mat_certificate"]["geometry"])
    for key in ("mat_sha256", "parameters", "seed", "geometry"):
        if certificate.get("input", {}).get(key) != bound[key]:
            raise ValueError("spectral evidence MAT binding mismatch: " + key)
    reports = certificate.get("configs", []); artifacts = certificate.get("artifacts", [])
    if len(reports) != len(FROG_CONFIGS) or len(artifacts) != len(FROG_CONFIGS):
        raise ValueError("incomplete evidence configurations")
    # the saved decision is bound to the recorded residual, which validate_certificate_portable has just tied to
    # this machine's read-back; the final decision below uses the read-back value itself
    if certificate.get("decision") != _decide(reports, certificate["mat_certificate"]["residual"]["value"]):
        raise ValueError("saved spectral decision does not match fixed policy")
    mass = np.concatenate([geom.M, geom.M])
    if recompute_pairs:
        u, v = read_mat_fields(mat, geom.n); J = jacobian(geom, params, u, v)
    verified = []
    for report, artifact in zip(reports, artifacts):
        for name, digest in (("report_file", "report_sha256"), ("vectors_file", "vectors_sha256")):
            relative = artifact[name]
            if not isinstance(relative, str) or Path(relative).name != relative or relative in ("", ".", ".."):
                raise ValueError("invalid evidence artifact path")
            if file_sha256(path.parent / relative) != artifact[digest]:
                raise ValueError("spectral artifact SHA256 mismatch")
        saved_report = json.loads((path.parent / artifact["report_file"]).read_text(encoding="utf-8"))
        if saved_report != report:
            raise ValueError("certificate/report mismatch")
        if report.get("input", {}).get("mat_sha256") != fresh["mat_sha256"]:
            raise ValueError("configuration MAT binding mismatch")
        checked = copy.deepcopy(report)
        if recompute_pairs:
            with np.load(path.parent / artifact["vectors_file"], allow_pickle=False) as data:
                values, vectors = data["eigenvalues"], data["eigenvectors"]
            if values.dtype != np.complex128 or vectors.dtype not in (np.dtype("complex128"), np.dtype("float64")):
                raise ValueError("eigenpairs must preserve float64/complex128 precision")
            if vectors.shape != (2 * geom.n, len(values)) or len(values) != report.get("returned_k"):
                raise ValueError("saved eigenpair count/shape mismatch")
            pairs = eigenpair_diagnostics(J, values, vectors, mass)
            if len(pairs) != len(report.get("pairs", [])):
                raise ValueError("saved eigenpair metric count mismatch")
            for a, b in zip(pairs, report["pairs"]):
                if a["eigenvalue"] != b.get("eigenvalue"):
                    raise ValueError("eigenvalues do not match saved metrics")
                for key in ("residual_absolute_M", "residual_relative_M"):
                    old = b.get(key)
                    if not isinstance(old, (float, int)) or not math.isclose(
                            a[key], old, rel_tol=PAIR_RECOMPUTE_RTOL, abs_tol=PAIR_RECOMPUTE_ATOL[key]):
                        raise ValueError("eigenpair residual does not match actual saved vector")
            checked["pairs"] = pairs
        verified.append(checked)
    decision = _decide(verified, fresh["residual"]["value"])
    if not decision["numerical_check_passed"]:
        raise ValueError("spectral evidence insufficient: " + ", ".join(decision["failure_reasons"]))
    return {"pde_residual_recomputed": True, "full_residual": fresh["residual"]["value"],
            "current_saved_vector_spectrum_status": "numerical_check_passed" if recompute_pairs else "metadata_only_not_current_acceptance",
            "mat_sha256": fresh["mat_sha256"], "saved_spectral_metadata_verified": True,
            "eigenpair_residuals_recomputed_from_saved_vectors": bool(recompute_pairs),
            "fresh_eigensolve_in_this_verification": False, "decision": decision,
            "mathematical_full_spectrum_proof": False}
