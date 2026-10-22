"""I03: independent finite-sample spectral check of the two formal frog MATs (v2).

The I02 stability machinery was built for the sphere (rotation-relative equilibria).
The frog surface has no continuous rotation symmetry, so this script reuses the same
Jacobian, mass-weighted similarity, ARPACK solver, eigenpair residual validation and
conservative sign policy, but applies **no rotation labelling or projection**: every
sampled eigenpair counts.

Configuration set (all shift-invert, three distinct shifts bracketing the rightmost
cluster):

    sigma = +1e-3, k = 12   (same setting as I02)
    sigma =  0,    k = 20
    sigma = +1e-5, k = 8    (sits above the sampled maximum; any positive mode would be
                             closer to this shift than the rightmost negative mode)

An unshifted ``which='LR'`` search was attempted first (run v1, preserved) and did not
converge on this operator; that failure is recorded in ``unshifted_lr_attempt`` and is
not used as passing evidence.  A finite sample with k << dim(J) is numerical evidence
only, never a proof of the full spectrum.

    python scripts/i03_frog_stability.py --out results/i03_delivery/frog_stability_v2
"""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for path in (str(ROOT), str(HERE)):
    if str(path) not in sys.path:
        sys.path.insert(0, path)

import numpy as np                                                  # noqa: E402
import scipy.sparse as sp                                           # noqa: E402
import scipy.sparse.linalg as spla                                  # noqa: E402

from json_safe import write_json_exclusive                          # noqa: E402
from turing_rd import Geometry, load_frog                            # noqa: E402
from turing_rd.diagnostics import jacobian                          # noqa: E402
from turing_rd.result_io import (file_sha256, geometry_identity, read_mat_fields,
                                 validate_certificate)              # noqa: E402
from turing_rd.stability import (PAIR_ABSOLUTE_TOL, PAIR_RELATIVE_TOL,             # noqa: E402
                                 SPECTRUM_LIMITATION, aggregate_spectra,
                                 eigenpair_diagnostics)
from turing_rd.solver import GSParams                               # noqa: E402

I01 = ROOT / "results" / "i01_float64_verified" / "tasks"
V1 = ROOT / "results" / "i03_delivery" / "frog_stability"
TASKS = {
    "frog_spots": dict(summary=I01 / "frog_spots" / "summary.json",
                       mat=I01 / "frog_spots" / "solution.mat"),
    "frog_stripes": dict(summary=I01 / "frog_stripes" / "summary.json",
                         mat=I01 / "frog_stripes" / "solution.mat"),
}
CONFIGS = (
    dict(name="sigma_1e-3_k12", sigma=1e-3, which="LM", k=12, ncv=48),
    dict(name="sigma_0_k20", sigma=0.0, which="LM", k=20, ncv=48),
    dict(name="sigma_1e-5_k8", sigma=1e-5, which="LM", k=8, ncv=48),
)
STEADY_TOL = 1e-11
SIGN_FLOOR = 1e-9
EVIDENCE_LEVEL = "i03_frog_three_shift_invert_bracketing_no_rotation_label"
CODE = ("scripts/i03_frog_stability.py", "scripts/json_safe.py", "turing_rd/stability.py",
        "turing_rd/diagnostics.py", "turing_rd/result_io.py", "turing_rd/geometry.py",
        "turing_rd/mesh.py", "turing_rd/solver.py", "turing_rd/steady.py")


def log(message):
    print(message, flush=True)


def solve_config(J, mass, root, cfg, tol=1e-10, maxiter=3000, seed=2026):
    """One ARPACK sample of the mass-similarity operator, without rotation labelling."""
    started = time.perf_counter()
    dimension = J.shape[0]
    if not 1 <= cfg["k"] < dimension - 1:
        raise ValueError("ARPACK requires 1 <= k < dimension-1")
    B = (sp.diags(root) @ J @ sp.diags(1.0 / root)).tocsc()
    v0 = np.random.default_rng(seed).standard_normal(dimension)
    v0 /= np.linalg.norm(v0)
    converged, error = True, None
    try:
        lam, weighted = spla.eigs(B, k=cfg["k"], sigma=cfg["sigma"], which=cfg["which"],
                                  tol=tol, maxiter=maxiter, ncv=cfg["ncv"], v0=v0)
    except spla.ArpackNoConvergence as exc:
        converged, error = False, str(exc)
        lam = np.asarray(exc.eigenvalues) if exc.eigenvalues is not None else np.empty(0, complex)
        weighted = (np.asarray(exc.eigenvectors) if exc.eigenvectors is not None
                    else np.empty((dimension, 0), complex))
    order = np.argsort(-lam.real)
    lam, weighted = lam[order], weighted[:, order]
    W = weighted / root[:, None]
    pairs = eigenpair_diagnostics(J, lam, W, mass, rotation_basis=None)
    absolute = [p["residual_absolute_M"] for p in pairs]
    report = {
        "diagnostic_version": "i03-frog-unprojected-spectrum-v2",
        "requested_k": cfg["k"], "returned_k": int(len(lam)),
        "solver_converged": bool(converged and len(lam) == cfg["k"]),
        "solver_error": error,
        "method": ("mass_similarity_arpack_shift_invert" if cfg["sigma"] is not None
                   else "mass_similarity_arpack_direct"),
        "which": cfg["which"], "sigma": cfg["sigma"], "ncv": cfg["ncv"],
        "arpack_tolerance": tol, "arpack_maxiter": maxiter, "v0_seed": seed,
        "similarity": "B=sqrt(Mfull)*J*invsqrt(Mfull); J=Mfull^-1*weak_J",
        "rotation_labelling": "none: the frog mesh has no continuous rotation symmetry",
        "pairs": pairs,
        "full_max_real": float(lam.real.max()) if len(lam) else None,
        "all_pairs_validated": bool(pairs and all(p["residual_passed"] for p in pairs)),
        "found_reliable_positive": any(p["positive_reliable"] for p in pairs),
        "max_residual_absolute_M": max(absolute) if absolute else None,
        "pair_absolute_tolerance": PAIR_ABSOLUTE_TOL,
        "pair_relative_tolerance": PAIR_RELATIVE_TOL,
        "wall_seconds": time.perf_counter() - started,
        "limitation": SPECTRUM_LIMITATION,
    }
    return report, W


def decide_frog(reports, full_residual, tol=STEADY_TOL):
    """Conservative frog decision: three validated shift-invert configurations bracketing zero."""
    pairs = [pair for report in reports for pair in report["pairs"]]
    positive = [pair for pair in pairs if pair["positive_reliable"]]
    negatives = [pair for pair in pairs if pair["negative_resolved"]]
    unresolved = [pair for pair in pairs if pair["residual_passed"]
                  and abs(pair["eigenvalue"][0]) <= SIGN_FLOOR]
    complete = bool(reports) and all(r["solver_converged"] and r["all_pairs_validated"] for r in reports)
    shifts = sorted({r["sigma"] for r in reports if r["sigma"] is not None
                     and r["solver_converged"] and r["all_pairs_validated"]})
    maxima = [r["full_max_real"] for r in reports if r["full_max_real"] is not None and r["solver_converged"]]
    residuals = [p["residual_absolute_M"] for p in pairs if p["residual_passed"]]
    agreement_tolerance = max(1e-10, 100 * max(residuals, default=0.0))
    maxima_agree = bool(maxima and (max(maxima) - min(maxima)) <= agreement_tolerance)
    steady = bool(full_residual <= tol)
    reasons = []
    if not steady:
        reasons.append("full_stationary_residual_failed")
    if not complete:
        reasons.append("solver_or_pair_validation_incomplete")
    if len(shifts) < 3:
        reasons.append("fewer_than_three_valid_distinct_shifts")
    if not maxima_agree:
        reasons.append("sampled_largest_real_parts_disagree")
    if positive:
        reasons.append("reliable_positive_full_jacobian_eigenvalue")
    elif unresolved or len(negatives) != len(pairs):
        reasons.append("not_all_checked_eigenvalue_signs_resolved_negative")
    passed = bool(steady and complete and len(shifts) >= 3 and maxima_agree and
                  not positive and not unresolved and len(negatives) == len(pairs))
    return {
        "diagnostic_version": "i03-frog-unprojected-spectrum-v2",
        "rule": ("numeric pass requires a complete steady state, three distinct validated "
                 "shift-invert configurations, agreeing sampled maxima and every sampled "
                 "eigenvalue resolved negative; any reliable positive eigenvalue fails the check"),
        "full_steady_state": steady,
        "full_residual": float(full_residual), "full_residual_tolerance": tol,
        "configurations": len(reports), "distinct_valid_shifts": shifts,
        "sampled_maxima": maxima, "sampled_maxima_agree": maxima_agree,
        "sampled_maxima_agreement_tolerance": agreement_tolerance,
        "checked_eigenpairs": len(pairs),
        "reliable_positive_eigenvalues": [p["eigenvalue"] for p in positive],
        "unresolved_near_zero_eigenvalues": [p["eigenvalue"] for p in unresolved],
        "all_checked_signs_resolved_negative": len(negatives) == len(pairs) and not unresolved,
        "numerical_check_passed": passed,
        "status": ("unstable_positive_mode" if positive else
                   ("numerical_check_passed" if passed else "insufficient_evidence")),
        "failure_reasons": reasons,
        "rotation_projection": False,
        "mathematical_full_spectrum_proof": False,
        "limitation": SPECTRUM_LIMITATION,
    }


def load_lr_attempt(task):
    """Record the v1 unshifted largest-real-part attempt without using it as pass evidence."""
    path = V1 / task / "spectrum_direct_LR_k8.json"
    if not path.is_file():
        return {"attempted": False}
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        "attempted": True,
        "file": str(path.relative_to(ROOT)), "sha256": file_sha256(path),
        "solver_converged": report.get("solver_converged"),
        "returned_k": report.get("returned_k"), "requested_k": report.get("requested_k"),
        "full_max_real_of_partial_result": report.get("full_max_real"),
        "solver_error": report.get("solver_error"),
        "used_as_pass_evidence": False,
        "note": ("The unshifted largest-real-part ARPACK search is ill-conditioned on this stiff "
                 "operator (spectral radius >> rightmost eigenvalue gap); it did not converge and "
                 "is therefore not used to award the numerical pass."),
    }


def run_task(task, out_root, tol=STEADY_TOL):
    spec = TASKS[task]
    out = out_root / task
    out.mkdir(parents=True, exist_ok=False)
    started = datetime.datetime.now(datetime.timezone.utc)
    summary = json.loads(spec["summary"].read_text(encoding="utf-8"))
    cert = summary["mat_certificate"]
    params = GSParams(**cert["parameters"])
    log(f"[{task}] building robust frog geometry and validating the I01 certificate")
    geom = Geometry(load_frog(ROOT / "frog.mat"), laplacian="robust", cache_dir=None)
    fresh = validate_certificate(spec["mat"], cert, expected_params=params,
                                 expected_seed=summary["seed"], expected_geometry=geom,
                                 tol=STEADY_TOL)
    u, v = read_mat_fields(spec["mat"], geom.n)
    mass = np.concatenate([geom.M, geom.M])
    root = np.sqrt(mass)
    log(f"[{task}] full steady residual={fresh['residual']['value']:.6e}; "
        f"building J ({2 * geom.n} x {2 * geom.n})")
    build_started = time.perf_counter()
    J = jacobian(geom, params, u, v)
    build_seconds = time.perf_counter() - build_started
    reports, artifacts = [], []
    for cfg in CONFIGS:
        report, W = solve_config(J, mass, root, cfg)
        values = np.asarray([complex(*p["eigenvalue"]) for p in report["pairs"]])
        npz = out / f"eigenpairs_{cfg['name']}.npz"
        np.savez_compressed(npz, eigenvalues=values, eigenvectors=W)
        report["eigenpairs_file"] = npz.name
        report["eigenpairs_sha256"] = file_sha256(npz)
        write_json_exclusive(out / f"spectrum_{cfg['name']}.json", report)
        reports.append(report)
        artifacts.append({"config": cfg["name"], "file": f"spectrum_{cfg['name']}.json",
                          "eigenpairs": npz.name, "eigenpairs_sha256": report["eigenpairs_sha256"]})
        maximum = report["full_max_real"]
        maximum_text = "n/a" if maximum is None else f"{maximum:+.12e}"
        residual = report["max_residual_absolute_M"]
        residual_text = "n/a" if residual is None else f"{residual:.3e}"
        log(f"[{task}] {cfg['name']}: converged={report['solver_converged']} "
            f"k={report['returned_k']}/{report['requested_k']} max Re lambda={maximum_text} "
            f"max |residual|={residual_text} ({report['wall_seconds']:.1f} s)")
    decision = decide_frog(reports, fresh["residual"]["value"], tol=STEADY_TOL)
    reference = aggregate_spectra(reports, fresh["residual"]["value"], tol=STEADY_TOL)
    meta = {
        "mat_file": str(spec["mat"].relative_to(ROOT)),
        "mat_sha256": file_sha256(spec["mat"]),
        "summary_file": str(spec["summary"].relative_to(ROOT)),
        "summary_sha256": file_sha256(spec["summary"]),
        "parameters": params.as_dict(),
        "seed": int(summary["seed"]),
        "geometry": geometry_identity(geom),
        "full_steady_residual": fresh["residual"],
        "I01_accepted": bool(fresh["accepted"]),
        "timestamp_utc": started.isoformat(),
        "runtime": {"python": sys.version, "numpy": np.__version__,
                    **{name: importlib.metadata.version(name)
                       for name in ("scipy", "robust-laplacian")}},
        "code_sha256": {name: file_sha256(ROOT / name) for name in CODE},
        "mass_convention": "J=Mfull^-1*weak_J; B=sqrt(Mfull)*J*invsqrt(Mfull), same eigenvalues",
        "rotation_projection": False,
        "rotation_modes": ("not_applicable: the frog mesh has no continuous rotation symmetry; "
                           "no rotation basis, labelling or exclusion was used"),
        "steady_full": bool(fresh["residual"]["value"] <= STEADY_TOL),
        "jacobian_wall_seconds": build_seconds,
        "evidence_level": EVIDENCE_LEVEL,
        "sign_floor": SIGN_FLOOR,
    }
    certificate = {
        "schema_version": "aic-frog-i03-frog-stability-v2",
        "scope": "unprojected_discrete_jacobian_finite_numerical_sample_no_rotation_label",
        "input": meta,
        "configurations": reports,
        "artifacts": artifacts,
        "decision": decision,
        "unshifted_lr_attempt": load_lr_attempt(task),
        "reference_i02_rule": {
            "decision": reference,
            "note": ("The I02 rule requires a converged direct largest-real-part run. Applied to these "
                     "reports it can only say stability_not_closed; the frog rule above substitutes a "
                     "third, positive-side shift-invert configuration and records the failed attempt."),
        },
        "perturbation": None,
        "perturbation_required_for_spectral_check": False,
        "mathematical_full_spectrum_proof": False,
        "unverified_risks": [
            "Finite Krylov sampling (k=8..20 of 58020) does not exhaust the full spectrum.",
            "Eigenpair residual is a backward error, not a rigorous forward eigenvalue bound.",
            "The unshifted largest-real-part search is not usable on this stiff operator; the third "
            "configuration is a positive-side shift-invert bracket instead.",
            "No time-domain perturbation test and no global nonlinear stability claim.",
            "The rotation-relative-equilibrium machinery of I02 is intentionally not used on frog.",
        ],
    }
    write_json_exclusive(out / "frog_stability_certificate.json", certificate)
    (out / "run.log").write_text(
        f"task={task}\nmat={meta['mat_file']}\nmat_sha256={meta['mat_sha256']}\n"
        f"full_residual={fresh['residual']['value']:.12e}\n"
        f"decision={decision['status']}\nconfigs={len(reports)}\n"
        f"lr_attempt={certificate['unshifted_lr_attempt'].get('solver_converged')}\n",
        encoding="utf-8")
    log(f"[{task}] decision={decision['status']} "
        f"positives={len(decision['reliable_positive_eigenvalues'])} "
        f"max Re lambda={decision['sampled_maxima'][0]:+.12e}")
    return {"task": task, "mat_sha256": meta["mat_sha256"],
            "full_residual": fresh["residual"]["value"], "status": decision["status"],
            "max_real": decision["sampled_maxima"][0], "passed": decision["numerical_check_passed"],
            "certificate": str(out.relative_to(ROOT) / "frog_stability_certificate.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASKS) + ("both",), default="both")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results" / "i03_delivery" / "frog_stability_v2")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = tuple(TASKS) if args.task == "both" else (args.task,)
    results = []
    for task in tasks:
        results.append(run_task(task, args.out))
    write_json_exclusive(args.out / "summary.json",
                         {"schema_version": "aic-frog-i03-frog-stability-summary-v2",
                          "tasks": results, "mathematical_full_spectrum_proof": False})
    print(json.dumps(results, indent=2), flush=True)
    return 0 if all(r["status"] in ("numerical_check_passed", "unstable_positive_mode") for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
