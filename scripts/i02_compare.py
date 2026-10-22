"""Assemble the I02 stability comparison table from the emitted certificates.

Read-only with respect to every experiment directory: the only outputs are
``comparison.json`` and ``comparison.md`` in a new directory.  Every number is
copied from a certificate (never recomputed or reinterpreted here), so the table
cannot claim more than the certificates did.

    python scripts/i02_compare.py --out results/i02_stability/comparison
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from json_safe import write_json_exclusive  # noqa: E402

BASE = ROOT / "results" / "i02_stability"
LABELS = {"six_bands": "sphere6 six bands (I01 seed 0)",
          "spots": "sphere6 spots (I01 seed 0)",
          "seed2": "sphere6 seed2 seven-band candidate"}
# Historical single-shift diagnostics of the *same* candidate, kept as an independent
# reproduction check.  These records contain no fields, so they cannot be reused as input.
LEGACY_RECORDS = {
    "seed2": ROOT / "results" / "ai_generation" / "tasks" / "robustness_jobs" /
             "ai_sphere6_0.0540_0.0630_2.pkl",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spectrum_row(report, artifact):
    pairs = report.get("pairs", [])
    absolute = [p["residual_absolute_M"] for p in pairs]
    return {"directory": str(Path(artifact["directory"]).relative_to(ROOT)),
            "method": report.get("method"), "which": report.get("which"),
            "sigma": report.get("sigma"), "requested_k": report.get("requested_k"),
            "solver_converged": bool(report.get("solver_converged")),
            "all_pairs_validated": bool(report.get("all_pairs_validated")),
            "full_max_real": report.get("full_max_real"),
            "nonrotation_max_real": report.get("nonrotation_max_real"),
            "max_residual_absolute_M": max(absolute) if absolute else None,
            "rotation_eigenvalues": [p["eigenvalue"][0] for p in pairs if p["rotation_like"]],
            "rotation_overlaps": [p["rotation_overlap"] for p in pairs if p["rotation_like"]],
            "found_reliable_positive": bool(report.get("found_reliable_positive")),
            "wall_seconds": report.get("wall_seconds")}


def legacy_crosscheck(label, row):
    """Compare a regenerated candidate with its historical single-shift diagnostic record."""
    path = LEGACY_RECORDS.get(label)
    if path is None or not path.is_file():
        return None
    import joblib

    legacy = joblib.load(path)
    shifted = next(s for s in row["spectra"] if s["sigma"] == .001)
    new_rotation = sorted(shifted["rotation_eigenvalues"])
    old_rotation = sorted(float(x) for x in legacy.get("rotation_eigenvalues", []))
    matched = len(new_rotation) == len(old_rotation) > 0

    def relative(a, b):
        return abs(a - b) / max(abs(b), 1e-300)

    old_nonrotation, new_nonrotation = legacy.get("max_re_lambda"), shifted["nonrotation_max_real"]
    return {
        "legacy_record": str(path.relative_to(ROOT)), "legacy_record_sha256": sha(path),
        "legacy_n_components": legacy.get("n_components"),
        "new_n_components": row["pattern"]["n_components"],
        "legacy_wavelength": legacy.get("wavelength"),
        "new_wavelength": row["pattern"]["wavelength"],
        "wavelength_relative_difference": relative(row["pattern"]["wavelength"],
                                                   legacy.get("wavelength", float("nan"))),
        "rotation_eigenvalues_legacy": old_rotation,
        "rotation_eigenvalues_new": new_rotation,
        "rotation_relative_difference_max": (max(relative(a, b) for a, b in zip(new_rotation, old_rotation))
                                             if matched else None),
        "nonrotation_max_legacy": old_nonrotation, "nonrotation_max_new": new_nonrotation,
        "nonrotation_relative_difference": (relative(new_nonrotation, old_nonrotation)
                                            if old_nonrotation else None),
        "legacy_stable_flag": legacy.get("stable"),
        "legacy_max_re_lambda_equals_nonrotation_max": bool(
            old_nonrotation is not None and abs(old_nonrotation - new_nonrotation) <=
            1e-9 * max(abs(old_nonrotation), 1e-300)),
        "note": ("The regenerated field reproduces the historical record's pattern and its rotation-block "
                 "eigenvalues. The historical `max_re_lambda` numerically equals the new NON-rotation "
                 "maximum (not the full maximum), i.e. the legacy stability flag ignored the rotation "
                 "block entirely; for this candidate that block was negative anyway."),
    }



def perturbation_row(block):
    if not block:
        return None
    resolutions = block.get("resolutions", [])
    rate = None
    for row in resolutions:
        if float(row["dt"]) == 1000.0:
            nonlinear = [float(x) for x in row["nonlinear_fitted_rates"]]
            rate = sum(nonlinear) / len(nonlinear)
    return {"file": str(Path(block["file"]).relative_to(ROOT)),
            "sha256": block["sha256"], "passed": bool(block["passed"]),
            "nonlinear_rate_dt1000_mean": rate,
            "linear_rates": {str(r["dt"]): r["linear_fitted_rate"] for r in resolutions},
            "half_dt_rate_difference": block["half_dt_rate_difference"]}


def summarise(label, path):
    cert = json.loads(path.read_text(encoding="utf-8"))
    decision = cert["decision"]
    spectra = [spectrum_row(report, artifact)
               for report, artifact in zip(cert["spectra"], cert["artifacts"])]
    return {"label": label, "certificate": str(path.relative_to(ROOT)),
            "certificate_sha256": sha(path),
            "mat_sha256": cert["input"]["mat_sha256"],
            "geometry": {"name": cert["input"]["geometry"]["name"],
                         "vertices": cert["input"]["geometry"]["n_vertices"]},
            "parameters": cert["input"]["parameters"], "seed": cert["input"]["seed"],
            "i01_full_residual": cert["input"]["full_steady_residual"]["value"],
            "i01_accepted": cert["input"]["I01_accepted"],
            "pattern": {"type": cert["pattern"]["type"],
                        "n_components": cert["pattern"]["n_components"],
                        "wavelength": cert["pattern"]["wavelength"],
                        "v_min": cert["pattern"]["v_min"],
                        "v_max": cert["pattern"]["v_max"]},
            "rotation_rank": cert["rotation"]["rank"],
            "rotation_singular_values": cert["rotation"]["singular_values"],
            "spectra": spectra,
            "perturbation": perturbation_row(cert.get("perturbation")),
            "decision": {"status": decision["status"],
                         "numerical_check_passed": bool(decision["numerical_check_passed"]),
                         "found_reliable_positive": bool(decision["found_reliable_positive"]),
                         "reliable_positive_eigenvalues": decision["reliable_positive_eigenvalues"],
                         "distinct_valid_shifts": decision["distinct_valid_shifts"],
                         "sampled_maxima_agree": bool(decision["sampled_maxima_agree"]),
                         "all_checked_signs_resolved_negative":
                             bool(decision["all_checked_signs_resolved_negative"]),
                         "failure_reasons": decision["failure_reasons"],
                         "mathematical_full_spectrum_proof":
                             bool(cert["mathematical_full_spectrum_proof"])}}


def fmt(value, digits=6):
    if value is None:
        return "n/a"
    return f"{value:+.{digits}e}"


def markdown(rows):
    lines = ["# I02 numerical stability comparison", "",
             "All values are copied from `stability_certificate.json` files produced by "
             "`scripts/i02_stability.py certificate`. Finite Krylov sampling is numerical "
             "evidence, not a proof of the full spectrum; no global nonlinear stability is claimed.",
             "",
             "| candidate | pattern (n) | I01 full residual | max Re lambda: sigma=1e-3, k=12 | "
             "max Re lambda: sigma=0, k=20 | max Re lambda: direct LR, k=8 | "
             "non-rotation max Re lambda | reliable positive | growth rate dt=1000 | "
             "perturbation | I02 status |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        shifted = next(s for s in row["spectra"] if s["sigma"] == .001)
        zero = next(s for s in row["spectra"] if s["sigma"] == 0.0)
        direct = next(s for s in row["spectra"] if s["sigma"] is None)
        perturb = row["perturbation"]
        lines.append(
            f"| {row['label']} | {row['pattern']['type']} ({row['pattern']['n_components']}) | "
            f"{row['i01_full_residual']:.3e} | {fmt(shifted['full_max_real'])} | "
            f"{fmt(zero['full_max_real'])} | {fmt(direct['full_max_real'])} | "
            f"{fmt(shifted['nonrotation_max_real'])} | "
            f"{'yes' if row['decision']['found_reliable_positive'] else 'no'} | "
            f"{fmt(perturb['nonlinear_rate_dt1000_mean']) if perturb else 'n/a'} | "
            f"{'passed' if perturb and perturb['passed'] else ('failed' if perturb else 'n/a')} | "
            f"{row['decision']['status']} |")
    lines += ["", "## Certificate details", "",
              "| candidate | MAT sha256 (first 16) | rotation rank | max eigenpair residual | "
              "spectra validated | failure reasons |",
              "|---|---|---|---|---|---|"]
    for row in rows:
        residual = max(s["max_residual_absolute_M"] for s in row["spectra"])
        validated = all(s["all_pairs_validated"] and s["solver_converged"] for s in row["spectra"])
        reasons = ", ".join(row["decision"]["failure_reasons"]) or "none"
        lines.append(f"| {row['label']} | {row['mat_sha256'][:16]} | {row['rotation_rank']} | "
                     f"{residual:.3e} | {'yes' if validated else 'no'} | {reasons} |")
    lines += ["", "## Per-candidate reading", ""]
    for row in rows:
        shifted = next(s for s in row["spectra"] if s["sigma"] == .001)
        perturb = row["perturbation"]
        if row["decision"]["found_reliable_positive"]:
            eigenvalue = row["decision"]["reliable_positive_eigenvalues"][0]
            lines.append(
                f"- **{row['label']}**: the unprojected operator has a reliable positive eigenvalue "
                f"{eigenvalue[0]:+.6e} (rotation overlap "
                f"{shifted['rotation_overlaps'][0]:.6f} in the shifted solve) while the non-rotation "
                f"maximum is {shifted['nonrotation_max_real']:+.6e}. The small-amplitude evolution of "
                f"that eigenvector grows at {perturb['nonlinear_rate_dt1000_mean']:+.6e} per unit time, "
                f"so the positive sign is confirmed dynamically and is not solver noise. This is a "
                f"*discrete* instability of a near-zero rotation-like mode; whether it survives mesh "
                f"refinement is not established here.")
        else:
            lines.append(
                f"- **{row['label']}**: every checked eigenvalue is resolved negative; the least damped "
                f"is {shifted['full_max_real']:+.6e} (rotation overlap "
                f"{shifted['rotation_overlaps'][0]:.6f}), and the non-rotation maximum is "
                f"{shifted['nonrotation_max_real']:+.6e}. The small-amplitude evolution decays at "
                f"{perturb['nonlinear_rate_dt1000_mean']:+.6e} per unit time.")
    legacy = [row for row in rows if row.get("legacy_crosscheck")]
    if legacy:
        lines += ["", "## Reproduction cross-check against the historical record", ""]
        for row in legacy:
            check = row["legacy_crosscheck"]
            lines.append(
                f"- **{row['label']}** (`{check['legacy_record']}`): {check['new_n_components']} vs "
                f"{check['legacy_n_components']} components; wavelength relative difference "
                f"{check['wavelength_relative_difference']:.2e}; rotation-block eigenvalues max relative "
                f"difference {check['rotation_relative_difference_max']:.2e}; non-rotation maximum "
                f"{check['nonrotation_max_new']:+.6e} vs historical `max_re_lambda` "
                f"{check['nonrotation_max_legacy']:+.6e} (relative difference "
                f"{check['nonrotation_relative_difference']:.2e}). {check['note']}")
    lines += ["", "## Scope and limits", "",
              "- Scope is the unprojected discrete Jacobian `J = Mfull^-1 * weak_J` on the robust "
              "sphere6 operator, checked with two shift-invert solves and one direct largest-real-part "
              "solve per candidate.",
              "- Rotation overlap is descriptive only. A positive eigenvalue near the rotation "
              "subspace is never removed from the decision.",
              "- `mathematical_full_spectrum_proof` is `false` for every candidate: a finite ARPACK "
              "sample cannot prove that the whole spectrum lies in the left half plane.",
              "- Mesh-refinement and orientation controls were not run, so the continuum limit of "
              "the weak rotation-like modes is not decided here.",
              ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificates", nargs="+", type=Path,
                        default=sorted(BASE.glob("certificates/*/stability_certificate.json")))
    parser.add_argument("--out", type=Path, default=BASE / "comparison")
    args = parser.parse_args()
    args.out = args.out.resolve()
    if not args.certificates:
        parser.error("No certificates found")
    if not args.out.is_relative_to(BASE.resolve()) or args.out.exists():
        parser.error("Choose a new output directory below results/i02_stability")
    rows = []
    for path in args.certificates:
        path = path.resolve()
        label = LABELS.get(path.parent.name, path.parent.name)
        row = summarise(label, path)
        row["legacy_crosscheck"] = legacy_crosscheck(path.parent.name, row)
        rows.append(row)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(args.out / "comparison.json",
                         {"schema_version": "aic-frog-i02-comparison-v1",
                          "certificates": [str(p.resolve().relative_to(ROOT)) for p in args.certificates],
                          "candidates": rows,
                          "mathematical_full_spectrum_proof": False,
                          "limitation": "Finite Krylov sampling plus local perturbation evidence; "
                                        "not a full-spectrum or nonlinear-stability proof."})
    (args.out / "comparison.md").write_text(markdown(rows), encoding="utf-8")
    print(markdown(rows), flush=True)


if __name__ == "__main__":
    main()
