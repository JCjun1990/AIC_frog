"""I04 unified evaluation criteria: keep every success concept separate and auditable.

The project historically mixed seven different notions into one ``success``/``stable``
flag: file validity, a complete steady state, a rotation-relative equilibrium, a finite
spectral sample, pattern morphology, the compute budget and the end-to-end task outcome.
This module defines them once, with explicit thresholds and scopes, and provides the
record/aggregation machinery used by ``scripts/i04_evaluate.py``.

Nothing here reads experiment files: it is pure classification and counting logic so it
can be unit tested without the large artifacts.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

EVALUATION_VERSION = "aic-frog-eval-v1"

#: Complete-steady-state threshold on the full (unprojected) joint residual/F.
STEADY_TOL = 1e-11
#: An eigenvalue with |Re lambda| below this cannot be signed from a legacy record that
#: carries no eigenpair residual.  1/time, same unit as the residuals.
SIGN_FLOOR = 1e-9

STATUS_PASS = "numerical_check_passed"
STATUS_POSITIVE = "unstable_positive_mode"
STATUS_INSUFFICIENT = "insufficient_evidence"
STATUS_NOT_CHECKED = "not_checked"
SPECTRAL_STATUSES = (STATUS_PASS, STATUS_POSITIVE, STATUS_INSUFFICIENT, STATUS_NOT_CHECKED)

FILE_VALID = "valid"
FILE_INVALID = "invalid"
FILE_NOT_SAVED = "not_saved"
FILE_NOT_CHECKED = "not_checked"
FILE_STATUSES = (FILE_VALID, FILE_INVALID, FILE_NOT_SAVED, FILE_NOT_CHECKED)

#: Task targets.  "stripes" is a band/maze family; six- and seven-band states both match,
#: the band count is reported separately and never folds into the shape criterion.
TARGET_PATTERN = {
    "spots": "spots",
    "stripes": "stripes_or_maze",
    "stripes_6": "stripes_or_maze",
    "stripes_7": "stripes_or_maze",
}

SPECTRAL_LIMITATION = (
    "A finite ARPACK sample with k << dim(J) is numerical evidence only; it is never a "
    "proof that the whole spectrum lies in the left half plane."
)

CRITERIA = {
    "file_validity": {
        "definition": ("Solution.u/v read back as float64 with the required (n,1) layout, all values "
                       "finite, u in [0,1], v >= 0, no float32-quantized nonuniform field, and any "
                       "attached stability certificate bound to the SHA256 of the same MAT bytes."),
        "threshold": "structure+precision+range+residual certificate accepted; certificate mat_sha256 == file sha256",
        "scope": "only runs whose field was saved (formal candidates); robustness/evaluation scans saved no field",
        "source": "turing_rd/result_io.py verify_mat / validate_certificate; results/i02_stability certificates",
    },
    "steady_full": {
        "definition": ("Complete stationary PDE residual sqrt((RMS_M(Ru)^2+RMS_M(Rv)^2)/2)/F computed from the "
                       "saved/recorded field with the actual robust L and M, with NO rotation projection."),
        "threshold": f"<= {STEADY_TOL:.0e}",
        "scope": "every run that recorded a full residual",
        "source": "turing_rd/diagnostics.full_steady_residual; I01 certificate residual; record rel_residual",
    },
    "relative_equilibrium": {
        "definition": ("Residual after removing the best rigid-rotation fit is below tolerance while the full "
                       "residual is not: the state is steady only up to a rigid rotation."),
        "threshold": f"rotation-removed <= {STEADY_TOL:.0e} AND full > {STEADY_TOL:.0e}",
        "scope": "spherical runs that recorded rel_residual_rot; not applicable to the frog mesh",
        "source": "turing_rd/steady.rotation_split; record rel_residual_rot",
    },
    "spectral": {
        "definition": ("Numerical linear stability of the recorded state from a finite eigenvalue sample. "
                       "Rotation/tilt-labelled eigenvalues are NEVER excluded from a pass decision."),
        "threshold": (f"pass iff every sampled |Re lambda| > {SIGN_FLOOR:.0e} and no sampled Re lambda > 0; "
                      "any Re lambda > 0 -> unstable_positive_mode; |Re lambda| <= floor -> insufficient"),
        "scope": "only runs where eigenvalues were sampled; otherwise not_checked",
        "source": "turing_rd/stability.py (I02 certificates) or legacy single-shift records",
    },
    "shape": {
        "definition": "Observed morphology equals the task/reference morphology for that parameter point.",
        "threshold": "type == 'spots' for spot tasks; type == 'stripes_or_maze' for stripe tasks; band count is reported separately",
        "scope": "every run with a recorded pattern type",
        "source": "turing_rd/diagnostics.pattern_stats; record type / predicted_type",
    },
    "budget": {
        "definition": "The run finished inside its declared physical-time cap instead of stopping at the cap.",
        "threshold": "relax_time < t_max (evaluation/robustness t_max = 2e5; training labels t_max = 1e5)",
        "scope": "every run with a recorded relax_time",
        "source": "record relax_time; scripts/ai_generation.py schedules",
    },
    "end_to_end": {
        "definition": ("Task success: steady_full AND spectral pass AND shape match AND within budget, with an "
                       "invalid saved field vetoing the result. A relative equilibrium alone is not a success."),
        "threshold": "all of the above",
        "scope": "every record",
        "source": "this module, task_success()",
    },
}


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rate(passed, planned):
    """Fraction with an explicit None instead of a ZeroDivisionError for empty groups."""
    if planned is None or planned <= 0:
        return None
    return passed / planned


def classify_spectrum(values, evidence_level="none", eigenpair_validated=False, note=None):
    """Classify a finite eigenvalue sample without ever letting a positive value pass.

    ``values`` are the real parts of every sampled eigenvalue, *including* the
    rotation/tilt-labelled ones.  Legacy records carry no eigenpair residual, so a
    value within ``SIGN_FLOOR`` of zero is reported as unresolved rather than passed.
    """
    numeric = []
    for value in values:
        if value is None:
            continue
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"Non-finite eigenvalue value {value!r}")
        numeric.append(value)
    positives = [v for v in numeric if v > 0]
    unresolved = [v for v in numeric if abs(v) <= SIGN_FLOOR]
    if not numeric:
        status = STATUS_NOT_CHECKED
    elif positives:
        status = STATUS_POSITIVE
    elif unresolved:
        status = STATUS_INSUFFICIENT
    else:
        status = STATUS_PASS
    return {
        "status": status,
        "max_real": max(numeric) if numeric else None,
        "checked_eigenvalues": len(numeric),
        "positive_eigenvalues": positives,
        "unresolved_eigenvalues": unresolved,
        "eigenpair_validated": bool(eigenpair_validated),
        "evidence_level": evidence_level,
        "sign_floor": SIGN_FLOOR,
        "limitation": SPECTRAL_LIMITATION,
        "note": note,
    }


def shape_matches(observed, target):
    if observed is None or target is None:
        return None
    return bool(observed == target)


def within_budget(relax_time, t_max):
    if relax_time is None:
        return None
    return bool(float(relax_time) < float(t_max))


def steady_flags(full_residual, rotation_removed_residual=None):
    """Separate 'complete steady state' from 'relative equilibrium' explicitly."""
    steady = None if full_residual is None else bool(float(full_residual) <= STEADY_TOL)
    if rotation_removed_residual is None:
        return steady, None
    relative = bool(float(rotation_removed_residual) <= STEADY_TOL and steady is not True)
    return steady, relative


def failure_reason(record: "EvaluationRecord"):
    """First failing criterion in report order, or None when everything passed."""
    reasons = []
    if record.file_validity == FILE_INVALID:
        reasons.append("invalid_saved_field")
    if record.steady_full is False:
        reasons.append("not_a_complete_steady_state")
    elif record.steady_full is None:
        reasons.append("steady_state_not_evaluated")
    if record.spectral_status == STATUS_POSITIVE:
        reasons.append("positive_sampled_eigenvalue")
    elif record.spectral_status == STATUS_INSUFFICIENT:
        reasons.append("spectral_sign_unresolved")
    elif record.spectral_status == STATUS_NOT_CHECKED:
        reasons.append("spectral_not_checked")
    if record.shape_match is False:
        reasons.append("shape_mismatch")
    if record.within_budget is False:
        reasons.append("time_budget_exhausted")
    return ";".join(reasons) if reasons else None


@dataclass
class EvaluationRecord:
    """One auditable experiment result with every criterion kept separate."""

    experiment_id: str
    record_kind: str                      # formal_candidate | robustness_seed | evaluation_run | training_label
    surface: str
    source_commit: str = ""
    source_fingerprint: str = ""
    input_hash: str | None = None
    result_hash: str | None = None
    target_pattern: str | None = None
    pattern: str | None = None          # observed morphology of this record (alias kept explicit)
    parameters: dict = field(default_factory=dict)
    seed: int | None = None
    model_id: str = "none"
    pipeline: str = "none"
    solver_config: dict = field(default_factory=dict)
    full_residual: float | None = None
    rotation_removed_residual: float | None = None
    steady_full: bool | None = None
    relative_equilibrium: bool | None = None
    spectral_status: str = STATUS_NOT_CHECKED
    spectral_max_real: float | None = None
    spectral_evidence_level: str = "none"
    spectral_detail: dict = field(default_factory=lambda: classify_spectrum([]))
    predicted_pattern: str | None = None
    observed_pattern: str | None = None
    shape_match: bool | None = None
    within_budget: bool | None = None
    failure_reason: str | None = None
    runtime: float | None = None
    integration_steps: int | None = None
    evaluation_version: str = EVALUATION_VERSION
    file_validity: str = FILE_NOT_SAVED
    end_to_end_success: bool = False
    i01_candidate_accepted: bool | None = None
    legacy_stability_flag: bool | None = None
    legacy_success: bool | None = None
    legacy_stability_scope: str | None = None
    evidence: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def finalise(self):
        """Fill derived fields after the raw ones are set.  Returns self."""
        if self.steady_full is None and self.relative_equilibrium is None:
            self.steady_full, self.relative_equilibrium = steady_flags(
                self.full_residual, self.rotation_removed_residual)
        if self.shape_match is None and self.observed_pattern is not None:
            self.shape_match = shape_matches(self.observed_pattern, self.target_pattern)
        if self.pattern is None:
            self.pattern = self.observed_pattern
        self.end_to_end_success = bool(task_success(self))
        self.failure_reason = failure_reason(self)
        return self

    def to_dict(self):
        return asdict(self)


def task_success(record: EvaluationRecord, require_saved_field=False):
    """End-to-end success = complete steady state AND spectral pass AND shape AND budget.

    A relative equilibrium alone, a merely converged run, a correct classifier
    prediction, or an unchecked spectrum never counts as success.
    """
    if record.steady_full is not True:
        return False
    if record.spectral_status != STATUS_PASS:
        return False
    if record.shape_match is not True:
        return False
    if record.within_budget is not True:
        return False
    if record.file_validity == FILE_INVALID:
        return False
    if require_saved_field and record.file_validity != FILE_VALID:
        return False
    return True


def i01_candidate_ready(record: EvaluationRecord):
    """I01-level readiness for submission: valid float64 field and a complete steady state."""
    return bool(record.file_validity == FILE_VALID and record.steady_full is True
                and record.i01_candidate_accepted is True)


def aggregate(records, planned=None):
    """Count every criterion separately; denominators are never shrunk silently."""
    records = list(records)
    planned = len(records) if planned is None else int(planned)
    spectral = {status: 0 for status in SPECTRAL_STATUSES}
    for record in records:
        spectral[record.spectral_status] = spectral.get(record.spectral_status, 0) + 1
    file_counts = {status: 0 for status in FILE_STATUSES}
    for record in records:
        file_counts[record.file_validity] = file_counts.get(record.file_validity, 0) + 1
    reasons = {}
    for record in records:
        for reason in (record.failure_reason or "").split(";"):
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
    strict = sum(1 for r in records if task_success(r))
    return {
        "planned": planned,
        "executed": len(records),
        "missing": max(planned - len(records), 0),
        "error": sum(1 for r in records if r.evidence.get("parse_error")),
        "valid": len(records) - sum(1 for r in records if r.evidence.get("parse_error")),
        "file_valid": file_counts[FILE_VALID],
        "file_invalid": file_counts[FILE_INVALID],
        "file_not_saved": file_counts[FILE_NOT_SAVED],
        "file_not_checked": file_counts[FILE_NOT_CHECKED],
        "steady_full": sum(1 for r in records if r.steady_full is True),
        "relative_equilibrium_only": sum(1 for r in records if r.relative_equilibrium is True),
        "spectral_checked": sum(1 for r in records if r.spectral_status != STATUS_NOT_CHECKED),
        "spectral_passed": spectral[STATUS_PASS],
        "spectral_positive": spectral[STATUS_POSITIVE],
        "spectral_insufficient": spectral[STATUS_INSUFFICIENT],
        "spectral_not_checked": spectral[STATUS_NOT_CHECKED],
        "shape_match": sum(1 for r in records if r.shape_match is True),
        "shape_checked": sum(1 for r in records if r.shape_match is not None),
        "within_budget": sum(1 for r in records if r.within_budget is True),
        "legacy_success": sum(1 for r in records if r.legacy_success is True),
        "strict_success": strict,
        "success_rate_strict": rate(strict, planned),
        "failure_reasons": dict(sorted(reasons.items())),
    }


def comparison_flags(a: dict, b: dict):
    """Decide whether two protocols may be compared at all (I04 fairness rules).

    ``a``/``b`` describe a protocol: host, load, concurrency, t_max, schedule.
    Wall-clock is comparable only under the same host *and* load *and* concurrency;
    integration steps only under the same time budget *and* schedule.
    """
    wall = (a.get("host") == b.get("host") and a.get("load") == b.get("load")
            and a.get("concurrency") == b.get("concurrency"))
    steps = (a.get("t_max") == b.get("t_max") and a.get("schedule") == b.get("schedule"))
    reasons = []
    if not wall:
        reasons.append("wall_clock_not_comparable: different host, load or concurrency")
    if not steps:
        reasons.append("integration_steps_not_comparable: different time cap or schedule")
    return {"wall_comparable": bool(wall), "steps_comparable": bool(steps),
            "reason": "; ".join(reasons) if reasons else "same protocol"}


def verify_certificate_binding(mat_path, certificate, expected_sha=None):
    """Fail closed unless a stability certificate points at the exact MAT bytes.

    Returns the MAT SHA256 on success.  Raises ValueError on any mismatch so a
    certificate for a different field can never be attached to a candidate.
    """
    actual = sha256_file(mat_path)
    if expected_sha is not None and actual != expected_sha:
        raise ValueError("MAT file hash does not match the expected I01 certificate hash")
    bound = (certificate or {}).get("input", {}).get("mat_sha256")
    if not bound:
        raise ValueError("stability certificate has no bound mat_sha256")
    if bound != actual:
        raise ValueError("stability certificate MAT hash mismatch")
    return actual


def denominator_consistency(summary: dict, expected_planned: int):
    """Every breakdown must account for all planned runs (executed + missing)."""
    problems = []
    if summary["planned"] != expected_planned:
        problems.append(f"planned {summary['planned']} != protocol {expected_planned}")
    if summary["executed"] + summary["missing"] != summary["planned"]:
        problems.append("executed + missing != planned")
    if summary["valid"] + summary["error"] != summary["executed"]:
        problems.append("valid + error != executed")
    if (summary["file_valid"] + summary["file_invalid"] + summary["file_not_saved"]
            + summary["file_not_checked"] != summary["executed"]):
        problems.append("file status counts != executed")
    if (summary["spectral_passed"] + summary["spectral_positive"] + summary["spectral_insufficient"]
            + summary["spectral_not_checked"] != summary["executed"]):
        problems.append("spectral status counts != executed")
    if summary["strict_success"] > summary["steady_full"]:
        problems.append("strict successes exceed complete steady states")
    return {"consistent": not problems, "problems": problems}
