"""I04 criteria tests: every success concept stays separate and fails closed."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from turing_rd.evaluation import (CRITERIA, EVALUATION_VERSION, FILE_INVALID, FILE_NOT_SAVED,
                                  FILE_VALID, SIGN_FLOOR, STATUS_INSUFFICIENT, STATUS_NOT_CHECKED,
                                  STATUS_PASS, STATUS_POSITIVE, STEADY_TOL, EvaluationRecord,
                                  aggregate, classify_spectrum, comparison_flags,
                                  denominator_consistency, i01_candidate_ready, rate,
                                  shape_matches, steady_flags, task_success,
                                  verify_certificate_binding, within_budget)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("i04_evaluate_under_test", ROOT / "scripts" / "i04_evaluate.py")
i04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(i04)


def make_record(**overrides):
    base = dict(
        experiment_id="test:record", record_kind="robustness_seed", surface="frog",
        target_pattern="spots", parameters={"F": 0.028, "k": 0.062}, seed=0,
        model_id="mlp_coarse_9x5", pipeline="ai", solver_config={"t_max": 2e5},
        full_residual=1e-14, observed_pattern="spots", within_budget=True,
        spectral_status=STATUS_PASS)
    base.update(overrides)
    return EvaluationRecord(**base).finalise()


class SteadyVersusRelativeTests(unittest.TestCase):
    def test_complete_steady_state_and_relative_equilibrium_are_different(self):
        steady, relative = steady_flags(1e-14, 1e-3)
        self.assertTrue(steady)
        self.assertFalse(relative)
        steady, relative = steady_flags(1e-3, 1e-14)
        self.assertFalse(steady)
        self.assertTrue(relative)

    def test_relative_equilibrium_is_not_task_success(self):
        record = make_record(full_residual=1e-3, rotation_removed_residual=1e-14)
        self.assertTrue(record.relative_equilibrium)
        self.assertFalse(record.steady_full)
        self.assertFalse(record.end_to_end_success)
        self.assertIn("not_a_complete_steady_state", record.failure_reason)

    def test_frog_has_no_relative_equilibrium_concept(self):
        steady, relative = steady_flags(1e-14, None)
        self.assertTrue(steady)
        self.assertIsNone(relative)


class SpectralDecisionTests(unittest.TestCase):
    def test_positive_eigenvalue_cannot_be_ignored(self):
        spectral = classify_spectrum([-4.6e-5, 1.665e-6], evidence_level="legacy")
        self.assertEqual(spectral["status"], STATUS_POSITIVE)
        record = make_record(spectral_status=STATUS_POSITIVE, spectral_max_real=1.665e-6)
        self.assertFalse(task_success(record))
        self.assertIn("positive_sampled_eigenvalue", record.failure_reason)

    def test_unknown_spectral_evidence_never_passes(self):
        for status in (STATUS_NOT_CHECKED, STATUS_INSUFFICIENT):
            record = make_record(spectral_status=status)
            self.assertFalse(task_success(record), status)

    def test_unresolved_near_zero_eigenvalue_is_insufficient(self):
        self.assertEqual(classify_spectrum([-1e-12])["status"], STATUS_INSUFFICIENT)
        self.assertEqual(classify_spectrum([0.0])["status"], STATUS_INSUFFICIENT)
        self.assertEqual(classify_spectrum([-1e-6])["status"], STATUS_PASS)
        self.assertEqual(classify_spectrum([])["status"], STATUS_NOT_CHECKED)
        self.assertGreater(SIGN_FLOOR, 0)

    def test_all_sampled_values_negative_and_resolved_pass(self):
        record = make_record(spectral_status=STATUS_PASS)
        self.assertTrue(record.end_to_end_success)

    def test_non_finite_eigenvalue_is_rejected(self):
        with self.assertRaises(ValueError):
            classify_spectrum([float("nan")])


class ShapeAndBudgetTests(unittest.TestCase):
    def test_math_converged_but_wrong_shape_is_not_success(self):
        record = make_record(target_pattern="stripes_or_maze", observed_pattern="spots")
        self.assertIs(record.shape_match, False)
        self.assertFalse(record.end_to_end_success)
        self.assertIn("shape_mismatch", record.failure_reason)

    def test_seven_band_state_matches_the_stripe_target(self):
        record = make_record(target_pattern="stripes_or_maze", observed_pattern="stripes_or_maze")
        self.assertTrue(record.end_to_end_success)
        self.assertTrue(shape_matches("stripes_or_maze", "stripes_or_maze"))
        self.assertIsNone(shape_matches(None, "spots"))

    def test_not_converged_with_correct_classification_is_not_success(self):
        record = make_record(full_residual=3.9e-3, spectral_status=STATUS_NOT_CHECKED,
                             predicted_pattern="spots")
        self.assertFalse(record.end_to_end_success)
        self.assertIn("not_a_complete_steady_state", record.failure_reason)

    def test_budget_boundary(self):
        self.assertTrue(within_budget(199999.0, 200000.0))
        self.assertFalse(within_budget(200000.0, 200000.0))
        self.assertIsNone(within_budget(None, 200000.0))
        record = make_record(within_budget=False)
        self.assertFalse(record.end_to_end_success)
        self.assertIn("time_budget_exhausted", record.failure_reason)

    def test_invalid_saved_field_vetoes_success(self):
        record = make_record(file_validity=FILE_INVALID)
        self.assertFalse(record.end_to_end_success)
        self.assertIn("invalid_saved_field", record.failure_reason)


class AggregationTests(unittest.TestCase):
    def test_all_samples_failing_keeps_the_full_denominator(self):
        records = [make_record(full_residual=1e-3, spectral_status=STATUS_NOT_CHECKED),
                   make_record(full_residual=1e-3, observed_pattern="stripes_or_maze"),
                   make_record(spectral_status=STATUS_POSITIVE)]
        summary = aggregate(records, planned=3)
        self.assertEqual(summary["planned"], 3)
        self.assertEqual(summary["executed"], 3)
        self.assertEqual(summary["missing"], 0)
        self.assertEqual(summary["strict_success"], 0)
        self.assertEqual(summary["success_rate_strict"], 0.0)
        self.assertTrue(denominator_consistency(summary, 3)["consistent"])

    def test_missing_and_unparsable_records_are_counted_not_dropped(self):
        error_record = EvaluationRecord(experiment_id="bad", record_kind="robustness_seed",
                                        surface="frog").finalise()
        error_record.evidence = {"parse_error": "KeyError: rel_residual"}
        summary = aggregate([error_record], planned=3)
        self.assertEqual(summary["executed"], 1)
        self.assertEqual(summary["missing"], 2)
        self.assertEqual(summary["error"], 1)
        self.assertEqual(summary["valid"], 0)
        self.assertEqual(summary["strict_success"], 0)
        self.assertTrue(denominator_consistency(summary, 3)["consistent"])

    def test_empty_group_has_no_rate_instead_of_zero_division(self):
        self.assertIsNone(rate(0, 0))
        summary = aggregate([], planned=0)
        self.assertIsNone(summary["success_rate_strict"])

    def test_missing_record_placeholder_never_passes(self):
        record = make_record(full_residual=None, spectral_status=STATUS_NOT_CHECKED,
                             observed_pattern=None, within_budget=None)
        self.assertFalse(record.end_to_end_success)

    def test_denominator_inconsistency_is_detected(self):
        summary = aggregate([make_record()], planned=3)
        self.assertTrue(denominator_consistency(summary, 3)["consistent"])
        self.assertFalse(denominator_consistency(summary, 4)["consistent"])
        tampered = dict(summary, strict_success=2)
        self.assertFalse(denominator_consistency(tampered, 3)["consistent"])

    def test_spectral_and_file_status_counts_partition_the_records(self):
        records = [make_record(spectral_status=STATUS_PASS),
                   make_record(spectral_status=STATUS_POSITIVE),
                   make_record(spectral_status=STATUS_INSUFFICIENT),
                   make_record(spectral_status=STATUS_NOT_CHECKED, file_validity=FILE_VALID)]
        summary = aggregate(records, planned=4)
        self.assertEqual(summary["spectral_passed"] + summary["spectral_positive"]
                         + summary["spectral_insufficient"] + summary["spectral_not_checked"], 4)
        self.assertEqual(summary["file_valid"] + summary["file_invalid"] + summary["file_not_saved"]
                         + summary["file_not_checked"], 4)


class ComparisonAndBindingTests(unittest.TestCase):
    def test_different_budgets_are_not_step_comparable(self):
        a = dict(host="A", load="alone", concurrency=1, t_max=100000.0, tol=1e-11)
        b = dict(host="A", load="alone", concurrency=1, t_max=200000.0, tol=1e-11)
        flags = comparison_flags(a, b)
        self.assertTrue(flags["wall_comparable"])
        self.assertFalse(flags["steps_comparable"])

    def test_different_load_is_not_wall_comparable(self):
        a = dict(host="A", load="alone", concurrency=20, t_max=200000.0, tol=1e-11)
        b = dict(host="A", load="interleaved", concurrency=4, t_max=200000.0, tol=1e-11)
        flags = comparison_flags(a, b)
        self.assertFalse(flags["wall_comparable"])
        self.assertTrue(flags["steps_comparable"])

    def test_stability_certificate_must_bind_the_same_mat_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            mat = Path(tmp) / "solution.mat"
            mat.write_bytes(b"mat-bytes")
            digest = i04.sha256_file(mat)
            good = {"input": {"mat_sha256": digest}}
            self.assertEqual(verify_certificate_binding(mat, good), digest)
            bad = {"input": {"mat_sha256": "0" * 64}}
            with self.assertRaisesRegex(ValueError, "mismatch"):
                verify_certificate_binding(mat, bad)
            with self.assertRaisesRegex(ValueError, "no bound"):
                verify_certificate_binding(mat, {"input": {}})
            with self.assertRaisesRegex(ValueError, "expected I01"):
                verify_certificate_binding(mat, good, expected_sha="f" * 64)

    def test_missing_and_corrupt_sources_are_recorded(self):
        i04.PROBLEMS.clear()
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.json"
            self.assertIsNone(i04.read_json(missing))
            corrupt = Path(tmp) / "corrupt.json"
            corrupt.write_text("{not json", encoding="utf-8")
            self.assertIsNone(i04.read_json(corrupt))
        kinds = [p["kind"] for p in i04.PROBLEMS]
        self.assertIn("missing_source", kinds)
        self.assertIn("parse_error", kinds)
        i04.PROBLEMS.clear()


class FormalCandidateTests(unittest.TestCase):
    def test_i01_ready_needs_valid_field_and_certificate(self):
        record = make_record(file_validity=FILE_VALID, i01_candidate_accepted=True)
        self.assertTrue(i01_candidate_ready(record))
        self.assertFalse(i01_candidate_ready(make_record(file_validity=FILE_NOT_SAVED,
                                                         i01_candidate_accepted=True)))
        self.assertFalse(i01_candidate_ready(make_record(file_validity=FILE_VALID,
                                                         i01_candidate_accepted=False)))

    def test_candidate_without_bound_spectrum_is_not_called_stable(self):
        record = make_record(record_kind="formal_candidate", file_validity=FILE_VALID,
                             i01_candidate_accepted=True, spectral_status=STATUS_NOT_CHECKED)
        self.assertTrue(i01_candidate_ready(record))
        self.assertFalse(record.end_to_end_success)

    def test_record_carries_the_required_fields(self):
        record = make_record().to_dict()
        for field in ("experiment_id", "source_commit", "source_fingerprint", "input_hash",
                      "result_hash", "surface", "target_pattern", "pattern", "parameters", "seed",
                      "model_id", "solver_config", "full_residual", "rotation_removed_residual",
                      "steady_full", "relative_equilibrium", "spectral_status", "spectral_max_real",
                      "spectral_evidence_level", "predicted_pattern", "observed_pattern",
                      "shape_match", "within_budget", "failure_reason", "runtime",
                      "integration_steps", "evaluation_version"):
            self.assertIn(field, record)
        self.assertEqual(record["pattern"], record["observed_pattern"])
        self.assertEqual(record["evaluation_version"], EVALUATION_VERSION)

    def test_every_criterion_has_a_definition_threshold_scope_and_source(self):
        for name in ("file_validity", "steady_full", "relative_equilibrium", "spectral",
                     "shape", "budget", "end_to_end"):
            spec = CRITERIA[name]
            for key in ("definition", "threshold", "scope", "source"):
                self.assertTrue(spec[key], f"{name}.{key}")
        self.assertIn(f"{STEADY_TOL:.0e}", CRITERIA["steady_full"]["threshold"])

    def test_record_serialises_to_strict_json(self):
        payload = json.dumps(make_record().to_dict(), allow_nan=False)
        self.assertIn("evaluation_version", payload)


if __name__ == "__main__":
    unittest.main()
