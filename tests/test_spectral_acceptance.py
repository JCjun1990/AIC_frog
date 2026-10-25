"""Fail-closed spectral policy regressions; no large eigenproblem or simulation."""
import copy
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import submission_workflow as workflow
from scripts import sphere_bandcount as bands
from scripts import sphere_spots_layout as spots
from scripts import inversion_best_match as inverse
from scripts.i03_frog_stability import decide_frog
from scripts import i03_frog_stability as frog_cli
from scripts import i02_stability as sphere_cli
from turing_rd.spectral_policy import (
    CANDIDATE_CONFIGS, FROG_CONFIGS, INVERSION_SCREEN_CONFIGS, SPHERE_CONFIGS,
    FROG_SIGN_FLOOR, candidate_decision, evaluate_spectra, failed_spectrum,
)

ROOT = Path(__file__).resolve().parents[1]


def report(cfg):
    pairs = [{"eigenvalue": [-1e-4 - i * 1e-5, 0.],
              "residual_absolute_M": 1e-14, "residual_relative_M": 1e-10,
              "residual_passed": True, "positive_reliable": False,
              "negative_resolved": True} for i in range(cfg.k)]
    return {"sigma": cfg.sigma, "which": cfg.which, "requested_k": cfg.k,
            "returned_k": cfg.k, "solver_converged": True, "solver_error": None,
            "all_pairs_validated": True, "pairs": pairs, "full_max_real": -1e-4,
            "found_reliable_positive": False, "pair_absolute_tolerance": 1e-9,
            "pair_relative_tolerance": 1e-7}


class StrictPolicyTests(unittest.TestCase):
    def decide(self, runs, residual=1e-14):
        return evaluate_spectra(runs, residual, configs=CANDIDATE_CONFIGS, scope="test_two_shift_screen")

    def runs(self):
        return [report(c) for c in CANDIDATE_CONFIGS]

    def test_both_configurations_pass(self):
        result = self.decide(self.runs())
        self.assertTrue(result["numerical_check_passed"])
        self.assertEqual(result["outcome"], "passed")
        self.assertFalse(result["mathematical_full_spectrum_proof"])

    def test_each_failure_combination_rejects(self):
        for mask in ((False, True), (True, False), (False, False)):
            runs = self.runs()
            for ok, run in zip(mask, runs):
                run["solver_converged"] = ok
            with self.subTest(mask=mask):
                result = self.decide(runs)
                self.assertFalse(result["numerical_check_passed"])
                self.assertEqual(result["status"], "insufficient_evidence")
                self.assertTrue(result["failure_reasons"])

    def test_arpack_partial_result_rejects_even_if_flags_say_valid(self):
        runs = self.runs()
        runs[0]["solver_converged"] = False
        runs[0]["solver_error"] = "ARPACK partial convergence"
        self.assertFalse(self.decide(runs)["numerical_check_passed"])

    def test_bad_absolute_and_relative_pair_residuals_reject(self):
        for key, bad in (("residual_absolute_M", 1e-5), ("residual_relative_M", 1e-4)):
            runs = self.runs(); runs[0]["pairs"][0][key] = bad
            with self.subTest(key=key):
                self.assertFalse(self.decide(runs)["numerical_check_passed"])

    def test_failed_pair_flag_cannot_be_hidden_by_run_summary(self):
        runs = self.runs()
        runs[0]["pairs"][0]["residual_passed"] = False
        self.assertFalse(self.decide(runs)["numerical_check_passed"])

    def test_positive_cannot_be_hidden_by_flags_rotation_or_another_failed_solve(self):
        runs = self.runs()
        runs[0]["pairs"][0].update(eigenvalue=[1.665e-6, 0.], rotation_like=True,
                                   rotation_overlap=1., positive_reliable=False)
        runs[0]["full_max_real"] = 1.665e-6
        runs[1]["solver_converged"] = False
        result = self.decide(runs)
        self.assertEqual(result["status"], "unstable_positive_mode")
        self.assertEqual(result["outcome"], "failed")

    def test_near_zero_and_error_ambiguous_negative_are_insufficient(self):
        for real, error in ((-5e-13, 1e-15), (0., 0.), (-5e-9, 8e-10)):
            runs = self.runs()
            for run in runs:
                run["pairs"][0].update(eigenvalue=[real, 0.], residual_absolute_M=error)
                run["full_max_real"] = real
            with self.subTest(real=real):
                result = self.decide(runs)
                self.assertEqual(result["status"], "insufficient_evidence")
                self.assertTrue(result["unresolved_near_zero_eigenvalues"])

    def test_frog_retains_existing_one_e_minus_nine_near_zero_floor(self):
        runs = [report(c) for c in FROG_CONFIGS]
        for run in runs:
            run["pairs"][0]["eigenvalue"][0] = -1e-10
            run["full_max_real"] = -1e-10
        result = decide_frog(runs, 1e-14)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(FROG_SIGN_FLOOR, 1e-9)

    def test_frog_decision_preserves_fields_used_by_its_report_writer(self):
        result = decide_frog([report(c) for c in FROG_CONFIGS], 1e-14)
        self.assertEqual(len(result["sampled_maxima"]), 3)
        self.assertEqual(result["configurations"], 3)

    def test_inflated_tolerances_do_not_change_acceptance(self):
        runs = self.runs()
        for run in runs:
            run["pair_absolute_tolerance"] = 1
            run["pairs"][0]["residual_absolute_M"] = 1e-5
        self.assertFalse(self.decide(runs)["numerical_check_passed"])

    def test_too_few_pairs_even_with_self_consistent_declared_count_rejects(self):
        runs = self.runs()
        for run in runs:
            run.update(requested_k=1, returned_k=1, pairs=run["pairs"][:1])
        self.assertFalse(self.decide(runs)["numerical_check_passed"])

    def test_missing_duplicate_or_unexpected_configurations_reject(self):
        runs = self.runs()
        for bad in (runs[:1], [runs[0], runs[0]], runs + [runs[0]]):
            with self.subTest(count=len(bad)):
                self.assertFalse(self.decide(bad)["numerical_check_passed"])

    def test_negative_nan_infinite_or_large_pde_residual_cannot_pass(self):
        for value in (-1., float("nan"), float("inf"), 1e-6):
            with self.subTest(value=value):
                self.assertEqual(self.decide(self.runs(), value)["outcome"], "failed")
        with self.assertRaises(ValueError):
            evaluate_spectra(self.runs(), 1e-6, configs=CANDIDATE_CONFIGS, scope="bad", tol=1.)

    def test_missing_pair_metrics_and_nonfinite_values_fail_closed(self):
        for bad in ({}, {"eigenvalue": [float("nan"), 0.]}, {"eigenvalue": "bad"}):
            runs = self.runs(); runs[0]["pairs"][0] = bad
            self.assertFalse(self.decide(runs)["numerical_check_passed"])

    def test_summary_only_history_is_not_promoted(self):
        old = {"stable": True, "full_residual": 1e-14, "spectra": [
            {"sigma": 0., "k": 12, "converged": True, "validated": True, "max_real": -1e-4},
            {"sigma": 1e-3, "k": 12, "converged": True, "validated": True, "max_real": -1e-4}]}
        result = candidate_decision(old)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertTrue(old["stable"])  # the historical claim is left intact
        self.assertEqual(result["checked_eigenpairs"], 0)


class CandidateEntryPointTests(unittest.TestCase):
    def invoke(self, module, runs):
        g = SimpleNamespace(n=2, area=1., V=np.array([[0., 0., 1.], [0., 0., -1.]]))
        uv = (np.ones(2), np.zeros(2)); info = {"relax_time": 0., "relax_steps": 0}
        with patch.object(module.A, "geom", return_value=g), \
                patch.object(module.A, "rotations", return_value=[]), \
                patch.object(module.A, "predict", return_value={"stats": {"kappa0": 12}, "density": 2}), \
                patch("joblib.load", return_value=object()), \
                patch.object(module.ai_init, "bands", return_value=uv), \
                patch.object(module.ai_init, "place_spots", return_value=uv), \
                patch.object(module, "relax_polish", return_value=(*uv, info)), \
                patch.object(module, "full_steady_residual", return_value={"full_joint_rms_over_F": 1e-14}), \
                patch.object(module, "pattern_stats", return_value={"type": "spots", "n_components": 2}), \
                patch.object(module, "solve_spectrum", side_effect=runs), \
                patch.object(module.np, "savez_compressed"), \
                patch.object(module, "riesz_points", return_value=g.V, create=True):
            return module.run("sphere6", 6, 1) if module is bands else module.run("sphere6", 1)

    def test_both_actual_candidate_scripts_require_every_configuration(self):
        for module in (bands, spots):
            for mask in ((True, True), (False, True), (True, False), (False, False)):
                runs = [report(c) for c in CANDIDATE_CONFIGS]
                for ok, run in zip(mask, runs):
                    run["solver_converged"] = ok
                with self.subTest(module=module.__name__, mask=mask):
                    row = self.invoke(module, [(r, None) for r in runs])
                    self.assertEqual(row["stable"], all(mask))
                    self.assertFalse(row["full_discrete_stability_certified"])
                    self.assertEqual(len(row["spectra"]), 2)
                    self.assertIn("pairs", row["spectra"][1])

    def test_candidate_exception_is_recorded_and_never_interpreted_as_negative(self):
        for module in (bands, spots):
            row = self.invoke(module, [RuntimeError("deliberate solve failure"),
                                       (report(CANDIDATE_CONFIGS[1]), None)])
            self.assertFalse(row["stable"])
            self.assertIn("deliberate solve failure", row["spectra"][0]["solver_error"])
            self.assertEqual(row["stability_status"], "insufficient_evidence")

    def test_inverse_screen_checks_pairs_and_not_only_scalar_maximum(self):
        good = report(INVERSION_SCREEN_CONFIGS[0])
        for change in ("none", "convergence", "validation", "residual", "count", "legacy_scalar"):
            run = copy.deepcopy(good)
            if change == "convergence": run["solver_converged"] = False
            if change == "validation": run["all_pairs_validated"] = False
            if change == "residual": run["pairs"][0]["residual_relative_M"] = 1.
            if change == "count": run["pairs"].pop()
            sample = {"max_real": -1e-4, "spectral_report": run}
            if change == "legacy_scalar": sample.pop("spectral_report")
            uv = (np.ones(2), np.zeros(2))
            with self.subTest(change=change), patch.dict(inverse.W, {"g": object()}, clear=True), \
                    patch("joblib.load", return_value=object()), \
                    patch("ai_generation.predict", return_value={}), \
                    patch.object(inverse, "initial_state", return_value=uv), \
                    patch.object(inverse, "relax_polish", return_value=(*uv, {"relax_time": 0.})), \
                    patch.object(inverse, "full_steady_residual", return_value={"full_joint_rms_over_F": 1e-14}), \
                    patch.object(inverse, "linear_stability", return_value=sample), \
                    patch.object(inverse, "measure", return_value={}), \
                    patch.object(inverse, "spot_centres", return_value=[]):
                row = inverse.job((0, .028, .062, "cvt", 2, 1))
                self.assertEqual(row["screen_decision"]["numerical_check_passed"], change == "none")


class SavedCertificatePolicyTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / "submission/task2_frog/spots"
        self.original = workflow.load(self.directory / "stability_certificate.json")
        self.hash = workflow.sha(self.directory / "solution.mat")

    def rejected(self, certificate):
        with self.assertRaises(ValueError):
            workflow.validate_spectral_record(certificate, self.hash)

    def test_all_four_original_certificates_pass_fixed_protocol(self):
        for task in workflow.TASKS:
            d = ROOT / "submission" / task
            result = workflow.validate_spectral_record(workflow.load(d / "stability_certificate.json"),
                workflow.sha(d / "solution.mat"), workflow.load(d / "summary.json")["mat_certificate"],
                surface="frog" if task.startswith("task2") else "sphere")
            self.assertEqual(result["checked_pairs"], 40)
            self.assertFalse(result["mathematical_full_spectrum_proof"])

    def test_malicious_saved_tolerance_cannot_accept_one_e_minus_five_absolute_error(self):
        c = copy.deepcopy(self.original)
        for run in c["configurations"]:
            run["pair_absolute_tolerance"] = 1.
            for pair in run["pairs"]:
                pair["residual_absolute_M"] = 1e-5
        self.rejected(c)


    def test_malicious_relative_or_steady_tolerance_cannot_pass(self):
        for kind in ("relative", "steady"):
            c = copy.deepcopy(self.original)
            if kind == "relative":
                c["configurations"][0]["pair_relative_tolerance"] = 1.
                c["configurations"][0]["pairs"][0]["residual_relative_M"] = 1e-4
            else:
                c["decision"]["full_residual_tolerance"] = 1.
                c["decision"]["full_residual"] = c["input"]["full_steady_residual"]["value"] = 1e-6
            with self.subTest(kind=kind): self.rejected(c)

    def test_saved_agreement_tolerance_cannot_be_inflated(self):
        c = copy.deepcopy(self.original)
        c["configurations"][0]["pairs"][0]["eigenvalue"][0] = -1e-6
        c["configurations"][0]["full_max_real"] = -1e-6
        c["decision"]["sampled_maxima_agreement_tolerance"] = 1.
        self.rejected(c)

    def test_saved_count_cannot_self_authorize_a_smaller_protocol(self):
        c = copy.deepcopy(self.original)
        for run in c["configurations"]:
            run.update(requested_k=1, returned_k=1, pairs=run["pairs"][:1])
        self.rejected(c)

    def test_mat_mismatch_and_cross_certificate_parameter_mismatch_reject(self):
        c = copy.deepcopy(self.original); c["input"]["mat_sha256"] = "0" * 64
        self.rejected(c)
        c = copy.deepcopy(self.original); c["input"]["parameters"]["F"] += .001
        mat_cert = workflow.load(self.directory / "summary.json")["mat_certificate"]
        with self.assertRaises(ValueError):
            workflow.validate_spectral_record(c, self.hash, mat_cert)

    def test_protocol_cannot_be_selected_by_changing_schema_or_task(self):
        with self.assertRaises(ValueError):
            workflow.validate_spectral_record(self.original, self.hash, surface="sphere")

    def test_near_zero_negative_with_forged_negative_flag_is_rejected(self):
        c = copy.deepcopy(self.original)
        for run in c["configurations"]:
            run["pairs"][0]["eigenvalue"][0] = -1e-10
            run["full_max_real"] = -1e-10
        self.rejected(c)


class CertificateCommandExitTests(unittest.TestCase):
    def test_sphere_certificate_command_never_exits_success_for_failed_acceptance(self):
        for passed in (True, False):
            argv = ["i02_stability.py", "certificate", "--worker", "--out",
                    str(sphere_cli.BASE / "unit-test-no-files"), "--mat", "unused.mat",
                    "--summary", "unused.json"]
            with self.subTest(passed=passed), patch.object(sys, "argv", argv), \
                    patch.object(sphere_cli, "certificate", return_value={"numerical_check_passed": passed}), \
                    self.assertRaises(SystemExit) as exc:
                sphere_cli.main()
            self.assertEqual(exc.exception.code, 0 if passed else 2)

    def test_frog_certificate_command_rejects_positive_and_insufficient_status(self):
        for status in ("numerical_check_passed", "unstable_positive_mode", "insufficient_evidence"):
            with tempfile.TemporaryDirectory() as tmp, self.subTest(status=status), \
                    patch.object(sys, "argv", ["i03_frog_stability.py", "--task", "frog_spots", "--out", tmp]), \
                    patch.object(frog_cli, "run_task", return_value={"status": status}), \
                    patch.object(frog_cli, "write_json_exclusive"):
                self.assertEqual(frog_cli.main(), 0 if status == "numerical_check_passed" else 2)


if __name__ == "__main__":
    unittest.main()
