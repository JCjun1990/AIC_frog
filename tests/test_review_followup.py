"""Low-cost regressions for seed statistics and residual-only portability.

Worker rows and cross-platform residual reductions are simulated, not new PDE runs.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scripts import ai_generation as A
from scripts import submission_workflow as W
from turing_rd import Geometry, SPOTS, icosphere
from turing_rd import result_io as IO


class RobustnessSeedMergeTests(unittest.TestCase):
    def test_seed_two_replaces_only_seed_two_and_keeps_zero(self):
        baseline = {"seed": 2, "source": "formal"}
        rows = [{"seed": 0}, {"seed": 2, "source": "worker"}]
        merged = A.merge_robustness_seeds(rows, baseline)
        self.assertEqual([r["seed"] for r in merged], [2, 0])
        self.assertEqual(merged[0]["source"], "formal")
        self.assertEqual(rows, [{"seed": 0}, {"seed": 2, "source": "worker"}])
        self.assertIsNot(merged[0], baseline)

    def test_zero_and_overridden_baselines_and_repeated_seeds(self):
        for seed in (0, 2, 5):
            with self.subTest(seed=seed):
                merged = A.merge_robustness_seeds([{"seed": s} for s in (0, 2, 2, 5)], {"seed": seed})
                self.assertEqual(merged[0]["seed"], seed)
                self.assertEqual(len(merged), 3)
                self.assertEqual({r["seed"] for r in merged}, {0, 2, 5})
        self.assertEqual(A.merge_robustness_seeds([]), [])

    def test_missing_or_conflicting_seed_is_never_guessed(self):
        for row in ({}, {"seed": True}, {"seed": -1}, {"seed": 2, "mat_certificate": {"seed": 0}}):
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, "seed"):
                A.merge_robustness_seeds([], row)
        row = {"mat_certificate": {"seed": 2}}
        self.assertEqual(A.merge_robustness_seeds([], row)[0]["seed"], 2)
        self.assertNotIn("seed", row)

    def run_stage(self, explicit=False):
        p = A.STRIPES
        def row(seed):
            return {"surface": "sphere6", "F": p.F, "k": p.k, "seed": seed, "stable": False,
                    "type": "stripes", "n_components": 7, "wavelength": .2, "rel_residual": 1e-14,
                    "rel_residual_rot": 1e-15, "max_re_lambda": None, "solve_s": 1}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, seed in (("tasks", 0), ("tasks_float64", 2), ("custom", 5)):
                target = root / name / "sphere6_stripes"; target.mkdir(parents=True)
                (target / "summary.json").write_text(json.dumps(row(seed)), encoding="utf-8")
            with patch.object(A, "OUT", root), patch.object(A, "TASKS", {"stripes": p}), \
                    patch.object(A, "geom", return_value=SimpleNamespace(eigenbasis=Mock())), \
                    patch.object(A, "pool_map", return_value=[row(0), row(2)]) as pool:
                A.stage_robustness(lambda _: None, ["sphere6"], [0, 2, 2], 1,
                                   baseline_dir=root / "custom" if explicit else None)
                self.assertEqual([job[1][3] for job in pool.call_args.args[0]], [0, 2])
            report = json.loads((root / "tasks" / "robustness.json").read_text(encoding="utf-8"))
            return report["sphere6_stripes"]["seeds"]

    def test_stage_uses_generated_float64_baseline_and_unique_jobs(self):
        self.assertEqual(self.run_stage(), [2, 0])

    def test_stage_accepts_explicit_baseline_with_recorded_override(self):
        self.assertEqual(self.run_stage(explicit=True), [5, 0, 2])


class ResidualOnlyPortabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = Geometry(icosphere(1), laplacian="robust", cache_dir=None)

    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.mat = Path(temp.name) / "state.mat"
        self.cert = IO.save_certified_mat(self.mat, self.g, SPOTS, np.ones(self.g.n), np.zeros(self.g.n), seed=0)
        self.verify_original = IO.verify_mat

    def validate(self, cert=None):
        return IO.validate_certificate_portable(self.mat, cert or self.cert, expected_geometry=self.g,
                                                expected_params=SPOTS, expected_seed=0)

    def reduction(self, value):
        def changed(*args, **kwargs):
            result = self.verify_original(*args, **kwargs)
            result["residual"]["value"] = value
            result["residual"]["passed"] = bool(np.isfinite(value) and 0 <= value <= result["residual"]["tolerance"])
            result["accepted"] = result["residual"]["passed"]
            if not result["accepted"]:
                result["failure_reasons"].append("simulated_full_residual_failed")
            return result
        return patch.object(IO, "verify_mat", side_effect=changed)

    def test_identical_fingerprint_roundoff_uses_fixed_portable_budget(self):
        with self.reduction(2.85e-14):
            with self.assertRaisesRegex(ValueError, "residual does not match"):
                IO.validate_certificate(self.mat, self.cert, expected_geometry=self.g, expected_params=SPOTS, expected_seed=0)
            fresh = self.validate()
        self.assertTrue(fresh["accepted"])
        self.assertIn("identical L/M fingerprint", fresh["verification_mode"])
        self.assertFalse(fresh["portable_comparison"]["operator_fingerprint_differs"])
        self.assertEqual(fresh["portable_comparison"]["residual_absolute_budget"], 1e-12)

    def test_agreement_budget_boundary_and_nonfinite_values(self):
        for value, passes in ((.99e-12, True), (1.01e-12, False), (float("nan"), False), (float("inf"), False)):
            with self.subTest(value=value), self.reduction(value):
                if passes:
                    self.assertTrue(self.validate()["accepted"])
                else:
                    with self.assertRaises(ValueError): self.validate()

    def test_current_full_pde_failure_remains_failure(self):
        with self.reduction(1.01e-11), self.assertRaisesRegex(ValueError, "read-back acceptance failed"):
            self.validate()

    def test_recorded_residual_must_also_satisfy_its_tolerance(self):
        for recorded, current in ((1.01e-11, .99e-11), (-2e-14, 0.)):
            cert = copy.deepcopy(self.cert); cert["residual"]["value"] = recorded
            with self.subTest(recorded=recorded), self.reduction(current), self.assertRaises(ValueError):
                self.validate(cert)

    def test_residual_fallback_cannot_hide_other_certificate_errors(self):
        changes = (lambda c: c.update(stability_status="strictly_stable"),
                   lambda c: c["residual"].update(formula="different"),
                   lambda c: c["precision"].update(passed=False),
                   lambda c: c["residual"].update(tolerance=1.),
                   lambda c: c.update(mat_sha256="0" * 64))
        for index, change in enumerate(changes):
            cert = copy.deepcopy(self.cert); change(cert)
            with self.subTest(index=index), self.reduction(2.85e-14), self.assertRaises(ValueError):
                self.validate(cert)

    def test_workflow_dispatches_residual_only_mismatch(self):
        exc = ValueError("MAT certificate residual does not match read-back recomputation")
        with self.reduction(2.85e-14):
            fresh = W.portable_acceptance(self.mat, self.cert, self.g, exc)
        self.assertTrue(fresh["accepted"])
        self.assertIn("identical L/M fingerprint", fresh["verification_mode"])


class ReleaseNoticeTests(unittest.TestCase):
    def test_generated_portable_scope_covers_same_fingerprint_roundoff(self):
        from scripts.update_scientific_docs import portable_verification_markdown
        text = portable_verification_markdown()
        self.assertIn("即使 L/M 指纹一致", text)
        self.assertIn("1e-11", text)
        self.assertIn("固定 1e-12", text)
        self.assertIn("不是完整 L/M 矩阵逐元素等价证明", text)
        self.assertNotIn("指纹不一致时才", text)

    def test_corrupted_report_note_fails_instead_of_reappearing_in_package(self):
        from scripts.update_scientific_docs import validate_release_note
        note = "Stage 8 报告中的 350 项为历史记录；本轮测试以实际执行 JSON 为准。"
        self.assertEqual(validate_release_note(note), note)
        for invalid in ("???? Stage 8 ????", "bad\ufffdtext", None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_release_note(invalid)

if __name__ == "__main__":
    unittest.main()
