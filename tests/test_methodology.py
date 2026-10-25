"""F8/F9 and saved-vector evidence regressions; no training or production grid solves."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from turing_rd import Geometry, SPOTS, icosphere
from turing_rd import quality as Q
from turing_rd import inverse as I
from turing_rd import spectrum_evidence as S
from turing_rd.result_io import save_certified_mat, file_sha256
from turing_rd.stability import eigenpair_diagnostics
from turing_rd.spectral_policy import INVERSION_SCREEN_CONFIGS, PAIR_ABSOLUTE_TOL, PAIR_RELATIVE_TOL
from scripts import methodology_audit as M
from scripts import train_quality as T
from scripts import invert_target as IT
from scripts import submission_workflow as SW


def sample_report(k=8, real=-1e-4):
    pairs = [{"eigenvalue": [real - j * 1e-5, 0.], "residual_absolute_M": 1e-14,
              "residual_relative_M": 1e-10, "residual_passed": True,
              "positive_reliable": real > 0, "negative_resolved": real < 0} for j in range(k)]
    return {"sigma": .001, "which": "LM", "requested_k": 8, "returned_k": k,
            "pairs": pairs, "solver_converged": True, "solver_error": None,
            "all_pairs_validated": True, "full_max_real": real, "found_reliable_positive": real > 0,
            "pair_absolute_tolerance": PAIR_ABSOLUTE_TOL, "pair_relative_tolerance": PAIR_RELATIVE_TOL}


def dense_sample(J, mass, root, cfg):
    values, vectors = np.linalg.eig(J.toarray())
    order = np.argsort(-values.real)[:cfg["k"]]
    values = values[order].astype(np.complex128); vectors = vectors[:, order].astype(np.complex128)
    pairs = eigenpair_diagnostics(J, values, vectors, mass)
    report = {"sigma": cfg["sigma"], "which": cfg["which"], "requested_k": cfg["k"], "returned_k": len(values),
              "solver_converged": True, "solver_error": None, "all_pairs_validated": True,
              "pairs": pairs, "full_max_real": float(values.real.max()), "found_reliable_positive": False,
              "method": "tiny-grid-independent-dense-eigensolve",
              "pair_absolute_tolerance": PAIR_ABSOLUTE_TOL, "pair_relative_tolerance": PAIR_RELATIVE_TOL}
    return report, vectors


class LabelQualityTests(unittest.TestCase):
    def row(self, **kwargs):
        return {"surface": "frog", "F": .028, "k": .062, "seed": 1, "type": "spots", "rel_residual": 1e-14, **kwargs}

    def test_bad_residual_overrides_converged_and_stable_flags(self):
        row = self.row(rel_residual=1., converged=True, stable=True)
        self.assertFalse(Q.label_quality(row)["complete_steady"])

    def test_good_residual_does_not_require_legacy_boolean(self):
        self.assertTrue(Q.label_quality(self.row(converged=False))["complete_steady"])

    def test_nonfinite_negative_missing_and_boolean_residuals_reject(self):
        for value in (None, float("nan"), float("inf"), -1., True):
            with self.subTest(value=value):
                self.assertFalse(Q.label_quality(self.row(rel_residual=value))["complete_steady"])

    def test_scalar_spectrum_summary_is_insufficient(self):
        q = Q.label_quality(self.row(stable=True, max_re_lambda=-1.))
        self.assertFalse(q["spectrum_checked_pass"]); self.assertTrue(q["spectrum_insufficient"])

    def test_declared_valid_finite_sample_passes_only_finite_scope(self):
        q = Q.label_quality(self.row(linear_stability_sample={"spectral_report": sample_report()}))
        self.assertTrue(q["spectrum_checked_pass"])
        self.assertFalse(q["spectral_decision"]["mathematical_full_spectrum_proof"])

    def test_missing_requested_count_does_not_become_pass(self):
        report = sample_report(); report.pop("requested_k")
        self.assertTrue(Q.label_quality(self.row(linear_stability_sample=report))["spectrum_insufficient"])

    def test_count_shortfall_rejects(self):
        self.assertFalse(Q.label_quality(self.row(linear_stability_sample=sample_report(k=7)))["spectrum_checked_pass"])

    def test_reliable_positive_cannot_be_hidden_by_summary(self):
        report = sample_report(real=1e-6); report["found_reliable_positive"] = False
        q = Q.label_quality(self.row(linear_stability_sample=report))
        self.assertTrue(q["spectrum_positive"]); self.assertFalse(q["spectrum_checked_pass"])

    def test_invalid_pair_rejects_even_if_certificate_tolerance_is_large(self):
        report = sample_report(); report["pairs"][0]["residual_absolute_M"] = 1e-5; report["pair_absolute_tolerance"] = 1.
        self.assertFalse(Q.label_quality(self.row(linear_stability_sample=report))["spectrum_checked_pass"])

    def test_policies_are_explicit_and_legacy_does_not_mutate_rows(self):
        rows = [self.row(rel_residual=.1), self.row(seed=2), self.row(seed=3, linear_stability_sample=sample_report())]
        before = copy.deepcopy(rows)
        self.assertEqual(Q.select_training_rows(rows), rows)
        self.assertEqual(len(Q.select_training_rows(rows, "all_endpoints")), 3)
        self.assertEqual(len(Q.select_training_rows(rows, "complete_steady")), 2)
        self.assertEqual(len(Q.select_training_rows(rows, "steady_and_spectrum")), 1)
        self.assertEqual(rows, before)
        with self.assertRaises(ValueError): Q.select_training_rows(rows, "guess")

    def test_mixed_terminal_types_are_not_automatically_steady_multistability(self):
        rows = [self.row(type="spots"), self.row(seed=2, type="holes", rel_residual=.1)]
        q = Q.quality_summary(rows)
        self.assertEqual(len(q["mixed_terminal_type_points"]), 1)
        self.assertEqual(len(q["mixed_complete_steady_type_points"]), 0)

    def test_default_quality_entry_never_trains(self):
        with tempfile.TemporaryDirectory() as tmp, patch("generalization_eval.fit", side_effect=AssertionError("must not train")):
            root = Path(tmp); source = root / "input.json"; source.write_text(json.dumps({"rows": [self.row()]}))
            result = T.run(source, root / "new", "complete_steady")
            self.assertFalse(result["training_executed"])
            self.assertFalse(list((root / "new").glob("*.joblib")))

    def test_quality_training_refuses_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError): T.run("missing", Path(tmp), "complete_steady", True)

    def test_empty_strict_evidence_refuses_training(self):
        with tempfile.TemporaryDirectory() as tmp, patch("generalization_eval.fit", side_effect=AssertionError("must not train")):
            root = Path(tmp); source = root / "input.json"; source.write_text(json.dumps({"rows": [self.row()]}))
            with self.assertRaises(ValueError): T.run(source, root / "new", "steady_and_spectrum", True)


class ReferenceQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = Geometry(icosphere(1), cache_dir=None)

    def record(self, uniform=True):
        u = np.ones(self.g.n) if uniform else .7 + .05 * self.g.V[:, 0]
        v = np.zeros(self.g.n) if uniform else .1 + .02 * self.g.V[:, 2]
        return {"F": SPOTS.F, "k": SPOTS.k, "u": u, "v": v, "status": "moving", "polished": False,
                "rate": 1e-8, "full_steady_diagnostic": Q.full_residual_record(self.g, SPOTS, u, v)}

    def test_low_rate_large_residual_is_legacy_only(self):
        r = self.record(False)
        self.assertTrue(I.is_steady(r)); self.assertTrue(I.reference_usable(r, I.LEGACY_REFERENCE_PROTOCOL))
        self.assertFalse(I.reference_usable(r)); self.assertTrue(I.reference_quality(r)["low_rate"])

    def test_polished_large_residual_is_not_full_steady(self):
        r = self.record(False); r.update(polished=True, status="steady")
        self.assertTrue(I.reference_quality(r)["newton_polished"])
        self.assertFalse(I.reference_usable(r))

    def test_full_two_equation_residual_can_pass_without_polish_flag(self):
        r = self.record()
        self.assertTrue(I.reference_usable(r)); self.assertFalse(I.reference_quality(r)["newton_polished"])

    def test_no_full_residual_record_rejects(self):
        r = self.record(); r.pop("full_steady_diagnostic"); r["residual"] = 0.; r["polished"] = True
        self.assertFalse(I.reference_usable(r)); self.assertFalse(I.reference_quality(r)["full_steady"])

    def test_null_and_malformed_diagnostics_reject(self):
        for value in (None, {}, {"parameters": None}):
            r = self.record(); r["full_steady_diagnostic"] = value
            self.assertFalse(I.reference_usable(r)); self.assertFalse(I.reference_quality(r)["full_steady"])

    def test_nonfinite_or_negative_residual_rejects(self):
        for value in (None, float("nan"), float("inf"), -1., True):
            r = self.record(); r["full_steady_diagnostic"]["value"] = value
            self.assertFalse(I.reference_usable(r))

    def test_missing_one_equation_rejects(self):
        r = self.record(); del r["full_steady_diagnostic"]["Ru_rms"]
        self.assertFalse(I.reference_usable(r))

    def test_wrong_normalization_or_rotation_projection_rejects(self):
        for key, value in (("normalization", "reaction"), ("rotation_projection", True), ("formula", "rate")):
            r = self.record(); r["full_steady_diagnostic"][key] = value
            self.assertFalse(I.reference_usable(r))

    def test_spectral_metadata_alone_is_not_steady(self):
        self.assertFalse(I.reference_usable({"spectral_report": sample_report(), "polished": True}))

    def test_other_parameters_or_quantized_states_do_not_match(self):
        r = self.record(); r["F"] += .001; self.assertFalse(I.reference_usable(r))
        r = self.record(); r["u"] = r["u"].astype(np.float32); self.assertFalse(I.reference_usable(r))
        r = self.record(); r["v"][0] += 1e-8; self.assertFalse(I.reference_usable(r))

    def test_new_finish_really_measures_nonuniform_fields(self):
        r = I.steady_states(self.g, [.028], [.062], [2], SPOTS.Du, SPOTS.Dv,
                            t_fast=1., t_max=3., dt=1., chunk=1., polish=False)[0]
        actual = Q.full_residual_record(self.g, SPOTS, r["u"], r["v"])
        self.assertEqual(actual, r["full_steady_diagnostic"])
        self.assertGreater(np.ptp(r["v"]), 0.)

    def test_unknown_reference_protocol_fails(self):
        with self.assertRaises(ValueError): I.reference_usable(self.record(), "guess")

    def test_strict_grid_persists_float64_and_legacy_grid_stays_float32(self):
        r = self.record(); r.update(seed=2, t=3., residual=0., ptc_iters=0, wall_s=0.)
        r["reference_quality"] = I.reference_quality(r)
        cfg = dict(t_max=3., dt=1., rate_tol=1e-6, polish_tol=1e-10)
        with patch.object(IT, "W", {"g": self.g, "lp": {1.: None}}), patch.object(IT, "steady_states", return_value=[r]), \
                patch.object(IT, "features_of", return_value={"type": "uniform"}):
            cfg["reference_protocol"] = I.STRICT_REFERENCE_PROTOCOL
            strict = IT.job_grid((.028, .062, [2], cfg))[0]
            cfg.pop("reference_protocol"); legacy = IT.job_grid((.028, .062, [2], cfg))[0]
        self.assertEqual(strict["u"].dtype, np.float64); self.assertTrue(I.reference_usable(strict))
        self.assertEqual(legacy["u"].dtype, np.float32); self.assertFalse(I.reference_usable(legacy))

    def test_strict_surrogate_does_not_fall_back_to_penalized_medians(self):
        with self.assertRaisesRegex(ValueError, "no legacy fallback"):
            IT.surrogate_support([0., .5, 0.], I.STRICT_REFERENCE_PROTOCOL)
        feasible, fallback = IT.surrogate_support([0., .5, 0.], I.LEGACY_REFERENCE_PROTOCOL)
        self.assertTrue(feasible.all()); self.assertTrue(fallback)
        feasible, fallback = IT.surrogate_support([.5, 1., 1., 0.], I.STRICT_REFERENCE_PROTOCOL)
        self.assertEqual(feasible.tolist(), [True, True, True, False]); self.assertFalse(fallback)

    def test_strict_representative_requires_final_full_residual_and_pair_evidence(self):
        good = self.record(); good.update(label="estimate", distance=2., spectral_report=sample_report())
        bad = self.record(False); bad.update(label="estimate", distance=0., spectral_report=sample_report())
        reps = IT.select_representatives([bad, good], ["estimate"], I.STRICT_REFERENCE_PROTOCOL)
        self.assertIs(reps["estimate"], good)
        for candidate in (bad, {**good, "spectral_report": None},
                          {**good, "spectral_report": sample_report(k=7)},
                          {**good, "spectral_report": sample_report(real=1e-6)}):
            with self.subTest(candidate=candidate.get("distance")):
                with self.assertRaisesRegex(ValueError, "no full-steady"):
                    IT.select_representatives([candidate], ["estimate"], I.STRICT_REFERENCE_PROTOCOL)

    def test_explicit_legacy_representative_fallback_is_preserved(self):
        r = {"label": "estimate", "distance": 1., "rel_residual": 1., "stability": None}
        self.assertIs(IT.select_representatives([r], ["estimate"], I.LEGACY_REFERENCE_PROTOCOL)["estimate"], r)

    def test_final_diagnostic_measures_fields_after_additional_polish(self):
        r = self.record(); r.update(t=1., residual=0., wall_s=0., status="steady")
        after = self.record(False)
        cfg = dict(t_max_final=3., dt=1., rate_tol=1e-6, polish_tol=1e-10,
                   reference_protocol=I.STRICT_REFERENCE_PROTOCOL)
        with patch.object(IT, "W", {"g": self.g, "fx": Mock(describe=Mock(return_value={"type": "spots"}))}), \
                patch.object(IT, "steady_states", return_value=[r]), \
                patch.object(IT, "features_of", return_value={"type": "spots"}), \
                patch("turing_rd.steady.ptc_polish", return_value=(after["u"], after["v"], [{"rel_residual": 1e-14}])), \
                patch("turing_rd.diagnostics.linear_stability", return_value={"spectral_report": sample_report()}):
            result = IT.job_final(("estimate", SPOTS.F, SPOTS.k, 2, cfg, 1.))
        self.assertEqual(result["full_steady_diagnostic"], after["full_steady_diagnostic"])
        self.assertFalse(I.reference_usable(result))


class SavedSpectralEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = Geometry(icosphere(1), cache_dir=None)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.mat = self.root / "state.mat"
        self.cert = save_certified_mat(self.mat, self.g, SPOTS, np.ones(self.g.n), np.zeros(self.g.n), seed=2)
        self.out = self.root / "evidence"
        self.evidence = S.save_evidence(self.mat, self.cert, self.g, SPOTS, self.out, dense_sample)

    def validate(self, **kwargs):
        return S.validate_evidence(self.out / "certificate.json", self.mat, self.g, SPOTS, 2, **kwargs)

    def rewrite_index(self):
        (self.out / "certificate.json").write_text(json.dumps(self.evidence), encoding="utf-8")

    def test_saved_vectors_independently_recompute_and_pass(self):
        result = self.validate()
        self.assertTrue(result["eigenpair_residuals_recomputed_from_saved_vectors"])
        self.assertEqual(result["decision"]["checked_eigenpairs"], 40)
        self.assertFalse(result["fresh_eigensolve_in_this_verification"])
        self.assertFalse(result["mathematical_full_spectrum_proof"])

    def test_metadata_only_mode_is_explicitly_not_vector_recomputation(self):
        self.assertFalse(self.validate(recompute_pairs=False)["eigenpair_residuals_recomputed_from_saved_vectors"])

    def test_wrong_mat_hash_rejects(self):
        self.evidence["input"]["mat_sha256"] = "0" * 64; self.rewrite_index()
        with self.assertRaises(ValueError): self.validate()

    def test_missing_vectors_rejects(self):
        (self.out / "eigenpairs_0.npz").unlink()
        with self.assertRaises((ValueError, FileNotFoundError)): self.validate()

    def test_corrupt_vectors_rejects(self):
        (self.out / "eigenpairs_0.npz").write_bytes(b"broken")
        with self.assertRaises(ValueError): self.validate()

    def test_changed_vectors_cannot_be_blessed_by_updated_file_hash(self):
        file = self.out / "eigenpairs_0.npz"
        with np.load(file) as data: values, vectors = data["eigenvalues"], data["eigenvectors"].copy()
        vectors[0, 0] += .1
        np.savez_compressed(file, eigenvalues=values, eigenvectors=vectors)
        self.evidence["artifacts"][0]["vectors_sha256"] = file_sha256(file); self.rewrite_index()
        with self.assertRaises(ValueError): self.validate()

    def test_insufficient_vector_count_rejects(self):
        file = self.out / "eigenpairs_0.npz"
        with np.load(file) as data: values, vectors = data["eigenvalues"][:-1], data["eigenvectors"][:, :-1]
        np.savez_compressed(file, eigenvalues=values, eigenvectors=vectors)
        self.evidence["artifacts"][0]["vectors_sha256"] = file_sha256(file); self.rewrite_index()
        with self.assertRaises(ValueError): self.validate()

    def test_historical_summary_cannot_be_upgraded_to_new_certificate(self):
        (self.out / "certificate.json").write_text(json.dumps({"configs": [{"solver_converged": True}]}))
        with self.assertRaises(ValueError): self.validate()

    def test_tampered_decision_rejects(self):
        self.evidence["decision"]["full_max_real_checked"] = -100.; self.rewrite_index()
        with self.assertRaises(ValueError): self.validate()

    def test_full_spectrum_claim_rejects(self):
        self.evidence["mathematical_full_spectrum_proof"] = True; self.rewrite_index()
        with self.assertRaises(ValueError): self.validate()

    def test_failed_solve_produces_failed_evidence_not_fake_pairs(self):
        def fail(*args): raise RuntimeError("deliberate test failure")
        out = self.root / "failed"
        evidence = S.save_evidence(self.mat, self.cert, self.g, SPOTS, out, fail)
        self.assertFalse(evidence["decision"]["numerical_check_passed"])
        self.assertTrue(all(not r["pairs"] for r in evidence["configs"]))
        with self.assertRaises(ValueError): S.validate_evidence(out / "certificate.json", self.mat, self.g, SPOTS, 2)

    def test_existing_evidence_cannot_be_overwritten(self):
        with self.assertRaises(FileExistsError): S.save_evidence(self.mat, self.cert, self.g, SPOTS, self.out, dense_sample)

    def package_fixture(self, declared=None):
        submission = self.root / "submission"; folder = submission / "task3_inversion/best_match"
        folder.mkdir(parents=True)
        (folder / "solution_best_match.mat").write_bytes(self.mat.read_bytes())
        summary = {"F": SPOTS.F, "k": SPOTS.k, "best": {"seed": 2}, "mat_certificate": self.cert,
                   "stability": {"configs": [{"solver_converged": True, "full_max_real": -1.}]}}
        if declared is not None:
            summary["stability"]["evidence_file"] = declared
        (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (folder.parent / "summary.json").write_text(json.dumps({"estimate": {"F": SPOTS.F, "k": SPOTS.k}}))
        return submission

    def test_package_verification_keeps_historical_summary_insufficient(self):
        result = SW.verify_best_match(self.package_fixture(), self.g)
        self.assertTrue(result["pde_residual_recomputed"])
        self.assertEqual(result["historical_spectrum_status"], "insufficient_evidence")
        self.assertFalse(result["saved_spectral_metadata_verified"])
        self.assertFalse(result["fresh_eigensolve_in_this_verification"])

    def test_package_verification_uses_independent_vectors_without_fresh_solve(self):
        result = SW.verify_best_match(self.package_fixture(), self.g, self.out / "certificate.json")
        self.assertTrue(result["saved_spectral_metadata_verified"])
        self.assertTrue(result["eigenpair_residuals_recomputed_from_saved_vectors"])
        self.assertFalse(result["fresh_eigensolve_in_this_verification"])

    def test_declared_missing_or_escaping_evidence_cannot_be_silently_ignored(self):
        submission = self.package_fixture("missing/certificate.json")
        with self.assertRaises(FileNotFoundError): SW.verify_best_match(submission, self.g)
        summary_path = submission / "task3_inversion/best_match/summary.json"
        summary = json.loads(summary_path.read_text()); summary["stability"]["evidence_file"] = "../../../escape.json"
        summary_path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "escapes"): SW.verify_best_match(submission, self.g)


class MethodologyGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = M.build(ROOT)

    def test_actual_raw_training_counts(self):
        expected = {"v1_frog": (45,32), "v1_sphere5": (45,15), "v2_frog": (306,257), "v2_sphere5": (306,90)}
        for name, (n, steady) in expected.items():
            row = self.data["datasets"][name]
            self.assertEqual((row["total"], row["complete_steady"]), (n, steady))
            self.assertEqual(row["spectrum_checked_pass"], 0)

    def test_inverse_counts_have_explicit_aggregate_scope(self):
        inv = self.data["inversion"]
        self.assertEqual(inv["total"], 294)
        self.assertEqual(inv["counts_from_aggregate_only"], {"polished": 130, "legacy_accepted": 176})
        self.assertEqual(inv["extra_legacy_accepted"], 46)
        self.assertIsNone(inv["actual_strict_steady_count"])
        self.assertIsNone(inv["strict_parameter_shift"])

    def test_loss_surface_is_recomputed_from_retained_seed_distances(self):
        self.assertEqual(len(self.data["inversion"]["landscape"]), 49)
        self.assertTrue(all(r["strict_loss"] is None for r in self.data["inversion"]["landscape"]))

    def test_sphere_steady_test_subset_is_uniform(self):
        self.assertEqual(self.data["test"]["sphere5"]["steady_types"], {"uniform": 16})

    def test_best_match_is_fixed_parameter_search_without_causal_ablation(self):
        b = self.data["best_match"]
        self.assertEqual((b["layout"], b["initial_spots"], b["final_spots"]), ("cvt_3", 20, 16))
        self.assertAlmostEqual(b["J_previous"], 1.9502571447760688)
        self.assertAlmostEqual(b["J_best"], 1.5539702580094354)
        self.assertFalse(b["equal_budget_non_AI_ablation_executed"])

    def test_pure_generator_preserves_unknowns_and_scientific_scope(self):
        text = M.render_report(self.data)
        for phrase in ("真实严格稳态数未知", "固定 F/k", "AI 独立因果贡献未证明", "不能由此推导严格损失曲面", "16/16 不证明"):
            self.assertIn(phrase, text)
        self.assertNotIn("128 个稳态", text)


if __name__ == "__main__":
    unittest.main()
