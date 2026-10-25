"""Exercise statistics and actual generators, without PDE solves or training."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import evidence_stats as E
from scripts import generalization_eval as G
from scripts import active_learning as A
from scripts import update_scientific_docs as D
from scripts import report_v3 as R

ROOT = Path(__file__).resolve().parents[1]
FONT = Path(os.environ.get("AIC_REPORT_FONT", r"C:\Windows\Fonts\Arial Unicode MS.ttf"))


def sample(real=-1e-4, absolute=1e-14):
    return {"eigenvalues": [[real, 0.]], "pairs": [{"eigenvalue": [real, 0.],
            "residual_absolute_M": absolute, "residual_relative_M": 1e-10,
            "residual_passed": True}], "solver_converged": True,
            "all_pairs_validated": True, "sampled_signs_resolved_negative": True}


class SpectralStrataTests(unittest.TestCase):
    def test_unattempted_gate_is_distinct_from_missing_training_evidence(self):
        row = {"rel_residual": 1e-4, "max_re_lambda": None,
               "stability_scope": "single_shift_unprojected_sample"}
        self.assertEqual(E.spectral_assessment(row), "not_attempted")
        self.assertEqual(E.spectral_assessment({"rel_residual": 1e-14, "stable": True}), "not_recorded")

    def test_attempted_error_without_solver_status_remains_unspecified(self):
        self.assertEqual(E.spectral_assessment({"stability_error": "failed"}), "attempted_insufficient_detail")

    def test_nonconvergence_is_not_invalid_pairs_or_unattempted(self):
        s = sample(); s["solver_converged"] = False
        self.assertEqual(E.spectral_assessment({"linear_stability_sample": s}), "solver_not_converged")

    def test_bad_pair_residual_overrides_a_true_pass_flag(self):
        self.assertEqual(E.spectral_assessment({"linear_stability_sample": sample(absolute=1e-5)}), "invalid_eigenpairs")

    def test_near_zero_is_unresolved_despite_negative_flag(self):
        self.assertEqual(E.spectral_assessment({"linear_stability_sample": sample(real=-1e-14)}), "sign_unresolved")

    def test_reliable_positive_cannot_be_hidden_by_solver_or_summary_flags(self):
        s = sample(real=1e-6); s["solver_converged"] = False; s["pairs"][0]["residual_passed"] = False
        self.assertEqual(E.spectral_assessment({"linear_stability_sample": s}), "reliable_positive")

    def test_valid_negative_sample_is_only_a_single_shift_pass(self):
        self.assertEqual(E.spectral_assessment({"linear_stability_sample": sample()}), "numerical_sample_passed")

    def test_boolean_only_summary_cannot_fabricate_pair_validation(self):
        for value in (True, False):
            row = {"linear_stability_sample": {"all_pairs_validated": value, "solver_converged": True}}
            self.assertEqual(E.spectral_assessment(row), "attempted_insufficient_detail")

    def test_missing_or_nonfinite_pair_metrics_are_insufficient(self):
        s = sample(); s["pairs"][0]["residual_absolute_M"] = float("nan")
        self.assertEqual(E.spectral_assessment({"linear_stability_sample": s}), "attempted_insufficient_detail")

    def test_inconsistent_pair_and_eigenvalue_records_cannot_pass(self):
        s = sample(); s["eigenvalues"][0][0] = -1e-2
        self.assertEqual(E.spectral_assessment({"linear_stability_sample": s}), "attempted_insufficient_detail")

    def test_full_steady_requires_positive_finite_residual_record(self):
        for value in (None, float("nan"), float("inf"), -1., 1e-5):
            self.assertFalse(E.full_steady({"rel_residual": value, "converged": True}))
        self.assertTrue(E.full_steady({"rel_residual": 1e-11, "converged": False}))


class ActualEvidenceGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q, cls.predictions = E.recompute_quality(ROOT)
        cls.test = E.load(ROOT / "submission/evidence/generalization/testset.json")["rows"]
        cls.c = E.context(ROOT, cls.q)
        cls.table = E.load(ROOT / "submission/evidence/generalization/summary.json")

    def test_duplicate_counts_filter_both_terms_by_surface(self):
        for surface in E.SURFACES:
            result = E.counts(self.test, surface)
            self.assertEqual(result["n"], 64)
            self.assertEqual(result["duplicate_rows"], 0)
        duplicate = next(r for r in self.test if r["surface"] == "frog")
        mixed = self.test + [duplicate]
        self.assertEqual(E.counts(mixed, "frog")["duplicate_rows"], 1)
        self.assertEqual(E.counts(mixed, "sphere5")["duplicate_rows"], 0)

    def test_frog_failed_attempts_are_not_counted_as_unchecked(self):
        r = self.q["test"]["frog"]
        self.assertEqual(r["spectrum_attempted"], 57)
        self.assertEqual(r["spectral_status"]["numerical_sample_passed"], 50)
        self.assertEqual(r["spectral_status"]["invalid_eigenpairs"], 7)
        self.assertEqual(r["spectral_status"]["not_attempted"], 7)
        self.assertEqual(sum(r["spectral_status"].values()), r["n"])

    def test_model_inference_matches_recorded_endpoint_and_steady_fractions(self):
        expected = {"frog": {"v1": (44,64,39,57), "v2": (59,64,53,57)},
                    "sphere5": {"v1": (57,64,16,16), "v2": (61,64,16,16)}}
        for surface, models in expected.items():
            for model, values in models.items():
                a = self.q["accuracy"][surface][model]
                self.assertEqual((a["all"]["correct"],a["all"]["n"],a["steady"]["correct"],a["steady"]["n"]), values)

    def test_sphere_steady_subset_has_no_nontrivial_samples(self):
        self.assertEqual(self.q["test"]["sphere5"]["steady_types"], {"uniform": 16})
        self.assertEqual(self.q["totals"], {"endpoints": 128, "steady": 73})
        for model in ("v1", "v2"):
            self.assertEqual(self.q["accuracy"]["sphere5"][model]["steady_nontrivial"]["n"], 0)

    def test_interpolation_and_independence_are_computed_from_actual_keys(self):
        for surface in E.SURFACES:
            d = self.q["independence"][surface]
            self.assertEqual(d["scope"], "training_box_interpolation")
            self.assertEqual(d["overlapping_points"], 0)
            self.assertEqual(d["overlapping_seeds"], 0)

    def test_training_missing_spectrum_is_not_a_failed_or_unattempted_solve(self):
        for surface in E.SURFACES:
            d = self.q["dense"][surface]
            rows=E.load(ROOT/f"submission/evidence/generalization/dense_dataset_{surface}.json")["rows"]
            errors=sum(bool(r.get("stability_error")) for r in rows)
            self.assertEqual(d["spectral_status"]["attempted_insufficient_detail"],errors)
            self.assertEqual(d["spectral_status"]["not_recorded"],d["n"]-errors)
            self.assertEqual(d["spectral_status"]["not_attempted"],0)

    def test_markdown_generators_do_not_call_training_or_pde(self):
        with patch.object(G, "fit", side_effect=AssertionError("training forbidden")), \
                patch.object(G.A, "run_pde", side_effect=AssertionError("PDE forbidden")), \
                patch.object(G.A, "run_ai", side_effect=AssertionError("PDE forbidden")):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "summary.md"
                G.write_markdown(self.table, self.q, path)
                text = path.read_text(encoding="utf8") + A.render_markdown(self.c["active"])
                self.assertNotIn("逐稳态", text)
                self.assertNotIn("128 个 PDE 稳态", text)
                self.assertIn("无样本", text)
                self.assertIn("57/64", text)
                self.assertIn("16/64", text)
                self.assertIn("2.38×", text)
                self.assertIn("不是完整墙钟加速", text)

    def test_generator_rejects_a_stale_metric_table(self):
        table = copy.deepcopy(self.table); table["frog"]["v2"]["type_accuracy_states"] = 1.
        with self.assertRaises(ValueError): G.render_markdown(table, self.q)

    def test_changed_data_changes_rendered_count_instead_of_using_a_literal(self):
        q = copy.deepcopy(self.q); q["totals"]["endpoints"] = 130
        self.assertIn("共 130 个", E.quality_markdown(q))

    def test_active_efficiency_is_recomputed_and_rejects_summary_disagreement(self):
        e = next(iter(E.end_to_end_statistics(self.c["active"]).values()))
        self.assertAlmostEqual(e["median_integration_step_ratio"], 25875/10875)
        self.assertFalse(e["wall_clock_speedup_verified"])
        bad = copy.deepcopy(self.c["active"])
        next(iter(bad["end_to_end"].values()))["summary"]["ai v2"]["median_steps"] += 1
        with self.assertRaises(ValueError): E.end_to_end_statistics(bad)

    def test_band_claim_and_match_objectives_come_from_records(self):
        facts = E.claims(self.c)
        self.assertIn("10/10", facts["bands"])
        self.assertIn("均为 6 条带", facts["bands"])
        self.assertIn("独立协议", facts["bands"])
        self.assertIn("固定 F=", facts["inverse"])
        for value in ("cvt_3", "1.950", "1.554", "1.748"):
            self.assertIn(value, facts["inverse"])
        self.assertIn("不能混用", facts["inverse"])
        c = copy.deepcopy(self.c); c["bands"]["selected_counts"] = {"7": 10}
        self.assertIn("均为 7 条带", E.claims(c)["bands"])

    def test_current_document_generator_is_repeatable_in_a_temporary_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in ("README.md", "HANDOVER.md", "submission/README_提交.md",
                        "submission/evidence/generalization/summary.json"):
                dest=root/rel; dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes((ROOT/rel).read_bytes())
            with patch.object(E, "context", return_value=self.c):
                first=D.render_documents(root,self.q,37)
                for rel,text in first.items():
                    dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text,encoding="utf8")
                self.assertEqual(D.render_documents(root,self.q,37),first)
            self.assertIn("37 项单元测试", first["README.md"])

    def test_report_sections_use_the_supplied_execution_count(self):
        sections=R.report_sections(self.c,37)
        self.assertIn("37 项单元测试",sections["verification"])
        self.assertIn("1.950",sections["inverse"])
        self.assertIn("不是完整墙钟加速",sections["efficiency"])

    @unittest.skipUnless(FONT.is_file(), "set AIC_REPORT_FONT to an available CJK scientific font")
    def test_actual_pdf_renderer_has_two_pages_and_preserves_scientific_symbols(self):
        import fitz
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"report.pdf"
            result=R.render_report(self.c,ROOT/"submission",font_path=FONT,output=out,tests_run=37,date="fixture")
            with fitz.open(out) as doc:
                text="".join(p.get_text() for p in doc)
                self.assertEqual(len(doc),2)
                for value in ("∂", "√", "≤", "1.950", "1.554", "1.748", "2.38", "37 项"):
                    self.assertIn(value,text)
                self.assertIn("1e-11",text)
                self.assertNotIn("\ufffd",text)
            # Test the actual wrapper at a forced word boundary. Concatenated
            # PDF text spans cannot distinguish wrapping from adjacent table cells.
            font=fitz.Font(fontfile=str(FONT))
            value="abcd abcd 1e-11 1.950 1.554 57/64"
            wrapped=R.wrap_lines(font,value,8.7,60)
            self.assertEqual("".join(wrapped),value)
            for token in ("1e-11","1.950","1.554","57/64"):
                self.assertTrue(any(token in line for line in wrapped))
            self.assertLess(result["page2_bottom"],800)
            before=out.read_bytes()
            with self.assertRaises(FileExistsError):
                R.render_report(self.c,ROOT/"submission",font_path=FONT,output=out,tests_run=37)
            self.assertEqual(out.read_bytes(),before)

    def test_model_missing_at_inference_never_triggers_automatic_fitting(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(G,"DENSE",Path(tmp)), \
                patch.object(G,"fit",side_effect=AssertionError("must not fit")):
            with self.assertRaises(FileNotFoundError): G.model_v2("frog")

    def test_test_evidence_must_match_executed_current_sources(self):
        record={"schema_version":"executed-unittest-v1","tests_run":37,"successful":True,
                "failures":0,"errors":0,"skipped":0,"source_fingerprint":E.source_fingerprint(ROOT)}
        self.assertEqual(E.validate_test_record(record,ROOT),37)
        for change in ({"successful":False},{"source_fingerprint":"old"},{"skipped":1},{"errors":1}):
            with self.assertRaises(ValueError): E.validate_test_record({**record,**change},ROOT)


if __name__ == "__main__":
    unittest.main()
