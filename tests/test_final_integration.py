"""Final generators and portable format checks, without production simulations."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import evidence_stats as E
from scripts import report_v3 as R
from scripts import update_scientific_docs as D
from turing_rd.delivery_checks import audit_json, LEGACY_JSON

ROOT = Path(__file__).resolve().parents[1]


class DeliveryJsonTests(unittest.TestCase):
    def check_text(self, text, relative="evidence/new.json"):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / relative; path.parent.mkdir(parents=True)
            path.write_text(text, encoding="utf-8")
            return audit_json(root, [relative])

    def test_strict_json_passes(self):
        self.assertEqual(self.check_text('{"value":null,"x":1.25}')["strict_count"], 1)

    def test_unknown_nan_and_infinity_reject(self):
        for value in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "unapproved"):
                self.check_text('{"value":' + value + '}')

    def test_duplicate_keys_reject(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.check_text('{"x":1,"x":2}')

    def test_numeric_overflow_rejects(self):
        with self.assertRaisesRegex(ValueError, "overflowing"):
            self.check_text('{"x":1e999}')

    def test_truncated_json_rejects(self):
        with self.assertRaises(json.JSONDecodeError): self.check_text('{"x":')

    def test_legacy_path_does_not_authorize_arbitrary_changed_content(self):
        with self.assertRaisesRegex(ValueError, "unapproved"):
            self.check_text('{"value":NaN}', "task3_inversion/summary.json")

    def test_three_actual_legacy_inputs_remain_explicitly_hash_bound(self):
        result = audit_json(ROOT / "submission", list(LEGACY_JSON))
        self.assertEqual(result["legacy_count"], 3)
        self.assertEqual(result["strict_count"], 0)


class FinalGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q = E.load(ROOT / "submission/evidence/generalization/QUALITY_BREAKDOWN.json")
        cls.c = E.context(ROOT, cls.q)

    def test_supplemental_spectrum_claim_uses_saved_pair_count(self):
        facts = E.claims(self.c)["inverse"]
        self.assertIn("40 个特征对及向量", facts)
        self.assertNotIn("未重新求谱", facts)
        c = copy.deepcopy(self.c); c["best_match_spectrum"] = None
        self.assertIn("不能凭摘要升级", E.claims(c)["inverse"])
        self.assertNotIn("40 个特征对及向量", E.claims(c)["inverse"])

    def test_supplemental_mat_mismatch_rejects_before_generating_claims(self):
        real_sha = E.sha
        def wrong_mat(path):
            return "0" * 64 if Path(path).name == "solution_best_match.mat" else real_sha(path)
        with patch.object(E, "sha", side_effect=wrong_mat), self.assertRaisesRegex(ValueError, "another MAT"):
            E.context(ROOT, self.q)

    def test_final_document_generator_is_repeatable_and_drops_obsolete_current_notes(self):
        release = {"report": "report/new/report.pdf", "zip": "drafts/new/draft.zip"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in ("README.md", "HANDOVER.md", "submission/README_提交.md",
                        "submission/evidence/generalization/summary.json"):
                path = root / rel; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((ROOT / rel).read_bytes())
            with patch.object(E, "context", return_value=self.c):
                first = D.render_documents(root, self.q, 41, release)
                for rel, text in first.items():
                    path = root / rel; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text, encoding="utf-8")
                self.assertEqual(D.render_documents(root, self.q, 41, release), first)
            text = first["submission/README_提交.md"]
            self.assertIn("41 项单元测试", text)
            self.assertIn("report/new/report.pdf", text)
            self.assertIn("drafts/new/draft.zip", text)
            self.assertNotIn("第 7 阶段再统一", text)
            self.assertNotIn("尚未包含阶段 3 修订", first["README.md"])
            self.assertIn("构建前生成", first["submission/VALIDATION.md"])

    def test_report_preserves_methodological_unknowns_and_finite_spectrum_scope(self):
        sections = R.report_sections(self.c, 41)
        self.assertIn("41 项单元测试", sections["verification"])
        self.assertIn("严格稳态数和参数偏移未知", sections["limits"])
        self.assertIn("不是全谱证明", sections["inverse"])


if __name__ == "__main__":
    unittest.main()
