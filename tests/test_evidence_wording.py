"""Regression guard for the 2026-10-23 evidence-consistency fixes.

The audit (results/evidence_consistency_audit/20261023T231500) found that four
public documents described 128 PDE numerical endpoints as 128 complete steady
states, credited the sphere band search with "10/10 stable" without saying that
every selected candidate was a six-band state, and described the task-3 best
match as an "anchored" candidate when the selected one was a uniform CVT layout.

These assertions pin the corrected wording so that a later edit cannot silently
reintroduce a claim the evidence does not support.  They deliberately test
prose only; numerical results are covered by the MAT/manifest tests.
"""
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


class EvidenceWordingTest(unittest.TestCase):
    def test_generalization_uses_numerical_endpoint_wording(self):
        for relative in (
            "README.md",
            "HANDOVER.md",
            "submission/README_提交.md",
            "submission/evidence/generalization/summary.md",
        ):
            with self.subTest(document=relative):
                self.assertIn("数值终态", text(relative))
                self.assertNotIn("128 个 PDE 稳态", text(relative))

    def test_complete_steady_count_is_stated(self):
        summary = text("submission/evidence/generalization/summary.md")
        self.assertIn("73", summary)
        self.assertIn("57/64", summary)
        self.assertIn("16/64", summary)

    def test_band_selection_wording_names_six_bands(self):
        for relative in ("README.md", "HANDOVER.md", "submission/README_提交.md"):
            with self.subTest(document=relative):
                self.assertIn("6 条带", text(relative))

    def test_best_match_wording_names_cvt(self):
        self.assertIn("cvt_3", text("submission/README_提交.md"))


if __name__ == "__main__":
    unittest.main()
