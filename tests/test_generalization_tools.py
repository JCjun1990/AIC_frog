import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


gen = module("generalization_eval_test", "generalization_eval.py")
bands = module("sphere_bandcount_test", "sphere_bandcount.py")


class GeneralizationToolTests(unittest.TestCase):
    def test_physics_features_follow_the_uniform_state_algebra(self):
        none = gen.physics_features(0.028, 0.062)                  # 4 (F + k)^2 > F: no nontrivial uniform state
        self.assertLess(none[2], 0)
        self.assertEqual(none[3], 0.0)
        F, k = 0.06, 0.057
        f = gen.physics_features(F, k)
        self.assertEqual(f[3], 1.0)
        v = F * (1 + np.sqrt(f[2])) / (2 * (F + k))
        u = (F + k) / v
        self.assertAlmostEqual(F * (1 - u), u * v * v, places=12)  # it is a uniform steady state
        self.assertAlmostEqual(f[4], k - v * v, places=12)          # trace
        self.assertAlmostEqual(f[5], (F + k) * (v * v - F), places=12)  # determinant

    def test_strict_evidence_drops_nonfinite_and_host(self):
        out = gen._finite({"host": "pc", "a": float("nan"), "b": [1.0, float("inf")], "c": {"host": "x", "d": 2.0}})
        self.assertEqual(out, {"a": None, "b": [1.0, None], "c": {"d": 2.0}})

    def test_band_candidates_bracket_the_wavenumber(self):
        self.assertEqual(bands.candidate_counts(12.9), [6, 7])
        self.assertEqual(bands.candidate_counts(14.0), [7])
        np.testing.assert_allclose(bands.axis(0), [0, 0, 1])
        self.assertAlmostEqual(float(np.linalg.norm(bands.axis(3))), 1.0)


if __name__ == "__main__":
    unittest.main()
