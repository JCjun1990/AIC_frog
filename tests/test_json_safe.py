"""Regression tests for the I02 truncated-JSON defect (`scripts/json_safe.py`).

The first I02 perturbation runs completed their numerics and then died with
``TypeError: Object of type bool is not JSON serializable`` *after* ~37 kB had
already been streamed to ``perturbation.json``, leaving two truncated evidence
files.  These tests pin both halves of the fix: NumPy scalars are converted, and
nothing is written unless the whole document serialized.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("json_safe_under_test", ROOT / "scripts" / "json_safe.py")
json_safe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(json_safe)


class NumPyConversionTests(unittest.TestCase):
    def test_numpy_scalars_become_python_builtins(self):
        converted = json_safe.jsonable({
            "flag": np.bool_(True),
            "rate": np.float64(1.665427891625e-06),
            "count": np.int64(7),
            "tiny": np.float32(.5),
            "nested": [np.bool_(False), {"inner": np.array([1.0, 2.0])}],
        })
        self.assertIs(type(converted["flag"]), bool)
        self.assertIs(converted["flag"], True)
        self.assertIs(type(converted["rate"]), float)
        self.assertIs(type(converted["count"]), int)
        self.assertIs(type(converted["tiny"]), float)
        self.assertIs(type(converted["nested"][0]), bool)
        self.assertEqual(converted["nested"][1]["inner"], [1.0, 2.0])

    def test_the_exact_production_expression_yields_numpy_bool(self):
        # This is the original defect: max(1e-9, abs(lam.real) * .03) returns numpy.float64
        # because lam.real is a NumPy scalar, so the comparison is numpy.bool_.
        eigenvalue_real = np.float64(-6.231004404613e-06)
        rate_difference = 1.4e-9
        passed = rate_difference < max(1e-9, abs(eigenvalue_real) * .03)
        self.assertIsInstance(passed, np.bool_)
        self.assertIs(type(json_safe.jsonable(passed)), bool)

    def test_unknown_object_is_rejected_instead_of_stringified(self):
        with self.assertRaises(TypeError):
            json_safe.jsonable({"bad": object()})
        with self.assertRaises(TypeError):
            json_safe.jsonable({"bad": {1, 2}})
        with self.assertRaises(TypeError):
            json_safe.jsonable({2: "non-string key"})


class ExclusiveWriteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "report.json"

    def test_complete_document_is_written_and_parsed(self):
        text = json_safe.write_json_exclusive(self.path, {"passed": np.bool_(True),
                                                          "rate": np.float64(1e-6)})
        self.assertTrue(text.endswith("\n"))
        loaded = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertIs(loaded["passed"], True)
        self.assertEqual(loaded["rate"], 1e-6)

    def test_serialization_failure_creates_no_file(self):
        with self.assertRaises(TypeError):
            json_safe.write_json_exclusive(self.path, {"input": {}, "passed": object()})
        self.assertFalse(self.path.exists())

    def test_non_finite_numbers_create_no_file(self):
        with self.assertRaises(ValueError):
            json_safe.write_json_exclusive(self.path, {"value": float("nan")})
        self.assertFalse(self.path.exists())

    def test_existing_file_is_never_replaced(self):
        json_safe.write_json_exclusive(self.path, {"first": 1})
        before = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            json_safe.write_json_exclusive(self.path, {"second": 2})
        self.assertEqual(before, self.path.read_bytes())

    def test_realistic_perturbation_document_roundtrips(self):
        document = {
            "eigenpair": {"eigenvalue": [1.665427891625e-06, 0.0],
                          "residual_absolute_M": np.float64(1.17e-15),
                          "residual_passed": np.bool_(True),
                          "rotation_like": np.bool_(True)},
            "resolutions": [{"dt": 2000.0, "steps": 300,
                             "nonlinear_fitted_rates": [np.float64(1.668e-06)] * 4,
                             "supports_eigenvalue_rate": np.bool_(True)}],
            "trajectories": [[{"time": 0.0,
                               "nonlinear_norm": [np.float64(1e-6), -1e-6, 1e-5, -1e-5]}],
                             [{"time": 0.0, "nonlinear_norm": [1e-6, -1e-6, 1e-5, -1e-5]}]],
            "passed": np.bool_(True),
        }
        json_safe.write_json_exclusive(self.path, document)
        loaded = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertIs(loaded["passed"], True)
        self.assertIs(loaded["eigenpair"]["residual_passed"], True)
        self.assertEqual(loaded["trajectories"][0][0]["nonlinear_norm"], [1e-6, -1e-6, 1e-5, -1e-5])


if __name__ == "__main__":
    unittest.main()
