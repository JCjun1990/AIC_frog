"""I01 persistence and rejection tests; no training or time integration."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

from turing_rd import Geometry, SPOTS, STRIPES, icosphere
from turing_rd.diagnostics import full_steady_residual, pde_residual
from turing_rd.result_io import (geometry_identity, read_mat_fields, save_certified_mat,
                                 validate_certificate, verify_mat)
from turing_rd.steady import rel_residual, weak_residual


class FormalMatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geom = Geometry(icosphere(1), laplacian="robust", cache_dir=None)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "solution.mat"
        self.u = np.ones(self.geom.n, dtype=np.float64)
        self.v = np.zeros(self.geom.n, dtype=np.float64)

    def accepted(self):
        return save_certified_mat(self.path, self.geom, SPOTS, self.u, self.v, seed=0)

    def validate(self, cert, **overrides):
        expected = dict(expected_params=SPOTS, expected_seed=0, expected_geometry=self.geom)
        expected.update(overrides)
        return validate_certificate(self.path, cert, **expected)

    def test_known_equilibrium_roundtrip_and_binding(self):
        cert = self.accepted()
        self.assertTrue(cert["accepted"])
        self.assertTrue(cert["precision"]["source_values_preserved"])
        self.assertEqual(cert["stability_status"], "not_evaluated_by_i01")
        self.assertLess(cert["residual"]["value"], 1e-14)
        self.assertTrue(self.validate(cert)["accepted"])
        u, v = read_mat_fields(self.path, self.geom.n)
        np.testing.assert_array_equal(u, self.u)
        np.testing.assert_array_equal(v, self.v)

    def test_non_float32_details_are_preserved_even_if_residual_fails(self):
        x = np.arange(self.geom.n)
        u, v = .7 + 1e-10 * np.sin(x), .1 + 1e-10 * np.cos(x)
        cert = save_certified_mat(self.path, self.geom, SPOTS, u, v, seed=0)
        saved = read_mat_fields(self.path, self.geom.n)
        np.testing.assert_array_equal(saved[0], u)
        np.testing.assert_array_equal(saved[1], v)
        self.assertFalse(np.array_equal(u, u.astype(np.float32).astype(np.float64)))
        self.assertTrue(cert["precision"]["passed"])
        self.assertFalse(cert["residual"]["passed"])
        self.assertFalse(cert["accepted"])

    def test_float32_input_is_rejected_without_creating_mat(self):
        with self.assertRaises(TypeError):
            save_certified_mat(self.path, self.geom, SPOTS, self.u.astype(np.float32), self.v, seed=0)
        self.assertFalse(self.path.exists())

    def test_upcast_quantized_nonuniform_field_is_not_recovered(self):
        u = np.linspace(.5, .8, self.geom.n).astype(np.float32).astype(np.float64)
        v = np.linspace(.1, .2, self.geom.n, dtype=np.float64)
        cert = save_certified_mat(self.path, self.geom, SPOTS, u, v, seed=0)
        self.assertFalse(cert["precision"]["passed"])
        self.assertIn("float32_quantized_nonuniform_field", cert["failure_reasons"])

    def test_existing_file_is_never_overwritten(self):
        self.accepted()
        before = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.accepted()
        self.assertEqual(before, self.path.read_bytes())

    def test_wrong_layout_is_rejected(self):
        sio.savemat(self.path, {"Solution": {"u": self.u, "v": self.v}})
        cert = verify_mat(self.path, self.geom, SPOTS, seed=0)
        self.assertFalse(cert["accepted"])
        self.assertIn("invalid_mat_structure", cert["failure_reasons"])

    def test_wrong_node_count_is_rejected(self):
        sio.savemat(self.path, {"Solution": {"u": self.u[:-1, None], "v": self.v[:-1, None]}})
        self.assertFalse(verify_mat(self.path, self.geom, SPOTS, seed=0)["accepted"])

    def test_mat_single_precision_is_rejected(self):
        sio.savemat(self.path, {"Solution": {"u": self.u[:, None].astype(np.float32),
                                            "v": self.v[:, None].astype(np.float32)}})
        cert = verify_mat(self.path, self.geom, SPOTS, seed=0)
        self.assertIn("not_float64", cert["failure_reasons"])

    def test_nan_and_physical_range_are_rejected(self):
        for value, reason in [(np.nan, "nonfinite_values"), (1.01, "outside_physical_range")]:
            with self.subTest(value=value):
                u = self.u.copy(); u[0] = value
                sio.savemat(self.path, {"Solution": {"u": u[:, None], "v": self.v[:, None]}})
                cert = verify_mat(self.path, self.geom, SPOTS, seed=0)
                self.assertFalse(cert["accepted"])
                self.assertIn(reason, cert["failure_reasons"])

    def test_sha_tampering_is_rejected(self):
        cert = self.accepted()
        with self.path.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "mat_sha256"):
            self.validate(cert)

    def test_parameters_seed_and_operator_must_match(self):
        cert = self.accepted()
        with self.assertRaisesRegex(ValueError, "parameters"):
            self.validate(cert, expected_params=STRIPES)
        with self.assertRaisesRegex(ValueError, "seed"):
            self.validate(cert, expected_seed=1)
        cotan = Geometry(icosphere(1), laplacian="cotan", cache_dir=None)
        with self.assertRaisesRegex(ValueError, "geometry"):
            self.validate(cert, expected_geometry=cotan)
        changed = copy.copy(self.geom)
        changed.L = changed.L + sp.eye(changed.n) * 1e-8
        with self.assertRaisesRegex(ValueError, "geometry"):
            self.validate(cert, expected_geometry=changed)

    def test_forged_acceptance_does_not_bypass_actual_pde(self):
        u, v = self.u * .7, self.v + .1
        cert = save_certified_mat(self.path, self.geom, SPOTS, u, v, seed=0)
        cert["accepted"] = True
        for key in ("structure", "precision", "physical_range", "residual"):
            cert[key]["passed"] = True
        cert["residual"]["value"] = 0
        with self.assertRaisesRegex(ValueError, "read-back acceptance failed"):
            self.validate(cert)

    def test_diagnostic_definition_and_stability_scope_are_enforced(self):
        cert = self.accepted()
        for field, value in [("rotation_projection", True), ("formula", "cotan crosscheck")]:
            bad = copy.deepcopy(cert); bad["residual"][field] = value
            with self.assertRaises(ValueError):
                self.validate(bad)
        bad = copy.deepcopy(cert); bad["stability_status"] = "stable"
        with self.assertRaisesRegex(ValueError, "cannot certify linear stability"):
            self.validate(bad)
        bad = copy.deepcopy(cert); bad["precision"]["passed"] = False
        with self.assertRaises(ValueError):
            self.validate(bad)
        bad = copy.deepcopy(cert); bad["structure"]["u_dtype"] = "float32"
        with self.assertRaisesRegex(ValueError, "structure.u_dtype"):
            self.validate(bad)

    def test_residual_matches_weak_equations_and_all_public_entries(self):
        u = .7 + .02 * np.sin(np.arange(self.geom.n))
        v = .1 + .01 * np.cos(np.arange(self.geom.n))
        weak = weak_residual(self.geom, SPOTS, u, v)
        mass = np.tile(self.geom.M, 2)
        expected = np.sqrt(np.dot(mass, (weak / mass) ** 2) / mass.sum()) / SPOTS.F
        current = full_steady_residual(self.geom, SPOTS, u, v)["full_joint_rms_over_F"]
        self.assertAlmostEqual(expected, current, places=14)
        self.assertEqual(rel_residual(self.geom, SPOTS, u, v), current)
        diag = pde_residual(self.geom, SPOTS, u, v, u_prev=u + .01, v_prev=v + .01, dt=1)
        self.assertEqual(diag["full_joint_rms_over_F"], current)


if __name__ == "__main__":
    unittest.main()
