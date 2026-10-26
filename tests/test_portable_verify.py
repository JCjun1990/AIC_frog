import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from turing_rd import SPOTS, Geometry, icosphere
from turing_rd.result_io import save_certified_mat, validate_certificate

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("portable_verify_test", ROOT / "scripts" / "submission_workflow.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class PortableVerifyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "solution.mat"
        self.geom = Geometry(icosphere(1), laplacian="robust", cache_dir=None)
        u, v = np.ones(self.geom.n), np.zeros(self.geom.n)          # uniform steady state, residual 0
        self.cert = save_certified_mat(self.path, self.geom, SPOTS, u, v, seed=0)

    def tearDown(self):
        self.tmp.cleanup()

    def rounding_only(self):
        cert = copy.deepcopy(self.cert)
        cert["geometry"]["actual_L_M_sha256"] = "0" * 64           # what another CPU's last-bit rounding looks like
        with self.assertRaisesRegex(ValueError, "geometry mismatch") as ctx:
            validate_certificate(self.path, cert, expected_params=SPOTS, expected_seed=0, expected_geometry=self.geom)
        return cert, ctx.exception

    def test_operator_rounding_alone_is_accepted_and_labelled(self):
        cert, exc = self.rounding_only()
        fresh = workflow.portable_acceptance(self.path, cert, self.geom, exc)
        self.assertTrue(fresh["accepted"])
        self.assertIn("portable", fresh["verification_mode"])

    def test_mesh_mass_mat_and_residual_changes_still_fail(self):
        for field, value, message in (("mesh_sha256", "1" * 64, "mesh_sha256"), ("n_faces", 1, "n_faces"),
                                      ("mass_sum", 1.0, "mass_sum")):
            cert, exc = self.rounding_only()
            cert["geometry"][field] = value
            with self.assertRaisesRegex(ValueError, message):
                workflow.portable_acceptance(self.path, cert, self.geom, exc)
        cert, exc = self.rounding_only()
        cert["mat_sha256"] = "2" * 64
        with self.assertRaisesRegex(ValueError, "mat_sha256"):
            workflow.portable_acceptance(self.path, cert, self.geom, exc)
        cert, exc = self.rounding_only()
        cert["residual"]["value"] = 5e-12                              # recorded value no longer matches
        with self.assertRaisesRegex(ValueError, "portable read-back"):
            workflow.portable_acceptance(self.path, cert, self.geom, exc)

    def test_other_validation_errors_are_not_softened(self):
        cert = copy.deepcopy(self.cert)
        cert["seed"] = 1
        with self.assertRaisesRegex(ValueError, "seed") as ctx:
            validate_certificate(self.path, cert, expected_params=SPOTS, expected_seed=0, expected_geometry=self.geom)
        with self.assertRaisesRegex(ValueError, "seed"):
            workflow.portable_acceptance(self.path, cert, self.geom, ctx.exception)

    def test_geometry_fallback_does_not_bypass_other_certificate_checks(self):
        changes = {
            "formula": lambda c: c["residual"].update(formula="wrong"),
            "normalization": lambda c: c["residual"].update(normalization="different"),
            "projection": lambda c: c["residual"].update(rotation_projection=True),
            "tolerance": lambda c: c["residual"].update(tolerance=1.),
            "structure": lambda c: c["structure"].update(passed=False),
            "precision": lambda c: c["precision"].update(passed=False),
            "physical_range": lambda c: c["physical_range"].update(passed=False),
            "scope": lambda c: c.update(stability_status="strictly_stable"),
        }
        for name, change in changes.items():
            cert, exc = self.rounding_only()
            change(cert)
            with self.subTest(name=name), self.assertRaises(ValueError):
                workflow.portable_acceptance(self.path, cert, self.geom, exc)


if __name__ == "__main__":
    unittest.main()
