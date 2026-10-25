"""Saved results must verify on another machine (another bit fingerprint of the assembled L/M, round-off in
every recomputed number) yet still reject real changes. Machine B is simulated by rewriting the recorded L/M
fingerprint, or by changing the numbers validate_evidence recomputes."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from turing_rd import Geometry, SPOTS, icosphere
from turing_rd import spectrum_evidence as S
from turing_rd.result_io import save_certified_mat, validate_certificate_portable
from turing_rd.spectral_policy import PAIR_ABSOLUTE_TOL, PAIR_RELATIVE_TOL
from turing_rd.stability import eigenpair_diagnostics


def dense_sample(J, mass, root, cfg):
    values, vectors = np.linalg.eig(J.toarray())
    order = np.argsort(-values.real)[:cfg["k"]]
    values = values[order].astype(np.complex128); vectors = vectors[:, order].astype(np.complex128)
    report = {"sigma": cfg["sigma"], "which": cfg["which"], "requested_k": cfg["k"], "returned_k": len(values),
              "solver_converged": True, "solver_error": None, "all_pairs_validated": True,
              "pairs": eigenpair_diagnostics(J, values, vectors, mass), "full_max_real": float(values.real.max()),
              "found_reliable_positive": False, "method": "tiny-grid-independent-dense-eigensolve",
              "pair_absolute_tolerance": PAIR_ABSOLUTE_TOL, "pair_relative_tolerance": PAIR_RELATIVE_TOL}
    return report, vectors


class SpectralEvidencePortabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = Geometry(icosphere(1), cache_dir=None)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        root = Path(tmp.name); self.mat = root / "state.mat"
        cert = save_certified_mat(self.mat, self.g, SPOTS, np.ones(self.g.n), np.zeros(self.g.n), seed=2)
        self.out = root / "evidence"
        S.save_evidence(self.mat, cert, self.g, SPOTS, self.out, dense_sample)

    def validate(self):
        return S.validate_evidence(self.out / "certificate.json", self.mat, self.g, SPOTS, 2)

    def rewrite(self, change):
        path = self.out / "certificate.json"; evidence = json.loads(path.read_text(encoding="utf-8"))
        change(evidence)
        path.write_text(json.dumps(evidence), encoding="utf-8")

    def with_recomputed_residual(self, value):
        real = S.validate_certificate_portable

        def fixed(*args, **kwargs):
            fresh = real(*args, **kwargs)
            fresh["residual"]["value"] = value
            return fresh
        return patch.object(S, "validate_certificate_portable", side_effect=fixed)

    def with_recomputed_pairs(self, change):
        real = S.eigenpair_diagnostics

        def changed(*args, **kwargs):
            rows = real(*args, **kwargs)
            for row in rows:
                change(row)
            return rows
        return patch.object(S, "eigenpair_diagnostics", side_effect=changed)

    def test_same_numbers_verify(self):
        self.assertTrue(self.validate()["eigenpair_residuals_recomputed_from_saved_vectors"])

    def test_read_back_residual_round_off_is_not_compared_bit_by_bit(self):
        with self.with_recomputed_residual(2.852356718394121e-14):       # ARM read-back of a 2.03e-14 record
            self.assertEqual(self.validate()["full_residual"], 2.852356718394121e-14)

    def test_tampered_saved_decision_is_rejected(self):
        self.rewrite(lambda e: e["decision"].update(full_residual=1e-20))
        with self.assertRaisesRegex(ValueError, "decision"):
            self.validate()

    def test_pair_residual_round_off_on_another_machine_is_accepted(self):
        def arm(row):                                                     # measured on ARM: 13% / 4e-16
            row["residual_absolute_M"] = row["residual_absolute_M"] * 1.13 + 4e-16
            row["residual_relative_M"] = row["residual_relative_M"] * 1.13 + 6e-13
        with self.with_recomputed_pairs(arm):
            self.validate()

    def test_pair_residual_not_matching_saved_vector_is_rejected(self):
        def off(row):                                                     # still passing, but not what was saved
            row["residual_absolute_M"] += 1e-11
        with self.with_recomputed_pairs(off), self.assertRaisesRegex(ValueError, "residual"):
            self.validate()

    def test_operator_rounding_on_another_machine_is_accepted(self):
        def fingerprint(e):
            for geometry in (e["mat_certificate"]["geometry"], e["input"]["geometry"]):
                geometry["actual_L_M_sha256"] = "0" * 64
        self.rewrite(fingerprint)
        self.validate()

    def test_evidence_bound_to_another_geometry_is_rejected(self):
        self.rewrite(lambda e: e["input"]["geometry"].update(actual_L_M_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "binding mismatch: geometry"):
            self.validate()

    def test_other_mesh_is_rejected(self):
        def mesh(e):
            for geometry in (e["mat_certificate"]["geometry"], e["input"]["geometry"]):
                geometry.update(actual_L_M_sha256="0" * 64, mesh_sha256="1" * 64)
        self.rewrite(mesh)
        with self.assertRaisesRegex(ValueError, "geometry mismatch"):
            self.validate()


class PortableCertificateTests(unittest.TestCase):
    """validate_certificate_portable softens exactly one check, the L/M bit fingerprint, and the residual
    agreement that depends on it."""
    @classmethod
    def setUpClass(cls):
        cls.g = Geometry(icosphere(1), cache_dir=None)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.mat = Path(tmp.name) / "state.mat"
        self.cert = save_certified_mat(self.mat, self.g, SPOTS, np.ones(self.g.n), np.zeros(self.g.n), seed=2)

    def validate(self, cert):
        return validate_certificate_portable(self.mat, cert, expected_params=SPOTS, expected_seed=2, expected_geometry=self.g)

    def changed(self, residual=None, **geometry):
        cert = copy.deepcopy(self.cert); cert["geometry"].update(geometry)
        if residual is not None:
            cert["residual"]["value"] = residual
        return cert

    def test_identical_machine_stays_strict(self):
        self.assertNotIn("verification_mode", self.validate(self.cert))
        with self.assertRaisesRegex(ValueError, "residual does not match"):
            self.validate(self.changed(residual=3e-14))

    def test_operator_rounding_alone_is_accepted_and_labelled(self):
        mass = self.cert["geometry"]["mass_sum"] * (1 + 2e-16)
        fresh = self.validate(self.changed(residual=3e-14, actual_L_M_sha256="0" * 64, mass_sum=mass))
        self.assertTrue(fresh["accepted"])
        self.assertIn("portable", fresh["verification_mode"])

    def test_mesh_counts_mass_and_operator_changes_still_fail(self):
        for field, value in (("mesh_sha256", "1" * 64), ("n_faces", 1), ("laplacian", "cotan"),
                             ("mass_sum", 1.0), ("mass_sum", "4.0"), ("extra", "field")):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "geometry mismatch"):
                self.validate(self.changed(actual_L_M_sha256="0" * 64, **{field: value}))

    def test_other_checks_are_not_softened(self):
        cert = self.changed(actual_L_M_sha256="0" * 64); cert["seed"] = 1
        with self.assertRaisesRegex(ValueError, "seed mismatch"):
            self.validate(cert)
        with self.assertRaisesRegex(ValueError, "residual does not match"):
            self.validate(self.changed(residual=5e-12, actual_L_M_sha256="0" * 64))


if __name__ == "__main__":
    unittest.main()
