"""I02 regression tests: actual PDE Jacobian, mass norms, and fail-closed spectra.

Only the 42-vertex robust icosphere and small synthetic matrices are used.
No artifact recovery, training, or time evolution is launched by this suite.
"""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from turing_rd import Geometry, SPOTS, icosphere
from turing_rd.diagnostics import jacobian, linear_stability
from turing_rd.steady import weak_jacobian, weak_residual
from turing_rd.spectral_policy import SpectrumConfig
from turing_rd.stability import (aggregate_spectra, eigenpair_diagnostics,
                                 rotation_subspace, solve_spectrum)


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ai_generation_i02_test", ROOT / "scripts" / "ai_generation.py")
ai = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ai)
spec_i02 = importlib.util.spec_from_file_location("i02_stability_script_test", ROOT / "scripts" / "i02_stability.py")
i02 = importlib.util.module_from_spec(spec_i02)
spec_i02.loader.exec_module(i02)


class DiscreteJacobianTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geom = Geometry(icosphere(1), laplacian="robust", cache_dir=None)
        cls.mass = np.tile(cls.geom.M, 2)
        cls.u = .65 + .08 * cls.geom.V[:, 0]
        cls.v = .15 + .03 * cls.geom.V[:, 2]

    def test_strong_jacobian_is_actual_mass_inverse_times_weak_jacobian(self):
        J = jacobian(self.geom, SPOTS, self.u, self.v)
        Jw = weak_jacobian(self.geom, SPOTS, self.u, self.v)
        difference = sp.diags(self.mass) @ J - Jw
        self.assertLess(float(np.max(np.abs(difference.toarray()))), 1e-15)

    def test_jacobian_action_matches_independent_centered_rhs_difference(self):
        rng = np.random.default_rng(1002)
        d = rng.standard_normal(2 * self.geom.n)
        d /= np.linalg.norm(d)
        state = np.concatenate([self.u, self.v])
        h = 1e-6

        def rhs(x):
            return weak_residual(self.geom, SPOTS, x[:self.geom.n], x[self.geom.n:]) / self.mass

        fd = (rhs(state + h * d) - rhs(state - h * d)) / (2 * h)
        exact = jacobian(self.geom, SPOTS, self.u, self.v) @ d
        self.assertLess(np.linalg.norm(fd - exact) / np.linalg.norm(exact), 2e-8)

    def test_low_resolution_complete_dense_and_generalized_mass_spectra_agree(self):
        J = jacobian(self.geom, SPOTS, self.u, self.v).toarray()
        Jw = weak_jacobian(self.geom, SPOTS, self.u, self.v).toarray()
        strong = la.eigvals(J)
        generalized = la.eigvals(Jw, np.diag(self.mass))
        # Match eigenvalues without relying on ordering of repeated/complex roots.
        distance = np.abs(strong[:, None] - generalized[None, :])
        self.assertLess(float(max(distance.min(0).max(), distance.min(1).max())), 1e-12)


class EigenpairEvidenceTests(unittest.TestCase):
    def test_absolute_and_relative_residual_use_the_mass_norm(self):
        J = sp.diags([-.2, .1], format="csc")
        w = np.array([[1.0], [2.0]])
        mass = np.array([9.0, .25])
        lam = np.array([.09 + 0j])
        report = eigenpair_diagnostics(J, lam, w, mass)[0]
        r = J @ w[:, 0] - lam[0] * w[:, 0]
        norm = lambda x: np.sqrt(np.sum(mass * np.abs(x) ** 2))
        expected_absolute = norm(r) / norm(w[:, 0])
        expected_relative = norm(r) / (norm(J @ w[:, 0]) + abs(lam[0]) * norm(w[:, 0]))
        self.assertAlmostEqual(report["residual_absolute_M"], expected_absolute, places=14)
        self.assertAlmostEqual(report["residual_relative_M"], expected_relative, places=14)
        self.assertFalse(report["residual_passed"])
        self.assertFalse(report["positive_reliable"])

    def test_exact_positive_rotation_mode_is_still_reliable_positive(self):
        J = sp.diags([1.665e-6, -.01], format="csc")
        mass = np.array([4.0, 2.0])
        # The supplied Q is an orthonormal basis in sqrt(M)-scaled coordinates.
        Q = np.array([[1.0], [0.0]])
        w = np.array([[.5], [0.0]])
        report = eigenpair_diagnostics(J, np.array([1.665e-6 + 0j]), w, mass,
                                       rotation_basis=Q)[0]
        self.assertTrue(report["residual_passed"])
        self.assertAlmostEqual(report["rotation_overlap"], 1.0)
        self.assertTrue(report["positive_reliable"])

    def test_complex_eigenpair_residual_uses_absolute_square(self):
        J = sp.csc_matrix([[0., -1.], [1., 0.]])
        w = np.array([[1.], [-1j]])
        pair = eigenpair_diagnostics(J, np.array([1j]), w, np.array([2., 3.]))[0]
        self.assertLess(pair["residual_absolute_M"], 1e-14)
        self.assertTrue(pair["residual_passed"])
        self.assertFalse(pair["positive_reliable"])


class NumericalSpectrumPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geom = Geometry(icosphere(1), laplacian="robust", cache_dir=None)
        cls.u = np.ones(cls.geom.n)
        cls.v = np.zeros(cls.geom.n)
        cls.configs = tuple(SpectrumConfig(s, w, 6) for s, w in [(1e-3, "LM"), (0., "LM"), (None, "LR")])
        cls.reports = []
        cls.vectors = []
        for sigma, which in [(1e-3, "LM"), (0.0, "LM"), (None, "LR")]:
            report, W = solve_spectrum(cls.geom, SPOTS, cls.u, cls.v, k=6,
                                       sigma=sigma, which=which, tol=1e-11,
                                       maxiter=2000, ncv=24, seed=2026)
            cls.reports.append(report)
            cls.vectors.append(W)

    def test_sparse_key_eigenpairs_match_complete_dense_spectrum(self):
        dense = la.eigvals(jacobian(self.geom, SPOTS, self.u, self.v).toarray())
        for report in self.reports:
            with self.subTest(method=report["method"]):
                self.assertTrue(report["solver_converged"])
                self.assertTrue(report["all_pairs_validated"])
                values = np.array([complex(*pair["eigenvalue"]) for pair in report["pairs"]])
                self.assertEqual(len(values), 6)
                self.assertLess(float(np.abs(values[:, None] - dense[None, :]).min(1).max()), 1e-10)
                self.assertAlmostEqual(report["full_max_real"], float(dense.real.max()), places=10)

    def test_uniform_field_has_no_active_rotation_directions(self):
        Q, metadata = rotation_subspace(self.geom, self.u, self.v)
        self.assertEqual(Q.shape, (2 * self.geom.n, 0))

    def test_two_shifts_and_independent_rightmost_negative_check_can_pass(self):
        result = aggregate_spectra(self.reports, full_residual=1e-15, configs=self.configs)
        self.assertTrue(result["numerical_check_passed"])

    def test_single_shift_is_insufficient_for_stability_acceptance(self):
        result = aggregate_spectra(self.reports[:1], full_residual=1e-15, configs=self.configs)
        self.assertFalse(result["numerical_check_passed"])

    def test_relative_equilibrium_cannot_receive_full_steady_stability_acceptance(self):
        result = aggregate_spectra(self.reports, full_residual=1e-6, configs=self.configs)
        self.assertFalse(result["numerical_check_passed"])

    def test_failed_or_partial_solver_cannot_receive_stability_acceptance(self):
        reports = copy.deepcopy(self.reports)
        reports[2]["solver_converged"] = False
        result = aggregate_spectra(reports, full_residual=1e-15, configs=self.configs)
        self.assertFalse(result["numerical_check_passed"])

    def test_confirmed_positive_cannot_be_hidden_by_nonrotation_maximum(self):
        reports = copy.deepcopy(self.reports)
        for report in reports:
            pair = report["pairs"][0]
            pair.update(eigenvalue=[1.665e-6, 0.0], residual_absolute_M=1e-15,
                        residual_relative_M=1e-10, residual_passed=True,
                        positive_reliable=True, rotation_overlap=1.0)
            report["full_max_real"] = 1.665e-6
            report["nonrotation_max_real"] = -4.61e-5
            report["found_reliable_positive"] = True
        result = aggregate_spectra(reports, full_residual=1e-15, configs=self.configs)
        self.assertFalse(result["numerical_check_passed"])

    def test_arpack_partial_convergence_is_recorded_and_fails_closed(self):
        known = self.reports[0]["pairs"][0]["eigenvalue"]
        lam = np.array([complex(*known)])
        W = self.vectors[0][:, :1]
        error = spla.ArpackNoConvergence("deliberate partial convergence", lam, W)
        with patch("turing_rd.stability.spla.eigs", side_effect=error):
            report, vectors = solve_spectrum(self.geom, SPOTS, self.u, self.v, k=6,
                                              sigma=1e-3, maxiter=2)
        self.assertFalse(report["solver_converged"])
        self.assertEqual(report["returned_k"], 1)
        self.assertEqual(vectors.shape[1], 1)
        self.assertFalse(aggregate_spectra([report], full_residual=1e-15)["numerical_check_passed"])


class LegacyEntryPointSafetyTests(unittest.TestCase):
    """Existing callers must not silently reintroduce rotation filtering/tolerance."""
    @classmethod
    def setUpClass(cls):
        cls.geom = Geometry(icosphere(1), laplacian="robust", cache_dir=None)
        cls.u = np.ones(cls.geom.n)
        cls.v = np.zeros(cls.geom.n)

    def test_linear_stability_maximum_includes_positive_rotation_mode(self):
        positive = {"eigenvalue": [5e-7, 0.0], "rotation_like": True,
                    "rotation_overlap": 1.0, "negative_resolved": False,
                    "positive_reliable": True, "residual_passed": True}
        negative = {"eigenvalue": [-4.61e-5, 0.0], "rotation_like": False,
                    "rotation_overlap": 0.0, "negative_resolved": True,
                    "positive_reliable": False, "residual_passed": True}
        report = {"pairs": [positive, negative], "full_max_real": 5e-7,
                  "solver_converged": True, "all_pairs_validated": True,
                  "found_reliable_positive": True}
        with patch("turing_rd.stability.rotation_subspace", return_value=(
                np.zeros((2*self.geom.n, 1)), {"rank": 1})), \
                patch("turing_rd.stability.solve_spectrum", return_value=(
                    report, np.zeros((2*self.geom.n, 2)))), \
                patch("turing_rd.stability.eigenpair_diagnostics", return_value=[positive, negative]):
            result = linear_stability(self.geom, SPOTS, self.u, self.v, rotations=[object()])
        self.assertEqual(result["max_real"], 5e-7)
        self.assertEqual(result["max_real_all"], 5e-7)
        self.assertEqual(result["nonrotation_max_real"], -4.61e-5)
        self.assertIn([5e-7, 0.0], result["rotation_eigenvalues"])
        self.assertFalse(result["sampled_signs_resolved_negative"])
        self.assertFalse(result["full_discrete_stability_certified"])

    def test_analyse_does_not_accept_a_positive_value_below_old_1e_minus_6_threshold(self):
        info = {"converged": True, "relax_time": 0., "relax_steps": 0,
                "newton": [], "wall_s": 0.}
        sample = {"max_real": 5e-7, "nonrotation_max_real": 5e-7,
                  "sampled_signs_resolved_negative": False}
        with patch.object(ai, "linear_stability", return_value=sample):
            result = ai.analyse(self.geom, SPOTS, self.u, self.v, info)
        self.assertTrue(result["converged"])
        self.assertEqual(result["max_re_lambda"], 5e-7)
        self.assertFalse(result["stable"])
        self.assertEqual(result["full_discrete_stability"], "not_certified_by_single_shift")

    def test_analyse_relative_equilibrium_flag_does_not_override_full_pde_residual(self):
        info = {"converged": True, "relax_time": 0., "relax_steps": 0,
                "newton": [], "wall_s": 0., "rel_residual_rot": 1e-15,
                "omega": [1e-5, 0., 0.], "steady": False}
        with patch.object(ai, "linear_stability") as spectrum:
            result = ai.analyse(self.geom, SPOTS, .8*self.u, .1*self.u, info, rot=[])
        self.assertGreater(result["rel_residual"], 1e-11)
        self.assertTrue(result["relative_equilibrium_converged"])
        self.assertFalse(result["converged"])
        self.assertFalse(result["steady"])
        self.assertFalse(result["stable"])
        spectrum.assert_not_called()


class PerturbationReportSerializationTests(unittest.TestCase):
    """The real perturb() report writer must produce a complete, parseable document.

    Both historical perturbation runs died here: `passed` was a numpy.bool_ and the
    old writer had already streamed ~37 kB into perturbation.json before raising, so
    the file ended in `"passed": `. The truncated files are preserved as evidence.
    """
    eigenvalue = 1.665427891625e-06

    def resolution(self, dt, rate, supports, ranges_ok=True):
        return {"dt": dt, "steps": steps_for(dt), "wall_seconds": 61.0,
                "linear_fitted_rate": rate, "linear_BE_expected_rate": rate,
                "continuous_eigenvalue_real": self.eigenvalue,
                "nonlinear_fitted_rates": [rate] * 4,
                "nonlinear_growth_factors": [1.1, 1.1, 1.1, 1.1],
                "linear_growth_factor": 1.1,
                "final_ranges": {"u_min": .70, "u_max": .79, "v_min": .01, "v_max": .21},
                "final_pattern_statistics": [{"type": "stripes_or_maze"}],
                "supports_eigenvalue_rate": supports, "physical_ranges_valid": ranges_ok}

    def report(self, rate_diff=1.39e-9, supports=True):
        return i02.perturbation_report(
            {"mat_sha256": "a" * 64}, "results/i02_stability/six_bands/shift_1e3_k12",
            {"eigenvalue": [self.eigenvalue, 0.0], "residual_passed": True},
            [1e-6, -1e-6, 1e-5, -1e-5, 0.0], 600000.0,
            [self.resolution(2000.0, self.eigenvalue, supports),
             self.resolution(1000.0, self.eigenvalue, supports)],
            [[], []], rate_diff, np.float64(self.eigenvalue), "b" * 64)

    def test_production_report_is_plain_json_and_passed_is_a_python_bool(self):
        report = self.report()
        self.assertIs(type(report["passed"]), bool)
        self.assertTrue(report["passed"])
        # A plain json.dumps must succeed: no numpy.bool_ may survive in the document.
        json.dumps(report, allow_nan=False)

    def test_production_report_can_be_written_and_read_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "perturbation.json"
            i02.write_json(path, self.report())
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertIs(loaded["passed"], True)
        self.assertEqual(loaded["half_dt_rate_difference"], 1.39e-9)
        self.assertEqual(loaded["perturbations_sha256"], "b" * 64)

    def test_failed_checks_stay_false_and_still_serialize(self):
        self.assertIs(self.report(supports=False)["passed"], False)
        self.assertIs(self.report(rate_diff=1e-6)["passed"], False)

    def test_unserializable_payload_leaves_no_truncated_file(self):
        report = self.report()
        report["eigenpair"] = object()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "perturbation.json"
            with self.assertRaises(TypeError):
                i02.write_json(path, report)
            self.assertFalse(path.exists())


def steps_for(dt):
    return int(round(600000.0 / dt))


if __name__ == "__main__":
    unittest.main()
