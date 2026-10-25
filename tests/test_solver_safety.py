"""Nonuniform tiny-grid regressions for checkpoint recomputation and strict CG."""
import json
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

from turing_rd import Geometry, icosphere
from turing_rd.solver import CGConvergenceError, GrayScottSolver, SPDOperator


def initial_fields(g, columns=None):
    u, v = .7 + .07 * g.V[:, 0], .12 + .02 * g.V[:, 2]
    if columns is not None:
        u = np.column_stack([u * (1 - .01 * j) for j in range(columns)])
        v = np.column_stack([v * (1 + .02 * j) for j in range(columns)])
    return u, v


def objective(g, u, v):
    weights = torch.as_tensor(g.M * (1 + .2 * g.V[:, 1]), dtype=u.dtype)
    if u.ndim == 2:
        weights = weights[:, None] / u.shape[1]
    return (weights * (u.square() + 3 * v.square() + .1 * u * v)).sum() / g.area


def trajectory(g, *, backend="lu", interval=0, columns=None, dtype=torch.float64,
               scheme="sbdf2", steps=7, record_every=0, callback=None):
    u, v = (torch.tensor(x, dtype=dtype, requires_grad=True) for x in initial_fields(g, columns))
    P = {k: torch.tensor(x, dtype=dtype, requires_grad=True)
         for k, x in dict(F=.028, k=.062, Du=.02, Dv=.01).items()}
    solver = GrayScottSolver(g, dt=.3, scheme=scheme, backend=backend, dtype=dtype,
                            cg_rtol=1e-13 if dtype == torch.float64 else 1e-6)
    U, V = solver.rollout(u, v, P, steps, checkpoint_every=interval,
                          record_every=record_every, callback=callback)
    loss = objective(g, U, V)
    gradients = torch.autograd.grad(loss, (u, v, *P.values()))
    return U.detach(), V.detach(), tuple(x.detach() for x in gradients)


def finite_difference_check(g, backend, interval, columns=None):
    """Double precision central differences, using the pre-existing gradcheck tolerances.

    h=1e-6 balances O(h^2) truncation with O(eps/h) cancellation for this
    short, smooth rollout. CG uses 1e-13, below the finite-difference error scale.
    """
    inputs = [*initial_fields(g, columns), .028, .062, .02, .01]
    tensors = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in inputs]
    solver = GrayScottSolver(g, dt=.3, backend=backend, cg_rtol=1e-13)

    def loss(values):
        u, v, F, k, Du, Dv = values
        U, V = solver.rollout(u, v, dict(F=F, k=k, Du=Du, Dv=Dv), 7, checkpoint_every=interval)
        return objective(g, U, V)

    gradients = torch.autograd.grad(loss(tensors), tensors)
    rng, records = np.random.default_rng(931), []
    for index, name in enumerate(("initial_u_direction", "initial_v_direction", "F", "k", "Du", "Dv")):
        direction = torch.ones_like(tensors[index])
        if index < 2:
            direction = torch.as_tensor(rng.normal(size=tensors[index].shape))
            direction /= torch.linalg.vector_norm(direction)
        h = 1e-6
        plus, minus = [x.detach().clone() for x in tensors], [x.detach().clone() for x in tensors]
        plus[index] += h * direction; minus[index] -= h * direction
        with torch.no_grad():
            fd = float((loss(plus) - loss(minus)) / (2 * h))
        ad = float((gradients[index] * direction).sum())
        records.append({"variable": name, "autograd": ad, "finite_difference": fd,
                        "absolute_error": abs(ad - fd),
                        "relative_error": abs(ad - fd) / max(abs(ad), abs(fd), 1e-30)})
    return records


class CheckpointSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = Geometry(icosphere(1), cache_dir=None)

    def compare_intervals(self, **kwargs):
        reference = trajectory(self.g, interval=0, **kwargs)
        eps = torch.finfo(kwargs.get("dtype", torch.float64)).eps
        for interval in (1, 3, 5):
            with self.subTest(interval=interval, settings=kwargs):
                actual = trajectory(self.g, interval=interval, **kwargs)
                # Forward operations are unchanged, so demand bit-identical fields.
                self.assertTrue(torch.equal(actual[0], reference[0]))
                self.assertTrue(torch.equal(actual[1], reference[1]))
                for a, b in zip(actual[2], reference[2]):
                    torch.testing.assert_close(a, b, rtol=16 * eps, atol=16 * eps)

    def test_seven_steps_vector_lu(self):
        self.compare_intervals()

    def test_seven_steps_single_column_lu(self):
        self.compare_intervals(columns=1)

    def test_seven_steps_three_columns_lu(self):
        self.compare_intervals(columns=3)

    def test_seven_steps_vector_cg(self):
        self.compare_intervals(backend="cg")

    def test_seven_steps_two_columns_cg(self):
        self.compare_intervals(backend="cg", columns=2)

    def test_sbdf1_vector_and_batch(self):
        for backend in ("lu", "cg"):
            for columns in (None, 2):
                self.compare_intervals(backend=backend, columns=columns, scheme="sbdf1")

    def test_float32_lu_and_cg(self):
        for backend in ("lu", "cg"):
            self.compare_intervals(backend=backend, columns=2, dtype=torch.float32)

    def test_single_step(self):
        self.compare_intervals(steps=1)

    def test_divisible_and_nondivisible_longer_rollouts(self):
        for steps in (8, 10, 11):
            self.compare_intervals(steps=steps)

    def test_callback_and_checkpoint_boundaries_are_independent(self):
        sequences = []
        for interval in (0, 1, 3, 5):
            visited = []
            result = trajectory(self.g, interval=interval, record_every=4,
                                callback=lambda step, *args: visited.append(step) or False)
            sequences.append((visited, result))
        for visited, result in sequences:
            self.assertEqual(visited, [1, 4])
            self.assertTrue(torch.equal(result[0], sequences[0][1][0]))
            for a, b in zip(result[2], sequences[0][1][2]):
                torch.testing.assert_close(a, b, rtol=16 * np.finfo(float).eps, atol=16 * np.finfo(float).eps)

    def test_callback_early_stop_has_correct_backward(self):
        first = trajectory(self.g, record_every=4, callback=lambda step, *args: step == 4)
        for interval in (1, 3, 5):
            actual = trajectory(self.g, interval=interval, record_every=4, callback=lambda step, *args: step == 4)
            self.assertTrue(torch.equal(actual[0], first[0]))
            for a, b in zip(actual[2], first[2]):
                torch.testing.assert_close(a, b, rtol=16 * np.finfo(float).eps, atol=16 * np.finfo(float).eps)

    def test_no_grad_does_not_invoke_checkpoint(self):
        import turing_rd.solver as module
        solver = GrayScottSolver(self.g, dt=.3)
        with torch.no_grad(), patch.object(module, "checkpoint", side_effect=AssertionError("no-grad path must not checkpoint")):
            reference = solver.rollout(*initial_fields(self.g), module.SPOTS, 7)
            actual = solver.rollout(*initial_fields(self.g), module.SPOTS, 7, checkpoint_every=3)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, reference)))

    def test_finite_difference_vector_lu(self):
        self.check_finite_differences("lu", None)

    def test_finite_difference_batch_lu(self):
        self.check_finite_differences("lu", 2)

    def test_finite_difference_vector_cg(self):
        self.check_finite_differences("cg", None)

    def test_finite_difference_batch_cg(self):
        self.check_finite_differences("cg", 2)

    def check_finite_differences(self, backend, columns):
        for interval in (0, 3, 5):
            for row in finite_difference_check(self.g, backend, interval, columns):
                with self.subTest(backend=backend, columns=columns, interval=interval, variable=row["variable"]):
                    self.assertLessEqual(row["absolute_error"], 1e-8 + 1e-5 * abs(row["finite_difference"]))


class CGSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = Geometry(icosphere(1), cache_dir=None)
        cls.b = torch.tensor(np.random.default_rng(0).normal(size=cls.g.n))
        cls.alpha = torch.tensor(.1, dtype=torch.float64)

    def op(self, **kwargs):
        return SPDOperator(self.g, backend="cg", **kwargs)

    def test_converged_vector_matches_independent_dense_solve(self):
        op = self.op(cg_rtol=1e-14)
        x = op.solve_raw(1., self.alpha, self.b)
        matrix = np.diag(self.g.M) - float(self.alpha) * self.g.L.toarray()
        reference = np.linalg.solve(matrix, self.b.numpy())
        np.testing.assert_allclose(x.numpy(), reference, atol=2e-13, rtol=2e-13)
        residual = np.linalg.norm(self.b.numpy() - matrix @ x.numpy()) / np.linalg.norm(self.b.numpy())
        self.assertLessEqual(residual, op.cg_rtol)
        self.assertTrue(op.last_cg_info["converged"])
        self.assertTrue(op.last_cg_info["true_residual_verified"])

    def test_batched_different_alpha_matches_independent_solves(self):
        B = torch.stack((self.b, self.b.flip(0), self.b.square()), dim=1)
        alphas = torch.tensor([.01, .1, .3], dtype=torch.float64)
        op = self.op(cg_rtol=1e-13); actual = op.solve_raw(1.5, alphas, B)
        for j in range(3):
            matrix = 1.5 * np.diag(self.g.M) - float(alphas[j]) * self.g.L.toarray()
            np.testing.assert_allclose(actual[:, j].numpy(), np.linalg.solve(matrix, B[:, j].numpy()), atol=2e-12, rtol=2e-12)
            self.assertLessEqual(np.linalg.norm(B[:, j].numpy() - matrix @ actual[:, j].numpy()) / np.linalg.norm(B[:, j].numpy()), op.cg_rtol)
        self.assertEqual(op.last_cg_info["converged_per_rhs"], [True] * 3)

    def test_float32_success_preserves_dtype_and_requested_tolerance(self):
        op = self.op(dtype=torch.float32, cg_rtol=1e-6)
        b, a = self.b.float(), self.alpha.float(); x = op.solve_raw(1., a, b)
        actual = torch.linalg.vector_norm(b - (op.M * x - a * op.Lmv(x))) / torch.linalg.vector_norm(b)
        self.assertEqual(x.dtype, torch.float32)
        self.assertLessEqual(float(actual), 1e-6)
        self.assertEqual(op.last_cg_info["rtol"], 1e-6)

    def test_one_iteration_refuses_unconverged_result(self):
        op = self.op(cg_rtol=1e-14, cg_maxiter=1)
        with self.assertRaises(CGConvergenceError) as caught:
            op.solve_raw(1., self.alpha, self.b)
        info = caught.exception.info
        self.assertFalse(info["converged"]); self.assertEqual(info["status"], "max_iterations")
        self.assertEqual(info["iterations"], 1); self.assertEqual(op.cg_iterations, 1)
        self.assertGreater(info["relative_residual"][0], 1e-2)
        self.assertEqual(info["rtol"], 1e-14); self.assertIsNone(info["fallback"])

    def test_zero_iteration_budget_refuses_non_solution(self):
        op = self.op(cg_maxiter=0)
        with self.assertRaises(CGConvergenceError) as caught:
            op.solve_raw(1., self.alpha, self.b)
        self.assertEqual(caught.exception.info["iterations"], 0)

    def test_zero_rhs_converges_without_updates(self):
        op = self.op(cg_rtol=1e-14, cg_maxiter=0)
        x = op.solve_raw(1., self.alpha, torch.zeros_like(self.b))
        self.assertTrue(torch.equal(x, torch.zeros_like(x)))
        self.assertEqual(op.last_cg_info["iterations"], 0)
        self.assertEqual(op.last_cg_info["relative_residual"], [0.])

    def test_exact_initial_guess_accepted_at_zero_budget(self):
        op = self.op(cg_rtol=1e-14, cg_maxiter=0)
        x0 = torch.ones_like(self.b)
        b = op.M * x0 - self.alpha * op.Lmv(x0)
        x = op._cg_solve(1., self.alpha, b, x0)
        self.assertTrue(torch.equal(x, x0)); self.assertTrue(op.last_cg_info["converged"])

    def test_tiny_nonzero_residual_cannot_pass_zero_rhs(self):
        op = self.op(cg_rtol=1e-14, cg_maxiter=0)
        with self.assertRaises(CGConvergenceError) as caught:
            op._cg_solve(1., self.alpha, torch.zeros_like(self.b), torch.full_like(self.b, 1e-200))
        self.assertEqual(caught.exception.info["status"], "residual_norm_underflow")
        self.assertFalse(caught.exception.info["converged"])

    def test_final_permitted_update_can_pass_true_residual(self):
        geom = SimpleNamespace(M=np.ones(2), L=sp.csr_matrix([[-1., 1.], [1., -1.]]))
        op = SPDOperator(geom, backend="cg", cg_rtol=1e-14, cg_maxiter=1)
        b = torch.tensor([1., -1.], dtype=torch.float64)
        x = op._cg_solve(1., torch.tensor(1., dtype=torch.float64), b, torch.zeros_like(b))
        self.assertEqual(op.cg_iterations, 1)
        self.assertTrue(op.last_cg_info["converged"])
        torch.testing.assert_close(x, b / 3, rtol=1e-14, atol=1e-14)

    def test_batch_requires_every_rhs_to_pass(self):
        op = self.op(cg_rtol=1e-14, cg_maxiter=1)
        with self.assertRaises(CGConvergenceError) as caught:
            op.solve_raw(1., self.alpha, torch.stack((torch.zeros_like(self.b), self.b), dim=1))
        self.assertEqual(caught.exception.info["converged_per_rhs"], [True, False])
        self.assertFalse(caught.exception.info["converged"])

    def test_nonfinite_rhs_is_explicit_failure(self):
        for value in (float("nan"), float("inf")):
            op = self.op(); b = self.b.clone(); b[0] = value
            with self.subTest(value=value), self.assertRaises(CGConvergenceError) as caught:
                op.solve_raw(1., self.alpha, b)
            self.assertFalse(caught.exception.info["finite"])
            json.dumps(caught.exception.info, allow_nan=False)

    def test_nonfinite_alpha_is_explicit_failure(self):
        for value in (float("nan"), float("inf")):
            op = self.op()
            with self.subTest(value=value), self.assertRaises(CGConvergenceError) as caught:
                op.solve_raw(1., torch.tensor(value), self.b)
            self.assertFalse(caught.exception.info["converged"])

    def test_nonfinite_initial_guess_is_explicit_failure(self):
        op = self.op(); x0 = torch.zeros_like(self.b); x0[0] = float("nan")
        with self.assertRaises(CGConvergenceError):
            op._cg_solve(1., self.alpha, self.b, x0)

    def test_nonfinite_operator_is_explicit_failure(self):
        geom = SimpleNamespace(M=np.ones(2), L=sp.csr_matrix([[-1., float("nan")], [float("nan"), -1.]]))
        op = SPDOperator(geom, backend="cg")
        with self.assertRaises(CGConvergenceError) as caught:
            op.solve_raw(1., self.alpha, torch.ones(2, dtype=torch.float64))
        self.assertFalse(caught.exception.info["finite"])

    def test_finite_input_norm_overflow_is_explicit_failure(self):
        op = self.op()
        with self.assertRaises(CGConvergenceError) as caught:
            op.solve_raw(1., self.alpha, torch.full_like(self.b, 1e200))
        self.assertFalse(caught.exception.info["finite"])

    def test_nonpositive_preconditioner_fails(self):
        op = self.op()
        with self.assertRaises(CGConvergenceError) as caught:
            op.solve_raw(-1., torch.tensor(0., dtype=torch.float64), self.b)
        self.assertEqual(caught.exception.info["status"], "nonpositive_preconditioner")

    def test_indefinite_matrix_breakdown_is_not_success(self):
        geom = SimpleNamespace(M=np.ones(2), L=sp.csr_matrix([[0., 2.], [2., 0.]]))
        op = SPDOperator(geom, backend="cg")
        b = torch.ones(2, dtype=torch.float64)
        with self.assertRaises(CGConvergenceError) as caught:
            op._cg_solve(1., torch.tensor(1., dtype=torch.float64), b, torch.zeros_like(b))
        self.assertEqual(caught.exception.info["status"], "breakdown")

    def test_float32_cannot_silently_relax_tolerance(self):
        op = self.op(dtype=torch.float32, cg_rtol=1e-14, cg_maxiter=100)
        with self.assertRaises(CGConvergenceError) as caught:
            op.solve_raw(1., self.alpha.float(), self.b.float())
        self.assertEqual(caught.exception.info["rtol"], 1e-14)
        self.assertGreater(caught.exception.info["relative_residual"][0], 1e-14)
        self.assertGreater(caught.exception.info["residual_restarts"], 0)

    def test_invalid_cg_options_fail_explicitly(self):
        for options in ({"cg_rtol": -1.}, {"cg_rtol": float("nan")}, {"cg_rtol": float("inf")},
                        {"cg_maxiter": -1}, {"cg_maxiter": 1.5}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.op(**options)

    def test_forward_failure_propagates_through_autograd(self):
        op = self.op(cg_rtol=1e-14, cg_maxiter=1)
        with self.assertRaises(CGConvergenceError) as caught:
            op.solve(1., self.alpha.clone().requires_grad_(), self.b.clone().requires_grad_())
        self.assertEqual(caught.exception.info["phase"], "forward")

    def test_backward_failure_does_not_return_gradient(self):
        op = self.op(cg_rtol=1e-14)
        alpha, b = self.alpha.clone().requires_grad_(), self.b.clone().requires_grad_()
        x = op.solve(1., alpha, b)
        op.cg_maxiter = 1
        with self.assertRaises(CGConvergenceError) as caught:
            x.square().sum().backward()
        self.assertEqual(caught.exception.info["phase"], "backward")
        self.assertIsNone(alpha.grad); self.assertIsNone(b.grad)

    def test_failure_propagates_through_time_integrator(self):
        solver = GrayScottSolver(self.g, dt=1., backend="cg", cg_rtol=1e-14, cg_maxiter=1)
        with self.assertRaises(CGConvergenceError) as caught:
            solver.rollout(*initial_fields(self.g), dict(F=.028, k=.062, Du=.1, Dv=.05), 3)
        self.assertEqual(caught.exception.info["phase"], "forward")

    def test_successful_adjoint_matches_lu(self):
        gradients = []
        for backend in ("lu", "cg"):
            op = SPDOperator(self.g, backend=backend, cg_rtol=1e-14)
            a, b = self.alpha.clone().requires_grad_(), self.b.clone().requires_grad_()
            x = op.solve(1., a, b); x.square().mean().backward(); gradients.append((a.grad, b.grad))
            if backend == "cg":
                self.assertEqual(op.last_cg_info["phase"], "backward")
                self.assertTrue(op.last_cg_info["converged"])
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, rtol=2e-12, atol=2e-12)

    def test_exception_report_survives_serialization(self):
        op = self.op(cg_maxiter=1, cg_rtol=1e-14)
        with self.assertRaises(CGConvergenceError) as caught:
            op.solve_raw(1., self.alpha, self.b)
        restored = pickle.loads(pickle.dumps(caught.exception))
        self.assertEqual(restored.info, caught.exception.info)
        json.dumps(restored.info, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
