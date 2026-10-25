"""F10 regression tests: real cache I/O, tiny actual operators, no training or PDE rollout."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from turing_rd import Geometry, icosphere
from turing_rd import experiment_cache as C

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ai_cache_tests", ROOT / "scripts/ai_generation.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def concurrent_writer(args):
    path, identity, value = args
    C.save_cache(path, identity, {"value": value})
    return C.load_cache(path, identity)[0]["value"]


class CacheFormatTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.identity = {"protocol": C.PROTOCOL, "F": .028, "settings": {"seed": 2}}
        self.path = C.cache_path(self.tmp.name, self.identity)

    def test_canonical_order_and_full_float_precision(self):
        self.assertEqual(C.fingerprint({"a": 1, "b": 2}), C.fingerprint({"b": 2, "a": 1}))
        self.assertNotEqual(C.fingerprint({"F": .028}), C.fingerprint({"F": np.nextafter(.028, 1.)}))
        for value in (float("nan"), float("inf"), object()):
            with self.assertRaises((ValueError, TypeError)):
                C.fingerprint({"value": value})

    def test_round_trip_identity_and_array_payload(self):
        result = {"u": np.array([.7 + 1e-12], dtype="float64")}
        C.save_cache(self.path, self.identity, result)
        row, identity = C.load_cache(self.path, self.identity)
        np.testing.assert_array_equal(row["u"], result["u"])
        self.assertEqual(row["u"].dtype, np.dtype("float64"))
        self.assertEqual(identity, self.identity)

    def test_different_expected_identity_rejected(self):
        C.save_cache(self.path, self.identity, {"value": 1})
        with self.assertRaises(C.CacheRejected):
            C.load_cache(self.path, {**self.identity, "F": .029})

    def test_changed_payload_rejected(self):
        C.save_cache(self.path, self.identity, {"value": 1})
        data = bytearray(self.path.read_bytes()); data[-2] ^= 1
        self.path.write_bytes(data)
        with self.assertRaises(C.CacheRejected):
            C.load_cache(self.path, self.identity)

    def test_truncated_header_and_payload_rejected(self):
        for data in (b"", C.MAGIC + b"{}", C.MAGIC + b"{}\n", C.MAGIC + b"[]\n"):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_bytes(data)
            with self.assertRaises(C.CacheRejected):
                C.load_cache(self.path, self.identity)

    def test_fingerprint_metadata_tamper_rejected(self):
        C.save_cache(self.path, self.identity, {"value": 1})
        magic, header, payload = self.path.read_bytes().split(b"\n", 2)
        meta = json.loads(header); meta["identity"]["F"] = .029
        self.path.write_bytes(magic + b"\n" + C.canonical_bytes(meta) + b"\n" + payload)
        with self.assertRaises(C.CacheRejected):
            C.load_cache(self.path, self.identity)

    def test_wrong_filename_rejected(self):
        C.save_cache(self.path, self.identity, {"value": 1})
        wrong = self.path.with_name("wrong.cache"); wrong.write_bytes(self.path.read_bytes())
        with self.assertRaises(C.CacheRejected):
            C.load_cache(wrong, self.identity)

    def test_legacy_pickle_rejected_and_preserved(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = pickle.dumps({"stable": True}); self.path.write_bytes(data)
        with self.assertRaises(C.CacheRejected):
            C.load_cache(self.path, self.identity)
        self.assertEqual(self.path.read_bytes(), data)

    def test_error_row_cannot_be_cached(self):
        with self.assertRaises(C.CacheRejected):
            C.save_cache(self.path, self.identity, {"error": "failed"})
        self.assertFalse(self.path.exists())

    def test_concurrent_process_writes_are_atomic(self):
        C.save_cache(self.path, self.identity, {"value": -1})
        with ProcessPoolExecutor(max_workers=2) as ex:
            values = list(ex.map(concurrent_writer, [(self.path, self.identity, i) for i in range(8)]))
        self.assertEqual(len(values), 8)
        self.assertIn(C.load_cache(self.path, self.identity)[0]["value"], range(8))
        self.assertEqual(list(self.path.parent.glob(".writing-*")), [])

    def test_interrupted_replace_preserves_previous_entry(self):
        C.save_cache(self.path, self.identity, {"value": 1})
        with patch.object(C.os, "replace", side_effect=OSError("simulated interruption")):
            with self.assertRaises(OSError):
                C.save_cache(self.path, self.identity, {"value": 2})
        self.assertEqual(C.load_cache(self.path, self.identity)[0]["value"], 1)
        self.assertEqual(list(self.path.parent.glob(".writing-*")), [])

    def test_transient_windows_sharing_error_retries(self):
        real_replace = C.os.replace
        attempts = []
        def retry(source, target):
            attempts.append(1)
            if len(attempts) == 1:
                raise PermissionError("simulated Windows sharing violation")
            return real_replace(source, target)
        with patch.object(C.os, "replace", side_effect=retry):
            C.save_cache(self.path, self.identity, {"value": 1})
        self.assertEqual(len(attempts), 2)
        self.assertEqual(C.load_cache(self.path, self.identity)[0]["value"], 1)

    def test_actual_code_hash_tracks_dirty_file(self):
        root = Path(self.tmp.name); (root / "scripts").mkdir(); (root / "turing_rd").mkdir()
        source = root / "scripts/ai_generation.py"; source.write_text("first")
        (root / "turing_rd/solver.py").write_text("solver")
        first = C.code_identity(root); source.write_text("second")
        self.assertNotEqual(first, C.code_identity(root))

    def test_dependency_snapshot_records_versions(self):
        record = C.dependency_versions()
        for key in ("python", "numpy", "scipy", "torch", "scikit-learn", "robust-laplacian", "joblib", "threads"):
            self.assertIn(key, record)

    def test_training_provenance_written_only_explicitly(self):
        weights = Path(self.tmp.name) / "new.joblib"; weights.write_bytes(b"test fixture, not a trained model")
        data = Path(self.tmp.name) / "dataset.json"; data.write_text('{"rows":[]}')
        record = C.write_training_provenance(weights, [C.file_identity(data)], {"stage": "test-only fixture"})
        saved = json.loads(weights.with_suffix(".provenance.json").read_text())
        self.assertEqual(record, saved)
        self.assertEqual(saved["model"]["sha256"], C.file_identity(weights)["sha256"])


class PoolFingerprintTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.cache = self.root / "jobs"
        self.weights = self.root / "model.joblib"; self.weights.write_bytes(b"fixture weights")
        self.data = self.root / "dataset.json"; self.data.write_text('{"rows":[]}')
        self.geometry = Geometry(icosphere(1), cache_dir=None)
        self.code_version = "original"; self.env_version = "original"; self.calls = 0; self.logs = []
        for target, value in (("geom", lambda _: self.geometry), ("ProcessPoolExecutor", ThreadPoolExecutor),
                              ("_safe", self.worker)):
            p = patch.object(A, target, value); p.start(); self.addCleanup(p.stop)
        p = patch.object(C, "code_identity", lambda _: {"test_source": self.code_version}); p.start(); self.addCleanup(p.stop)
        p = patch.object(C, "dependency_versions", lambda: {"test_dependency": self.env_version}); p.start(); self.addCleanup(p.stop)

    def worker(self, fn, args):
        self.calls += 1
        return {"pipeline": "ai" if fn is A.run_ai else "pde", "surface": args[0], "F": args[1], "k": args[2],
                "seed": args[3], "type": "fixture", "n_components": 1, "total_s": 0., "token": self.calls}

    def args(self):
        return ["sphere1", .028, .062, 2, 100., str(self.weights), False, False]

    def run_job(self, args=None, fn=None):
        return A.pool_map([(fn or A.run_ai, tuple(self.args() if args is None else args))], 1, self.logs.append, self.cache)[0]

    def assert_invalidates(self, change):
        args = self.args(); self.run_job(args); change(args); result = self.run_job(args)
        self.assertEqual(self.calls, 2, self.logs)
        self.assertNotIn("error", result, self.logs)

    def test_identical_job_hits_without_starting_executor(self):
        first = self.run_job()
        with patch.object(A, "ProcessPoolExecutor", side_effect=AssertionError("no new workers for cache hit")):
            second = self.run_job()
        self.assertEqual(self.calls, 1); self.assertEqual(first, second)

    def test_F_one_ulp_invalidates(self):
        self.assert_invalidates(lambda a: a.__setitem__(1, float(np.nextafter(a[1], 1.))))

    def test_k_one_ulp_invalidates(self):
        self.assert_invalidates(lambda a: a.__setitem__(2, float(np.nextafter(a[2], 1.))))

    def test_model_content_invalidates(self):
        self.assert_invalidates(lambda _: self.weights.write_bytes(b"updated weights"))

    def test_model_path_invalidates_even_identical_content(self):
        path = self.root / "other.joblib"; path.write_bytes(self.weights.read_bytes())
        self.assert_invalidates(lambda a: a.__setitem__(5, str(path)))

    def test_training_data_invalidates(self):
        self.assert_invalidates(lambda _: self.data.write_text('{"rows":[1]}'))

    def test_input_data_sha_invalidates(self):
        source = self.root / "frog.mat"; source.write_bytes(b"original mesh input")
        args = self.args(); args[0] = "frog"
        with patch.object(A, "ROOT", self.root):
            self.run_job(args); source.write_bytes(b"modified mesh input"); result = self.run_job(args)
        self.assertEqual(self.calls, 2); self.assertNotIn("error", result)

    def test_training_provenance_invalidates(self):
        self.assert_invalidates(lambda _: self.weights.with_suffix(".provenance.json").write_text('{"version":2}'))

    def test_budget_invalidates(self):
        self.assert_invalidates(lambda a: a.__setitem__(4, 200.))

    def test_orient_invalidates(self):
        self.assert_invalidates(lambda a: a.__setitem__(7, True))

    def test_tol_invalidates(self):
        self.run_job()
        with patch.object(A, "TOL", 1e-12):
            result = self.run_job()
        self.assertEqual(self.calls, 2); self.assertNotIn("error", result)

    def test_mesh_invalidates(self):
        self.assert_invalidates(lambda _: setattr(self, "geometry", Geometry(icosphere(2), cache_dir=None)))

    def test_actual_operator_changes_invalidate_same_mesh(self):
        self.assert_invalidates(lambda _: setattr(self.geometry, "L", self.geometry.L * 1.01))

    def test_mass_changes_invalidate_same_mesh(self):
        self.assert_invalidates(lambda _: self.geometry.M.__imul__(1.01))

    def test_basis_input_changes_invalidate(self):
        self.assert_invalidates(lambda _: setattr(self.geometry, "_eig", (np.array([0., 1.]), np.ones((self.geometry.n, 2)))))

    def test_dtype_invalidates(self):
        args = ["sphere1", .028, .062, 2, 100., True, "float32"]
        self.run_job(args, A.run_pde); args[-1] = "float64"; self.run_job(args, A.run_pde)
        self.assertEqual(self.calls, 2)

    def test_keep_state_invalidates(self):
        self.assert_invalidates(lambda a: a.__setitem__(6, True))

    def test_code_invalidates(self):
        self.assert_invalidates(lambda _: setattr(self, "code_version", "changed"))

    def test_algorithm_version_invalidates(self):
        self.run_job()
        with patch.object(A, "EXPERIMENT_ALGORITHM_VERSION", "new-algorithm"):
            result = self.run_job()
        self.assertEqual(self.calls, 2); self.assertNotIn("error", result)

    def test_dependency_version_invalidates(self):
        self.assert_invalidates(lambda _: setattr(self, "env_version", "changed"))

    def test_schedule_invalidates(self):
        self.run_job()
        with patch.object(A, "AI_SCHEDULE", lambda t: ((.5, 1000.), (2., t))):
            result = self.run_job()
        self.assertEqual(self.calls, 2); self.assertNotIn("error", result)

    def test_corruption_recomputes(self):
        self.run_job()
        path = next(self.cache.rglob("*.cache")); path.write_bytes(b"broken")
        self.assertNotIn("error", self.run_job()); self.assertEqual(self.calls, 2)
        self.assertTrue(any("rejected" in line for line in self.logs))

    def test_payload_job_binding_tamper_recomputes(self):
        self.run_job()
        path = next(self.cache.rglob("*.cache")); row, identity = C.load_cache(path)
        row["F"] = .029; C.save_cache(path, identity, row)
        self.run_job(); self.assertEqual(self.calls, 2)

    def test_legacy_file_never_hit_or_deleted(self):
        self.cache.mkdir()
        legacy = self.cache / "ai_sphere1_0.0280_0.0620_2.pkl"; data = pickle.dumps({"token": -1})
        legacy.write_bytes(data); result = self.run_job()
        self.assertEqual(self.calls, 1); self.assertEqual(result["token"], 1)
        self.assertEqual(legacy.read_bytes(), data)

    def test_legacy_only_summarize_refuses(self):
        self.cache.mkdir(); (self.cache / "old.pkl").write_bytes(pickle.dumps({"stable": True}))
        with self.assertRaises(C.CacheRejected):
            A.checkpoint_rows(self.cache, self.logs.append)

    def test_current_summarize_and_explicit_load_pass(self):
        first = self.run_job()
        self.assertEqual(A.checkpoint_rows(self.cache, self.logs.append), [first])
        self.assertEqual(A.cached_results([(A.run_ai, tuple(self.args()))], self.cache), [first])

    def test_stale_code_summarize_refuses(self):
        self.run_job(); self.code_version = "changed"
        with self.assertRaises(C.CacheRejected):
            A.checkpoint_rows(self.cache, self.logs.append)

    def test_summarize_refuses_mixed_budgets_but_explicit_jobs_work(self):
        args = self.args(); first = self.run_job(args); args[4] = 200.; self.run_job(args)
        with self.assertRaisesRegex(C.CacheRejected, "multiple configurations"):
            A.checkpoint_rows(self.cache, self.logs.append)
        self.assertEqual(A.cached_results([(A.run_ai, tuple(self.args()))], self.cache), [first])

    def test_missing_model_fails_before_worker(self):
        self.weights.unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_job()
        self.assertEqual(self.calls, 0)

    def test_nonfinite_parameter_fails_before_worker(self):
        args = self.args(); args[1] = float("nan")
        with self.assertRaises(ValueError):
            self.run_job(args)
        self.assertEqual(self.calls, 0)

    def test_dependency_change_during_run_is_not_cached(self):
        original = self.worker
        def changed(fn, args):
            result = original(fn, args); self.weights.write_bytes(b"mutated mid-run"); return result
        with patch.object(A, "_safe", changed):
            result = self.run_job()
        self.assertIn("dependencies changed", result["error"])
        self.assertEqual(list(self.cache.rglob("*.cache")), [])

    def test_worker_error_is_retried(self):
        with patch.object(A, "_safe", return_value={"pipeline": "ai", "surface": "sphere1", "error": "fixture failure"}):
            result = self.run_job()
        self.assertEqual(result["error"], "fixture failure")
        self.assertEqual(list(self.cache.rglob("*.cache")), [])
        self.assertNotIn("error", self.run_job()); self.assertEqual(self.calls, 1)


class ModelInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        import active_learning
        cls.active = active_learning

    def test_v1_v2_resolve_to_distinct_existing_weights(self):
        active = self.active
        for surface in ("frog", "sphere5"):
            v1, v2 = active.model_path("v1", surface), active.model_path("v2", surface)
            self.assertTrue(v1.is_file()); self.assertTrue(v2.is_file()); self.assertNotEqual(v1, v2)

    def test_missing_inference_model_does_not_train(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(self.active.G, "DENSE", Path(tmp)), \
                patch.object(self.active.G, "fit", side_effect=AssertionError("inference cannot train")):
            with self.assertRaises(FileNotFoundError):
                self.active.model_path("v2", "frog")
            with self.assertRaises(FileNotFoundError):
                self.active.G.model_v2("frog")

    def test_missing_run_ai_model_does_not_generate(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(A, "geom"), patch.object(A, "rotations"), \
                patch.object(A, "ai_initial_state", side_effect=AssertionError("cannot generate without weights")):
            with self.assertRaises(FileNotFoundError):
                A.run_ai("frog", .028, .062, 2, 10., str(Path(tmp) / "missing.joblib"))


if __name__ == "__main__":
    unittest.main()
