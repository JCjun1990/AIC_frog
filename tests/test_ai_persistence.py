import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import scipy.io as sio

from turing_rd import Geometry, icosphere

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ai_generation_i01_test", ROOT / "scripts" / "ai_generation.py")
ai = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ai)


class StateRetentionTests(unittest.TestCase):
    def setUp(self):
        self.u = np.array([.7 + 1e-10, .8 - 1e-10], dtype=np.float64)
        self.v = np.array([.1 + 1e-10, .2 - 1e-10], dtype=np.float64)

    def test_formal_and_training_storage_are_separate(self):
        formal = ai._retain_state(self.u, self.v)
        self.assertEqual(formal["state_storage"], {"dtype": "float64", "purpose": "formal_field"})
        np.testing.assert_array_equal(formal["state"][0], self.u)
        self.assertFalse(np.array_equal(formal["state"][0], self.u.astype(np.float32).astype(np.float64)))
        cached = ai._retain_state(self.u, self.v, "float32")
        self.assertEqual(cached["state"][0].dtype, np.float32)
        self.assertEqual(cached["state_storage"]["purpose"], "training_cache")
        with self.assertRaises(ValueError):
            ai._retain_state(self.u.astype(np.float32), self.v)

    def test_run_ai_retains_actual_float64_solver_values(self):
        with patch.object(ai, "geom", return_value=object()), patch.object(ai, "rotations", return_value=None), \
                patch("joblib.load", return_value={}), \
                patch.object(ai, "ai_initial_state", return_value=(self.u, self.v, {"type": "spots"})), \
                patch.object(ai, "relax_polish", return_value=(self.u, self.v, {"wall_s": .1, "newton": []})), \
                patch.object(ai, "analyse", return_value={"stable": False, "wall_s": .1}):
            result = ai.run_ai("frog", .028, .062, 0, 1, "unused", keep_state=True)
        np.testing.assert_array_equal(result["state"][0], self.u)
        np.testing.assert_array_equal(result["state"][1], self.v)
        self.assertEqual(result["initial_storage"]["purpose"], "visualization_only")

    def test_run_pde_formal_default_preserves_float64(self):
        with patch.object(ai, "geom", return_value=object()), patch.object(ai, "rotations", return_value=None), \
                patch.object(ai, "perturbed_ic", return_value=(self.u, self.v)), \
                patch.object(ai, "relax_polish", return_value=(self.u, self.v, {"wall_s": .1})), \
                patch.object(ai, "analyse", return_value={"type": "spots", "wall_s": .1}), \
                patch.object(ai, "labels", return_value={}):
            result = ai.run_pde("frog", .028, .062, 0, 1, keep_state=True)
        np.testing.assert_array_equal(result["state"][0], self.u)
        self.assertEqual(result["state"][0].dtype, np.float64)

    def test_generation_rejects_existing_directory_before_running(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(ai, "run_ai") as run:
            (Path(tmp) / "frog_spots").mkdir()
            with self.assertRaises(FileExistsError):
                ai.stage_generate(lambda message: None, ["frog"], output_dir=tmp)
            run.assert_not_called()

    def test_generate_writes_actual_mat_bound_certificate_and_records_failed_candidates(self):
        geometry = Geometry(icosphere(1), laplacian="robust", cache_dir=None)
        for valid in (True, False):
            with self.subTest(valid=valid), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "frog").mkdir()
                (root / "frog" / "model.joblib").write_bytes(b"mock model; never deserialized")
                u = np.ones(geometry.n) if valid else np.full(geometry.n, .7)
                v = np.zeros(geometry.n) if valid else np.full(geometry.n, .1)

                def generated(*args, **kwargs):
                    return {"pipeline": "ai", "surface": "frog", "F": args[1], "k": args[2], "seed": 0,
                            "state": (u.copy(), v.copy()), "initial": (u.astype(np.float32), v.astype(np.float32)),
                            "rel_residual": 0, "total_s": 0, "stable": False, "max_re_lambda": None}

                with patch.object(ai, "OUT", root), patch.object(ai, "geom", return_value=geometry), \
                        patch.object(geometry, "eigenbasis"), patch.object(ai, "run_ai", side_effect=generated), \
                        patch("turing_rd.viz.render_field"):
                    if valid:
                        ai.stage_generate(lambda message: None, ["frog"], output_dir=root / "new")
                    else:
                        with self.assertRaisesRegex(RuntimeError, "MAT readback rejected"):
                            ai.stage_generate(lambda message: None, ["frog"], output_dir=root / "new")
                for task in ("frog_spots", "frog_stripes"):
                    target = root / "new" / task
                    record = json.loads((target / "summary.json").read_text(encoding="utf-8"))
                    self.assertEqual(record["mat_certificate"]["accepted"], valid)
                    self.assertEqual(record["rel_residual"], record["mat_certificate"]["residual"]["value"])
                    saved = sio.loadmat(target / "solution.mat", struct_as_record=False)["Solution"][0, 0]
                    np.testing.assert_array_equal(saved.u[:, 0], u)
                    self.assertEqual(record["full_discrete_stability"], "not_certified_by_I01")


if __name__ == "__main__":
    unittest.main()
