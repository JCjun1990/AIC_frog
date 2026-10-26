import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from turing_rd import Geometry, icosphere

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ai_generation_seed_test", ROOT / "scripts" / "ai_generation.py")
ai = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ai)


class GenerateSeedTests(unittest.TestCase):
    def test_override_parser_accepts_only_requested_unique_nonnegative_seeds(self):
        self.assertEqual(ai.parse_generate_seeds(["sphere6:stripes=0"], ["frog", "sphere6"]),
                         {("sphere6", "stripes"): 0})
        for items in (["sphere6:stripes=-1"], ["sphere6:stripes=--1"], ["sphere6:stripes=x"],
                      ["sphere5:stripes=2"], ["sphere6:typo=2"], ["stripes=2"],
                      ["sphere6:stripes=2", "sphere6:stripes=3"]):
            with self.subTest(items=items), self.assertRaises(ValueError):
                ai.parse_generate_seeds(items, ["frog", "sphere6"])

    def test_invalid_worker_count_fails_before_geometry_or_solver(self):
        for workers in (0, -1, True):
            with self.subTest(workers=workers), patch.object(ai, "geom") as geometry, \
                    self.assertRaisesRegex(ValueError, "positive integer"):
                ai.stage_generate(lambda message: None, workers=workers)
            geometry.assert_not_called()

    def test_default_task_seeds(self):
        self.assertEqual(ai.task_seed("sphere6", "stripes"), 2)          # the submitted seven-band state
        for surface, name in (("sphere6", "spots"), ("frog", "spots"), ("frog", "stripes")):
            self.assertEqual(ai.task_seed(surface, name), 0)
        self.assertEqual(ai.task_seed("sphere6", "stripes", {("sphere6", "stripes"): 0}), 0)

    def check_generation(self, workers):
        geometry = Geometry(icosphere(1), laplacian="robust", cache_dir=None)
        u, v = np.ones(geometry.n), np.zeros(geometry.n)
        calls = []

        def generated(surface, F, k, seed, *args, **kwargs):
            calls.append((surface, F, k, seed))
            return {"pipeline": "ai", "surface": surface, "F": F, "k": k, "seed": seed,
                    "state": (u.copy(), v.copy()), "initial": (u.astype(np.float32), v.astype(np.float32)),
                    "rel_residual": 0, "total_s": 0, "stable": False, "max_re_lambda": None}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for train in ("frog", "sphere5"):
                (root / train).mkdir()
                (root / train / "model.joblib").write_bytes(b"mock model; never deserialized")
            with patch.object(ai, "OUT", root), patch.object(ai, "geom", return_value=geometry), \
                    patch.object(geometry, "eigenbasis"), patch.object(ai, "run_ai", side_effect=generated), \
                    patch("turing_rd.viz.render_field"), patch.object(ai, "ProcessPoolExecutor") as pool:
                pool.return_value.__enter__.return_value.map.side_effect = lambda fn, *args: map(fn, *args)
                ai.stage_generate(lambda message: None, ["frog", "sphere6"], output_dir=root / "new", workers=workers)
                if workers > 1:
                    pool.assert_called_once_with(max_workers=4)
                else:
                    pool.assert_not_called()
            seeds = {(s, "spots" if F == ai.SPOTS.F else "stripes"): sd for s, F, k, sd in calls}
            self.assertEqual(seeds, {("frog", "spots"): 0, ("frog", "stripes"): 0,
                                     ("sphere6", "spots"): 0, ("sphere6", "stripes"): 2})
            import json
            cert = json.loads((root / "new" / "sphere6_stripes" / "summary.json").read_text(encoding="utf-8"))["mat_certificate"]
            self.assertEqual(cert["seed"], 2)

    def test_generate_runs_and_certifies_each_task_with_its_seed(self):
        self.check_generation(workers=1)

    def test_parallel_dispatch_preserves_task_seeds_and_caps_worker_count(self):
        self.check_generation(workers=8)


if __name__ == "__main__":
    unittest.main()
