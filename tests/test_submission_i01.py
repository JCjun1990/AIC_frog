import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from turing_rd import Geometry, SPOTS, STRIPES, icosphere
from turing_rd.result_io import save_certified_mat

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("submission_i01_test", ROOT / "scripts" / "make_submission.py")
submission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submission)
KEYS = ("sphere6_spots", "sphere6_stripes", "frog_spots", "frog_stripes")


class SubmissionValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh = icosphere(1)
        cls.geom = Geometry(cls.mesh, laplacian="robust", cache_dir=None)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tasks = self.root / "tasks"
        self.records = {}
        for key in KEYS:
            surface, pattern = key.rsplit("_", 1)
            params = SPOTS if pattern == "spots" else STRIPES
            directory = self.tasks / key
            cert = save_certified_mat(directory / "solution.mat", self.geom, params,
                                       np.ones(self.geom.n), np.zeros(self.geom.n), seed=0)
            rec = {"surface": surface, **params.as_dict(), "seed": 0, "mat_certificate": cert,
                   "rel_residual": cert["residual"]["value"], "stable": True,
                   "pattern_after": {"type": "uniform", "n_components": 0}, "wall_seconds": 1}
            self.records[key] = rec
            self.write_record(key)

    def write_record(self, key):
        (self.tasks / key / "summary.json").write_text(json.dumps(self.records[key]), encoding="utf-8")

    def validate_small(self):
        # Exercise real robust/Geometry/residual validation, using only a small test mesh.
        with patch("turing_rd.icosphere", return_value=self.mesh), patch("turing_rd.load_frog", return_value=self.mesh):
            return submission.validate_task_results(self.tasks)

    def test_all_four_mats_are_recomputed_with_same_operator(self):
        validated = self.validate_small()
        self.assertEqual(set(validated), set(KEYS))
        self.assertTrue(all(rec["mat_certificate"]["accepted"] for rec in validated.values()))

    def test_legacy_missing_certificate_fails_closed(self):
        del self.records["frog_spots"]["mat_certificate"]
        self.write_record("frog_spots")
        with self.assertRaisesRegex(ValueError, "missing I01 mat_certificate"):
            self.validate_small()

    def test_summary_residual_parameter_and_mat_tampering_are_rejected(self):
        for field, value in [("rel_residual", 1e-5), ("F", .029), ("seed", 1)]:
            original = copy.deepcopy(self.records["frog_spots"])
            self.records["frog_spots"][field] = value
            self.write_record("frog_spots")
            with self.assertRaises(ValueError):
                self.validate_small()
            self.records["frog_spots"] = original
            self.write_record("frog_spots")
        with (self.tasks / "frog_spots" / "solution.mat").open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "mat_sha256"):
            self.validate_small()

    def test_validate_only_writes_no_package(self):
        output = self.root / "never_created"
        with patch.object(submission, "validate_task_results", return_value=self.records), \
                patch("sys.argv", ["make_submission.py", "Test", "--validate-only", "--out", str(output)]), \
                redirect_stdout(io.StringIO()):
            submission.main()
        self.assertFalse(output.exists())

    def test_existing_package_is_preserved(self):
        output = self.root / "packages"
        package = output / "Test_Round_AI-MECH-2026-002_result"
        package.mkdir(parents=True)
        sentinel = package / "keep.txt"; sentinel.write_text("original")
        with patch.object(submission, "validate_task_results", return_value=self.records), \
                patch("sys.argv", ["make_submission.py", "Test", "--round", "Round", "--out", str(output)]):
            with self.assertRaisesRegex(SystemExit, "already exists"):
                submission.main()
        self.assertEqual(sentinel.read_text(), "original")

    def test_new_package_mat_hashes_match_certificates(self):
        manifest = []
        for key in KEYS:
            surface, pattern = key.rsplit("_", 1)
            target = "task2_frog" if surface == "frog" else "task1_sphere"
            for name in ("solution.mat", "summary.json"):
                manifest.append((f"{target}/ai_pde_{pattern}/{name}", self.tasks / key / name))
        output = self.root / "packages"
        with patch.object(submission, "validate_task_results", return_value=self.records), \
                patch.object(submission, "manifest", return_value=manifest), \
                patch.object(submission, "readme", return_value="I01 only; stability not certified"), \
                patch("sys.argv", ["make_submission.py", "Test", "--round", "Round", "--out", str(output)]), \
                redirect_stdout(io.StringIO()):
            submission.main()
        path = next(output.glob("*.zip"))
        with zipfile.ZipFile(path) as archive:
            self.assertIsNone(archive.testzip())
            for target, source in manifest:
                self.assertEqual(archive.read(target), source.read_bytes())

    def test_readme_does_not_upgrade_legacy_stable_to_new_certificate(self):
        text = submission.readme("Test", "Round", self.tasks, self.records)
        self.assertIn("I01", text)
        self.assertIn("+1.7e-6", text)
        self.assertNotIn("稳定，max Re", text)


if __name__ == "__main__":
    unittest.main()
