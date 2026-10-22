import copy
import importlib.util
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    obj = importlib.util.module_from_spec(spec); spec.loader.exec_module(obj)
    return obj


workflow = module("portable_workflow_test", "submission_workflow.py")
verifier = module("portable_verifier_test", "i03_verify_package.py")
packer = module("portable_packer_test", "i03_package.py")


class PortableSubmissionTests(unittest.TestCase):
    def test_saved_certificates_bind_all_four_actual_mats(self):
        for task in workflow.TASKS:
            directory = ROOT / "submission" / task
            c = workflow.load(directory / "stability_certificate.json")
            result = workflow.validate_spectral_record(c, workflow.sha(directory / "solution.mat"))
            self.assertEqual(result["status"], "numerical_check_passed")
            self.assertLess(result["sampled_max_real"], 0)

    def test_spectral_metadata_tampering_fails_closed(self):
        directory = ROOT / "submission/task2_frog/spots"
        original = workflow.load(directory / "stability_certificate.json")
        for change in ("positive", "residual", "convergence", "lr", "hash"):
            c = copy.deepcopy(original)
            if change == "positive": c["configurations"][0]["pairs"][0]["eigenvalue"][0] = 1e-6
            if change == "residual": c["configurations"][0]["pairs"][0]["residual_relative_M"] = 1
            if change == "convergence": c["configurations"][0]["solver_converged"] = False
            if change == "lr": c["unshifted_lr_attempt"]["used_as_pass_evidence"] = True
            if change == "hash": c["input"]["mat_sha256"] = "0" * 64
            with self.subTest(change=change), self.assertRaises(ValueError):
                workflow.validate_spectral_record(c, workflow.sha(directory / "solution.mat"))

    def test_zip_escape_drive_backslash_and_duplicates_rejected(self):
        for names in (["../escape"], ["C:/escape"], ["/escape"], ["dir\\escape"], ["a", "A"], ["trailing./a"]):
            data = io.BytesIO()
            with zipfile.ZipFile(data, "w") as z:
                for name in names: z.writestr(name, "test")
            if names == ["dir\\escape"]:
                # Windows ZipInfo normalizes separators while writing. Patch both
                # headers to represent an externally supplied malicious archive.
                data = io.BytesIO(data.getvalue().replace(b"dir/escape", b"dir\\escape"))
            with zipfile.ZipFile(data) as z:
                self.assertTrue(verifier.unsafe_zip_members(z), names)

    def test_zip_symlink_rejected_and_valid_layout_accepted(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as z:
            info = zipfile.ZipInfo("link"); info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(info, "../target")
        with zipfile.ZipFile(data) as z:
            self.assertTrue(verifier.unsafe_zip_members(z))
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as z: z.writestr("task1_sphere/spots/solution.mat", "test")
        with zipfile.ZipFile(data) as z: self.assertFalse(verifier.unsafe_zip_members(z))

    def test_model_copy_layout_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); source, out = tmp / "package", tmp / "inputs"
            for task in ("task1_sphere", "task2_frog"):
                directory = source / task / "ai_model"; directory.mkdir(parents=True)
                for name in ("model.joblib", "dataset.json", "cv.json"): (directory / name).write_bytes(b"fixture")
            result = workflow.prepare_models(source, out)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual((out / "frog/model.joblib").read_bytes(), b"fixture")
            self.assertEqual((out / "sphere5/model.joblib").read_bytes(), b"fixture")
            with self.assertRaises(FileExistsError): workflow.prepare_models(source, out)

    def test_packer_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); existing = out / "Test_初赛_AI-MECH-2026-002_result"
            existing.mkdir(); (existing / "keep").write_text("original")
            with self.assertRaises(FileExistsError):
                packer.build(out / "absent-source", out, team="Test")
            self.assertEqual((existing / "keep").read_text(), "original")


if __name__ == "__main__":
    unittest.main()
