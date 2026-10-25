"""Regression tests for the submission manifest invariants.

The 2026-10-23 audit found `submission/PACKAGE_MANIFEST.json` stale: 105 listed
entries while the package carried 107, and three hashes still described files
from before the wording fixes.  `i03_package.build()` only ever wrote a refreshed
manifest *into* the package, so nothing regenerated the workspace copy.

These tests pin the failure modes that let that happen silently: missing file,
tampered file, unregistered file, duplicate path, ZIP/workspace drift, plus the
new sync path that re-derives the manifest and refuses to bless changed formal
artifacts.
"""
import importlib.util
import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


workflow = module("submission_workflow_test", "submission_workflow.py")
packager = module("i03_package_test", "i03_package.py")

FILES = {
    "README_提交.md": "# 待填 · 初赛答案包\n",
    "VALIDATION.md": "validation\n",
    "evidence/generalization/summary.md": "summary\n",
    "task1_sphere/spots/solution.mat": "mat-bytes",
}


def write(root, relative, data):
    path = Path(root) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")
    return path


def make_submission(root, files=None):
    files = dict(FILES if files is None else files)
    for relative, data in files.items():
        write(root, relative, data)
    entries = [{"path": relative,
                "sha256": workflow.sha(Path(root) / relative),
                "bytes": (Path(root) / relative).stat().st_size,
                "reason": "test", "source": "submission/" + relative}
               for relative in sorted(files)]
    manifest = {"schema_version": "test", "files": entries}
    (Path(root) / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return files


def make_zip(root, zip_path, files):
    files = set(files)
    with zipfile.ZipFile(zip_path, "w") as archive:
        for relative in sorted(files):
            archive.write(Path(root) / relative, relative)
        manifest = json.loads((Path(root) / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
        manifest["files"] = [entry for entry in manifest["files"] if entry["path"] in files]
        archive.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=1))


class ManifestCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.files = make_submission(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_complete_manifest_passes(self):
        result = workflow.check_manifest(self.root)
        self.assertEqual(result["manifest_files"], len(self.files))
        self.assertEqual(result["payload_files"], len(self.files))

    def test_missing_file_fails(self):
        (self.root / "evidence/generalization/summary.md").unlink()
        with self.assertRaisesRegex(ValueError, "missing or escaping"):
            workflow.check_manifest(self.root)

    def test_tampered_file_fails(self):
        write(self.root, "README_提交.md", "edited after packaging\n")
        with self.assertRaisesRegex(ValueError, "hash/path mismatch"):
            workflow.check_manifest(self.root)

    def test_unregistered_file_fails(self):
        write(self.root, "evidence/generalization/QUALITY_BREAKDOWN.md", "new\n")
        with self.assertRaisesRegex(ValueError, "unregistered payload file"):
            workflow.check_manifest(self.root)

    def test_duplicate_path_fails(self):
        manifest_path = self.root / "PACKAGE_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"].append(dict(manifest["files"][0]))
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate manifest paths"):
            workflow.check_manifest(self.root)

    def test_non_payload_directories_are_ignored(self):
        write(self.root, "drafts/待填_result.zip", "zip")
        write(self.root, "report/初赛技术报告_v3.pdf", "pdf")
        write(self.root, "待填_初赛_AI-MECH-2026-002_result.zip", "zip")
        self.assertEqual(workflow.check_manifest(self.root)["manifest_files"], len(self.files))

    def test_zip_matching_workspace_passes(self):
        archive = self.root / "package.zip"
        make_zip(self.root, archive, self.files)
        result = workflow.compare_zip_to_workspace(self.root, archive)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["different"], {})

    def test_zip_workspace_drift_fails(self):
        archive = self.root / "package.zip"
        make_zip(self.root, archive, self.files)
        write(self.root, "README_提交.md", "edited after the zip was built\n")
        result = workflow.compare_zip_to_workspace(self.root, archive)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("README_提交.md", result["different"])

    def test_zip_missing_a_workspace_file_fails(self):
        archive = self.root / "package.zip"
        make_zip(self.root, archive, [p for p in self.files if p != "VALIDATION.md"])
        result = workflow.compare_zip_to_workspace(self.root, archive)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("VALIDATION.md", result["only_workspace"])

    def test_wrong_manifest_hash_fails(self):
        path = self.root / "PACKAGE_MANIFEST.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["files"][0]["sha256"] = "0" * 64
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash/path mismatch"):
            workflow.check_manifest(self.root)

    def test_wrong_payload_count_fails(self):
        path = self.root / "PACKAGE_MANIFEST.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["payload_file_count"] = len(self.files) + 1
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "payload count mismatch"):
            workflow.check_manifest(self.root)

    def test_alias_and_escaping_paths_fail(self):
        path = self.root / "PACKAGE_MANIFEST.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        for relative in ("../escape", "evidence/../alias", "code//alias", "C:/escape", "code\\alias"):
            manifest = copy.deepcopy(original)
            manifest["files"][0]["path"] = relative
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.subTest(path=relative), self.assertRaisesRegex(ValueError, "unsafe manifest path"):
                workflow.check_manifest(self.root)

    def test_case_insensitive_duplicates_fail(self):
        path = self.root / "PACKAGE_MANIFEST.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        entry = dict(manifest["files"][0])
        entry["path"] = entry["path"].swapcase()
        manifest["files"].append(entry)
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate manifest paths"):
            workflow.check_manifest(self.root)

    def test_nested_zip_is_not_an_implicit_exclusion(self):
        write(self.root, "evidence/unregistered.zip", "unexpected archive")
        with self.assertRaisesRegex(ValueError, "unregistered payload file"):
            workflow.check_manifest(self.root)

    def test_zip_tampered_payload_fails(self):
        archive = self.root / "package.zip"
        make_zip(self.root, archive, self.files)
        with zipfile.ZipFile(archive) as z:
            contents = {name: z.read(name) for name in z.namelist()}
        contents["VALIDATION.md"] = b"tampered"
        with zipfile.ZipFile(archive, "w") as z:
            for name, data in contents.items():
                z.writestr(name, data)
        with self.assertRaisesRegex(ValueError, "packed hash mismatch"):
            workflow.zip_manifest_hashes(archive)

    def test_zip_duplicate_manifest_entries_fail(self):
        path = self.root / "PACKAGE_MANIFEST.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["files"].append(dict(manifest["files"][0]))
        path.write_text(json.dumps(manifest), encoding="utf-8")
        archive = self.root / "package.zip"
        make_zip(self.root, archive, self.files)
        with self.assertRaisesRegex(ValueError, "duplicate manifest paths"):
            workflow.zip_manifest_hashes(archive)

    def test_zip_cannot_make_a_bad_workspace_manifest_pass(self):
        archive = self.root / "package.zip"
        make_zip(self.root, archive, self.files)
        path = self.root / "PACKAGE_MANIFEST.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["files"][0]["sha256"] = "0" * 64
        path.write_text(json.dumps(manifest), encoding="utf-8")
        result = workflow.compare_zip_to_workspace(self.root, archive)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("hash/path mismatch", result["workspace_manifest_error"])


class ManifestSyncTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.files = make_submission(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_sync_registers_a_new_evidence_file(self):
        write(self.root, "evidence/generalization/QUALITY_BREAKDOWN.md", "new evidence\n")
        with self.assertRaisesRegex(ValueError, "unregistered"):
            workflow.check_manifest(self.root)
        result = packager.sync_source_manifest(self.root)
        self.assertIn("evidence/generalization/QUALITY_BREAKDOWN.md", result["added"])
        self.assertEqual(workflow.check_manifest(self.root)["manifest_files"], len(self.files) + 1)

    def test_sync_refuses_a_changed_formal_artifact(self):
        write(self.root, "task1_sphere/spots/solution.mat", "different bytes")
        with self.assertRaisesRegex(ValueError, "formal result/model/summary changed"):
            packager.sync_source_manifest(self.root)

    def test_sync_refuses_a_vanished_file(self):
        (self.root / "VALIDATION.md").unlink()
        with self.assertRaisesRegex(ValueError, "disappeared"):
            packager.sync_source_manifest(self.root)

    def test_sync_is_idempotent(self):
        packager.sync_source_manifest(self.root)
        first = (self.root / "PACKAGE_MANIFEST.json").read_bytes()
        packager.sync_source_manifest(self.root)
        second = (self.root / "PACKAGE_MANIFEST.json").read_bytes()
        self.assertEqual(json.loads(first)["files"], json.loads(second)["files"])
        self.assertEqual(first, second)

    def test_sync_keeps_code_and_refreshes_only_from_repository_source(self):
        files = {**FILES, "code/scripts/submission_workflow.py": "old runtime"}
        make_submission(self.root, files)
        result = packager.sync_source_manifest(self.root)
        relative = "code/scripts/submission_workflow.py"
        self.assertIn(relative, result["refreshed_code"])
        self.assertEqual((self.root / relative).read_bytes(), (ROOT / "scripts/submission_workflow.py").read_bytes())
        self.assertEqual(workflow.check_manifest(self.root)["manifest_files"], len(files))

    def test_sync_keeps_all_existing_provenance_fields(self):
        path = self.root / "PACKAGE_MANIFEST.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        entry = manifest["files"][0]
        entry.update(source_sha256="1" * 64, sanitized=True,
                     transformations=["original sanitization"],
                     historical_provenance={"original": "archived input"}, custom_field={"keep": True})
        original = copy.deepcopy(entry)
        path.write_text(json.dumps(manifest), encoding="utf-8")
        packager.sync_source_manifest(self.root)
        after = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(next(r for r in after["files"] if r["path"] == entry["path"]), original)

    def test_sync_rejects_an_unexplained_document_change(self):
        path = self.root / "PACKAGE_MANIFEST.json"
        before = path.read_bytes()
        write(self.root, "README_提交.md", "unknown replacement")
        with self.assertRaisesRegex(ValueError, "unexplained payload change"):
            packager.sync_source_manifest(self.root)
        self.assertEqual(path.read_bytes(), before)

    def test_sync_records_explicit_update_provenance(self):
        path = self.root / "PACKAGE_MANIFEST.json"
        before = json.loads(path.read_text(encoding="utf-8"))
        old = next(r for r in before["files"] if r["path"] == "README_提交.md")
        write(self.root, "README_提交.md", "reviewed change")
        packager.sync_source_manifest(self.root, update_reasons={"README_提交.md": "reviewed wording correction"})
        after = json.loads(path.read_text(encoding="utf-8"))
        entry = next(r for r in after["files"] if r["path"] == "README_提交.md")
        self.assertEqual(entry["manifest_updates"][-1]["previous_entry"], old)
        self.assertEqual(entry["manifest_updates"][-1]["reason"], "reviewed wording correction")
        workflow.check_manifest(self.root)

    def test_sync_traces_a_known_change_to_validated_zip_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory)
            files = {**FILES, "README_提交.md": "accepted earlier correction"}
            make_submission(reference, files)
            archive = reference / "accepted.zip"
            make_zip(reference, archive, files)
            write(self.root, "README_提交.md", files["README_提交.md"])
            packager.sync_source_manifest(self.root, reference_zip=archive)
            manifest = json.loads((self.root / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
            entry = next(r for r in manifest["files"] if r["path"] == "README_提交.md")
            self.assertEqual(entry["manifest_updates"][-1]["reference_zip_sha256"], workflow.sha(archive))
            workflow.check_manifest(self.root)

    def test_sync_refuses_duplicate_entries_before_indexing(self):
        path = self.root / "PACKAGE_MANIFEST.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["files"].append(dict(manifest["files"][0]))
        path.write_text(json.dumps(manifest), encoding="utf-8")
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "duplicate manifest paths"):
            packager.sync_source_manifest(self.root)
        self.assertEqual(path.read_bytes(), before)

    def test_wrong_size_fails_before_manifest_replacement(self):
        path = self.root / "PACKAGE_MANIFEST.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["files"][0]["bytes"] += 1
        path.write_text(json.dumps(manifest), encoding="utf-8")
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "byte count mismatch"):
            packager.sync_source_manifest(self.root)
        self.assertEqual(path.read_bytes(), before)

    def test_atomic_serialization_failure_preserves_manifest(self):
        path = self.root / "PACKAGE_MANIFEST.json"
        before = path.read_bytes()
        with self.assertRaises(TypeError):
            packager._write_manifest(path, {"not_serializable": object()})
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(list(self.root.glob(".manifest-*.tmp")))


if __name__ == "__main__":
    unittest.main()
