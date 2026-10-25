"""Exercise the real package check on tiny saved-vector fixtures, no long eigensolve."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import test_methodology as T
from scripts import i03_verify_package as V
from scripts import submission_workflow as W
from scripts import update_scientific_docs as D
from scripts import evidence_stats as E
from turing_rd import Geometry, icosphere
from turing_rd.result_io import file_sha256

ROOT = Path(__file__).resolve().parents[1]


class StrictBestMatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        T.SavedSpectralEvidenceTests.setUpClass()

    def setUp(self):
        self.fixture = T.SavedSpectralEvidenceTests('test_saved_vectors_independently_recompute_and_pass')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root; self.g = self.fixture.g
        self.package = self.fixture.package_fixture()
        self.out = self.package / 'evidence/methodology/best_match_spectrum'
        shutil.copytree(self.fixture.out, self.out)
        self.cert = json.loads((self.out / 'certificate.json').read_text())

    def save(self):
        (self.out / 'certificate.json').write_text(json.dumps(self.cert), encoding='utf-8')

    def verify_checks(self):
        with patch.object(V, 'CHECKS', []):
            V.verify_best_match_checks(self.package, self.g, W)
            return copy.deepcopy(V.CHECKS)

    def assert_spectrum_rejected(self):
        with self.assertRaises((ValueError, FileNotFoundError, KeyError)):
            W.verify_best_match(self.package, self.g)
        checks = self.verify_checks()
        self.assertTrue(checks[0]['passed'], checks)
        self.assertFalse(checks[1]['passed'], checks)
        self.assertEqual(checks[1]['status'], 'FAIL')

    def test_complete_evidence_passes_both_independent_checks(self):
        r = W.verify_best_match(self.package, self.g)
        self.assertEqual(r['status'], 'PASS')
        self.assertEqual(r['historical_spectrum_status'], 'insufficient_evidence')
        self.assertEqual(r['current_saved_vector_spectrum_status'], 'numerical_check_passed')
        self.assertEqual(r['decision']['checked_eigenpairs'], 40)
        self.assertTrue(W.current_saved_vectors_passed(r))
        self.assertTrue(all(c['passed'] for c in self.verify_checks()))

    def test_missing_certificate_cannot_fall_back_to_historical_summary(self):
        (self.out / 'certificate.json').unlink(); self.assert_spectrum_rejected()

    def test_missing_one_vector_file_rejects(self):
        (self.out / 'eigenpairs_1.npz').unlink(); self.assert_spectrum_rejected()

    def test_missing_one_configuration_rejects(self):
        self.cert['configs'].pop(); self.save(); self.assert_spectrum_rejected()

    def test_wrong_mat_hash_rejects(self):
        self.cert['input']['mat_sha256'] = '0' * 64; self.save(); self.assert_spectrum_rejected()

    def test_tampered_pair_residual_rejects_even_if_report_hash_is_updated(self):
        self.cert['configs'][0]['pairs'][0]['residual_absolute_M'] = 1e-5
        report = self.out / self.cert['artifacts'][0]['report_file']
        report.write_text(json.dumps(self.cert['configs'][0]), encoding='utf-8')
        self.cert['artifacts'][0]['report_sha256'] = file_sha256(report)
        self.save(); self.assert_spectrum_rejected()

    def test_changed_vectors_with_updated_hash_still_reject(self):
        path = self.out / 'eigenpairs_0.npz'
        with np.load(path, allow_pickle=False) as a:
            values = a['eigenvalues']; vectors = a['eigenvectors'].copy()
        vectors[0, 0] += .1
        np.savez_compressed(path, eigenvalues=values, eigenvectors=vectors)
        self.cert['artifacts'][0]['vectors_sha256'] = file_sha256(path)
        self.save(); self.assert_spectrum_rejected()

    def test_failed_spectral_decision_rejects(self):
        self.cert['decision']['numerical_check_passed'] = False
        self.save(); self.assert_spectrum_rejected()

    def test_only_historical_summary_does_not_pass_current_protocol(self):
        (self.out / 'certificate.json').unlink()
        r = W.verify_best_match(self.package, self.g, historical_compatibility=True)
        self.assertEqual(r['status'], 'LIMITED_EVIDENCE')
        self.assertEqual(r['historical_spectrum_status'], 'insufficient_evidence')
        self.assertEqual(r['current_saved_vector_spectrum_status'], 'missing_evidence')
        self.assertFalse(W.current_saved_vectors_passed(r))
        self.assert_spectrum_rejected()

    def test_explicit_historical_package_check_is_limited_not_pass(self):
        (self.out / 'certificate.json').unlink()
        with patch.object(V, 'CHECKS', []):
            V.verify_best_match_checks(self.package, self.g, W, historical_compatibility=True)
            self.assertFalse(V.CHECKS[-1]['passed'])
            self.assertEqual(V.CHECKS[-1]['status'], 'LIMITED_EVIDENCE')
            args = types.SimpleNamespace(out=self.root/'review', zip=self.root/'unused.zip', report=None,
                                         historical_compatibility=True)
            work = self.root/'temporary_extract'; work.mkdir()
            with patch.object(V.shutil, 'rmtree'):
                self.assertEqual(V.write_report(args, self.package, work, ref=None), 2)
            report = json.loads((args.out/'acceptance_report.json').read_text())
            self.assertEqual(report['status'], 'LIMITED_EVIDENCE')
            self.assertFalse(report['checks'][-1]['passed'])

    def test_metadata_only_result_never_satisfies_current_saved_vector_predicate(self):
        r = W.verify_best_match(self.package, self.g)
        for key, value in [('eigenpair_residuals_recomputed_from_saved_vectors', False),
                           ('current_saved_vector_spectrum_status', 'metadata_only_not_current_acceptance'),
                           ('saved_spectral_metadata_verified', False)]:
            changed = copy.deepcopy(r); changed[key] = value
            self.assertFalse(W.current_saved_vectors_passed(changed))

    def test_missing_best_match_mat_fails_both_checks(self):
        (self.package/'task3_inversion/best_match/solution_best_match.mat').unlink()
        self.assertFalse(any(c['passed'] for c in self.verify_checks()))

    def test_cotan_operator_cannot_certify_robust_result(self):
        g = Geometry(icosphere(1), laplacian='cotan', cache_dir=None)
        with self.assertRaisesRegex(ValueError, 'robust operator'):
            W.verify_best_match_steady(self.package, g)


class PublicReleaseGeneratorTests(unittest.TestCase):
    def test_new_release_paths_and_execution_record_replace_previous_current_paths(self):
        q = E.load(ROOT/'submission/evidence/generalization/QUALITY_BREAKDOWN.json')
        release = {'report':'report/stage8_fixture/new.pdf', 'zip':'drafts/stage8_fixture/new.zip',
                   'unit_tests':'evidence/releases/stage8_fixture/unit_tests.json'}
        docs = D.render_documents(ROOT, q, 47, release)
        self.assertIn('47 项单元测试', docs['submission/VALIDATION.md'])
        self.assertIn(release['unit_tests'], docs['submission/VALIDATION.md'])
        self.assertIn('LIMITED_EVIDENCE', docs['submission/VALIDATION.md'])
        self.assertIn(release['zip'], docs['README.md'])
        self.assertNotIn('C:\\Users\\18650', docs['HANDOVER.md'])
        self.assertNotIn('C:\\Windows\\Fonts', docs['HANDOVER.md'])
        self.assertIn('不改写历史记录或 Git 历史', docs['HANDOVER.md'])
        self.assertIn('不在提交内预写自身最终 SHA', docs['HANDOVER.md'])


if __name__ == '__main__':
    unittest.main()
