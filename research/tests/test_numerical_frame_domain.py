"""Adversarial CPU checks: analytic GL covariance vs raw numeric acceptance.

No Munich propagation, no model fitting; one already-known Gaussian detector.
The representation cutoff stays at its previous 1e10 safety threshold.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
import whitened_calibration_bridge as bridge

H0 = np.diag([1., 0.5]).astype(np.complex128)
S0 = np.eye(2, dtype=np.complex128)
TAU = RHO = 0.001


def mat_sqrt(A, inverse=False):
    w, V = np.linalg.eigh(A)
    if w.min() <= 0:
        raise ValueError('positive definite input required')
    factor = w**(-.5 if inverse else .5)
    return (V * factor) @ V.conj().T


class NumericalFrameDomainTests(unittest.TestCase):
    def test_GL_transitive_on_hermitian_positive_covariances(self):
        S1 = np.array([[2., .1+.3j], [.1-.3j, 1.3]], complex)
        S2 = np.array([[.7, -.13+.2j], [-.13-.2j, 3.]], complex)
        W = mat_sqrt(S2) @ mat_sqrt(S1, inverse=True)
        np.testing.assert_allclose(W @ S1 @ W.conj().T, S2, rtol=0, atol=2e-14)
        self.assertGreater(abs(np.linalg.det(W)), 1e-12)

    def test_stable_nonunitary_frame_preserves_actual_eligibility_and_budget(self):
        original = bridge.gram_radius_from_calibration(H0, S0, TAU, RHO)
        self.assertTrue(original['ready_for_existing_ser_certificate'])
        self.assertEqual(original['numerical_frame_domain_result'],
                         'accepted_raw_representation')
        self.assertEqual(bridge.COND_LIMIT, 1e10)
        U = bridge._frame_su2(23., .38)
        W = np.diag([1e3, 1.]).astype(complex)
        Hp = W @ H0 @ U.conj().T
        Sp = W @ S0 @ W.conj().T
        moved = bridge.gram_radius_from_calibration(Hp, Sp, TAU, RHO)
        self.assertTrue(moved['ready_for_existing_ser_certificate'])
        self.assertEqual(moved['status'], original['status'])
        self.assertAlmostEqual(moved['nominal_covariance_condition'], 1e6, places=4)
        self.assertLess(abs(moved['effective_gram_operator_norm_error_bound']
                            -original['effective_gram_operator_norm_error_bound']), 1e-12)
        G0 = H0.conj().T @ np.linalg.solve(S0, H0)
        Gp = Hp.conj().T @ np.linalg.solve(Sp, Hp)
        np.testing.assert_allclose(Gp, U @ G0 @ U.conj().T, atol=1e-12, rtol=0)

    def test_extreme_GL_frame_rejected_as_numeric_not_physical_failure(self):
        original = bridge.gram_radius_from_calibration(H0, S0, TAU, RHO)
        self.assertTrue(original['ready_for_existing_ser_certificate'])
        W = np.diag([1e6, 1.]).astype(complex)
        Hp = W @ H0
        Sp = W @ S0 @ W.conj().T
        self.assertAlmostEqual(np.linalg.cond(Sp), 1e12, delta=1.)
        G0 = H0.conj().T @ np.linalg.solve(S0, H0)
        Gp = Hp.conj().T @ np.linalg.solve(Sp, Hp)
        np.testing.assert_allclose(Gp, G0, atol=1e-12, rtol=0)
        with self.assertRaises(bridge.NumericalFrameConditionError) as exc:
            bridge.gram_radius_from_calibration(Hp, Sp, TAU, RHO)
        err = exc.exception
        self.assertIsInstance(err, ValueError)
        self.assertEqual(err.reason_code, 'numerical_frame_domain_rejection')
        self.assertEqual(err.condition_limit, 1e10)
        self.assertAlmostEqual(err.observed_condition, 1e12, delta=1.)
        self.assertIn('re-express both channel and covariance', str(err))
        self.assertNotIn('abstain_positive_definiteness_gate', str(err))

    def test_pd_error_budget_abstention_remains_separate_and_fail_closed(self):
        wide = bridge.gram_radius_from_calibration(H0, S0, .5, .5)
        self.assertFalse(wide['ready_for_existing_ser_certificate'])
        self.assertEqual(wide['status'], 'abstain_positive_definiteness_gate')
        self.assertEqual(wide['numerical_frame_domain_result'],
                         'accepted_raw_representation')
        self.assertGreaterEqual(wide['effective_gram_operator_norm_error_bound'],
                                wide['nominal_gram_min_eigenvalue'])

    def test_no_raw_cutoff_relaxation_or_spurious_physical_provenance(self):
        self.assertEqual(bridge.COND_LIMIT, 1e10)
        C = np.diag([1e5, 1.]).astype(complex)
        near = bridge.gram_radius_from_calibration(C @ H0, C @ S0 @ C.conj().T,
                                                   TAU, RHO)
        self.assertTrue(near['ready_for_existing_ser_certificate'])
        self.assertLessEqual(near['nominal_covariance_condition'], bridge.COND_LIMIT)
        self.assertIn('assumptions_only_no_measured_calibration',
                      near['evidence_status'])
        self.assertIn('condition <= 1e10', near['numerical_frame_acceptance'])


if __name__ == '__main__':
    unittest.main()
