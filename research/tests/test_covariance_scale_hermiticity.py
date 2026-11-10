"""Scale-relative covariance validation: fail closed on relative non-Hermiticity.

CPU-only research input gate. No real Munich receiver or calibration samples.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import whitened_calibration_bridge as bridge


TAU = 0.001
RHO = 0.001
D = np.diag([1., 0.5]).astype(complex)
S0 = np.array([[1., 0.25 + 0.1j],
               [0.25 - 0.1j, 1.5]], dtype=complex)
H0 = np.linalg.cholesky(S0) @ D


def cert(scale, covariance=None):
    H = np.sqrt(scale) * H0
    S = scale * S0 if covariance is None else covariance
    return bridge.gram_radius_from_calibration(H, S, TAU, RHO)


class CovarianceScaleHermiticityTests(unittest.TestCase):
    def test_bad_tiny_covariance_was_false_accepted_by_absolute_check(self):
        H = 1e-10 * D
        bad = 1e-20*np.eye(2, dtype=complex) + 1j*1e-14*np.eye(2)
        absolute_defect = float(np.max(np.abs(bad - bad.conj().T)))
        relative_defect = absolute_defect / float(np.max(np.abs(bad)))
        self.assertLess(absolute_defect, 1e-12)   # Defect in old gate
        self.assertGreater(relative_defect, 1.)
        with self.assertRaisesRegex(ValueError, "relative skew"):
            bridge.gram_radius_from_calibration(H, bad, TAU, RHO)

    def test_test_only_oracle_does_not_erase_bad_covariance(self):
        H = 1e-10 * D
        bad = 1e-20*np.eye(2, dtype=complex) + 1j*1e-14*np.eye(2)
        with self.assertRaisesRegex(ValueError, "relative skew"):
            bridge.oracle_relative_errors(H, bad, H, bad)

    def test_valid_scalar_GL_frames_have_same_nominal_gram_and_eligibility(self):
        baseline = cert(1.)
        self.assertTrue(baseline["ready_for_existing_ser_certificate"])
        self.assertEqual(bridge.HERMITIAN_RELATIVE_TOL, 1e-12)
        for scale in (1e-20, 1., 1e20):
            with self.subTest(scale=scale):
                result = cert(scale)
                self.assertEqual(result["status"], baseline["status"])
                self.assertTrue(result["ready_for_existing_ser_certificate"])
                self.assertLess(
                    abs(result["effective_gram_operator_norm_error_bound"]
                        - baseline["effective_gram_operator_norm_error_bound"]), 1e-11
                )
                self.assertLess(
                    abs(result["nominal_gram_min_eigenvalue"]
                        - baseline["nominal_gram_min_eigenvalue"]), 1e-11
                )
                self.assertLess(
                    abs(result["nominal_covariance_condition"]
                        - baseline["nominal_covariance_condition"]), 1e-11
                )

    def test_small_relative_skew_is_symmetrized_consistently_at_all_scales(self):
        for scale in (1e-20, 1., 1e20):
            with self.subTest(scale=scale):
                reference = cert(scale)
                near = (scale*S0).copy()
                near[0,0] += 1j * (3e-14*scale)
                absolute_skew = np.max(np.abs(near - near.conj().T))
                relative_skew = absolute_skew / np.max(np.abs(near))
                self.assertLess(relative_skew, bridge.HERMITIAN_RELATIVE_TOL)
                if scale == 1e20:
                    self.assertGreater(absolute_skew, 1e-12)  # Old false rejection
                result = cert(scale, near)
                self.assertTrue(result["ready_for_existing_ser_certificate"])
                self.assertLess(
                    abs(result["effective_gram_operator_norm_error_bound"]
                        - reference["effective_gram_operator_norm_error_bound"]), 1e-11
                )

    def test_large_relative_skew_rejected_at_all_scales(self):
        for scale in (1e-20, 1., 1e20):
            with self.subTest(scale=scale):
                bad = (scale*S0).copy()
                bad[0,0] += 1j*(1e-8*scale)
                self.assertGreater(
                    np.max(np.abs(bad-bad.conj().T))/np.max(np.abs(bad)), 1e-9
                )
                with self.assertRaisesRegex(ValueError, "relative skew"):
                    cert(scale, bad)

    def test_extreme_hermitian_scale_avoids_avoidable_symmetrization_overflow(self):
        # Auxiliary numeric edge case: S+S^H would overflow near 1e308.
        scale = 1e308
        H = np.sqrt(scale) * D
        S = scale * np.eye(2, dtype=complex)
        result = bridge.gram_radius_from_calibration(H, S, TAU, RHO)
        self.assertTrue(result["ready_for_existing_ser_certificate"])
        self.assertLess(abs(result["nominal_gram_min_eigenvalue"]-.25),1e-12)

    def test_existing_condition_number_policy_still_fails_closed(self):
        self.assertEqual(bridge.COND_LIMIT, 1e10)
        H = D
        S = np.diag([1e11,1.]).astype(complex)
        with self.assertRaises(bridge.NumericalFrameConditionError):
            bridge.gram_radius_from_calibration(H,S,TAU,RHO)


if __name__ == "__main__":
    unittest.main()
