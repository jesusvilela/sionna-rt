"""Subnormal complex covariance whitening: independent reference and controls.

Research helper only; no physical receiver measurements.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mpmath as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import whitened_calibration_bridge as bridge

S0 = np.array([[1., .25 + .1j], [.25 - .1j, 1.5]], dtype=complex)
H0 = np.linalg.cholesky(S0) @ np.diag([1., .5]).astype(complex)


def direct_high_precision_gram(H, S):
    """Independent mpmath inversion on ACTUAL complex128 inputs, not ideal floats."""
    with mp.workdps(100):
        def c(x):
            return mp.mpc(mp.mpf(float(x.real)), mp.mpf(float(x.imag)))
        sm = mp.matrix([[c(S[i, j]) for j in range(2)] for i in range(2)])
        hm = mp.matrix([[c(H[i, j]) for j in range(2)] for i in range(2)])
        gm = hm.transpose_conj() * sm**-1 * hm
        return np.array([[complex(gm[i, j]) for j in range(2)] for i in range(2)])


class SubnormalComplexWhiteningTests(unittest.TestCase):
    def test_original_complex_division_breaks_but_scaled_whitening_survives(self):
        scale = 1e-309
        H, S = np.sqrt(scale)*H0, scale*S0
        with np.errstate(all="ignore"):
            naive = S / np.max(np.abs(S))
        self.assertFalse(np.isfinite(naive).all())
        result = bridge.gram_radius_from_calibration(H, S, .001, .001)
        self.assertTrue(result["ready_for_existing_ser_certificate"])
        self.assertLess(abs(result["nominal_gram_min_eigenvalue"] - .25), 1e-11)
        self.assertEqual(bridge.COND_LIMIT, 1e10)

    def test_independent_100_digit_actual_input_oracle_all_scales(self):
        for scale in (1., 1e-20, 1e20, 1e-200, 1e-300, 1e-309, 1e-320,
                      np.nextafter(0., 1.)):
            with self.subTest(scale=scale):
                H, S = np.sqrt(scale)*H0, scale*S0
                Z, L, G, eigmin, cond = bridge._nominal_whitened(H, S)
                self.assertTrue(np.isfinite(G).all())
                oracle = direct_high_precision_gram(H, S)
                np.testing.assert_allclose(G, oracle, rtol=0, atol=1e-12)
                self.assertGreater(eigmin, 0)
                self.assertLess(cond, 2.)
                self.assertTrue(np.isfinite(Z).all())
                self.assertTrue(np.isfinite(L).all())

    def test_nominal_gram_remains_frame_stable_above_quantization_floor(self):
        base = bridge.gram_radius_from_calibration(H0, S0, .001, .001)
        for scale in (1e-200, 1e-300, 1e-309):
            with self.subTest(scale=scale):
                moved = bridge.gram_radius_from_calibration(
                    np.sqrt(scale)*H0, scale*S0, .001, .001
                )
                self.assertEqual(moved["status"], base["status"])
                self.assertLess(
                    abs(moved["nominal_gram_min_eigenvalue"]
                        - base["nominal_gram_min_eigenvalue"]), 1e-11
                )

    def test_relative_antihermitian_subnormal_rejected(self):
        scale = 1e-309
        S = scale*S0.copy()
        S[0, 0] += 1j*1e-8*scale
        with self.assertRaisesRegex(ValueError, "relative skew"):
            bridge.gram_radius_from_calibration(np.sqrt(scale)*H0, S, .001, .001)

    def test_existing_condition_policy_unmodified(self):
        with self.assertRaises(bridge.NumericalFrameConditionError):
            bridge.gram_radius_from_calibration(
                np.diag([1., .5]).astype(complex),
                np.diag([1e11, 1.]).astype(complex), .001, .001
            )

    def test_unrepresentable_whitening_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "whitening exceeds"):
            bridge.gram_radius_from_calibration(
                np.eye(2, dtype=complex)*1e308,
                np.eye(2, dtype=complex)*1e-309, .001, .001
            )

    def test_test_only_oracle_at_subnormal_scale(self):
        scale = 1e-309
        H, S = np.sqrt(scale)*H0, scale*S0
        # Use a real scalar channel perturbation; no invented calibration bound.
        actual_H = H*1.001
        a, tau, rho = bridge.oracle_relative_errors(H, S, actual_H, S)
        a0, tau0, rho0 = bridge.oracle_relative_errors(H0, S0, H0*1.001, S0)
        self.assertLess(abs(a-a0), 1e-11)
        self.assertLess(abs(tau-tau0), 1e-11)
        self.assertLess(abs(rho-rho0), 1e-11)


if __name__ == "__main__":
    unittest.main()
