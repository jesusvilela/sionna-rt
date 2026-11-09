"""Adversarial sharpness checks: trace-preserving and unrestricted Gram balls.

The 75-cell source and codebooks are frozen. No new propagation data.
Worst-case constructions are classical 2x2 Hermitian operator norm examples.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import norm

EXPERIMENTS = Path(__file__).resolve().parents[1] / 'experiments'
sys.path.insert(0, str(EXPERIMENTS))
import gram_uncertainty_certificate as m
from gram_decision_sweep import sweep

PAIRS = ((0, 1), (0, 7), (2, 11), (4, 15))
FRAME_CELLS = (0, 18, 37, 55, 74)
PORTS = (np.diag([1., 0.]), np.diag([0., 1.]))


def source_gram(case):
    a = case['channel']
    z = complex(a['cross_correlation_real'], a['cross_correlation_imag'])
    return np.array([[a['port_gain_split'][0], z],
                     [z.conjugate(), a['port_gain_split'][1]]], dtype=complex)


def direct_output_probabilities(G, C):
    """Direct 2x2 Hermitian-square-root channel, independent of Gram einsum."""
    w, V = np.linalg.eigh(G)
    if min(w) <= 0:
        raise ValueError('independently built channel must be full rank')
    H = (V * np.sqrt(w)) @ V.conj().T
    Y = C @ H.T
    d2 = np.sum(abs(Y[:, None, :] - Y[None, :, :])**2, axis=-1)
    return norm.sf(np.sqrt(d2 / (2*m.N0)))


class AdversarialSharpnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = sweep()['cases']
        cls.codebooks = m.books()
        if len(cls.cases) != 75 or len(cls.codebooks) != 3:
            raise AssertionError('frozen grid/cardinality drift')

    def test_all_1800_traceless_extremizers_reach_individual_pep_bounds(self):
        count = 0
        max_err = 0.
        for case in self.cases:
            G = source_gram(case)
            mineig = np.linalg.eigvalsh(G)[0]
            eps = min(1e-3, mineig / 4)
            for C in self.codebooks.values():
                lower, upper = m.event_brackets(G, C, eps)
                for i, j in PAIRS:
                    d = C[i] - C[j]
                    d2 = float(np.vdot(d, d).real)
                    self.assertGreater(d2, 0)
                    v = d / np.sqrt(d2)
                    S = 2 * np.outer(v, v.conj()) - np.eye(2)
                    for sign in (1., -1.):
                        E = sign * eps * S
                        trueG = G + E
                        self.assertGreater(np.linalg.eigvalsh(trueG)[0], 0)
                        self.assertLess(abs(np.trace(E)), 1e-12)
                        self.assertLess(abs(np.linalg.norm(E, 2) - eps), 1e-12)
                        self.assertLess(abs(np.trace(trueG) - np.trace(G)), 1e-12)
                        w, V = np.linalg.eigh(trueG)
                        H = (V * np.sqrt(w)) @ V.conj().T
                        actual_q = float(np.linalg.norm(H @ d)**2)
                        nominal_q = float(np.vdot(d, G @ d).real)
                        target_q = nominal_q + sign * eps * d2
                        self.assertLess(abs(actual_q - target_q), 1e-11)
                        actual_p = float(norm.sf(np.sqrt(actual_q / (2*m.N0))))
                        target_p = float(lower[i, j] if sign == 1 else upper[i, j])
                        max_err = max(max_err, abs(actual_p - target_p))
                        self.assertLess(abs(actual_p - target_p), 1e-11)
                        count += 1
        self.assertEqual(count, 1800)
        self.assertLess(max_err, 1e-11)

    def test_unrestricted_gram_ball_joint_extrema_all_225_books(self):
        # Unlike the trace-preserving case, +/- eps*I hits *every* pair endpoint.
        comparisons = 0
        for case in self.cases:
            G = source_gram(case)
            eps = min(1e-3, np.linalg.eigvalsh(G)[0] / 4)
            for C in self.codebooks.values():
                bounds = m.robust_interval(G, C, eps)
                low_p = direct_output_probabilities(G + eps * np.eye(2), C)
                high_p = direct_output_probabilities(G - eps * np.eye(2), C)
                np.fill_diagonal(low_p, 0.)
                np.fill_diagonal(high_p, 0.)
                min_lower = float(np.mean(np.max(low_p, axis=1)))
                max_upper = float(np.mean(np.sum(high_p, axis=1)))
                self.assertLess(abs(min_lower - bounds['L_guaranteed']), 1e-11)
                self.assertLess(abs(max_upper - bounds['U_guaranteed']), 1e-11)
                comparisons += 1
        self.assertEqual(comparisons, 225)

    def test_trace_preserving_single_beam_sharpness_vs_multibeam(self):
        for case in self.cases:
            G = source_gram(case)
            eps = min(1e-3, np.linalg.eigvalsh(G)[0] / 4)
            C = self.codebooks['circular_16psk']
            v = C[0] / np.linalg.norm(C[0])
            E = eps * (2 * np.outer(v, v.conj()) - np.eye(2))
            low, high = m.event_brackets(G, C, eps)
            for sign, target in ((1., low), (-1., high)):
                actual = direct_output_probabilities(G + sign * E, C)
                mask = ~np.eye(16, dtype=bool)
                self.assertLess(float(np.max(abs(actual[mask] - target[mask]))), 1e-11)
        # Independent differences, therefore no *traceless* fixed E can be
        # extremal for every direction in either multi-beam book.
        expected_cosines = {'rank_one': .75, 'qpsk_squared': 2./3.,
                            'circular_16psk': 1.}
        for name, C in self.codebooks.items():
            x, y = C[0]-C[1], C[0]-C[7]
            cosine2 = float(abs(np.vdot(x, y))**2 /
                            (np.vdot(x, x).real * np.vdot(y, y).real))
            self.assertAlmostEqual(cosine2, expected_cosines[name], places=12)

    def test_two_moving_input_frames_and_nonunitary_noise_covariance(self):
        U1 = m.su2(31, .71)
        U2 = m.su2(17, -.49) @ U1
        W = np.array([[1.2+.15j, .27-.13j],
                      [.08+.22j, .82-.09j]], dtype=complex)
        L = np.linalg.cholesky(W @ W.conj().T)
        count = 0
        for index in FRAME_CELLS:
            G = source_gram(self.cases[index])
            eps = min(1e-3, np.linalg.eigvalsh(G)[0] / 4)
            for C in self.codebooks.values():
                d = C[0] - C[1]
                v = d / np.linalg.norm(d)
                E = eps * (2*np.outer(v, v.conj()) - np.eye(2))
                for sign in (1., -1.):
                    Gtrue = G + sign * E
                    w, V = np.linalg.eigh(Gtrue)
                    H = (V*np.sqrt(w)) @ V.conj().T
                    baseline = float(np.linalg.norm(H @ d)**2)
                    for U in (U1, U2):
                        Hnew = W @ H @ U.conj().T
                        dnew = U @ d
                        whitened_q = float(np.linalg.norm(np.linalg.solve(L, Hnew @ dnew))**2)
                        self.assertLess(abs(baseline - whitened_q), 1e-11)
                        for P in PORTS:
                            Pnew = U @ P @ U.conj().T
                            original = np.einsum('mi,ij,mj->m', C.conj(), P, C).real
                            moved = C @ U.T
                            transformed = np.einsum('mi,ij,mj->m', moved.conj(), Pnew, moved).real
                            self.assertLess(float(np.max(abs(original-transformed))), 1e-11)
                        count += 1
        self.assertEqual(count, 60)
        # Deliberately incorrect control: rotate Gram without rotating codewords.
        G = source_gram(self.cases[37])
        C = self.codebooks['rank_one']
        d = C[0] - C[1]
        correct = float(np.vdot(d, G @ d).real)
        incorrect = float(np.vdot(d, (U1 @ G @ U1.conj().T) @ d).real)
        self.assertGreater(abs(correct-incorrect), 1e-4)


if __name__ == '__main__':
    unittest.main()
