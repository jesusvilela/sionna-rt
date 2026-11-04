"""CPU-only tests of phase-frame covariance of dual-polarized codebooks.

Requires NumPy but not Sionna RT, Mitsuba, a GPU, or stochastic ray tracing.
These tests establish a representation identity, not a wireless performance gain.
"""

import unittest
import numpy as np


def null_form(x):
    """q(x) = det [[Re(x1), Im(x1)], [Re(x2), Im(x2)]]."""
    x = np.asarray(x, dtype=np.complex128)
    return np.imag(np.conj(x[..., 0]) * x[..., 1])


def rank_one_constellation():
    """Reproduce the published 16-point rank-1 construction from PR #82."""
    pts = []
    for m in range(8):
        v = np.array([np.cos(2 * np.pi * m / 8),
                      np.sin(2 * np.pi * m / 8)])
        for w in (np.array([1.0, 1.0]), np.array([1.0, -1.0])):
            X = np.outer(v, w)
            pts.append([X[0, 0] + 1j * X[0, 1],
                        X[1, 0] + 1j * X[1, 1]])
    pts = np.asarray(pts, dtype=np.complex128)
    return pts / np.sqrt(np.sum(np.abs(pts)**2, axis=1).mean())


def qpsk_squared():
    g = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    pts = np.array([[a, b] for a in g for b in g])
    return pts / np.sqrt(np.sum(np.abs(pts)**2, axis=1).mean())


def phase_frame(a, b):
    """Unitary input-basis transformation D = diag(exp(ia), exp(ib))."""
    return np.diag(np.exp(1j * np.array([a, b])))


def transform_codebook_and_channel(C, H, D):
    """Coordinates x' = D x and H' = H D^-1, for columns x."""
    return C @ D.T, H @ D.conj().T


def channel_outputs(C, H):
    return C @ H.T


def pairwise_squared_distances(Y):
    return np.sum(np.abs(Y[:, None, :] - Y[None, :, :])**2, axis=-1)


class PhaseFrameCovarianceTests(unittest.TestCase):
    def test_null_form_matches_real_determinant(self):
        rng = np.random.default_rng(21)
        for _ in range(100):
            x = rng.normal(size=2) + 1j * rng.normal(size=2)
            X = np.stack((x.real, x.imag), axis=-1)
            self.assertAlmostEqual(null_form(x), np.linalg.det(X), places=12)

    def test_exact_counterexample_of_null_cone_invariance(self):
        x = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2)
        D = phase_frame(0.0, np.pi / 2)
        self.assertAlmostEqual(null_form(x), 0.0, places=12)
        self.assertAlmostEqual(null_form(D @ x), 0.5, places=12)

    def test_rank_one_codebook_loses_null_condition_under_relative_phase(self):
        C = rank_one_constellation()
        self.assertEqual(C.shape, (16, 2))
        np.testing.assert_allclose(null_form(C), 0, atol=1e-14)
        D = phase_frame(0.0, np.pi / 2)
        C_new = C @ D.T
        self.assertGreater(np.max(np.abs(null_form(C_new))), 0.2)
        np.testing.assert_allclose(np.sum(np.abs(C_new)**2, axis=1),
                                   np.sum(np.abs(C)**2, axis=1), atol=1e-14)

    def test_coherence_pair_transports_as_rotation(self):
        rng = np.random.default_rng(72)
        x = rng.normal(size=2) + 1j * rng.normal(size=2)
        a, b = 0.17, 0.83
        D = phase_frame(a, b)
        z = np.conj(x[0]) * x[1]
        z_new = np.conj((D @ x)[0]) * (D @ x)[1]
        np.testing.assert_allclose(z_new, np.exp(1j * (b-a)) * z, atol=1e-14)
        np.testing.assert_allclose(np.abs(z_new), np.abs(z), atol=1e-14)

    def test_coherency_matrix_is_covariant_not_component_invariant(self):
        rng = np.random.default_rng(100)
        x = rng.normal(size=2) + 1j * rng.normal(size=2)
        D = phase_frame(-0.2, 0.9)
        Q = np.outer(x, np.conj(x))
        x_new = D @ x
        Q_new = np.outer(x_new, np.conj(x_new))
        np.testing.assert_allclose(Q_new, D @ Q @ D.conj().T, atol=1e-14)

    def test_received_points_distances_and_ml_decisions_are_frame_invariant(self):
        rng = np.random.default_rng(20261104)
        for C in (rank_one_constellation(), qpsk_squared()):
            for a, b in ((0.0, 0.0), (0.0, np.pi/2),
                         (0.3, 1.1), (np.pi/3, -np.pi/7)):
                H = (rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
                D = phase_frame(a, b)
                C_new, H_new = transform_codebook_and_channel(C, H, D)
                Y = channel_outputs(C, H)
                Y_new = channel_outputs(C_new, H_new)
                np.testing.assert_allclose(Y_new, Y, rtol=1e-13, atol=1e-13)
                np.testing.assert_allclose(pairwise_squared_distances(Y_new),
                                           pairwise_squared_distances(Y),
                                           rtol=1e-13, atol=1e-13)
                transmitted = rng.integers(len(C), size=500)
                noise = 0.1 * (rng.normal(size=(500, 2)) +
                               1j * rng.normal(size=(500, 2)))
                received = Y[transmitted] + noise
                cost = np.sum(np.abs(received[:, None, :] - Y[None, :, :])**2,
                              axis=-1)
                cost_new = np.sum(np.abs(received[:, None, :] - Y_new[None, :, :])**2,
                                  axis=-1)
                np.testing.assert_array_equal(np.argmin(cost, axis=1),
                                              np.argmin(cost_new, axis=1))

    def test_symbol_only_rotation_is_a_physical_change_not_a_frame_change(self):
        x = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2)
        H = np.array([[1, 0], [0, 1]], dtype=np.complex128)
        D = phase_frame(0, np.pi/2)
        self.assertGreater(np.linalg.norm(H @ (D @ x) - H @ x), 0.5)
        H_new = H @ D.conj().T
        np.testing.assert_allclose(H_new @ (D @ x), H @ x, atol=1e-14)

    def test_degenerate_component_is_null_in_all_phase_frames(self):
        for x in (np.array([1, 0], dtype=complex),
                  np.array([0, 1], dtype=complex)):
            for a, b in ((0, np.pi/2), (0.31, 1.78)):
                self.assertAlmostEqual(null_form(phase_frame(a, b) @ x),
                                       0.0, places=12)


if __name__ == "__main__":
    unittest.main()
