"""CPU-only geometric controls for PR #82: Stokes, Hopf, and a (2,2) null cone.

These are analytical identities and fixed-codebook diagnostics, not evidence
of a causal SER benefit or of a previously unknown mathematical theorem.
"""

import unittest
import numpy as np
from test_phase_frame_covariance import (
    null_form, rank_one_constellation, qpsk_squared,
    channel_outputs, pairwise_squared_distances,
)

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = (SX, SY, SZ)


def stokes(x):
    x = np.asarray(x, dtype=complex)
    z = np.conj(x[..., 0]) * x[..., 1]
    return np.stack((2 * z.real, 2 * z.imag,
                     np.abs(x[..., 0])**2 - np.abs(x[..., 1])**2), axis=-1)


def clifford_coordinates(x):
    """Real orthogonal change R^4 -> R^4, returned as two complex coordinates."""
    a, b, c, d = x[0].real, x[0].imag, x[1].real, x[1].imag
    return ((a + d + 1j*(b-c))/np.sqrt(2),
            (a - d + 1j*(b+c))/np.sqrt(2))


def stokes_rotation(U):
    """Adjoint U(2) action on traceless Hermitian 2x2 matrices."""
    return np.array([[0.5 * np.trace(A @ U @ B @ U.conj().T).real
                      for B in PAULI] for A in PAULI])


def random_unitary(rng):
    Z = rng.normal(size=(2,2)) + 1j*rng.normal(size=(2,2))
    Q, R = np.linalg.qr(Z)
    return Q @ np.diag(np.exp(-1j*np.angle(np.diag(R))))


def min_codeword_distance(C):
    d2 = pairwise_squared_distances(C)
    return np.sqrt(np.min(d2[np.triu_indices(len(C), 1)]))


def stokes_histogram(C):
    return np.unique(np.round(stokes(C), decimals=10), axis=0, return_counts=True)


class StokesHopfGeometryTests(unittest.TestCase):
    def test_stokes_has_hopf_sphere_identity(self):
        rng=np.random.default_rng(1001)
        for _ in range(250):
            x = rng.normal(size=2) + 1j*rng.normal(size=2)
            s = stokes(x)
            p = np.vdot(x, x).real
            self.assertAlmostEqual(float(np.dot(s, s)), p*p, places=10)
            self.assertAlmostEqual(float(s[1]), 2*null_form(x), places=12)
        np.testing.assert_allclose(stokes([0j,0j]), [0,0,0], atol=1e-15)

    def test_signature_two_plus_two_and_clifford_coordinates(self):
        # q = ad-bc = v^T Q v, for v=(a,b,c,d).
        Q = np.zeros((4,4))
        Q[0,3] = Q[3,0] = 0.5
        Q[1,2] = Q[2,1] = -0.5
        np.testing.assert_allclose(np.linalg.eigvalsh(Q), [-0.5,-0.5,0.5,0.5])
        rng = np.random.default_rng(2002)
        for _ in range(250):
            x = rng.normal(size=2) + 1j*rng.normal(size=2)
            u, v = clifford_coordinates(x)
            p = np.vdot(x,x).real
            self.assertAlmostEqual(abs(u)**2 + abs(v)**2, p, places=11)
            self.assertAlmostEqual((abs(u)**2-abs(v)**2)/2,
                                   null_form(x), places=11)
        for x in rank_one_constellation():
            u, v = clifford_coordinates(x)
            self.assertAlmostEqual(abs(u)**2, 0.5, places=12)
            self.assertAlmostEqual(abs(v)**2, 0.5, places=12)

    def test_stokes_transports_under_arbitrary_unitary(self):
        rng=np.random.default_rng(3003)
        for _ in range(60):
            U = random_unitary(rng)
            R = stokes_rotation(U)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-13)
            self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=12)
            x=rng.normal(size=2)+1j*rng.normal(size=2)
            np.testing.assert_allclose(stokes(U@x), R@stokes(x), atol=1e-12)
            n=np.array([0.0, 1.0, 0.0])
            self.assertAlmostEqual(float(np.dot(R@n, stokes(U@x))),
                                   float(np.dot(n, stokes(x))), places=11)

    def test_real_orthogonal_normalizer_preserves_null_form(self):
        rng=np.random.default_rng(4004)
        for ang in np.linspace(0,2*np.pi,11):
            R0=np.array([[np.cos(ang), -np.sin(ang)],
                         [np.sin(ang),  np.cos(ang)]])
            for reflection in (np.eye(2),np.diag([1,-1])):
                A=R0@reflection
                U=np.exp(0.27j)*A
                for _ in range(8):
                    x=rng.normal(size=2)+1j*rng.normal(size=2)
                    self.assertAlmostEqual(float(null_form(U@x)),
                                           float(np.linalg.det(A)*null_form(x)),
                                           places=12)

    def test_non_normalizer_moves_null_cone_but_preserves_moving_axis(self):
        D=np.diag([1,1j])
        x=np.array([1,1],dtype=complex)/np.sqrt(2)
        self.assertAlmostEqual(float(null_form(x)),0.0,places=12)
        self.assertAlmostEqual(float(null_form(D@x)),0.5,places=12)
        R=stokes_rotation(D)
        n=np.array([0.,1.,0.])
        self.assertAlmostEqual(float(np.dot(R@n,stokes(D@x))),0.0,places=12)
        self.assertGreater(abs(float(np.dot(n,stokes(D@x)))),0.9)

    def test_both_codebooks_project_to_four_equally_weighted_stokes_points(self):
        C0=rank_one_constellation()
        C1=qpsk_squared()
        U=(np.eye(2,dtype=complex)-1j*SX)/np.sqrt(2)
        S0,n0=stokes_histogram(C0@U.T)
        S1,n1=stokes_histogram(C1)
        np.testing.assert_allclose(S0,S1,atol=1e-12)
        np.testing.assert_array_equal(n0,[4,4,4,4])
        np.testing.assert_array_equal(n1,[4,4,4,4])
        np.testing.assert_allclose(stokes(C0)[:,1],0,atol=1e-14)
        np.testing.assert_allclose(stokes(C1)[:,2],0,atol=1e-14)
        # Four equally spaced points in the Poincare sphere DO NOT fix their
        # complex Hopf-fiber phases, and therefore do not fix codeword distances.
        self.assertAlmostEqual(min_codeword_distance(C0),
                               np.sqrt(2-np.sqrt(2)),places=12)
        self.assertAlmostEqual(min_codeword_distance(C1),1.0,places=12)
        self.assertNotAlmostEqual(min_codeword_distance(C0),
                                  min_codeword_distance(C1),places=3)

    def test_phase_sections_differ_after_base_circle_alignment(self):
        C0=rank_one_constellation()
        C1=qpsk_squared()
        U=(np.eye(2,dtype=complex)-1j*SX)/np.sqrt(2)
        # Under this EXPLICITLY CHOSEN U, the polarization base points align.
        # First-channel phases after 4x amplification encode the relative
        # Hopf-fiber section; this marker is NOT a coordinate-free invariant.
        marker0=np.exp(4j*np.angle((C0@U.T)[:,0]))
        marker1=np.exp(4j*np.angle(C1[:,0]))
        self.assertEqual(int(np.sum(np.isclose(marker0.real,+1,atol=1e-12))),8)
        self.assertEqual(int(np.sum(np.isclose(marker0.real,-1,atol=1e-12))),8)
        np.testing.assert_allclose(marker1,-np.ones(len(C1)),atol=1e-12)

    def test_full_u2_frame_change_preserves_received_codeword_geometry(self):
        rng=np.random.default_rng(5005)
        for C in (rank_one_constellation(),qpsk_squared()):
            H=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))
            Y=channel_outputs(C,H)
            for _ in range(30):
                U=random_unitary(rng)
                Cp=C@U.T
                Hp=H@U.conj().T
                np.testing.assert_allclose(channel_outputs(Cp,Hp),Y,
                                           atol=1e-12,rtol=1e-12)
                np.testing.assert_allclose(pairwise_squared_distances(
                                           channel_outputs(Cp,Hp)),
                                           pairwise_squared_distances(Y),
                                           atol=1e-11,rtol=1e-11)

if __name__ == '__main__':
    unittest.main()
