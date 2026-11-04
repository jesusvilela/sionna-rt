"""CPU-only received-distance analysis for the two 16-point codebooks of PR #82.

This module proves nothing about ray-traced error rates. It checks an exact
finite-orbit identity and a diagonal-channel packing crossover against
independently enumerated codeword pairs, including counterexamples.

Run: python -m unittest discover -s research/tests -p 'test_*.py' -v
Requires: NumPy. No GPU, ray tracing, or Sionna runtime.
"""

import unittest
import numpy as np

from test_phase_frame_covariance import rank_one_constellation, qpsk_squared, null_form


MU4 = (1+0j, 1j, -1+0j, -1j)
A = 1.5 - np.sqrt(2.0)
CROSSOVER = 3.0 - 2.0 * np.sqrt(2.0)


def four_phase_representatives(C, tol=1e-10):
    """Partition a finite codebook into whole orbits under x -> i*x.

    Raise ValueError rather than silently discard a symbol if not closed.
    Return one representative of each orbit in original index order.
    """
    C = np.asarray(C, dtype=np.complex128)
    if C.ndim != 2 or C.shape[1] != 2:
        raise ValueError('codebook must have shape (number_of_symbols, 2)')
    used = set()
    reps = []
    for i, x in enumerate(C):
        if i in used:
            continue
        indices = []
        for phase in MU4:
            candidate = phase * x
            matches = np.flatnonzero(
                np.linalg.norm(C-candidate, axis=1) < tol)
            if len(matches) != 1:
                raise ValueError('not a distinct complete four-phase orbit')
            indices.append(int(matches[0]))
        if len(set(indices)) != 4:
            raise ValueError('orbit degeneracy (e.g. a zero symbol)')
        used.update(indices)
        reps.append(x)
    if len(used) != len(C):
        raise ValueError('incomplete partition')
    return np.asarray(reps)


def raw_received_d2(C, H):
    """Independent direct pairwise enumeration; returns real squared distances."""
    Y = np.asarray(C) @ np.asarray(H).T
    D = Y[:, None, :] - Y[None, :, :]
    return np.sum(np.abs(D)**2, axis=-1)


def brute_min_received_d2(C, H):
    d2 = raw_received_d2(C, H)
    return float(np.min(d2[np.triu_indices(len(C), 1)]))


def four_phase_min_received_d2(C, H):
    """Exact mu_4 orbit quotient using the channel Gram matrix G=H*H.

    If C is a union of distinct four-phase orbits with representatives u_k,
    d_min^2 is the minimum of:
      2*u_k*G*u_k,
      u_k*G*u_k + u_l*G*u_l - 2*max(|Re u_k*G*u_l|,|Im ...|).
    The first term covers pairs within a phase orbit, the second pairs in
    different orbits. The formula works also for singular channel matrices.
    """
    U = four_phase_representatives(C)
    H = np.asarray(H, dtype=np.complex128)
    G = H.conj().T @ H
    e = np.array([np.vdot(u, G @ u).real for u in U])
    candidates = list(2.0 * e)
    for k in range(len(U)):
        for j in range(k+1, len(U)):
            z = np.vdot(U[k], G @ U[j])
            candidates.append(float(e[k]+e[j] -
                                    2*max(abs(z.real), abs(z.imag))))
    return min(candidates)


def expected_diagonal_min_d2(eta):
    """Analytically classified lower envelope for 0 <= eta <= 1."""
    if eta < 0 or eta > 1:
        raise ValueError('diagonal closed form only stated for 0 <= eta <= 1')
    return min(2.0*eta, A+eta/2.0), eta


def coordinate_distance_pairs(C):
    """Distinct (|dx1|^2, |dx2|^2) over unequal indices (not channels)."""
    D = C[:, None, :] - C[None, :, :]
    P = np.abs(D)**2
    return sorted({(round(float(x), 12), round(float(y), 12))
                   for x, y in P[np.triu_indices(len(C), 1)]})


def pareto_frontier(P, tol=1e-9):
    """Coordinatewise undominated distance vectors for nonnegative weights."""
    return [p for p in P if not any(
        q[0] <= p[0]+tol and q[1] <= p[1]+tol and
        (q[0] < p[0]-tol or q[1] < p[1]-tol)
        for q in P)]


class ChannelOrbitTests(unittest.TestCase):
    def test_orbit_decomposition_exactly_covers_both_codebooks(self):
        for C in (rank_one_constellation(), qpsk_squared()):
            self.assertEqual(C.shape, (16, 2))
            self.assertEqual(len(four_phase_representatives(C)), 4)
            np.testing.assert_allclose(np.sum(np.abs(C)**2, axis=1),
                                       np.ones(16), atol=1e-13)

    def test_orbit_quotient_matches_independent_brute_force(self):
        rng = np.random.default_rng(20261104)
        channels = [np.zeros((2, 2), complex),
                    np.diag([1., 0.]),
                    np.diag([1., 0.1]),
                    np.diag([1., np.sqrt(0.01)]),
                    np.eye(2),
                    np.array([[1j, 2], [0, 0]], complex)]
        for _ in range(40):
            channels.append(rng.normal(size=(3, 2)) +
                            1j*rng.normal(size=(3, 2)))
        for C in (rank_one_constellation(), qpsk_squared()):
            for H in channels:
                self.assertAlmostEqual(four_phase_min_received_d2(C, H),
                                       brute_min_received_d2(C, H), places=10)

    def test_diagonal_closed_forms_against_brute_force(self):
        grid = np.r_[0., 1e-4, 0.01, 0.05, CROSSOVER,
                     np.linspace(0, 1, 81), 1.]
        for eta in grid:
            H = np.diag([1., np.sqrt(eta)])
            r, q = expected_diagonal_min_d2(eta)
            self.assertAlmostEqual(r, brute_min_received_d2(
                rank_one_constellation(), H), places=10)
            self.assertAlmostEqual(q, brute_min_received_d2(
                qpsk_squared(), H), places=10)

    def test_coordinate_distance_certificate(self):
        # These four lower-envelope vertices are independently enumerated.
        front = pareto_frontier(coordinate_distance_pairs(
            rank_one_constellation()))
        expected = [(0., 2.), (A, 0.5), (0.5, A), (2., 0.)]
        self.assertEqual(len(front), 4)
        np.testing.assert_allclose(front, expected, atol=1e-11)
        # For eta <= 1, (0.5,A) is no better than (A,0.5),
        # and (2,0) is no better than (0,2).
        for eta in np.linspace(0, 1, 41):
            lo = min(a+eta*b for a,b in front)
            self.assertAlmostEqual(lo, min(2*eta, A+eta/2), places=10)

    def test_unequal_metric_reverses_euclidean_ranking(self):
        examples = [(0.01, 0.02, 0.01),
                    (0.10, A+0.05, 0.10),
                    (1.0, 2-np.sqrt(2), 1.0)]
        for eta, rank_value, qpsk_value in examples:
            H = np.diag([1., np.sqrt(eta)])
            self.assertAlmostEqual(brute_min_received_d2(
                rank_one_constellation(), H), rank_value, places=11)
            self.assertAlmostEqual(brute_min_received_d2(
                qpsk_squared(), H), qpsk_value, places=11)
        # Just below crossover rank-one wins; above it loses.
        low, high = CROSSOVER-1e-3, CROSSOVER+1e-3
        for eta, sign in ((low,1), (CROSSOVER,0), (high,-1)):
            H=np.diag([1., np.sqrt(eta)])
            dr=brute_min_received_d2(rank_one_constellation(), H)
            dq=brute_min_received_d2(qpsk_squared(), H)
            self.assertTrue(abs(dr-dq) < 1e-10 if sign==0
                            else (dr-dq)*sign > 0)

    def test_representative_choice_and_orbit_permutation_do_not_change_result(self):
        C=rank_one_constellation()
        rng=np.random.default_rng(50480)
        H=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))
        expected=four_phase_min_received_d2(C,H)
        for _ in range(40):
            shuffled=C[rng.permutation(len(C))].copy()
            self.assertAlmostEqual(four_phase_min_received_d2(shuffled,H),
                                   expected,places=10)

    def test_unitary_coordinate_changes_leave_metric_invariant(self):
        rng=np.random.default_rng(4277)
        for C in (rank_one_constellation(), qpsk_squared()):
            H=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))
            expected=four_phase_min_received_d2(C,H)
            for _ in range(20):
                Z=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))
                U,R=np.linalg.qr(Z)
                U=U@np.diag(np.exp(-1j*np.angle(np.diag(R))))
                rotated=C@U.T
                Hrot=H@U.conj().T
                self.assertAlmostEqual(four_phase_min_received_d2(
                    rotated, Hrot),expected,places=10)

    def test_nonclosed_and_degenerate_codebooks_rejected(self):
        C=rank_one_constellation().copy()
        C[0] += np.array([0.012, -0.007j])
        with self.assertRaises(ValueError):
            four_phase_representatives(C)
        with self.assertRaises(ValueError):
            four_phase_representatives(np.zeros((16,2),complex))


    def test_singular_values_do_not_determine_codebook_ranking(self):
        # A right-side input-polarization rotation changes the relative
        # orientation of the physical channel and fixed transmit codebooks.
        # Its singular values remain unchanged, yet the ranking reverses.
        eta=0.01
        diagonal=np.diag([1., np.sqrt(eta)])
        theta=np.pi/8
        rotation=np.array([[np.cos(theta),-np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
        rotated=diagonal@rotation
        np.testing.assert_allclose(np.linalg.svd(diagonal,compute_uv=False),
                                   np.linalg.svd(rotated,compute_uv=False),
                                   atol=1e-14)
        rank=rank_one_constellation()
        qpsk=qpsk_squared()
        r0=brute_min_received_d2(rank,diagonal)
        q0=brute_min_received_d2(qpsk,diagonal)
        r1=brute_min_received_d2(rank,rotated)
        q1=brute_min_received_d2(qpsk,rotated)
        self.assertGreater(r0,q0)
        self.assertLess(r1,q1)
        self.assertAlmostEqual(r1,0.005857864376269049,places=10)
        self.assertAlmostEqual(q1,0.1549821433126589,places=10)

    def test_explicit_off_cone_isometric_twin(self):
        # Change the physical transmitted codebook but hold H fixed.
        # For a diagonal channel, D commutes with H: H D = D H.
        # Therefore outputs rotate by D, preserving pairwise distances.
        C = rank_one_constellation()
        D = np.diag([1.0, 1j])
        C_twin = C @ D.T
        eta = 0.01
        H = np.diag([1., np.sqrt(eta)])
        np.testing.assert_allclose(null_form(C), 0., atol=1e-13)
        self.assertGreater(float(np.max(np.abs(null_form(C_twin)))), 0.49)
        original = raw_received_d2(C, H)
        twin = raw_received_d2(C_twin, H)
        np.testing.assert_allclose(twin, original, atol=1e-14)
        self.assertAlmostEqual(brute_min_received_d2(C_twin, H), 2*eta,
                               places=12)
        # The same ML cost vector results after rotating both received
        # signals and noisy observations by D. Circular white Gaussian noise
        # is invariant in distribution under this receiver-side rotation.
        rng = np.random.default_rng(2317)
        Y = C @ H.T
        indices = rng.integers(0, 16, 200)
        noise = 0.2*(rng.normal(size=(200, 2)) +
                     1j*rng.normal(size=(200, 2)))
        R = Y[indices] + noise
        Yt = C_twin @ H.T
        Rt = R @ D.T
        costs = np.sum(np.abs(R[:,None,:]-Y[None,:,:])**2,axis=-1)
        costs_twin = np.sum(np.abs(Rt[:,None,:]-Yt[None,:,:])**2,
                            axis=-1)
        np.testing.assert_allclose(costs_twin, costs, atol=1e-13)
        np.testing.assert_array_equal(costs_twin.argmin(axis=1),
                                      costs.argmin(axis=1))


if __name__ == '__main__':
    unittest.main()
