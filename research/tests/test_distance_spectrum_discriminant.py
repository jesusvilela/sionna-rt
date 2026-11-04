"""Finite codebook distance spectra, ML bounds, and rank-one collision loci.

CPU-only, NumPy and Python stdlib. No Sionna, GPU, SciPy, or ray tracing.
The codebooks reproduce the fixed 16-point constructors of Sionna RT PR #82.

What the tests do NOT prove: a new physical 6G law or that the null-cone
constraint causally improves symbol error rate in ray-traced channels.
"""

import math
import unittest
from collections import Counter

import numpy as np


PHASES = (1 + 0j, 1j, -1 + 0j, -1j)


def original_rank_one_codebook():
    """Independent transcription of the 16-point notebook construction."""
    points = []
    for m in range(8):
        v = np.array([np.cos(2*np.pi*m/8), np.sin(2*np.pi*m/8)])
        for w in (np.array([1., 1.]), np.array([1., -1.])):
            X = np.outer(v, w)
            points.append((X[0, 0] + 1j*X[0, 1],
                           X[1, 0] + 1j*X[1, 1]))
    C = np.asarray(points, dtype=np.complex128)
    return C / np.sqrt((np.abs(C)**2).sum(axis=1).mean())


def original_qpsk2_codebook():
    g = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    C = np.array([[a, b] for a in g for b in g])
    return C / np.sqrt((np.abs(C)**2).sum(axis=1).mean())


def orbit_representatives(C, tol=1e-10):
    """Require *complete, distinct* orbits under multiplication by i."""
    C = np.asarray(C, dtype=np.complex128)
    if C.ndim != 2 or C.shape[1] != 2:
        raise ValueError('expected M x 2 complex codebook')
    remaining = set(range(len(C)))
    reps = []
    while remaining:
        idx = min(remaining)
        x = C[idx]
        members = []
        for phase in PHASES:
            matches = np.flatnonzero(np.linalg.norm(C-phase*x, axis=1) < tol)
            if len(matches) != 1:
                raise ValueError('codebook not a complete set of unique mu4 orbits')
            members.append(int(matches[0]))
        if len(set(members)) != 4 or not set(members).issubset(remaining):
            raise ValueError('overlapping, zero, or duplicate orbit')
        reps.append(x)
        remaining -= set(members)
    return np.asarray(reps)


def brute_squared_spectrum(C, H):
    """All unordered Euclidean received squared distances, including zeros."""
    Y = np.asarray(C, complex) @ np.asarray(H, complex).T
    diff = Y[:, None, :] - Y[None, :, :]
    values = np.sum(np.abs(diff)**2, axis=-1)
    return np.sort(values[np.triu_indices(len(C), 1)])


def orbit_squared_spectrum(C, H):
    """Complete weighted distance enumerator from the four-phase quotient.

    For K orbits, there are 4K + 2K intra-orbit pairs, and
    16*binom(K,2) cross-orbit pairs; counts sum to binom(4K,2).
    """
    U = orbit_representatives(C)
    H = np.asarray(H, complex)
    G = H.conj().T @ H
    e = np.array([np.vdot(u, G@u).real for u in U])
    values = []
    for energy in e:
        values.extend([2*energy]*4 + [4*energy]*2)
    for k in range(len(U)):
        for j in range(k+1, len(U)):
            z = np.vdot(U[k], G@U[j])
            for phase in PHASES:
                # each relative quarter-phase occurs for four ordered phase
                # choices in the first and second orbit (unordered code pairs).
                values.extend([e[k]+e[j]-2*(phase*z).real]*4)
    return np.sort(np.asarray(values, float))


def pairwise_q(d2, N0):
    """PEP Q(sqrt(d2/(2*N0))) for proper CN(0,N0 I) receiver noise."""
    if N0 <= 0 or d2 < 0:
        raise ValueError('N0 positive and distance squared nonnegative')
    return 0.5*math.erfc(math.sqrt(d2)/(2*math.sqrt(N0)))


def exact_qpsk2_diagonal_ser(eta, N0):
    """Exact coherent AWGN SER for QPSK^2 on H=diag(1,sqrt(eta)).

    Four independent real BPSK decisions have error probabilities
    Q(1/sqrt(2*N0)) (twice) and Q(sqrt(eta/(2*N0))) (twice).
    For eta=0, each erased real bit is unobservable and its correct
    probability is 1/2, giving a 3/4 floor as N0 approaches zero.
    """
    if eta < 0 or N0 <= 0:
        raise ValueError('eta nonnegative and N0 positive')
    p1 = 0.5*math.erfc(1/(2*math.sqrt(N0)))
    p2 = 0.5*math.erfc(math.sqrt(eta)/(2*math.sqrt(N0)))
    return 1-(1-p1)**2*(1-p2)**2


def ml_ser_bounds(C, H, N0):
    """Nearest-neighbor lower and all-pair union upper SER bounds.

    Only for distinct received candidates: at exact collisions, a pairwise
    'Q(0)' treatment is tie-convention-dependent. Use collision floor instead.
    Returns *raw* upper bound (possibly >1), never silently clips it.
    """
    if N0 <= 0:
        raise ValueError('N0 must be positive')
    C = np.asarray(C, complex)
    Y = C @ np.asarray(H, complex).T
    d2 = np.sum(np.abs(Y[:, None, :] - Y[None, :, :])**2, axis=-1)
    iu = np.triu_indices(len(C), 1)
    if np.min(d2[iu]) < 1e-12:
        raise ValueError('received codeword collisions; use collision floor')
    nearest = np.min(np.where(np.eye(len(C), dtype=bool), np.inf, d2), axis=1)
    lower = sum(pairwise_q(x, N0) for x in nearest)/len(C)
    upper = 2*sum(pairwise_q(x, N0) for x in d2[iu])/len(C)
    return lower, upper


def collision_classes(C, h, tol=1e-7):
    """Numerical grouping of scalar outputs for one nonzero channel row.

    Tolerance accounts for floating trig/phase values and rounded ray
    representatives. It is *not* a proof of the collision class structure;
    the separate symbolic certificate uses exact algebraic equalities.
    """
    vals = np.asarray(C, complex) @ np.asarray(h, complex)
    centers, counts = [], []
    for z in vals:
        for i, center in enumerate(centers):
            if abs(z-center) <= tol:
                counts[i] += 1
                break
        else:
            centers.append(z)
            counts.append(1)
    return sorted(counts)


def canonical_ray(h, decimals=9):
    h = np.asarray(h, complex)
    h = h/np.linalg.norm(h)
    pivot = 0 if abs(h[0]) > 1e-9 else 1
    h *= h[pivot].conjugate()/abs(h[pivot])
    return tuple(np.round([h[0].real, h[0].imag, h[1].real, h[1].imag], decimals))


def collision_direction_histogram(C):
    """Numerical witness for exceptional CP1 directions and collision pairs."""
    C = np.asarray(C, complex)
    out = {}
    for i in range(len(C)):
        for j in range(i+1, len(C)):
            d = C[i]-C[j]
            h = np.array([d[1], -d[0]])
            ray = canonical_ray(h)
            out.setdefault(ray, []).append((i, j))
    return Counter((len(edges), len(C)-len(collision_classes(
        C, np.array([ray[0]+1j*ray[1], ray[2]+1j*ray[3]]))))
        for ray, edges in out.items()), len(out)


def ser_monte_carlo(C, H, N0, count, seed):
    """Finite-sample coherent complex-AWGN ML detector; no physical claims."""
    rng = np.random.default_rng(seed)
    C = np.asarray(C, complex)
    Y = C @ np.asarray(H, complex).T
    errors = 0
    for start in range(0, count, 2000):
        batch = min(2000, count-start)
        tx = rng.integers(0, len(C), size=batch)
        noise = np.sqrt(N0/2)*(rng.normal(size=(batch, Y.shape[1])) +
                               1j*rng.normal(size=(batch, Y.shape[1])))
        received = Y[tx]+noise
        costs = np.sum(np.abs(received[:, None, :]-Y[None, :, :])**2, axis=-1)
        errors += int(np.count_nonzero(np.argmin(costs, axis=1) != tx))
    return errors/count


class DistanceSpectrumDiscriminantTests(unittest.TestCase):
    def test_codebooks_have_equal_energy_and_distinct_symbols(self):
        for C in (original_rank_one_codebook(), original_qpsk2_codebook()):
            self.assertEqual(C.shape, (16, 2))
            np.testing.assert_allclose(np.sum(np.abs(C)**2, axis=1), 1, atol=1e-13)
            self.assertEqual(len(orbit_representatives(C)), 4)
            self.assertGreater(brute_squared_spectrum(C, np.eye(2))[0], 0)

    def test_full_mu4_spectrum_matches_independent_120_pair_enumeration(self):
        rng = np.random.default_rng(7777)
        channels = [np.zeros((2, 2)), np.eye(2),
                    np.diag([1., 0.]), np.diag([1., 0.1]),
                    np.array([[1+2j, -0.5j]], complex)]
        channels += [rng.normal(size=(3, 2))+1j*rng.normal(size=(3, 2))
                     for _ in range(40)]
        for C in (original_rank_one_codebook(), original_qpsk2_codebook()):
            for H in channels:
                expected = brute_squared_spectrum(C, H)
                actual = orbit_squared_spectrum(C, H)
                self.assertEqual(len(actual), 120)
                np.testing.assert_allclose(actual, expected,
                                           rtol=1e-11, atol=5e-12)

    def test_full_spectrum_invariant_under_joint_unitary_frame(self):
        rng = np.random.default_rng(20261104)
        for C in (original_rank_one_codebook(), original_qpsk2_codebook()):
            H = rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2))
            expected = brute_squared_spectrum(C, H)
            for _ in range(20):
                Z = rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2))
                U, R = np.linalg.qr(Z)
                U = U@np.diag(np.exp(-1j*np.angle(np.diag(R))))
                Cp = C@U.T
                Hp = H@U.conj().T
                np.testing.assert_allclose(orbit_squared_spectrum(Cp, Hp),
                                           expected, atol=1e-11, rtol=1e-12)

    def test_gaussian_pairwise_closed_form_and_bounds_input_validation(self):
        self.assertAlmostEqual(pairwise_q(0., 1.), 0.5)
        self.assertAlmostEqual(pairwise_q(4., 1.), 0.5*math.erfc(1.))
        self.assertLess(pairwise_q(4., 0.01), pairwise_q(4., 1.))
        with self.assertRaises(ValueError):
            ml_ser_bounds(original_rank_one_codebook(), np.diag([1, 0]), 0.1)
        with self.assertRaises(ValueError):
            pairwise_q(1., 0.)

    def test_union_bounds_enclose_finite_sample_ml_ser(self):
        # Independent detector versus geometric bounds; statistical tolerance.
        cases = [(0.01, 0.15), (0.2, 0.10), (1.0, 0.15)]
        for C in (original_rank_one_codebook(), original_qpsk2_codebook()):
            for eta, N0 in cases:
                H = np.diag([1, np.sqrt(eta)])
                lower, upper = ml_ser_bounds(C, H, N0)
                observed = ser_monte_carlo(C, H, N0, 30000, 4111)
                sigma = math.sqrt(max(observed*(1-observed), 1e-6)/30000)
                self.assertGreaterEqual(observed+6*sigma, lower)
                self.assertLessEqual(observed-6*sigma, upper)

    def test_exact_qpsk_diagonal_awgn_formula_against_independent_ml(self):
        C = original_qpsk2_codebook()
        for eta, N0 in ((0.,0.005), (.01,.02), (.2,.02), (1.,.08)):
            expected = exact_qpsk2_diagonal_ser(eta, N0)
            actual = ser_monte_carlo(C, np.diag([1., np.sqrt(eta)]),
                                     N0, 60000, 9317)
            sigma = math.sqrt(max(expected*(1-expected), 1e-6)/60000)
            self.assertLess(abs(actual-expected), 6*sigma)
        self.assertAlmostEqual(exact_qpsk2_diagonal_ser(0.,0.01),
                               1-(1-0.5*math.erfc(1/(2*math.sqrt(.01))))**2/4)

    def test_exploratory_distance_and_ser_rankings_can_disagree(self):
        # Model-internal replication of an exploratory finding; this point
        # was chosen after a grid scan and is NOT pre-registered physical proof.
        eta, N0 = 0.2, 0.02
        H = np.diag([1.,np.sqrt(eta)])
        rank = original_rank_one_codebook()
        qpsk = original_qpsk2_codebook()
        self.assertLess(brute_squared_spectrum(rank,H)[0],
                        brute_squared_spectrum(qpsk,H)[0])
        baseline = exact_qpsk2_diagonal_ser(eta, N0)
        estimate = ser_monte_carlo(rank,H,N0,200000,8851)
        se = math.sqrt(max(estimate*(1-estimate),1e-6)/200000)
        self.assertLess(estimate+6*se,baseline)

    def test_exact_axis_projection_multiplicities_and_floors(self):
        rank, qpsk = original_rank_one_codebook(), original_qpsk2_codebook()
        self.assertEqual(collision_classes(rank, [1, 0]),
                         [1, 1, 1, 1, 2, 2, 2, 2, 4])
        self.assertEqual(collision_classes(qpsk, [1, 0]), [4, 4, 4, 4])
        self.assertEqual(1-len(collision_classes(rank, [1,0]))/16, 7/16)
        self.assertEqual(1-len(collision_classes(qpsk, [1,0]))/16, 3/4)

    def test_orientation_changes_collision_floors(self):
        rank, qpsk = original_rank_one_codebook(), original_qpsk2_codebook()
        for C in (rank, qpsk):
            self.assertEqual(len(collision_classes(C, [1, 1])), 9)
            self.assertEqual(len(collision_classes(C, [1, 2*np.exp(0.23j)])), 16)
        self.assertEqual(len(collision_classes(qpsk, [1,0])), 4)
        self.assertEqual(len(collision_classes(rank, [1,0])), 9)

    def test_discriminant_histograms_numerical_witness(self):
        rank_hist, rank_n = collision_direction_histogram(original_rank_one_codebook())
        qpsk_hist, qpsk_n = collision_direction_histogram(original_qpsk2_codebook())
        self.assertEqual(rank_n, 18)
        self.assertEqual(qpsk_n, 14)
        self.assertEqual(rank_hist, Counter({(4,4): 8, (8,8): 6, (10,7): 4}))
        self.assertEqual(qpsk_hist, Counter({(4,4): 8, (10,7): 4, (24,12): 2}))
        # Each unordered pair of distinct codewords has exactly one collision ray.
        self.assertEqual(sum(pair_count*nrays
                             for (pair_count,_), nrays in rank_hist.items()), 120)
        self.assertEqual(sum(pair_count*nrays
                             for (pair_count,_), nrays in qpsk_hist.items()), 120)

    def test_off_cone_twin_retains_full_spectrum_and_floor(self):
        C = original_rank_one_codebook()
        D = np.diag([1, 1j])
        twin = C@D.T
        H = np.diag([1., 0.])
        np.testing.assert_allclose(orbit_squared_spectrum(twin, H),
                                   orbit_squared_spectrum(C, H), atol=1e-13)
        self.assertEqual(collision_classes(C, [1,0]), collision_classes(twin,[1,0]))
        self.assertGreater(max(abs(np.imag(np.conj(twin[:,0])*twin[:,1]))), 0.49)

    def test_noncommuting_zero_noise_and_rank_loss_limits(self):
        C = original_rank_one_codebook()
        for eta in (1e-2, 1e-4, 1e-8):
            self.assertGreater(brute_squared_spectrum(C,
                               np.diag([1., np.sqrt(eta)]))[0], 0.)
        self.assertLess(brute_squared_spectrum(C, np.diag([1., 0.]))[0], 1e-25)
        self.assertEqual(1-len(collision_classes(C,[1,0]))/16, 7/16)

    def test_empirical_high_snr_axis_floor_approaches_class_count(self):
        H = np.diag([1., 0.])
        for C, floor in ((original_rank_one_codebook(),7/16),
                         (original_qpsk2_codebook(),3/4)):
            measured = ser_monte_carlo(C, H, 1e-4, 40000, 9182)
            self.assertLess(abs(measured-floor), 0.015)

    def test_toy_collision_locus_three_directions(self):
        C = np.array([[0,0], [1,0], [0,1]], dtype=complex)
        hist, n = collision_direction_histogram(C)
        self.assertEqual(n,3)
        self.assertEqual(hist,Counter({(1,1):3}))


if __name__ == '__main__':
    unittest.main()
