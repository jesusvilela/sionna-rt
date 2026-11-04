"""Boundary-layer decoding when a 2-polarization channel approaches rank loss.

Frozen 16-point codebooks from Sionna RT PR #82.  Pure CPU/NumPy + stdlib.
No Sionna RT, GPU, HF billing, or ray tracing.  Physical causal benefit NOT claimed.
"""

import math
import unittest

import numpy as np


def rank_one_codebook():
    points = []
    for m in range(8):
        v = np.array([np.cos(m*np.pi/4), np.sin(m*np.pi/4)])
        for w in (np.array([1., 1.]), np.array([1., -1.])):
            X = np.outer(v, w)
            points.append([X[0, 0]+1j*X[0, 1], X[1, 0]+1j*X[1, 1]])
    C = np.asarray(points, dtype=np.complex128)
    return C / np.sqrt(np.sum(np.abs(C)**2, axis=1).mean())


def qpsk2_codebook():
    g = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    C = np.array([[x, y] for x in g for y in g])
    return C / np.sqrt(np.sum(np.abs(C)**2, axis=1).mean())


def first_coordinate_classes(C, tol=1e-10):
    """Partition into equality classes of x[0] (the channel h_0 output)."""
    groups = []
    for i, x in enumerate(C):
        for g in groups:
            if abs(C[g[0], 0] - x[0]) < tol:
                g.append(i)
                break
        else:
            groups.append([i])
    return [np.asarray(g, dtype=int) for g in groups]


def Q(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def rank_boundary_ser(rho):
    """Exact double-scaling limit for rank-one book; rho=|eps|^2/N0."""
    if rho < 0:
        raise ValueError('rho is nonnegative')
    q = Q(math.sqrt(rho))
    return q - 0.25*q*q


def qpsk_boundary_ser(rho):
    """Exact double-scaling limit for QPSK-squared; rho=|eps|^2/N0."""
    if rho < 0:
        raise ValueError('rho is nonnegative')
    q = Q(math.sqrt(rho/2))
    return 2*q - q*q


def oracle_ml_ser(C, rho, seed, samples=80000, phase=0.):
    """ML error with the x[0] collision class supplied to the receiver.

    The observation within each class is alpha*x[1] + CN(0,1), where
    alpha=sqrt(rho)*exp(1j*phase).  This is independent of N0 exactly.
    """
    rng = np.random.default_rng(seed)
    tx = rng.integers(0, len(C), size=samples)
    noise = (rng.normal(size=samples) + 1j*rng.normal(size=samples))/np.sqrt(2)
    alpha = math.sqrt(rho)*np.exp(1j*phase)
    z = alpha*C[tx,1] + noise
    errors = 0
    for g in first_coordinate_classes(C):
        mask = np.isin(tx, g)
        if not mask.any():
            continue
        costs = np.abs(z[mask,None] - alpha*C[g,1][None,:])**2
        pred = g[np.argmin(costs,axis=1)]
        errors += int(np.count_nonzero(pred != tx[mask]))
    return errors/samples


def full_ml_ser(C, rho, N0, seed, samples=100000, phase=0.):
    """Scalar complex channel h=(1, eps), eps=sqrt(rho*N0)*e^(i phase)."""
    if N0 <= 0 or rho < 0:
        raise ValueError('N0 must be positive; rho must be nonnegative')
    eps = math.sqrt(rho*N0)*np.exp(1j*phase)
    Y = C[:,0] + eps*C[:,1]
    rng = np.random.default_rng(seed)
    errors = 0
    for start in range(0, samples, 2000):
        size = min(2000, samples-start)
        tx = rng.integers(len(C),size=size)
        noise = math.sqrt(N0/2)*(rng.normal(size=size)+1j*rng.normal(size=size))
        obs = Y[tx] + noise
        pred = np.argmin(np.abs(obs[:,None]-Y[None,:])**2,axis=1)
        errors += int(np.count_nonzero(pred != tx))
    return errors/samples


def distinct_received_count(C, eps, tol=1e-9):
    values=C[:,0]+eps*C[:,1]
    seen=[]
    for x in values:
        if not any(abs(x-y) < tol for y in seen):
            seen.append(x)
    return len(seen)


def cross_group_upper_excess(C, rho, N0):
    """Union bound on full SER minus class-oracle SER.

    delta=min gap between distinct x[0] classes, B=max(abs(x[1])).
    For delta>2*B*sqrt(rho*N0), cross-class point separation is at least
    delta-2*B*sqrt(rho*N0). Bound: (M-1)*Q(gap/sqrt(2*N0)).
    Return None if the positive-separation certificate fails.
    """
    groups=first_coordinate_classes(C)
    coords=[C[g[0],0] for g in groups]
    delta=min(abs(a-b) for i,a in enumerate(coords) for b in coords[i+1:])
    B=float(np.max(abs(C[:,1])))
    gap=delta-2*B*math.sqrt(rho*N0)
    if gap<=0:
        return None
    return (len(C)-1)*Q(gap/math.sqrt(2*N0))


class BoundaryLayerTests(unittest.TestCase):
    def test_collision_class_multiplicities_and_per_symbol_energy(self):
        for C, expected in ((rank_one_codebook(),[1]*4+[2]*4+[4]),
                            (qpsk2_codebook(),[4]*4)):
            self.assertEqual(sorted(map(len,first_coordinate_classes(C))),
                             sorted(expected))
            np.testing.assert_allclose(np.sum(abs(C)**2,axis=1),1,atol=1e-13)
            self.assertEqual(distinct_received_count(C,0),len(expected))

    def test_exact_boundary_at_zero_matches_collision_floors(self):
        self.assertAlmostEqual(rank_boundary_ser(0),7/16)
        self.assertAlmostEqual(qpsk_boundary_ser(0),3/4)
        for rho in (0,0.25,1,4,9):
            self.assertLess(rank_boundary_ser(rho),qpsk_boundary_ser(rho))
        self.assertLess(rank_boundary_ser(100),1e-20)
        self.assertLess(qpsk_boundary_ser(100),1e-10)

    def test_oracle_ml_agrees_with_two_independent_analytic_formulas(self):
        cases=((rank_one_codebook(),rank_boundary_ser),
               (qpsk2_codebook(),qpsk_boundary_ser))
        for C,F in cases:
            for rho in (0,0.25,1,4):
                estimate=oracle_ml_ser(C,rho,seed=10892,samples=80000)
                p=F(rho)
                sigma=math.sqrt(p*(1-p)/80000)
                self.assertLess(abs(estimate-p),6*sigma+0.001)

    def test_full_ml_converges_to_oracle_boundary_at_fixed_rho(self):
        for C,F in ((rank_one_codebook(),rank_boundary_ser),
                    (qpsk2_codebook(),qpsk_boundary_ser)):
            for rho in (0,0.25,1,4):
                estimate=full_ml_ser(C,rho,1e-4,seed=30825,samples=80000)
                p=F(rho)
                sigma=math.sqrt(p*(1-p)/80000)
                self.assertLess(abs(estimate-p),6*sigma+0.002)

    def test_boundary_limit_is_independent_of_relative_phase(self):
        C=rank_one_codebook()
        for phi in (0,np.pi/3,1.71):
            estimate=full_ml_ser(C,1.,1e-4,seed=8871,
                                 samples=80000,phase=phi)
            p=rank_boundary_ser(1.)
            sigma=math.sqrt(p*(1-p)/80000)
            self.assertLess(abs(estimate-p),6*sigma+0.002)

    def test_oracle_lower_bound_and_exponentially_small_cross_error(self):
        for C,F in ((rank_one_codebook(),rank_boundary_ser),
                    (qpsk2_codebook(),qpsk_boundary_ser)):
            rho,N0=1.,1e-4
            bound=cross_group_upper_excess(C,rho,N0)
            self.assertIsNotNone(bound)
            self.assertLess(bound,1e-10)
            sample=full_ml_ser(C,rho,N0,seed=5517,samples=100000)
            p=F(rho)
            sigma=math.sqrt(p*(1-p)/100000)
            self.assertLess(abs(sample-p),6*sigma+0.002)

    def test_exact_group_known_awgn_scaling_invariance(self):
        C=rank_one_codebook()
        g=next(g for g in first_coordinate_classes(C) if len(g)==4)
        rng=np.random.default_rng(27)
        tx=rng.integers(len(g),size=6000)
        z=(rng.normal(size=6000)+1j*rng.normal(size=6000))/np.sqrt(2)
        rho,phase=2.3,0.71
        alpha=math.sqrt(rho)*np.exp(1j*phase)
        scores_ref=np.abs((alpha*C[g[tx],1]+z)[:,None]-alpha*C[g,1][None,:])**2
        decisions_ref=np.argmin(scores_ref,axis=1)
        for N0 in (1e-2,1e-4,1e-7):
            eps=math.sqrt(N0)*alpha
            received=C[g[0],0]+eps*C[g[tx],1]+math.sqrt(N0)*z
            candidates=C[g[0],0]+eps*C[g,1]
            decisions=np.argmin(abs(received[:,None]-candidates[None,:])**2,
                                axis=1)
            np.testing.assert_array_equal(decisions,decisions_ref)

    def test_off_cone_twin_same_collision_profile_and_boundary(self):
        C=rank_one_codebook()
        twin=C.copy(); twin[:,1]*=1j
        self.assertEqual(sorted(map(len,first_coordinate_classes(C))),
                         sorted(map(len,first_coordinate_classes(twin))))
        self.assertGreater(np.max(np.abs(np.imag(np.conj(twin[:,0])*twin[:,1]))),
                           0.49)
        # For each x1 collision class, multiplication of all x2 by i is
        # an exact isometry, so all complex AWGN class risks coincide.
        for g in first_coordinate_classes(C):
            lhs=np.abs(C[g,1][:,None]-C[g,1][None,:])
            rhs=np.abs(twin[g,1][:,None]-twin[g,1][None,:])
            np.testing.assert_allclose(lhs,rhs,atol=1e-14)

    def test_nearby_secondary_discriminant_invalidates_uniform_finite_noise_claim(self):
        # Selected adversarially, not a preregistered performance benchmark.
        # rho=9 and N0=.01 => eps=.3, close to another real collision
        # direction eps=sqrt(2)-1.  This is outside the tiny-noise regime.
        C=rank_one_codebook()
        estimate=full_ml_ser(C,9,.01,seed=3221,samples=60000)
        self.assertGreater(estimate,rank_boundary_ser(9)+0.10)
        self.assertIsNone(cross_group_upper_excess(C,9,.01))
        self.assertEqual(distinct_received_count(C,math.sqrt(2)-1),8)
        self.assertEqual(distinct_received_count(qpsk2_codebook(),
                                                math.sqrt(2)-1),16)

    def test_separation_certificate_at_axis_and_generic_direction(self):
        C=rank_one_codebook()
        groups=first_coordinate_classes(C)
        vals=[C[g[0],0] for g in groups]
        gap=min(abs(a-b) for i,a in enumerate(vals) for b in vals[i+1:])
        self.assertAlmostEqual(gap,1-1/math.sqrt(2),places=12)
        self.assertEqual(distinct_received_count(C,0),9)
        self.assertEqual(distinct_received_count(C,.17+0.11j),16)
        self.assertLess(cross_group_upper_excess(C,1,1e-5),1e-100)


if __name__ == '__main__':
    unittest.main()
