"""CPU-only tests of coherent white-noise Gram sufficiency and its boundary."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments') if Path(__file__).resolve().parent.name=='tests' else str(Path(__file__).resolve().parent))
import unittest
import math
import numpy as np
from rayleigh_counterworld import qpsk2, rank1


def haar(rng,n):
    z=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
    q,r=np.linalg.qr(z)
    return q@np.diag(np.exp(-1j*np.angle(np.diag(r))))


def scores(H,C,obs):
    Y=C@H.T
    return np.sum(np.abs(obs[:,None,:]-Y[None,:,:])**2,axis=-1)


class GramSufficiencyTests(unittest.TestCase):
    def test_left_unitary_preserves_full_decision_cost_matrix(self):
        rng=np.random.default_rng(24044)
        for M in (2,3,4):
            for _ in range(12):
                H=rng.normal(size=(M,2))+1j*rng.normal(size=(M,2))
                U=haar(rng,M); Ht=U@H
                np.testing.assert_allclose(H.conj().T@H,Ht.conj().T@Ht,atol=1e-12)
                for C in (qpsk2(), rank1()):
                    idx=rng.integers(0,16,size=512)
                    z=0.1*(rng.normal(size=(512,M))+1j*rng.normal(size=(512,M)))
                    Y=C@H.T
                    obs=Y[idx]+z
                    cost=scores(H,C,obs)
                    cost2=scores(Ht,C,obs@U.T)
                    np.testing.assert_allclose(cost,cost2,rtol=1e-12,atol=1e-12)
                    np.testing.assert_array_equal(cost.argmin(axis=1),cost2.argmin(axis=1))

    def test_rank_deficient_channels_still_obey_sufficiency(self):
        rng=np.random.default_rng(1722)
        H=np.array([[1, 2j],[0,0],[0,0]],complex)
        U=haar(rng,3); Ht=U@H
        np.testing.assert_allclose(H.conj().T@H,Ht.conj().T@Ht,atol=1e-12)
        C=rank1(); z=rng.normal(size=(200,3))+1j*rng.normal(size=(200,3))
        np.testing.assert_allclose(scores(H,C,z),scores(Ht,C,z@U.T),atol=1e-11)

    def test_nonwhite_noise_breaks_unweighted_gram_sufficiency(self):
        # Two channels with identical H*H, but different distances under
        # a fixed non-white receiver covariance. This is a conditional
        # pairwise-error counterexample (no stochastic sampling needed).
        H1=np.eye(2,dtype=complex)
        H2=np.array([[0,1],[1,0]],complex)
        delta=np.array([1,0],complex)
        Sinv=np.diag([100.,1.])
        np.testing.assert_allclose(H1.conj().T@H1,H2.conj().T@H2)
        d1=float(np.vdot(H1@delta,Sinv@(H1@delta)).real)
        d2=float(np.vdot(H2@delta,Sinv@(H2@delta)).real)
        self.assertEqual((d1,d2),(100.,1.))
        pep=lambda d:0.5*math.erfc(math.sqrt(d)/2)
        self.assertLess(pep(d1),pep(d2))
        self.assertGreater(pep(d2),0.1)

    def test_gram_contains_orientation_not_just_singular_values(self):
        H=np.diag([1.,0.1])
        theta=np.pi/8
        R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        Ht=H@R
        np.testing.assert_allclose(np.linalg.svd(H,compute_uv=False),
                                   np.linalg.svd(Ht,compute_uv=False),atol=1e-14)
        self.assertGreater(np.linalg.norm(H.T@H-Ht.T@Ht),0.2)

if __name__=='__main__':
    unittest.main()
