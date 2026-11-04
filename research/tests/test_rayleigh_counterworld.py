"""Independent deterministic checks of the Rayleigh counterworld model.

No external radio simulator or GPU. Run: python -m unittest -v
"""
import math
import unittest
import numpy as np
from scipy.integrate import quad
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments") if Path(__file__).resolve().parent.name == "tests" else str(Path(__file__).resolve().parent))
from rayleigh_counterworld import (
    qpsk2,rank1,pep_rayleigh_averaged,union_bound_averaged_iid,
    errors_by_channel,numerical_invariants)


def distance_spectrum(C):
    y=np.sum(np.abs(C[:,None,:]-C[None,:,:])**2,axis=-1)
    return np.sort(y[np.triu_indices(16,1)])


def direct_craig_pep(d2,N0,r=2):
    return quad(lambda th:(1+d2/(4*r*N0*math.sin(th)**2))**(-r)/math.pi,
                0,math.pi/2,epsabs=1e-12)[0]


class RayleighCounterworldTests(unittest.TestCase):
    def test_unchanged_PR82_codebooks_and_offcone_control(self):
        rank=rank1(); qpsk=qpsk2(); D=np.diag([1,1j]); twin=rank@D.T
        numerical_invariants(qpsk,rank,twin)
        np.testing.assert_allclose(np.sort(distance_spectrum(rank)),
                                   np.sort(distance_spectrum(twin)),atol=1e-13)

    def test_complete_pair_distance_spectra(self):
        for C,expected in [(rank1(),[(2-np.sqrt(2),16),(2.,80),(2+np.sqrt(2),16),(4.,8)]),
                           (qpsk2(),[(1.,32),(2.,48),(3.,32),(4.,8)])]:
            v=distance_spectrum(C)
            self.assertEqual(len(v),120)
            for d,count in expected:
                self.assertEqual(np.sum(np.isclose(v,d,atol=1e-12)),count)
            self.assertAlmostEqual(float(np.sum(v)),256.0,places=10)

    def test_rayleigh_craig_kernel_against_independent_quad(self):
        for N0 in (0.02,0.04,0.08,1.0):
            for d2 in (2-np.sqrt(2),1.,2.,2+np.sqrt(2)):
                self.assertAlmostEqual(pep_rayleigh_averaged(d2,N0),
                                       direct_craig_pep(d2,N0),places=12)

    def test_unitary_offcone_expected_union_bound_exactly_equal(self):
        rank=rank1(); twin=rank@np.diag([1,1j]).T
        for N0 in (0.01,0.04,0.08,0.5):
            self.assertAlmostEqual(union_bound_averaged_iid(rank,N0),
                                   union_bound_averaged_iid(twin,N0),places=12)

    def test_exact_high_snr_union_coefficients(self):
        rank=rank1(); qpsk=qpsk2()
        sr=np.sum(1/distance_spectrum(rank)**2)
        sq=np.sum(1/distance_spectrum(qpsk)**2)
        self.assertAlmostEqual(sr,137/2,places=10)
        self.assertAlmostEqual(sq,865/18,places=10)
        tiny=1e-6
        self.assertAlmostEqual(union_bound_averaged_iid(rank,tiny)/tiny**2,
                               411/4,delta=0.02)
        self.assertAlmostEqual(union_bound_averaged_iid(qpsk,tiny)/tiny**2,
                               865/12,delta=0.02)

    def test_cyclic_orbit_antithetic_control_has_zero_empirical_delta(self):
        rng=np.random.default_rng(14089)
        Cr=rank1(); D=np.diag([1,1j]); Ct=Cr@D.T
        idx=rng.integers(0,16,size=(1,200))
        noise=0.1*(rng.normal(size=(1,200,2))+1j*rng.normal(size=(1,200,2)))
        for _ in range(12):
            H=(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))/2
            errors_r=[];errors_t=[]
            for k in range(4):
                Hk=(H@np.linalg.matrix_power(D,k))[None,:,:]
                errors_r.append(int(errors_by_channel(Cr,Hk,idx,noise)[0]))
                errors_t.append(int(errors_by_channel(Ct,Hk,idx,noise)[0]))
            self.assertEqual(sum(errors_r),sum(errors_t))
            self.assertEqual(errors_t,errors_r[1:]+errors_r[:1])

    def test_known_answer_qpsk_identity_ser_against_model(self):
        rng=np.random.default_rng(111)
        N0=.08
        tx=rng.integers(0,16,size=(1,100000))
        noise=np.sqrt(N0/2)*(rng.normal(size=(1,100000,2))+1j*rng.normal(size=(1,100000,2)))
        observed=errors_by_channel(qpsk2(),np.eye(2)[None,:,:],tx,noise)[0]/100000
        p=.5*math.erfc(1/(2*math.sqrt(N0)))
        exact=1-(1-p)**4
        self.assertLess(abs(observed-exact),5*math.sqrt(exact*(1-exact)/100000)+0.001)

if __name__=='__main__':
    unittest.main()
