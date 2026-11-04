"""Source-matched but SINGLE-RECEIVER tests; not a 16-channel replication."""
import unittest
import numpy as np
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from printed_channel_control import (
    H_PRINTED,NOISE_VARIANCE,rank_one,qpsk_squared,
    equal_port_circular_psk16,pairwise_gaussian_bounds,
    circular_psk16_ser_exact,printed_precision_error_bound,report,
)

class PrintedChannelControls(unittest.TestCase):
    def test_source_constellations_and_power_parity(self):
        for C in (rank_one(),qpsk_squared(),equal_port_circular_psk16()):
            self.assertEqual(C.shape,(16,2))
            np.testing.assert_allclose((abs(C)**2).sum(axis=1),np.ones(16),atol=1e-12)
            np.testing.assert_allclose((abs(C)**2).mean(axis=0),[.5,.5],atol=1e-12)
        np.testing.assert_allclose((abs(equal_port_circular_psk16())**2),np.full((16,2),.5),atol=1e-12)

    def test_circular_is_strictly_off_original_null_cone(self):
        rank=rank_one();other=equal_port_circular_psk16()
        q=lambda x:np.imag(np.conj(x[:,0])*x[:,1])
        np.testing.assert_allclose(q(rank),0,atol=1e-12)
        np.testing.assert_allclose(q(other),.5,atol=1e-12)

    def test_printed_matrix_roughly_normalized_and_strong_column(self):
        self.assertLess(abs(np.linalg.norm(H_PRINTED)-1.),1e-6)
        e=(abs(H_PRINTED)**2).sum(axis=0)
        self.assertLess(abs(e[0]-0.98104893199),1e-9)
        self.assertGreater(e[0]/sum(e),.98)

    def test_strict_ser_certificate_single_printed_channel(self):
        lo_rank,_=pairwise_gaussian_bounds(rank_one(),H_PRINTED,NOISE_VARIANCE)
        _,up_circ=pairwise_gaussian_bounds(equal_port_circular_psk16(),H_PRINTED,NOISE_VARIANCE)
        self.assertAlmostEqual(lo_rank,.06808739425621839,places=12)
        self.assertAlmostEqual(up_circ,.05182352183597337,places=12)
        self.assertGreater(lo_rank-up_circ,.0162)

    def test_the_same_design_order_reverses_on_balanced_H(self):
        H=np.eye(2)/np.sqrt(2)
        _,up_rank=pairwise_gaussian_bounds(rank_one(),H,NOISE_VARIANCE)
        lo_circ,_=pairwise_gaussian_bounds(equal_port_circular_psk16(),H,NOISE_VARIANCE)
        self.assertLess(up_rank,0.00014)
        self.assertGreater(lo_circ,0.025)
        self.assertLess(up_rank,lo_circ)

    def test_printed_receiver_energy_almost_matched(self):
        gain=lambda C:np.mean((abs(C@H_PRINTED.T)**2).sum(axis=1))
        self.assertAlmostEqual(gain(rank_one()),.4999999619704065,places=10)
        self.assertAlmostEqual(gain(equal_port_circular_psk16()),.4973473628457301,places=10)
        self.assertLess(abs(gain(rank_one())-gain(equal_port_circular_psk16())),.003)

    def test_finite_print_precision_cannot_reverse_the_certificate(self):
        e=printed_precision_error_bound(NOISE_VARIANCE)
        a=report()['strict_printed_margin']
        self.assertGreater(a-e['lower_error_max']-e['upper_error_max'],0.0162)
        rng=np.random.default_rng(5527)
        for _ in range(16):
            p=(rng.uniform(-.5e-8,.5e-8,(2,2))+1j*rng.uniform(-.5e-8,.5e-8,(2,2)))
            lr,_=pairwise_gaussian_bounds(rank_one(),H_PRINTED+p,NOISE_VARIANCE)
            _,uc=pairwise_gaussian_bounds(equal_port_circular_psk16(),H_PRINTED+p,NOISE_VARIANCE)
            self.assertGreater(lr,uc)

    def test_gaussian_bounds_consistent_with_scalar_phase_integral(self):
        for H in (H_PRINTED,np.eye(2)/np.sqrt(2)):
            lo,up=pairwise_gaussian_bounds(equal_port_circular_psk16(),H,NOISE_VARIANCE)
            exact=circular_psk16_ser_exact(H,NOISE_VARIANCE)
            self.assertLessEqual(lo,exact)
            self.assertGreaterEqual(up,exact)
        self.assertAlmostEqual(circular_psk16_ser_exact(H_PRINTED,NOISE_VARIANCE),
                               0.051688226653448965,places=11)

    def test_pairwise_bound_invariant_under_joint_input_frame(self):
        rng=np.random.default_rng(631)
        C=rank_one()
        a=.31;b=.97
        D=np.diag(np.exp(1j*np.array([a,b])))
        before=pairwise_gaussian_bounds(C,H_PRINTED,NOISE_VARIANCE)
        after=pairwise_gaussian_bounds(C@D.T,H_PRINTED@D.conj().T,NOISE_VARIANCE)
        np.testing.assert_allclose(after,before,rtol=0,atol=1e-13)

    def test_seeded_decoder_falls_between_mathematical_bounds(self):
        rng=np.random.default_rng(29245)
        n=75000
        idx=rng.integers(16,size=n)
        noise=np.sqrt(NOISE_VARIANCE/2)*(rng.normal(size=(n,2))+1j*rng.normal(size=(n,2)))
        for C in (rank_one(),equal_port_circular_psk16()):
            Y=C@H_PRINTED.T
            errors=0
            for k in range(0,n,3000):
                y=Y[idx[k:k+3000]]+noise[k:k+3000]
                d=np.sum(abs(y[:,None,:]-Y[None,:,:])**2,axis=2)
                errors+=np.count_nonzero(np.argmin(d,axis=1)!=idx[k:k+3000])
            empirical=errors/n
            lo,up=pairwise_gaussian_bounds(C,H_PRINTED,NOISE_VARIANCE)
            self.assertGreater(empirical,lo-0.002)
            self.assertLess(empirical,up+0.002)
        self.assertLess(circular_psk16_ser_exact(H_PRINTED,NOISE_VARIANCE),.052)

if __name__=='__main__':
    unittest.main()
