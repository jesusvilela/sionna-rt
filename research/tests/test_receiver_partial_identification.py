"""Counterworld tests: no Sionna, no artificial promotion of scene authenticity."""
from __future__ import annotations
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from mpmath import mp

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import receiver_partial_identification as p

mp.dps=60


def independent_mp_channels():
    """Direct analytic Hermitian square root, not NumPy eigendecomposition."""
    h0=((mp.mpc('0.92982936','0.33584663'),mp.mpc('0.01006189','0.00676795')),
        (mp.mpc('-0.06026739','-0.00641686'),mp.mpc('-0.13674566','-0.01022593')))
    ans={'H0':h0}
    for name,r,t in [('A_rank_one_better',mp.mpf('.1'),mp.mpf('0')),
                     ('B_rank_one_worse',mp.mpf('.01'),mp.pi/8)]:
        a=mp.sqrt(1/(1+r));b=mp.sqrt(r/(1+r))
        u=(mp.cos(t),mp.sin(t));v=(-u[1],u[0])
        H=tuple(tuple(a*u[i]*u[j]+b*v[i]*v[j] for j in range(2)) for i in range(2))
        ans[name]=H
    return ans


def independent_mp_books():
    g=[mp.mpc(a,b)/mp.sqrt(2) for a,b in ((1,1),(1,-1),(-1,1),(-1,-1))]
    q=[(a/mp.sqrt(2),b/mp.sqrt(2)) for a in g for b in g]
    rank=[]
    for m in range(8):
        v=(mp.cos(mp.pi*m/4),mp.sin(mp.pi*m/4))
        for s in (1,-1):
            rank.append((v[0]*(1+mp.j*s)/mp.sqrt(2),
                         v[1]*(1+mp.j*s)/mp.sqrt(2)))
    circ=[(mp.exp(2*mp.j*mp.pi*k/16)/mp.sqrt(2),
           mp.j*mp.exp(2*mp.j*mp.pi*k/16)/mp.sqrt(2)) for k in range(16)]
    return {'rank_one':rank,'qpsk_squared':q,'circular_16psk':circ}


def independent_mp_bounds(H,C):
    """Full 16×15 direct-H event enumeration in 60-digit mpmath."""
    y=[(H[0][0]*x[0]+H[0][1]*x[1],H[1][0]*x[0]+H[1][1]*x[1]) for x in C]
    lower=[];upper=[]
    for i in range(16):
        probs=[]
        for j in range(16):
            if i==j:continue
            ds=mp.fsum((abs(y[i][a]-y[j][a])**2 for a in (0,1)))
            probs.append(mp.erfc(mp.sqrt(ds)/(2*mp.sqrt(mp.mpf('.01'))))/2)
        lower.append(max(probs));upper.append(mp.fsum(probs))
    return float(mp.fsum(lower)/16),float(mp.fsum(upper)/16)


class ReceiverPartialIdentificationTests(unittest.TestCase):
    def test_both_worlds_share_exact_printed_site_and_have_15_alive_normalized(self):
        w=p.worlds()
        self.assertEqual(set(w),set(p.WITNESSES))
        for channels in w.values():
            self.assertEqual(len(channels),16)
            np.testing.assert_array_equal(channels[0],p.PRINTED)
            for H in channels[1:]:
                self.assertGreater(np.linalg.svd(H)[1].min(),0)
                np.testing.assert_allclose(np.linalg.norm(H,'fro'),1,atol=1e-14)

    def test_no_extra_physical_scene_claim_and_same_original_codebooks(self):
        out=p.report()
        self.assertTrue(out['not_independently_verified_munich_data'])
        self.assertEqual(out['real_original_munich_matrices_recovered'],0)
        self.assertIn('NOT established',out['H'])
        for c in p.source_codebooks().values():
            np.testing.assert_allclose(np.sum(abs(c)**2,axis=1),1,atol=1e-13)
            np.testing.assert_allclose(np.mean(abs(c)**2,axis=0),[.5,.5],atol=1e-13)

    def test_two_strict_opposite_average_ser_orderings(self):
        out=p.report()
        self.assertEqual(len(out['worlds']),2)
        for name,world in out['worlds'].items():
            for cmp,d in world['comparisons'].items():
                winner='rank_one' if name.startswith('A_') else cmp
                self.assertEqual(d['winner'],winner)
                self.assertGreater(d['gap'],1e-4)
            self.assertGreater(world['min_rounding_adjusted_gap'],1e-4)

    def test_independent_60_digit_direct_channel_bounds_for_both_worlds(self):
        ref=independent_mp_channels();books=independent_mp_books()
        for name,channels in p.worlds().items():
            for key,C in p.source_codebooks().items():
                lo0,up0=independent_mp_bounds(ref['H0'],books[key])
                los,ups=independent_mp_bounds(ref[name],books[key])
                expected=((lo0+15*los)/16,(up0+15*ups)/16)
                actual=p.mixture_bounds(channels)[key]
                self.assertAlmostEqual(actual['lower'],expected[0],places=11)
                self.assertAlmostEqual(actual['upper'],expected[1],places=11)

    def test_original_h0_interval_matches_archived_high_precision(self):
        expected={
            'rank_one':(.06808739425621835466,.10646272262942519657),
            'qpsk_squared':(.16517190571537943992,.41465776263146507660),
            'circular_16psk':(.02584411332672455628,.05182352183597335205)
        }
        for name,C in p.source_codebooks().items():
            bounds=p.direct_bounds(p.PRINTED,C)
            self.assertAlmostEqual(bounds['lower'],expected[name][0],places=11)
            self.assertAlmostEqual(bounds['upper'],expected[name][1],places=11)

    def test_direct_H_and_gram_are_equivalent_under_both_sitewise_frames(self):
        books=p.source_codebooks()
        for world in p.worlds().values():
            for site in (0,1,7,15):
                H=world[site]
                for C in books.values():
                    self.assertLess(p.check_direct_vs_gram(H,C),1e-12)
                    for level in (1,2):
                        U=p.su2(np.deg2rad(17),.49)
                        W=p.site_receive_change(site,level)
                        hp=W@H@U.conj().T
                        sigma=W@W.conj().T
                        self.assertLess(p.check_direct_vs_gram(hp,C@U.T,sigma),1e-12)

    def test_all_average_bound_intervals_survive_two_moving_frames(self):
        for channels in p.worlds().values():
            data=p.mixture_bounds(channels)
            checks=p.frame_recheck(channels,data,p.source_codebooks())
            self.assertEqual(len(checks),2)
            for check in checks:self.assertLess(check['max_metric_port_and_bounds_deviation'],1e-10)

    def test_rounding_budget_correctly_scales_by_one_of_sixteen(self):
        a=p.rounding_allowance()
        expected=2*math.sqrt(8)*5e-9/math.sqrt(4*math.pi*p.N0)
        self.assertAlmostEqual(a['ensemble_gap_loss_max'],expected,places=16)
        # One printed H0, so aggregate L and U each carry factor 1/16.
        self.assertLess(a['ensemble_gap_loss_max'],1e-6)
        for world in p.report()['worlds'].values():
            self.assertGreater(min(v['gap'] for v in world['comparisons'].values()),a['ensemble_gap_loss_max'])

    def test_witness_specific_number_of_unknown_sites_needed_for_flip(self):
        books=p.source_codebooks()
        reference=[p.PRINTED]*16
        self.assertEqual(p.comparisons(p.mixture_bounds(reference,books))['qpsk_squared']['winner'],'rank_one')
        self.assertEqual(p.comparisons(p.mixture_bounds(reference,books))['circular_16psk']['winner'],'circular_16psk')
        good=p.worlds()['A_rank_one_better'][1]
        bad=p.worlds()['B_rank_one_worse'][1]
        a_at13=p.comparisons(p.mixture_bounds([p.PRINTED]*(16-13)+[good]*13,books))
        a_at14=p.comparisons(p.mixture_bounds([p.PRINTED]*(16-14)+[good]*14,books))
        b_at10=p.comparisons(p.mixture_bounds([p.PRINTED]*(16-10)+[bad]*10,books))
        b_at11=p.comparisons(p.mixture_bounds([p.PRINTED]*(16-11)+[bad]*11,books))
        self.assertNotEqual(a_at13['circular_16psk']['winner'],'rank_one')
        self.assertEqual(a_at14['circular_16psk']['winner'],'rank_one')
        self.assertNotEqual(b_at10['qpsk_squared']['winner'],'qpsk_squared')
        self.assertEqual(b_at11['qpsk_squared']['winner'],'qpsk_squared')

    def test_malformed_channel_and_noise_covariance_fail(self):
        with self.assertRaises(ValueError):p.mixture_bounds([p.PRINTED])
        C=p.source_codebooks()['rank_one']
        for bad in (np.zeros((2,2)),np.diag([1,-1]),np.array([[1,1j],[0,1]])):
            with self.assertRaises(ValueError):p.direct_bounds(p.PRINTED,C,bad)

    def test_report_refuses_silent_output_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'test.json';output.write_text('evidence')
            with self.assertRaises(FileExistsError):p.main(['--out',str(output)])
            self.assertEqual(output.read_text(),'evidence')


if __name__=='__main__':unittest.main()
