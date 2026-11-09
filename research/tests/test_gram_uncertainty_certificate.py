"""Adversarial CPU tests of nonparametric Gram uncertainty certificates."""
import json
import os
import sys
import unittest
from pathlib import Path
from collections import Counter

import numpy as np
from scipy.special import erfc

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parent/'experiments'))
import gram_uncertainty_certificate as m

GRID_FILE=Path(os.environ.get('SIONNA_GRAM_SWEEP',
    HERE.parent/'experiments'/'gram_sweep_2026-11-04.json'))


def gram_from_case(c):
    a=c['channel']
    z=complex(a['cross_correlation_real'],a['cross_correlation_imag'])
    return np.array([[a['port_gain_split'][0],z],[z.conjugate(),a['port_gain_split'][1]]],complex)


class GramUncertaintyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if GRID_FILE.is_file():
            cls.grid=json.loads(GRID_FILE.read_text())
        else:
            from gram_decision_sweep import sweep
            cls.grid=sweep()
        assert len(cls.grid['cases'])==75
        cls.books=m.books()

    def test_frozen_grid_baseline_reproduces_decisions_and_complete_bounds(self):
        count={'qpsk_squared':Counter(),'circular_16psk':Counter()}
        for case in self.grid['cases']:
            G=gram_from_case(case)
            ref={name:m.robust_interval(G,C,0.) for name,C in self.books.items()}
            for label in count:
                decision,_=m.robust_winner(ref['rank_one'],ref[label])
                historical=case['rank_one_vs'][label]
                historical=historical if historical!='undecided' else 'abstain'
                self.assertEqual(decision,historical)
                count[label][decision]+=1
            for name in ref:
                old=case['bounds'][name]
                self.assertAlmostEqual(ref[name]['L_guaranteed'],old['lower'],places=11)
                self.assertAlmostEqual(ref[name]['U_guaranteed'],old['upper'],places=11)
        self.assertEqual(count['qpsk_squared'],Counter({'A':18,'B':46,'abstain':11}))
        self.assertEqual(count['circular_16psk'],Counter({'A':45,'B':18,'abstain':12}))

    def test_robust_certification_only_abstains_as_radius_grows(self):
        counts={}
        for eps in (0.,.0001,.001):
            hist={'qpsk_squared':Counter(),'circular_16psk':Counter()}
            for case in self.grid['cases']:
                G=gram_from_case(case)
                ref={name:m.robust_interval(G,C,eps) for name,C in self.books.items()}
                for name in hist:
                    decision,_=m.robust_winner(ref['rank_one'],ref[name])
                    hist[name][decision]+=1
                    if eps>0 and decision!='abstain':
                        zero={k:m.robust_interval(G,C,0.) for k,C in self.books.items()}
                        self.assertEqual(decision,m.robust_winner(zero['rank_one'],zero[name])[0])
            counts[str(eps)]={name:dict(h) for name,h in hist.items()}
        print('GRID_REMAINING_CERTIFICATES',json.dumps(counts,sort_keys=True))

    def test_seeded_225_psd_perturbations_respect_pairwise_and_ser_envelopes(self):
        rng=np.random.default_rng(20261109)
        checks=0
        for case in self.grid['cases']:
            base=gram_from_case(case)
            eps=min(.001,.4*np.linalg.eigvalsh(base)[0])
            for _ in range(3):
                z=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))
                D=(z+z.conj().T)/2
                D*=eps*rng.uniform(.2,.99)/np.linalg.norm(D,2)
                actual=base+D
                self.assertLessEqual(np.linalg.norm(D,2),eps*(1+1e-12))
                self.assertGreater(np.linalg.eigvalsh(actual)[0],0)
                C=self.books[['rank_one','qpsk_squared','circular_16psk'][checks%3]]
                plo,phi=m.event_brackets(base,C,eps)
                d=C[:,None,:]-C[None,:,:]
                q=np.einsum('...i,ij,...j->...',d.conj(),actual,d).real
                p=.5*erfc(np.sqrt(np.maximum(q,0))/(2*np.sqrt(m.N0)))
                mask=~np.eye(16,dtype=bool)
                self.assertTrue(np.all(plo[mask]<=p[mask]+1e-14))
                self.assertTrue(np.all(p[mask]<=phi[mask]+1e-14))
                cert=m.robust_interval(base,C,eps)
                p=np.where(mask,p,0.)
                lower=float(np.max(p,axis=1).mean())
                upper=float(p.sum(axis=1).mean())
                self.assertLessEqual(cert['L_guaranteed'],lower+1e-14)
                self.assertLessEqual(upper,cert['U_guaranteed']+1e-14)
                checks+=1
        self.assertEqual(checks,225)

    def test_closed_operator_norm_ball_requires_positive_definiteness(self):
        G=m.principal_gram(.01,22.5)
        lam=float(np.linalg.eigvalsh(G)[0])
        with self.assertRaises(ValueError):m.robust_interval(G,self.books['rank_one'],lam)
        with self.assertRaises(ValueError):m.robust_interval(G,self.books['rank_one'],-1e-4)
        with self.assertRaises(ValueError):m.robust_interval(np.zeros((2,2)),self.books['rank_one'],0.)
        with self.assertRaises(ValueError):m.robust_interval(np.array([[1,2j],[2,1]],complex),self.books['rank_one'],0.)
        with self.assertRaises(ValueError):m.robust_interval(G,np.zeros((3,2)),0.)

    def test_two_counterworld_robust_radii_and_pd_cliff(self):
        r=m.report()
        self.assertTrue(r['all_positive_gate'])
        self.assertEqual(r['original_munich_matrices_recovered'],0)
        a=r['two_counterworlds']['A_rank_one_better']
        b=r['two_counterworlds']['B_rank_one_worse']
        self.assertAlmostEqual(a['smallest_certified_radius'],.0203648167138,places=9)
        self.assertAlmostEqual(b['smallest_certified_radius'],.0099009900990,places=9)
        self.assertFalse(a['comparisons']['qpsk_squared']['ceiling_is_PD_gate'])
        self.assertTrue(b['comparisons']['qpsk_squared']['ceiling_is_PD_gate'])
        for world in (a,b):
            self.assertLess(world['moving_frame_max_delta_at_half_budget'],1e-10)
            for comp in world['comparisons'].values():
                self.assertLess(comp['numerical_bracket_width'],1e-9)
                self.assertGreater(comp['nominal_strict_margin'],.01)

    def test_certificate_fails_above_A_qpsk_limit_without_retuning(self):
        G=m.worlds()['A_rank_one_better']
        C=self.books
        x=m.radius_for_witness(G,C,'qpsk_squared','A')
        e=x['certified_radius_lower']+1e-4
        a=m.ensemble_bounds(G,C['rank_one'],e)
        b=m.ensemble_bounds(G,C['qpsk_squared'],e)
        self.assertEqual(m.robust_winner(a,b)[0],'abstain')

    def test_independent_direct_H_whitening_for_two_site_local_changes(self):
        for world in m.worlds().values():
            for i in (0,1,7,15):
                G=world[i]
                H=m.positive_root(G)
                for stage,U in enumerate((m.su2(31,.71),m.su2(17,-.49)@m.su2(31,.71)),1):
                    W=m.site_W(i,stage)
                    Hp=W@H@U.conj().T
                    Sigma=W@W.conj().T
                    L=np.linalg.cholesky(Sigma)
                    Hwhite=np.linalg.solve(L,Hp)
                    for name,C in self.books.items():
                        Cp=C@U.T
                        Y=C@H.T
                        Yw=Cp@Hwhite.T
                        D=Y[:,None,:]-Y[None,:,:]
                        Dw=Yw[:,None,:]-Yw[None,:,:]
                        self.assertLess(np.max(abs(np.sum(abs(D)**2,axis=-1)-np.sum(abs(Dw)**2,axis=-1))),1e-12)

    def test_false_input_frame_without_codebook_transport_changes_distances(self):
        G=m.PRINTED.conj().T@m.PRINTED
        U=m.su2(31,.71)
        C=self.books['rank_one']
        d=C[:,None,:]-C[None,:,:]
        baseline=np.einsum('...i,ij,...j->...',d.conj(),G,d).real
        false=np.einsum('...i,ij,...j->...',d.conj(),U@G@U.conj().T,d).real
        self.assertGreater(np.max(abs(baseline-false)),.1)

    def test_rounding_gram_radius_conservative(self):
        E=np.sqrt(8)*5e-9
        self.assertAlmostEqual(m.printed_rounding_gram_radius(),2*np.linalg.norm(m.PRINTED,'fro')*E+E*E,places=18)
        self.assertLess(m.printed_rounding_gram_radius(),3e-8)

    def test_no_silent_file_overwrite(self):
        from unittest.mock import patch
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as d:
            p=Path(d)/'result.json';p.write_text('preexisting')
            with patch.object(sys,'argv',['audit','--out',str(p)]):
                with self.assertRaises(FileExistsError):m.main()
            self.assertEqual(p.read_text(),'preexisting')

if __name__=='__main__':unittest.main()
