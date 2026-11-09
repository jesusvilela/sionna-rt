"""Fixed-trace aggregate union-envelope theorem and adversarial tests."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.special import erfc
from scipy.stats import norm
from mpmath import mp

EXP=Path(__file__).resolve().parents[1]/'experiments'
sys.path.insert(0,str(EXP))
import gram_uncertainty_certificate as base
from gram_decision_sweep import sweep

N0=base.N0
PAIRS={
 'rank_one':((0,1),(0,2)),
 'qpsk_squared':((0,1),(0,4)),
 'circular_16psk':((0,1),(0,7)),
}
PAULI=(
 np.array([[0,1],[1,0]],complex),
 np.array([[0,-1j],[1j,0]],complex),
 np.array([[1,0],[0,-1]],complex),
)


def source_gram(case):
    a=case['channel'];z=complex(a['cross_correlation_real'],a['cross_correlation_imag'])
    return np.array([[a['port_gain_split'][0],z],[z.conjugate(),a['port_gain_split'][1]]],complex)


def direct_union_upper(G,C):
    w,V=np.linalg.eigh(G)
    if w[0]<=0:raise ValueError('full-rank channel expected')
    H=(V*np.sqrt(w))@V.conj().T
    Y=C@H.T
    d=Y[:,None,:]-Y[None,:,:]
    d2=np.sum(abs(d)**2,axis=-1)
    p=norm.sf(np.sqrt(d2/(2*N0)))
    np.fill_diagonal(p,0.)
    return float(p.sum(axis=1).mean())


def trace_free_random(rng,eps):
    a=rng.normal(size=3);a/=np.linalg.norm(a)
    return eps*sum((x*S for x,S in zip(a,PAULI)),np.zeros((2,2),complex))


def pair_delta(G,C,pair1,pair2,eps,c):
    out=[]
    for i,j in (pair1,pair2):
        d=C[i]-C[j]
        n2=float(np.vdot(d,d).real)
        q=float(np.vdot(d,G@d).real)
        f=lambda x:.5*erfc(np.sqrt(x)/(2*np.sqrt(N0)))
        p_sep=f(q-eps*n2)
        p_common=f(q-eps*c*n2)
        out.append(p_sep-p_common)
    return out


def line_coherence(C,pair1,pair2):
    d1=C[pair1[0]]-C[pair1[1]]
    d2=C[pair2[0]]-C[pair2[1]]
    return float(abs(np.vdot(d1,d2))/(np.linalg.norm(d1)*np.linalg.norm(d2)))


class FixedTraceAggregateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=sweep()['cases']
        cls.books=base.books()
        if len(cls.cases)!=75:raise AssertionError('source grid changed')

    def test_pair_direction_coherence_is_frozen(self):
        expected={'rank_one':np.sin(np.pi/8),'qpsk_squared':0.,'circular_16psk':1.}
        for name,C in self.books.items():
            c=line_coherence(C,*PAIRS[name])
            self.assertAlmostEqual(c,expected[name],places=12)

    def test_two_line_theorem_gives_strict_noncollinear_union_gap(self):
        mins={name:float('inf') for name in ('rank_one','qpsk_squared')}
        counts={name:0 for name in mins}
        for case in self.cases:
            G=source_gram(case)
            eps=min(1e-3,float(np.linalg.eigvalsh(G)[0])/4)
            for name in mins:
                C=self.books[name]
                c=line_coherence(C,*PAIRS[name])
                self.assertLess(c,1-1e-12)
                ds=pair_delta(G,C,*PAIRS[name],eps,c)
                self.assertTrue(all(x>0 for x in ds),(name,ds))
                gap=min(ds)/8. # two ordered orientations divided by 16 symbols
                self.assertGreater(gap,0)
                mins[name]=min(mins[name],gap);counts[name]+=1
        self.assertEqual(counts,{'rank_one':75,'qpsk_squared':75})
        self.assertGreater(mins['rank_one'],1e-12)
        self.assertGreater(mins['qpsk_squared'],1e-12)
        print('FIXED_TRACE_STRICT_UNION_GAP_MIN',mins)

    def test_19200_random_traceless_perturbations_respect_quantitative_gap(self):
        rng=np.random.default_rng(20261109)
        total=0;min_observed_gap=float('inf');min_theorem_gap=float('inf')
        for case in self.cases:
            G=source_gram(case)
            lam=float(np.linalg.eigvalsh(G)[0])
            eps=min(1e-3,lam/4)
            for name in ('rank_one','qpsk_squared'):
                C=self.books[name]
                c=line_coherence(C,*PAIRS[name])
                ds=pair_delta(G,C,*PAIRS[name],eps,c)
                theorem_gap=min(ds)/8.
                sep=base.robust_interval(G,C,eps)['U_guaranteed']
                min_theorem_gap=min(min_theorem_gap,theorem_gap)
                for _ in range(128):
                    E=trace_free_random(rng,eps)
                    self.assertLess(abs(np.trace(E)),1e-12)
                    self.assertLess(abs(np.linalg.norm(E,2)-eps),1e-12)
                    Gtrue=G+E
                    self.assertGreater(np.linalg.eigvalsh(Gtrue)[0],0)
                    actual=direct_union_upper(Gtrue,C)
                    gap=sep-actual
                    min_observed_gap=min(min_observed_gap,gap)
                    self.assertGreaterEqual(gap+1e-12,theorem_gap)
                    total+=1
        self.assertEqual(total,19200)
        print('TRACE_FREE_GAP_CHECKS',total,'minimum_theorem_gap',min_theorem_gap,
              'minimum_observed_gap',min_observed_gap)

    def test_circular_single_beam_jointly_attains_both_separable_functionals(self):
        C=self.books['circular_16psk']
        beam=C[0]/np.linalg.norm(C[0])
        P=np.outer(beam,beam.conj())
        for case in self.cases:
            G=source_gram(case);eps=min(1e-3,float(np.linalg.eigvalsh(G)[0])/4)
            lo,hi=base.event_brackets(G,C,eps)
            mask=~np.eye(16,dtype=bool)
            for sign,target in ((1.,lo),(-1.,hi)):
                E=sign*eps*(2*P-np.eye(2))
                self.assertLess(abs(np.trace(E)),1e-12)
                self.assertAlmostEqual(np.linalg.norm(E,2),eps,places=12)
                actual=direct_pair_probabilities(G+E,C)
                self.assertLess(float(np.max(abs(actual[mask]-target[mask]))),1e-11)
                if sign<0:
                    sep=base.robust_interval(G,C,eps)['U_guaranteed']
                    self.assertLess(abs(direct_union_upper(G+E,C)-sep),1e-11)

    def test_80_digit_minimum_gap_is_not_float_subtraction_artifact(self):
        with mp.workdps(60):
            min_gap=mp.inf
            for case in self.cases:
                G=source_gram(case)
                lam=float(np.linalg.eigvalsh(G)[0])
                eps=mp.mpf(str(min(1e-3,lam/4)))
                for name in ('rank_one','qpsk_squared'):
                    C=self.books[name];pair1,pair2=PAIRS[name]
                    c=mp.mpf(str(line_coherence(C,pair1,pair2)))
                    deltas=[]
                    for i,j in (pair1,pair2):
                        d=C[i]-C[j]
                        n2=mp.mpf(str(float(np.vdot(d,d).real)))
                        q=mp.mpf(str(float(np.vdot(d,G@d).real)))
                        f=lambda x:mp.erfc(mp.sqrt(x)/(2*mp.sqrt(mp.mpf(str(N0)))))/2
                        deltas.append(f(q-eps*n2)-f(q-eps*c*n2))
                    gap=min(deltas)/8
                    self.assertGreater(gap,0)
                    min_gap=min(min_gap,gap)
        self.assertGreater(min_gap,mp.mpf('1e-12'))
        print('MP60_MINIMUM_CERTIFIED_TRACE_FREE_GAP',mp.nstr(min_gap,40))

    def test_colored_noise_does_not_inherit_h_frobenius_trace(self):
        H=np.eye(2,dtype=complex)/np.sqrt(2)
        Sigma=np.diag([1.,4.]).astype(complex)
        np.testing.assert_allclose(np.linalg.norm(H,'fro'),1.,atol=1e-15)
        Geff=H.conj().T@np.linalg.solve(Sigma,H)
        self.assertAlmostEqual(float(np.trace(Geff).real),.625,places=14)
        self.assertNotAlmostEqual(float(np.trace(Geff).real),1.,places=8)

    def test_two_stage_moving_frame_preserves_the_strict_gap(self):
        U1=base.su2(31,.71)
        U2=base.su2(17,-.49)@U1
        W=np.array([[1.2+.15j,.27-.13j],[.08+.22j,.82-.09j]],complex)
        L=np.linalg.cholesky(W@W.conj().T)
        maxerr=0.
        for index in (0,18,37,55,74):
            G=source_gram(self.cases[index]);eps=min(1e-3,np.linalg.eigvalsh(G)[0]/4)
            for name in ('rank_one','qpsk_squared','circular_16psk'):
                C=self.books[name];base_lo,base_hi=base.event_brackets(G,C,eps)
                mask=~np.eye(16,dtype=bool)
                for U in (U1,U2):
                    H=base.positive_root(G);Hn=W@H@U.conj().T;Sn=W@W.conj().T
                    Cn=C@U.T
                    Gw=np.linalg.solve(L,Hn).conj().T@np.linalg.solve(L,Hn)
                    lo,hi=base.event_brackets(Gw,Cn,eps)
                    maxerr=max(maxerr,float(np.max(abs(lo[mask]-base_lo[mask]))),
                               float(np.max(abs(hi[mask]-base_hi[mask]))))
        self.assertLess(maxerr,1e-10)

    def test_unrestricted_joint_endpoint_theorem_remains_distinct(self):
        for case in self.cases:
            G=source_gram(case);eps=min(1e-3,np.linalg.eigvalsh(G)[0]/4)
            for C in self.books.values():
                plus=direct_union_upper(G-eps*np.eye(2),C)
                sep=base.robust_interval(G,C,eps)['U_guaranteed']
                self.assertLess(abs(plus-sep),1e-11)


def direct_pair_probabilities(G,C):
    w,V=np.linalg.eigh(G);H=(V*np.sqrt(w))@V.conj().T
    Y=C@H.T;d=Y[:,None,:]-Y[None,:,:]
    d2=np.sum(abs(d)**2,axis=-1)
    p=norm.sf(np.sqrt(d2/(2*N0)))
    np.fill_diagonal(p,0.)
    return p


if __name__=='__main__':unittest.main()
