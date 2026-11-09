"""Independent and metamorphic controls for two nonunitary receiver frames.

No Sionna, no GPU; no inference of physical channel observations from synthetic G.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
import colored_noise_frame_audit as a
import frame_resource_audit as old

SOURCE = ROOT/'experiments'/'gram_sweep_2026-11-04.json'


def direct_score_whitened(y,H,C,S):
    """Independent colored ML via Cholesky whitening of each candidate residual."""
    chol = np.linalg.cholesky(S)
    outputs = C@H.T
    residual=y[:,None,:]-outputs[None,:,:]
    r=np.linalg.solve(chol,residual.reshape(-1,2).T)
    return np.sum(abs(r)**2,axis=0).reshape((len(y),len(C)))


class ColoredNoiseAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_dir=None
        cls.source_path=SOURCE
        if not SOURCE.is_file():
            # The 75-cell JSON is intentionally not checked in. Reproduce
            # it from the frozen source, then validate its exact SHA-256.
            cls.fixture_dir=tempfile.TemporaryDirectory()
            cls.source_path=Path(cls.fixture_dir.name)/'gram_sweep_2026-11-04.json'
            from gram_decision_sweep import main as generate_source
            generate_source(['--out',str(cls.source_path)])
        cls.data=old.load_source(cls.source_path)

    @classmethod
    def tearDownClass(cls):
        if cls.fixture_dir is not None:
            cls.fixture_dir.cleanup()

    def test_source_sha_and_frozen_grid(self):
        self.assertEqual(hashlib.sha256(self.source_path.read_bytes()).hexdigest(),a.SOURCE_SHA256)
        self.assertEqual(self.data['n_cases'],75)
        self.assertEqual(len(self.data['cases']),75)

    def test_known_answer_diagonal_covariance(self):
        H=np.diag([2.,3.]).astype(complex)
        S=np.diag([4.,9.]).astype(complex)
        np.testing.assert_allclose(a.effective_gram(H,S),np.eye(2),atol=1e-15)
        np.testing.assert_allclose(a.whitened_gram_independent(H,S),np.eye(2),atol=1e-15)

    def test_random_hermitian_covariance_two_independent_formulas(self):
        rng=np.random.default_rng(5743)
        for _ in range(80):
            z=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))
            S=z@z.conj().T+.5*np.eye(2)
            H=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))
            A=a.effective_gram(H,S)
            B=a.whitened_gram_independent(H,S)
            np.testing.assert_allclose(A,B,atol=2e-13,rtol=3e-13)
            d=rng.normal(size=2)+1j*rng.normal(size=2)
            L=np.linalg.cholesky(S)
            received_energy=np.linalg.norm(np.linalg.solve(L,H@d))**2
            self.assertAlmostEqual(np.vdot(d,A@d).real,received_energy,places=11)

    def test_two_nonunitary_receiver_transforms_and_two_input_unities_compose(self):
        G=old.source_gram(self.data['cases'][30]);H=a.spectral_channel(G)
        C=old.source_codebooks(G)['rank_one'];S=np.eye(2,dtype=complex)
        h1,s1,c1=a.stage_map(H,S,C,a.W1,a.U1)
        h2,s2,c2=a.stage_map(h1,s1,c1,a.W2,a.U2)
        hf,sf,cf=a.stage_map(H,S,C,a.W2@a.W1,a.U2@a.U1)
        for x,y in ((h2,hf),(s2,sf),(c2,cf)):
            np.testing.assert_allclose(x,y,atol=5e-15)
        np.testing.assert_allclose(a.effective_gram(h2,s2),a.U2@a.U1@G@a.U1.conj().T@a.U2.conj().T,atol=1e-12)

    def test_bad_covariances_and_singular_frame_fail_closed(self):
        H=np.eye(2,dtype=complex);C=old.source_codebooks(np.eye(2)/2)['rank_one']
        for bad in (np.zeros((2,2)),np.diag([1,-1]),np.diag([1,1e-15]),
                    np.array([[1,1j],[0,1]]),np.ones((3,3))):
            with self.assertRaises(ValueError):a.effective_gram(H,bad)
        with self.assertRaises(ValueError):a.stage_map(H,np.eye(2),C,np.array([[1,0],[0,0]]),a.U1)
        with self.assertRaises(ValueError):a.stage_map(H,np.eye(2),C,a.W1,2*np.eye(2))

    def test_direct_colored_ml_scores_invariant_with_random_proper_noise(self):
        rng=np.random.default_rng(222)
        H=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))
        z=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))
        S=z@z.conj().T+np.eye(2)
        C=old.source_codebooks(H.conj().T@H)['qpsk_squared']
        trials=700
        ix=rng.integers(0,16,size=trials)
        n=np.sqrt(a.N0/2)*(rng.normal(size=(trials,2))+1j*rng.normal(size=(trials,2)))
        Y=C[ix]@H.T+n@np.linalg.cholesky(S).T
        baseline=direct_score_whitened(Y,H,C,S)
        for _,W,U in a.STAGES:
            Hp,Sp,Cp=a.stage_map(H,S,C,W,U)
            Z=Y@W.T
            new=direct_score_whitened(Z,Hp,Cp,Sp)
            np.testing.assert_allclose(new,baseline,rtol=1e-10,atol=1e-10)
            np.testing.assert_array_equal(new.argmin(axis=1),baseline.argmin(axis=1))

    def test_transmitted_physical_port_energies_follow_input_unitaries(self):
        G=old.source_gram(self.data['cases'][2]);C=old.source_codebooks(G)
        U=a.U2@a.U1
        P=tuple(U@p@U.conj().T for p in old.PORTS)
        for book in C.values():
            np.testing.assert_allclose(old.physical_powers(book,old.PORTS),
                                       old.physical_powers(book@U.T,P),atol=3e-14)
            np.testing.assert_allclose(np.linalg.norm(book,axis=1),np.linalg.norm(book@U.T,axis=1),atol=3e-14)

    def test_printed_single_channel_blackbox(self):
        result=a.printed_control()
        self.assertEqual(result['n_symbols'],2048)
        self.assertEqual(result['decision_differences'],{'first':0,'second':0})
        self.assertLess(result['max_full_symbol_metric_delta'],1e-12)
        self.assertIn('single',result['noise_model']+' single')

    def test_frozen_375_pairs_and_all_120_distances(self):
        result=a.run(self.data)
        self.assertEqual(result['source_cells'],75)
        self.assertEqual(result['comparisons_per_frame'],375)
        self.assertEqual(len(result['observations']),75)
        for name in ('first','second'):
            for key in ('max_gram_delta','max_d2_delta','max_bound_delta','max_port_energy_delta'):
                self.assertLess(result['stages'][name][key],1e-10,key)
            for row in result['observations']:
                self.assertEqual(row['source_decisions'],row['moving_frames'][name]['corrected_decisions'])

    def test_deliberately_wrong_white_noise_metric_changes_certificates(self):
        result=a.run(self.data)
        for name in ('first','second'):
            s=result['stages'][name]
            self.assertGreater(s['n_wrong_decision_changes'],0)
            self.assertGreater(s['n_trace_wrong_decision_changes'],0)
            self.assertGreater(s['wrong_strict_flips'],0)
            self.assertGreater(s['trace_wrong_strict_flips'],0)
            self.assertGreater(s['n_wrong_cells_any_change'],0)
            self.assertLessEqual(s['n_wrong_decision_changes'],375)
            self.assertLessEqual(s['n_trace_wrong_decision_changes'],375)

    def test_naive_trace_correction_changes_no_physical_parameters(self):
        # The equal-trace control removes a gain-scaling confound but remains
        # the WRONG physical metric. It should have exactly trace G_eff.
        H=a.spectral_channel(old.source_gram(self.data['cases'][44]))
        for _,W,U in a.STAGES:
            Hp,S,Cp=a.stage_map(H,np.eye(2),old.source_codebooks(H.conj().T@H)['rank_one'],W,U)
            expected=a.effective_gram(Hp,S)
            bad=Hp.conj().T@Hp
            adjusted=bad*(np.trace(expected).real/np.trace(bad).real)
            self.assertAlmostEqual(np.trace(adjusted).real,np.trace(expected).real,places=13)
            self.assertGreater(np.linalg.norm(adjusted-expected),1e-6)

    def test_logical_scope_no_measured_munich_claim(self):
        r=a.run(self.data)
        self.assertTrue(r['evidence']['no_Munich_16_rx_matrices'])
        self.assertTrue(r['evidence']['no_novel_physical_result'])
        self.assertEqual(r['status'],'CPU_synthetic_whitening_covariance_audit_only')


if __name__=='__main__':unittest.main()
