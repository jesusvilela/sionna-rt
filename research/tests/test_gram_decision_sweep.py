"""Analytic and metamorphic regression of every Gram/oracle/sweep component."""
import json
import math
from pathlib import Path
import sys
import tempfile
from unittest import TestCase, main

import numpy as np
from scipy.special import ndtr

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import gram_decision_sweep as g
import export_munich_16 as exp


class GramTests(TestCase):
    def test_codebooks_match_source_construction_and_per_port_power(self):
        C=g.fixed_codebooks()
        self.assertEqual(set(C),{'rank_one','qpsk_squared','circular_16psk'})
        for name,x in C.items():
            self.assertEqual(x.shape,(16,2))
            np.testing.assert_allclose(np.mean(np.abs(x)**2,axis=0),.5,atol=1e-14)
            np.testing.assert_allclose(np.sum(np.abs(x)**2,axis=1),1,atol=1e-14)
        np.testing.assert_allclose(g.stokes(C['rank_one'])[:,1],0,atol=1e-14)
        np.testing.assert_allclose(g.stokes(C['circular_16psk'])[:,1],1,atol=1e-14)
        self.assertAlmostEqual(np.min(np.linalg.norm(C['rank_one'][:,None,:]-C['rank_one'][None,:,:]+
            np.eye(16)[:,:,None]*100,axis=-1)),np.sqrt(2-np.sqrt(2)),places=11)

    def test_stokes_matches_direct_realification_for_complex_arrays(self):
        z=np.array([[1+2j,3-1j],[0+1j,0+2j]])
        S=g.stokes(z)
        self.assertEqual(S.shape,(2,3))
        self.assertAlmostEqual(S[0,1],2*np.imag(np.conj(z[0,0])*z[0,1]))
        self.assertAlmostEqual(S[0,2],abs(z[0,0])**2-abs(z[0,1])**2)

    def test_generated_gram_has_frozen_spectrum_and_intended_principal_vector(self):
        for r in g.RATIOS:
            for theta in (0,22.5,45,67.5,90):
                for phi in (0,90,180):
                    G=g.gram_from_spectrum(r,theta,phi)
                    w=np.linalg.eigvalsh(G)
                    self.assertAlmostEqual(w[0]/w[1],r,places=10)
                    self.assertAlmostEqual(w.sum(),1,places=12)
                    ratio,th,p,trace=g.gram_spectrum(G)
                    self.assertAlmostEqual(ratio,r,places=10)
                    self.assertAlmostEqual(trace,1,places=12)
                    if r<1 and theta in (22.5,45,67.5):
                        self.assertAlmostEqual(th,theta,places=10)
                        self.assertAlmostEqual(np.sin(np.deg2rad(p-phi)),0,places=10)
                    if r==1:
                        self.assertIsNone(th)
                        self.assertIsNone(p)
        for invalid in (-.1,0,1.1):
            with self.assertRaises(ValueError):g.gram_from_spectrum(invalid,30,0)

    def test_gram_validation_rejects_non_psd_nonhermitian_and_nonfinite(self):
        for bad in (np.array([[1,2],[2,-1.]]),np.array([[1,1j],[0,1]]),
                    np.array([[np.inf,0],[0,1]])):
            with self.assertRaises(ValueError):g._validated_gram(bad)
        with self.assertRaises(ValueError):g._validated_gram(np.zeros((2,2)))
        with self.assertRaises(ValueError):g._validated_gram(np.eye(3))
        np.testing.assert_allclose(g._validated_gram(np.diag([1.,0.])),np.diag([1.,0.]))

    def test_direct_gram_quadratic_and_independent_stokes_form_match_channel(self):
        rng=np.random.default_rng(1011)
        for _ in range(30):
            H=rng.normal(size=(3,2))+1j*rng.normal(size=(3,2))
            G=H.conj().T@H
            delta=rng.normal(size=(16,16,2))+1j*rng.normal(size=(16,16,2))
            v0=g.gram_quadratic(G,delta)
            v1=g.gram_quadratic_stokes(G,delta)
            direct=np.sum(np.abs(np.einsum('ab,...b->...a',H,delta))**2,axis=-1)
            np.testing.assert_allclose(v0,v1,atol=2e-13,rtol=2e-13)
            np.testing.assert_allclose(v1,direct,atol=5e-13,rtol=2e-13)
        with self.assertRaises(ValueError):g.gram_quadratic(np.eye(2),np.ones((3,3)))
        # Cross-polarization Im term must *not* drop out.
        G=g.gram_from_spectrum(.1,45,90)
        d=np.array([1,1j])
        self.assertAlmostEqual(g.gram_quadratic(G,d),g.gram_quadratic_stokes(G,d))

    def test_principal_eigenbeam_maximizes_gain(self):
        rng=np.random.default_rng(999)
        for _ in range(20):
            H=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))
            G=H.conj().T@H
            v=g.principal_beam(G)
            np.testing.assert_allclose(np.vdot(v,v),1,atol=1e-14)
            self.assertAlmostEqual(np.vdot(v,G@v).real,np.linalg.eigvalsh(G)[-1],places=10)
        np.testing.assert_allclose(g.principal_beam(np.eye(2)),[1,0])

    def test_equal_port_phase_analytic_oracle_dominates_untuned_phases(self):
        for r in (.01,.1,.8,1.):
            for theta in (0,20,45,67):
                for phi in (0,90,175):
                    G=g.gram_from_spectrum(r,theta,phi)
                    v=g.equal_port_beam(G)
                    np.testing.assert_allclose(np.abs(v)**2,[.5,.5],atol=1e-14)
                    e=np.vdot(v,G@v).real
                    all_gain=[np.vdot(w,G@w).real for w in
                              (np.array([1,np.exp(1j*p)])/np.sqrt(2) for p in np.linspace(0,2*np.pi,127))]
                    self.assertGreaterEqual(e+1e-12,max(all_gain))
        # A phase that aligns to G12 must use minus arg(G12), not plus.
        G=g.gram_from_spectrum(.1,45,90)
        self.assertGreater(np.vdot(g.equal_port_beam(G),G@g.equal_port_beam(G)).real,.9)

    def test_beamed_constellations_resource_parity_and_peak_caveat(self):
        v=np.array([1,1j])/np.sqrt(2)
        p=g.beam_psk16(v);q=g.beam_qam16(v)
        for C in (p,q):
            self.assertEqual(C.shape,(16,2))
            self.assertAlmostEqual(np.mean(np.sum(abs(C)**2,axis=1)),1)
            np.testing.assert_allclose(np.mean(abs(C)**2,axis=0),[.5,.5])
        np.testing.assert_allclose(np.sum(abs(p)**2,axis=1),1,atol=1e-13)
        self.assertAlmostEqual(np.max(np.sum(abs(q)**2,axis=1)),1.8,places=12)
        with self.assertRaises(ValueError):g.beam_psk16(np.array([1,1]))
        with self.assertRaises(ValueError):g.beam_qam16(np.array([1,1]))

    def test_oracles_and_fixed_books_all_present(self):
        G=g.gram_from_spectrum(.02,30,45)
        C=g.codebooks_for_gram(G)
        self.assertEqual(set(C),set(g.LABELS))
        self.assertTrue(all(len(x)==16 for x in C.values()))
        np.testing.assert_allclose(abs(C['equal_port_psk16_oracle'])**2,.5,atol=1e-12)
        self.assertAlmostEqual(np.mean(np.sum(abs(C['eigen_qam16_oracle'])**2,axis=1)),1)

    def test_pep_bounds_match_printed_notebook_and_scipy_norm_sf(self):
        H=np.array([[.92982936+.33584663j,.01006189+.00676795j],
                    [-.06026739-.00641686j,-.13674566-.01022593j]])
        G=H.conj().T@H
        C=g.codebooks_for_gram(G)
        rank=g.gaussian_ser_bounds(G,C['rank_one'],.01)
        circ=g.gaussian_ser_bounds(G,C['circular_16psk'],.01)
        qpsk=g.gaussian_ser_bounds(G,C['qpsk_squared'],.01)
        self.assertAlmostEqual(rank['lower'],.06808739425621839,places=11)
        self.assertAlmostEqual(circ['upper'],.05182352183597337,places=11)
        self.assertAlmostEqual(qpsk['lower'],.16517190571537954,places=11)
        self.assertEqual(g.bound_decision(rank,circ),'B')
        self.assertEqual(g.bound_decision(rank,qpsk),'A')
        # independent SciPy survival function for a fixed pair PEP
        d=C['rank_one'][0]-C['rank_one'][4]
        dist=np.sqrt(np.vdot(H@d,H@d).real)
        expected=float(ndtr(-dist/np.sqrt(2*.01)))
        actual=float(.5*math.erfc(dist/(2*math.sqrt(.01))))
        self.assertAlmostEqual(expected,actual,places=15)
        with self.assertRaises(ValueError):g.gaussian_ser_bounds(G,C['rank_one'],0)
        with self.assertRaises(ValueError):g.gaussian_ser_bounds(G,C['rank_one'][:15],.01)

    def test_bounds_are_invariant_under_compensated_input_unitary(self):
        G=g.gram_from_spectrum(.03,40,60)
        rng=np.random.default_rng(20)
        z=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))
        U,R=np.linalg.qr(z);U=U@np.diag(np.exp(-1j*np.angle(np.diag(R))))
        C=g.fixed_codebooks()['rank_one']
        before=g.gaussian_ser_bounds(G,C,.01)
        transformed=g.gaussian_ser_bounds(U@G@U.conj().T,C@U.T,.01)
        for key in ('lower','upper','min_d2'):
            self.assertAlmostEqual(before[key],transformed[key],places=12)

    def test_bound_decision_abstains_unless_intervals_disjoint(self):
        a={'lower':.1,'upper':.2};b={'lower':.3,'upper':.4};c={'lower':.15,'upper':.25}
        self.assertEqual(g.bound_decision(a,b),'A')
        self.assertEqual(g.bound_decision(b,a),'B')
        self.assertEqual(g.bound_decision(a,c),'undecided')
        self.assertEqual(g.bound_decision(a,{'lower':.2,'upper':.3}),'undecided')

    def test_describe_gram_and_evaluate_single_fixed_cases(self):
        a=g.describe_gram(np.eye(2)/2)
        self.assertEqual(a['eigenvalue_ratio'],1)
        self.assertIsNone(a['principal_theta_deg'])
        self.assertIsNone(a['principal_phi_deg'])
        b=g.evaluate_gram(g.gram_from_spectrum(.1,0,0))
        self.assertEqual(b['rank_one_vs']['qpsk_squared'],'A')
        self.assertEqual(b['rank_one_vs']['circular_16psk'],'A')
        c=g.evaluate_gram(g.gram_from_spectrum(1,45,90))
        self.assertEqual(c['rank_one_vs']['qpsk_squared'],'B')

    def test_fixed_sweep_and_every_certified_pair_is_disjoint(self):
        data=g.sweep()
        self.assertEqual(data['n_cases'],75)
        self.assertEqual(len(data['cases']),75)
        for name,counts in data['summary'].items():
            self.assertEqual(sum(counts.values()),75)
            self.assertIn(name,g.LABELS)
        for point in data['cases']:
            a=point['bounds']['rank_one']
            for name,decision in point['rank_one_vs'].items():
                b=point['bounds'][name]
                if decision=='A':self.assertLess(a['upper']+1e-12,b['lower'])
                elif decision=='B':self.assertLess(b['upper']+1e-12,a['lower'])
                else:self.assertFalse(a['upper']+1e-12<b['lower'] or b['upper']+1e-12<a['lower'])

    def test_archive_provenance_gate_rejects_fake_and_accepts_typed_synthetic_fixture(self):
        H=np.tile(np.eye(2,dtype=complex)[None,:,:]/np.sqrt(2),(16,1,1))
        with tempfile.TemporaryDirectory() as tmp:
            archive=Path(tmp)/'channels.npz'
            with self.assertRaises(ValueError):g.evaluate_receiver_npz(archive)
            exp.archive_export(archive,H,np.ones(16,dtype=bool),{'mode':'test-fixture'})
            # A structurally valid synthetic fixture is NOT Munich evidence.
            with self.assertRaisesRegex(ValueError,'synthetic fixture'):
                g.evaluate_receiver_npz(archive)
            output=g.evaluate_receiver_npz(archive,allow_synthetic_fixture=True)
            self.assertEqual(output['status'],'synthetic_fixture_only')
            self.assertFalse(output['origin_authenticated'])
            self.assertEqual(output['n_receivers'],16)
            self.assertEqual(len(output['receivers']),16)
            self.assertEqual(output['receivers'][15]['y_group'],7)
            # Mutate checksum provenance: must fail rather than silently accept.
            meta_path=archive.with_suffix('.manifest.json')
            meta=json.loads(meta_path.read_text());meta['archive_sha256']='deadbeef'
            meta_path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'SHA256'):
                g.evaluate_receiver_npz(archive,allow_synthetic_fixture=True)

    def test_main_records_no_real_receiver_results_without_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'result.json'
            g.main(['--out',str(path)])
            x=json.loads(path.read_text())
            self.assertEqual(x['per_receiver']['status'],'not_available')
            self.assertEqual(x['summary'],g.sweep()['summary'])
            with self.assertRaises(FileExistsError):g.main(['--out',str(path)])


if __name__=='__main__':main()
