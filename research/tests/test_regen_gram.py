"""Deterministic CPU/analytic tests. No ray tracing or network required."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
import munich_cpu_runner as m
import gram_boundary_tightening as g

ARCHIVE=Path('/mnt/data/sionna_gram_checkpoint/gram_sweep_2026-11-04.json')

class MunichRunnerTests(unittest.TestCase):
    def test_pins_and_cpu_only(self):
        status=m.pins()
        self.assertEqual(set(status),{'sionna-rt','mitsuba','drjit'})
        self.assertEqual(m.CPU_VARIANT,'llvm_ad_mono_polarized')
        self.assertEqual(m.SIONNA_PIN,'2.2.0')

    def test_original_and_reduced_config_never_confused(self):
        old=m.config(m.MODES[0]);det=m.config(m.MODES[1]);low=m.config(m.MODES[2])
        self.assertTrue(old['source_match']);self.assertFalse(old['deterministic'])
        self.assertTrue(det['deterministic']);self.assertEqual(det['max_depth'],5)
        self.assertEqual(low['max_depth'],2);self.assertEqual(low['samples_per_src'],20000)
        self.assertFalse(low['source_match'])
        with self.assertRaises(ValueError):m.config('fake')

    def test_positions_original_source(self):
        p=m.receiver_positions()
        self.assertEqual(p.shape,(16,3))
        np.testing.assert_allclose(p[[0,1,14,15]],[[25,15,1.5],[35,15,1.5],[25,60,1.5],[35,60,1.5]])

    def test_scene_and_paths_source_solver_calls(self):
        class FakePaths:
            @staticmethod
            def cir(**kwargs):
                assert kwargs=={'out_type':'numpy'}
                return np.ones((16,2,2,1),dtype=complex),np.zeros(1)
        class FakeRT:
            scene=type('S',(),{'munich':'munich'})
            def __init__(self):self.calls=[];self.scene=type('S',(),{'munich':'munich'})()
            def load_scene(self,name):
                assert name=='munich'
                class Scene:
                    frequency=3.5e9
                    def __init__(self):self.devices=[]
                    def add(self,device):self.devices.append(device)
                self.s=Scene();return self.s
            def PlanarArray(self,**kw):return kw
            def Transmitter(self,name,position):return ('tx',name,position)
            def Receiver(self,name,position):return ('rx',name,position)
            def PathSolver(self,**init):
                self.calls.append(('init',init))
                def solve(scene,**cfg):
                    self.calls.append(('run',cfg))
                    return FakePaths()
                return solve
        rt=FakeRT()
        for mode,settings in [(m.MODES[0],{'max_depth':5}),
                              (m.MODES[1],{'max_depth':5,'seed':42}),
                              (m.MODES[2],{'max_depth':2,'samples_per_src':20000,'seed':42})]:
            a,scene=m.scene_and_paths(rt,mode)
            self.assertEqual(a.shape,(16,2,2,1));self.assertEqual(len(scene.devices),17)
            self.assertEqual(rt.calls[-1],('run',settings))
        self.assertEqual(rt.calls[0],('init',{}))
        self.assertEqual(rt.calls[2],('init',{'deterministic':True}))

    def test_cir_reduction_mask_and_bad_input(self):
        a=np.zeros((16,2,2,2),complex);a[:,0,0,0]=1
        a[3]=0
        raw,H,alive=m.cir_to_channels(a)
        self.assertEqual(H.shape,(16,2,2));self.assertEqual(alive.sum(),15)
        self.assertAlmostEqual(np.linalg.norm(H[0]),1)
        np.testing.assert_allclose(H[3],0)
        with self.assertRaises(ValueError):m.cir_to_channels(np.ones((1,2)))
        with self.assertRaises(ValueError):m.cir_to_channels(np.full((16,2,2,1),np.nan))

    def test_compare_outputs_correctly_separates_frame_and_gram(self):
        H=np.tile(np.eye(2,dtype=complex)[None,:,:]/math.sqrt(2),(16,1,1))
        U=np.diag([1,1j]);other=U@H
        result=m.compare_outputs(H,other)
        self.assertGreater(result['mean_Fro_H'],0.5)
        self.assertLess(result['max_Fro_Gram'],1e-12)
        with self.assertRaises(ValueError):m.compare_outputs(H[:2],other)

    def test_children_have_deadline_and_reject_negative_budget(self):
        with self.assertRaises(ValueError):m.run_mode(m.MODES[0],Path('/tmp'),0,1024)
        # A child can fail before ray tracing. Recording failure is correct.
        with tempfile.TemporaryDirectory() as tmp:
            report=m.run_mode(m.MODES[1],tmp,5,2048)
            self.assertEqual(report['mode'],m.MODES[1]);self.assertTrue(Path(tmp,report['stdout_log']).is_file())
            self.assertEqual(report['output_exists'],False)
            self.assertIn(report['status'],('failed','wall_timeout'))


class GramBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.archive=json.loads(ARCHIVE.read_text())

    def test_frozen_codebooks_and_equality_with_archived(self):
        C=g.codebooks()
        self.assertEqual(set(C),{'rank_one','qpsk_squared','circular_16psk'})
        for x in C.values():
            self.assertEqual(x.shape,(16,2))
            np.testing.assert_allclose(np.sum(np.abs(x)**2,axis=1),1,atol=1e-12)
            np.testing.assert_allclose(np.mean(np.abs(x)**2,axis=0),[.5,.5],atol=1e-12)

    def test_mufour_orbit_envelope_against_brute(self):
        rng=np.random.default_rng(20261105)
        for c in g.codebooks().values():
            self.assertEqual(len(g.orbit_representatives(c)),4)
            for _ in range(20):
                H=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))
                G=H.conj().T@H
                self.assertAlmostEqual(g.orbit_envelope_min_d2(G,c),g.min_d2(G,c),places=11)
        with self.assertRaises(ValueError):g.orbit_representatives(np.zeros((16,2),complex))

    def test_circular_closed_form_exact_for_random_gram(self):
        for r in (0.01,0.1,0.9,1.):
            for theta in (0,30,45,90):
                for phi in (0,60,180):
                    G=g.gram_from_spec(r,theta,phi)
                    self.assertAlmostEqual(g.circular_analytic_min_d2(G),g.min_d2(G,g.codebooks()['circular_16psk']),places=12)

    def test_gram_reconstruction_and_saved_envelopes(self):
        for c in self.archive['cases']:
            G=g.gram_of_case(c)
            w=np.linalg.eigvalsh(G)
            self.assertAlmostEqual(w.sum(),1,places=12)
            for name,C in g.codebooks().items():
                self.assertAlmostEqual(g.min_d2(G,C),c['bounds'][name]['min_d2'],places=10)

    def test_psk_wedge_bounds_cover_monte_carlo_model(self):
        rng=np.random.default_rng(123)
        for gain in (.06,.15,.5,.9):
            lo,up=g.rigorous_beam_16psk_bounds(gain,.01)
            self.assertLessEqual(lo,up)
            # Direct complex AWGN 16PSK ML; independent event construction.
            R=math.sqrt(gain);sig=math.sqrt(.01/2)
            z=R+sig*(rng.normal(size=30000)+1j*rng.normal(size=30000))
            errors=(np.abs(np.angle(z))>math.pi/16).mean()
            std=math.sqrt(max(errors*(1-errors),1e-5)/30000)
            self.assertLessEqual(lo,errors+6*std)
            self.assertLessEqual(errors-6*std,up)

    def test_qam_exact_agrees_with_independent_4pam_formula(self):
        for gain in (.01,.1,.4,.9):
            val=g.exact_beam_16qam_ser(gain,.01)
            p=0.5*math.erfc(math.sqrt(gain/(10*.01)))
            self.assertAlmostEqual(val,3*p-2.25*p*p,places=12)
        with self.assertRaises(ValueError):g.exact_beam_16qam_ser(-1,.01)

    def test_diagonal_qpsk_exact_against_mc(self):
        rng=np.random.default_rng(123)
        C=g.codebooks()['qpsk_squared']
        for ratio in (.01,.25,1.):
            G=g.gram_from_spec(ratio,0,0)
            predicted=g.exact_diagonal_qpsk2_ser(G,.05)
            H=np.diag(np.sqrt(np.diag(G)))
            Y=C@H.T;tx=rng.integers(0,16,size=40000)
            obs=Y[tx]+math.sqrt(.05/2)*(rng.normal(size=(40000,2))+1j*rng.normal(size=(40000,2)))
            err=np.mean(np.argmin(np.sum(abs(obs[:,None,:]-Y[None,:,:])**2,axis=-1),axis=1)!=tx)
            se=math.sqrt(max(predicted*(1-predicted),1e-8)/40000)
            self.assertLess(abs(err-predicted),6*se+0.002)
        with self.assertRaises(ValueError):g.exact_diagonal_qpsk2_ser(g.gram_from_spec(.1,45,0),.01)

    def test_tightening_never_weakens_and_new_decisions_are_strict(self):
        result=g.run_sweep(self.archive)
        self.assertEqual(result['n_cases'],75)
        for label,hist in result['summary'].items():
            self.assertEqual(sum(hist['new'].values()),75)
            self.assertEqual(hist['conflict'],0)
            self.assertEqual(hist['newly_decided'],hist['old']['undecided']-hist['new']['undecided'])
        for case,old in zip(result['cases'],self.archive['cases']):
            for label,new in case['tightened'].items():
                self.assertGreaterEqual(new['lower']+1e-11,old['bounds'][label]['lower'])
                self.assertLessEqual(new['upper'],old['bounds'][label]['upper']+1e-11)
            for label,dec in case['decisions'].items():
                newa=case['tightened']['rank_one'];newb=case['tightened'][label]
                if dec['new']=='A':self.assertLess(newa['upper']+1e-12,newb['lower'])
                if dec['new']=='B':self.assertLess(newb['upper']+1e-12,newa['lower'])

    def test_high_snr_aligned_analytic_phase_thresholds(self):
        C=g.codebooks()
        crit_q=3-2*math.sqrt(2)
        k=2-2*math.cos(math.pi/8)
        crit_c=k/(4-k)
        self.assertLess(crit_c,crit_q)
        for r in (.01,crit_c-.0001,crit_c+.0001,.10,crit_q-.0001,crit_q+.0001,.99):
            G=g.gram_from_spec(r,0,0)
            rank=g.orbit_envelope_min_d2(G,C['rank_one'])
            q=g.orbit_envelope_min_d2(G,C['qpsk_squared'])
            circ=g.circular_analytic_min_d2(G)
            self.assertEqual(rank>q,r<crit_q)
            self.assertEqual(rank>circ,r>crit_c)

    def test_fixed_grid_high_snr_regions_and_finite_mismatch(self):
        out=g.run_sweep(self.archive)
        self.assertEqual(out['summary']['qpsk_squared']['high_snr_A'],18)
        self.assertEqual(out['summary']['qpsk_squared']['high_snr_B'],50)
        self.assertEqual(out['summary']['qpsk_squared']['high_snr_equal_or_unresolved'],7)
        self.assertEqual(out['summary']['circular_16psk']['high_snr_A'],45)
        self.assertEqual(out['summary']['circular_16psk']['high_snr_B'],30)
        self.assertEqual(out['summary']['circular_16psk']['finite_vs_high_snr_disagreement'],4)


if __name__=='__main__':unittest.main()
