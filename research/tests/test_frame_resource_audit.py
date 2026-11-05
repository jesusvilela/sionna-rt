"""Independent/metamorphic verification of source-bound moving-frame resource audit."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
import sys
import unittest

import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'experiments'))
import frame_resource_audit as audit

SOURCE=Path(os.environ.get('SIONNA_GRAM_SWEEP', HERE/'experiments'/'gram_sweep_2026-11-04.json'))


def independent_direct_received_d2(H,C):
    """Black-box channel-output oracle independent of Gram quadratic code path."""
    Y=np.einsum('ab,mb->ma',H,C)
    return np.sum(np.abs(Y[:,None,:]-Y[None,:,:])**2,axis=-1)


class FrameAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._fixture_dir = None
        cls.source_path = SOURCE
        if not cls.source_path.exists():
            # Regenerate from the previously committed frozen sweep source;
            # do not require any model-specific /mnt/data directory.
            from gram_decision_sweep import sweep
            data=sweep()
            data['per_receiver']={'status':'not_available',
                'reason':'no 16-channel verified export supplied'}
            cls._fixture_dir=tempfile.TemporaryDirectory()
            cls.source_path=Path(cls._fixture_dir.name)/'gram_sweep_2026-11-04.json'
            cls.source_path.write_text(json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+'\n')
        cls.source=audit.load_source(cls.source_path)

    @classmethod
    def tearDownClass(cls):
        if cls._fixture_dir is not None:
            cls._fixture_dir.cleanup()

    def test_frozen_source_integrity_with_tamper_rejection(self):
        self.assertEqual(hashlib.sha256(self.source_path.read_bytes()).hexdigest(),audit.EXPECTED_SHA256)
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'tampered.json'
            f.write_bytes(self.source_path.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'SHA256'):
                audit.load_source(f)

    def test_su2_unitarity_and_nested_frame_composition(self):
        U=audit.fixed_frame();V=audit.u2_mixing(*audit.SU2_FRAMES[1]);T=audit.moving_frame()
        np.testing.assert_allclose(T,V@U,atol=1e-15)
        for mat in (U,V,T):
            np.testing.assert_allclose(mat@mat.conj().T,np.eye(2),atol=1e-14)
            self.assertAlmostEqual(np.linalg.det(mat),1.+0j,places=12)
        # A genuine two-level change; not merely a diagonal phase rotation.
        self.assertGreater(abs(T[0,1]),0.1)

    def test_two_step_tensor_transport_equals_direct_composition(self):
        G=audit.source_gram(self.source['cases'][23])
        C=audit.source_codebooks(G)['circular_16psk']
        objs=(*audit.PORTS,audit.NULL_OPERATOR)
        U=audit.fixed_frame();V=audit.u2_mixing(*audit.SU2_FRAMES[1])
        G1,C1,O1=audit.transport(G,C,objs,U)
        G2,C2,O2=audit.transport(G1,C1,O1,V)
        directG,directC,directObjs=audit.transport(G,C,objs,V@U)
        np.testing.assert_allclose(G2,directG,atol=1e-14)
        np.testing.assert_allclose(C2,directC,atol=1e-14)
        for a,b in zip(O2,directObjs): np.testing.assert_allclose(a,b,atol=1e-14)
        for object_ in O2:
            np.testing.assert_allclose(object_,object_.conj().T,atol=1e-14)

    def test_port_measurement_and_null_form_move_together(self):
        G=audit.source_gram(self.source['cases'][31])
        C=audit.source_codebooks(G)
        U=audit.moving_frame()
        ports2=tuple(U@P@U.conj().T for P in audit.PORTS)
        operator2=U@audit.NULL_OPERATOR@U.conj().T
        for label,c in C.items():
            moved=c@U.T
            np.testing.assert_allclose(audit.physical_powers(c,audit.PORTS),
                                       audit.physical_powers(moved,ports2),atol=1e-14)
            np.testing.assert_allclose(audit.q_values(c),audit.q_values(moved,operator2),atol=1e-14)
        self.assertLess(np.max(abs(audit.q_values(C['rank_one']))),1e-12)
        np.testing.assert_allclose(audit.q_values(C['circular_16psk']),.5,atol=1e-12)
        self.assertGreater(max(abs(audit.q_values(C['circular_16psk']@U.T)-.5)),.05)

    def test_each_source_gram_matches_explicit_channel_factorization(self):
        # Independent direct-H received-distance witness rather than only
        # two algebraically equivalent computations of the same Gram.
        for k in (0,1,5,19,32,48,74):
            G=audit.source_gram(self.source['cases'][k]);w,V=np.linalg.eigh(G)
            H=np.diag(np.sqrt(np.clip(w,0,None)))@V.conj().T
            C=audit.source_codebooks(G)
            for name,c in C.items():
                np.testing.assert_allclose(audit.pair_distances(G,c),
                                           independent_direct_received_d2(H,c),atol=1e-14,rtol=1e-12)

    def test_all_75_x_6_bound_intervals_and_375_decisions(self):
        data=audit.run(self.source)
        self.assertTrue(data['all_75_x_5_strict_decisions_preserved'])
        for key,v in data['full_coordinate_transport_max_abs'].items():
            self.assertLess(v,1e-10, key)
        self.assertEqual(data['source_munich_real_channels'],0)
        self.assertEqual(data['original_sweep_summary'],self.source['summary'])

    def test_deliberately_wrong_frame_violates_physical_resources(self):
        s=audit.run(self.source)['naive_fixed_diagonal_port_control']
        for frame in ('first','second'):
            r=s[frame]
            self.assertEqual(r['naive_physical_port_constraint_violations'],75)
            self.assertGreater(r['max_naive_port_violation'],0.4)
            self.assertGreater(r['naive_interval_changes'],0)
            self.assertGreater(r['naive_certified_decision_changes'],0)
        # Frozen-result regression; these were not promotion thresholds.
        self.assertEqual(s['first']['naive_certified_decision_changes'],14)
        self.assertEqual(s['second']['naive_certified_decision_changes'],18)

    def test_steered_beam_remains_optimal_with_transported_physical_port_projectors(self):
        for k in (6,8,29,41,55):
            G=audit.source_gram(self.source['cases'][k]);T=audit.moving_frame()
            gp=T@G@T.conj().T
            beam=audit.equal_port_beam(G)
            transported=T@beam
            target=float(np.vdot(transported,gp@transported).real)
            for phi in np.linspace(0,2*np.pi,65):
                physical=np.array([1,np.exp(1j*phi)])/np.sqrt(2)
                candidate=T@physical
                gain=float(np.vdot(candidate,gp@candidate).real)
                self.assertLessEqual(gain,target+1e-12)
                powers=audit.physical_powers(candidate[None,:],tuple(T@P@T.conj().T for P in audit.PORTS))
                np.testing.assert_allclose(powers,[[.5,.5]],atol=1e-13)

    def test_printed_channel_independent_transmit_and_receive_frame(self):
        info=audit.printed_channel_control()
        self.assertEqual(info['physical_channel'],'single rounded printed notebook receiver only')
        self.assertLess(info['max_distance_and_bound_delta'],1e-11)

    def test_known_printed_channel_analytic_bounds(self):
        H=np.array([[.92982936+.33584663j,.01006189+.00676795j],
                    [-.06026739-.00641686j,-.13674566-.01022593j]])
        G=H.conj().T@H
        books=audit.source_codebooks(G)
        rank=audit.bounds(G,books['rank_one'])
        circ=audit.bounds(G,books['circular_16psk'])
        qpsk=audit.bounds(G,books['qpsk_squared'])
        self.assertAlmostEqual(rank['lower'],.06808739425621839,places=11)
        self.assertAlmostEqual(circ['upper'],.05182352183597337,places=11)
        self.assertEqual(audit.decision(rank,circ),'B')
        self.assertEqual(audit.decision(rank,qpsk),'A')
        self.assertAlmostEqual(qpsk['lower'],.16517190571537954,places=11)


if __name__=='__main__': unittest.main()
