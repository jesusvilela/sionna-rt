"""Adversarial tests for a transported projector-valued polarization field."""
import json
import math
import tempfile
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
import moving_frame as m


def projector(x):
    x=np.asarray(x,complex)
    return np.outer(x,x.conj())/np.vdot(x,x).real


def random_u2(rng):
    z=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))
    q,r=np.linalg.qr(z)
    return q@np.diag(np.exp(-1j*np.angle(np.diag(r))))


class MovingFrameTests(unittest.TestCase):
    def setUp(self):
        self.P=[projector([1,0]),projector([1,1]),projector([1,1j])]

    def test_exact_nontrivial_loop_and_orientation(self):
        self.assertAlmostEqual(m.bargmann(*self.P),(1+1j)/4)
        self.assertAlmostEqual(m.bargmann(self.P[0],self.P[2],self.P[1]),(1-1j)/4)
        self.assertAlmostEqual(np.angle(m.bargmann(*self.P)),math.pi/4)

    def test_mirror_preserves_pairwise_overlaps_but_flips_chirality(self):
        mirrored=[self.P[0],self.P[1],self.P[2].conj()]
        overlaps=lambda P: np.array([[np.trace(a@b).real for b in P] for a in P])
        np.testing.assert_allclose(overlaps(self.P),overlaps(mirrored))
        self.assertAlmostEqual(m.bargmann(*mirrored),m.bargmann(*self.P).conjugate())

    def test_stokes_analytic_independent_trace_formula(self):
        rng=np.random.default_rng(7)
        for _ in range(300):
            P=[projector(rng.normal(size=2)+1j*rng.normal(size=2)) for i in range(3)]
            s=[m.stokes_from_projector(p) for p in P]
            np.testing.assert_allclose([np.linalg.norm(x) for x in s],[1,1,1],atol=1e-12)
            self.assertAlmostEqual(m.bargmann(*P),m.bargmann_stokes(*s),places=12)

    def test_global_u2_conjugation_invariant(self):
        rng=np.random.default_rng(8);baseline=m.bargmann(*self.P)
        for _ in range(100):
            U=random_u2(rng);P=[U@p@U.conj().T for p in self.P]
            self.assertAlmostEqual(m.bargmann(*P),baseline)

    def test_independent_local_frames_require_typed_transports(self):
        rng=np.random.default_rng(11);baseline=m.bargmann(*self.P)
        broken=[]
        for _ in range(50):
            gauges=[random_u2(rng) for _ in range(3)]
            P,T=m.reframe(self.P,gauges)
            self.assertAlmostEqual(m.transported_bargmann(P,T,(0,1,2)),baseline)
            broken.append(abs(m.bargmann(*P)-baseline))
        self.assertGreater(max(broken),.1)

    def test_two_moving_frames_compose_with_correct_order(self):
        rng=np.random.default_rng(42);g1=[random_u2(rng) for _ in range(3)]
        g2=[random_u2(rng) for _ in range(3)]
        combined=[g2[i]@g1[i] for i in range(3)]
        P,T=m.reframe(self.P,combined)
        first,_=m.reframe(self.P,g1)
        second,_=m.reframe(first,g2)
        for a,b in zip(P,second):np.testing.assert_allclose(a,b,atol=1e-12)
        self.assertAlmostEqual(m.transported_bargmann(P,T,(0,1,2)),m.bargmann(*self.P))

    def test_eigenprojector_without_arbitrary_eigenvector_phase(self):
        rng=np.random.default_rng(123)
        for _ in range(70):
            u=random_u2(rng);G=u@np.diag([.13,.87])@u.conj().T
            P=m.projector_from_gram(G)
            self.assertAlmostEqual(np.trace(P),1.)
            np.testing.assert_allclose(P@P,P,atol=1e-12)
            np.testing.assert_allclose(G@P,.87*P,atol=1e-12)
        self.assertIsNone(m.projector_from_gram(np.eye(2)/2))
        with self.assertRaises(ValueError):m.projector_from_gram(np.array([[1,2j],[2,1]]))
        with self.assertRaises(ValueError):m.projector_from_gram(np.zeros((2,2)))

    def test_zero_bargmann_phase_is_undefined_not_reported_zero(self):
        self.assertEqual(m.bargmann(projector([1,0]),projector([0,1]),projector([1,1])),0)
        H=np.tile(np.eye(2)[None,:,:]/np.sqrt(2),(16,1,1)).astype(complex)
        report=m.field_diagnostics(H)
        self.assertTrue(all(s['projector_available'] is False for s in report['sites']))
        self.assertTrue(all(t['phase_rad'] is None and t['state']=='eigenline_degenerate' for t in report['triangles']))

    def test_covariant_edge_derivative_is_matrix_valued_and_composeable(self):
        rng=np.random.default_rng(23)
        gauges=[random_u2(rng) for _ in range(3)]
        P,T=m.reframe(self.P,gauges)
        dij=m.covariant_edge_difference(P[0],P[1],T[(0,1)])
        djk=m.covariant_edge_difference(P[1],P[2],T[(1,2)])
        dik=m.covariant_edge_difference(P[0],P[2],T[(0,2)])
        np.testing.assert_allclose(dik,dij+T[(0,1)]@djk@T[(0,1)].conj().T,atol=1e-12)
        np.testing.assert_allclose(dij,gauges[0]@(self.P[1]-self.P[0])@gauges[0].conj().T,atol=1e-12)
        # Pure coordinate motion of a constant projector has zero covariant difference.
        fixed=[self.P[0]]*3
        P2,T2=m.reframe(fixed,gauges)
        np.testing.assert_allclose(m.covariant_edge_difference(P2[0],P2[1],T2[(0,1)]),0,atol=1e-12)
        with self.assertRaises(ValueError):
            m.covariant_edge_difference(P[0],P[1],np.eye(2)*2)

    def test_munich_mesh_triangles_are_oriented(self):
        points=m.receiver_positions();tri=m.oriented_triangles()
        self.assertEqual(len(tri),14)
        self.assertEqual(len(set(tri)),14)
        for i,j,k in tri:
            u=points[j,:2]-points[i,:2];v=points[k,:2]-points[i,:2]
            self.assertGreater(np.linalg.det(np.stack([u,v])),0)

    def test_mesh_preserves_covariant_matrix_edge_field(self):
        rng=np.random.default_rng(300)
        H=np.array([np.diag([.9,.4])@random_u2(rng) for _ in range(16)])
        H=H/np.linalg.norm(H,axis=(1,2))[:,None,None]
        d=m.field_diagnostics(H)
        self.assertEqual(len(d['edges']),29)
        self.assertTrue(all(e['state']=='defined' for e in d['edges']))
        self.assertTrue(all(s['projector_re_im'] is not None for s in d['sites']))
        for e in d['edges']:
            arr=np.array([[complex(*z) for z in row] for row in e['matrix_re_im']])
            np.testing.assert_allclose(arr,arr.conj().T,atol=1e-12)
            self.assertAlmostEqual(np.trace(arr).real,0.,places=12)

    def test_polarization_field_is_left_receiver_unitary_invariant(self):
        rng=np.random.default_rng(25)
        H=np.array([random_u2(rng)@np.diag([.9,np.sqrt(.19)])@random_u2(rng) for _ in range(16)])
        H=H/np.linalg.norm(H,axis=(1,2))[:,None,None]
        other=np.array([random_u2(rng)@h for h in H])
        a=m.field_diagnostics(H);b=m.field_diagnostics(other)
        for s,t in zip(a['sites'],b['sites']):
            np.testing.assert_allclose(s['stokes'],t['stokes'],atol=1e-12)
        for s,t in zip(a['triangles'],b['triangles']):
            self.assertEqual(s['state'],t['state'])
            np.testing.assert_allclose(s['bargmann'],t['bargmann'],atol=1e-12)

    def test_projector_field_rejects_bad_shapes_unscaled_channels(self):
        with self.assertRaises(ValueError):m.field_diagnostics(np.zeros((15,2,2),complex))
        with self.assertRaises(ValueError):m.field_diagnostics(np.ones((16,2,2),complex))

    def test_missing_archive_does_not_fabricate_receiver_data(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'toy.npz';np.savez_compressed(path,H=np.ones((16,2,2),complex))
            with self.assertRaises(ValueError):m.load_verified_munich(path)
            out=Path(temp)/'demo.json';m.main(['--out',str(out)])
            obj=json.loads(out.read_text())
            self.assertEqual(obj['per_receiver']['status'],'not_available')
            self.assertEqual(obj['status'],'synthetic_toy_only')
            self.assertEqual(len(obj['triangles_munich_lattice']),14)

if __name__=='__main__':unittest.main()
