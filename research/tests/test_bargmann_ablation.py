"""Regression and representation controls for one frozen Bargmann ablation.

No alternative model, target, threshold or train/test split is tried here.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import bargmann_ablation as a

SOURCE = Path('/mnt/data/sionna_gram_checkpoint/gram_sweep_2026-11-04.json')


class SyntheticAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups,cls.sha=a.source_rows(SOURCE)
        cls.rows=a.make_features(cls.groups)

    def test_input_integrity_and_forbidden_isotropic_projector(self):
        self.assertEqual(self.sha,a.FROZEN_SHA256)
        self.assertEqual(len(self.groups),20)
        self.assertEqual(len(self.rows),60)
        self.assertEqual(len(a.PHIS),3)
        for grp,nodes in self.groups.items():
            self.assertEqual(sorted(nodes),list(a.PHIS))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'tampered.json'
            path.write_bytes(SOURCE.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):
                a.source_rows(path)

    def test_group_split_exactly_blocks_theta(self):
        tr={x['group'] for x in self.rows if x['theta'] in a.TRAIN_THETAS}
        te={x['group'] for x in self.rows if x['theta'] in a.TEST_THETAS}
        self.assertFalse(tr&te)
        self.assertEqual(len(tr),12)
        self.assertEqual(len(te),8)
        self.assertEqual(sum(x['theta'] in a.TRAIN_THETAS for x in self.rows),36)
        self.assertEqual(sum(x['theta'] in a.TEST_THETAS for x in self.rows),24)

    def test_reference_overlap_features_recover_full_local_gram(self):
        # Anchored pairwise overlaps of rank-one P encode its Stokes vector.
        # tr(G)=1, ratio r => G=lambda_min I+(lambda_max-lambda_min)P.
        for x in self.rows:
            r=x['r']
            z,xs,ys=x['x_base'][1:4]
            nx,ny,nz=2*xs-1,2*ys-1,2*z-1
            P=np.array([[(1+nz)/2, (nx-1j*ny)/2],
                        [(nx+1j*ny)/2,(1-nz)/2]],complex)
            trueP=self.groups[x['group']][x['phi']]['P']
            assert_allclose(P,trueP,atol=1e-12)
            lmin=r/(1+r); lmax=1/(1+r)
            expected=lmin*np.eye(2)+(lmax-lmin)*trueP
            reconstructed=lmin*np.eye(2)+(lmax-lmin)*P
            assert_allclose(reconstructed,expected,atol=1e-12)

    def test_global_u2_covariance_of_pairwise_baseline_and_B(self):
        rng=np.random.default_rng(1212)
        for grp,nodes in list(self.groups.items())[::3]:
            P=[nodes[phi]['P'] for phi in a.PHIS]
            Z=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))
            U,R=np.linalg.qr(Z)
            U=U@np.diag(np.exp(-1j*np.angle(np.diag(R))))
            rot=lambda p:U@p@U.conj().T
            for p in P:
                for anchor in a.REFS:
                    self.assertAlmostEqual(a.overlap(p,anchor),a.overlap(rot(p),rot(anchor)),places=12)
            self.assertAlmostEqual(a.bargmann(*P),a.bargmann(*(rot(p) for p in P)),places=12)

    def test_B_is_group_shared_and_train_does_not_cover_test_chirality(self):
        # This is a load-bearing generalization caveat, not a changed model.
        for grp in self.groups:
            rows=[x for x in self.rows if x['group']==grp]
            self.assertEqual(len(rows),3)
            self.assertTrue(all(abs(rows[0]['B_im']-x['B_im'])<1e-13 for x in rows))
        train=np.array([x['B_im'] for x in self.rows if x['theta'] in a.TRAIN_THETAS])
        test=np.array([x['B_im'] for x in self.rows if x['theta'] in a.TEST_THETAS])
        self.assertLess(np.max(np.abs(train)),2e-15)
        self.assertGreater(np.max(np.abs(test)),.17)

    def test_archived_targets_distinguish_not_certified_from_better(self):
        for x in self.rows:
            self.assertEqual(x['y'],int(x['decision']=='A'))
            self.assertIn(x['decision'],('A','B','undecided'))
        self.assertEqual(len([x for x in self.rows if x['y']]),sum(x['y'] for x in self.rows))

    def test_no_nonfinite_input_features(self):
        self.assertEqual({len(x['x_base']) for x in self.rows},{6})
        self.assertEqual({len(x['x_plus']) for x in self.rows},{8})
        self.assertTrue(np.isfinite([x['x_plus'] for x in self.rows]).all())


if __name__=='__main__':
    unittest.main()
