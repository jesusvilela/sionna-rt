"""CPU adversarial tests of the conditional calibration-to-effective-Gram bridge.

All actual channel/noise values in this file are synthetic test oracles.
No external noise calibration, original Munich H archive, or propagation data.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'experiments'))
import whitened_calibration_bridge as bridge
from gram_decision_sweep import sweep
from gram_uncertainty_certificate import books, event_brackets, PORTS


class WhitenedCalibrationBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=sweep()['cases']
        if len(cls.cases)!=75:
            raise AssertionError('frozen 75-case synthetic grid missing')

    def test_analytic_equality_witness_matches_exact_formula(self):
        r=bridge.saturation_witness()
        self.assertAlmostEqual(r['true_gram_error'],.55,places=13)
        self.assertAlmostEqual(r['derived_error_bound'],.55,places=13)
        self.assertAlmostEqual(r['equality_gap'],0.,places=13)

    def test_nonwhite_nominal_covariance_preserves_equality_witness(self):
        Z=np.diag([1.25,.3]).astype(complex)
        F=np.diag([.05,0]).astype(complex)
        R=np.diag([-.2,0]).astype(complex)
        S=np.array([[1.,.2+.15j],[.2-.15j,1.3]],complex)
        L=np.linalg.cholesky(S)
        Hhat=L@Z
        Htrue=L@(Z+F)
        Strue=L@(np.eye(2)+R)@L.conj().T
        ghat=Hhat.conj().T@np.linalg.solve(S,Hhat)
        gtrue=Htrue.conj().T@np.linalg.solve(Strue,Htrue)
        true_norm=float(np.linalg.norm(gtrue-ghat,2))
        b=bridge.gram_radius_from_calibration(Hhat,S,.05,.2)
        self.assertAlmostEqual(b['effective_gram_operator_norm_error_bound'],.55,places=13)
        self.assertAlmostEqual(true_norm,.55,places=12)

    def test_frozen_600_samples_satisfy_gram_bound(self):
        x=bridge.synthetic_audit(self.cases)
        self.assertEqual(x['cases'],600)
        self.assertEqual(x['frame_checks'],48)
        self.assertEqual(x['source_sha256'],bridge.SOURCE_SHA256)
        self.assertLessEqual(x['max_bound_excess'],1e-11)
        self.assertLess(x['max_frame_budget_parameter_difference'],2e-10)
        self.assertLess(x['max_frame_true_gram_difference'],2e-10)
        self.assertGreater(x['max_invalid_white_noise_metric_change'],1e-3)
        self.assertEqual(x['physical_munich_channels'],0)

    def test_receiver_GL_and_transmitter_SU2_preserve_calibration_norms(self):
        H=np.array([[.93+.33j,.01+.006j],[-.06-.006j,-.14-.01j]],complex)
        S=np.array([[1.,.2+.15j],[.2-.15j,1.3]],complex)
        L=np.linalg.cholesky(S)
        F=np.array([[.0008+.0002j,0],[.0001j,-.0002]],complex)
        R=np.array([[.01,.002j],[-.002j,-.007]],complex)
        Htrue=H+L@F
        Strue=L@(np.eye(2)+R)@L.conj().T
        a,tau,rho=bridge.oracle_relative_errors(H,S,Htrue,Strue)
        original=bridge.gram_radius_from_calibration(H,S,tau,rho)
        assert original['status']=='conditionally_certifiable'
        Gtrue=Htrue.conj().T@np.linalg.solve(Strue,Htrue)
        for W,U in ((bridge.W1,bridge.U1),(bridge.W2@bridge.W1,bridge.U2)):
            Hp=W@H@U.conj().T
            Sp=W@S@W.conj().T
            Htp=W@Htrue@U.conj().T
            Stp=W@Strue@W.conj().T
            ap,tt,rr=bridge.oracle_relative_errors(Hp,Sp,Htp,Stp)
            changed=bridge.gram_radius_from_calibration(Hp,Sp,tt,rr)
            for x,y in ((a,ap),(tau,tt),(rho,rr),
                        (original['effective_gram_operator_norm_error_bound'],
                         changed['effective_gram_operator_norm_error_bound'])):
                self.assertLess(abs(x-y),2e-10)
            Gp=Htp.conj().T@np.linalg.solve(Stp,Htp)
            np.testing.assert_allclose(Gp,U@Gtrue@U.conj().T,atol=2e-11,rtol=0)
            for P in PORTS:
                Pp=U@P@U.conj().T
                C=books()['rank_one']
                np.testing.assert_allclose(
                    np.einsum('mi,ij,mj->m',C.conj(),P,C),
                    np.einsum('mi,ij,mj->m',(C@U.T).conj(),Pp,C@U.T),
                    atol=2e-12)
        # This raw-coordinate error changes under nonunitary W. It is not
        # an admissible invariant calibration-radius convention.
        self.assertGreater(abs(np.linalg.norm(bridge.W1@(Htrue-H),2)
                               -np.linalg.norm(Htrue-H,2)),1e-5)

    def test_when_budget_exceeds_minimum_eigenvalue_abstain(self):
        H=np.diag([1.,.0001]).astype(complex)
        S=np.eye(2)
        b=bridge.gram_radius_from_calibration(H,S,.001,.01)
        self.assertFalse(b['ready_for_existing_ser_certificate'])
        self.assertEqual(b['status'],'abstain_positive_definiteness_gate')
        Hbad=np.diag([1.,0.]).astype(complex)
        self.assertFalse(bridge.gram_radius_from_calibration(Hbad,S,0,0)['ready_for_existing_ser_certificate'])

    def test_near_singular_relative_covariance_rho_diverges_then_abstains(self):
        H=np.diag([1.,.5]).astype(complex)
        result=bridge.gram_radius_from_calibration(H,np.eye(2),.01,.9999)
        self.assertGreater(result['effective_gram_operator_norm_error_bound'],1e3)
        self.assertFalse(result['ready_for_existing_ser_certificate'])
        with self.assertRaisesRegex(ValueError,'rho must be <1'):
            bridge.gram_radius_from_calibration(H,np.eye(2),.01,1.)

    def test_rejects_invalid_nominal_covariance_and_error_claims(self):
        H=np.eye(2,dtype=complex)
        bad_S=(np.zeros((2,2)),np.diag([1.,-1.]),np.diag([1.,1e-12]),
               np.array([[1.,1j],[0,1.]]),np.eye(3),np.full((2,2),np.nan))
        for S in bad_S:
            with self.assertRaises(ValueError):
                bridge.gram_radius_from_calibration(H,S,.001,.001)
        for tau,rho in ((-1,0),(0,-.1),(np.nan,0),(0,np.inf),(np.inf,.1),
                        (0,1.1),(True,0),('not a number',0)):
            with self.assertRaises(ValueError):
                bridge.gram_radius_from_calibration(H,np.eye(2),tau,rho)

    def test_bridge_can_feed_existing_certificate_without_different_model(self):
        S=np.eye(2)
        H=np.array([[.8+.1j,.2],[-.1j,.35]],complex)
        b=bridge.gram_radius_from_calibration(H,S,1e-4,1e-4)
        self.assertTrue(b['ready_for_existing_ser_certificate'])
        eps=b['effective_gram_operator_norm_error_bound']
        G=H.conj().T@H
        for C in books().values():
            lo,hi=event_brackets(G,C,eps)
            self.assertEqual(lo.shape,(16,16))
            self.assertEqual(hi.shape,(16,16))
            off=~np.eye(16,dtype=bool)
            self.assertTrue(np.all(lo[off]<=hi[off]))

    def test_no_claim_of_measured_munich_calibration(self):
        r=bridge.report(self.cases)
        self.assertEqual(r['asserted_physical_channel_data'],0)
        self.assertEqual(r['synthetic_adversarial']['source_cells'],75)
        for name,world in r['illustrative_not_measured'].items():
            self.assertTrue(world['within_prior_required_bound'],name)
            self.assertEqual(world['measurement_status'],'not measured or independently calibrated')
        b=bridge.gram_radius_from_calibration(np.eye(2),np.eye(2),.001,.001)
        self.assertIn('assumptions_only',b['evidence_status'])


if __name__=='__main__':
    unittest.main()
