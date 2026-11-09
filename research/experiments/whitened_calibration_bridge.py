"""Convert *externally justified* channel/noise error budgets into a Gram bound.

No error radius is inferred from observed data by this module. The public
interface takes declared deterministic bounds for whitened channel error and
relative noise-covariance error, and returns a sufficient operator-norm
radius for G = H^H Sigma^{-1} H. There is no Sionna dependency or claim of
original Munich receiver observations.

The underlying inequality is a classical inverse/resolvent norm estimate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


COND_LIMIT = 1e10
SOURCE_SHA256 = "16ad7ca1eda73d9c6a73776555be2f93333d248ad44d11a32ac9be3b05d2f4c6"
DEMO_BUDGETS = ((0., 0.), (1e-4, 1e-3), (1e-3, 1e-2), (5e-3, 0.1))


def _nominal_whitened(Hhat, Sigmahat):
    """Check contract and return calibrated Z, Cholesky factor, Gram, lambda_min."""
    H = np.asarray(Hhat, dtype=np.complex128)
    S = np.asarray(Sigmahat, dtype=np.complex128)
    if H.shape != (2, 2) or S.shape != (2, 2):
        raise ValueError("finite complex 2x2 Hhat and Sigmahat required")
    if not np.isfinite(H).all() or not np.isfinite(S).all():
        raise ValueError("nominal channel and covariance must be finite")
    if np.max(np.abs(S - S.conj().T)) > 1e-12:
        raise ValueError("nominal covariance must be Hermitian")
    S = (S + S.conj().T) / 2
    w = np.linalg.eigvalsh(S)
    if w[0] <= 0:
        raise ValueError("nominal covariance must be strictly positive definite")
    if w[-1] / w[0] > COND_LIMIT:
        raise ValueError("nominal covariance too ill-conditioned for this CPU gate")
    L = np.linalg.cholesky(S)
    Z = np.linalg.solve(L, H)
    G = Z.conj().T @ Z
    G = (G + G.conj().T) / 2
    eig = np.linalg.eigvalsh(G)
    return Z, L, G, float(eig[0]), float(w[-1] / w[0])


def _nonnegative_finite(value, name):
    if isinstance(value, (bool, complex)):
        raise ValueError(f"{name} must be a real finite nonnegative bound")
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a real finite nonnegative bound") from None
    if not np.isfinite(scalar) or scalar < 0:
        raise ValueError(f"{name} must be a real finite nonnegative bound")
    return scalar


def gram_radius_from_calibration(Hhat, Sigmahat, tau, rho):
    """Sufficient ||Gtrue-Ghat||op bound from DECLARED whitened error radii.

    tau: independently justified upper bound on ||L^{-1}(H-Hhat)||op.
    rho: independently justified upper bound on
         ||L^{-1}(Sigma-Sigmahat)L^{-H}||op, strictly less than one.
    L L^H = Sigmahat.

    The result is only a conditional engineering acceptance test, and never
    validates the instrument that supplied tau or rho.
    """
    tau = _nonnegative_finite(tau, "tau")
    rho = _nonnegative_finite(rho, "rho")
    if rho >= 1:
        raise ValueError("rho must be <1 to ensure every admissible Sigma is PD")
    Z, _, G, lam_min, condition = _nominal_whitened(Hhat, Sigmahat)
    a = float(np.linalg.norm(Z, 2))
    eps = (a*a*rho + 2*a*tau + tau*tau) / (1-rho)
    if not np.isfinite(eps):
        raise ValueError("derived Gram radius is not numerically finite")
    ready = bool(lam_min > 0 and eps < lam_min)
    return {
        "whitened_nominal_channel_operator_norm":a,
        "declared_whitened_channel_error_bound":tau,
        "declared_relative_covariance_error_bound":rho,
        "effective_gram_operator_norm_error_bound":float(eps),
        "nominal_gram_min_eigenvalue":lam_min,
        "nominal_covariance_condition":condition,
        "ready_for_existing_ser_certificate":ready,
        "status":"conditionally_certifiable" if ready else "abstain_positive_definiteness_gate",
        "evidence_status":"assumptions_only_no_measured_calibration",
        "receiver_frame":"arbitrary GL(2,C) with full covariance transport",
        "noise_model":"CN(0, N0*Sigma) with independent fixed N0 > 0",
        "source":"Neumann inverse perturbation and spectral submultiplicativity",
    }


def oracle_relative_errors(Hhat, Sigmahat, Htrue, Sigmatrue):
    """TEST-ONLY oracle; should not be used to assign experimental guarantees.

    Computes realized whitened errors when BOTH actual objects are known.
    It does not produce an independently justified uncertainty bound.
    """
    Z, L, _, _, _ = _nominal_whitened(Hhat, Sigmahat)
    Htrue = np.asarray(Htrue, dtype=np.complex128)
    Strue = np.asarray(Sigmatrue, dtype=np.complex128)
    if Htrue.shape != (2,2) or Strue.shape != (2,2):
        raise ValueError("test oracle actual objects must be 2x2")
    if not np.isfinite(Htrue).all() or not np.isfinite(Strue).all():
        raise ValueError("test oracle actual objects must be finite")
    _nominal_whitened(Htrue, Strue)
    F = np.linalg.solve(L, Htrue - Hhat)
    tmp = np.linalg.solve(L, Strue - Sigmahat)
    R = np.linalg.solve(L, tmp.conj().T).conj().T
    if np.max(abs(R-R.conj().T))>2e-10:
        raise ValueError("relative covariance difference not Hermitian")
    return (float(np.linalg.norm(Z,2)),
            float(np.linalg.norm(F,2)),
            float(np.linalg.norm((R+R.conj().T)/2,2)))


def _source_gram(case):
    a=case["channel"]
    b=complex(a["cross_correlation_real"],a["cross_correlation_imag"])
    return np.array([[a["port_gain_split"][0],b],
                     [b.conjugate(),a["port_gain_split"][1]]],complex)


def _hermitian_root(G):
    eigen,V=np.linalg.eigh(G)
    if eigen[0]<=0:
        raise ValueError("synthetic source Gram must be positive definite")
    return (V*np.sqrt(eigen))@V.conj().T


def _frame_su2(angle_deg, phase):
    t=np.deg2rad(angle_deg); c,s=np.cos(t),np.sin(t);z=np.exp(1j*phase)
    return np.array([[c,z*s],[-z.conjugate()*s,c]],complex)


W1=np.array([[1.3+.2j,.6-.3j],[.1+.4j,.75]],complex)
W2=np.array([[.9,-.35+.15j],[.2+.05j,1.1-.1j]],complex)
U1=_frame_su2(23.,.38)
U2=_frame_su2(13.,-.61)@U1


def _random_unit_operator(rng):
    X=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))
    return X/np.linalg.norm(X,2)


def _random_hermitian_unit(rng):
    X=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))
    E=(X+X.conj().T)/2
    return E/np.linalg.norm(E,2)


def _realized_gram(H, Sigma):
    return H.conj().T @ np.linalg.solve(Sigma,H)


def synthetic_audit(cases):
    """Frozen seeded tests only. Never consumes/produces physical Munich data."""
    if len(cases)!=75:
        raise ValueError("frozen experiment requires exactly 75 synthetic cells")
    rng=np.random.default_rng(20261109)
    S0=np.array([[1.,.2+.15j],[.2-.15j,1.3]],complex)
    L=np.linalg.cholesky(S0)
    max_excess=-float("inf")
    max_frame_parameter_difference=0.
    max_frame_gram_difference=0.
    max_false_metric_change=0.
    n=0
    frame_count=0
    all_budgets=[]
    for k,case in enumerate(cases):
        G=_source_gram(case)
        Hroot=_hermitian_root(G)
        Hhat=L @ Hroot
        nominal=_realized_gram(Hhat,S0)
        if np.max(abs(nominal-G))>2e-12:
            raise AssertionError("input Gram not preserved by nominal whitening")
        for j in range(8):
            tau,rho=DEMO_BUDGETS[j%4]
            F=tau*rng.uniform(.1,1.)*_random_unit_operator(rng)
            R=rho*rng.uniform(.1,1.)*_random_hermitian_unit(rng)
            true_H=Hhat+L@F
            true_S=L@(np.eye(2)+R)@L.conj().T
            actual=_realized_gram(true_H,true_S)
            budget=gram_radius_from_calibration(Hhat,S0,tau,rho)
            realized=float(np.linalg.norm(actual-nominal,2))
            diff=realized-budget['effective_gram_operator_norm_error_bound']
            max_excess=max(max_excess,diff)
            if diff>1e-11:
                raise AssertionError(("calibration bridge upper bound violated",k,j,diff))
            all_budgets.append(budget['effective_gram_operator_norm_error_bound'])
            n+=1
            if k<12 and j%4==2:
                a0,t0,r0=oracle_relative_errors(Hhat,S0,true_H,true_S)
                original=gram_radius_from_calibration(Hhat,S0,t0,r0)
                for W,U in ((W1,U1),(W2@W1,U2)):
                    Hn=W@Hhat@U.conj().T
                    Sn=W@S0@W.conj().T
                    Ht=W@true_H@U.conj().T
                    St=W@true_S@W.conj().T
                    aa,tt,rr=oracle_relative_errors(Hn,Sn,Ht,St)
                    floated=gram_radius_from_calibration(Hn,Sn,tt,rr)
                    deviation=max(abs(aa-a0),abs(tt-t0),abs(rr-r0),
                                  abs(floated['effective_gram_operator_norm_error_bound']
                                      -original['effective_gram_operator_norm_error_bound']))
                    max_frame_parameter_difference=max(max_frame_parameter_difference,deviation)
                    transformed=_realized_gram(Ht,St)
                    expected=U@actual@U.conj().T
                    max_frame_gram_difference=max(max_frame_gram_difference,
                                                  float(np.max(abs(transformed-expected))))
                    # Intentionally invalid: move receiver channel but not its noise model.
                    false=_realized_gram(Ht,Sn)
                    max_false_metric_change=max(max_false_metric_change,
                                                float(np.linalg.norm(false-expected,2)))
                    frame_count+=1
                    if deviation>2e-10 or max_frame_gram_difference>2e-10:
                        raise AssertionError(("frame-consistent calibration bridge failed",k,j,deviation))
    if n!=600 or frame_count!=48:
        raise AssertionError("predeclared fixture population changed")
    return {
        "cases":n,"frame_checks":frame_count,"seed":20261109,
        "source_cells":75,"source_sha256":SOURCE_SHA256,
        "budgets":[list(v) for v in DEMO_BUDGETS],
        "max_bound_excess":max_excess,
        "max_frame_budget_parameter_difference":max_frame_parameter_difference,
        "max_frame_true_gram_difference":max_frame_gram_difference,
        "max_invalid_white_noise_metric_change":max_false_metric_change,
        "min_assumed_gram_budget":min(all_budgets),
        "max_assumed_gram_budget":max(all_budgets),
        "physical_munich_channels":0,
        "status":"synthetic_only_not_calibration_measurements",
    }


def saturation_witness():
    """Analytic equality witness: not a simulated radio channel."""
    a,b,tau,rho=1.25,.3,.05,.2
    Z=np.diag([a,b]).astype(complex)
    F=np.diag([tau,0]).astype(complex)
    R=np.diag([-rho,0]).astype(complex)
    Ghat=Z.conj().T@Z
    Gtrue=(Z+F).conj().T@np.linalg.solve(np.eye(2)+R,Z+F)
    direct=float(np.linalg.norm(Gtrue-Ghat,2))
    bound=gram_radius_from_calibration(Z,np.eye(2),tau,rho)
    if abs(direct-.55)>1e-12 or abs(bound['effective_gram_operator_norm_error_bound']-.55)>1e-12:
        raise AssertionError("analytic saturating witness drift")
    return {"true_gram_error":direct,"derived_error_bound":bound['effective_gram_operator_norm_error_bound'],
            "equality_gap":direct-bound['effective_gram_operator_norm_error_bound'],
            "status":"exact conditional norm bound saturation"}


def report(cases):
    synth=synthetic_audit(cases)
    sat=saturation_witness()
    # Prespecified illustrative assumptions only, not derived from any receiver.
    comparison={}
    for label,G in (
        ("A_rank_one_better", np.array([[1/1.1,0],[0,.1/1.1]],complex)),
        ("B_rank_one_worse",
          (0.01*np.eye(2)+.99*np.outer([np.cos(np.pi/8),np.sin(np.pi/8)],
                                           [np.cos(np.pi/8),np.sin(np.pi/8)]))/1.01)
    ):
        assumed=gram_radius_from_calibration(_hermitian_root(G),np.eye(2),.001,.001)
        previous=0.020364816713798736 if label.startswith('A_') else 0.009900990099009875
        comparison[label]={
            "assumed_tau":.001,"assumed_rho":.001,
            "illustrative_effective_gram_error_bound":assumed["effective_gram_operator_norm_error_bound"],
            "prior_synthetic_required_bound":previous,
            "within_prior_required_bound":bool(assumed['effective_gram_operator_norm_error_bound']<previous),
            "measurement_status":"not measured or independently calibrated",
        }
    return {
        "source":"existing frozen 75 synthetic Gram cells",
        "saturation":sat,"synthetic_adversarial":synth,
        "illustrative_not_measured":comparison,
        "asserted_physical_channel_data":0,
        "scope":"calibration acceptance bridge, known matrix perturbation mathematics",
        "evidence":{
            "P":"conditional relative noise inverse resolvent and spectral norm",
            "M":"synthetic seeded tests and analytic saturator",
            "A":"tau and rho must be independently certified before physical use",
            "R":"no Munich calibrated H or Sigma, no claims of physical uncertainty achievement",
        }
    }


def main(argv=None):
    from gram_decision_sweep import sweep
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("refusing to overwrite prior experimental evidence")
    data=report(sweep()['cases'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(data,sort_keys=True,indent=2)+'\n')
    print(json.dumps(data,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
