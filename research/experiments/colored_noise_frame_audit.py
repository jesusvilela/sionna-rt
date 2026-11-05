"""Frozen CPU audit of colored-noise coherent ML under two moving receiver frames.

Research scope is numeric representation correctness, NOT a new propagation
result, geometric theory, or 6G-performance claim. Source: 75-case Gram sweep.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import frame_resource_audit as old

SOURCE_SHA256 = old.EXPECTED_SHA256
N0 = old.N0
W1 = np.array([[1.3 + 0.2j, 0.6 - 0.3j],
               [0.1 + 0.4j, 0.75]], dtype=complex)
W2 = np.array([[0.9, -0.35 + 0.15j],
               [0.2 + 0.05j, 1.1 - 0.1j]], dtype=complex)
U1 = old.u2_mixing(23., 0.38)
U2 = old.u2_mixing(13., -0.61)
STAGES = (("first", W1, U1), ("second", W2 @ W1, U2 @ U1))
ATOL = 1e-10


def _validate_covariance(S):
    S = np.asarray(S, dtype=complex)
    if S.shape != (2, 2) or not np.isfinite(S).all():
        raise ValueError("noise covariance must be finite 2x2 complex")
    if not np.allclose(S, S.conj().T, rtol=0, atol=1e-12):
        raise ValueError("noise covariance must be Hermitian")
    S = (S + S.conj().T)/2
    eig = np.linalg.eigvalsh(S)
    if eig[0] <= 0:
        raise ValueError("noise covariance must be positive definite")
    if eig[-1]/eig[0] > 1e12:
        raise ValueError("ill-conditioned covariance cannot support this numeric check")
    return S


def effective_gram(H, Sigma):
    """Physical ML Gram, preserving the full colored receiver covariance."""
    H = np.asarray(H, dtype=complex)
    if H.shape != (2, 2) or not np.isfinite(H).all():
        raise ValueError("source channel must be finite 2x2 complex")
    Sigma = _validate_covariance(Sigma)
    G = H.conj().T @ np.linalg.solve(Sigma, H)
    return (G + G.conj().T)/2


def whitened_gram_independent(H, Sigma):
    """Second numerical witness via triangular whitening, not Sigma inverse."""
    T = np.linalg.cholesky(_validate_covariance(Sigma))
    whitened = np.linalg.solve(T, H)
    return whitened.conj().T @ whitened


def spectral_channel(G):
    """Construct a source-coordinate H from a trace-1 positive Gram."""
    w, v = np.linalg.eigh(G)
    if w[0] <= 0:
        raise ValueError("source synthetic Gram should be positive definite")
    H = np.diag(np.sqrt(w)) @ v.conj().T
    np.testing.assert_allclose(H.conj().T @ H, G, atol=1e-13, rtol=0)
    return H


def stage_map(H, Sigma, C, W, U):
    """Active receiver representation W and passive input coordinates U."""
    if abs(np.linalg.det(W)) < 1e-9:
        raise ValueError("W must be invertible")
    if not np.allclose(U.conj().T @ U, np.eye(2), rtol=0, atol=1e-12):
        raise ValueError("U must be unitary")
    return W @ H @ U.conj().T, W @ Sigma @ W.conj().T, C @ U.T


def bounds_and_d2(G, C):
    return old.bounds(G, C, N0), old.pair_distances(G, C)


def decision_pair(B):
    a = B["rank_one"]
    return {k: old.decision(a, B[k]) for k in old.LABELS if k != "rank_one"}


def printed_control(sample_count=2048, seed=20261105):
    """Independent black-box coherent-ML-score round-trip on one printed H."""
    H = np.array([[0.92982936+0.33584663j,0.01006189+0.00676795j],
                  [-0.06026739-0.00641686j,-0.13674566-0.01022593j]])
    G = H.conj().T @ H
    C = old.source_codebooks(G)["rank_one"]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0,16,size=sample_count)
    noise = np.sqrt(N0/2)*(rng.normal(size=(sample_count,2))+
                           1j*rng.normal(size=(sample_count,2)))
    y = np.einsum("ab,mb->ma", H, C[indices]) + noise

    def scores(y_, H_, C_, sigma_):
        Y = C_ @ H_.T
        resid = y_[:,None,:] - Y[None,:,:]
        inv_resid = np.linalg.solve(sigma_, resid.reshape(-1,2).T).T.reshape(resid.shape)
        return np.sum((resid.conj()*inv_resid).real,axis=-1)

    baseline = scores(y,H,C,np.eye(2))
    maximum_score_delta = 0.
    num_diff = {}
    for name,W,U in STAGES:
        Hp,Sp,Cp = stage_map(H,np.eye(2),C,W,U)
        yp = y @ W.T
        new = scores(yp,Hp,Cp,Sp)
        maximum_score_delta = max(maximum_score_delta,float(np.max(abs(new-baseline))))
        num_diff[name] = int(np.sum(new.argmin(axis=1)!=baseline.argmin(axis=1)))
    return {"noise_model":"original source-printed H; sampled CN(0,N0 I)",
            "n_symbols":sample_count,"seed":seed,
            "decision_differences":num_diff,
            "max_full_symbol_metric_delta":maximum_score_delta}


def run(frozen):
    if frozen.get('n_cases') != 75 or len(frozen.get('cases',[])) != 75:
        raise ValueError('expected 75 frozen source cases')
    stats = {}
    for name,W,U in STAGES:
        stats[name] = {"max_gram_delta":0.,"max_d2_delta":0.,"max_bound_delta":0.,
                       "max_port_energy_delta":0.,"n_wrong_decision_changes":0,
                       "n_trace_wrong_decision_changes":0,
                       "n_wrong_cells_any_change":0,"n_trace_wrong_cells_any_change":0,
                       "wrong_strict_flips":0,"trace_wrong_strict_flips":0,
                       "max_wrong_gram_delta":0.,"max_trace_wrong_gram_delta":0.,
                       "max_wrong_interval_delta":0.,"max_trace_wrong_interval_delta":0.,
                       "worst_covariance_condition":0.}
    rows = []
    for case in frozen['cases']:
        G = old.source_gram(case)
        H = spectral_channel(G)
        Sigma = np.eye(2,dtype=complex)
        C=old.source_codebooks(G)
        original={label:old.bounds(G,C[label]) for label in old.LABELS}
        d2={label:old.pair_distances(G,C[label]) for label in old.LABELS}
        baseline_dec = decision_pair(original)
        if baseline_dec != case['rank_one_vs']:
            raise AssertionError('frozen source classification mismatch')
        case_row = {'grid':case['grid'],'source_decisions':baseline_dec,'moving_frames':{}}
        for name,W,U in STAGES:
            st=stats[name]
            Gp = None
            trans_bounds={}
            wrong_bounds={}
            trace_wrong_bounds={}
            for label in old.LABELS:
                Hp,Sp,Cp=stage_map(H,Sigma,C[label],W,U)
                if Gp is None:
                    Gp = effective_gram(Hp,Sp)
                    independent = whitened_gram_independent(Hp,Sp)
                    st['max_gram_delta']=max(st['max_gram_delta'],float(np.max(abs(independent-Gp))))
                    expected=U@G@U.conj().T
                    st['max_gram_delta']=max(st['max_gram_delta'],float(np.max(abs(Gp-expected))))
                    st['worst_covariance_condition']=max(st['worst_covariance_condition'],float(np.linalg.cond(Sp)))
                    wrong=Hp.conj().T@Hp
                    trace_wrong=wrong*(np.trace(Gp).real/np.trace(wrong).real)
                    st['max_wrong_gram_delta']=max(st['max_wrong_gram_delta'],float(np.max(abs(wrong-Gp))))
                    st['max_trace_wrong_gram_delta']=max(st['max_trace_wrong_gram_delta'],float(np.max(abs(trace_wrong-Gp))))
                corrected = old.bounds(Gp,Cp)
                trans_bounds[label] = corrected
                wrong_bounds[label] = old.bounds(wrong,Cp)
                trace_wrong_bounds[label] = old.bounds(trace_wrong,Cp)
                st['max_d2_delta']=max(st['max_d2_delta'],float(np.max(abs(old.pair_distances(Gp,Cp)-d2[label]))))
                st['max_bound_delta']=max(st['max_bound_delta'],max(abs(corrected[k]-original[label][k]) for k in ('lower','upper','min_d2')))
                st['max_wrong_interval_delta']=max(st['max_wrong_interval_delta'],max(abs(wrong_bounds[label][k]-original[label][k]) for k in ('lower','upper','min_d2')))
                st['max_trace_wrong_interval_delta']=max(st['max_trace_wrong_interval_delta'],max(abs(trace_wrong_bounds[label][k]-original[label][k]) for k in ('lower','upper','min_d2')))
                ports=tuple(U@p@U.conj().T for p in old.PORTS)
                p0=old.physical_powers(C[label],old.PORTS)
                p1=old.physical_powers(Cp,ports)
                st['max_port_energy_delta']=max(st['max_port_energy_delta'],float(np.max(abs(p0-p1))))
            corrected_dec=decision_pair(trans_bounds)
            if corrected_dec!=baseline_dec:
                raise AssertionError(('corrected frame changed classification',case['grid'],name,corrected_dec,baseline_dec))
            wrong_dec=decision_pair(wrong_bounds)
            trace_dec=decision_pair(trace_wrong_bounds)
            wrong_n=0; trace_n=0
            for cmp in baseline_dec:
                a=baseline_dec[cmp]
                b=wrong_dec[cmp]
                c=trace_dec[cmp]
                wrong_n+=int(a!=b);trace_n+=int(a!=c)
                st['wrong_strict_flips']+=int(a in ('A','B') and b in ('A','B') and a!=b)
                st['trace_wrong_strict_flips']+=int(a in ('A','B') and c in ('A','B') and a!=c)
            st['n_wrong_decision_changes']+=wrong_n
            st['n_trace_wrong_decision_changes']+=trace_n
            st['n_wrong_cells_any_change']+=int(wrong_n>0)
            st['n_trace_wrong_cells_any_change']+=int(trace_n>0)
            case_row['moving_frames'][name] = {'corrected_decisions':corrected_dec,'naive_white_decisions':wrong_dec,
                       'naive_white_trace_normalized_decisions':trace_dec}
        rows.append(case_row)
    for name,st in stats.items():
        for metric in ('max_gram_delta','max_d2_delta','max_bound_delta','max_port_energy_delta'):
            if st[metric]>ATOL:
                raise AssertionError(('covariance invariance failed',name,metric,st[metric]))
    return {'status':'CPU_synthetic_whitening_covariance_audit_only',
            'source_cells':75,'comparisons_per_frame':375,'codebooks':list(old.LABELS),
            'source_sha256':SOURCE_SHA256,'N0':N0,
            'stages':stats,'printed_channel_blackbox':printed_control(),
            'observations':rows,
            'evidence':{'P':'conditional coherent AWGN/colored-noise algebra',
                        'M':'frozen synthetic sweep and local CPU black-box regression',
                        'R':'naive use of Euclidean receiver metric after nonunitary frame transform',
                        'no_Munich_16_rx_matrices':True,
                        'no_novel_physical_result':True}}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    source_blob=args.source.read_bytes()
    if hashlib.sha256(source_blob).hexdigest()!=SOURCE_SHA256:
        raise ValueError('source SHA256 does not match frozen data')
    if args.out.exists():
        raise FileExistsError('refusing to overwrite prior result')
    result=run(json.loads(source_blob))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    compact={k:v for k,v in result.items() if k!='observations'}
    print(json.dumps(compact,sort_keys=True,indent=2))


if __name__=='__main__':
    main()
