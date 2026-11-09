"""Quantify the strict fixed-trace gap in the coordinatewise Gaussian union bound."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import erfc
from gram_uncertainty_certificate import books, N0
from gram_decision_sweep import sweep

SOURCE_SHA256='16ad7ca1eda73d9c6a73776555be2f93333d248ad44d11a32ac9be3b05d2f4c6'
PAIRS={'rank_one':((0,1),(0,2)),
       'qpsk_squared':((0,1),(0,4)),
       'circular_16psk':((0,1),(0,7))}


def gram(case):
    a=case['channel'];z=complex(a['cross_correlation_real'],a['cross_correlation_imag'])
    return np.array([[a['port_gain_split'][0],z],[z.conjugate(),a['port_gain_split'][1]]],complex)


def line_coherence(C,pairs):
    d=C[pairs[0][0]]-C[pairs[0][1]]
    e=C[pairs[1][0]]-C[pairs[1][1]]
    return float(abs(np.vdot(d,e))/(np.linalg.norm(d)*np.linalg.norm(e)))


def pairwise_gap(G,C,pairs,eps):
    c=line_coherence(C,pairs)
    if not 0<=c<=1+1e-12:raise ValueError('invalid projective line coherence')
    f=lambda q:.5*erfc(np.sqrt(q)/(2*np.sqrt(N0)))
    ds=[]
    for i,j in pairs:
        d=C[i]-C[j]
        n2=float(np.vdot(d,d).real)
        q=float(np.vdot(d,G@d).real)
        if q-eps*n2<=0:raise ValueError('uncertainty ball could create collision')
        ds.append(float(f(q-eps*n2)-f(q-eps*c*n2)))
    return {'coherence':c,'pair_deltas':ds,'union_gap_lower_bound':min(ds)/8.}


def trace_effective_example():
    H=np.eye(2,dtype=complex)/np.sqrt(2)
    Sigma=np.diag([1.,4.]).astype(complex)
    G=H.conj().T@np.linalg.solve(Sigma,H)
    return {'frobenius_H_squared':float(np.linalg.norm(H,'fro')**2),
            'trace_Hdagger_Sigma_inverse_H':float(np.trace(G).real),
            'Sigma_diagonal':[1.,4.]}


def audit_grid():
    cases=sweep()['cases']
    if len(cases)!=75:raise ValueError('frozen grid size changed')
    C=books()
    rows=[]
    for index,case in enumerate(cases):
        G=gram(case)
        eps=min(1e-3,float(np.linalg.eigvalsh(G)[0])/4)
        for name,pairs in PAIRS.items():
            row=pairwise_gap(G,C[name],pairs,eps)
            rows.append({'cell':index,'book':name,'epsilon':eps,**row})
    result={}
    for name in PAIRS:
        sub=[x for x in rows if x['book']==name]
        result[name]={
            'cells':len(sub),
            'coherence':sub[0]['coherence'],
            'minimum_individual_delta':min(min(x['pair_deltas']) for x in sub),
            'minimum_aggregate_union_gap_lower_bound':min(x['union_gap_lower_bound'] for x in sub),
            'maximum_aggregate_union_gap_lower_bound':max(x['union_gap_lower_bound'] for x in sub),
            'all_pairwise_gaps_positive':all(x['union_gap_lower_bound']>0 for x in sub),
            'joint_endpoint_attained_for_all_cells':name=='circular_16psk',
        }
    return {'scope':'75 synthetic Gram cells; fixed codebooks; fixed-trace perturbations',
            'source_sha256':SOURCE_SHA256,'noise_N0':N0,
            'assumption':'trace-preserving Hermitian effective-Gram error; valid only if independently justified for the estimated effective Gram',
            'pair_choices':{k:[list(x) for x in v] for k,v in PAIRS.items()},
            'books':result,'colored_noise_trace_control':trace_effective_example(),
            'interpretation':'rank_one and QPSK2 separable union upper endpoints are strictly unattainable under one fixed traceless E; circular single-beam endpoints are jointly attainable',
            'no_new_physical_data':True}


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv)
    if a.out.exists():raise FileExistsError('refuse to overwrite prior evidence')
    r=audit_grid()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    print(json.dumps(r,indent=2,sort_keys=True))


if __name__=='__main__':main()
