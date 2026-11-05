"""Representation audit of existing Gram SER comparisons and physical resources.

No added geometric model or new wireless evidence. CPU-only NumPy/SciPy.
Source: frozen 75-cell synthetic Gram sweep at PR #82 research fork.
Re-run from repository root with:
 python research/experiments/gram_decision_sweep.py --out /tmp/sionna_gram.json
 python research/experiments/frame_resource_audit.py --source /tmp/sionna_gram.json --out /tmp/frame_resource.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import erfc

EXPECTED_SHA256 = "16ad7ca1eda73d9c6a73776555be2f93333d248ad44d11a32ac9be3b05d2f4c6"
N0 = 0.01
SU2_FRAMES = ((31.0, 0.71), (17.0, -0.49))
PORTS = (np.diag([1.0, 0.0]).astype(complex),
         np.diag([0.0, 1.0]).astype(complex))
NULL_OPERATOR = np.array([[0, -1j], [1j, 0]], dtype=complex)
LABELS = ("rank_one", "qpsk_squared", "circular_16psk", "equal_port_psk16_oracle",
          "eigen_psk16_oracle", "eigen_qam16_oracle")


def u2_mixing(angle_deg: float, phase_rad: float) -> np.ndarray:
    """Nontrivial complex SU(2) change of coordinates, unitary by construction."""
    angle = np.deg2rad(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    z = np.exp(1j * phase_rad)
    U = np.array([[c, z*s], [-np.conj(z)*s, c]], dtype=complex)
    np.testing.assert_allclose(U.conj().T @ U, np.eye(2), atol=3e-15)
    return U


def source_gram(case: dict) -> np.ndarray:
    a = case["channel"]
    b = complex(a["cross_correlation_real"], a["cross_correlation_imag"])
    G = np.array([[a["port_gain_split"][0], b],
                  [b.conjugate(), a["port_gain_split"][1]]], dtype=complex)
    if not np.allclose(G, G.conj().T) or np.linalg.eigvalsh(G).min() < -1e-11:
        raise ValueError("invalid original Gram")
    return G


def psk(beam: np.ndarray) -> np.ndarray:
    return np.exp(2j*np.pi*np.arange(16)/16)[:, None] * beam[None, :]


def qam(beam: np.ndarray) -> np.ndarray:
    symbols = np.array([a + 1j*b for a in (-3,-1,1,3) for b in (-3,-1,1,3)], complex)
    return (symbols/np.sqrt(10))[:, None] * beam[None, :]


def equal_port_beam(G: np.ndarray) -> np.ndarray:
    """Optimizes beam gain subject to |v0|^2=|v1|^2=1/2 in *this* basis."""
    phi = -np.angle(G[0,1]) if np.abs(G[0,1]) > 1e-14 else 0.0
    return np.array([1, np.exp(1j*phi)], dtype=complex)/np.sqrt(2)


def source_codebooks(G: np.ndarray) -> dict[str,np.ndarray]:
    g = np.array([1+1j, 1-1j, -1+1j, -1-1j])/np.sqrt(2)
    qpsk = np.array([[a,b] for a in g for b in g])/np.sqrt(2)
    rank = []
    for m in range(8):
        v = np.array([np.cos(2*np.pi*m/8), np.sin(2*np.pi*m/8)])
        for w in (np.array([1.,1.]),np.array([1.,-1.])):
            X = np.outer(v,w)
            rank.append([X[0,0]+1j*X[0,1], X[1,0]+1j*X[1,1]])
    rank = np.array(rank,complex)/np.sqrt(2)
    circ = psk(np.array([1,1j],complex)/np.sqrt(2))
    lam,V = np.linalg.eigh(G)
    beam = np.array([1.,0.],complex) if lam[-1]-lam[0] <= 1e-12 else V[:,-1]
    return {"rank_one":rank,"qpsk_squared":qpsk,"circular_16psk":circ,
            "equal_port_psk16_oracle":psk(equal_port_beam(G)),
            "eigen_psk16_oracle":psk(beam),"eigen_qam16_oracle":qam(beam)}


def transport(G: np.ndarray, C: np.ndarray, objects: tuple[np.ndarray,...], U: np.ndarray):
    """Coordinate change preserving all *physical* objects; no re-optimization."""
    return U @ G @ U.conj().T, C @ U.T, tuple(U @ obj @ U.conj().T for obj in objects)


def pair_distances(G: np.ndarray, C: np.ndarray) -> np.ndarray:
    d = C[:,None,:]-C[None,:,:]
    d2 = np.einsum("...i,ij,...j->...",d.conj(),G,d)
    if np.max(np.abs(d2.imag)) > 1e-9:
        raise ValueError("complex received squared distance")
    if np.min(d2.real) < -1e-9:
        raise ValueError("negative received squared distance")
    return np.maximum(d2.real,0.)


def bounds(G: np.ndarray, C: np.ndarray, N0_:float=N0) -> dict[str,float]:
    if N0_ <= 0 or C.shape != (16,2):
        raise ValueError("expected 16 complex 2-port symbols and N0>0")
    d = pair_distances(G,C)
    minimum = float(np.min(d[np.triu_indices(16,1)]))
    if minimum < 1e-12:
        raise ValueError("coincident codewords require tie-aware SER bounds")
    prob=0.5*erfc(np.sqrt(d)/(2*np.sqrt(N0_)))
    np.fill_diagonal(prob,0.)
    return {"min_d2":minimum,
            "lower":float(np.mean(np.max(prob,axis=1))),
            "upper":float(np.mean(np.sum(prob,axis=1)))}


def physical_powers(C:np.ndarray,ports:tuple[np.ndarray,np.ndarray]):
    return np.array([np.einsum('mi,ij,mj->m',C.conj(),Ea,C).real for Ea in ports]).T


def q_values(C:np.ndarray,Q:np.ndarray=NULL_OPERATOR):
    return np.einsum('mi,ij,mj->m',C.conj(),Q,C).real/2


def decision(a:dict,b:dict,tol=1e-12)->str:
    if a['upper']+tol < b['lower']:
        return 'A'
    if b['upper']+tol < a['lower']:
        return 'B'
    return 'undecided'


def fixed_frame():
    return u2_mixing(*SU2_FRAMES[0])


def moving_frame():
    return u2_mixing(*SU2_FRAMES[1]) @ fixed_frame()


def verify_case(case:dict) -> dict:
    G=source_gram(case)
    C=source_codebooks(G)
    original={name:bounds(G,C[name]) for name in LABELS}
    # This is a *separate* source-archive consistency check.
    for label in LABELS:
        archived=case['bounds'][label]
        for metric in ('lower','upper','min_d2'):
            if abs(original[label][metric]-archived[metric]) > 1e-10:
                raise AssertionError((case['grid'],label,metric,'archive drift'))
    for label in LABELS[1:]:
        if decision(original['rank_one'],original[label]) != case['rank_one_vs'][label]:
            raise AssertionError((case['grid'],label,'decision drift'))
    baseline_phys=physical_powers(C['equal_port_psk16_oracle'],PORTS)
    np.testing.assert_allclose(baseline_phys,np.full((16,2),0.5),atol=1e-13)
    max_bound_delta,max_d2_delta,max_port_delta,max_q_delta=0.,0.,0.,0.
    naive={}
    for frame_name,U in [('first',fixed_frame()),('second',moving_frame())]:
        Gp=U@G@U.conj().T
        Ps=tuple(U@P@U.conj().T for P in PORTS)
        Q=U@NULL_OPERATOR@U.conj().T
        transformed={}
        for label in LABELS:
            Cp=C[label]@U.T
            b=bounds(Gp,Cp)
            transformed[label]=b
            max_d2_delta=max(max_d2_delta,float(np.max(abs(pair_distances(G,C[label])-pair_distances(Gp,Cp)))))
            for key in ('lower','upper','min_d2'):
                max_bound_delta=max(max_bound_delta,abs(b[key]-original[label][key]))
            max_port_delta=max(max_port_delta,float(np.max(abs(physical_powers(C[label],PORTS)-physical_powers(Cp,Ps)))))
            max_q_delta=max(max_q_delta,float(np.max(abs(q_values(C[label])-q_values(Cp,Q)))))
        for label in LABELS[1:]:
            if decision(transformed['rank_one'],transformed[label])!=case['rank_one_vs'][label]:
                raise AssertionError((case['grid'],frame_name,label,'frame-induced false decision'))
        # Deliberately WRONG: the power constraint is left attached to the new
        # coordinate axes instead of transported with the physical transmitter.
        naiveC=psk(equal_port_beam(Gp))
        naive_bounds=bounds(Gp,naiveC)
        bcorrect=original['equal_port_psk16_oracle']
        pnaive=physical_powers(naiveC,Ps)
        correct_dec=decision(original['rank_one'],bcorrect)
        naive_dec=decision(transformed['rank_one'],naive_bounds)
        naive[frame_name]={
            'port_violation':float(np.max(abs(pnaive-.5))),
            'max_interval_delta':float(max(abs(naive_bounds[x]-bcorrect[x]) for x in ('lower','upper','min_d2'))),
            'changed_decision':bool(naive_dec!=correct_dec),
            'old_decision':correct_dec,'naive_decision':naive_dec,
        }
    return {'max_bound_delta':max_bound_delta,'max_d2_delta':max_d2_delta,
            'max_port_delta':max_port_delta,'max_q_delta':max_q_delta,'naive':naive}


def load_source(path:Path) -> dict:
    blob=Path(path).read_bytes()
    actual=hashlib.sha256(blob).hexdigest()
    if actual != EXPECTED_SHA256:
        raise ValueError(f'expected SHA256 {EXPECTED_SHA256}, got {actual}')
    data=json.loads(blob)
    if data.get('n_cases')!=75 or len(data.get('cases',[]))!=75:
        raise ValueError('not source 75-cell dataset')
    return data


def printed_channel_control() -> dict:
    H = np.array([[.92982936+.33584663j, .01006189+.00676795j],
                  [-.06026739-.00641686j, -.13674566-.01022593j]])
    G=H.conj().T@H
    books=source_codebooks(G)
    U=moving_frame()
    # An arbitrary (independent) complex unitary change of receiver basis.
    W=u2_mixing(25.0,-0.83)
    Hnew=W@H@U.conj().T
    delta=0.
    for C in books.values():
        Y=C@H.T
        Yp=(C@U.T)@Hnew.T
        delta=max(delta,float(np.max(abs(np.sum(abs(Y[:,None,:]-Y[None,:,:])**2,axis=-1)-
                                         np.sum(abs(Yp[:,None,:]-Yp[None,:,:])**2,axis=-1)))))
        a=bounds(G,C)
        b=bounds(Hnew.conj().T@Hnew,C@U.T)
        delta=max(delta,max(abs(a[k]-b[k]) for k in ('min_d2','lower','upper')))
    return {'max_distance_and_bound_delta':delta,'physical_channel': 'single rounded printed notebook receiver only'}


def run(data:dict)->dict:
    values=[verify_case(c) for c in data['cases']]
    overall={key:float(max(v[key] for v in values)) for key in
             ('max_bound_delta','max_d2_delta','max_port_delta','max_q_delta')}
    hist={}
    for frame_name in ('first','second'):
        hist[frame_name]={
            'naive_physical_port_constraint_violations':sum(v['naive'][frame_name]['port_violation']>1e-9 for v in values),
            'max_naive_port_violation':max(v['naive'][frame_name]['port_violation'] for v in values),
            'naive_interval_changes':sum(v['naive'][frame_name]['max_interval_delta']>1e-10 for v in values),
            'naive_certified_decision_changes':sum(v['naive'][frame_name]['changed_decision'] for v in values),
            'max_naive_interval_delta':max(v['naive'][frame_name]['max_interval_delta'] for v in values),
        }
    if max(overall.values()) >= 1e-10:
        raise AssertionError('covariant invariant failed at 1e-10 gate')
    return {
        'status':'representation_audit_only', 'source_sha256':EXPECTED_SHA256,
        'source_cells':75, 'source_munich_real_channels':0,
        'fixed_frame_degrees_radians':SU2_FRAMES[0], 'second_frame_degrees_radians':SU2_FRAMES[1],
        'original_sweep_summary':data['summary'],
        'all_75_x_5_strict_decisions_preserved':True,
        'full_coordinate_transport_max_abs':overall,
        'naive_fixed_diagonal_port_control':hist,
        'printed_receiver_control':printed_channel_control(),
        'scope':'no new channel data, no added geometry, no Bargmann predictive claim',
        'model':'known coherent channel, circular white complex AWGN and N0=0.01',
    }


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(argv)
    if args.out.exists():
        raise FileExistsError('refusing to overwrite existing audit evidence')
    r=run(load_source(args.source))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(r,sort_keys=True,indent=2)+'\n')
    print(json.dumps(r,indent=2,sort_keys=True))


if __name__=='__main__':
    main()
