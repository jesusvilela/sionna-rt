"""Two conditional counterworlds for the 15 unpublished Munich receiver channels.

No ray tracing and NO claim that either completion occurs in the Munich scene.
The one source-printed H0 is held fixed. Both completions satisfy only the
published per-slot full-rank, trace/Frobenius and receiver-count constraints.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import erfc

N0 = 0.01
SOURCE_NOTEBOOK = "jesusvilela/sionna-rt@57435845d3395b2d5e167b0893035e15ffbdaa0d"
OBSERVED_SLOTS = 1
TOTAL_SLOTS = 16
PRINTED = np.array([
    [0.92982936+0.33584663j, 0.01006189+0.00676795j],
    [-0.06026739-0.00641686j, -0.13674566-0.01022593j],
], dtype=np.complex128)

PORTS = (
    np.array([[1,0],[0,0]],dtype=complex),
    np.array([[0,0],[0,1]],dtype=complex),
)
BOOK_NAMES = ("rank_one", "qpsk_squared", "circular_16psk")


def source_codebooks():
    g = np.array([1+1j,1-1j,-1+1j,-1-1j],complex)/np.sqrt(2)
    q = np.array([[a,b] for a in g for b in g],complex)/np.sqrt(2)
    r = []
    for m in range(8):
        v=np.array([np.cos(np.pi*m/4),np.sin(np.pi*m/4)])
        for s in (1,-1):
            r.append([(v[0]*(1+1j*s))/np.sqrt(2),
                      (v[1]*(1+1j*s))/np.sqrt(2)])
    r=np.asarray(r,complex)
    k=np.arange(16)
    circle=np.exp(2j*np.pi*k/16)[:,None]*np.array([1,1j])[None,:]/np.sqrt(2)
    out=dict(rank_one=r,qpsk_squared=q,circular_16psk=circle)
    for c in out.values():
        assert c.shape==(16,2)
        np.testing.assert_allclose(np.sum(abs(c)**2,axis=1),1,atol=1e-13)
        np.testing.assert_allclose(np.mean(abs(c)**2,axis=0),[.5,.5],atol=1e-13)
    return out


def gram_from_principal(ratio,theta_deg,phi_deg=0.):
    if not 0<ratio<=1:raise ValueError('ratio outside (0,1]')
    t,p=np.deg2rad([theta_deg,phi_deg])
    u=np.array([np.cos(t),np.exp(1j*p)*np.sin(t)],complex)
    G=(ratio*np.eye(2)+(1-ratio)*np.outer(u,u.conj()))/(1+ratio)
    return G


def positive_hermitian_root(G):
    """A full-rank artificial H with H†H=G. No physical scene assertion."""
    w,V=np.linalg.eigh(G)
    if w[0] <= 0:raise ValueError('positive-definite Gram required')
    H=(V*np.sqrt(w))@V.conj().T
    np.testing.assert_allclose(H.conj().T@H,G,atol=3e-15,rtol=0)
    np.testing.assert_allclose(np.linalg.norm(H,'fro'),1,atol=5e-15)
    return H


WITNESSES={
    "A_rank_one_better":dict(ratio=0.10,theta_deg=0.,phi_deg=0.,
                             expected="rank_one"),
    "B_rank_one_worse":dict(ratio=0.01,theta_deg=22.5,phi_deg=0.,
                            expected="comparators"),
}


def worlds():
    return {name:[PRINTED.copy()]+[positive_hermitian_root(
        gram_from_principal(spec['ratio'],spec['theta_deg'],spec['phi_deg'])) for _ in range(15)]
        for name,spec in WITNESSES.items()}


def direct_bounds(H,C,Sigma=None,N0_=N0):
    """Direct H×symbol event bounds, with full complex colored covariance."""
    if Sigma is None:Sigma=np.eye(2)
    if H.shape!=(2,2) or C.shape!=(16,2) or N0_<=0:raise ValueError('bad detector shape')
    sigma=np.asarray(Sigma,complex)
    if not np.allclose(sigma,sigma.conj().T,atol=1e-12) or np.linalg.eigvalsh(sigma).min()<=0:
        raise ValueError('noise covariance must be Hermitian positive definite')
    Y=C@H.T
    d=Y[:,None,:]-Y[None,:,:]
    d2=np.einsum('...i,ij,...j->...',d.conj(),np.linalg.solve(sigma,d.reshape(-1,2).T).T.reshape(d.shape)*0+sigma*0 if False else np.eye(2),d) if False else None
    # Compute the Mahalanobis quadratic on all 256 ordered differences.
    solved=np.linalg.solve(sigma,d.reshape(-1,2).T).T.reshape(d.shape)
    d2=np.einsum('...i,...i->...',d.conj(),solved).real
    if d2[np.triu_indices(16,1)].min()<=1e-13:raise ValueError('received duplicate symbols')
    if d2.min() < -1e-11:raise ValueError('negative Mahalanobis distance')
    p=.5*erfc(np.sqrt(np.maximum(d2,0))/(2*np.sqrt(N0_)))
    np.fill_diagonal(p,0)
    return {'lower':float(np.max(p,axis=1).mean()),
            'upper':float(p.sum(axis=1).mean()),
            'min_d2':float(np.min(d2[np.triu_indices(16,1)]))}


def gram_distances(H,C,Sigma=None):
    if Sigma is None:Sigma=np.eye(2)
    G=H.conj().T@np.linalg.solve(Sigma,H)
    d=C[:,None,:]-C[None,:,:]
    return np.einsum('...i,ij,...j->...',d.conj(),G,d).real


def check_direct_vs_gram(H,C,Sigma=None):
    if Sigma is None:Sigma=np.eye(2)
    Y=C@H.T
    d=Y[:,None,:]-Y[None,:,:]
    solved=np.linalg.solve(Sigma,d.reshape(-1,2).T).T.reshape(d.shape)
    direct=np.einsum('...i,...i->...',d.conj(),solved).real
    g=gram_distances(H,C,Sigma)
    discrepancy=float(np.max(abs(direct-g)))
    if discrepancy>1e-10:raise AssertionError(('quadratic direct/Gram drift',discrepancy))
    return discrepancy


def mixture_bounds(channels,codebooks=None):
    if codebooks is None:codebooks=source_codebooks()
    if len(channels)!=TOTAL_SLOTS:raise ValueError('expected 16 receiver slots')
    for v,H in enumerate(channels):
        if H.shape!=(2,2) or np.linalg.matrix_rank(H)!=2:
            raise ValueError(('dead/singular synthetic channel',v))
        if v>0 and abs(np.linalg.norm(H,'fro')-1)>5e-12:
            raise ValueError('unobserved slots must be normalized')
    rows={}
    for name,c in codebooks.items():
        per=[direct_bounds(H,c) for H in channels]
        rows[name]={k:float(np.mean([v[k] for v in per])) for k in ('lower','upper','min_d2')}
        # Mean min_d2 is auxiliary, not the aggregate constellation min distance.
        rows[name]['per_site_bounds']=[{'lower':v['lower'],'upper':v['upper']} for v in per]
    return rows


def comparisons(data):
    out={}
    rank=data['rank_one']
    for name in ('qpsk_squared','circular_16psk'):
        other=data[name]
        if rank['upper']+1e-10<other['lower']:
            out[name]={'winner':'rank_one','gap':other['lower']-rank['upper']}
        elif other['upper']+1e-10<rank['lower']:
            out[name]={'winner':name,'gap':rank['lower']-other['upper']}
        else:
            out[name]={'winner':'abstain','gap':None}
    return out


def su2(angle,phase):
    c,s=np.cos(angle),np.sin(angle);z=np.exp(1j*phase)
    U=np.array([[c,z*s],[-np.conj(z)*s,c]],complex)
    np.testing.assert_allclose(U.conj().T@U,np.eye(2),atol=1e-14)
    return U


def site_receive_change(site,level):
    """Deterministic, invertible and site-dependent GL(2,C) coordinate motion."""
    A=np.array([[1.2+.15j,.27-.13j],[.08+.22j,.82-.09j]],complex)
    B=np.array([[.91,.15+.17j],[-.12+.07j,1.09-.11j]],complex)
    p=np.diag([np.exp(1j*(site+1)*.051),np.exp(-1j*(site+1)*.023)])
    q=np.diag([np.exp(1j*(site+1)*.017),np.exp(-1j*(site+1)*.031)])
    W1=p@A
    return W1 if level==1 else (q@B)@W1


def frame_recheck(channels,base,codebooks):
    """Two moving input frames and 16 independent receiver-local frames."""
    out=[]
    U1=su2(np.deg2rad(31),.71)
    U2=su2(np.deg2rad(17),-.49)@U1
    for level,U in ((1,U1),(2,U2)):
        tports=tuple(U@P@U.conj().T for P in PORTS)
        maxi=0.
        for name,C in codebooks.items():
            Cp=C@U.T
            for P,Pp in zip(PORTS,tports):
                x=np.einsum('mi,ij,mj->m',C.conj(),P,C).real
                y=np.einsum('mi,ij,mj->m',Cp.conj(),Pp,Cp).real
                maxi=max(maxi,float(np.max(abs(x-y))))
            bounds_new=[]
            for site,H in enumerate(channels):
                W=site_receive_change(site,level)
                Hnew=W@H@U.conj().T
                sigma=W@W.conj().T
                bounds_new.append(direct_bounds(Hnew,Cp,sigma))
                maxi=max(maxi,check_direct_vs_gram(Hnew,Cp,sigma))
            for key in ('lower','upper'):
                maxi=max(maxi,abs(float(np.mean([b[key] for b in bounds_new]))-base[name][key]))
        if maxi>1e-10:raise AssertionError(('moving frame changed classification interval',level,maxi))
        out.append({'level':level,'max_metric_port_and_bounds_deviation':maxi})
    return out


def rounding_allowance():
    E=np.sqrt(8)*5e-9
    pep=2*E/np.sqrt(4*np.pi*N0)
    return {'H0_fro_error_max':E,'one_H0_PEP_error_max':pep,
            'ensemble_gap_loss_max':pep}


def report():
    C=source_codebooks()
    all_worlds=worlds()
    result={}
    eps=rounding_allowance()['ensemble_gap_loss_max']
    for name,channels in all_worlds.items():
        baseline=mixture_bounds(channels,C)
        for H in (channels[0],channels[1]):
            for code in C.values():check_direct_vs_gram(H,code)
        decided=comparisons(baseline)
        expected=WITNESSES[name]['expected']
        for comparator,result_ in decided.items():
            actual=result_['winner']
            want='rank_one' if expected=='rank_one' else comparator
            if actual!=want or result_['gap']<=1e-4:
                raise AssertionError((name,comparator,actual,want,result_['gap']))
            if result_['gap']<=eps:
                raise AssertionError('H0 eight-decimal rounding can erase strict gap')
        frames=frame_recheck(channels,baseline,C)
        result[name]={
            'construction':{k:v for k,v in WITNESSES[name].items() if k!='expected'},
            'missing_slots':15,'same_source_printed_H0':True,
            'all_synthetic_missing_H_full_rank_frobenius_one':True,
            'per_codebook_interval':{k:{p:b[p] for p in ('lower','upper')} for k,b in baseline.items()},
            'comparisons':decided,
            'min_rounding_adjusted_gap':min(v['gap']-eps for v in decided.values()),
            'two_successive_moving_frame_checks':frames,
            'scientific_origin':'synthetic counterworld, NOT the Munich scene',
        }
    return {'scope':'Partial identification from H0+16 alive trace-normalized H slots only',
            'source':SOURCE_NOTEBOOK,
            'P':'linear averaging of conditional SER lower/upper bounds, and moving-frame covariance',
            'M':'two numeric synthetic mathematical counterexamples, independent of Sionna runtime',
            'H':'physical Sionna Munich scene realizability is NOT established',
            'R':'claim that printed H0 plus basic per-slot constraints identifies 16-site ordering',
            'not_independently_verified_munich_data':True,'real_original_munich_matrices_recovered':0,
            'N0':N0,'rounding_allowance':rounding_allowance(),
            'worlds':result}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv)
    if args.out.exists():raise FileExistsError('do not overwrite an earlier audit')
    data=report()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(data,sort_keys=True,indent=2)+'\n')
    print(json.dumps({name: {'intervals':x['per_codebook_interval'],
                              'comparisons':x['comparisons'],
                              'two_stage_frame_checks':x['two_successive_moving_frame_checks']}
                      for name,x in data['worlds'].items()},indent=2))


if __name__=='__main__':main()
