"""Spectral-norm robustness certificate for fixed coherent 16-symbol codebooks.

A conservative uncertainty acceptance test, NOT an inference about 16 Munich H.
Only standard Gaussian pairwise SER envelopes and known 2x2 matrix geometry.
"""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import numpy as np
from scipy.special import erfc

N0 = 0.01
GAP_TOL = 1e-12
PRINTED = np.array([
    [.92982936+.33584663j,.01006189+.00676795j],
    [-.06026739-.00641686j,-.13674566-.01022593j],
],complex)
PORTS = (np.diag([1.,0.]).astype(complex),np.diag([0.,1.]).astype(complex))
WORLD_SPECS = {'A_rank_one_better':(.10,0.), 'B_rank_one_worse':(.01,22.5)}


def books():
    g=np.array([1+1j,1-1j,-1+1j,-1-1j])/np.sqrt(2.)
    q=np.array([[a,b] for a in g for b in g])/np.sqrt(2.)
    r=[]
    for m in range(8):
        v=(np.cos(np.pi*m/4),np.sin(np.pi*m/4))
        for s in (1,-1):
            r.append([v[0]*(1+1j*s)/np.sqrt(2),v[1]*(1+1j*s)/np.sqrt(2)])
    z=np.exp(2j*np.pi*np.arange(16)/16)
    c=z[:,None]*np.array([1,1j])[None,:]/np.sqrt(2)
    return {'rank_one':np.array(r,complex),'qpsk_squared':q,'circular_16psk':c}


def principal_gram(r,theta):
    t=np.deg2rad(theta)
    u=np.array([np.cos(t),np.sin(t)],complex)
    return (r*np.eye(2)+(1-r)*np.outer(u,u.conj()))/(1+r)


def worlds():
    return {name:[PRINTED.conj().T@PRINTED]+[principal_gram(*spec)]*15
            for name,spec in WORLD_SPECS.items()}


def _valid_gram(G):
    G=np.asarray(G,complex)
    if G.shape!=(2,2) or not np.isfinite(G).all():
        raise ValueError('finite complex 2x2 Gram required')
    if np.max(abs(G-G.conj().T))>1e-11:
        raise ValueError('Gram is not Hermitian')
    w=np.linalg.eigvalsh(G)
    if w[0]<=0:
        raise ValueError('strictly positive definite nominal Gram required')
    return G, float(w[0])


def event_brackets(G,C,eps):
    """Pointwise guaranteed p_minus<=p_true<=p_plus for all admissible G.

    Explicitly excludes possible received-symbol collision from the uncertainty
    family by requiring eps < lambda_min(Ghat).
    """
    G,lmin=_valid_gram(G)
    C=np.asarray(C,complex)
    if C.shape!=(16,2) or not np.isfinite(C).all():
        raise ValueError('finite 16x2 codebook required')
    if not np.isfinite(eps) or eps<0 or eps>=lmin:
        raise ValueError('uncertainty radius must be nonnegative and below nominal min eigenvalue')
    d=C[:,None,:]-C[None,:,:]
    q=np.einsum('...i,ij,...j->...',d.conj(),G,d)
    if np.max(abs(q.imag))>1e-10:
        raise ArithmeticError('non-real Hermitian quadratic')
    power=np.sum(abs(d)**2,axis=-1)
    allowance=eps*power
    minus=np.maximum(0,q.real-allowance)
    plus=np.maximum(0,q.real+allowance)
    f=lambda x:.5*erfc(np.sqrt(x)/(2*np.sqrt(N0)))
    return f(plus),f(minus)


def robust_interval(G,C,eps):
    plo,phi=event_brackets(G,C,eps)
    offdiag=~np.eye(16,dtype=bool)
    # Diagonal i=j is not an error event; in particular its PEP is NOT 1/2.
    plo=np.where(offdiag,plo,0.)
    phi=np.where(offdiag,phi,0.)
    return {'L_guaranteed':float(np.max(plo,axis=1).mean()),
            'U_guaranteed':float(phi.sum(axis=1).mean())}


def printed_rounding_gram_radius():
    e=np.sqrt(8)*5e-9
    return 2*np.linalg.norm(PRINTED,'fro')*e+e*e


def ensemble_bounds(grams,C,unknown_epsilon):
    if len(grams)!=16:raise ValueError('requires 16 nominal Grams including H0')
    if unknown_epsilon <0:raise ValueError('negative radius')
    eps0=printed_rounding_gram_radius()
    per=[robust_interval(G,C,eps0 if i==0 else unknown_epsilon)
         for i,G in enumerate(grams)]
    return {key:float(np.mean([p[key] for p in per]))
            for key in ('L_guaranteed','U_guaranteed')}


def robust_winner(a,b):
    if a['U_guaranteed']+GAP_TOL<b['L_guaranteed']:
        return 'A',b['L_guaranteed']-a['U_guaranteed']
    if b['U_guaranteed']+GAP_TOL<a['L_guaranteed']:
        return 'B',a['L_guaranteed']-b['U_guaranteed']
    return 'abstain',None


def radius_for_witness(grams,books_,competitor,expected):
    if expected not in ('A','B'):raise ValueError('unknown comparison direction')
    floor=min(float(np.linalg.eigvalsh(g)[0]) for g in grams[1:])
    ceiling=min(0.1,floor)
    def compare(e):
        ra=ensemble_bounds(grams,books_['rank_one'],e)
        rb=ensemble_bounds(grams,books_[competitor],e)
        return robust_winner(ra,rb)
    initial=compare(0.)
    if initial[0]!=expected:
        raise AssertionError((competitor,'nominal strict sign not certified',initial))
    low,high=0.,ceiling
    # Take the limit from BELOW the strict positive-definite boundary.
    for _ in range(48):
        mid=(low+high)/2
        if mid>=floor or compare(mid)[0]!=expected:
            high=mid
        else:
            low=mid
    return {'certified_radius_lower':low,'first_failure_or_PD_upper':high,
            'numerical_bracket_width':high-low,
            'min_eigenvalue_of_nominal_unknown_sites':floor,
            'nominal_strict_margin':initial[1],
            'envelope_at_safe_radius':compare(low)[0],
            'ceiling_is_PD_gate':high>=floor*(1-1e-10)}


def su2(angle_deg,phase):
    t=np.deg2rad(angle_deg);c,s=np.cos(t),np.sin(t);z=np.exp(1j*phase)
    return np.array([[c,z*s],[-z.conjugate()*s,c]],complex)


def site_W(site,level):
    """Two nested invertible receiver transforms with site-dependent phases."""
    A=np.array([[1.2+.15j,.27-.13j],[.08+.22j,.82-.09j]],complex)
    B=np.array([[.91,.15+.17j],[-.12+.07j,1.09-.11j]],complex)
    p=np.diag([np.exp(1j*(site+1)*.051),np.exp(-1j*(site+1)*.023)])
    q=np.diag([np.exp(1j*(site+1)*.017),np.exp(-1j*(site+1)*.031)])
    first=p@A
    return first if level==1 else (q@B)@first


def positive_root(G):
    w,V=np.linalg.eigh(G)
    if w[0]<=0:raise ValueError('positive-definite nominal Gram required')
    H=(V*np.sqrt(w))@V.conj().T
    np.testing.assert_allclose(H.conj().T@H,G,atol=1e-12)
    return H


def moving_frame_check(grams,books_,e):
    """Two input rotations and independent nonunitary receiver transforms;
    the noise covariance and physical port projectors move with the frames.
    """
    base={name:ensemble_bounds(grams,C,e) for name,C in books_.items()}
    stages=(su2(31,.71),su2(17,-.49)@su2(31,.71))
    max_delta=0.
    for level,U in enumerate(stages,1):
        newG=[]
        for site,G in enumerate(grams):
            H=positive_root(G)
            W=site_W(site,level)
            Hnew=W@H@U.conj().T
            Sigma=W@W.conj().T
            # Independent Cholesky-whitening witness (no explicit inverse).
            L=np.linalg.cholesky(Sigma)
            Hwhite=np.linalg.solve(L,Hnew)
            Gwhite=Hwhite.conj().T@Hwhite
            Gsolve=Hnew.conj().T@np.linalg.solve(Sigma,Hnew)
            Gtarget=U@G@U.conj().T
            max_delta=max(max_delta,float(np.max(abs(Gwhite-Gsolve))),
                          float(np.max(abs(Gwhite-Gtarget))))
            newG.append((Gwhite+Gwhite.conj().T)/2)
        transformed_ports=[U@P@U.conj().T for P in PORTS]
        for key,C in books_.items():
            Cp=C@U.T
            result=ensemble_bounds(newG,Cp,e)
            max_delta=max(max_delta,*[abs(result[k]-base[key][k]) for k in result])
            for old,new in zip(PORTS,transformed_ports):
                pre=np.einsum('mi,ij,mj->m',C.conj(),old,C).real
                post=np.einsum('mi,ij,mj->m',Cp.conj(),new,Cp).real
                max_delta=max(max_delta,float(np.max(abs(pre-post))))
    if max_delta>1e-10:raise AssertionError(('full covariance/frame transport failed',max_delta))
    return max_delta


def report():
    codebooks=books()
    for c in codebooks.values():
        np.testing.assert_allclose(np.sum(abs(c)**2,axis=1),1.,atol=5e-14)
        np.testing.assert_allclose(np.mean(abs(c)**2,axis=0),[.5,.5],atol=5e-14)
    results={}
    for name,grams in worlds().items():
        expected='A' if name.startswith('A_') else 'B'
        radii={comp:radius_for_witness(grams,codebooks,comp,expected)
               for comp in ('qpsk_squared','circular_16psk')}
        small=min(x['certified_radius_lower'] for x in radii.values())
        results[name]={'comparisons':radii,
                       'smallest_certified_radius':small,
                       'positive_budget_gate':bool(small>1e-6),
                       'moving_frame_max_delta_at_half_budget':moving_frame_check(grams,codebooks,small/2)}
    return {'scope':'synthetic uncertainty certificate, no real Munich data',
            'model':'fixed source codebooks; CN(0,N0 Sigma) ML; effective Gram, operator-norm balls',
            'unknown_sites':15,'original_munich_matrices_recovered':0,
            'printed_source_rounding_gram_radius':printed_rounding_gram_radius(),
            'two_counterworlds':results,
            'all_positive_gate':all(w['positive_budget_gate'] for w in results.values()),
            'kill_gate':'if any radius<=1e-6, no practical stability conclusion',
            'P':'conditional PSD operator-norm Gaussian PEP envelopes and frame covariance',
            'M':'numeric sufficient robustness radii on two preexisting synthetic worlds',
            'A':'unknown future source-matched Gram error radii, not measured',
            'R':'new physics, physical Munich uncertainty calibration or performance claims'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('refuse to overwrite experiment output')
    r=report()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    print(json.dumps(r,indent=2,sort_keys=True))

if __name__=='__main__':main()
