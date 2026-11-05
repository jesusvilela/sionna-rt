"""Gram-metric high-SNR phase boundaries and strict finite-SNR SER bounds.

CPU-only, no Sionna import. Calibrated against saved 75-cell PR #82 sweep.
The finite-noise bounds retain guaranteed inequalities in exact arithmetic.
The EPS guard covers normal floating-point roundoff, not a proof of IEEE error.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import erfc

N0_DEFAULT = 0.01
EPS = 1e-12


def codebooks():
    g = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    q = np.array([[a,b] for a in g for b in g]) / np.sqrt(2)
    rank = []
    for m in range(8):
        v = np.array([math.cos(m*math.pi/4), math.sin(m*math.pi/4)])
        for w in (np.array([1.,1.]), np.array([1.,-1.])):
            X=np.outer(v,w)
            rank.append([complex(X[0,0],X[0,1]),complex(X[1,0],X[1,1])])
    rank = np.array(rank,complex) / np.sqrt(2)
    # The rank-one constructor's raw X has two entries with total energy 2:
    # after /sqrt(2), the complex symbol is unit-energy.
    circ = np.exp(2j*np.pi*np.arange(16)/16)[:,None]*np.array([1,1j])/np.sqrt(2)
    for C in (q,rank,circ):
        if not np.allclose(np.linalg.norm(C,axis=1),1,atol=1e-13):
            raise ArithmeticError('codebook energy error')
    return {'rank_one':rank,'qpsk_squared':q,'circular_16psk':circ}


def gram_from_spec(ratio,theta_deg,phi_deg):
    t,p=np.deg2rad((theta_deg,phi_deg))
    u=np.array([np.cos(t), np.exp(1j*p)*np.sin(t)])
    lam0=ratio/(1+ratio); lam1=1/(1+ratio)
    return lam0*np.eye(2)+(lam1-lam0)*np.outer(u,u.conj())


def pair_d2(G,C):
    D=C[:,None,:]-C[None,:,:]
    return np.einsum('...i,ij,...j->...',D.conj(),G,D).real


def min_d2(G,C):
    d2=pair_d2(G,C)
    return float(d2[np.triu_indices(len(C),1)].min())


def orbit_representatives(C):
    """Generate mu4 representatives; reject malformed/incomplete orbits."""
    reps=[];used=set()
    for i,x in enumerate(C):
        if i in used: continue
        orbit=[]
        for phase in (1,1j,-1,-1j):
            matches=np.flatnonzero(np.linalg.norm(C-phase*x,axis=1)<1e-10)
            if len(matches)!=1:raise ValueError('not a complete mu4 orbit')
            orbit.append(int(matches[0]))
        if len(set(orbit))!=4 or set(orbit)&used:raise ValueError('orbit overlapping')
        used.update(orbit);reps.append(x)
    if len(used)!=len(C):raise ValueError('orbit omitted')
    return np.asarray(reps)


def orbit_envelope_min_d2(G,C):
    """Exact mu4 formula: min(2*e_k, e_k+e_l - 2 max(|Re z|,|Im z|))."""
    V=orbit_representatives(C)
    e=np.einsum('ki,ij,kj->k',V.conj(),G,V).real
    candidates=list(2*e)
    for k in range(len(e)):
        for j in range(k+1,len(e)):
            z=np.vdot(V[k],G@V[j]);candidates.append(e[k]+e[j]-2*max(abs(z.real),abs(z.imag)))
    return float(min(candidates))


def circular_analytic_min_d2(G):
    """(2-2cos(pi/8)) * (tr G/2 - Im G01) for fixed v=(1,i)/sqrt2."""
    gain=float(np.trace(G).real/2 - G[0,1].imag)
    return float((2-2*np.cos(np.pi/8))*gain)


def high_snr_order(G,code_a,code_b):
    """Conditional eventual SER sign when min received squared distances differ.

    Larger d_min implies smaller SER for all sufficiently small N0, by the
    PEP lower/union upper sandwich. Equal minima leave multiplicities open.
    """
    a=orbit_envelope_min_d2(G,code_a) if len(code_a)==16 and _is_mu4(code_a) else min_d2(G,code_a)
    b=orbit_envelope_min_d2(G,code_b) if len(code_b)==16 and _is_mu4(code_b) else min_d2(G,code_b)
    if a>b+1e-10:return 'A'
    if b>a+1e-10:return 'B'
    return 'equal_or_unresolved'


def _is_mu4(C):
    try:orbit_representatives(C);return True
    except ValueError:return False


def standard_bounds(G,C,N0):
    d2=pair_d2(G,C)
    if d2[np.triu_indices(len(C),1)].min()<=1e-13:raise ValueError('colliding received points')
    p=.5*erfc(np.sqrt(np.maximum(d2,0))/(2*np.sqrt(N0)))
    np.fill_diagonal(p,0)
    return float(np.max(p,axis=1).mean()),float(p.sum(axis=1).mean())


def Q(x):
    return .5*math.erfc(x/math.sqrt(2))


def exact_beam_16qam_ser(gain,N0):
    """Square 16QAM, one beam, mean total energy 1; coherent complex AWGN."""
    if gain<0 or N0<=0:raise ValueError('power and N0')
    p=Q(math.sqrt(gain/(5*N0)))
    return 1-(1-1.5*p)**2


def rigorous_beam_16psk_bounds(gain,N0):
    """2p-Q(A/sigma) <= SER <= 2p from the two wedge halfspace events.

    Intersection of both wrong halfspaces entails the longitudinal receiver
    noise has reversed the mean: X < -A. No numerical quadrature required.
    """
    if gain<=0 or N0<=0:raise ValueError('positive gain and N0 required')
    A=math.sqrt(gain);sigma=math.sqrt(N0/2)
    p=Q(A*math.sin(math.pi/16)/sigma)
    overlap_ub=Q(A/sigma)
    return max(0,2*p-overlap_ub), min(1,2*p)


def exact_diagonal_qpsk2_ser(G,N0):
    """Exact only when receive columns are orthogonal under white noise."""
    if abs(G[0,1])>1e-12:raise ValueError('nonorthogonal input channel')
    p1=Q(math.sqrt(max(G[0,0].real,0)/(2*N0)))
    p2=Q(math.sqrt(max(G[1,1].real,0)/(2*N0)))
    return 1-(1-p1)**2*(1-p2)**2


def beam_gain_from_label(G,label):
    if label=='circular_16psk':
        v=np.array([1,1j])/math.sqrt(2)
    elif label=='equal_port_psk16_oracle':
        ph=-np.angle(G[0,1]) if abs(G[0,1])>1e-14 else 0.
        v=np.array([1,np.exp(1j*ph)])/math.sqrt(2)
    elif label in ('eigen_psk16_oracle','eigen_qam16_oracle'):
        lam,V=np.linalg.eigh(G);v=V[:,-1]
    else:raise ValueError('not a beam-based codebook')
    return float(np.vdot(v,G@v).real)


def tightened_bounds(G,label,original,N0):
    """Intersect existing certified bounds with analytic PSK/QAM/orthogonal QPSK bounds.

    Includes 1e-12 numeric guard for computed exact expressions. This is
    a mathematical bound in real arithmetic; IEEE rounding itself is tested,
    not formally interval-certified.
    """
    lower,upper=original
    if label in ('circular_16psk','equal_port_psk16_oracle','eigen_psk16_oracle'):
        l,u=rigorous_beam_16psk_bounds(beam_gain_from_label(G,label),N0)
    elif label=='eigen_qam16_oracle':
        p=exact_beam_16qam_ser(beam_gain_from_label(G,label),N0)
        l,u=p,p
    elif label=='qpsk_squared' and abs(G[0,1])<1e-13:
        p=exact_diagonal_qpsk2_ser(G,N0)
        l,u=p,p
    else:
        return lower,upper
    lo=max(lower,l-EPS)
    hi=min(upper,u+EPS)
    if lo>hi+2e-10:raise AssertionError((label,original,(l,u),'inconsistent'))
    return lo,hi


def decision(a,b,tol=1e-12):
    if a[1]+tol<b[0]:return 'A'
    if b[1]+tol<a[0]:return 'B'
    return 'undecided'


def gram_of_case(c):
    z=c['channel']
    return np.array([[z['port_gain_split'][0],z['cross_correlation_real']+1j*z['cross_correlation_imag']],
                     [z['cross_correlation_real']-1j*z['cross_correlation_imag'],z['port_gain_split'][1]]],dtype=complex)


def run_sweep(archived):
    books=codebooks()
    summary={k:{'old':{'A':0,'B':0,'undecided':0},'new':{'A':0,'B':0,'undecided':0},'newly_decided':0,
                'conflict':0,'high_snr_A':0,'high_snr_B':0,'high_snr_equal_or_unresolved':0,
                'finite_vs_high_snr_disagreement':0} for k in archived['summary']}
    cell_outputs=[]
    for c in archived['cases']:
        G=gram_of_case(c)
        saved={k:(v['lower'],v['upper']) for k,v in c['bounds'].items()}
        # Independent implementations match archived distance diagnostics.
        for label,C in books.items():
            d=min_d2(G,C)
            if not math.isclose(d,c['bounds'][label]['min_d2'],rel_tol=2e-9,abs_tol=2e-10):
                raise AssertionError((c['grid'],label,d,c['bounds'][label]['min_d2']))
        new={k:tightened_bounds(G,k,(v['lower'],v['upper']),archived['N0']) for k,v in c['bounds'].items()}
        case={'grid':c['grid'],'decisions':{},'tightened':{k:{'lower':v[0],'upper':v[1]} for k,v in new.items()}}
        for other in summary:
            old=decision(saved['rank_one'],saved[other]);now=decision(new['rank_one'],new[other]);hist=summary[other]
            hs=high_snr_order(G,books['rank_one'],books[other]) if other in books else 'not_applicable'
            hist['old'][old]+=1; hist['new'][now]+=1
            hist['newly_decided']+=int(old=='undecided' and now!='undecided')
            hist['conflict']+=int(old!='undecided' and now!='undecided' and old!=now)
            if hs in ('A','B','equal_or_unresolved'):hist['high_snr_'+hs]+=1
            if hs in ('A','B') and now in ('A','B') and hs!=now:hist['finite_vs_high_snr_disagreement']+=1
            case['decisions'][other]={'old':old,'new':now,'high_snr':hs}
        cell_outputs.append(case)
    for label,s in summary.items():
        if s['old']!=archived['summary'][label]:raise AssertionError((label,'archive mismatch',s['old']))
        if s['conflict']:raise AssertionError((label,'finite-bound certificate reversed',s['conflict']))
    return {'model':'fixed 16-symbol coherent complex AWGN','N0':archived['N0'],
            'source_json_sha256':None,'n_cases':len(cell_outputs),'summary':summary,'cases':cell_outputs}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    d=json.loads(a.archive.read_text())
    result=run_sweep(d)
    import hashlib
    result['source_json_sha256']=hashlib.sha256(a.archive.read_bytes()).hexdigest()
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('n_cases',result['n_cases'])
    print(json.dumps(result['summary'],indent=2))

if __name__=='__main__':main()
