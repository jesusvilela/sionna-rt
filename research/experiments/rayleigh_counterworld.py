"""Preregistered CPU-only counterworld for Sionna RT PR #82.

Uses the exact PR #82 16-point source constructors. No ray tracing or GPU.
See PRE_REG_2026-11-04.txt, frozen before simulation.
"""
import json
import math
import hashlib
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
SEED=74283
N_CHANNELS=512
N_SYMBOLS=256
N0_GRID=(0.04, 0.08)
OUT=HERE/'rayleigh_results_2026-11-04.json'


def qpsk2():
    g=np.array([1+1j,1-1j,-1+1j,-1-1j])/np.sqrt(2)
    pts=np.array([[x1,x2] for x1 in g for x2 in g])
    return pts/np.sqrt((np.abs(pts)**2).sum(axis=1).mean())


def rank1():
    pts=[]
    for m in range(8):
        v=np.array([np.cos(2*np.pi*m/8),np.sin(2*np.pi*m/8)])
        for w in (np.array([1.0,1.0]),np.array([1.0,-1.0])):
            X=np.outer(v,w)
            pts.append([X[0,0]+1j*X[0,1],X[1,0]+1j*X[1,1]])
    pts=np.array(pts)
    return pts/np.sqrt((np.abs(pts)**2).sum(axis=1).mean())


def unitary_haar_2(rng,n):
    out=np.empty((n,2,2),dtype=np.complex128)
    for k in range(n):
        z=(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))/np.sqrt(2)
        u,r=np.linalg.qr(z)
        out[k]=u@np.diag(np.exp(-1j*np.angle(np.diag(r))))
    return out


def generate_channels(rng,regime,n):
    if regime=='isotropic_rayleigh':
        return (rng.normal(size=(n,2,2))+1j*rng.normal(size=(n,2,2)))/2
    if regime=='rotated_anisotropic':
        U=unitary_haar_2(rng,n)
        return np.einsum('ab,nbc->nac',np.diag([1,np.sqrt(0.1)]),U)
    raise ValueError(regime)


def errors_by_channel(C,H,tx,noise):
    # H: [N,r,d], C: [M,d]. Broadcast over receiver dimension.
    n=H.shape[0]
    errs=np.zeros(n,dtype=np.int64)
    for k in range(0,n,64):
        HH=H[k:k+64]
        Y=np.einsum('nrd,md->nmr',HH,C)
        obs=Y[np.arange(Y.shape[0])[:,None], tx[k:k+64]]+noise[k:k+64]
        cost=np.sum(np.abs(obs[:,:,None,:]-Y[:,None,:,:])**2,axis=-1)
        predict=np.argmin(cost,axis=-1)
        errs[k:k+64]=np.sum(predict!=tx[k:k+64],axis=1)
    return errs


def cluster_ci(errA,errB,number_symbols,seed):
    diff=(errA-errB)/number_symbols
    rng=np.random.default_rng(seed)
    # Only channels are independent sampling units; symbols within each channel correlate.
    idx=rng.integers(0,len(diff),size=(2000,len(diff)))
    boots=diff[idx].mean(axis=1)
    lo,hi=np.percentile(boots,[2.5,97.5])
    return dict(delta=float(diff.mean()), ci95=[float(lo),float(hi)], bootstrap_channels=2000)


def numerical_invariants(Cq,Cr,Ct):
    for label,C in [('QPSK2',Cq),('rank1',Cr),('off_cone_twin',Ct)]:
        assert C.shape==(16,2),label
        np.testing.assert_allclose(np.sum(abs(C)**2,axis=1),1,atol=1e-13)
        sq=np.sum(abs(C[:,None,:]-C[None,:,:])**2,axis=2)
        assert sq[np.triu_indices(16,1)].min()>1e-10
    q=lambda C: np.imag(np.conj(C[:,0])*C[:,1])
    assert np.max(abs(q(Cr)))<1e-12
    assert np.max(abs(q(Ct)))>0.49
    # Frame covariance for the off-cone twin, exact received candidate arrays.
    rng=np.random.default_rng(20261104)
    H=(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))/2
    D=np.diag([1,1j]); np.testing.assert_allclose(Ct@ (H @ D.conj().T).T,Cr@H.T,atol=1e-13)


def pep_rayleigh_averaged(distance_sq,N0,r=2):
    """Craig integral, H_ij proper CN(0,1/r)."""
    if distance_sq<=0: return .5
    xi,wi=np.polynomial.legendre.leggauss(96)
    theta=(xi+1)*np.pi/4
    w=wi*np.pi/4
    return float(np.sum(w*(1+distance_sq/(4*r*N0*np.sin(theta)**2))**(-r))/np.pi)


def union_bound_averaged_iid(C,N0,r=2):
    sq=np.sum(np.abs(C[:,None,:]-C[None,:,:])**2,axis=-1)
    return 2/len(C)*sum(pep_rayleigh_averaged(float(d),N0,r=r) for d in sq[np.triu_indices(len(C),1)])


def known_answer_detector_check():
    C=qpsk2(); H=np.eye(2,dtype=complex)[None,:,:]
    rng=np.random.default_rng(18993)
    for n0 in (.08,):
        n=120000
        tx=rng.integers(0,16,size=(1,n))
        noise=np.sqrt(n0/2)*(rng.normal(size=(1,n,2))+1j*rng.normal(size=(1,n,2)))
        obs=errors_by_channel(C,H,tx,noise)[0]/n
        q=0.5*math.erfc(1/(2*math.sqrt(n0)))
        exact=1-(1-q)**4
        tolerance=6*math.sqrt(exact*(1-exact)/n)+.0005
        assert abs(obs-exact)<tolerance,(obs,exact,tolerance)
    return {'estimated':float(obs),'exact':float(exact),'six_sigma_tolerance':float(tolerance)}


def main():
    Cq=qpsk2(); Cr=rank1(); D=np.diag([1.,1j]); Ct=Cr@D.T
    C={'qpsk2':Cq,'rank1':Cr,'off_cone_twin':Ct}
    numerical_invariants(Cq,Cr,Ct)
    known=known_answer_detector_check()
    out={'protocol':'PRE_REG_2026-11-04.txt','seed':SEED,
         'codebook_source':'NVlabs/sionna-rt PR #82 head 57435845d3395b2d5e167b0893035e15ffbdaa0d',
         'H_normalization':'H_ij~CN(0,1/r), r=2',
         'noise_model':'proper complex CN(0,N0 I_2)',
         'num_channels':N_CHANNELS,'symbols_per_channel':N_SYMBOLS,
         'known_answer_detector':known,'runs':{}}
    for regime_index,regime in enumerate(('isotropic_rayleigh','rotated_anisotropic')):
        rng=np.random.default_rng(SEED+100*regime_index)
        H=generate_channels(rng,regime,N_CHANNELS)
        tx=rng.integers(0,16,size=(N_CHANNELS,N_SYMBOLS))
        for j,N0 in enumerate(N0_GRID):
            noise=np.sqrt(N0/2)*(rng.normal(size=(N_CHANNELS,N_SYMBOLS,2)) + 1j*rng.normal(size=(N_CHANNELS,N_SYMBOLS,2)))
            errs={name:errors_by_channel(points,H,tx,noise) for name,points in C.items()}
            key=f'{regime}_N0_{N0:.2f}'
            out['runs'][key]={'regime':regime,'N0':N0,'sers':{key1:float(err.mean()/N_SYMBOLS) for key1,err in errs.items()},
                              'paired_diff_rank1_minus_qpsk2':cluster_ci(errs['rank1'],errs['qpsk2'],N_SYMBOLS,SEED+10000*regime_index+j),
                              'paired_diff_rank1_minus_off_cone':cluster_ci(errs['rank1'],errs['off_cone_twin'],N_SYMBOLS,SEED+20000*regime_index+j)}
            if regime=='isotropic_rayleigh':
                bounds={name:union_bound_averaged_iid(points,N0,r=2) for name,points in C.items()}
                out['runs'][key]['unclipped_averaged_union_bounds']=bounds
                assert abs(bounds['off_cone_twin']-bounds['rank1'])<1e-12
            print(key, json.dumps(out['runs'][key],sort_keys=True),flush=True)
    OUT.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print('RESULTS_FILE',OUT,'sha256',hashlib.sha256(OUT.read_bytes()).hexdigest(),flush=True)


if __name__=='__main__':
    main()
