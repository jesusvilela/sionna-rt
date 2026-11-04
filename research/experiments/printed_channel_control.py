"""GPU-free, source-matched control for Sionna RT PR #82's printed channel.

The 2x2 channel is copied from the notebook's *printed output*, rounded to
8 decimals. It is ONE channel, not the full 16-receiver Munich ensemble.
Statements about SER bounds are conditional on coherent CN(0,N0 I) noise.
"""
import json
import math
from scipy.integrate import quad
import numpy as np

SOURCE = 'jesusvilela/sionna-rt@57435845d3395b2d5e167b0893035e15ffbdaa0d:tutorials/Null-Cone-Constellations.ipynb'
H_PRINTED = np.array([
    [0.92982936+0.33584663j, 0.01006189+0.00676795j],
    [-0.06026739-0.00641686j, -0.13674566-0.01022593j],
], dtype=np.complex128)
NOISE_VARIANCE = 0.01  # PR #82: 20 dB -> N0=10^(-20/10).


def qpsk_squared():
    g = np.array([1+1j, 1-1j, -1+1j, -1-1j])/np.sqrt(2)
    C = np.array([[x1, x2] for x1 in g for x2 in g])
    return C/np.sqrt(np.mean(np.sum(abs(C)**2,axis=1)))


def rank_one():
    p=[]
    for m in range(8):
        v=np.array([np.cos(2*np.pi*m/8),np.sin(2*np.pi*m/8)])
        for w in (np.array([1.,1.]),np.array([1.,-1.])):
            X=np.outer(v,w)
            p.append([X[0,0]+1j*X[0,1], X[1,0]+1j*X[1,1]])
    C=np.array(p)
    return C/np.sqrt(np.mean(np.sum(abs(C)**2,axis=1)))


def equal_port_circular_psk16():
    """Off-null q=+1/2 on every symbol; equal power 1/2 each port."""
    k=np.arange(16)
    return np.exp(2j*np.pi*k/16)[:,None]*np.array([1,1j])[None,:]/np.sqrt(2)


def pairwise_gaussian_bounds(C,H,N0):
    """Rigorous coherent-ML lower/upper probability bounds (not CIs)."""
    if N0<=0: raise ValueError('N0 must be positive')
    C=np.asarray(C,complex); Y=C@np.asarray(H,complex).T
    D2=np.sum(abs(Y[:,None,:]-Y[None,:,:])**2,axis=-1)
    i,j=np.triu_indices(len(C),1)
    P=lambda d: np.array([.5*math.erfc(math.sqrt(max(0,float(x)))/(2*math.sqrt(N0))) for x in np.ravel(d)]).reshape(np.shape(d))
    # For proper CN(0,N0): pairwise PEP = Q(d/sqrt(2N0))
    upper=2*P(D2[i,j]).sum()/len(C)
    np.fill_diagonal(D2,np.inf)
    lower=P(D2.min(axis=1)).mean()
    return (float(lower),float(upper))


def circular_psk16_ser_exact(H,N0):
    """1D quadrature for coherent ML with 16 equally spaced phases."""
    v=np.array([1.,1j])/np.sqrt(2)
    A=float(np.linalg.norm(np.asarray(H)@v))
    sigma=math.sqrt(N0/2)
    t=math.tan(math.pi/16)
    # Correct iff Re(A+z)>0 and |Im(z)|<tan(pi/16)*Re(A+z).
    f=lambda u: math.exp(-u*u/2)/math.sqrt(2*math.pi)*math.erf((A/sigma+u)*t/math.sqrt(2))
    correct=quad(f,-A/sigma,12.,epsabs=1e-12,epsrel=1e-12)[0]
    return 1-correct


def printed_precision_error_bound(N0):
    """Safe error in PEP from rounding each real/imag H entry to 8 digits.

    |E_ij| <= sqrt(2)*0.5e-8, so ||E||_F <= sqrt(8)*0.5e-8.
    Every pair difference of unit vectors has norm <= 2, whence
    |distance(H+E)-distance(H)| <= 2||E||_F.
    Q is globally 1/sqrt(2pi)-Lipschitz; pairwise error <= L.
    In M=16: per-symbol closest-neighbor lower-bound error <=L,
    full union upper-bound error <=(M-1)L.
    """
    Emax=math.sqrt(8)*0.5e-8
    L=2*Emax/(math.sqrt(2*N0)*math.sqrt(2*math.pi))
    return {'H_fro_rounding_max':Emax,'pep_error_max':L,
            'lower_error_max':L,'upper_error_max':15*L}


def report():
    C={'rank_one':rank_one(),'qpsk_squared':qpsk_squared(),
       'circular_16psk':equal_port_circular_psk16()}
    channels={'printed_munich_rx0':H_PRINTED,'balanced':np.eye(2)/np.sqrt(2)}
    bounds={}
    for name,H in channels.items():
        bounds[name]={}
        for label,code in C.items():
            lo,up=pairwise_gaussian_bounds(code,H,NOISE_VARIANCE)
            D2=np.sum(abs((code@H.T)[:,None,:]-(code@H.T)[None,:,:])**2,axis=-1)
            bounds[name][label]={
                'ser_lower':lo,'ser_union_upper':up,
                'min_squared_distance':float(D2[np.triu_indices(16,1)].min()),
                'mean_received_energy':float(np.mean(np.sum(abs(code@H.T)**2,axis=1))),
                'mean_tx_port_power':np.mean(abs(code)**2,axis=0).tolist(),
            }
        bounds[name]['circular_16psk']['ser_exact_via_quadrature']=circular_psk16_ser_exact(H,NOISE_VARIANCE)
        bounds[name]['norm_H_fro']=float(np.linalg.norm(H))
    rp=bounds['printed_munich_rx0']['rank_one']['ser_lower']
    cp=bounds['printed_munich_rx0']['circular_16psk']['ser_union_upper']
    rb=bounds['balanced']['rank_one']['ser_union_upper']
    cb=bounds['balanced']['circular_16psk']['ser_lower']
    rounding=printed_precision_error_bound(NOISE_VARIANCE)
    assert rp-cp > rounding['lower_error_max']+rounding['upper_error_max']
    assert rb<cb
    return {'source':SOURCE,'noise_model':'CN(0,N0 I_2)', 'noise_variance':NOISE_VARIANCE,
            'caveat':'printed 8-digit channel only; no full 16-receiver dataset available',
            'results':bounds,'printed_precision_cert':rounding,
            'strict_printed_margin':rp-cp,'strict_balanced_margin':cb-rb}


if __name__=='__main__':
    print(json.dumps(report(),indent=2,sort_keys=True))
