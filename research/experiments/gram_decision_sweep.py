"""Bound-certified Gram-matrix SER comparisons for dual-polarized 16-symbol books.

No Sionna, Mitsuba, GPU or Monte Carlo. Every decision compares mathematical
pairwise Gaussian error bounds for a fixed known 2x2 Gram matrix G=H^H H.
The 16-point fixed source designs match NVlabs/sionna-rt PR #82.
The channel-aware beamformed controls REQUIRE transmitter CSI; do not describe
these oracle references as deployed no-CSI modems or globally optimal designs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import erfc
from munich_provenance_gate import load_munich_archive, receiver_positions as source_receiver_positions

NOISE_N0 = 0.01
RATIOS = (0.01, 0.03, 0.10, 0.35, 1.00)  # positive-definite only: avoids tie ambiguity
THETAS_DEG = (0., 22.5, 45., 67.5, 90.)
PHIS_DEG = (0., 90., 180.)
SOURCE_COMMIT = "57435845d3395b2d5e167b0893035e15ffbdaa0d"
LABELS = ("rank_one", "qpsk_squared", "circular_16psk",
          "equal_port_psk16_oracle", "eigen_psk16_oracle", "eigen_qam16_oracle")


def fixed_codebooks() -> dict[str, np.ndarray]:
    """Frozen PR #82 sources plus deliberately off-null equal-port 16PSK."""
    g = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2.)
    qpsk = np.array([[a, b] for a in g for b in g])
    qpsk /= np.sqrt((np.abs(qpsk)**2).sum(axis=1).mean())
    rank = []
    for m in range(8):
        v = np.array([np.cos(2*np.pi*m/8), np.sin(2*np.pi*m/8)])
        for w in (np.array([1.,1.]), np.array([1.,-1.])):
            X = np.outer(v,w)
            rank.append([X[0,0]+1j*X[0,1], X[1,0]+1j*X[1,1]])
    rank = np.asarray(rank, complex)
    rank /= np.sqrt((np.abs(rank)**2).sum(axis=1).mean())
    phases = np.exp(2j*np.pi*np.arange(16)/16)
    circ = phases[:, None] * np.array([1.,1j])[None,:] / np.sqrt(2.)
    return {"rank_one":rank, "qpsk_squared":qpsk,
            "circular_16psk":circ}


def stokes(z: np.ndarray) -> np.ndarray:
    """For any Jones vector z, return (S1,S2,S3) without norm division."""
    z = np.asarray(z, complex)
    prod = np.conj(z[...,0])*z[...,1]
    return np.stack((2*prod.real,2*prod.imag,
                     np.abs(z[...,0])**2-np.abs(z[...,1])**2),axis=-1)


def gram_from_spectrum(ratio: float, theta_deg: float, phi_deg: float) -> np.ndarray:
    """trace(G)=1, lambda_min/lambda_max=ratio; principal u=(cos t,e^iφ sin t)."""
    if not (0 < ratio <= 1) or not (0 <= theta_deg <= 90):
        raise ValueError("require 0 < lambda_ratio <= 1 and 0 <= theta_deg <= 90")
    t,p = np.deg2rad([theta_deg,phi_deg])
    u = np.array([np.cos(t),np.exp(1j*p)*np.sin(t)])
    lmax,lmin = 1/(1+ratio),ratio/(1+ratio)
    G = lmin*np.eye(2) + (lmax-lmin)*np.outer(u,u.conj())
    return (G+G.conj().T)/2  # remove harmless floating Hermitian drift


def gram_spectrum(G: np.ndarray) -> tuple[float, float | None, float | None, float]:
    """Eigenvalue ratio, theta, phi, and trace; orientation undefined if equal."""
    G = _validated_gram(G)
    w,V = np.linalg.eigh(G)
    ratio = float(w[0]/w[1])
    if abs(w[1]-w[0]) < 1e-12:
        return ratio, None, None, float(w.sum())
    u = V[:,1]
    theta = float(np.rad2deg(np.arctan2(abs(u[1]),abs(u[0]))))
    phi = float(np.rad2deg(np.angle(u[1]/u[0]))) if abs(u[0])>1e-12 else None
    return ratio,theta,phi,float(w.sum())


def _validated_gram(G: np.ndarray) -> np.ndarray:
    G = np.asarray(G,complex)
    if G.shape != (2,2) or not np.isfinite(G).all():
        raise ValueError("G must be finite Hermitian 2x2")
    if np.max(abs(G-G.conj().T)) > 1e-11:
        raise ValueError("G not Hermitian")
    G = (G+G.conj().T)/2
    if np.linalg.eigvalsh(G)[0] < -1e-11 or np.trace(G).real <= 0:
        raise ValueError("G must be nonzero positive semidefinite")
    return G


def gram_quadratic(G: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Received squared displacement, direct quadratic form Delta^H G Delta."""
    G = _validated_gram(G)
    d = np.asarray(delta,complex)
    if d.shape[-1] != 2:
        raise ValueError("delta last dimension must be 2")
    val = np.einsum("...i,ij,...j->...",d.conj(),G,d)
    if np.max(np.abs(val.imag)) > 1e-10 or np.min(val.real) < -1e-10:
        raise ArithmeticError("quadratic is not real nonnegative")
    return np.maximum(val.real,0.)


def gram_quadratic_stokes(G: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Independent Pauli/Stokes derivation, retains split and both cross-terms."""
    G = _validated_gram(G)
    d=np.asarray(delta,complex)
    a = float(np.trace(G).real/2)
    b = (G[0,0].real-G[1,1].real)/2
    # d^H G d = a ||d||^2 + Re(G01) S1 + Im(-G01) S2 + b S3.
    # Stokes S2=2 Im(conj(d1)*d2), and 2 Re(G01 * conj(d1)*d2)
    # = Re(G01)*S1 - Im(G01)*S2.
    s = stokes(d)
    val = a*np.sum(np.abs(d)**2,axis=-1) + G[0,1].real*s[...,0] - G[0,1].imag*s[...,1] + b*s[...,2]
    return np.asarray(val, dtype=float)


def principal_beam(G: np.ndarray) -> np.ndarray:
    """Unit eigenvector for max eigenvalue (fixed deterministic gauge)."""
    G = _validated_gram(G)
    w,V = np.linalg.eigh(G)
    if w[1]-w[0] <= 1e-12:
        return np.array([1+0j,0j]) # degenerate; no preferred eigenvector
    u = V[:,1]
    idx = int(np.argmax(np.abs(u)))
    return u*np.exp(-1j*np.angle(u[idx]))


def equal_port_beam(G: np.ndarray) -> np.ndarray:
    """Optimal phase for equal-port single-beam PSK: v=(1,e^-i argG01)/sqrt2."""
    G = _validated_gram(G)
    phase = -np.angle(G[0,1]) if abs(G[0,1])>1e-14 else 0.
    return np.array([1, np.exp(1j*phase)],complex)/np.sqrt(2)


def beam_psk16(v: np.ndarray) -> np.ndarray:
    """Constant-energy 16PSK along a given unit-norm Jones beam."""
    v=np.asarray(v,complex)
    if v.shape!=(2,) or not np.all(np.isfinite(v)) or abs(np.vdot(v,v)-1)>1e-10:
        raise ValueError("beam must be a unit C^2 vector")
    return np.exp(2j*np.pi*np.arange(16)/16)[:,None]*v[None,:]


def beam_qam16(v: np.ndarray) -> np.ndarray:
    """16QAM along beam, mean symbol energy 1, nonconstant energy up to 1.8."""
    v=np.asarray(v,complex)
    if v.shape!=(2,) or abs(np.vdot(v,v)-1)>1e-10:
        raise ValueError("beam must be a unit C^2 vector")
    grid=np.array([a+1j*b for a in (-3,-1,1,3) for b in (-3,-1,1,3)])/np.sqrt(10)
    return grid[:,None]*v[None,:]


def codebooks_for_gram(G: np.ndarray) -> dict[str, np.ndarray]:
    """Fixed codebooks plus CSI-aware equal-port and unrestricted-port oracles."""
    C=fixed_codebooks()
    C['equal_port_psk16_oracle']=beam_psk16(equal_port_beam(G))
    u=principal_beam(G)
    C['eigen_psk16_oracle']=beam_psk16(u)
    C['eigen_qam16_oracle']=beam_qam16(u)
    return C


def gaussian_ser_bounds(G: np.ndarray, C: np.ndarray, N0: float) -> dict:
    """Strict nearest-neighbor lower / all-pair union upper for coherent AWGN.

    For nondegenerate received codebook only; no hidden clipping.
    P(i prefers j)=Q(sqrt(d2/(2*N0)))=0.5 erfc(sqrt(d2)/(2sqrt(N0))).
    """
    if N0<=0:
        raise ValueError("N0 must be > 0")
    C=np.asarray(C,complex)
    if C.shape!=(16,2):
        raise ValueError("expected 16x2 codebook")
    D=C[:,None,:]-C[None,:,:]
    d2=gram_quadratic(G,D)
    upper_iu = np.triu_indices(16,1)
    min_d2=float(np.min(d2[upper_iu]))
    if min_d2 < 1e-12:
        raise ValueError("received codebook has collisions; tie-aware rule required")
    p=0.5*erfc(np.sqrt(d2)/(2*np.sqrt(N0)))
    np.fill_diagonal(p,0.)
    lower=float(np.mean(np.max(p,axis=1)))
    upper=float(np.mean(np.sum(p,axis=1)))
    return {'lower':lower, 'upper':upper, 'min_d2':min_d2,
            'average_port_power':np.mean(np.abs(C)**2,axis=0).tolist(),
            'mean_symbol_energy':float(np.mean(np.sum(np.abs(C)**2,axis=1))),
            'peak_symbol_energy':float(np.max(np.sum(np.abs(C)**2,axis=1)))}


def bound_decision(a: dict, b: dict, tol: float = 1e-12) -> str:
    """Certify 'A' iff its U<L_B, 'B' iff U_B<L_A, else abstain."""
    if a['upper']+tol < b['lower']:
        return 'A'
    if b['upper']+tol < a['lower']:
        return 'B'
    return 'undecided'


def describe_gram(G: np.ndarray) -> dict:
    """Sufficient channel statistic and polarization orientation diagnostics."""
    G=_validated_gram(G)
    ratio,theta,phi,tr=gram_spectrum(G)
    w=np.linalg.eigvalsh(G)
    return {'eigenvalue_ratio':float(ratio),'principal_theta_deg':theta,
            'principal_phi_deg':phi,'trace':tr,
            'lambda_max':float(w[-1]),'lambda_min':float(w[0]),
            'port_gain_split':[float(G[0,0].real),float(G[1,1].real)],
            'cross_correlation_real':float(G[0,1].real),
            'cross_correlation_imag':float(G[0,1].imag)}


def evaluate_gram(G: np.ndarray, N0: float = NOISE_N0) -> dict:
    C=codebooks_for_gram(G)
    bounds={label:gaussian_ser_bounds(G,C[label],N0) for label in LABELS}
    a=bounds['rank_one']
    decisions={key:bound_decision(a,bounds[key]) for key in LABELS if key!='rank_one'}
    # 'A' means rank one wins; 'B' means comparator wins.
    return {'channel':describe_gram(G),'bounds':bounds,'rank_one_vs':decisions}


def sweep() -> dict:
    """Frozen, finite synthetic sweep; mathematical bounds, no stochastic sampling."""
    cases=[]
    for ratio in RATIOS:
        for theta in THETAS_DEG:
            for phi in PHIS_DEG:
                G=gram_from_spectrum(ratio,theta,phi)
                datum=evaluate_gram(G)
                datum['grid']={'eigenvalue_ratio':ratio,'theta_deg':theta,'phi_deg':phi}
                cases.append(datum)
    summaries={}
    for name in LABELS:
        if name=='rank_one': continue
        values=[c['rank_one_vs'][name] for c in cases]
        summaries[name]={k:values.count(k) for k in ('A','B','undecided')}
    return {'protocol':'SYNTHETIC_SWEEP_PREREG_2026-11-04.md',
            'source_commit':SOURCE_COMMIT, 'N0':NOISE_N0,
            'trace_G':1.,'ratios':RATIOS,'thetas_deg':THETAS_DEG,'phis_deg':PHIS_DEG,
            'comparison_label':'A=rank_one certified better; B=comparator certified better; undecided=overlapping bounds',
            'reference_note':'oracle controls use perfect transmitter G; eigen-QAM peak energy 1.8; not global optimum',
            'n_cases':len(cases),'summary':summaries,'cases':cases}


def evaluate_receiver_npz(path: Path, N0: float = NOISE_N0, *,
                          allow_synthetic_fixture: bool = False,
                          allow_unverified_import: bool = False) -> dict:
    """Evaluate only structurally checked channels; NEVER promote origin from hashes.

    Source-scene metadata is a declaration by the archive creator, not
    independent evidence of a physical or ray-traced Munich observation.
    """
    H, meta, assessment = load_munich_archive(
        path, allow_synthetic_fixture=allow_synthetic_fixture,
        allow_unverified_import=allow_unverified_import)
    pos = source_receiver_positions()
    data=[]
    for idx,h in enumerate(H):
        e=evaluate_gram(h.conj().T@h,N0)
        e['rx_index']=idx
        e['rx_xyz_m']=pos[idx].tolist()
        e['y_group']=int(idx//2)
        data.append(e)
    return {'status':assessment['classification'],
            'origin_authenticated':False,
            'provenance':assessment,
            'archive':Path(path).name,
            'archive_sha256':meta['archive_sha256'],
            'source_commit':SOURCE_COMMIT,
            'n_receivers':len(data),'receivers':data}


def main(argv: list[str] | None = None) -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('gram_sweep_2026-11-04.json'))
    parser.add_argument('--channels-npz',type=Path,help='source-structured archive, diagnostics only')
    parser.add_argument('--allow-unverified-import',action='store_true',help='permit a manually imported NPZ for diagnostic calculations only')
    args=parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f'refusing to overwrite {args.out}')
    data=sweep()
    if args.channels_npz is None:
        data['per_receiver']={'status':'not_available', 'reason':'no 16-channel verified export supplied'}
    else:
        data['per_receiver']=evaluate_receiver_npz(args.channels_npz, allow_unverified_import=args.allow_unverified_import)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    print('cases',data['n_cases'],'summary',json.dumps(data['summary']))
    print('per_receiver',data['per_receiver']['status'])
    print('json_sha256',hashlib.sha256(args.out.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
