"""Transport-covariant Bargmann projector loops on a 16-site Munich lattice.

P, transports, and complex loop invariants are retained as typed objects.
This is geometry of *channel Gram principal eigenlines*, not radio-wave
propagation holonomy, and does not add information beyond full channel Grams.

CPU only: Python stdlib + NumPy. No Sionna/GPU/HF/network.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SOURCE_COMMIT = "57435845d3395b2d5e167b0893035e15ffbdaa0d"
IDENTITY = np.eye(2, dtype=complex)


def projector_from_gram(G, relative_gap=1e-7):
    """Return principal Hermitian rank-one projector or None at degeneracy.

    The abstention is obligatory at equal eigenvalues, where no principal
    eigenline is canonically selected by the channel.
    """
    G = np.asarray(G, dtype=complex)
    if G.shape != (2, 2) or not np.isfinite(G).all():
        raise ValueError('expected finite complex 2x2 Gram')
    if np.max(np.abs(G - G.conj().T)) > 1e-10:
        raise ValueError('Gram is not Hermitian')
    G = (G + G.conj().T) / 2
    values = np.linalg.eigvalsh(G)
    if values[0] < -1e-10 or values.sum() <= 0:
        raise ValueError('Gram must be nonzero positive semidefinite')
    gap = float(values[1] - values[0])
    if gap <= relative_gap * float(values.sum()):
        return None
    P = (G - values[0] * IDENTITY) / gap
    return (P + P.conj().T) / 2


def stokes_from_projector(P):
    """Jones Stokes convention (S1,S2,S3), with S2=-2 Im(P01)."""
    P = np.asarray(P, dtype=complex)
    return np.array((2 * P[0,1].real, -2 * P[0,1].imag,
                     (P[0,0]-P[1,1]).real), dtype=float)


def bargmann(Pa, Pb, Pc):
    """Complex three-projector Bargmann invariant in one shared input frame."""
    return complex(np.trace(Pa @ Pb @ Pc))


def bargmann_stokes(na, nb, nc):
    """Independent Pauli-vector formula (orientation retained)."""
    a, b, c = (np.asarray(v, dtype=float) for v in (na, nb, nc))
    return complex((1 + a@b + b@c + c@a) / 4,
                   (a@np.cross(b, c)) / 4)


def transported_bargmann(projectors, transports, triangle):
    """Typed transport T[(i,j)]: maps the *j* fiber into the *i* fiber."""
    i,j,k = triangle
    return complex(np.trace(projectors[i] @ transports[(i,j)] @
                            projectors[j] @ transports[(j,k)] @
                            projectors[k] @ transports[(k,i)]))


def reframe(projectors, gauges):
    """Independent receiver-local U(2) coordinate changes, with transitions."""
    if len(projectors) != len(gauges):
        raise ValueError('gauges and projectors must have equal site count')
    n = len(projectors)
    P = [gauges[i] @ projectors[i] @ gauges[i].conj().T for i in range(n)]
    T = {(i,j): gauges[i] @ gauges[j].conj().T
         for i in range(n) for j in range(n) if i != j}
    return P,T


def covariant_edge_difference(Pi, Pj, T_i_from_j):
    """Matrix-valued edge derivative in site i's frame, not a scalar norm.

    The connection/transport must be given. With source-coordinate identity
    transport it reduces to Pj-Pi; it is not inferred from the radio channel.
    """
    T=np.asarray(T_i_from_j,dtype=complex)
    if T.shape!=(2,2) or np.max(abs(T.conj().T@T-IDENTITY))>1e-10:
        raise ValueError('a unitary typed transport is required')
    return T @ Pj @ T.conj().T - Pi


def receiver_positions():
    return np.array([(x,y,1.5) for y in np.linspace(15,60,8)
                     for x in (25.,35.)], dtype=float)


def oriented_triangles():
    """14 positively oriented triangles, two per consecutive y-strip."""
    return [(2*y,2*y+1,2*y+2) for y in range(7)] + [
            (2*y+1,2*y+3,2*y+2) for y in range(7)]


def oriented_boundary_edges(triangles=None):
    """Return the oriented boundary of the source triangulation (16 edges)."""
    from collections import Counter
    triangles = oriented_triangles() if triangles is None else triangles
    counts = Counter((a,b) for i,j,k in triangles
                     for a,b in ((i,j),(j,k),(k,i)))
    boundary=[]
    for (i,j),num in counts.items():
        reverse=counts.get((j,i),0)
        if num != 1 or reverse > 1:
            raise ValueError('nonmanifold or duplicated oriented edge')
        if reverse == 0:
            boundary.append((i,j))
    return boundary


def surface_phase(projectors, transports=None, floor=1e-10):
    """Discrete Stokes/Pancharatnam gluing on the 14 oriented mesh faces.

    Product of normalized complex face Bargmann loops = product of normalized
    transported Jones overlaps on oriented boundary. No quantization claim.
    """
    if len(projectors) != 16 or any(P is None for P in projectors):
        raise ValueError('need 16 nondegenerate principal projectors')
    vectors = [np.linalg.eigh(p)[1][:,1] for p in projectors]
    def T(i,j):
        return IDENTITY if transports is None else transports[(i,j)]
    def edge(i,j):
        z=np.vdot(vectors[i],T(i,j)@vectors[j])
        if abs(z)<=floor:
            raise ValueError('zero-overlap edge: phase undefined')
        return z/abs(z)
    face_product=1+0j
    for i,j,k in oriented_triangles():
        B=transported_bargmann(projectors,transports,(i,j,k)) if transports is not None else bargmann(projectors[i],projectors[j],projectors[k])
        if abs(B)<=floor:
            raise ValueError('zero-overlap triangle: phase undefined')
        face_product*=B/abs(B)
    boundary=oriented_boundary_edges()
    boundary_product=1+0j
    for i,j in boundary:
        boundary_product*=edge(i,j)
    return {'faces':14,'boundary_edges':boundary,
            'euler_characteristic':16-29+14,
            'face_product':[float(face_product.real),float(face_product.imag)],
            'boundary_product':[float(boundary_product.real),float(boundary_product.imag)],
            'phase_rad':float(np.angle(face_product)),
            'gluing_residual':float(abs(face_product-boundary_product))}


def field_diagnostics(H, relative_gap=1e-7, loop_floor=1e-10):
    """Keep projectors and complex loops; abstain at degeneracy/zero overlap."""
    H=np.asarray(H,dtype=complex)
    if H.shape != (16,2,2) or not np.isfinite(H).all():
        raise ValueError('expected 16 alive, finite 2x2 complex channel matrices')
    norms = np.linalg.norm(H,axis=(1,2))
    if not np.allclose(norms, 1, atol=5e-7):
        raise ValueError('channels must be source-style Frobenius normalized')
    positions=receiver_positions()
    P=[]; sites=[]
    for i,h in enumerate(H):
        G=h.conj().T @ h
        w=np.linalg.eigvalsh(G)
        p=projector_from_gram(G,relative_gap)
        P.append(p)
        sites.append({'rx':i,'xyz_m':positions[i].tolist(),
                      'eigenvalues':w.tolist(),
                      'relative_gap':float((w[1]-w[0])/sum(w)),
                      'projector_available':p is not None,
                      'projector_re_im':[[[float(z.real),float(z.imag)] for z in row] for row in p] if p is not None else None,
                      'stokes':stokes_from_projector(p).tolist() if p is not None else None,
                      'gram_re_im':[[[float(z.real),float(z.imag)] for z in row] for row in G]})
    # Matrix-valued edges of the oriented triangular receiver complex.
    # Source transmit coordinates are shared, so T_{i<-j}=I here.
    # If local frames are later supplied, transport MUST accompany them.
    pairs=sorted({tuple(sorted((a,b))) for tri in oriented_triangles()
                  for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0]))})
    edges=[]
    for i,j in pairs:
        if P[i] is None or P[j] is None:
            edges.append({'edge':[i,j],'matrix_re_im':None,'state':'degenerate_endpoint'})
        else:
            D=covariant_edge_difference(P[i],P[j],IDENTITY)
            edges.append({'edge':[i,j],
                          'matrix_re_im':[[[float(z.real),float(z.imag)] for z in row] for row in D],
                          'state':'defined'})
    loops=[]
    for i,j,k in oriented_triangles():
        if any(P[s] is None for s in (i,j,k)):
            loops.append({'triangle':[i,j,k],'state':'eigenline_degenerate','bargmann':None,'phase_rad':None})
            continue
        z=bargmann(P[i],P[j],P[k])
        state='phase_undefined_zero_overlap' if abs(z)<=loop_floor else 'defined'
        loops.append({'triangle':[i,j,k],'state':state,
                      'bargmann':[float(z.real),float(z.imag)],
                      'phase_rad':float(np.angle(z)) if state=='defined' else None,
                      'overlap_magnitude':float(abs(z))})
    try:
        surface=surface_phase(P)
    except ValueError as err:
        surface={'state':'undefined','reason':str(err)}
    return {'type':'synthetic_or_verified_channel_Gram_projector_field',
            'sites':sites,'edges':edges,'triangles':loops,'surface_gluing':surface,
            'warning':'Bargmann loops of principal Gram eigenlines; not Maxwell propagation holonomy or incremental information beyond full G.'}


def load_verified_munich(path):
    """Require file SHA and full 16-slot source manifest before any real labeling."""
    path=Path(path)
    meta_path=path.with_suffix('.manifest.json')
    if not meta_path.is_file():
        raise ValueError('missing verified exporter manifest')
    info=json.loads(meta_path.read_text())
    if (info.get('source_commit')!=SOURCE_COMMIT or
        info.get('num_receiver_positions')!=16 or
        info.get('all_16_available') is not True):
        raise ValueError('incomplete or mismatched source provenance')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=info.get('archive_sha256'):
        raise ValueError('NPZ SHA-256 mismatch')
    with np.load(path,allow_pickle=False) as d:
        H=np.asarray(d['H_all_normalized'],complex)
        mask=np.asarray(d['alive'],bool)
        coords=np.asarray(d['rx_positions_xyz_m'],float)
    if H.shape!=(16,2,2) or mask.shape!=(16,) or not mask.all():
        raise ValueError('not 16 alive channels')
    if not np.allclose(coords,receiver_positions(),atol=1e-8):
        raise ValueError('receiver geometry does not match source order')
    return H,info


def demo():
    north=np.array([1.,0j]); east=np.array([1.,1.])/np.sqrt(2)
    north_y=np.array([1.,1j])/np.sqrt(2)
    def proj(u):return np.outer(u,u.conj())
    P=[proj(u) for u in (north,east,north_y)]
    z=bargmann(*P)
    anti=bargmann(P[0],P[1],np.conj(P[2]))
    return {'status':'synthetic_toy_only','B_forward':[z.real,z.imag],
            'B_mirror':[anti.real,anti.imag],
            'phase_forward_rad':float(np.angle(z)),
            'phase_mirror_rad':float(np.angle(anti)),
            'pairwise_overlap_matrix':[[float(np.trace(a@b).real) for b in P] for a in P],
            'triangles_munich_lattice':oriented_triangles()}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channels-npz',type=Path)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args(argv)
    if args.channels_npz is None:
        result=demo()
        result['per_receiver']={'status':'not_available','reason':'no verified 16-Munich-channel archive supplied'}
    else:
        H,manifest=load_verified_munich(args.channels_npz)
        result=field_diagnostics(H)
        result['per_receiver']={'status':'derived_from_verified_archive',
                                'archive_sha256':manifest['archive_sha256'],
                                'mode':manifest.get('mode')}
    payload=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if args.out:
        if args.out.exists():raise FileExistsError(args.out)
        args.out.write_text(payload)
    else:
        print(payload)


if __name__=='__main__':main()
