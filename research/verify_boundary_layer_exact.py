"""Exact Q(sqrt(2), i) certificate for two 16-point boundary-layer codebooks.

SymPy-only. Checks the x1 collision-class structure used in the analytical
AWGN double-scaling limit and a secondary exact collision direction.
"""

from collections import Counter
import sympy as sp

S = sp.sqrt(2)
I = sp.I


def equal(a, b):
    return sp.simplify(sp.expand(a-b)) == 0


def codebooks():
    c = [1,S/2,0,-S/2,-1,-S/2,0,S/2]
    s = [0,S/2,1,S/2,0,-S/2,-1,-S/2]
    rank=[(sp.expand(t*c[m]/S), sp.expand(t*s[m]/S))
          for m in range(8) for t in (1+I,1-I)]
    g=[(a+b*I)/2 for a in (1,-1) for b in (1,-1)]
    qpsk=[(a,b) for a in g for b in g]
    return rank,qpsk


def classes(C, eps):
    groups=[]
    for x in C:
        val=sp.expand(x[0]+eps*x[1])
        for g in groups:
            if equal(g[0][0],val):
                g.append((val,x))
                break
        else:
            groups.append([(val,x)])
    return groups


def verify():
    rank,qpsk=codebooks()
    a=classes(rank,0)
    b=classes(qpsk,0)
    assert sorted(map(len,a)) == [1]*4+[2]*4+[4],list(map(len,a))
    assert sorted(map(len,b)) == [4]*4,list(map(len,b))
    for C in (rank,qpsk):
        assert len(C)==16
        assert all(equal(sp.expand(sp.conjugate(x[0])*x[0]
                                  + sp.conjugate(x[1])*x[1]),1) for x in C)
    for g in a:
        xs=[v[1][1] for v in g]
        if len(g)==2:
            assert equal(sp.expand(sp.conjugate(xs[0]-xs[1])*(xs[0]-xs[1])),2)
        elif len(g)==4:
            assert set((sp.simplify(sp.re(x)),sp.simplify(sp.im(x))) for x in xs) == {
                (a*S/2,b*S/2) for a in (1,-1) for b in (1,-1)}
        else:
            assert len(g)==1
    for g in b:
        xs=[v[1][1] for v in g]
        assert set((sp.simplify(sp.re(x)),sp.simplify(sp.im(x))) for x in xs) == {
            (sp.Rational(a,2),sp.Rational(b,2)) for a in (1,-1) for b in (1,-1)}
    rho_ray=S-1
    ranks=classes(rank,rho_ray)
    qpsks=classes(qpsk,rho_ray)
    assert Counter(map(len,ranks)) == Counter({2:8})
    assert Counter(map(len,qpsks)) == Counter({1:16})
    assert equal(sp.expand(rho_ray), S-1)
    print('PASS: exact x1 classes rank:',Counter(map(len,a)), 'qpsk:',Counter(map(len,b)))
    print('PASS: exact intra-class BPSK/QPSK geometry')
    print('PASS: secondary ray eps=sqrt(2)-1 rank class sizes:',Counter(map(len,ranks)),
          'qpsk:',Counter(map(len,qpsks)))
    print('At this secondary ray, rank-one has SER floor 1/2, QPSK2 floor 0')


if __name__=='__main__':
    verify()
