"""Symbolic certificate of the rank-one channel collision discriminant.

Optional dependency: SymPy. Independently uses exact Q(sqrt(2), i) arithmetic
rather than the floating-point constellation or numerical ray clustering.
No GPU, Sionna RT, external service, or Monte Carlo. ~15s on a CPU.

Usage: python research/verify_collision_discriminant_exact.py
"""

from collections import Counter
import sympy as sp


SQRT2 = sp.sqrt(2)
I = sp.I


def zero(expr):
    """Decide zero for the algebraic expressions in Q(sqrt(2), i)."""
    return sp.expand(sp.simplify(expr)) == 0


def codebooks():
    c = [1, SQRT2/2, 0, -SQRT2/2, -1, -SQRT2/2, 0, SQRT2/2]
    s = [0, SQRT2/2, 1, SQRT2/2, 0, -SQRT2/2, -1, -SQRT2/2]
    rank = [(sp.expand(t*c[m]/SQRT2), sp.expand(t*s[m]/SQRT2))
            for m in range(8) for t in (1+I, 1-I)]
    scalar = [(a+b*I)/2 for a in (1, -1) for b in (1, -1)]
    qpsk = [(a, b) for a in scalar for b in scalar]
    return {'rank_one': rank, 'qpsk_squared': qpsk}


def exact_projection_class_count(C, h):
    values = [sp.expand(h[0]*x[0] + h[1]*x[1]) for x in C]
    representatives = []
    sizes = []
    for y in values:
        for k, representative in enumerate(representatives):
            if zero(y-representative):
                sizes[k] += 1
                break
        else:
            representatives.append(y)
            sizes.append(1)
    return len(representatives), tuple(sorted(sizes))


def exact_discriminant(C):
    """All exceptional points [h1:h2] in CP1, exact group and full proof.

    Every unordered pair (i,j) determines the unique collision ray
    h=(delta_x2,-delta_x1), grouped by determinant zero. The 120 edges
    are partitioned into pair-direction groups. Connected components of
    each collision graph are the exactly coincident received symbols.
    """
    n = len(C)
    groups = []
    for a in range(n):
        for b in range(a+1, n):
            d = (sp.expand(C[a][0]-C[b][0]),
                 sp.expand(C[a][1]-C[b][1]))
            if zero(d[0]) and zero(d[1]):
                raise AssertionError('transmitted symbols are not distinct')
            for group in groups:
                e = group['delta']
                if zero(e[0]*d[1]-e[1]*d[0]):
                    group['edges'].append((a,b))
                    break
            else:
                groups.append({'delta':d, 'edges':[(a,b)]})
    assert sum(len(group['edges']) for group in groups) == n*(n-1)//2
    for k, group in enumerate(groups):
        d = group['delta']
        for other in groups[:k]:
            e = other['delta']
            assert not zero(e[0]*d[1]-e[1]*d[0]), 'duplicate direction'
        h = (d[1], -d[0])
        count, sizes = exact_projection_class_count(C, h)
        # Expected collision-pair count from the *symbol classes*
        # agrees with the symbolic projective grouping (cross-check).
        assert sum(t*(t-1)//2 for t in sizes) == len(group['edges'])
        group['num_received'] = count
        group['class_sizes'] = sizes
    return groups


def main():
    expected = {
        'rank_one': Counter({(4,12):8, (8,8):6, (10,9):4}),
        'qpsk_squared': Counter({(4,12):8, (10,9):4, (24,4):2}),
    }
    for name,C in codebooks().items():
        assert len(C) == 16
        groups = exact_discriminant(C)
        histogram = Counter((len(g['edges']), g['num_received']) for g in groups)
        assert histogram == expected[name], (name, histogram)
        axis_R, axis_sizes = exact_projection_class_count(C, (1,0))
        diagonal_R, _ = exact_projection_class_count(C, (1,1))
        generic_R, _ = exact_projection_class_count(C, (1,2+I))
        assert (axis_R, diagonal_R, generic_R) == (
            (9,9,16) if name=='rank_one' else (4,9,16))
        print(name, 'exceptional CP1 directions:',len(groups))
        print('  (colliding unordered pairs, distinct received points) -> rays:')
        for (pair_count,R), count in sorted(histogram.items()):
            print(' ',(pair_count,R), '->',count,
                  '    SER_floor:', sp.Rational(16-R,16))
        print('  axis class sizes:',axis_sizes,
              '  axis SER floor:',sp.Rational(16-axis_R,16))
        print('  R at row (1,1):',diagonal_R,
              '  R at row (1,2+i):',generic_R)
    print('PASS: exact collision discriminant and class-count certificate')


if __name__=='__main__':
    main()
