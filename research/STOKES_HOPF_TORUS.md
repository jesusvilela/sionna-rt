# From the (2,2) null cone to a Hopf torus — Sionna RT PR #82

**2026-11-04 | bounded mathematical continuation | fork only**

This is a **classical-geometry reconstruction** of the *actual finite codebooks* in [NVlabs/sionna-rt PR #82](https://github.com/NVlabs/sionna-rt/pull/82), not a new communication-physics claim. The later [resource-matched optimizer control](https://github.com/NVlabs/sionna-rt/pull/82#issuecomment-6491527495) retired the causal assertion that the determinant/null cone created the observed symbol-error-rate gain. That correction remains binding.

## Objects and assumptions

The signal is one Jones vector x = (x1,x2) in C² (two complex polarization components). Here every nonzero symbol in the two compared codebooks has ||x||² = 1. This **Jones-vector polarization model** does not by itself specify a physical antenna calibration, XPD model, propagation distribution or noisy receiver. All transformations below are algebraic **coordinate** changes unless explicitly identified as a physical codebook edit.

Write x1=a+ib, x2=c+id and let

    q(x) = det [[a,b],[c,d]] = ad-bc = Im(conj(x1)*x2).

It is a real quadratic form with signature (2,2). Its zero set in R⁴ is a cone with a singular vertex at x=0. We do not identify this determinant with channel rank, nor with a determinant of the transmitted/received channel matrix.

## Result 1 — the normalized null cone is a Clifford torus [P, direct algebra]

Make this **real-orthogonal** change of coordinates:

    u = ((a+d) + i(b-c))/sqrt(2)
    v = ((a-d) + i(b+c))/sqrt(2).

It obeys

    |u|²+|v|² = ||x||²,             |u|²-|v|² = 2 q(x).

Therefore on ||x||²=1 and q(x)=0,

    |u|=|v|=1/sqrt(2).

This is precisely a (unit-radius S³) Clifford torus S¹(1/sqrt2) × S¹(1/sqrt2). The unnormalized (2,2) cone is recovered by allowing the overall radius to vary.

**Important:** the (u,v) change is real-orthogonal on R⁴ but is not a claimed physically implementable complex-unitary polarization transform. Geometric coordinates and physical signal operations are different types.

## Result 2 — its Poincaré projection is a great circle [P, direct algebra]

Define the Stokes/Hopf map

    s(x) = (2 Re(conj(x1)*x2), 2 Im(conj(x1)*x2),
            |x1|²-|x2|²).

Direct expansion gives ||s(x)|| = ||x||². For unit-norm x, s(x) lies on S². A global phase x -> exp(iθ)x leaves s(x) unchanged, yielding the usual S³ -> S² Hopf map. Moreover

    q(x) = s_y(x)/2.

Thus the normalized null cone is the *full Hopf preimage* of the great circle s_y=0 on the Poincaré sphere. This agrees with the classical theorem that a Hopf preimage of a great circle is a Clifford torus; it is **not** a novelty claim.

For an arbitrary U in U(2), its adjoint action induces R(U) in SO(3), and

    s(Ux) = R(U) s(x).

The fixed-axis condition s_y=0 is generally NOT invariant. If the axis is also transported, n' = R(U)n, the typed moving-frame scalar

    q_n(x) = (n dot s(x))/2

satisfies q_{n'}(Ux) = q_n(x). That identity is coordinate covariance, not an improved communication system.

## Result 3 — full null-cone-preserving unitary subgroup [P, elementary proof]

For the **fixed** null cone q=0, the unitaries preserving its zero set for *every x* are exactly

    N = { exp(iθ) A : A in O(2), θ real }  subset U(2).

Sufficiency: for real A, q(exp(iθ)Ax)=det(A) q(x); hence q=0 is preserved.

Necessity: let U preserve q=0. The columns u1=U e1 and u2=U e2 separately lie on the cone, so each has one common component phase: uj=exp(iθj)aj, with aj a real nonzero vector. Unitarity makes a1,a2 a real orthonormal basis. Because e1+e2 is also on the cone, q(u1+u2)=0 forces sin(θ2-θ1) det(a1,a2)=0. Since det(a1,a2)=±1, the phase difference is 0 or π modulo 2π. The columns therefore share one common phase up to sign: U=exp(iθ)A, A in O(2).

Thus the fixed null cone has a definite stabilizer subgroup. Outside this subgroup, q changes even when a compensated channel/frame transformation leaves the received waveform unchanged.

## Result 4 — the two 16-point codebooks share a projective skeleton but not a fiber section [P for finite identities; M for numeric regression]

The actual rank-one constructor used in PR #82 is equivalent to

    x_rank(m, ±) = exp(± iπ/4) (cos(mπ/4), sin(mπ/4)),
    m = 0,...,7.

Its 16 Jones vectors project to **four** equally spaced points on s_y=0, each with four global-phase representatives.

The QPSK² constructor (also 16 unit-energy vectors) has

    x_QPSK = (exp(iθ1), exp(iθ2))/sqrt(2),
    θ1, θ2 in {π/4, 3π/4, 5π/4, 7π/4}.

It projects to **four** equally spaced points on s_z=0, each with four global-phase representatives. The unitary

    U0 = (I - i σ_x)/sqrt(2)

rotates the rank-one base-point set onto the QPSK² base-point set on S². But the *full* codebooks do not coincide after U0: their Hopf-fiber phase assignments differ. They cannot be globally U(2)-equivalent because global unitaries preserve Euclidean symbol distances while

    min_{i!=j} ||x_rank_i-x_rank_j|| = sqrt(2-sqrt(2)) ≈ 0.76537,
    min_{i!=j} ||x_QPSK_i-x_QPSK_j|| = 1.00000.

Equivalently, the Stokes-base arrangements and the **complex phase section over them** must not be collapsed into a single polarization-only object.

This gives a clean explanation of *where to look* for differences between these two engineered codebooks. It does **not** establish that a particular phase section caused the previously measured SER effect; that requires separate physical-channel and resource-matched tests.

## Local verification and boundary

Command (NumPy; no GPU, Sionna RT or Mitsuba):

    python -m unittest discover -s research/tests -p 'test_*.py' -v

**2026-11-04 local execution:** the original 8 frame-covariance tests and the 8 new Stokes/Hopf tests all pass: **16/16**. This check compares mathematical representations and the fixed codebook, not the original ray-traced SER data. The finite-precision tests are supporting regression evidence; the equations above carry the elementary proofs.

### Pre-registration for the next research gate

- **Prediction:** projected four-point Stokes sets align under U0; the complete codebooks have different phase sections and different minimum Euclidean distances. Consistent x'=Ux, H'=HU† changes no received candidate for any U in U(2).
- **Controls:** original rank-one and QPSK² constructors; real rotations and reflections; arbitrary complex unitaries; identity/global phase; non-normalizer relative phase; moving Stokes axis; received pairwise distances.
- **Primary metric:** algebraic equality (or <1e-11 numerical residue) of covariant quantities; fixed-set codebook distance differences are diagnostics, not an optimized modem score.
- **Kill condition:** if analytic or implementation identities fail, fix the formalization before discussing signal performance. If any future causal claim is matched by an unconstrained equal-budget optimizer, retire that causal claim.
- **Stop rule:** no GPU ray tracing, external NVIDIA PR or hypercomplex escalation on the strength of this geometry alone.
- **Promotion rule:** exact algebra can be called proved after checking its derivation; simulations remain measured and no claim of a newly established physical advantage is made.

## Evidence map and provenance

| Object | Status | Reason |
|---|---|---|
| Signature (2,2), Clifford-torus and Hopf-preimage identities | P (classical/direct derivation) | explicit coordinate transformations and norm identities |
| Null-cone stabilizer U(1)·O(2) | P (direct derivation) | column-phase proof |
| Both codebooks' four Stokes base points and unequal full-codebook distances | P for mathematical constructors; M for NumPy regression | explicit finite point sets |
| Earlier PR #82 SER improvement in the tested narrowband setup | M (prior simulation, not rerun here) | original notebook |
| Null-cone geometry as an independently causal explanation for SER advantage | R | resource-matched negative control |
| A genuinely improved physical constellation or a novel theorem | Not established | requires new preregistration and external evidence |

The *topological facts* are classical: see [Pinkall, Hopf Tori (1984)](https://archive.mpim-bonn.mpg.de/2356/1/preprint_1984_52.pdf) and [ETH Differential Geometry exercises on Hopf fibrations and Clifford tori (2023)](https://metaphor.ethz.ch/x/2023/hs/401-3531-00L/ex/ex7.pdf).

Human-supplied framing: geometric/non-scalar research direction, causal skepticism and evidence governance. AI-assisted: this mathematical reconstruction, test implementation and repository documentation. These roles are not evidence of independent human derivation or third-party scientific validation.
