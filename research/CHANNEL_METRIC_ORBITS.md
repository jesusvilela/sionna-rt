# Channel-weighted phase orbits: exact packing crossover and a non-null-cone twin

**Checkpoint:** 2026-11-04, research branch only. **Scope:** reproduce the exact 16-symbol codebooks in [Sionna RT PR #82](https://github.com/NVlabs/sionna-rt/pull/82) under a **fixed, memoryless, narrowband, coherent complex channel**. No 6G field claim, GPU ray tracing, or external NVIDIA PR. These results are an *analytic counterworld* for the causal interpretation already [retired by matched controls](https://github.com/NVlabs/sionna-rt/pull/82#issuecomment-6491527495).

## 1. Precise carrier and metric

Let `C ⊂ C²` be one of the two codebooks, with 16 **distinct** unit-energy vectors `x`. A fixed channel `H ∈ C^{r×2}` produces `y = Hx+n`. Define the Hermitian positive-semidefinite Gram matrix `G=H†H` and the **received squared pairwise distance**

\[
d_G^2(x,y)=(x-y)^\dagger G(x-y)=\|H(x-y)\|_2^2.
\]

The comparison is `d_{\min,G}^2(C)=\min_{x\ne y\in C} d_G^2(x,y)`. This is one *high-SNR-relevant distance diagnostic*, not by itself a theorem about entire symbol-error-rate curves.

Both published codebooks are unions of exactly four orbits of the global quarter-phase action `μ₄={1,i,-1,-i}`:

\[
C=\bigsqcup_{k=0}^{3}\{i^a u_k : a=0,1,2,3\}.
\]

Here `u_k` is one chosen representative in each orbit; the quotient is a mathematical device, not a new modulation scheme.

## 2. Exact orbit-quotient theorem [P: elementary algebra]

Put `e_k=u_k† G u_k ≥0` and `z_{k\ell}=u_k†G u_\ell`. Then

\[
\boxed{
d_{\min,G}^2(C)=\min\left[
\min_k 2e_k,\
\min_{k<\ell}\left(e_k+e_\ell
-2\max\{|\Re z_{k\ell}|,|\Im z_{k\ell}|\}\right)
\right].}
\]

**Proof.** Inside one orbit, `\|H(i^a u_k-i^b u_k)\|²=|i^a-i^b|²e_k`, and the minimum for `a≠b` is `2e_k`. For distinct orbits `k,\ell`, their cross-distance is

\[
e_k+e_\ell-2\Re(i^{b-a}z_{k\ell}).
\]

Over the four possible quarter turns, the maximum of the real cross term is exactly `max(|Re z|,|Im z|)`. Taking the minimum over all orbit pairs proves the formula. The theorem includes singular `G` and even zero `H`; it assumes the codebook itself has no duplicate symbol and is a complete union of the four-phase orbits.

**Semantic boundary:** `G` is the effective channel geometry. The formula does not rely on the realified determinant `q(x)=Im(conj(x₁)x₂)` and does not impute any independent role to the original null-cone constraint.

## 3. Closed-form anisotropic channel theorem [P: finite algebraic enumeration]

Choose a simple fixed channel with polarization *power* imbalance `η`:

\[
H_\eta=\mathrm{diag}(1,\sqrt{\eta}), \qquad 0\leq\eta\leq1.
\]

The original rank-one constructor can be written as `x=e^{\pm i\pi/4}(\cos(m\pi/4),\sin(m\pi/4))`, for `m=0,\ldots,7`. The QPSK² constructor is the independently normalized Cartesian product of ordinary QPSK with total energy one.

For the rank-one codebook, each pair yields a vector `(A,B)=(|\Delta x₁|²,|\Delta x₂|²)` and `d_{H_\eta}²=A+\eta B`. An exhaustive *finite* certificate gives these **12 distinct coordinate-distance pairs**, where `a=3/2-\sqrt2` and `b=3/2+\sqrt2`:

```text
(0,2)      (0,4)        (a,1/2)      (1/2,a)
(1/2,3/2)  (1/2,b)      (1,1)        (3/2,1/2)
(2,0)      (2,2)        (b,1/2)      (4,0)
```

To reconstruct the enumeration without trusting a simulator, use the four orbit representatives `u_m=e^{i\pi/4}v_m` with `v_m=(\cos(m\pi/4),\sin(m\pi/4))`, `m=0,1,2,3`, and the four relative multipliers `i^k`. If `k` is even, component distances are `(v_{m,j}∓v_{n,j})²`; if `k` is odd, they are `v_{m,j}²+v_{n,j}²`. Exclude the identical-symbol case, then collect the distinct pairs above.

Their nonnegative-coordinate Pareto frontier is exactly

\[
(0,2),\quad(a,\tfrac12),\quad(\tfrac12,a),\quad(2,0).
\]

For `0≤η≤1` the last two are never better than the first two respectively. QPSK² has scalar coordinate pair differences `0,1,2`, and its minimum received squared distance between distinct transmitted symbols (possibly zero when η=0) is `η`. Thus:

\[
\boxed{
d_{\min,H_\eta}^2(C_{\rm rank1})
=\min\{2\eta,a+\eta/2\},\qquad
d_{\min,H_\eta}^2(C_{\rm QPSK²})=\eta.}
\]

Both codebooks contain 16 symbols and have exactly unit energy per symbol in these constructors, so there is no hidden average-power rescaling in the comparison.

The inequalities give the **exact crossover**:

\[
d_{\min,H_\eta}^2(C_{\rm rank1})>
d_{\min,H_\eta}^2(C_{\rm QPSK²})
\quad\Longleftrightarrow\quad
0<\eta<2a=3-2\sqrt2\approx0.171572875.
\]

- `η=0.01`: rank-one `0.020000` vs QPSK² `0.010000` (squared distances).
- `η=0.10`: rank-one `0.135786` vs QPSK² `0.100000`.
- `η=1`: rank-one `2-\sqrt2≈0.585786` vs QPSK² `1`.

**Orientation counterexample (same singular values, opposite ranking).** The crossover is conditional on alignment of the codebooks with the diagonal channel axes; it is *not* a theorem determined by the singular-value ratio alone. At `eta=0.01`, replace the diagonal channel by `H' = H_eta R(pi/8)`, with `R` a real 2×2 rotation on the input coordinates. Both channels have singular values `(1,0.1)`, but direct finite enumeration gives:

| Channel at `eta=0.01` | Rank-one received `d_min²` | QPSK² received `d_min²` |
|---|---:|---:|
| `H_eta` | **0.020000** | 0.010000 |
| `H_eta R(pi/8)` | 0.005857864 | **0.154982143** |

This is a *physical channel-orientation change with the transmitted codebooks held fixed*, not a compensated frame change. It is an explicit falsifier of any orientation-free interpretation of the `3−2sqrt(2)` crossover. Its values are reproduced by the tenth CPU regression test. The equivalent **amplitude-gain threshold** in the aligned family is `sqrt(eta)<sqrt(2)-1`, not a universal condition-number law.

**Interpretation:** rank-one is better packed *for this metric* when one polarization is sufficiently weak; QPSK² wins for balanced channels. This does not establish that null-cone geometry is the reason or that an arbitrarily sampled ray-traced channel satisfies this criterion.

## 4. Decisive counterworld — an off-cone isometric twin [P: exact construction]

Let `D=diag(1,i)`, hold the *physical* diagonal channel `H_\eta` fixed, and actually change the transmitted codebook to `C_{\rm twin}=D C_{\rm rank1}`. Because `D` commutes with the diagonal channel,

\[
H_\eta D = D H_\eta,\qquad D^\dagger D=I.
\]

Consequently, for **every** pair `x,y`,

\[
\|H_\eta(Dx-Dy)\|^2
=\|D H_\eta(x-y)\|^2
=\|H_\eta(x-y)\|^2.
\]

And `q(Dx)=Im(conj(x₁) i x₂)=Re(conj(x₁)x₂)` is nonzero for many points originally satisfying `q(x)=0`. The transformed codebook is therefore **not contained in the original null cone**.

For complex circular white Gaussian noise, rotation `n↦Dn` preserves the noise distribution. The received candidate set is the unitary image of the original, so **ML symbol-error rates are identical in distribution at every SNR** for these two codebooks under the fixed diagonal channel, using the same codeword-index labeling. This is a physical symbol-only modification at fixed `H`, not the compensated coordinate transform `H↦H D^{-1}` studied in the earlier branch.

**Failure boundary:** for a general channel where `HD≠DH` and `D† H†HD≠H†H`, this conclusion need not hold. Isotropic receiver noise, equal-energy symbols, coherent ML and the fixed channel are all material assumptions.

The twin is a direct falsifier of any statement that the **original** realified condition `det X=0` is *necessary* for the performance of this design in this channel family. It does **not** show that all geometric structure is irrelevant, nor refute the measured PR #82 notebook.

## 5. Executable independent checks

`research/tests/test_channel_metric_orbits.py` contains 10 NumPy/unittest tests of:

- four-orbit partition of **both** actual PR #82 codebook constructors;
- orbit quotient against a separately written full 16×16 enumeration for zero, rank-deficient, diagonal and random rectangular complex channels;
- the diagonal formulas and crossover against brute force across a grid;
- coordinate-distance certificate and positive-weight Pareto filtering;
- reordering and representative invariance, and full `U(2)` compensated frame covariance;
- rejection of malformed/unclosed orbits;
- the off-null-cone twin and equal ML distance-cost vectors under rotated observations.

Command from repository root (NumPy installed):

```bash
python -m unittest discover -s research/tests -p 'test_*.py' -v
```

**Verification in this chat session:** the **10 new tests passed locally** using a local copy of the two codebook constructors from the GitHub source. The 16 earlier tests were *not rerun in the same local container*. NVIDIA CI, ray-traced Sionna tests and the original performance results were not rerun. These numerical checks corroborate, but do not replace, the exact finite derivations above.

## Evidence ledger, novelty boundary and stop rule

| Claim | Status | Evidence / restriction |
|---|---|---|
| Four-phase orbit-quotient identity for arbitrary PSD `G` | **P**, elementary algebra | direct orbit-pair expansion |
| Diagonal channel crossover at `3−2√2` | **P**, finite enumeration | complete distance-pair certificate |
| Non-null-cone twin with same received distances and Gaussian ML SER in diagonal channel | **P**, conditional analytical identity | `HD=DH` and noise rotational symmetry |
| 10 new CPU-only regression tests | **M**, local test | NumPy, seeds and checks in test file |
| PR #82 constellation performance in specific ray-traced ensembles | **M**, previous notebook | no rerun; limited to its setup |
| Original null-cone determinant is a uniquely causal source of benefit | **R**, retired | earlier matched unconstrained optimization and current off-cone control |
| Novel information-theoretic law, superior general-purpose modem, new mathematical theorem | **Not established** | do not claim without literature and physical-channel verification |

**Next discriminating gate (future, not executed):** freeze a small *non-diagonal* channel ensemble, compare original rank-one, QPSK², off-cone twin, and a matched unconstrained 16-point codebook with **identical power, CSI, labeling and optimization budget**. Prediction: the exact twin symmetry generally breaks when `[H†H,D]≠0`. Primary metric: received minimum squared distance plus complete SER vs SNR with confidence intervals and nearest-neighbor multiplicities. **Kill condition:** a matched unconstrained control reproduces any purported special-geometry gain, or the claimed symmetry fails in a channel that does satisfy the mathematical commutant condition. **Stop condition:** do not escalate to GPU ray tracing, higher algebras or an upstream PR without a source-matched physical effect.

**Authorship and prior-art boundary:** user supplied the conceptual geometric programme and evidence-governance direction; mathematical reconstruction, code and prose are AI-assisted. No literature novelty search or independent human proof review was performed; the mathematical identities are elementary and are *not claimed novel*. The current work is a controlled research note in a fork, not a request for NVIDIA to adopt a physical theory.
