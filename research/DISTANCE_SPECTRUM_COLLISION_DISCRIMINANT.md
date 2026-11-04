# When a radio channel erases a distinction

**NVIDIA Sionna RT / PR #82 — collision discriminant and complete distance spectrum**  
**Research checkpoint:** 4 November 2026 · CPU-only · fork research branch · no new NVIDIA PR.

This note continues [the channel-metric orbit audit](./CHANNEL_METRIC_ORBITS.md), which itself extends the [Stokes/Hopf reconstruction](./STOKES_HOPF_TORUS.md). The underlying [PR #82](https://github.com/NVlabs/sionna-rt/pull/82) remains an empirical constellation-shaping comparison. Its earlier claim that the realified determinant `det X=0` *causes* an SER improvement was retired after matched unconstrained-optimizer controls. **Nothing here reopens that claim.**

## For a busy human

Imagine 16 letters travelling through a two-lane radio channel. A rank-one channel preserves only one combination of the lanes. Some distinct letters may become indistinguishable. The number of surviving different letters, not just the distance between the two closest ones, determines the error rate when the noise becomes vanishingly small.

The choice of *which combination survives* matters. Two equally rank-deficient channels can erase different numbers of letters, and most one-dimensional complex projections actually keep all 16 distinct.

The mathematics makes these distinctions explicit and inexpensive to test.

## 1. Typed mathematical objects

Fix a finite set `C={x_1,...,x_M} ⊂ C²` of **distinct transmitted Jones vectors** with equal prior probability, and a fixed coherent channel `H ∈ C^{r×2}`. The receiver observes

\[
y=Hx+n,\qquad n\sim\mathcal{CN}(0,N_0 I_r),\quad N_0>0.
\]

For each pair of transmitted symbols define the received squared distance

\[
d_{ij}^2(H)=\|H(x_i-x_j)\|_2^2
           =(x_i-x_j)^\dagger (H^\dagger H)(x_i-x_j).
\]

The two fixed PR #82 codebooks have `M=16`, unit energy per symbol, and four complete global quarter-phase orbits each:

\[
C=\bigsqcup_{k=1}^4\{u_k,\,iu_k,\,-u_k,\,-iu_k\}.
\]

The original rank-one constructor and the QPSK² constructor are separately reproduced in the new CPU tests. This is a *frozen source-code comparison*, not a general result about all rank-one or unconstrained 16-point constellations.

## 2. An exact distance-spectrum identity [P — elementary algebra]

Write `G=H†H`, `e_k=u_k†Gu_k`, `z_{k\ell}=u_k†Gu_\ell`. The complete **unordered** received-distance multiset (all 120 pairs, not merely its minimum) is the union of:

- *Within one quarter-phase orbit* `k`: four copies of `2e_k` and two of `4e_k`.
- *Between distinct orbits* `k<\ell`: for each `r∈{0,1,2,3}`, four copies of

\[
e_k+e_\ell-2\Re(i^r z_{k\ell}).
\]

There are `6×4+16×6=120` distances. **Proof:** factor out the four possible differences of quarter-phase exponents and expand the Hermitian norm. The identity holds for any complex `H`, including rank-deficient channels; a numerical Gram implementation can produce tiny signed roundoff near exact zero and should be compared to direct distances within tolerance.

The implementation checks all 120 distances against an independently computed brute-force `16×16` table for the zero channel, rank-one channels, diagonal channels and random complex rectangular channels. It also tests compensated changes of input frame. This is a meaningful alternative to assuming that one shortest distance captures the receiver's entire geometry.

### From distances to Gaussian pairwise error

If all received candidates are distinct, an individual wrong-codeword half-space has probability

\[
P(i\to j)=Q\!\left(\sqrt{\frac{d_{ij}^2}{2N_0}}\right),
\quad Q(t)=\frac12\operatorname{erfc}\!\left(\frac{t}{\sqrt2}\right).
\]

Let `d_{i,\min}` be the closest *received* neighbor of `x_i`. Then coherent-ML symbol error satisfies

\[
\frac1M\sum_i Q\!\left(\frac{d_{i,\min}}{\sqrt{2N_0}}\right)
\;\leq\; P_{\mathrm{SER}}\;\leq\;
\frac{2}{M}\sum_{i<j}Q\!\left(\frac{d_{ij}}{\sqrt{2N_0}}\right).
\]

The upper expression is an **unclipped union bound** and may exceed one. The test code preserves that raw value. These bounds are **not** silently extended to coincident received symbols, where `d_{ij}=0` creates an ML tie and a naïve `Q(0)` pairwise event can be misleading.

## 3. The collision discriminant in CP¹ [P — finite geometric argument]

For a **rank-one** channel, the effective received scalar has the form

\[
z=h_1x_1+h_2x_2,\qquad
[h_1:h_2]\in\mathbb{CP}^1.
\]

Two distinct symbols `x_i,x_j` collide exactly when

\[
h_1\Delta x_1+h_2\Delta x_2=0,\qquad
\Delta x=x_i-x_j\ne0.
\]

Each unordered pair contributes exactly one projective direction `[\Delta x_2:-\Delta x_1]`. The constellation's **collision discriminant** is the finite union of those directions. Outside it, the projection is injective on the whole codebook, despite the channel having matrix rank one. This is elementary projective linear algebra, *not a novelty claim about algebraic geometry*.

### Exhaustive exact classification for the two frozen 16-point codebooks

The independent SymPy certificate constructs every point over `Q(\sqrt2,i)`, groups all 120 pair differences by exact vanishing of a 2×2 determinant, and checks each projected coincidence class.

| Number of distinct output points | Number of rank-one directions | Number of QPSK² directions | Zero-noise SER floor |
|---|---:|---:|---:|
| 12 | 8 | 8 | 1/4 |
| 9 | 4 | 4 | 7/16 |
| 8 | 6 | 0 | 1/2 |
| 4 | 0 | 2 | 3/4 |
| **Exceptional projective directions** | **18** | **14** | — |

Every one of the 120 unordered symbol pairs is assigned to exactly one direction. The exact certificate also checks that the collision-pair count equals `Σ_j binom(m_j,2)` for the corresponding output-class sizes `m_j`. This prevents a convenient numerical clustering tolerance from determining the mathematical result.

### Why an exact symbol-error floor follows [P — conditional theorem]

Suppose a rank-one channel produces `R` distinct outputs from `M` equiprobable symbols. Each output class of size `m` induces the *same conditional noise distribution* for its `m` inputs. Any decoder has probability of correct decoding at most `1/m` conditioned on that class. With zero noise, distinct output classes are themselves distinguishable. Therefore

\[
\boxed{\lim_{N_0\downarrow0}P_{\mathrm{SER}}
=1-\frac{R}{M}.}
\]

The lower bound `P_{\mathrm{SER}}\ge 1-R/M` holds for **every** `N_0>0` under the same equal-prior, channel and noise assumptions. The limit is attained by optimal coherent decoding.

Examples for row directions `h`:

| Direction | Rank-one `R` | QPSK² `R` |
|---|---:|---:|
| `(1,0)` — discard second polarization | 9 | 4 |
| `(1,1)` — equal-sum projection | 9 | 9 |
| `(1,2+i)` — generic example | 16 | 16 |

Thus the `(1,0)` projection has **43.75% versus 75%** irreducible error, respectively, in the vanishing-noise limit. A *different* rank-one channel may have **zero** such floor for both. The rank of `H` and the singular values alone do not determine the coincidence classes.

The *off-null-cone twin* `D C_{\rm rank1}` with `D=diag(1,i)` has the same first component, so it retains the same `(1,0)` collision classes and floor while violating the original fixed null-cone condition. **The determinant constraint is not necessary for this effect.**

## 4. Singular limits do not commute [P — under the stated Gaussian model]

Let `H_\eta=diag(1,\sqrt\eta)`. For every fixed `\eta>0`, this map is injective on both codebooks, so SER tends to zero as noise vanishes. At `\eta=0` it has the collision floors above. Consequently,

\[
\lim_{\eta\downarrow0}\lim_{N_0\downarrow0}P_{\mathrm{SER}}=0,
\qquad
\lim_{N_0\downarrow0}\lim_{\eta\downarrow0}P_{\mathrm{SER}}
=1-\frac{R(H_0,C)}{16}.
\]

The order of 'weakening the polarization' and 'removing the noise' changes the answer. This is the mathematical reason a low-but-nonzero polarization component can recover distinctions only at sufficiently low noise. It is *not* evidence of a discontinuity in a real radio's behavior at fixed positive noise.

## 5. The complete spectrum matters at finite noise [P/M boundaries]

For **QPSK² only**, the diagonal channel has four independent BPSK sign decisions (the real and imaginary parts of both components). There is an exact coherent-ML SER formula:

\[
P_{\mathrm{SER,QPSK}^2}
=1-\left[1-Q\!\left(\frac{1}{\sqrt{2N_0}}\right)\right]^2
     \left[1-Q\!\left(\sqrt{\frac{\eta}{2N_0}}\right)\right]^2.
\]

For the rank-one constellation, this note does **not** claim a comparable closed-form full SER. It instead retains the pairwise bounds and a directly simulated finite-sample coherent-ML detector.

**Exploratory, post-selection example:** `\eta=0.2` and `N_0=0.02`, 200,000 equiprobable symbols, complex circular Gaussian noise, fixed perfect-CSI channel and exact ML nearest-codeword decoder.

| Quantity | Rank-one | QPSK² |
|---|---:|---:|
| Minimum squared received distance | 0.185786 | **0.200000** |
| SER | 0.015985 (seed 20261104) | 0.0251872559 (**exact model formula**) |
| Same-model fresh-seed check | 0.016265 (seed 8851) | Same exact formula |

The approximate normal 95% binomial sampling interval for the first rank-one run is 0.01544–0.01653. The *smaller-minimum-distance codebook* nevertheless has the *lower observed finite-noise SER*, because the rest of the distance spectrum matters. This setting was selected after a small exploratory parameter scan. The fresh seed is an **internal model replication**, not independent physics or a preregistered real-world validation.

## 6. How to reproduce without GPU resources

From the fork's research branch:

```bash
# NumPy required; 14 new CPU-only unit tests
python -m unittest discover -s research/tests -p 'test_distance_spectrum_discriminant.py' -v

# Optional SymPy dependency; exact arithmetic, no Monte Carlo
python research/verify_collision_discriminant_exact.py
```

**Executed in the isolated local container (2026-11-04):** 14/14 new NumPy tests pass; exact SymPy projective-collision certificate passes for both codebooks. The Git blob SHA-1 of each of these two source files was compared to the tested file and matches. The **26 earlier branch tests were not rerun in this local container**, nor were NVIDIA's Sionna RT simulations or its upstream CI.

### Epistemic ledger

| Claim | Class | What the artifact establishes |
|---|---|---|
| Complete `μ₄` distance enumerator | **P** | finite orbit-pair identity; brute-force numerical cross-check |
| Collision discriminant finite in CP¹ | **P** | one projective line condition per symbol pair |
| 18 versus 14 exceptional directions and their strata | **P / exact symbolic certificate** | exhaustive finite algebraic computation on these frozen codebooks |
| SER floor `1-R/16` | **P** | equal-prior indistinguishability argument |
| Exact diagonal QPSK² SER | **P** | factorized Gaussian sign decisions |
| Rank-one finite-noise numbers | **M** | seeded CPU Monte Carlo in a narrow coherent-AWGN model |
| General ray-traced 6G performance advantage | **Not established** | no new Sionna runs or source-matched held-out channels |
| The fixed null-cone determinant uniquely *causes* SER improvement | **R (retired)** | previous matched controls and an off-cone isometric counterexample |

### Preregistered next test — *not executed*

**Object:** a one-row channel `h_\epsilon=(1,\epsilon)` approaching an exceptional axis, for two original codebooks and their phase-rotated control. **Prediction:** within-axis collision classes split at order `|\epsilon|` in amplitude, so near-axis decision behavior depends on a boundary-layer ratio such as `|\epsilon|^2/N_0`; directions away from the finite discriminant remain distinguishable. **Primary measurement:** class-conditioned SER, not only a scalar minimum distance. **Controls:** uniform priors/energy, exact QPSK formula where applicable, independent direct distance enumeration, identical noise streams, and a matched unconstrained 16-point codebook if the claim is escalated to physical superiority. **Kill:** no reproducible boundary-layer collapse or no benefit beyond matched controls. **Stop:** no GPU simulation or upstream submission without new preregistration tied to a real Sionna channel.

**Human/AI provenance:** the conceptual geometric programme and constraints come from the human researcher; definitions, finite deductions, test implementations and explanatory writing in this checkpoint are AI-assisted. Agreement between two AI-authored code paths or exact arithmetic checks is not independent human peer review.
