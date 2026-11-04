# Almost losing a polarization: the exact decoder boundary layer

**2026-11-04 | bounded follow-up to the collision-discriminant checkpoint**

[Parent research](./DISTANCE_SPECTRUM_COLLISION_DISCRIMINANT.md) · [NVIDIA Sionna RT PR #82](https://github.com/NVlabs/sionna-rt/pull/82) · **No upstream PR or GPU run**

The source-matched 16-point rank-one and QPSK² constellations are compared in a deliberately simple, **one-row coherent complex-AWGN channel**. Their original ray-tracing SER results remain **M**; their unsupported *null-cone causal* interpretation remains **R**. This note does not establish 6G physical performance or new mathematical novelty.

## A human explanation

A two-polarization signal carries information in two complex components. Consider a receiver that sees the first component clearly and the second only faintly. When the second component becomes weaker *at the same rate as the noise*, the decoding problem does not vanish into an on/off singularity. It becomes a small, exactly solvable set of Gaussian discrimination problems **inside the groups of symbols that share the same first component**.

That is the operative distinction: first choose the correct surviving *group*, then recover which symbol within that group was sent. In the singular limit the group is known almost perfectly; distinguishing its members remains noisy.

## Object, assumptions and limit

Let `C={x_i=(a_i,b_i)}_{i=1}^{16} subset C²` be one of the frozen **unit-energy, distinct-symbol** codebooks from the notebook, transmitted equiprobably. The measured scalar channel and proper noise are

\[
y=a_i+\varepsilon b_i+n,\qquad
n\sim\mathcal{CN}(0,N_0),\qquad N_0>0.
\]

Define `G_g={i:a_i=a_g}`, the *exact* first-coordinate collision classes, and set

\[
\varepsilon=\sqrt{\rho N_0}\,e^{i\phi},\qquad
\rho\ge0,\quad \phi\in\mathbb R.
\]

The **double-scaling limit** means `N_0 downarrow0` with `rho` and `phi` fixed, **not** lowering `N_0` at fixed nonzero `eps`.

If the receiver is told the true class `G_g`, subtract `a_g` and rescale the observation by `1/sqrt(N0)`. The exact class-conditional channel, for *every* positive `N0`, is

\[
z=\sqrt{\rho}\,e^{i\phi}b_i+Z,\qquad Z\sim\mathcal{CN}(0,1).
\]

Hence its optimal coherent-ML error probability depends on the amplitude-to-noise ratio `rho` but **not on `N0` or `phi`**, because circular complex Gaussian noise is rotation invariant.

## Theorem: exact error functions and finite-noise lower bounds [P, conditional]

Let `Q(t)=1/2 erfc(t/sqrt2)`.

The first-coordinate fibers of the rank-one constellation have sizes

\[
(1,1,1,1,\,2,2,2,2,\,4),
\]

while the QPSK² codebook has four classes of size four.

For rank-one, the four 2-symbol classes each contain two antipodal second-component symbols of distance `sqrt2`; their error probability is `q=Q(sqrt rho)`. The single 4-symbol class has four second-component points forming QPSK of **unit** complex energy; its error is `2q-q²`. Singleton classes contribute no error. Weighted by 8/16 and 4/16, respectively, we obtain

\[
\boxed{F_R(\rho)=Q(\sqrt\rho)-\tfrac14 Q(\sqrt\rho)^2.}
\]

For QPSK², every first-component class contains a conventional four-point QPSK with energy `1/2` in the second component. Each of its independent real decisions has error `p=Q(sqrt(rho/2))`. Therefore

\[
\boxed{F_Q(\rho)=2Q(\sqrt{\rho/2})-Q(\sqrt{\rho/2})^2.}
\]

These are *exact class-informed decoder risks at every positive `N0`* when `rho=|eps|²/N0`; they are not merely fits to the numerical sweeps.

Let `P_R(N0,eps)` and `P_Q(N0,eps)` be the optimal full-codebook SER with **no class information**. More information cannot increase Bayes error, giving the **finite-noise lower bounds**

\[
P_R(N_0,\varepsilon)\ge F_R(\rho),\qquad
P_Q(N_0,\varepsilon)\ge F_Q(\rho).
\]

In particular, the named `F` functions are not universal upper bounds on physical decoder error.

### Why the full-decoder risk tends to the oracle risk

Let `delta=min_{g not=h}|a_g-a_h|>0` be the separation of distinct first-coordinate class centers, `B=max_i |b_i|`, and `M=16`. Any cross-class received pair has distance at least

\[
d_{\mathrm{cross}}\ge \delta-2B|\varepsilon|.
\]

For `delta>2B|eps|`, a standard Gaussian pairwise-error union bound implies

\[
0\le P_{\mathrm{SER}}(N_0,\varepsilon)-F_C(\rho)
\le (M-1)Q\!\left(
\frac{\delta-2B\sqrt{\rho N_0}}{\sqrt{2N_0}}
\right).
\]

A full decoder can only make an *additional* mistake relative to the class-informed decoder when some candidate from another group beats the true symbol. The inequality union-bounds precisely those cross-group comparisons. The right-hand side tends exponentially to zero for fixed `rho` as `N0 downarrow0`; therefore

\[
\boxed{\lim_{\substack{N_0\downarrow0\\|\varepsilon|^2/N_0=\rho}}
P_{\mathrm{SER}}(N_0,\varepsilon)=F_C(\rho).}
\]

This proof requires finite distinct first-coordinate groups, equal priors, proper circular complex Gaussian noise, a fixed known coherent channel, and ML decoding. It does **not** apply without modification to correlated noise, unknown CSI, a varying source codebook or an arbitrary physical ray-tracing ensemble.

For the rank-one codebook `delta=1-1/sqrt2≈0.292893` and `B=1`. The bound is intentionally conservative; its positivity certificate is a **sufficient** condition, not a necessary condition for convergence.

### Exact comparison within this narrow model

With `q=Q(sqrt rho)≤p=Q(sqrt(rho/2))≤1/2`, we have

\[
F_R=q-\tfrac14q^2<q\le p\le 2p-p^2=F_Q
\]

for all **finite** `rho≥0`. This proves a strict ordering of these **double-scaling boundary-layer risks**, not a uniform ordering of the actual finite-noise SER across every channel geometry.

| `rho` | Rank-one `F_R` | QPSK² `F_Q` |
|---:|---:|---:|
| 0 | 0.437500 | 0.750000 |
| 0.25 | 0.284739 | 0.592748 |
| 1 | 0.152362 | 0.422020 |
| 2 | 0.077103 | 0.292139 |
| 4 | 0.022621 | 0.151113 |
| 9 | 0.001349 | 0.033608 |
| 16 | 0.000032 | 0.004672 |

Both functions tend to zero as `rho→infinity`, consistent with resolving all transmitted symbols whenever the weak-component signal is strong relative to the noise.

## A second, *exact* collision direction: counterexample to global superiority [P, exact algebra]

At `epsilon=sqrt2-1` (a **fixed nonzero** real relative channel gain), the scalar channel `h=(1,epsilon)` produces:

- rank-one: **8** distinct output points, each hit by two transmitted symbols, hence zero-noise SER floor **1/2**;
- QPSK²: **16** distinct output points, hence zero-noise SER floor **0**.

The separate SymPy certificate constructs the two codebooks in `Q(sqrt2,i)` and verifies these coincidence classes using *exact* algebraic equality, not a numerical clustering tolerance. This is a different order of limits from the boundary-layer theorem. It prevents treating small-noise asymptotics near the `epsilon=0` direction as performance predictions for arbitrary nearby channels at finite noise.

An exploratory Monte Carlo check at `rho=9, N0=0.01` gives `epsilon=0.3`, close enough to this secondary collision direction that the full rank-one error can be more than 0.1 **above** its theoretical oracle limit. This is a counterworld chosen after exploration, not preregistered performance evidence.

The original `det X=0` hypothesis is **not rescued** here. Rotating the second component of every rank-one symbol by `i` gives an *off-null-cone* codebook with the same within-class distances and thus the same `F_R` function. The fixed determinant constraint is **not necessary** for the boundary-layer risk.

## Executable verification and resource discipline

Files added on this branch:

- `research/tests/test_boundary_layer_scaling.py` — **10 passing NumPy unittest methods**. Covers the original constructors, group sizes, exact limits at `rho=0`, independent Gaussian class-oracle simulation, full-coherent-ML convergence, phase rotation, the finite-noise lower bound and separation certificate, the off-cone twin, an exact second collision ray (numerical corroboration), and adversarial finite-noise failure.
- `research/verify_boundary_layer_exact.py` — **passing independent SymPy algebraic certificate**. Establishes the exact first-coordinate class sizes, second-component BPSK/QPSK geometries, and the secondary direction `epsilon=sqrt2-1`.

Run from repository root, with ordinary CPU and NumPy/SymPy:

```bash
python -m unittest discover -s research/tests -p 'test_boundary_layer_scaling.py' -v
python research/verify_boundary_layer_exact.py
```

**Verification (2026-11-04):** NumPy **10/10** passed; SymPy algebraic certificate passed. The exact Git blob SHA of each committed source matches the independently executed local file. The previous 40 regression tests and original Sionna RT notebook were **not** re-executed in this turn.

**Compute cost:** local CPU only; **no Hugging Face jobs, Spaces, or ZeroGPU minutes used**. The connected Hugging Face account is available, but its Space-search endpoint currently reports unavailable, and its connector does not expose Space creation. For later *heavier* CPU research, the official Hugging Face [Spaces overview](https://huggingface.co/docs/hub/spaces-overview) documents a CPU Basic tier with 2 vCPUs / 16 GB RAM and no hourly cost, subject to plan eligibility and usage rules. No Space is needed for this experiment.

## Evidence, remaining obstruction and next experiment

| Claim | State | Limitation |
|---|---|---|
| Class profile, formulas for `F_R` and `F_Q` | **P** | Exact algebra, equal priors, coherent complex-AWGN model |
| Finite-noise class-oracle lower bound and asymptotic convergence | **P** | Channel `h=(1,epsilon)`, fixed `rho` and correct noise scaling |
| Exact secondary collision ray at `epsilon=sqrt2-1` | **P** | 16-point frozen source constructors, SymPy witness |
| New NumPy simulated outcomes | **M** | CPU Monte Carlo, same authored modelling assumptions |
| Generalized radio/6G improvement | **H / not demonstrated** | No source-matched Sionna evaluation, no held-out real channel data |
| Null-cone determinant is the unique cause of performance gain | **R** | Earlier matched optimizer control and off-cone twin |

**Preregister next gate, not executed:** replace the perfectly aligned one-row channel with a *held-out small family of non-diagonal complex channels* and predeclare the channel distribution, input-energy constraints, baseline optimization budget, matched unconstrained 16-point codebook, confidence interval procedure, and primary metric (complete SER versus SNR, stratified by distance-spectrum and collision-neighborhood statistics). **Kill** any claimed special geometric advantage matched by an equally optimized unconstrained codebook. **Stop** if the proposed new geometry does not improve prediction beyond standard channel Gram-matrix features. No GPU or NVIDIA upstream PR until that discriminator is specified.

**Human/AI provenance:** geometric direction and explicit evidence discipline are human-led; this mathematical construction, code, derivations, and documentation are AI-assisted. Neither two tests by the same modelling workflow nor an exact SymPy calculation amount to independent scientific peer review.
