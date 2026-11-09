# Frozen uncertainty-certificate audit (9 November 2026)

Parent: \`research/receiver-partial-identification-2026-11-09\` @ \`087c579a58dd4693106c2316f276b9c2d54db719\`. CPU only; only the one source-printed rounded H, the previously fixed 75 synthetic Gram cells and the two known mathematical counterworlds. No new channel data, novel 6G mechanism, geometric lifts, tuning, upstream PR activity or Work.

## Living question
What precision of a **full, frame-aligned, noise-weighted Gram** \(G_v=H_v^\dagger\Sigma_v^{-1}H_v\) would suffice to retain an already certified constellation comparison? The two prior counterworlds show missing 15 receivers permit opposite results. This experiment asks a narrower, practically measurable *what-if*: if future slot-Grams were supplied only within bounded spectral-norm error, would the previous aggregate sign be stable?

## Model / theorem target (known classical inequalities, not an originality claim)
For each fixed, non-adaptive 16-symbol codebook \(C\), Hermitian positive semidefinite true \(G_v\) and estimate \(\widehat G_v\) with \(\|G_v-\widehat G_v\|_{\mathrm{op}}\le\varepsilon_v\), let \(\Delta_{ij}=x_i-x_j\), \(q_{ij}=\Delta^\dagger G_v\Delta\), \(\widehat q_{ij}=\Delta^\dagger\widehat G_v\Delta\), \(\eta_{ij}=\varepsilon_v\|\Delta\|^2\). Then \(q_{ij}\in[\max(0,\widehat q_{ij}-\eta_{ij}),\widehat q_{ij}+\eta_{ij}]\). Since \(f(q)=Q(\sqrt{q/(2N_0)})\) is decreasing, put
\[
p^-_{ij}=f(\widehat q_{ij}+\eta_{ij}),\qquad
p^+_{ij}=f(\max(0,\widehat q_{ij}-\eta_{ij})).
\]
Define \(\underline L_{C,v}=M^{-1}\sum_i\max_{j\ne i}p^-_{ij}\), \(\overline U_{C,v}=M^{-1}\sum_i\sum_{j\ne i}p^+_{ij}\). For every admissible true Gram and standard coherent ML (perfect receiver CSI, \(N_0=0.01\)), \(\underline L_{C,v}\le\mathrm{SER}_{C,v}\le\overline U_{C,v}\). With uniformly sampled receivers, \(\overline U_A<\underline L_B\) is a **sufficient, conservative certificate** that A wins throughout the whole declared Gram uncertainty product, not a necessary condition.

**Moving moving frame requirement**: input-unitary \(U\) maps \((\widehat G,\Delta,G)\mapsto(U\widehat G U^\dagger,U\Delta,UGU^\dagger)\), conserving \(\varepsilon\), distances and robust bounds. A nonunitary receiver coordinate map \(W\) is permitted only with \(\Sigma\mapsto W\Sigma W^\dagger\), so the effective Gram transforms as stated. Do not flatten to a ratio, assume untransported physical port projectors or reoptimize CSI codebooks.

## Source inputs, exact test plan and independent controls (frozen before execution)
1. Use the three **fixed** original 16-symbol codebooks rank-one, QPSK², circular 16-PSK; no channel-reoptimized beam references. The one source-printed H0 is unchanged; the two counterworlds A/B retain 15 full-rank, trace-1 missing-channel slots. The 75 grid and codebooks are not refit.
2. For H0 decimal printing, assume nearest rounding in eight real components, \(\|E\|_F\le\sqrt8\cdot 5\times10^{-9}\). Bound its Gram error by \(2\|\widehat H_0\|_F\|E\|_F+\|E\|_F^2\). Apply that radius to site 0; apply a common variable operator-norm radius \(\varepsilon\) to the 15 missing counterworld slots.
3. **Primary quantitative output**: for each world × two comparisons, bisect the largest \(\varepsilon\in[0,0.1]\) retaining strict robust interval separation, to numerical bisection resolution \(10^{-9}\); report the minimum of the four radii. Do not choose a new radius after seeing results. A numerical radius is only a *sufficient uncertainty budget for these synthetic worlds*, not a claim about future Munich measurements.
4. **Kill gate**: if the \(\varepsilon=0\) rules do not reproduce all four baseline strict signs, or any radius is \(\le 10^{-6}\), classify the uncertainty method as impractically conservative here; preserve the failure and do not change worlds, design, bracket or threshold. If radii are positive, report without engineering deployment claim.
5. **Internal validity controls**: on all 75 frozen grid cells at \(\varepsilon=0\), reproduce rank-one-vs-QPSK²/circular certified signs; at fixed \(\varepsilon\in\{10^{-4},10^{-3}\}\), record how many prior strict decisions remain certified and abstain rather than force a sign. At least 100 seeded random Hermitian PSD perturbations of frozen cells (seed 20261109) must satisfy the robust pointwise Gaussian and aggregate envelope inequalities; positive-definite and near-collision edge cases included.
6. **Separate, independent numerical witness**: use explicit direct-H channel square roots / Cholesky whitened complex-noise distances on selected cells, not merely a second copy of the same Gram function. Verify two sequential transmitter SU(2) frame rotations and independent site-local invertible receive transformations: *all numerical envelopes and four aggregate decisions unchanged to 1e-10* with full covariance/physical port transport.
7. **False-frame control**: deliberately rotate the nominal Gram without codewords; require at least one altered pair-distance/envelope, report as a deliberately different physical problem (not a new observation). Invalid Hermitian/negative spectral-radius/mismatched shape must fail closed.
8. There is **no new fitting, no grid choice after results, no GPU, HF/ZeroGPU, Sionna CPU regeneration, Work, upstream PR or comments**. Stop after a concise addendum to the existing technical note, code/tests/results and clean CPU CI if available.

## Evidence promotion
P: the conditional PSD spectral norm inequality and monotonic Gaussian bounds, including composition in moving frames.
M: computed sufficient radii and finite metamor­phic controls on 75 synthetic cells and two synthetic counterworlds, with explicit seeds and thresholds.
A: bounds on the not-yet-measured effective-Gram errors are assumptions, NOT observed calibration.
R: any suggestion that these synthetic radii establish physically achievable 16-Munich-channel measurements or performance superiority.
Novelty: no check or claim beyond standard matrix perturbation and coherent detection.

If the new code or CI fails the primary gate, state it plainly; no retuning or substitute promotion.
