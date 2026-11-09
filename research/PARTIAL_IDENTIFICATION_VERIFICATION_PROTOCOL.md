# Verification protocol — receiver-ensemble partial identification (2026-11-09)

## Discovery disclosure
Two explicit counterworlds were **selected after exploratory inspection** of the already-frozen 75-cell synthetic Gram grid. This is NOT a prospectively preregistered discovery. The formulas, controls, numerical tolerances and stop condition below are frozen **before the new independent direct-H verification and CI tests**.

## Question
Does the one 8-decimal-place printed PR #82 channel, the 16-slot receiver count, and individual Frobenius normalization logically determine the **sign** of the ensemble-average conditional SER ordering between the fixed rank-one constellation and BOTH fixed QPSK² and off-cone circular 16-PSK?

## Exact domain and quantifiers
Let H0 be the printed complex 2×2 matrix in PR #82 (not the other 15 H). An admissible abstract completion is an ordered tuple (H0,H1,...,H15), where each unobserved Hv for v≥1 is finite, full-rank complex 2×2, with ||Hv||_F=1. Receiver index is sampled uniformly; noise is independent proper CN(0,0.01 I_2) and receiver CSI is perfect. All designs use the same 16-symbol unit-energy fixed source codebooks and physical per-port average power.

**Important omission:** this domain DOES NOT enforce the physical Munich scene geometry, Maxwell propagation, raytracer solver equations, spatial smoothness, path depths or realistic correlation between receiver slots. It is a mathematical partial-identification result under the **published H0 + basic per-slot constraints only**, NOT a proof that either completion is realizable by the actual fixed scene.

## Frozen witnesses
- World A: each missing receiver has source-frame Gram G_A=(1/1.1) diag(1,0.1), ratio r=0.1 and principal theta=phi=0.
- World B: each missing receiver has source-frame Gram G_B=(r I + (1-r)u u†)/(1+r), r=0.01, u=(cos(pi/8),sin(pi/8)), phi=0.
- Each Hv is the positive-definite Hermitian matrix sqrt(G_v), with source transmit port basis fixed, no simulated propagation.
- Both worlds contain exactly the SAME printed H0, 15 alive/normalized missing receivers, the same port-power allocation and the same noise SNR.

## Observable, gate and numerical reference
For any 16-symbol code C at site v compute p_ij=Q(||Hv(x_i-x_j)||/sqrt(2 N0)).
L_v(C)=(1/16) sum_i max_{j != i} p_ij and U_v(C)=(1/16) sum_i sum_{j != i} p_ij.
Then average across sites: mean_L(C)=(1/16)sum_v L_v(C), mean_U(C)=(1/16)sum_v U_v(C).
A wins B only if mean_U(A) < mean_L(B); else if mean_U(B) < mean_L(A), B wins; otherwise abstain.

The primary gate passes iff world A strictly certifies rank-one better than BOTH comparators, world B strictly certifies rank-one worse than BOTH, and every certified gap exceeds 1e-4 (far beyond float error and H0 printed rounding). Reject, rather than force, any contrary outcome.

Independent controls: reconstruct the 16 receiver H matrices using spectral square roots, verify H†H=G and positive rank; compare direct-H distances against Gram-quadratic distances without using the source codebook/Gram implementation; test source-printed H bounds against 80-digit historical results within 1e-10; run two successive nontrivial transmitter unitaries and receiver-local invertible nonunitary frame maps with fully transported covariances and physical transmit-port projectors. All conditional intervals and strict decisions must survive each stage within 1e-10.

## Rounding proof obligation
Assume each real/imag printed H0 entry is rounded to nearest 1e-8. Each Gaussian PEP is perturbed by ≤ epsilon=2 sqrt(8)*5e-9 / sqrt(4 pi N0). At H0 a lower bound varies by ≤epsilon and upper by ≤15epsilon. As H0 carries 1/16 ensemble weight, each **aggregate strict comparison gap** loses at most epsilon, not 16epsilon. Require every gap >epsilon before describing the two conclusions as robust to printed precision.

## Stop and evidence
One execution, one narrow note addendum, regressions. If one witness fails, retain the failure and do not replace it with a selected replacement under the same 'frozen' protocol. 'P' applies to the conditional aggregation and gauge algebra; 'M' to finite computed witness inequalities; 'H' to physical realizability of those synthetic Gram completions; 'R' to aggregate ranking claims supposedly established by the one printed H. Nothing is learned from the inaccessible 15 Munich channel realizations.

**No Work, GPU, HF/ZeroGPU credits, no Sionna regeneration, no comments to NVlabs, PR #82 untouched.** This protocol and the ensuing code do NOT assert mathematical novelty: the statistical logic is classical partial identification with missing observations.
