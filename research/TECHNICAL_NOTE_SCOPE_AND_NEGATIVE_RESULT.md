# Sionna RT constellation audit: what survives

**Scope, stated first.** This note contains **one printed, rounded Munich receiver channel** from Sionna RT PR #82; **75 deterministic synthetic Gram-matrix grid cells**; **0 of 16 original Munich channel matrices recovered**; and **known** Pancharatnam/Bargmann geometric-phase mathematics. It is **not** a new ray-tracing study, a 6G performance claim, or a novel geometry theorem. CPU only; no GPU/HF credits. The NVIDIA repository and PR #82 were not modified.

## PR #83: an independently checkable software contribution

[PR #83](https://github.com/NVlabs/sionna-rt/pull/83), submitted from `jesusvilela/sionna-rt`, was **merged** into `NVlabs/sionna-rt:main` on **2026-11-04 16:18:46 UTC** (not merely closed). Its merge commit is [3b45582](https://github.com/NVlabs/sionna-rt/commit/3b45582f79827630af7e6a270fdad0fd822fe458), which is an ancestor of current `main`. The commit is attributed to maintainer `merlinND` and committed via GitHub `web-flow`; [the maintainer explicitly thanked the contributor](https://github.com/NVlabs/sionna-rt/pull/83#issuecomment-6635783872). It adds an actionable Mitsuba-variant validation check to `Scene.all_set()` and corresponding regression tests: **six files, +119 lines**. This is a substantive *software reliability* merge, not acceptance of the separate constellation hypothesis.

## The defensible constellation result: one source-printed channel

The printed complex matrix from [PR #82](https://github.com/NVlabs/sionna-rt/pull/82) is

```text
[[ 0.92982936+0.33584663j,  0.01006189+0.00676795j],
 [-0.06026739-0.00641686j, -0.13674566-0.01022593j]]
```

Its first input-polarization column contributes approximately **98.1% of total squared channel gain**. The notebook normalizes receiver H matrices individually and archives only this one matrix in its printed output.

For equal-prior, perfect-CSI coherent detection with circular white complex AWGN of variance `N0=0.01`, fix three 16-symbol, per-symbol unit-energy codebooks: original rank-one/null-cone, original QPSK-squared, and a simple fixed *off-null-cone* circular 16-PSK,

`x_k = exp(2*pi*i*k/16) * (1,i)/sqrt(2)`, for `k=0,...,15`.

All three average **half the transmit power on each port**; the circular book has `Im(conj(x1)*x2)=1/2`, not zero. This is a matched *average-port-power* control, not a claim of matched RF implementation or CSI overhead.

For any fixed codebook and channel, the Gaussian pairwise preference probability is `p_ij=Q(||H(x_i-x_j)||/sqrt(2*N0))`. Averaging each symbol's **largest** wrong-symbol pairwise event yields a lower bound `L` on its ML SER; averaging the **sum** of wrong-symbol events yields an upper bound `U`. The following intervals are **analytic probability bounds, not Monte Carlo confidence intervals**:

| 16-symbol design | Single printed H: [L,U] | Balanced H=I/sqrt(2): [L,U] |
|---|---:|---:|
| Circular 16-PSK (off cone) | [0.025844, 0.051824] | [0.025534, 0.051198] |
| Original rank-one | [0.068087, 0.106463] | [0.0000649, 0.0001298] |
| Original QPSK-squared | [0.165172, 0.414658] | [0.000000287, 0.000001147] |

The intervals certify **circular < rank-one < QPSK-squared** (smaller SER) on the printed channel, and the exact **reverse ranking** on the balanced example. The circular design does **not** have greater mean received energy on the printed H: about 0.49735, versus 0.50000 for rank-one. This falsifies *universal* fixed-null-cone superiority; it says **nothing** about the other 15 unrecovered Munich matrices. The circular baseline was selected **after** seeing the printed matrix, so it is an adversarial diagnostic, not a held-out field result.

**Independent numerical recheck (9 November 2026).** A separate 80-digit, direct-channel computation of all six Gaussian lower/upper values on the one printed H and balanced control agrees with the previous floating-point bounds to within **2.94×10⁻¹⁶**. The two printed strict gaps remain positive even after a worst-case **1.277×10⁻⁶** allowance for eight-decimal-place source rounding; the smaller remaining gap is approximately **0.0162626**. See the [frozen protocol and precise arithmetic audit](./PRINTED_PRECISION_AUDIT_2026-11-09.md). This is **not a formal directed-interval proof** or any additional receiver data.

## Gram decision rule; 75 synthetic cells

For each receiver Gram `G=H^H H`, define `d_ij^2=(x_i-x_j)^H G (x_i-x_j)` and compute bounds `L_C(G),U_C(G)` above. Declare A better than B **only if** `U_A < L_B`; declare B better **only if** `U_B < L_A`; otherwise abstain. The *full* Gram, including eigenvector orientation relative to the fixed transmit codebooks, matters; its eigenvalue ratio alone cannot decide the outcome.

The existing **75-case synthetic** sweep of five ratios, five principal angles and three phases (including repeated Gram matrices at axis/isotropy) yielded:

| Comparison to rank-one | Rank-one certified better | Comparator certified better | Abstain |
|---|---:|---:|---:|
| Fixed QPSK-squared | 18 | 46 | 11 |
| Fixed circular 16-PSK | 45 | 18 | 12 |

These are **grid counts**, not samples from radio propagation or a success rate. They can be reproduced from the checked-in `gram_decision_sweep.py`; the archived 75-cell JSON has SHA-256 `16ad7ca1eda73d9c6a73776555be2f93333d248ad44d11a32ac9be3b05d2f4c6`.

## What the missing 15 channels leave undecidable

**Conditional mathematical counterworld (9 November 2026).** The printed H0, 16-slot count and per-receiver Frobenius normalization **alone** do not identify the aggregate 16-receiver constellation ranking. This is a classical missing-data/partial-identification issue, not new geometric physics.

Hold H0, all three fixed 16-symbol codebooks, physical per-port power and coherent white noise at \(N_0=0.01\) unchanged. Two explicit sets of 15 *unmeasured* positive-definite channel completions yield opposite, strictly non-overlapping ensemble-average SER bounds:

| Mathematical completion for H1–H15 | Rank-one [L,U] | QPSK² [L,U] | Circular 16-PSK [L,U] |
|---|---|---|---|
| A: all \(G=\mathrm{diag}(1,0.1)/1.1\) | [0.009119, 0.013945] | [0.025795, 0.058064] | [0.025554, 0.051237] |
| B: all \(G=(0.01I+0.99uu^\dagger)/1.01\), \(u=(\cos(\pi/8),\sin(\pi/8))\) | [0.187958, 0.190544] | [0.012952, 0.031258] | [0.025554, 0.051237] |

Thus rank-one is better than both in A, and worse than both in B. The conditional inequalities follow by averaging the existing Gaussian SER lower/upper bounds across uniformly sampled receiver slots. A separate 60-digit direct-channel implementation checked both completions; two successive moving-frame changes preserved the bounds when channel, codebook, physical ports and receiver-noise covariance were transported consistently. The smallest strict margin (0.011609) dwarfs the at-most \(7.98\times10^{-8}\) ensemble-gap change from the printed H0 rounding assumption. The [executable construction](./experiments/receiver_partial_identification.py), [frozen verification protocol](./PARTIAL_IDENTIFICATION_VERIFICATION_PROTOCOL.md), [results ledger](./PARTIAL_IDENTIFICATION_RESULTS_2026-11-09.json) and [clean-checkout CPU run](https://github.com/jesusvilela/sionna-rt/actions/runs/42587624459) record the evidence.

**Critical domain restriction:** these completions satisfy matrix-level normalization and rank constraints, but have **not** been generated by, or shown to satisfy, the *specific Munich scene's ray-tracing equations and spatial channel correlations*. They do not establish non-identifiability given the full scene and solver, only **insufficiency of the published printed H0 plus basic slot constraints**. Their coordinates were chosen after exploring the frozen grid, not by a prospectively preregistered discovery experiment. This correction does not change the source notebook or supply any missing Munich measurements.

## Covariance-aware decoding: a reproducible engineering check

The Gram decision rule above assumes **white** complex Gaussian receiver noise. For a known positive-definite covariance \(\Sigma\) in \(n\sim\mathcal{CN}(0,N_0\Sigma)\), use the *noise-weighted* Gram \(G_\Sigma=H^\dagger\Sigma^{-1}H\). Under an invertible receiver-coordinate change \(W\) and unitary transmitter-coordinate change \(U\), transport **all** objects: \(H'=WHU^\dagger\), \(\Sigma'=W\Sigma W^\dagger\), \(x'=Ux\), including the physical transmit-port projectors. Then \(G'_\Sigma=UG_\Sigma U^\dagger\), preserving all conditional ML pair distances and Gaussian SER bounds. This is **standard whitening/coherent-detection mathematics**, not new 6G physics.

The [frozen CPU protocol](./COLORED_NOISE_FRAME_PREREG.md) checked six existing codebooks across the **same 75 synthetic Gram cells**, with two fixed nonunitary receiver-frame changes and \(N_0=0.01\):

| Test | First frame | Two successive frames |
|---|---:|---:|
| Correctly transported decisions preserved | **375/375** | **375/375** |
| Incorrectly assume transformed noise remains white: changed decisions | **96/375** | **77/375** |
| Same incorrect white metric, adjusted to the correct trace: changed decisions | **96/375** | **79/375** |
| Maximum correct Gaussian-bound deviation | 1.22×10⁻¹⁵ | 2.22×10⁻¹⁵ |

Changed decisions include transitions into or out of *undecided*, not only strict reversals. An independent numerical decoding path on **2,048 seeded noise realizations using the one printed H** returned zero changed ML decisions under the correctly transported metric. Neither check is a new channel measurement: **0/16 Munich matrices and no measured colored-noise covariance** remain available. The engineering requirement is to preserve the covariance and physical polarization frame whenever receiver data are transformed.

The [compact result ledger](./COLORED_NOISE_FRAME_RESULTS.json), [executable audit](./experiments/colored_noise_frame_audit.py) and [12 regression tests](./tests/test_colored_noise_frame_audit.py) record the evidence. Reproduce on CPU with NumPy and SciPy, from the fork root (use fresh output filenames):

~~~bash
python research/experiments/gram_decision_sweep.py --out research/experiments/gram_sweep_2026-11-04.json
python research/experiments/colored_noise_frame_audit.py --source research/experiments/gram_sweep_2026-11-04.json --out colored_noise_frame_results.json
python -m unittest discover -s research/tests -p 'test_colored_noise_frame_audit.py' -v
~~~

## Retired prediction claim: three-site geometric phase

The classical invariant `B_ijk=tr(P_i P_j P_k)` is mathematically legitimate. We froze a grouped train/test ablation before fitting: 60 nonisotropic synthetic cells, 36 training rows and 24 held-out rows, target the archived strict rank-one-better-than-circular decision, pairwise overlaps plus eigenvalue ratio versus the same model with `Re B, Im B` added.

**Held-out Brier loss:** pairwise-only **0.185431**; plus-B **0.500000**; difference **+0.314569** (worse; grouped bootstrap 95% interval **[+0.121535,+0.549670]**). It failed the preregistered criterion for provisional value (improvement at least 0.02, with interval entirely below zero). Thus **NO demonstrated predictive benefit**, and the proposed engineering advantage is **retired** for this task.

Important diagnostic: training triangles had near-zero imaginary B, while held-out triangles had `|Im B|≈0.1768`. A fixed standardizer amplified this out-of-support extrapolation. Do **not** present the magnitude of deterioration as evidence that B is universally harmful, and do **not** reselect the split after seeing it. More fundamentally, the baseline's three anchored overlaps reconstruct each local principal projector; with eigenvalue ratio and fixed trace, it reconstructs full local G. The per-site SER-bound target is *already fixed by local G*. B carries no **additional conditional information** for this model and target, regardless of learner.

## One-command-at-a-time reproduction

The [frozen protocol](./PREREG_BARGMANN_ABLATION_2026-11-05.md) was committed before the first fit. The source grid and ablation are deterministic on Python 3.13.5, NumPy 2.3.5, scikit-learn 1.8.0 (plus SciPy for the preexisting Gram sweep):

```bash
python research/experiments/gram_decision_sweep.py --out gram_sweep_2026-11-04.json
python research/experiments/bargmann_ablation.py --source gram_sweep_2026-11-04.json --out bargmann_ablation_results.json
python -m unittest discover -s research/tests -p 'test_bargmann_ablation.py' -v
```

The [negative-result ledger](./ABLATION_OUTCOME_2026-11-05.json) preserves source/result SHA-256 hashes and the frozen decision threshold. Seven structural/metamorphic regression tests pass; they are self-authored, not independent scientific replication. These scripts create no physical channel data.

**Data-ingestion correction (5 November).** A self-authored NPZ plus matching SHA256 is not independently authenticated Munich propagation evidence. The [provenance gate](./MUNICH_PROVENANCE_AUDIT.md) now rejects synthetic fixtures and unverified imports by default; CPU regenerated channels remain explicitly labeled *candidates*. There are still **0/16 authenticated Munich matrices**. This changes only the evidence interface, not any constellation bound or negative-result conclusion.

## Clean-checkout CPU reproducibility checkpoint (9 November 2026)

At its initial clean-checkout checkpoint, the [fork-only CPU regression workflow](../.github/workflows/research-cpu-regression.yml) checked **175 research unittests across 16 files**, with Python 3.13 and pinned NumPy/SciPy/scikit-learn packages; **no Sionna, GPU, HF or external propagation samples**. In a clean checkout, it first regenerates the **75 synthetic Gram cells** and checks the archived SHA-256 `16ad7ca1eda73d9c6a73776555be2f93333d248ad44d11a32ac9be3b05d2f4c6` independently of the test fixtures. It then executes the entire test suite **without** committing or pre-installing that JSON.

The [successful CI run of 9 November 2026](https://github.com/jesusvilela/sionna-rt/actions/runs/42577583312) reports **175/175 passing**. A first clean-checkout run revealed two **test-harness integration** defects: a colored-noise test expected an untracked JSON fixture, and a provenance test expected a checksum error without enabling its explicit synthetic-fixture diagnostic mode. Both tests were corrected; the numerical models, original notebook, PR #82, fixed codebooks and research claims were unchanged.

**Evidence ceiling:** this confirms reproducibility of the *self-authored CPU research tests* against the frozen synthetic data. It does not independently verify the science, authenticate Munich propagation, remove the 0/16 receiver-data gap, or revive the retired geometric-performance claims.

## Error-aware receiver acceptance test (9 November 2026)

A future channel acquisition must provide **both** a frame-aligned effective Gram estimate \(\widehat G_v=H_v^\dagger\Sigma_v^{-1}H_v\) and an independently supported **spectral-norm error budget** \(\|G_v-\widehat G_v\|_{\mathrm{op}}\le\varepsilon_v\). For every codeword difference \(\Delta\), its received squared distance differs from the estimate by at most \(\varepsilon_v\|\Delta\|^2\). Applying the monotone Gaussian tail to those two distance endpoints, then averaging the *maximal pairwise lower* and *all-pairs upper* probabilities, produces a **conservative, simultaneous SER certificate** for all admissible channels. The current implementation requires \(\varepsilon_v<\lambda_{\min}(\widehat G_v)\) to exclude receiver-codeword collisions without an explicit tie policy. Under moving frames, the input Gram, codebook and physical ports transform together; nonunitary receiver changes additionally transport the noise covariance.

In the two existing, wholly synthetic **15-missing-slot** counterworlds (with the one printed H0 fixed and its eight-decimal rounding budget included), the sufficient uniform error thresholds for the unknown slots are:

| Counterworld | Rank-one vs QPSK² | Rank-one vs circular 16-PSK | Interpretation |
|---|---:|---:|---|
| A, rank-one better | \(\varepsilon<0.020365\) | \(\varepsilon<0.029026\) | Conservative Gaussian-bound separation expires |
| B, rank-one worse | \(\varepsilon<0.009901\) | \(\varepsilon<0.009901\) | **Positive-definiteness gate**, not observed ranking failure |

On the previously frozen 75-Gram grid, the comparisons at \(\varepsilon=0,10^{-4},10^{-3}\) yielded the **same strict/abstention counts** for the two fixed competing designs. A further 225 seeded positive-definite perturbation trials obeyed all predicted event and SER envelopes. These are *software regression controls*, not independent source-channel measurements. Source: [pre-execution protocol](./GRAM_UNCERTAINTY_PREREG_2026-11-09.md), [executable CPU test](./experiments/gram_uncertainty_certificate.py), [adversarial regression suite](./tests/test_gram_uncertainty_certificate.py), and [numerical ledger](./GRAM_UNCERTAINTY_RESULTS_2026-11-09.json).

**Tightness correction (9 November 2026).** For the *unrestricted* Hermitian spectral-norm Gram ball, \(G_\pm=\widehat G\pm\varepsilon I\) jointly attain all pairwise distance endpoints. The propagated Gaussian lower \(L\) and union upper \(U\) are therefore attained extrema of those **pairwise-derived functionals** over that ball, but not extrema of exact ML SER. For the *additional trace-preserving* model \(E=E^\dagger,\operatorname{tr}E=0,\|E\|_{\rm op}\le\varepsilon\), each individual PEP endpoint remains attainable with \(E_\pm=\pm\varepsilon(2vv^\dagger-I)\), \(v=\Delta/\|\Delta\|\). Joint attainment is a different question.

For two unit difference directions \(u_1,u_2\), \(c=|u_1^\dagger u_2|<1\), any traceless 2×2 Hermitian E with \(\|E\|_{\rm op}\le\varepsilon\) has one negative eigenline \(w\), and the two-line inequality gives \(\min_k|w^\dagger u_k|^2\le(1+c)/2\). Thus for at least one selected pair, \(u_k^\dagger E u_k\ge-\varepsilon c\). Its PEP falls short of its independently maximized upper endpoint by at least
\[
\delta_k=Q\!\left(\sqrt{\frac{q_k-\varepsilon\|\Delta_k\|^2}{2N_0}}\right)-
Q\!\left(\sqrt{\frac{q_k-\varepsilon c\|\Delta_k\|^2}{2N_0}}\right)>0.
\]
The symmetric ordered pair appears twice in the 16-symbol union sum, so the *actual maximum trace-preserving Gaussian union-upper functional* is below the separable endpoint by at least \(2\min(\delta_1,\delta_2)/16\). The bound is strict for any book containing two non-collinear differences. For single-beam circular 16-PSK all differences are collinear, so one fixed traceless E attains all upper endpoints simultaneously (and its opposite attains all lower endpoints). This proves the fixed-trace union-envelope distinction without asserting exact ML SER extrema.

On the frozen **75-cell synthetic grid** at \(\varepsilon=\min(10^{-3},\lambda_{\min}(\widehat G)/4)\), the chosen difference-line coherences are \(c=0.3826834324\) (rank-one), \(c=0\) (QPSK²), and \(c=1\) (circular 16-PSK). The two-line theorem's smallest per-cell aggregate-gap lower bound is only **\(5.0689\times10^{-26}\)** for rank-one and **\(3.1924\times10^{-15}\)** for QPSK²; the circular gap is zero because joint attainment holds. The largest such lower bounds across the grid are \(7.8672\times10^{-8}\) and \(9.4138\times10^{-10}\), respectively. **Conclusion: the fixed-trace nonattainment is strict but the worst-grid quantitative guarantee is negligible**; it does not materially tighten the earlier robustness thresholds. In 19,200 seeded random traceless perturbations, the smallest *sampled* actual union-functional gap was \(1.5803\times10^{-9}\); this is a finite sample, not a global extremum. The minimum analytic gap was independently recomputed at 60 decimal digits. The [frozen protocol](./FIXED_TRACE_AGGREGATE_SHARPNESS_PREREG.md), [CPU calculation](./experiments/fixed_trace_aggregate_sharpness.py), [eight-test audit](./tests/test_fixed_trace_aggregate_sharpness.py) and [result ledger](./FIXED_TRACE_AGGREGATE_SHARPNESS_RESULTS_2026-11-09.json) retain these limits. Clean-checkout CI passed **214/214 tests**: [run](https://github.com/jesusvilela/sionna-rt/actions/runs/42654843602). An initial test draft added an unregistered \(10^{-12}\) minimum-gap requirement and failed; it was removed after the 60-digit result showed the true positive gap can be \(5.07\times10^{-26}\). This is recorded as a test-design correction, not hidden as a scientific pass.

**Trace premise correction:** \(\operatorname{tr}(H^\dagger H)=\|H\|_F^2=1\) follows from Frobenius-normalized H under *white* noise. It does **not** imply \(\operatorname{tr}(H^\dagger\Sigma^{-1}H)=1\) for colored-noise effective Grams; for example \(H=I/\sqrt2,\Sigma=\mathrm{diag}(1,4)\) gives trace \(5/8\). A trace-preserving uncertainty set for a colored-noise effective Gram therefore requires a separately stated and justified constraint. The original uncertainty thresholds remain for the **unrestricted** spectral-norm ball; the fixed-trace result is a separate mathematical audit.

**Research boundary:** the error budgets are assumptions about not-yet-measured noise-weighted Grams and are **not observed calibration precision or a specification verified in Munich**. This is standard matrix perturbation and coherent Gaussian detection applied to an existing conditional result; it adds neither data nor geometric novelty. The retired geometric advantage claims stay retired.

## From independently calibrated errors to a Gram-radius certificate

An operational missing bridge is how to convert **external instrument-error guarantees** into the already defined effective-Gram uncertainty \(\varepsilon_G\). Given \(\widehat H,\widehat\Sigma\succ0\), let \(LL^\dagger=\widehat\Sigma\), \(Z=L^{-1}\widehat H\), \(a=\|Z\|_{\mathrm{op}}\), and **assume independently justified** \(\|L^{-1}(H-\widehat H)\|_{\mathrm{op}}\le\tau\), \(\|L^{-1}(\Sigma-\widehat\Sigma)L^{-\dagger}\|_{\mathrm{op}}\le\rho<1\). The classical inverse-resolvent bound gives

\[
\varepsilon_G=\frac{a^2\rho+2a\tau+\tau^2}{1-\rho}.
\]

The triple \((a,\tau,\rho)\), and hence this bound, is invariant under a simultaneous \(GL(2,\mathbb C)\) receiver-coordinate change and \(U(2)\) transmitter-frame change when **both true and estimated** channel/noise objects are transported; an unwhitened absolute error need not be. The existing conditional SER certificate may use \(\varepsilon_G\) only if \(\varepsilon_G<\lambda_{\min}(\widehat G)\); otherwise it **abstains**. The bound is mathematically tight over general independent operator-norm balls: one aligned \(2\times2\) diagonal witness attains \(\varepsilon_G=0.55\).

A [frozen protocol](./WHITENED_CALIBRATION_BRIDGE_PREREG_2026-11-09.md), [CPU implementation](./experiments/whitened_calibration_bridge.py), [nine adversarial tests](./tests/test_whitened_calibration_bridge.py) and [result ledger](./WHITENED_CALIBRATION_BRIDGE_RESULTS_2026-11-09.json) document **600 synthetic source/noise perturbations, 48 composed frame checks and 223 passing clean-checkout CPU tests** ([successful run](https://github.com/jesusvilela/sionna-rt/actions/runs/42691468179)). Assuming *hypothetically* \(\tau=\rho=0.001\), the two prior synthetic counterworlds yield \(\varepsilon_G=0.00282\) and \(0.00298\), below their earlier sufficient thresholds. **Neither bound is an observed calibration accuracy.** This interface does not authenticate a receiver, justify a probabilistic confidence level, or recover any of the 16 missing Munich matrices. No earlier performance/novelty claim changes.

**Operational moving-frame qualification (9 November 2026).** The analytic \((a,\tau,\rho,\varepsilon_G)\) invariance under receiver \(GL(2,\mathbb C)\) is **not** unrestricted software acceptance invariance. The existing CPU validator rejects nominal noise covariance representations with raw condition number \(>10^{10}\), a deliberate numeric safety rule. For \(\widehat H=\operatorname{diag}(1,0.5)\), \(\widehat\Sigma=I\), and \(W=\operatorname{diag}(10^6,1)\), the transported effective Gram remains \(\operatorname{diag}(1,0.25)\) but the raw covariance condition becomes \(10^{12}\), forcing rejection. The exact obstruction is structural: for any two positive-definite covariances, \(W=\Sigma_2^{1/2}\Sigma_1^{-1/2}\) maps one to the other by congruence. Hence **no nontrivial covariance-only numerical condition cutoff can be invariant under every GL receiver transformation**. The code now distinguishes a typed \`NumericalFrameConditionError\` (*cannot safely evaluate that raw representation*) from \`abstain_positive_definiteness_gate\` (*declared physical uncertainty radius not sufficient*), without increasing the numerical limit. A trustworthy re-expression into a well-conditioned receiver frame must transport channel, noise and independently validated error attestations together; this does **not** create missing measurements. This is a correction of the numerical acceptance claim, not a new result about radio propagation. See the [post-hoc defect and frozen verification protocol](./NUMERICAL_FRAME_DOMAIN_PREREG_2026-11-09.md) and [five new CPU regression tests](./tests/test_numerical_frame_domain.py).

**Covariance input-scale correction (10 November 2026).** A post-hoc input-validation audit found that the calibration bridge's former absolute Hermitian skew test \(\max|\widehat\Sigma-\widehat\Sigma^\dagger|\le10^{-12}\) could accept a **grossly non-Hermitian covariance** at small raw scale, then silently symmetrize it into a seemingly certifiable Gram. For example, \(\widehat\Sigma=10^{-20}I+i10^{-14}I\) has absolute skew \(2\times10^{-14}\) but relative skew approximately 2. The revised CPU gate checks the dimensionless maximum-entry-normalized defect \(e_H=\max|\widehat\Sigma/s-(\widehat\Sigma/s)^\dagger|\le10^{-12}\), with \(s=\max|\widehat\Sigma_{ij}|\), and symmetrizes **only after** that check. This preserves the input's Hermiticity decision under common scalar receiver gains; it does **not** claim unrestricted \(GL(2,\mathbb C)\) numerical acceptance invariance. The existing \(10^{10}\) raw-covariance condition cutoff and physical-error abstention semantics remain unchanged. The [post-discovery frozen repair protocol](./COVARIANCE_SCALE_HERMITICITY_PREREG_2026-11-10.md) and [seven adversarial CPU tests](./tests/test_covariance_scale_hermiticity.py) cover invalid tiny-scale inputs, benign scaling, tolerated roundoff and fail-closed rejection. This is input correctness, **not** a physical Munich calibration result.

## Evidence boundary and smallest next action

**Proved under model:** Gaussian pairwise bounds, printed-channel conditional ordering, white-noise Gram sufficiency and covariance-aware frame equivalence; **measured in synthetic data:** 75-case classifications, a frozen held-out ablation and a covariance-transport regression; **known prior mathematics:** Pancharatnam/Bargmann phases; **open:** actual 16 Munich receivers, external propagation generalization and fair optimized-baseline testing; **retired:** determinant's unique causal performance role, and a demonstrated Bargmann engineering advantage for the synthetic per-site decision.

**Next smallest permissible action:** preserve this concise, source-linked note as the fork's scientific entry point. Stop building geometric layers or fitting new synthetic variations. Only reopen the propagation question if a source-matched, versioned 16-receiver H archive becomes available; the prior CPU regeneration attempt was blocked by Sionna/Mitsuba/Dr.Jit dependency resolution. No synthetic approximation substitutes for those measurements.
