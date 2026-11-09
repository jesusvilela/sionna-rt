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

The [fork-only CPU regression workflow](../.github/workflows/research-cpu-regression.yml) now checks **all 175 research unittests across 16 files**, with Python 3.13 and pinned NumPy/SciPy/scikit-learn packages; **no Sionna, GPU, HF or external propagation samples**. In a clean checkout, it first regenerates the **75 synthetic Gram cells** and checks the archived SHA-256 `16ad7ca1eda73d9c6a73776555be2f93333d248ad44d11a32ac9be3b05d2f4c6` independently of the test fixtures. It then executes the entire test suite **without** committing or pre-installing that JSON.

The [successful CI run of 9 November 2026](https://github.com/jesusvilela/sionna-rt/actions/runs/42577583312) reports **175/175 passing**. A first clean-checkout run revealed two **test-harness integration** defects: a colored-noise test expected an untracked JSON fixture, and a provenance test expected a checksum error without enabling its explicit synthetic-fixture diagnostic mode. Both tests were corrected; the numerical models, original notebook, PR #82, fixed codebooks and research claims were unchanged.

**Evidence ceiling:** this confirms reproducibility of the *self-authored CPU research tests* against the frozen synthetic data. It does not independently verify the science, authenticate Munich propagation, remove the 0/16 receiver-data gap, or revive the retired geometric-performance claims.

## Evidence boundary and smallest next action

**Proved under model:** Gaussian pairwise bounds, printed-channel conditional ordering, white-noise Gram sufficiency and covariance-aware frame equivalence; **measured in synthetic data:** 75-case classifications, a frozen held-out ablation and a covariance-transport regression; **known prior mathematics:** Pancharatnam/Bargmann phases; **open:** actual 16 Munich receivers, external propagation generalization and fair optimized-baseline testing; **retired:** determinant's unique causal performance role, and a demonstrated Bargmann engineering advantage for the synthetic per-site decision.

**Next smallest permissible action:** preserve this concise, source-linked note as the fork's scientific entry point. Stop building geometric layers or fitting new synthetic variations. Only reopen the propagation question if a source-matched, versioned 16-receiver H archive becomes available; the prior CPU regeneration attempt was blocked by Sionna/Mitsuba/Dr.Jit dependency resolution. No synthetic approximation substitutes for those measurements.
