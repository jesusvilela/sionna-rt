# Engineering appendix: non-unitary receiver frames require noise covariance

**Research checkpoint, 2026-11-05. CPU only, fork only. No Sionna execution, HF/ZeroGPU credits, NVlabs comments or updates.** The scientifically defensible [short technical note](./TECHNICAL_NOTE_SCOPE_AND_NEGATIVE_RESULT.md) is unchanged: **one printed rounded Munich H, 75 synthetic Gram cells, 0/16 recovered Munich matrices, known geometry**.

## Verdict (P/M/R)

**P, conditional linear algebra:** for proper complex Gaussian noise, the physically relevant coherent ML metric is not \(\|H\Delta\|_2^2\) after an arbitrary change of receiver coordinates. It is

\[
d_\Sigma^2(\Delta)=\Delta^\dagger H^\dagger\Sigma^{-1}H\Delta,
\quad n\sim\mathcal{CN}(0,N_0\Sigma).
\]

For a lossless, invertible *receiver-side* coordinate map \(W\in GL(2,\mathbb C)\) and input-coordinate unitary \(U\in U(2)\),

\[
H'=WHU^\dagger,\qquad
\Sigma'=W\Sigma W^\dagger,\qquad
x'=Ux
\]

imply

\[
\boxed{H'^\dagger\Sigma'^{-1}H'=U(H^\dagger\Sigma^{-1}H)U^\dagger}.
\]

Therefore all 120 pairwise ML squared distances and the nearest-error-lower / all-pair-union-upper Gaussian SER bounds are preserved when codebooks and physical transmit-port projectors are transported with \(U\). The pairwise probability remains \(Q(\sqrt{d_\Sigma^2/(2N_0)})\).

This is standard Mahalanobis/whitened coherent detection, **not a new mathematical theorem**, a novel quantum/holonomy layer, or a measured radio improvement.

## Frozen CPU audit (M: finite synthetic calculation)

The [pre-registration](./COLORED_NOISE_FRAME_PREREG.md) was committed before experiment execution; the input was the earlier unchanged 75-case synthetic Gram dataset (SHA256 `16ad7ca1eda73d9c6a73776555be2f93333d248ad44d11a32ac9be3b05d2f4c6`). The same six 16-symbol codebooks and \(N_0=0.01\) were used. A deterministic square root of each source \(G\) generated one **synthetic** channel matrix \(H\) with \(H^\dagger H=G\). This is not propagation data.

Two frozen complex, nonunitary receiver maps were applied in sequence, along with two independent transmit-input unitaries. All error bounds were calculated from \(G_\Sigma=H^\dagger\Sigma^{-1}H\). Negative controls intentionally discarded \(\Sigma\), first entirely, then after rescaling the resulting false Gram to the correct trace.

| Audit measure | Receiver frame 1 | Two successive frames |
|---|---:|---:|
| **Correctly transported strict/abstention decisions unchanged** | **375/375** | **375/375** |
| Maximum absolute change in conditional Gaussian error bound | 1.22e-15 | 2.22e-15 |
| Maximum absolute change in one pair's squared distance | 3.55e-15 | 6.22e-15 |
| Incorrect white-noise metric: changed decisions | **96/375** | **77/375** |
| Incorrect white-noise metric: direct strict A↔B reversals | 40 | 28 |
| Incorrect white-noise metric, corrected to same trace: changed decisions | **96/375** | **79/375** |
| Trace-corrected incorrect metric: direct strict A↔B reversals | 31 | 28 |

Changes include transitions into/out of **undecided**, not only strict reversals. The 75 Gram cells contain repeats at axis and isotropy configurations; these are *grid-level artifacts*, **not percentages of wireless receivers**. Multiplying a receiver's data by \(W\) without transforming its noise covariance amounts to a different assumed statistical model. The mistaken apparent SER-ranking changes are entirely representational, not physical.

The source-printed Munich **single** H was used only for a black-box ML-scoring equivalence test with 2,048 seeded proper complex-noise symbols: **zero changed ML decisions** and a maximum full-symbol score discrepancy \(4.44\times10^{-15}\). Another independent numerical path using Cholesky whitening tested non-identity correlated noise; both paths remain AI-authored and are **not independent scientific replication**.

## Implementation and evidence scope

- [Executable experiment](./experiments/colored_noise_frame_audit.py): source SHA check; full \(G_\Sigma\) via linear solve; independent Cholesky whitened reference; two composed receiver \(GL(2,\mathbb C)\) maps plus transmitter \(U(2)\) coordinate changes; all six fixed/CSI codebooks; Gaussian decision bounds; two naive negative controls; printed-H black-box test.
- [Twelve new CPU tests](./tests/test_colored_noise_frame_audit.py): known-answer diagonal covariance, 80 random Hermitian positive-definite covariance witnesses, direct Gaussian receiver ML sample equivalence under nonunitary maps, frame composition, singular/invalid covariance rejection, port-power parity and all frozen 75×6 codebook checks.
- [Compact result ledger](./COLORED_NOISE_FRAME_RESULTS.json), with SHA256 of the full 75-cell [reproducible JSON](./experiments/colored_noise_frame_audit.py). The full local result file's SHA256 is `3478aa1973a859da38da327efebfcd536ca68ae690aa24f0bd3fc7458f5505f8`.
- [Audit manifest](./COLORED_NOISE_FRAME_AUDIT_MANIFEST.json): verified with installed Recursive Research Auditor validator, **0 errors, 0 warnings**.

In the local CPU container, **12/12 new tests**, **12/12 earlier provenance tests**, and **10/10 earlier frame-resource tests** passed: 34 focused tests total. The entire Sionna/PR #82 notebook CI and all historical repository suites were **not** run. GitHub blob SHA verification protects the source/test consistency, not the external scientific claim.

## Boundaries and stopping condition

**P:** conditional Gaussian model invariance under simultaneously transported channel, noise and codebook.

**M:** synthetic 75-grid invariance and deliberately wrong whitening negative controls; sampled printed-H coherent decoder equivalence.

**A/H:** whether any real Sionna dual-polarization receiver has a materially nonwhite calibrated covariance or nonunitary front-end; none was measured here.

**R:** interpreting \(H'^\dagger H'\) as the same coherent detector metric after nonunitary \(W\) without transforming \(\Sigma\). Previous determinant/null-cone unique causality and Bargmann predictive usefulness remain retired.

**Next smallest step, gated on real data:** if a calibrated receiver chain becomes available, record \(H_v,\Sigma_v\), the transmit/receive polarization port bases and noise/SNR convention together, then compare invariant whitened-bound calculations against the existing conditional white-noise rule. Do **not** synthesize a measured covariance, launch GPU jobs, or add geometry in order to escape the missing 0/16 Munich data.

This engineering appendix is a reproducibility acceptance test, not a new publishable 6G physics result. PR #82 was not touched.
