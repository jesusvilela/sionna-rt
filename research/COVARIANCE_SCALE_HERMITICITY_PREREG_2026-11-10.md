# Frozen covariance Hermiticity scale audit — 10 November 2026

**Parent:** \`research/numerical-frame-domain-audit-2026-11-09\`. **Scope:** existing \`research/experiments/whitened_calibration_bridge.py\` nominal covariance input gate, no physical-channel claims. One specific **post-hoc defect was already discovered and locally reproduced** before writing this protocol; this freezes the subsequent repair and regression checks, not the original discovery. Fork-only, CPU-only, no Work, GPU, HF, Sionna run, or changes to upstream PR #82.

## Falsified operational claim

The current bridge accepts only finite Hermitian positive-definite nominal noise covariances. But it checks \`max(abs(S-S†)) <= 1e-12\` in absolute raw units, then replaces \`S\` with \`(S+S†)/2\`. For

\[
S_{\rm bad}=10^{-20}I+i\,10^{-14}I,\qquad H_{\rm bad}=10^{-10}{\rm diag}(1,0.5),
\]

the old gate sees \`2e-14<1e-12\`, silently erases the enormous relative anti-Hermitian component and returns a positive \`conditionally_certifiable\` status (with \(\tau=\rho=0.001\)) despite no valid covariance having been supplied. The actual relative entrywise defect \(2e^{-14}/\max|S_{\rm bad}|\) is ~2.

## Minimal fix and frozen gates

Use an explicit **dimensionless** Hermitian defect
\[
e_H(S)=\max_{ij}|S_{ij}/s-\overline{S_{ji}}/s|,\qquad s=\max_{ij}|S_{ij}|>0.
\]
Reject \(e_H(S)>10^{-12}\) or nonfinite values. If the relative skew is at or below tolerance, explicitly replace by the symmetric part using \((S/2)+(S^\dagger/2)\) to avoid avoidable overflow of \`S+S†\`. Keep the **original** numerical condition-number policy \(10^{10}\) and typed \`NumericalFrameConditionError\`; preserve the \`abstain_positive_definiteness_gate\` distinction.

**Frozen acceptance tests before patching:**
1. Reject the adversarial non-Hermitian diagonal at scale \`1e-20\` (both direct \`gram_radius_from_calibration\` and test-only \`oracle_relative_errors\` paths); report the old accepted result in the correction log.
2. For valid Hermitian SPD \(S_0=[[1,0.25+0.1i],[0.25-0.1i,1.5]]\), and \(H_0=S_0^{1/2}\operatorname{diag}(1,0.5)\), transport by common **scalar** \(W=s^{1/2}I\) for \`s in {1e-20, 1, 1e20}\`. Assert same whitened effective-Gram spectrum \(\{1,.25\}\), same declared error budget \(\tau=\rho=.001\), and same conditional eligibility within \`1e-11\`; this is a \`GL(2,C)\` transformation with well-conditioned covariance.
3. For each scale, deliberately insert \(3e^{-14}s\,i\) on one diagonal of \(S\), representing a relative perturbation below \`1e-12\`: allow the input only after explicit symmetrization; the resulting Gram/radius differs from the Hermitian reference by at most \`1e-11\`. For the same three scales, insert \(1e^{-8}s\,i\): reject for relative defect far above threshold, independent of absolute scale.
4. Preserve existing malformed, non-PD and condition-number-rejection tests; do not increase the condition limit, change any SER thresholds, codebook, source Gram dataset, or physical provenance status.
5. Run the **entire fork CPU research regression suite** from clean checkout, plus the 75-cell source hash and prior independent checks. If the new tests fail, report it instead of changing the frozen tolerance or replacing the adversarial input.

## Epistemic statement

This is an input-validation/numerical-scale correctness repair. Relative Hermiticity is invariant under a common nonzero scalar receiver frame. It is **not** claimed invariant under all ill-conditioned \`GL(2,C)\` transformations; prior numerical-domain caveat remains. \`10^{-12}\` is a numerical engineering tolerance, not a mathematically necessary physical level and not an externally calibrated noise bound.

DERIVED: a covariance must be Hermitian, and scalar frame scaling preserves its Hermiticity; ARTIFACT-VERIFIED if the fork code+tests and exact head CI pass; MEASURED: synthetic numerical behavior only; UNKNOWN: original 16 Munich channel matrices/noise calibrations; RETIRED: any claim that the old absolute check was a scale-safe Hermitian validity gate.
