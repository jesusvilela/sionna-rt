# Operational frame-domain falsification — 9 November 2026

Parent: research/whitened-calibration-bridge-2026-11-09. Explicitly a **post-hoc defect found during repository inspection**; the counterexample was reproduced locally before this protocol was written. This file freezes the repair and CI acceptance tests, **not** the discovery.

## Claim being corrected
The conditional error estimate \(\|G-\widehat G\|_2\le(a^2\rho+2a\tau+\tau^2)/(1-\rho)\) is invariant under consistent receiver \(GL(2,\mathbf C)\) and transmitter \(U(2)\) coordinate transport. The implementation, however, rejects nominal covariance matrices whose raw floating-point condition exceeds \(10^{10}\). That computational domain is *not* GL-invariant, so the earlier generic \`receiver_frame\` metadata risks conflating a mathematical invariance with an operational accept/reject guarantee.

## Exact non-novel mathematical obstruction
For any two Hermitian positive-definite \(\Sigma_1,\Sigma_2\), \(W=\Sigma_2^{1/2}\Sigma_1^{-1/2}\in GL(2,\mathbf C)\) satisfies \(W\Sigma_1W^\dagger=\Sigma_2\). Thus the GL congruence action on positive-definite covariance matrices is transitive. An invariant predicate dependent **only on \(\Sigma\)** is constant; no nontrivial raw condition-number cutoff can be invariant over the entire GL group. This is known linear algebra, not a new theorem.

## Frozen adversarial witness and stop conditions
- Nominal \(\widehat H=\operatorname{diag}(1,0.5)\), \(\widehat\Sigma=I\), externally **assumed** but not measured error bounds \(\tau=\rho=0.001\); effective Gram diag(1,0.25), so old SER eligibility is true. No actual radio observation.
- One physically equivalent receiver frame \(W_1=\operatorname{diag}(1000,1)\) with raw covariance condition \(10^6\): old API should keep identical Gram radius/eligibility within 1e-12.
- One more frame \(W_2=\operatorname{diag}(10^6,1)\) with raw condition \(10^{12}>10^{10}\). The physical effective Gram must be identical to 1e-12 by independent calculation, but the safe raw-coordinate implementation must **fail closed** with a typed \`NumericalFrameConditionError\`, *not* report \`abstain_positive_definiteness_gate\` and never mark a real radio problem as invalid.
- The exception must remain a \`ValueError\` subtype for existing callers, have clear observed condition/limit and instruct validated re-expression in a better-conditioned receiver frame; **do not increase or remove the 10^10 cutoff**.
- Distinguish \`numerical_frame_domain_rejection\` (raw representation unsuitable for this CPU path) from ordinary \`abstain_positive_definiteness_gate\` (declared Gram error set could contain a degenerate channel).
- The original public API and its 223 CPU tests must stay green. Add at least 4 focused regression checks: GL-transitivity witness, moderate-frame acceptance, extreme-frame typed rejection, and genuinely too-wide error-budget abstention.
- Update the original output's misleading unrestricted receiver-frame metadata to explicitly scope **theoretical invariance** versus **numerical acceptance only on well-conditioned representations**. Update one paragraph in the existing technical note; no new scientific models, geometry, frozen Gram grid refitting, new SER thresholds or sources.
- No physical Munich receiver data; **PR #82 untouched**, no NVlabs modifications, GPU, HF or Work. No external model execution.

## Evidence and failure contract
- DERIVED: transitivity proof and exact nominal Gram transformation; no Lean/kernel check or originality assertion.
- ARTIFACT-VERIFIED if and only if final fork CI tests pass on exact last head SHA.
- A: error budgets still assumed, not independently calibrated; raw CPU cutoff a numerical policy.
- R: claim that *all raw coordinate representations* produce identical software eligibility status.
- If the new test reveals a physical Gram mismatch, reject the implementation rather than waiving tolerances. If numerical cutoff changes, treat as scope expansion and stop.
