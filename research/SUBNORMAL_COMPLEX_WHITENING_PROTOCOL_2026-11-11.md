# Subnormal complex covariance whitening — bounded numerical repair (11 Nov 2026)

**Status:** post-discovery validation protocol. The defect was reproduced before this document; it is not presented as prospectively discovered. **Parent:** `research/covariance-scale-hermiticity-2026-11-10`. Fork-only research helper; no changes to the open upstream constellation PR, no Sionna/GPU/ray tracing and no new Munich observations.

## Frozen failure and mathematical object

The effective Gram is `G = H† Sigma^{-1} H`, with `Sigma` a finite complex128 Hermitian positive-definite covariance. The previous relative-skew validation computed `S / max(abs(S))` using NumPy complex division. For `S = 1e-309*S0`, `S0=[[1,0.25+0.1j],[0.25-0.1j,1.5]]`, this produces nonfinite entries, although the *represented* S is finite and positive definite with small condition number.

A positive scalar frame change `H'=sqrt(s)H`, `S'=sS` must preserve the effective Gram whenever both are represented accurately enough; deep subnormal input quantization itself is not invariant under finite-precision encoding.

## Candidate repair

1. Divide the real and imaginary components of each complex entry separately by the positive real `scale=max(abs(S))`. Reject nonfinite scaled results.
2. Validate relative Hermitian skew, positive definiteness and the **unchanged 1e10 covariance condition limit** on the scaled covariance. Symmetrize *after* scaling to avoid subnormal half-entry loss.
3. Compute `L_unit=chol(S/scale)`, `Z=solve(L_unit,H/sqrt(scale))`; scale complex H by its real and imaginary components separately. Return the same raw-coordinate Cholesky factor `sqrt(scale)*L_unit` for the existing test-only oracle. Do not silently clip errors.
4. Retain the prior distinction between `NumericalFrameConditionError` and `abstain_positive_definiteness_gate`. An unrepresentable whitened channel must fail closed with a numerical ValueError.

## Frozen tests and controls

- The original `1e-309` SPD test must now pass with nonzero finite Gram, and agree within `1e-12` entrywise with an independent 100-digit mpmath calculation on the **actual binary-float inputs** (not an idealized unrounded S).
- Repeat on `s in {1,1e-20,1e20,1e-200,1e-300,1e-309,1e-320}` and on the smallest subnormal. Compare each to its exact *represented input* mpmath reference at `1e-12` absolute; the deepest subnormals need not reproduce the ideal unscaled model because entry quantization is large.
- At `s=1e-309`, the effective Gram should agree with the well-scaled base within `1e-11` and remain conditionally eligible for illustrative `tau=rho=0.001`.
- Negative controls: large relative anti-Hermitian input at the same scale must reject; `diag(1e11,1)` covariance must retain the exact condition-limit rejection; nonsensically large H with subnormal covariance must reject without claiming physical abstention.
- Verify oracle realized whitened errors on the subnormal frame against the matching well-scaled control. Run all existing CPU tests plus frozen 75-case SHA-256 regeneration, printed high-precision test and synthetic partial-identification controls.

## Promotion and stop rule

Promote only to an **artifact-tested research numerical correctness repair** after exact-head clean-checkout CI. Never reinterpret this as an error budget provided by physical calibration or as an improvement to SER. If exact-head CI is unavailable, report the patch as committed but unverified. Stop here unless a new independently observed numerical defect or authenticated receiver data provides a reason to continue.
