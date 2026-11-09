# Frozen source-calibration-to-Gram certificate protocol — 9 November 2026

Parent \`research/fixed-trace-aggregate-sharpness-2026-11-09\`; fork-only. No NVlabs edits, no PR #82 changes, CPU only, no Sionna/HF/Work. The prior scientific scope stays **one printed H, 75 synthetic Gram cells, 0/16 original Munich receiver matrices**.

## Gap to close, no extra geometry
The existing \`gram_uncertainty_certificate.py\` accepts an *assumed* effective-Gram absolute spectral-norm error \(\varepsilon_G\) for \(G=H^\dagger\Sigma^{-1}H\). Future acquisition is more likely to report separate estimates \(\widehat H,\widehat\Sigma\) and validated relative uncertainty bounds. A useful interface must convert genuinely externally justified channel+noise uncertainties to \(\varepsilon_G\) **without** treating a self-reported confidence score or synthetic fixture as authentication.

## Conditional theorem
Assume proper Gaussian receiver noise \(\mathcal{CN}(0,N_0\Sigma)\), fixed known \(N_0>0\), \(\widehat\Sigma\succ0\), actual \(\Sigma\succ0\), estimate \(\widehat H\), and a declared physical transmitter-port frame. Let \(L L^\dagger=\widehat\Sigma\) be its Cholesky factor. Define
\[
Z=L^{-1}\widehat H,\quad
F=L^{-1}(H-\widehat H),\quad
R=L^{-1}(\Sigma-\widehat\Sigma)L^{-\dagger}.
\]
Assume deterministically justified operator-norm bounds \(\|F\|_2\le\tau\), \(\|R\|_2\le\rho<1\), where \(R\) is Hermitian. Write \(a=\|Z\|_2\), \(G=\!H^\dagger\Sigma^{-1}H\), \(\widehat G=\widehat H^\dagger\widehat\Sigma^{-1}\widehat H=Z^\dagger Z\). Neumann resolvent/submultiplicativity yields the **conditional** bound
\[
\|G-\widehat G\|_2\le \varepsilon_{\rm eff}
  =\frac{a^2\rho+2a\tau+\tau^2}{1-\rho}.
\]
No new mathematics or wireless-performance claim. This estimate is simultaneously valid for every actual pair satisfying the *declared* upper bounds. It becomes an input for the **already existing** Gaussian error certificate only if \(\varepsilon_{\rm eff}<\lambda_{\min}(\widehat G)\), otherwise **abstain**.

## Exact moving-frame law
A simultaneous transmitter \(U\in U(2)\) and arbitrary invertible receiver \(W\in GL(2,\mathbb C)\) representation change maps both nominal and true objects by \(H\mapsto WHU^\dagger\), \(\Sigma\mapsto W\Sigma W^\dagger\), physical port projectors by \(E_j\mapsto UE_jU^\dagger\). The Cholesky factors differ by a unitary \(V=L_{\rm new}^{-1}W L_{\rm old}\), hence \((Z,F,R)\mapsto(VZU^\dagger,VFU^\dagger,VRV^\dagger)\). Thus \(a,\tau,\rho,\varepsilon_{\rm eff}\) are *exactly invariant* in mathematics. Numeric test tolerance is \(2\times10^{-10}\) for well-conditioned transforms. Do not assert coordinate invariance for the naive **absolute** norms \(\|\delta H\|,\|\delta\Sigma\|\), which change under \(W\).

## Frozen controls before execution
1. Deterministic saturating case: \(Z=\mathrm{diag}(1.25,0.3)\), \(F=\mathrm{diag}(0.05,0)\), \(R=\mathrm{diag}(-0.2,0)\), baseline \(\widehat\Sigma=I\). The exact norm gap and calculated \(\varepsilon_{\rm eff}\) both equal **0.55** (no unverified objective). Also verify a non-diagonal \(\widehat\Sigma\) is just a representation of that same case.
2. Source grid SHA256 \`16ad7ca1eda73d9c6a73776555be2f93333d248ad44d11a32ac9be3b05d2f4c6\`; use exactly 75 synthetic Grams and fixed seed **20261109**. Reconstruct a spectral H with \(H^\dagger H=G\); set the nominal noise covariance to the fixed SPD \(\widehat\Sigma=[[1,0.2+0.15i],[0.2-0.15i,1.3]]\), \(\widehat H=L H\). Run 8 seeded random complex channel-error/covariance-error pairs per cell (**600 source cases**), alternating fixed budgets \((\tau,\rho)=(0,0),(10^{-4},10^{-3}),(10^{-3},10^{-2}),(5\times10^{-3},0.1)\). Errors have independently normalized spectral norm \(\le\) their budgets. Require \(\|G_{\rm true}-\widehat G\|_2\le\varepsilon_{\rm eff}+10^{-11}\) and actual/nominal covariance SPD. Record max excess over the bound (non-positive mathematically).
3. On the first 12 grid cells, for each pair with \((\tau,\rho)=(10^{-3},10^{-2})\), use two frozen receiver \(W\) frames \([[1.3+0.2i,0.6-0.3i],[0.1+0.4i,0.75]]\) and the product with \([[0.9,-0.35+0.15i],[0.2+0.05i,1.1-0.1i]]\), and two frozen input SU(2) frames (angles 23°,13°, phases 0.38,-0.61 radians). Recompute Cholesky whitened norms and \(\varepsilon_{\rm eff}\) from transported *true and estimated* objects as a **test oracle only**, not in the public certification API. Require errors \(\le2\times10^{-10}\). Include a deliberate incorrect control that moves H but leaves \(\Sigma\) fixed; expect detectable metric change.
4. On existing synthetic counterworlds A/B, compute \(\varepsilon_{\rm eff}\) for **illustrative assumed**, not measured, \(\tau=\rho=0.001\). Compare against the previous immutable required budgets 0.020364816713798736 (A) and 0.009900990099009875 (B); if either exceeds, report failure without tuning. Do not claim an instrument achieved those errors.
5. Fail closed on \(\tau<0\), \(\rho<0\), \(\rho\ge1\), NaN/infinity, malformed/non-Hermitian/non-PD nominal covariance, and numerical condition number \(>10^{10}\). If \(\varepsilon_{\rm eff}\ge\lambda_{\min}(\widehat G)\), do not issue an SER-ordering certificate. At \(\rho\to1\), record divergence and abstention.
6. Standalone CPU tests, add to clean-checkout research workflow, and append at most one concise qualification to the main technical note. No Sionna CPU regeneration and no external data. **If any test fails:** report the specific counterexample; do not alter saturation witness, seed, thresholds or frame choices to force a pass.

## Authority and evidence
P/DERIVED: conditional resolvent spectral inequality and GL×U covariance; not a kernel-checked theorem.
M: exact algebraic saturator and finite self-authored CPU tests.
A: the input \(\tau,\rho\) must be independently calibrated upper bounds; a confidence interval must additionally carry explicit frequentist coverage, joint error events and union accounting.
R: any inference that Munich per-site \(H,\Sigma\) or their errors have been measured from these numerical controls. CI validates code behavior in tested cases, not physical data provenance.
Known mathematical provenance: Neumann-series inverse norm inequality, e.g. Cornell CS6210 Matrix Computations lecture 2025-08-27 (https://www.cs.cornell.edu/courses/cs6210/2025fa/lec/2025-08-27.html).
