# Frozen fixed-trace aggregate sharpness test — 9 November 2026

Parent: \`research/adversarial-pairwise-sharpness-2026-11-09\`. CPU-only; no new channel samples, no Sionna run, no new geometry. This resolves one narrow open statement in the note: whether the coordinatewise Gaussian union upper envelope is jointly attainable when the effective Gram perturbation is constrained to be traceless.

## Scope distinction (important)
- The existing Gram uncertainty certificate uses the **unrestricted** Hermitian spectral-norm ball \(\|G-\widehat G\|_{\rm op}\le\varepsilon\). Its joint PEP-derived \(L\) and \(U\) extrema are attained by \(+\varepsilon I\) and \(-\varepsilon I\), respectively, when the ball remains positive definite.
- A separate *trace-preserving* uncertainty model additionally requires \(\operatorname{tr}(G)=\operatorname{tr}(\widehat G)\), so the error \(E\) is traceless. This is implied by Frobenius-normalized \(H\) **only** for white-noise \(G=H^\dagger H\), or must be declared as a separate assumption on the effective noise-weighted Gram \(G_\Sigma=H^\dagger\Sigma^{-1}H\). It does NOT follow from \(\|H\|_F=1\) alone when \(\Sigma\ne I\).
- No trace-preserving claim is transferred to colored-noise effective Grams without a separately measured/assumed trace constraint.

## Frozen theorem and quantitative gap
Let \(\widehat G\succ0\), \(\|E\|_{\rm op}\le\varepsilon<\lambda_{\min}(\widehat G)\), \(E=E^\dagger\), \(\operatorname{tr}E=0\). Then \(E\) has eigenvalues \(\pm\rho\), \(0\le\rho\le\varepsilon\). For any unit \(u\), writing \(w\) for its negative-eigenvector, \(u^\dagger Eu=\rho(1-2|w^\dagger u|^2)\).

For two unit difference directions \(u_1,u_2\), define \(c=|u_1^\dagger u_2|<1\). The elementary two-line bound is \(\min(|w^\dagger u_1|^2,|w^\dagger u_2|^2)\le(1+c)/2\). Therefore at least one direction has \(u_k^\dagger E u_k\ge-\varepsilon c\), and its PEP is at most \(f(q_k-\varepsilon c\|\Delta_k\|^2)\), where \(f(q)=Q(\sqrt{q/(2N_0)})\). The independently maximized PEP endpoint is \(f(q_k-\varepsilon\|\Delta_k\|^2)\). The positive difference
\[
\delta_k=f(q_k-\varepsilon\|\Delta_k\|^2)-f(q_k-\varepsilon c\|\Delta_k\|^2)>0
\]
means that **for every single traceless E**, the ordered-pair union sum misses its separable endpoint by at least \(2\min(\delta_1,\delta_2)/16\) when both directions are chosen from the same 16-point book. Thus the *aggregate union upper Gaussian functional* is strictly less than the sum of individual PEP maxima for any codebook containing two non-collinear difference directions. This is not a statement about exact ML SER.

For a single-beam book \(x_k=z_k b\), all differences are collinear; \(E=-\varepsilon(2bb^\dagger-I)\) jointly attains all union-upper PEP endpoints, while the opposite sign attains all lower-PEP endpoints.

## Fixed choices and tests
- Use exactly the existing 75 synthetic Gram cells, SHA256 \`16ad7ca1eda73d9c6a73776555be2f93333d248ad44d11a32ac9be3b05d2f4c6\`, fixed codebooks and \(N_0=0.01\).
- Fixed pair directions: rank-one \((0,1),(0,2)\); QPSK² \((0,1),(0,4)\); circular 16-PSK \((0,1),(0,7)\).
- At each grid cell and each book set \(\varepsilon=\min(10^{-3},\lambda_{\min}(\widehat G)/4)\). Compute \(c\), \(\delta_1,\delta_2\) and the rigorous minimum aggregate upper-functional gap \(2\min(\delta_1,\delta_2)/16\).
- For every grid cell and the two multi-beam books, test 128 seeded random traceless E matrices (seed 20261109), each \(\|E\|_{\rm op}\le\varepsilon\); verify direct-H actual union upper is below the separable endpoint by at least the derived gap (tolerance \(10^{-12}\)).
- For circular 16-PSK, use the fixed beam projector to show all 120 ordered PEP upper endpoints and the aggregate upper functional are jointly attained by one traceless E for all 75 cells.
- Check the exact two-line coherence constants and at least 60-digit arithmetic on the minimum computed \(\delta\) to rule out float subtraction artifacts.
- Test the trace premise: for \(\|H\|_F=1\) and nonidentity positive-definite \(\Sigma\), \(\operatorname{tr}(H^\dagger\Sigma^{-1}H)\) need not equal 1. Require the example \(H=I/\sqrt2,\Sigma=\operatorname{diag}(1,4)\) to produce trace \(5/8\).
- Preserve the prior unrestricted-ball joint endpoint result. No re-fitting, no alternative pair selection, no tuning, no claim of exact SER extrema. Stop after tests, ledger, and correction to the existing technical note.

**Evidence ceiling:** P = conditional two-line eigenvector inequality and trace distinction; M = 75×2×128 numerical witnesses and 75 circular joint extremizers; H = actual Munich trace-preserving calibration and 15 missing channels unknown; R = any inference of physical measurement precision or new geometry. No Work/GPU/HF, no NVlabs action, PR #82 untouched.
