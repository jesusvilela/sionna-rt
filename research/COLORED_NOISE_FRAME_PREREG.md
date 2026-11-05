# Frozen audit: non-unitary receiver frames and colored receiver noise

**2026-11-05; CPU only; source:** existing 75 synthetic Gram cells from `gram_sweep_2026-11-04.json`, SHA256 `16ad7ca1eda73d9c6a73776555be2f93333d248ad44d11a32ac9be3b05d2f4c6`. No new wireless channel observations. Parent `research/munich-provenance-boundary-2026-11-05`. Do not change PR #82 or post upstream.

**One engineering issue, no new geometry.** Earlier frame audit was correctly restricted to receiver *unitaries* with white noise. For any invertible complex receive-chain transformation W, the same statistical observation can be represented by `y'=Wy`, `H'=WH`, `Sigma'=W Sigma W^H`; using a white-noise Euclidean metric after that change generally changes the detection problem. Retain full complex covariance `Sigma`.

**Model and closed algebra.** Coherent ML for noise `n~CN(0,N0 Sigma)` and fixed source-unit-energy C uses `d_Sigma^2(Delta) = Delta^H H^H Sigma^{-1} H Delta` and Gaussian PEP `Q(sqrt(d_Sigma^2/(2 N0)))`. For an arbitrary invertible W, and independent transmit-coordinate unitaries U (which also move all source codewords, physical port projectors and the coordinate representation of G), the transformed objects are:
`H'=W H U^H`, `Sigma'=W Sigma W^H`, `C'=UC`. Therefore `G'_Sigma = U G_Sigma U^H`, preserving every received squared pair distance, Gaussian PEP interval, and strict comparison.

**Frozen transforms (no tuning).** Start with each source G as trace-one positive-definite, take a deterministic spectral factor H with `H^H H=G`, and `Sigma=I`. Apply two receive-frame transformations in sequence:
`W1 = [[1.3+0.2i, 0.6-0.3i],[0.1+0.4i, 0.75]]` and
`W2 = [[0.9, -0.35+0.15i],[0.2+0.05i, 1.1-0.1i]]`;
two transmit-coordinate unitaries `U1=SU2(23°, 0.38 rad)`, `U2=SU2(13°, -0.61 rad)` and compose them in that order. No receiver optimization or channel-adaptive W.

**Primary gate.** All 75×5 rank-one-vs-other design decisions and all six full Gaussian error intervals must be unchanged, as must all 120 pairwise squared distances per codebook, with an absolute numerical tolerance of `1e-10`. Also require exact codebook physical energy parity under unitary input reframe, and noise-covariance Hermitian positivity.

**Negative controls (frozen before running).** First treat nonunitarily transformed data with incorrect white noise: `G_wrong = H'^H H'`. Second remove the scalar-trace change by normalizing `G_wrong` to the correct trace and *still* use a false white metric. Report the number of changed strict/undecided comparisons for both controls (changes to/from abstention count separately; no invented "performance benefit"). Do not assume a particular nonzero count in advance; if neither changes, claim only mathematical covariance, not a measured confounder for this test.

**Independent witnesses and stop rules.** Check `H^H Sigma^-1 H` two ways: linear solve and Cholesky whitening of H. Compare actual black-box coherent ML scores and decisions on the same complex-noise samples before/after W. For nonidentity original Sigma, run randomized positive-definite test cases. Reject singular W or non-PD Sigma rather than hiding failure by pseudoinverse. If a correctly transported interval or decision changes above tolerance, do not claim the implementation is covariance-safe; report the exact failing test. Do not retune transforms or metrics after seeing outcomes.

**Interpretation boundary.** This is established Mahalanobis/whitened coherent detection, not novel 6G math, measured receiver-chain noise, or improved constellation SER. The source notebook assumes white noise; its published conditional conclusions remain in that scope. Continue to report 0/16 real Munich H matrices and no supported Bargmann engineering advantage.

**Stop:** one CPU-only bounded experiment; commit source, tests, results and a short appendix, then stop. No Sionna run, GitHub upstream PR, Work, GPU, HF credits, or new geometric/holonomy layers.
