# CPU covariance whitening: matched-scale stress and test-harness correction (12 November 2026)

## Verdict and scope

The fork's optional PyTorch portability tool had a concrete **test-harness** bug: `compile_probe` called the boolean `torch.dtype.is_complex` property as a function. This prevented the advertised four-dtype graph-capture check from executing. The call was corrected at commit `77d7943`, locally replayed, and the existing-path errors were demoted from mandatory assertions to measured outcomes at `bfef045`. The research helper's focused subnormal tests were separately replayed using an extracted numerical core: **7/7 passed**. None of this is an exact-head full GitHub Actions run.

The narrower PyTorch observation survives: scaling a particular *represented* subnormal 2×2 complex128 covariance to order `1e-320` yields substantial error in the old Cholesky path relative to high-precision exact-input inversion, while componentwise normalization corrects it on the tested CPU. This is **not** a Sionna RT receiver-performance result, not a new theorem, and not yet a justified Sionna PHY hot-path change.

## Inputs, controls, and evidence

- Upstream-derived numerical primitive: `NVlabs/sionna` `src/sionna/phy/utils/linalg.py` blob `cbc53d3e8791834ebc8cdc8001441867c997320d`. The fork's `torch_whitening_portability.py` replicates the `cholesky_ex` and triangular inverse sequence without importing Sionna. It is **not** an exact checkout of production Sionna.
- Matched deterministic synthetic generator: `torch_whitening_stress.py` — NumPy PCG64 seed **20261112**; exactly **64 fixed** complex 2×2 SPD covariance shapes `S0=B B†+0.8I`, normalized and Hermitian-symmetrized, and channels `H0=chol(S0) diag(1,0.5)`. For each shape, compare the **same** matrices rescaled by `s`.
- Numerical acceptance: the *actual rounded complex128 covariance* must remain exactly Hermitian and positive definite, tested with extended-range `longdouble` leading principal minors. Invalid represented inputs are counted and **excluded before measuring either algorithm**.
- Independent arithmetic oracle: mpmath direct `H†S^-1 H` at 100 decimal digits on **actual IEEE binary64 inputs**, not the unquantized ideal model. All reported errors are maximum absolute Gram-entry errors.
- Environment: local CPU, Python 3.13.5, NumPy 2.3.5, PyTorch 2.10.0+cpu, mpmath 1.3.0, one PyTorch compute thread. No GPU, original Munich receiver data, calibrated noise, radio propagation, or wireless measurements.
- Code reproduction: `python research/experiments/torch_whitening_stress.py --out /tmp/matched-stress.json`. Requires optional PyTorch, NumPy and mpmath dependencies. The `--out` file uses exclusive creation to avoid overwriting evidence.

## Matched 64-shape experiment

| Covariance scale | Represented SPD | Existing Cholesky, median/max error | Component-normalized, median/max error | Error above 1e-8 (old/new) |
|---|---:|---:|---:|---:|
| `1` | 64/64 | 5.55e-17 / 2.22e-16 | 5.55e-17 / 4.44e-16 | 0 / 0 |
| `1e-200` | 64/64 | 5.55e-17 / 2.22e-16 | 1.11e-16 / 4.44e-16 | 0 / 0 |
| `1e-309` | 64/64 | 5.69e-16 / 4.11e-15 | 1.11e-16 / 4.44e-16 | 0 / 0 |
| `1e-320` | 64/64 | **7.96e-5 / 3.57e-4** | **7.62e-17 / 6.66e-16** | **64 / 0** |
| `5e-324` | **25/64** | 0 / 2.22e-16 | 0 / 2.22e-16 | 0 / 0 |

At the smallest subnormal, 39/64 scaled covariance matrices ceased to be SPD **after binary64 quantization** and were excluded from *both* algorithms. These cannot be treated as failures of Cholesky on valid SPD inputs. The 25 retained cases establish no special superiority of either algorithm. The experiment is deterministic and exploratory, not independently replicated.

This matched control strengthens the scale-specific local CPU finding, but does not show practical receiver-channel incidence. The normal-scale negative control shows that normalization does not always improve floating-point error.

## Test correction and resource parity

The earlier `--compile` option was **nonfunctional** because `dtype.is_complex()` raised `TypeError`. After correcting it, rerunning `--compile --benchmark` locally completed four `torch.compile(..., fullgraph=True, backend="eager")` dtype controls with exact equality. These controls **do not** exercise Inductor, CUDA graphs, GPUs or production Sionna.

Repeated warmup/median single-thread CPU measurements after the fix: approximately 2.39x the existing time for batches of 128 and 1.59x for batches of 1024, using the always-normalized candidate. The ratios vary by run; they are evidence of a resource cost **on this CPU**, not general hardware performance estimates.

A second correction removes assertions requiring the upstream-derived baseline to fail at a particular scale or perform well at another. Such assertions confound the measured observable with a chosen expected outcome and could reject a legitimate improvement in upstream or hardware versions. The candidate's measured error is still checked against the independently computed oracle.

## Specification → implementation → test audit

1. Requirement: preserve the noise-weighted Gram under physically consistent representation changes, but do not infer actual receiver error budgets without instruments.
2. Numerical target: stable recovery of `H†S^-1H` for represented Hermitian SPD matrices, not unconditional equivalence to unquantized `s*S0`.
3. Existing production-related path: Sionna PHY torch Cholesky; the NumPy fork helper is not that path.
4. Test infrastructure: an optional, focused CPU script with independent high-precision numerical oracle, synthetically constructed SPD inputs, matched scalar-scale controls and a deliberately wrong covariance-frame negative control.
5. Measurement: extreme-subnormal precision gap on this CPU only; no source-channel calibration, SER, BER, latency or throughput inference.
6. Interpretation: **numerical edge case**. Do not promote as universal Sionna bug or 6G sensitivity breakthrough.

## P/A/M/H/S/R ledger

| Claim | Evidence tag and explanation |
|---|---|
| `chol(sS)=sqrt(s)chol(S)` over positive real `s` and SPD `S` | **P**, exact algebra |
| Fork NumPy helper now handles selected subnormal cases | **M**, focused local replay, 7/7 extracted-core tests |
| Existing PyTorch path has elevated Gram error at `1e-320` | **M**, local matched 64-shape CPU experiment |
| Always-normalized candidate resolves those selected errors | **M**, the same local matched cases |
| Relative condition-number policy and provenance gates are physically validated | **A/H**, not established by software tests |
| Original Munich receiver claim improved or new SER result | **R**, unsupported; no authenticated field input |
| Unconditional upstream Sionna patch warranted | **R**, unsupported because normal-scale overhead and absent external relevance |
| Exact-head GitHub CI succeeded | **H**, not verified; GitHub combined statuses and PR-filtered workflow list are not a push CI witness |

## Stop rule and next bounded experiment

Keep `NVlabs/sionna-rt` PR #82 untouched and do not submit this research script as a production fix. The next admissible *upstream* experiment is exact source-checkout reproduction on a second supported CPU or GPU, comparing the existing method against a compatible alternative with matched batches, dtypes, input scales, compilations and end-to-end detector outputs. **Kill condition:** no practical-scale accuracy benefit or material throughput regression. **Promotion rule:** only a reproducible source-level numerical fix with relevant operating domain, tests, review and DCO sign-off. Do not generalize from artificial subnormal inputs to physical radio performance.
