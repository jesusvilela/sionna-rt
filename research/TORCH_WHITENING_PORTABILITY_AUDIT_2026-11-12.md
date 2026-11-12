# Covariance whitening portability audit — 12 November 2026

**Verdict:** Keep the scale-normalized NumPy research correction in the fork; **do not** claim the same defect or port the fix wholesale into Sionna RT. The actual upstream covariance whitening lives in `NVlabs/sionna` (Sionna PHY). Its PyTorch Cholesky/triangular-solve arithmetic already handles the previously problematic `1e-309` input accurately. A narrower discrepancy appears around `1e-320` in one complex128 CPU control, and separate-component normalization removes it, but unconditional normalization incurs overhead on ordinary matrices. The measurement does **not** establish a practical radio-performance improvement or justify an upstream PR.

## Frozen objects, provenance and representation

- Fork implementation: `research/experiments/whitened_calibration_bridge.py` and `research/tests/test_subnormal_complex_whitening.py` on `research/subnormal-covariance-whitening-2026-11-10`. These are **NumPy research helpers**, not Sionna production APIs.
- Upstream Sionna RT `NVlabs/sionna-rt`: ray tracing and polarized propagation; code search did not locate a production receiver-noise whitening API.
- Upstream Sionna PHY `NVlabs/sionna`, file `src/sionna/phy/utils/linalg.py`, blob SHA `cbc53d3e8791834ebc8cdc8001441867c997320d`: `inv_cholesky` computes `cholesky_ex(A, check_errors=False)` then triangular inverse. The matching `whiten_channel` entry point is `src/sionna/phy/mimo/utils.py` (blob SHA `47bafd8ba318b580e7763a4aafc35c9af734f4bd`).
- Upstream precondition: Hermitian positive-definite covariance; `inv_cholesky` deliberately avoids runtime tensor-value validation for CUDA graph capture. Our research-only `NumericalFrameConditionError` is **not** part of its contract.
- Instrument: local CPU, Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5, mpmath 1.3.0. No production Sionna package invocation, no Nvidia CUDA execution, no GPU, no Munich receiver dataset.
- Independent oracle: 100 decimal digits, direct `H† S⁻¹ H` inversion of **the actual complex128 input bit patterns** in mpmath. The oracle is independent of the PyTorch Cholesky code path but not an external, independently obtained channel measurement.
- Executable reproducer: [optional standalone CPU script](./experiments/torch_whitening_portability.py); torch is **not** installed by the fork's lightweight research CPU workflow, so this cross-framework experiment is deliberately opt-in.

## Exact bounded experiment

Use `S0=[[1,0.25+0.1j],[0.25-0.1j,1.5]]` and `H0=chol(S0) diag(1,0.5)`; transport by `S=s S0` and `H=sqrt(s) H0`. This is a **scalar receiver-frame equivalence experiment** away from quantization; at the smallest subnormal, do not equate the represented S with the ideal scale multiple of S0.

| `s` | Existing PyTorch max entrywise Gram error vs 100-digit oracle | Scaled candidate |
|---:|---:|---:|
| 1 | < 1e-15 | < 1e-15 |
| 1e-20 | 2.22e-16 | 2.22e-16 |
| 1e20 | < 1e-15 | < 1e-15 |
| 1e-200 | 5.55e-17 | 2.22e-16 |
| 1e-300 | 2.78e-17 | 4.44e-16 |
| 1e-309 | 1.11e-16 | 1.11e-16 using componentwise normalization |
| 1e-320 | **2.94e-5** | **2.22e-16** using componentwise normalization |
| 5e-324 | 2.78e-17 | 4.44e-16 using componentwise normalization |

The defect is **not** that upstream PyTorch fails at `1e-309`; it does not in this control. It is the narrower numerically ill-scaled `1e-320` case, where current Cholesky processing departs from an independent exact-input oracle. The smallest subnormal case shows that representing a changed covariance can dominate the *ideal-frame* comparison while the computed Gram still matches the **actual input**.

A naïve `S/scale` performed using complex division in this PyTorch control produced NaNs at `1e-309` and smaller. The successful candidate divides real and imaginary components separately by a real positive scale *before* Cholesky, then rescales the inverse factor by `1/sqrt(scale)`. The exact-arithmetic identity is
`L(sA)^-1 = L(A)^-1/sqrt(s)` for positive real `s`.

Moving nonunitary receiver-frame control at ordinary scale:
- Correctly transporting `(H,S)` by `(W H, W S W†)` changed the effective Gram by at most `1.55e-16` per entry.
- Changing `H` but leaving `S` fixed produced an operator-norm Gram difference of `0.61997665`, a deliberately invalid control.
- A 2x2 covariance is not assumed to represent a real Munich scene. No channel data were acquired.

## Compatibility and cost control

The candidate's `torch.compile(..., fullgraph=True, backend="eager")` graph capture matched direct evaluation for `torch.float32`, `float64`, `complex64`, and `complex128`. **This is not Inductor, CUDA graph, multi-GPU or production Sionna validation.**

Matched warmup + repeated local CPU median timings, single torch thread, complex128 positive-definite batches:

| Batch of 2x2 matrices | Existing path | Always-normalized candidate | Relative cost |
|---:|---:|---:|---:|
| 128 | 0.0389 ms/call | 0.0770 ms/call | 1.98x |
| 1024 | 0.2148 ms/call | 0.3283 ms/call | 1.53x |

Timing measures are noisy and machine-dependent; no general performance conclusion is licensed. They do establish that **an unconditional hot-path rewrite is not a free fix** on the inspected CPU.

## Evidence ledger

| Claim | Evidence | Tag |
|---|---|---|
| Cholesky's positive-scalar normalization identity | Exact algebra on SPD matrices | P |
| Original NumPy helper could form nonfinite normalized complex entries on tiny SPD input | Fork code, bounded numerical reproduction | M |
| PyTorch upstream-derived `inv_cholesky` loses ~2.94e-5 on one `1e-320` matrix | Local faithful micro-reimplementation + independent 100-digit oracle | M (local only) |
| Componentwise normalized candidate improves that specific case | Local matched negative control | M (local only) |
| Fullgraph eager works for four dtypes | Local capture experiment; not production compiled backend | M (local only) |
| A universally enabled upstream rewrite is warranted | Not supported; baseline is faster on ordinary tests | R |
| New 6G SER or receiver sensitivity result | No authentic channel/noise calibration, no transmission experiment | H, unsupported |

## Decision / gates

**No upstream PR or NVIDIA issue on this evidence alone.** `NVlabs/sionna-rt` has no matching live receiver-whitening path. `NVlabs/sionna` is the appropriate *future* venue, but the 1e-320 CPU edge case does not establish real-world value and unconditional normalization adds ordinary-case cost.

**Next smallest test, if justified:** independently reproduce on the exact upstream Sionna PHY checkout (not only the matched function reproduction) on another CPU and supported GPU/dtype; include normal noise levels, dynamic shapes, compiled Inductor, throughput and stable SER/LLR baselines. **Kill condition:** no effective error in a supported practical regime or the overhead erases the benefit. **Stop condition:** no independent reproduction/data; leave the experimental correction in the fork and do not promote radio claims.

## CI boundary

At the time this note was authored, no exact-head GitHub Actions green result had been retrieved for the new subnormal research branch. A green run must be tied to its actual commit SHA, not inferred from previously passing 235/235 tests or from a missing GitHub commit-status list. The optional torch script is outside the lightweight workflow by design.
