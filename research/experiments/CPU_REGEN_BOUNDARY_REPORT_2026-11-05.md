# Munich CPU regeneration and Gram phase-boundary audit

**Date:** 2026-11-05 · **Branch:** `research/munich-cpu-regeneration-boundary-2026-11-05` · **Scope:** CPU-only, fork-only · **Upstream:** NVlabs PR #82 **untouched**.

## Verdict

**The 16 receiver channels were not regenerated.** This execution environment has no `sionna`, `mitsuba` or `drjit`, and PyPI cannot be resolved by the sandbox network (error: `[Errno -3] Temporary failure in name resolution`). An explicit `pip install sionna-rt==2.2.0` failed before downloading a wheel. Two isolated child processes attempted the exact-notebook and deterministic CPU modes, but both stopped before initializing ray tracing. No real or synthetic surrogate was committed as a Munich channel array; no Frobenius or Gram difference between nonexistent runs is reported.

We completed the theory and regression part: an exact finite quarter-phase-orbit minimum-distance envelope gives the leading high-SNR boundary, and stronger coherent-AWGN bounds resolve **19** of the previous **44** undecided grid comparisons. All **19 new CPU tests passed**. They are local self-authored validations, not NVIDIA CI.

## 1. Environment and bounded attempts

- Python: 3.13.5; NumPy 2.3.5; SciPy 1.17.0. Linux CPU-only sandbox, no swap.
- Requested package: `sionna-rt==2.2.0`, **not installed** because package index DNS fails; Mitsuba and Dr.Jit versions are **unresolved**, not fabricated version pins.
- CPU backend selected *in code*: `llvm_ad_mono_polarized`; never a CUDA fallback.
- Source: `jesusvilela/sionna-rt` PR #82 head `57435845d3395b2d5e167b0893035e15ffbdaa0d`.
- Original notebook: `PathSolver()(scene,max_depth=5)` with implicit default seed 42; NVIDIA does not guarantee determinism from a fixed seed alone.
- Deterministic alternative: `PathSolver(deterministic=True)(scene,max_depth=5,seed=42)`. This changes the candidate generator and cannot be called an exact original channel reproduction.
- Reduced config *available but not run*: deterministic, `max_depth=2`, `samples_per_src=20000` and seed 42. Reduction is for a proven wall-time bottleneck, **not** a workaround for missing packages.

| Attempt | Hard wall budget | Virtual-memory cap | Actual elapsed | Result |
|---|---:|---:|---:|---|
| Exact-notebook best effort | 30 s | 4096 MiB | 0.874 s | Dependency failure before solver |
| Deterministic CPU | 30 s | 4096 MiB | 0.837 s | Dependency failure before solver |

The bounded subprocess runner isolates each mode, enforces `RLIMIT_AS` and a wall-clock deadline, records exit status, and terminates a timed-out process group. It refuses a non-CPU Mitsuba variant. If a run succeeds in a network-enabled environment, the output contains **all 16 positional slots**, alive mask, raw and Frobenius-normalized complex matrices, receiver coordinates, source revision, runtime package versions and SHA-256 manifest.

The final runner also refuses stale, partial or tampered outputs: a comparison is permitted only when both **current** solver attempts succeeded and each current NPZ checksum, mode, source revision and CPU backend match its manifest. An extra adversarial regression corrupts a temporary archive and verifies rejection.

The exporter keys were matched to the previous `gram_decision_sweep.py` contract: `H_all_normalized`, `H`, `alive` and `rx_positions_xyz_m`, plus `num_receiver_positions=16`, `all_16_available` and `archive_sha256`. A synthetic **temporary-file regression** verifies the layout; no such temporary file is represented as physical evidence.

**No output NPZ was committed**, because there was no regenerated 16-receiver dataset. A fictional NPZ or one built by repeating the printed receiver would violate provenance and the requested scientific stop rule.

## 2. High-SNR boundary: exact finite algebra, conditional conclusion

Let `G=H†H`, with eigenvalues `lambda_plus≥lambda_minus>0`, ratio `r=lambda_minus/lambda_plus` and normalized principal vector

\[
u=(\cos\theta,e^{i\phi}\sin\theta).
\]

For every codeword difference `Delta`,

\[
d_G^2(\Delta)=
\lambda_- \|\Delta\|^2+
(\lambda_+-\lambda_-)
|\cos\theta\Delta_1+e^{-i\phi}\sin\theta\Delta_2|^2.
\]

The actual rank-one and QPSK² constellations are each unions of **four complete global quarter-phase orbits**. Choose representatives `v_0,...,v_3` and write `e_k=v_k†Gv_k`, `z_{k\ell}=v_k†Gv_\ell`. Their exact minimum received squared distance is

\[
M_G(C)=
\min\left\{
\min_k 2e_k,\quad
\min_{k<\ell}
[e_k+e_\ell-2\max(|\Re z_{k\ell}|,|\Im z_{k\ell}|)]
\right\}.
\]

For rank-one one may take `v_k=e^{iπ/4}(\cos(kπ/4),\sin(kπ/4))`. For QPSK² take `v_k=e^{iπ/4}(1,i^k)/\sqrt2`, in both cases `k=0,1,2,3`.

For the **fixed off-null-cone circular 16-PSK**, with `v=(1,i)/sqrt2`,

\[
M_G(C_{\rm circ})=\kappa
(\operatorname{tr}G/2-\Im G_{12}),
\quad
\kappa=2-2\cos(\pi/8).
\]

**The leading high-SNR phase regions are explicitly the finite analytic inequalities**

\[
\boxed{
\begin{aligned}
\mathcal R_{\mathrm{rank>QPSK}}
&=\{(r,\theta,\phi):M_G(C_{\rm rank})>M_G(C_{\rm QPSK^2})\},\\
\mathcal R_{\mathrm{rank>circ}}
&=\{(r,\theta,\phi):M_G(C_{\rm rank})>
\kappa(\operatorname{tr}G/2-\Im G_{12})\}.
\end{aligned}}
\]

These are *piecewise analytic* in `r,theta,phi` (a finite minimum of explicit trigonometric expressions), with no hidden numerical optimization or claim of new literature-level novelty. For distinct minimum distances, the nearest-pair lower and union upper Gaussian bounds show that the constellation with larger `M_G` has lower SER for **all sufficiently small N0**. Equal minimum distances are **unresolved** by that criterion; multiplicities and complete distance spectra matter.

**Aligned diagonal special case.** For `G=diag(1,r)/(1+r)` the two exact thresholds are

\[
r_{\rm circ}=\frac{\kappa}{4-\kappa}
\approx0.0395661299,\quad
r_{\rm QPSK}=3-2\sqrt2\approx0.1715728753.
\]

Thus rank-one wins *both* asymptotic minimum-distance comparisons precisely when

\[
\boxed{0.0395661299<r<0.1715728753,}
\]

within that aligned diagonal family. Outside it, orientation and relative phase can reverse the order; no ratio-only rule is universal.

## 3. Tightening the original 75-cell Gaussian decision rule

The previous frozen sweep used `r={.01,.03,.10,.35,1}`, `theta={0,22.5,45,67.5,90}` degrees, `phi={0,90,180}` degrees, `tr G=1` and `N0=.01`. Axial and isotropic samples include repeated physical Gram matrices: 75 means **grid cells**, not independent radio channels.

At fixed G and noise, keep the source's *strict interval rule*: **A is better iff U_A + 1e-12 < L_B**, B better iff `U_B+1e-12<L_A`, otherwise undecided. New bounds intersect the original nearest-competitor lower and all-pair union upper; they cannot weaken any preexisting decision.

The new **16-PSK two-halfspace wedge bound** is especially effective. Let `g=v†Gv>0`, `A=sqrt g`, `sigma=sqrt(N0/2)` and `p=Q(A sin(π/16)/sigma)`. The error event is the union of two Gaussian halfspaces, each with probability `p`. Their intersection entails a *longitudinal reversal* `X<-A`. Thus

\[
\boxed{\max(0,2p-Q(A/\sigma))\le SER_{\rm16PSK}\le\min(1,2p).}
\]

This uses no quadrature or Monte Carlo and can be extremely sharp at high SNR.

For beamformed square **16-QAM**, average-energy normalization gives the *exact ideal-model formula*

\[
SER_{\rm16QAM}
=1-\left[1-\frac32Q\!\left(\sqrt{\frac{g}{5N_0}}\right)\right]^2.
\]

For QPSK² with `G_{12}=0`, the two received port channels decouple and an exact four-real-decision expression applies. These analytic bounds use a `1e-12` numerical guard, but the float evaluation is **not a formally verified directed-interval computation**. New decisions have gaps vastly above that guard (smallest observed margin approximately `4.79e-5`).

| Comparator | Old A / B / undecided | New A / B / undecided | Newly resolved |
|---|---:|---:|---:|
| Fixed QPSK² | 18 / 46 / 11 | 18 / 46 / 11 | 0 |
| Fixed circular 16-PSK | 45 / 18 / 12 | **49 / 18 / 8** | 4 |
| CSI equal-port 16-PSK | 43 / 24 / 8 | **45 / 24 / 6** | 2 |
| CSI eigen-beam 16-PSK | 33 / 30 / 12 | **45 / 30 / 0** | 12 |
| CSI eigen-beam 16-QAM | 15 / 59 / 1 | **16 / 59 / 0** | 1 |

**Total: 19 newly certified decisions; zero reversals of previously certified decisions.** A means rank-one, B means comparator. The beamformed oracles have transmitter CSI and their power/peak constraints differ; they are *not* global power-matched optima.

## 4. High-SNR boundary versus finite N0 grid

At the strict `M_G` criterion, rank-one versus QPSK² gives **18 better / 50 worse / 7 tied or unresolved**, while rank-one versus fixed circular 16-PSK gives **45 better / 30 worse / 0 tied**. These are eventual high-SNR *distance* regions.

At the finite `N0=.01` bound-certified rule, there are **four circular-codebook cells** whose strict finite-noise classification disagrees with the eventual high-SNR ordering. This is expected: minimum distance alone does not order full SER at every finite SNR. The other comparisons are either consistent or abstain; no mathematical transport between different SNR regimes is assumed.

## 5. Tests, hashes and remaining gate

**19/19 new CPU regression tests pass**. They independently enumerate all received 16-point distances against the quarter-phase envelope on random Gram matrices, test exact diagonal thresholds, check 75 archived cells, run Gaussian-AWGN sample sanity checks, test 16-QAM algebra, verify the 16-PSK wedge bounds, prove-by-counterworld that H and Gram distances differ under receive-unitary transformations, and confirm isolated subprocess/dependency failure handling. The newest test verifies the exporter schema, SHA-256 manifest and no-GPU validation.

Git blob hashes matched the **actual tested local source**:

- `gram_boundary_tightening.py`: `7ffb3655428f62abd461f41664a7c6599b552dc3`
- `munich_cpu_runner.py`: `eb1ddd869e64fee76e408755e42961b0825a933d`
- `test_regen_gram.py`: `aa3d795f9d804eb2799591c7b5891e48bfc80ddf`

Input 75-case JSON SHA256: `16ad7ca1eda73d9c6a73776555be2f93333d248ad44d11a32ac9be3b05d2f4c6`. Full tightened output JSON SHA256: `910c533b298c2ba9d790dd259017832819a0f0732492911e4d385bc8ccfcb2c6`. The compact count ledger and blocked-attempt manifest are committed, while the full 75-cell JSON can be regenerated from the saved source sweep.

**Evidence:** P = exact phase-orbit / Gaussian identities conditional on the declared model; M = finite 75-cell computation and local test executions; H = source-matched real Munich predictions and transfer to other radio scenes; R = original unique null-cone causal explanation already retired. No new physical channel data were obtained.

## Next smallest permissible action

Use a **network-enabled, CPU-only** sandbox or local machine with a fresh Python environment to install `sionna-rt==2.2.0`; record the resolved exact Mitsuba/Dr.Jit versions and wheel hashes before comparing results. Run the branch's `munich_cpu_runner.py --mode both --wall-seconds 60 --memory-mib 4096`. Only if **deterministic full** exceeds its wall budget and the dependencies work should one run `--mode reduced`, explicitly labelling it `deterministic CPU regeneration, reduced config`. If both full modes produce 16 alive receiver matrices, compute their per-index `||H_d-H_e||_F` and `||H_d†H_d-H_e†H_e||_F` and import **each separate archive** into `gram_decision_sweep.py`. Report printed example matching, per-receiver ratio, angles, bounds and abstentions, without representing either CPU regeneration as original CUDA output.

Until that data gate opens, **all 16 per-receiver slots remain unmeasured**. This is a resource/provenance stop, not a failure of the mathematical boundary.

**No NVlabs submissions or comments; PR #82 unchanged; no GPU/HF credits or paid compute used.**
