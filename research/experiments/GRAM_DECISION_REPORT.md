# Source-matched Munich export + Gram-matrix decision gates

**Research checkpoint, 2026-11-04.** Fork-only. No modifications to PR #82, no NVlabs comments/submissions, no GPU or Hugging Face credits. The numerical sweep is CPU-only synthetic **mathematical error bounds**, not a real Munich propagation measurement.

Source frozen at [PR #82 fork head 57435845](https://github.com/jesusvilela/sionna-rt/blob/57435845d3395b2d5e167b0893035e15ffbdaa0d/tutorials/Null-Cone-Constellations.ipynb).

## 1. The actual 16 Munich matrices: not available in this environment

The original notebook constructs the scene and 16 receivers, executes `PathSolver()(scene,max_depth=5)`, sums paths via `reshape(16,2,2,-1).sum(-1)`, excludes channels below Frobenius norm `1e-12`, and normalizes each living receiver separately. Only **one rounded channel** is printed in the committed notebook, not the full H array.

The sandbox CPU has Python 3.13.5, NumPy 2.3.5 and SciPy 1.17.0, but **no sionna, mitsuba or drjit installed**. We did not install packages or start a ray-tracing job.

The new `export_munich_16.py` reproduces transmitter/receiver positions, port arrays, path maximum depth, alive filtering and normalization. It has separate modes:

- `--mode check`: safely report dependencies, no ray tracing.
- `--mode from-npz --input EXISTING.npz`: repackage an existing live notebook's saved `np.savez_compressed(...,H=H)` data with checksums and receiver-index manifest, **no ray tracing**.
- `--mode regenerate-default`: explicitly opt-in to CPU `llvm_ad_mono_polarized`, Sionna RT pinned to 2.2.0, source `PathSolver()` defaults and seed 42. NVIDIA states that **a fixed seed alone does not guarantee deterministic paths**.
- `--mode regenerate-deterministic`: same scene but uses `PathSolver(deterministic=True)` and seed 42. This **changes** the source solver's candidate generator and does not claim byte-for-byte recovery of the original CUDA run.

The exporter never installs Sionna, silently changes hardware variants, modifies the notebook or overwrites an existing output. It stores all 16 positional slots, normalized H, optional raw H, alive mask, receiver locations, source commit, backend/mode, archive SHA256 and a diagnostic distance from the notebook's printed H0.

**Cheapest route:** if an old notebook session still contains the `H` variable, save it there with `np.savez_compressed("existing_munich_H.npz",H=H)`, then run the exporter `--mode from-npz`. Otherwise the smallest external action is CPU install of version-pinned Sionna RT/Mitsuba/Dr.Jit followed by one explicitly opted-in CPU regeneration. Computational feasibility and exact CPU-vs-CUDA numerical matching have **not been established**, so we stopped there.

### Source interpretation versus saved notebook output

The executed PR #82 notebook still prints a categorical explanation to the effect that rank-one wins **because** its worst-channel minimum distance is about twice QPSK²'s. That minimum-distance comparison is a recorded diagnostic, **not** a causal proof: the later matched unconstrained-optimizer control retired the unique null-cone mechanism, and even a larger minimum distance need not order the entire finite-noise SER. We preserve the original notebook unchanged as requested and treat that printed text as an unresolved communication/interpretation correction, **not** a new NVIDIA issue or comment.

## 2. Exact Gram/Stokes predictor [P: elementary algebra]

For a fixed 16-point complex constellation `C`, proper white receiver noise `CN(0,N0 I)` and known 2-by-2 complex H, the **entire conditional ML SER** is determined by `G=H†H` (Gram sufficiency, established in the previous checkpoint).

Let the channel's eigenvalues obey `lambda_plus >= lambda_minus > 0`, define `r=lambda_minus/lambda_plus` and the principal eigenvector

\[
u=(\cos\theta,e^{i\phi}\sin\theta).
\]

Then for **every pair difference** `Delta=x_i-x_j`,

\[
\boxed{
d_G^2(\Delta)=\lambda_- \|\Delta\|^2+
(\lambda_+-\lambda_-)
\left|\cos\theta\,\Delta_1+
e^{-i\phi}\sin\theta\,\Delta_2\right|^2.
}
\]

The equivalent Pauli/Stokes expansion, with `S_1=2Re(conj(Delta_1)Delta_2)`, `S_2=2Im(conj(Delta_1)Delta_2)` and `S_3=|Delta_1|²-|Delta_2|²`, is

\[
d_G^2=\tfrac{\operatorname{tr}G}{2}\|\Delta\|^2
       +(\Re G_{12}) S_1-(\Im G_{12}) S_2
       +\tfrac{G_{11}-G_{22}}{2} S_3 .
\]

**Typed warning:** port-power split corresponds to `S3`, and the original null form `Im(conj(x1)x2)` to one half of `S2(x)`; but the SER geometry involves **S2 of pair differences**, not merely S2 of codewords. The real cross term `S1` is equally indispensable. A ratio-only or null-form-only predictor is not sufficient.

For fixed H and codebook, Gaussian pairwise probability is `p_ij=Q(sqrt(d_G²/(2N0)))`. For each i, let `p_i^*=max_{j !=i}p_ij`. Elementary nearest-neighbor lower and union upper bounds give

\[
L_C(G)=\frac1{16}\sum_i p_i^*,\qquad
U_C(G)=\frac1{16}\sum_i\sum_{j\ne i}p_{ij},
\quad L_C\le {\rm SER}_C\le U_C.
\]

Thus our **certified three-way decision rule** is:

\[
\mathrm{rank1\ wins}\quad\Longleftarrow\quad
U_{\rm rank1}+10^{-12}<L_{\rm other};
\quad
\mathrm{other\ wins}\quad\Longleftarrow\quad
U_{\rm other}+10^{-12}<L_{\rm rank1}.
\]

If neither inequality holds, **abstain**. This is a sufficient, *not necessary*, decision rule; there is no claimed one-scalar sharp boundary for arbitrary G. It rejects exact channel collisions and leaves tie-aware analysis to the earlier discriminant research.

## 3. Design-agnostic references and resource boundaries

In addition to fixed PR #82 rank-one, QPSK², and the previously defined **off-cone, equal-port circular 16-PSK**, the code includes:

1. **Equal-port, channel-steered 16-PSK** on `v=(1,exp(-i arg G12))/sqrt(2)`. For a 16-PSK beam, this **maximizes `v†Gv` under per-port power exactly 1/2**; it is a closed-form *within-family optimum*, with perfect transmitter knowledge of G.
2. **Eigen-beamformed 16-PSK** on the principal eigenvector, maximizing channel gain among all unit-energy single-beam 16-PSK. **Per-port power may be imbalanced**, so it is a distinct relaxed-resource oracle.
3. **Eigen-beamformed 16-QAM** with **average total symbol energy one** and peak symbol energy 1.8. This is an additional design-agnostic reference, *not* globally optimal among all 16-point codebooks. Peak and per-port resource restrictions differ.

None of these channel-aware oracles is an ordinary fixed/no-CSI deployed baseline. No equality of implementation cost or channel acquisition overhead is claimed.

## 4. CPU-only sweep: 75 frozen cells, N0=0.01, tr(G)=1 [M: finite calculation of P-bounds]

Frozen grid: `r={.01,.03,.10,.35,1.0}`, `theta={0,22.5,45,67.5,90}` degrees, `phi={0,90,180}` degrees. Values at isotropy/axis endpoints repeat the same physical G; the 75 cells are **not 75 independent propagation draws**.

**All-pair Gaussian bounds, number of cells in each class:**

| Comparator | Rank-one certified better | Comparator certified better | Bounds overlap / undecided |
|---|---:|---:|---:|
| Fixed QPSK² | 18 | 46 | 11 |
| Fixed off-cone circular 16-PSK | 45 | 18 | 12 |
| CSI-steered equal-port 16-PSK | 43 | 24 | 8 |
| CSI eigen-beam 16-PSK | 33 | 30 | 12 |
| CSI eigen-beam 16-QAM | 15 | 59 | 1 |

**Selected synthetic cells (rank-one versus two fixed alternatives):**

| r | theta | phi | Rank-one error [L,U] | QPSK² [L,U] | Circular 16-PSK [L,U] | Strict conclusions |
|---:|---:|---:|---:|---:|---:|---|
| .01 | 0° | 0° | [.12415,.19678] | [.24084,.64154] | [.02553,.05120] | Rank-one beats QPSK²; circular beats rank-one |
| .01 | 22.5° | 90° | [.00289,.00536] | [.00280,.00570] | [.00557,.01113] | Rank-one beats circular; undecided vs QPSK² |
| .10 | 0° | 0° | [.00519,.00778] | [.01650,.03429] | [.02553,.05120] | Rank-one beats both |
| .10 | 45° | 0° | [.00519,.00778] | [.00096,.00129] | [.02553,.05120] | QPSK² beats rank-one; rank-one beats circular |
| .10 | 45° | 90° | [.00128,.00142] | [.00096,.00129] | [.00426,.00852] | Rank-one beats circular; undecided vs QPSK² |
| .35 | 22.5° | 0° | [.00150,.00153] | [.000024,.000049] | [.02553,.05120] | QPSK² beats rank-one; rank-one beats circular |
| 1.0 | 45° | 0° | [.0000649,.0001298] | [.000000287,.000001147] | [.02553,.05120] | QPSK² beats rank-one; rank-one beats circular |

**Counterexample to a ratio-only rule:** the two `r=.10, phi=0°` rows with `theta=0°` and `theta=45°` have the **same Gram eigenvalues**, yet their certified rank-one-versus-QPSK² orderings are opposite. The principal-eigenvector orientation matters. The full output also has genuine abstentions rather than auto-promoted winners.

## 5. Reproducibility and evidence

Files:

- `research/experiments/export_munich_16.py` — safe CPU exporter/importer + manifest.
- `research/experiments/gram_decision_sweep.py` — source-matched codebooks, Stokes/Gram equality, all six controls, certified interval rule, 75-cell deterministic sweep, and optional 16-receiver import.
- `research/tests/test_export_munich_16.py` — 11 CPU tests with mocked Sionna scene/solver and temporary saved data.
- `research/tests/test_gram_decision_sweep.py` — 16 CPU tests including independent direct-H and Stokes formulas, unitary frame invariance, source printed-channel error-bound anchor, oracle family optimality, spectrum degeneracy, no false decisions, and checksum rejection.
- `research/experiments/SYNTHETIC_SWEEP_PREREG_2026-11-04.md` — frozen numeric gate and scope, with explicit warning that commit occurred *after* first exploratory execution.

Commands from the fork checkout:

```bash
python -m unittest discover -s research/tests -p 'test_export_munich_16.py' -v
python -m unittest discover -s research/tests -p 'test_gram_decision_sweep.py' -v
python research/experiments/export_munich_16.py --mode check
python research/experiments/gram_decision_sweep.py --out gram_sweep.json
# Only after an actual existing H export is available:
python research/experiments/gram_decision_sweep.py --channels-npz munich_16_channels.npz --out munich_gram_results.json
```

**Observed 2026-11-04 CPU run:** **27/27 tests pass** in this isolated container. The exact full JSON, generated without randomness, has SHA256 `16ad7ca1eda73d9c6a73776555be2f93333d248ad44d11a32ac9be3b05d2f4c6`. A compact 75-cell CSV of its bounds and decisions has SHA256 `49d5c49ff16b381f843745a106b031209b845b8201e3bbf30dd2e6fe82d07ca6`. Both are reproducible by the committed code; neither is a real channel dataset. **The 16-receiver assessment is not available**, and no aggregate Munich validation is asserted.

### Evidence and stop

- **P:** exact Gram/Stokes pairwise-distance identity; conditional Gaussian PEP and bound-based certified ordering; analytic within-family beam-optimal directions under stated power restrictions.
- **M:** 75-cell finite synthetic sweep and 27 self-authored regression tests; not independent third-party evidence.
- **H/open:** same decision behavior on the actual 16 Munich H matrices or held-out physical channels; source-matched multipath / phase calibration.
- **R:** a unique causal performance gain due to the fixed null-cone determinant, already retired by earlier matched unconstrained controls.

**Next smallest step:** capture `H` from the *existing* PR #82 notebook if available, write `np.savez_compressed("existing_munich_H.npz", H=H)` without rerunning ray tracing, package it with the exporter, and execute `--channels-npz`. Inspect receiver-indexed strict decisions against measured source-setup SER and source-matched equal-budget unconstrained controls. The synthetic ratio/angle grid **cannot** substitute for those physical matrices.

No NVlabs communication or PR, no GPU/ZeroGPU/Hugging Face billing, no new ray tracing or model installation was performed.

*Provenance:* human-specified geometric research programme and constraints; AI-authored derivation, code and tests. No novelty or third-party reproducibility claims.
