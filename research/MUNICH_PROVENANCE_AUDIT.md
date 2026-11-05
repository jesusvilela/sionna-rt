# Munich channel data: evidence-bearing moving-frame contract

**2026-11-05 · fork-only CPU audit · source commit `57435845d3395b2d5e167b0893035e15ffbdaa0d` · 0/16 recovered Munich channels**

## Verdict

The original per-receiver code correctly computed Gram/constellation diagnostics **conditionally on an input H**, but its NPZ acceptance logic promoted a source-shaped, self-checksummed archive to an ambiguously labeled "evaluated" result. Its own regression harness used **16 synthetic diagonal matrices** and demonstrated that this shape+hash gate accepted test data. That is a software trust-boundary defect, not a new mathematical obstruction.

The repair preserves the full complex channel / polarization frame / conditional Gram decisions while distinguishing a syntactically valid archive from independently authenticated ray-tracing provenance. It adds **no geometric layer and no measured Munich channel**.

## What was repaired

1. [Single source-structured data gate](./experiments/munich_provenance_gate.py): requires 16 complex 2×2 matrices, source-order receiver and transmitter positions, all-alive boolean mask, Frobenius normalization, agreement of the compact and positional H arrays, raw-to-normalized agreement for CPU candidates, SHA256, source commit and recorded mode. It checks the declared shared transmitter VH frame; that declaration is **not an independent calibration**.
2. [Gram receiver evaluator](./experiments/gram_decision_sweep.py): no longer returns a plain "evaluated" status. Every per-receiver table includes an explicit unverified origin assessment and mode, even for CPU-generated candidates. Synthetic fixtures are refused unless testing code explicitly passes `allow_synthetic_fixture=True`. Manually imported NPZ archives require `--allow-unverified-import` and are returned as `unverified_import_only`.
3. [Moving-frame reader](./experiments/moving_frame.py): uses the same source and provenance gate, retains the archive's evidence class in its output, and never says "derived from verified archive" solely from a self-authored SHA.
4. [Exporter](./experiments/export_munich_16.py): forbids caller metadata from overriding reserved scientific evidence fields such as source commit, receiver count, archive hashes and frame declaration.
5. [Budgeted CPU runner](./experiments/munich_cpu_runner.py): labels its declared transmitter source frame in the manifest. Its source-default, deterministic full and deterministic reduced modes remain distinguishable; no solver execution was performed this round.

## Explicit non-equivalence: these statuses must not collapse

| Incoming mode | Default ingestion | Reported evidence |
|---|---|---|
| `test-fixture` | Refused | With test-only opt-in: `synthetic_fixture_only` |
| `from-npz` | Refused | With diagnostic opt-in: `unverified_import_only` |
| `regenerate-default`, `regenerate-deterministic` | Source-structured candidate admitted after raw/back-end/config validation | `cpu_regeneration_candidate_unverified` |
| `exact_notebook_best_effort`, `deterministic_cpu`, `deterministic_cpu_reduced_config` | Source-structured candidate admitted after raw/back-end/config validation | `cpu_regeneration_candidate_unverified` |

No accepted class means “the original 16 Munich matrices were recovered.” A malicious or mistaken actor can construct internally consistent raw and normalized arrays, hashes and metadata. **This gate does not perform cryptographic provenance authentication.** Such a claim would require independently obtained source execution records, environment/scene assets, frame-calibration metadata, retained solver logs or genuinely third-party reproduction.

The source-frame declaration is not a license to rotate the input channel while leaving the transmit codebook and physical port projectors unchanged. The prior [resource-aware moving-frame audit](./FRAME_RESOURCE_COVARIANCE_FINDINGS.md) remains the engineering covariance rule; its 375/375 synthetic decisions are unchanged.

## Verification and evidence ledger

- **P (conditional):** the schema gate prevents a *declared* synthetic fixture from being silently reported as a physical receiver observation; this is a software rule. The Gram algebra is unchanged.
- **M (local):** **12/12 adversarial NumPy unit tests passed** before committing the module, including self-consistent checksum fabrication, wrong receiver order, swapped H payload, incorrect raw normalization, forbidden frame labels, falsely labeled reduced config and candidate-mode compatibility. Both new module and test file were compared against GitHub Git blob hashes after committing.
- **A:** source-reported commit, scene, polarization frame and run mode remain declarations. The numerical validators cannot prove an external process actually generated the data.
- **R:** “a matching self-authored SHA and source-commit string proves these are authentic original Munich propagation matrices.”
- **Not promoted:** neither Sionna scene reproducibility nor a physical constellation advantage. The prior Bargmann prediction advantage remains retired.
- **Integration boundary:** new shared gate is wired into the two existing reader paths and exporter metadata protection; the already existing regression fixtures were updated to require explicit synthetic mode. The local 12 tests are not third-party peer review or a complete upstream CI run.

Run the gate tests from the fork checkout without any Sionna install:

```bash
python -m unittest discover -s research/tests -p 'test_munich_provenance_gate.py' -v
```

**Next smallest step:** keep the printed-channel technical note [as the substantive research kernel](./TECHNICAL_NOTE_SCOPE_AND_NEGATIVE_RESULT.md). If a real CPU regeneration becomes possible, preserve its scene version, solver logs, environment and SHA separately, then analyze it only as a **CPU candidate** until its origin is independently checked. No new holonomy, Berry or 6G superiority assertions are supported.

No Work, GPU, ZeroGPU/HF credits or NVlabs comments/PRs. PR #82 unchanged.
