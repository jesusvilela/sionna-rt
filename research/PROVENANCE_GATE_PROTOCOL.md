# Frozen channel-provenance boundary audit — 2026-11-05

Scope: existing 16-Munich-antenna NPZ consumers only, no ray tracing, no new geometric identities, no revisions to PR #82, no NVIDIA communications. Run NumPy-only tests on synthetic fixtures, never label those fixtures as recovered Munich matrices.

**Observed code defect to address before authoring a fix:** `gram_decision_sweep.evaluate_receiver_npz` and `moving_frame.load_verified_munich` accept a self-authored `.manifest.json` with matching source_commit/count and self-computed archive SHA256 as if it verifies channel origin; existing `test_archive_provenance_gate_rejects_fake_and_accepts_typed_synthetic_fixture` explicitly demonstrates the fixture is accepted. The provenance conclusion is stronger than its evidence. The exporter also merges caller metadata after source-derived fields, permitting reserved-key overrides.

**Required behavior**:
1. Accept an archive for *numeric diagnostic analysis* only after checking exact 16-slot shape and complex dtype, all-alive mask, Frobenius normalization, source receiver coordinate order, optional raw-channel normalization agreement, source commit and CPU mode consistency, SHA256 and metadata. Reject altered/missing payloads, index changes, and raw/normalized mismatch.
2. For `mode=test-fixture`, reject by default; allow only via an explicit `allow_synthetic_fixture` flag, with output marked `synthetic_fixture_only`.
3. For `mode=from-npz`, reject by default; allow only with `allow_unverified_import`, marked `unverified_import`, and never silently relabel as original 16 Munich channels.
4. For `regenerate-default`, `regenerate-deterministic`, `exact_notebook_best_effort`, `deterministic_cpu`, `deterministic_cpu_reduced_config`, allow source-structured CPU *candidate* data with typed mode and `candidate_not_independently_authenticated`. No fabricated physical-truth promotion. Require raw-vs-normalized consistency and correct reduced settings when relevant.
5. Preserve existing geometry/Poincare/Gram math and per-receiver analysis, but return provenance class and true limitations with every result, never a bare `evaluated` status that implies real data. Stop if compatibility breaks with the authentic CPU-runner schema.
6. Reject caller `archive_export` metadata overriding reserved provenance keys. Do not rely on manifests to establish external provenance: a malicious author can synthesize self-consistent payloads and hashes. External capture logs, a source runtime/scene record and independent attestation are separate future requirements.
7. Use an adversarial fixture with all valid checksums and 16 normalized synthetic channels as a negative control. A checksum-only gate must be shown insufficient before treating a file as independently supported.
8. CPU-only, no dependency installs, no GPU/ZeroGPU/HF credits, no Work. Stop after strict importer tests, a minimal code change and one correction note. Explicitly maintain 0/16 verified real Munich matrices.

Primary gate: **zero synthetic/unverified archives mislabeled as independently authenticated Munich channel observations**; acceptance of authentic-schema CPU-runner outputs only as provenance-labeled candidates, and preservation of all numeric values/decisions under the new loader. If failed, do not claim the evidence gate is repaired.
