# Frozen frame-resource covariance audit — 5 November 2026

**Scope.** Audit the existing, *already frozen* 75-cell Gram sweep and one previously printed channel. Do not add geometric objects, refit the Bargmann ablation, claim new propagation data or change the notebook. This is a coordinate-invariance / engineering-constraint reproducibility check.

## Exact objects and transformations

For each Hermitian source channel Gram G, 16-point transmit codebook C and source physical port projectors E0=diag(1,0), E1=diag(0,1), change transmitter input coordinates with one or two non-diagonal unitaries U and V:

    G' = (VU) G (VU)^†
    C' = (VU) C
    Ea' = (VU) Ea (VU)^†
    Q' = (VU) sigma_y (VU)^†

The source null functional is q(x)=x^† sigma_y x/2, with sigma_y=[[0,-i],[i,0]]. Physical-port average and per-symbol power constraints are x^†Ea x. For perfect-CSI one-row beam controls, do NOT re-run the original `equal_port_beam(G')` in the new basis, since that function assumes the untransformed coordinate projectors and therefore changes the resource constraints.

## Pre-specified tests and gates

Data: SHA256 `16ad7ca1eda73d9c6a73776555be2f93333d248ad44d11a32ac9be3b05d2f4c6`, 75 cells; noise N0=0.01; no new fitting. Use three fixed constellations, the two rank-one beam controls and the 16-QAM beam control from the existing source; label CSI-aware baselines as non-deployed controls. Compare all 120 received-pair squared distances and all bound intervals L,U and min_d2 in each representation. Use tolerance 1e-10 for distance/bounds and 1e-12 for average and per-symbol physical-port power (absolute).

Frame changes: one fixed nontrivial complex SU(2) mixing with angle 31 degrees, relative phase 0.71 radians, followed by an independent SU(2) mixing with angle 17 degrees, relative phase -0.49 radians. No tuning of mixing matrices after viewing outcomes. Also include independent left-receiver rotations as an optional direct-H verification on the source-printed channel.

Primary gate: zero representation-induced changes to any of the 75 × 5 strict certified comparisons to rank-one, with complete coordinated transport of the physical resource basis and codebooks. At least one deviation retires the claim of existing pipeline's frame covariance until repaired. The bounds themselves must match, not merely the decision labels. Repeat all tests on the single printed channel separately without interpreting it as held out.

Negative control: in each new frame, deliberately **forget** to transport physical E0,E1; recompute equal-port 16-PSK using the old diagonal port basis. Record (1) physical port-power violations and (2) the count of changed Gram-bound decisions relative to the correctly transported equal-port oracle. No predicted count is specified; zero is allowed. **Never treat the naive recomputation as an equivalent physical design.** Avoid confusion with passive gauge changes: the original C and the resource projectors must both transform.

No ML training, no new Sionna runs, no NVlabs posts, no GPU/ZeroGPU/HF/Work, no new physical-data claim. Outcome should be a negative/positive reproducibility statement plus a minimal interface correction if necessary, and an addendum to the already short technical note. Stop after one frozen audit.
