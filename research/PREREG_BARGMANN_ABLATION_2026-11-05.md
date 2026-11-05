# Frozen synthetic test: does Bargmann B improve held-out prediction?

Date: 2026-11-05. Frozen in the fork BEFORE feature construction, model fitting or inspection of held-out outcomes. One confirmatory comparison; no hyperparameter search or alternate target after seeing results.

## Data and exact source
Source: archived `gram_sweep_2026-11-04.json` from the 75-case Gram sweep; SHA-256 `16ad7ca1eda73d9c6a73776555be2f93333d248ad44d11a32ac9be3b05d2f4c6`. Each case contains G, r, theta, phi, and the fixed bound-certified rank-one-vs-circular-16PSK judgment. This is synthetic deterministic material, NOT 16 Munich channels, and should never be called a wireless measurement.

Exclude all 15 isotropic r=1 cases, since the Gram matrix has no uniquely defined principal eigenline and B is not physically identifiable there. The remaining 60 cases form 20 groups/triples indexed by (r, theta), with three relative input phases phi=(0,90,180) in fixed order. A triple is an *artificial adjacency convention*, not a measured receiver triangle. Preserve exact group identity and labels; never randomly split individual nodes.

## Target and split
Primary target: for each of the 60 nondegenerate cases, binary `1` iff the ARCHIVED strict union-upper < comparator lower certified rank-one better than **fixed circular 16-PSK**; `0` otherwise. Label 0 includes certified comparator better and abstentions: it must not be reported as actual comparator superiority.

Training groups: theta in {0,45,90} degrees for each of four r values {.01,.03,.1,.35}, i.e. 12 groups, 36 nodes. Held-out groups: theta in {22.5,67.5} for each r, i.e. 8 groups, 24 nodes. No fit, scaling, threshold, or feature selection on held-out labels.

## Physically typed pairwise baseline and B increment
For each row, reconstruct principal rank-one projector P from its full G (not from a phase-chosen eigenvector). Choose fixed transmitter polarization reference projectors E0=|e1><e1|, Ex=|(1,1)/sqrt2><...|, Ey=|(1,i)/sqrt2><...|. The baseline features are:
  (i) local eigenvalue ratio r;
  (ii) the three **pairwise projective overlaps** tr(P E0), tr(P Ex), tr(P Ey), which anchor orientation to the FIXED transmit-codebook coordinates;
  (iii) the two pairwise overlaps tr(P P_other) for the other two nodes in the declared (r,theta) triple.
The reference overlaps are required: omitting the fixed codebook basis would give B an artificial advantage on a target that depends on absolute transmitter orientation. This baseline can reconstruct local P and r, and hence full trace-normalized G.

The increment adds exactly two features, Re B and Im B, where B=tr(P_phi0 P_phi90 P_phi180), shared within its triple. No new geometric object or identity is introduced.

Both models: one scikit-learn `StandardScaler()` fitted on TRAIN only, then `LogisticRegression(C=1.0,solver="lbfgs",max_iter=2000,class_weight=None)`, same optimizer and probability mapping; no tuning, no resampling class balance, no feature engineering from labels. Additional features cost is counted, not hidden.

## Metric, uncertainty and decision
One primary score: held-out **Brier loss** mean[(predicted probability - binary label)^2], lower better. Define delta = Brier(B-added) - Brier(pairwise-baseline).
Bootstrap 5000 replicates of the 8 INDEPENDENT TEST GROUPS with replacement, retaining all three within-group rows; fixed RNG seed 6105; report percentile 95% interval for delta. No independent-channel inference is licensed by the bootstrap; it quantifies sensitivity to these eight synthetic grid groups only.

For B to earn a *provisional synthetic predictive-usefulness* designation BOTH must hold:
  (a) delta <= -0.02 absolute Brier points;
  (b) upper endpoint of the 95% grouped-bootstrap interval is strictly < 0.
Otherwise answer **NO** and explicitly RETIRE the proposed engineering value of B for synthetic design-certification prediction. The mathematical identity remains valid classical Pancharatnam/Bargmann geometry. Even a YES does not warrant a real-wireless claim without source-matched channels.

## Hard stop and controls
- Audit source SHA256 and exact 75/60/36/24 counts before modeling.
- Assert no (r,theta) group overlap across train/test.
- Assert projectors Hermitian/idempotent and baseline anchor overlaps in [0,1].
- Assert B is consistent under one common global U(2) reframe of all projectors **and the three fixed anchors**, to guard coordinate artifact.
- Treat theta=0/90 repeated phi projectors as duplicates *within their one group*, not independent channels.
- Do not try alternate model, target, split, threshold, or larger geometry in this round regardless of outcome.
- No NVlabs comments/PR, no Sionna CPU regeneration unless independently necessary, no GPU/HF or Work.
- Finish by publishing a short, accurate technical note focused on: PR #83 verified reliability merge; one printed H; 75 synthetic Gram cases; 0/16 measured Munich H; source-matched baseline; retired causal and predictive claims.
