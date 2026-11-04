# Frozen design gate — synthetic Gram sweep, 2026-11-04

Source-matched constellations: PR #82 fork head `57435845d3395b2d5e167b0893035e15ffbdaa0d`.
No actual Sionna channels are present; this sweep **cannot** validate ray-traced performance.

- Fixed single-channel coherent complex AWGN with `N0=0.01`, known 2x2 positive-definite Hermitian Gram G, tr(G)=1.
- Exactly 75 indexed grid cells: eigenvalue ratios `r=(.01,.03,.10,.35,1.0)`, principal theta in degrees `(0,22.5,45,67.5,90)`, relative phi `(0,90,180)`. Axis and isotropic cells may represent repeated physical G; counts are **grid counts**, not IID samples.
- Original fixed source codebooks: rank-one and QPSK-squared (16 points, unit energy); fixed off-cone circular 16-PSK (16 points, equal half-power ports).
- Design-agnostic CSI-aware references: (a) equal-port 16-PSK with analytically optimal relative phase, (b) principal-eigenvector 16-PSK with arbitrary port split but unit total symbol energy, (c) principal-eigenvector 16-QAM with unit **mean** total symbol energy (peak 1.8). These are **family-optimal controls**, not global codebook optimizers; perfect transmitter CSI changes the design resource theory.
- For each codebook, compute received pairwise squared distances by G and Gaussian pairwise error `Q(sqrt(d2/(2*N0)))`; derive symbol SER lower bound L=average largest pairwise error probability and raw union upper bound U=average sum of all pairwise probabilities. No Monte Carlo or clipping.
- **Primary classification:** certify rank-one better than comparator **only** if its U+1e-12 < comparator L; certify worse only if comparator U+1e-12 < rank-one L; otherwise `undecided`. No effect-size rethresholding or post-hoc tuning.
- **Counterexample/kill:** if a purported decision is based solely on eigenvalue ratio or one Stokes component while varied theta/phi reverses it, reject the scalar threshold. If a purported oracle is called globally optimal, retire that overclaim immediately. No physical-channel claim from synthetic G.
- **Data gate:** no per-receiver classification unless a 16x2x2 normalized H archive with accompanying exporter manifest and SHA-256 is supplied. One rounded printed receiver does not qualify. A deterministic solver regeneration does not reproduce the notebook's nondeterministic default path generator bit-for-bit.
- Source scripts and tests were developed before the final frozen run, but this registration was committed **after** initial exploratory execution; do not claim third-party independent prospective registration.
- Stop: no Sionna install/run here, no GPU/HF credits, no NVlabs comments/PRs, no broad hypercomplex lift until held-out real channels.
