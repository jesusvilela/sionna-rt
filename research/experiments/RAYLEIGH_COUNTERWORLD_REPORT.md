# A Rayleigh counterworld for the null-cone constellation

**04 November 2026 | NVIDIA Sionna RT PR #82 | CPU-only, preregistered synthetic channels**

**What changed:** the actual 16-symbol rank-one and QPSK² constructors from [NVlabs/sionna-rt PR #82](https://github.com/NVlabs/sionna-rt/pull/82) were evaluated in an *independent*, explicitly defined synthetic fading ensemble. QPSK² achieved lower measured SER under both frozen isotropic Rayleigh noise settings. The result is a **channel-distribution counterexample** to universal rank-one superiority, **not** a rerun or contradiction of the original ray-traced scene measurements.

The original interpretation that the determinant/null-cone constraint *causes* the earlier observed SER gain had already been retired by resource-matched optimization; that retirement remains in force. This new control gives further reason not to assign a universal physical effect to the (2,2) geometry.

## 1. Protocol and provenance

A plain-text **pre-registration was written locally before executing the numerical experiment**, then committed here along with its exact source, the raw JSON, and independent deterministic tests. GitHub commit timestamps are *after* experiment execution; they do not themselves establish prospective registration.

- [Frozen protocol](./PRE_REG_2026-11-04.txt)
- [Reproducible NumPy experiment](./rayleigh_counterworld.py)
- [Raw results, not summary-only](./rayleigh_results_2026-11-04.json)
- [Seven separate CPU regression checks](../tests/test_rayleigh_counterworld.py)

**Execution environment:** Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, CPU only. No Sionna RT, Mitsuba, GPU, Hugging Face job, paid Space, NVIDIA upstream PR or physical measurements were used. The tests execute the exact PR #82 codebook constructors and check their normalization. The frozen seed is 74283; 512 independent channels and 256 uniformly sampled symbols per channel per condition (131,072 symbols), with common synthetic channels, transmitted symbol indices and noise draws for the candidate codebooks.

**Models:** coherent maximum-likelihood (minimum Euclidean distance), known fixed 2×2 channel per block, per-dimension receiver noise proper CN(0,N0). The fixed codebooks are:

- **Rank-one**: 16 unit-energy source symbols, q(x)=Im(conj(x1)*x2)=0.
- **QPSK²**: 16 unit-energy source symbols, fixed Cartesian product from PR #82.
- **Off-cone twin**: D C_rank1 with D=diag(1,i); still 16 unit-energy symbols, generally q(x)≠0.

The two previously frozen synthetic channel regimes are:

1. **Isotropic Rayleigh**: independent H_ij ~ CN(0,1/2) (two receive rows); the input and output dimensions are each two.
2. **Rotated anisotropic**: H=diag(1,sqrt(0.1)) U, U Haar-uniform in U(2). This family is *still right-unitarily invariant in distribution* because U is Haar. Output singular values are fixed but its input orientation is randomized.

Neither ensemble substitutes for Sionna RT's ray-traced H distributions.

## 2. Frozen numerical results [M — finite Monte Carlo]

The primary outcome is SER(rank-one) minus SER(QPSK²); positive means the fixed rank-one book is worse. Confidence intervals are 2,000-resample **channel-cluster** percentile bootstraps; 256 symbols in one fading block are not treated as 256 independent channels.

| Channel family | N0 | Rank-one SER | QPSK² SER | Difference, percentage points | Paired 95% interval, pp |
|---|---:|---:|---:|---:|---:|
| i.i.d. Rayleigh | 0.04 | 8.3542% | 6.4354% | +1.9188 | [+1.4724, +2.3659] |
| i.i.d. Rayleigh | 0.08 | 18.9911% | 15.9416% | +3.0495 | [+2.5466, +3.5683] |
| rotated anisotropic | 0.04 | 11.3380% | 8.9027% | +2.4353 | [+2.0096, +2.8443] |
| rotated anisotropic | 0.08 | 27.7046% | 24.0334% | +3.6713 | [+3.1479, +4.1993] |

No hypothesis-sign or noise values were chosen after viewing outcomes; only the two prespecified N0 values and two regimes were reported. These finite-sample intervals assess Monte Carlo sampling of these **synthetic ensembles**, not transfer uncertainty to genuine urban radio propagation.

The off-cone twin offers an important epistemic challenge:

| Channel family | N0 | Rank-one minus off-cone twin, pp | Unadjusted 95% interval, pp |
|---|---:|---:|---:|
| i.i.d. Rayleigh | 0.04 | +0.1663 | [-0.1526,+0.4906] |
| i.i.d. Rayleigh | 0.08 | +0.0031 | [-0.3808,+0.3723] |
| rotated anisotropic | 0.04 | -0.4898 | [-0.8172,-0.1427] |
| rotated anisotropic | 0.08 | +0.1358 | [-0.1847,+0.4875] |

**Theorem below:** the exact ensemble expectations of rank-one and its fixed unitary twin are equal for **both** channel families. The single 95% interval excluding zero is therefore a finite-sample miscoverage in one of four unadjusted comparisons, not evidence of geometry-induced performance. We retain it, do not select another seed, and do not promote a contradiction to the exact symmetry. Our cyclic antithetic control reproduces exact identity for a group-balanced set of channels with common noise.

### Independent detector oracle

For H=I_2 and QPSK², the known coherent-AWGN SER is

    SER = 1 - [1-Q(1/sqrt(2*N0))]^4.

At N0=.08 the exact formula gives 0.02460826; the independent 120,000-symbol detector check gives 0.02375 (within the prespecified six-standard-error plus numerical tolerance).

## 3. A representation-independent Rayleigh diversity theorem [P — elementary proof, known theory]

Let C⊂C² be any **finite constellation of M>=2 distinct symbols**, H∈C^{r×2} have independent entries CN(0,1/r), and the receiver know H exactly. There is one transmitted vector per independent block (no channel-time coding); receiver noise is CN(0,N0 I_r), with equiprobable symbols and coherent ML decoding.

**Claim:** averaging over H and noise, the symbol-error rate is bounded above and below by positive multiples of N0^r as N0 tends to zero. In notation,

    E_H[SER(N0|H)] = Θ(N0^r).

**Derivation:** For a pair Δ=x_i-x_j≠0, each component of HΔ is CN(0, ||Δ||²/r), independently across receive rows. Hence ||HΔ||²/(||Δ||²/r) has a Gamma(r,1) distribution. Conditional Gaussian pairwise error is

    P(i→j|H) = Q( sqrt( ||HΔ||²/(2*N0) ) ).

Craig's identity and the Gamma Laplace transform yield the *exact averaged pairwise error*

    Pbar_ij(N0) = (1/π) ∫_0^(π/2)
                      (1 + ||Δ||²/(4*r*N0*sin²θ))^(-r) dθ.

At N0→0,

    Pbar_ij(N0) ~ (binom(2r,r)/2) * (r*N0/||Δ||²)^r.

A fixed pairwise event that makes a wrong codeword closer to the observation implies full-ML error. Therefore, for any fixed i≠j,

    (1/M)*Pbar_ij <= E[SER] <= (2/M)*Σ_{i<j} Pbar_ij.

Both sides have strictly positive finite coefficients times N0^r. This proves **diversity order exactly r** for every such distinct finite codebook, including rank-one, QPSK² and the off-cone twin. The prefactor of the true SER is not identified by these bounds.

**Non-transport:** this result assumes i.i.d. Gaussian channel entries and independent receive rows. It does not assert an identical exponent in rank-deficient, correlated, spatially structured, ray-traced, multi-tap or poorly calibrated channels. The mathematical argument is well-established MIMO/PEP theory, not a new theorem of 6G.

## 4. Exact distance-spectrum certificate and union-bound constants [P]

Every source codebook has 120 unordered symbol pairs. Direct finite enumeration yields:

| Euclidean squared distance | Rank-one count | QPSK² count |
|---|---:|---:|
| 2-sqrt(2) ≈ 0.585786 | 16 | 0 |
| 1 | 0 | 32 |
| 2 | 80 | 48 |
| 3 | 0 | 32 |
| 2+sqrt(2) ≈ 3.414214 | 16 | 0 |
| 4 | 8 | 8 |

Both have Σ_{pairs} ||Δ||²=256; they do **not** have identical distributions. In the i.i.d. channel law, averaged PEP and its union bound depend only on these Euclidean squared distances, not on how the constellation is positioned relative to a preferred polarization basis.

The unclipped **Rayleigh pairwise union bound** is

    B_C(N0) = (2/M) * Σ_{i<j} Pbar_ij(N0).

For r=2 the leading term is (3/2) N0² Σ_{pairs} (||Δ||²)^(-2). Direct algebra gives

    Σ_rankone 1/(||Δ||²)^2 = 137/2,
    Σ_qpsk²  1/(||Δ||²)^2 = 865/18,

so the **upper bounds**, not the actual SER curves, satisfy

    B_rankone(N0) ~ (411/4) N0² = 102.75 N0²,
    B_qpsk²(N0)   ~ (865/12) N0² ≈ 72.0833 N0².

These do not prove a universal ordering of actual SER at every finite SNR, and the union bound is not asserted to have the true leading SER prefactor in fading. The numerical Craig quadrature was independently checked against SciPy integration, as well as Monte Carlo bounds.

At N0=.04 the evaluated unclipped union bounds are 0.09510437 (rank-one) and 0.07895131 (QPSK²). At N0=.08 they are 0.26242135 and 0.23430559, respectively. For the off-cone unitary twin, both bounds equal the corresponding rank-one bounds to numerical precision.

## 5. Exact unitary-twin covariance and its test [P]

Let D=diag(1,i), C_twin=D C_rankone. For either *right-unitarily invariant* channel ensemble listed above, H and HD have the same probability distribution. Therefore

    E_H[SER(H, C_twin)] = E_H[SER(HD, C_rankone)]
                        = E_H[SER(H, C_rankone)]

at every N0 and under exactly the same proper AWGN decoder. This is a **law-level equality**. It does *not* imply that the two SER measurements must coincide in a finite sample with common fixed H.

The new metamorphic regression takes each base H and its four-channel cyclic orbit

    {H, H D, H D², H D³}.

With identical transmitted indices and noise draws at each member, its *aggregate* rank-one error count agrees exactly with the aggregate off-cone count, because the twin shifts the orbit index by one. This verifies the machinery without statistical approximations; these orbit members are deliberately dependent and are **not** counted as independent confidence-interval samples.

## 6. Reproduction and evidence ledger

From repository root:

    python research/experiments/rayleigh_counterworld.py
    python -m unittest discover -s research/tests -p 'test_rayleigh_counterworld.py' -v

The first command writes the result JSON next to the script. **This execution passed the prespecified known-answer check and produced all four results. All 7/7 deterministic CPU tests pass**, including an independently evaluated Craig integral, exact spectra, high-SNR coefficients, source-codebook normalization, and a group-orbit metamorphic check. Source/test/result/prereg files were all confirmed identical to the locally executed/created versions by Git blob SHA, not filename similarity.

- **P:** exact Rayleigh PEP integral, diversity order r under stated assumptions, distance-spectrum enumerations, off-cone expected SER symmetry.
- **M:** the four frozen synthetic-ensemble Monte Carlo outcomes, their channel-cluster intervals, and the detector checks.
- **H:** geometric or codebook features predicting source-matched, held-out ray-traced performance better than simpler channel spectral features; untested here.
- **R:** the previously retired claim that the fixed null-cone determinant uniquely *causes* the earlier Sionna SER advantage.
- **No claim:** no newly proven mathematical novelty, no demonstrated broad 6G superiority, no Sionna GPU reproducibility, and no independently verified physical channel.

Classical foundations: Craig-formula averaged pairwise error, coding gain and diversity analysis have a substantial pre-existing communications literature, including Lindell, *Some exact union bound results for maximum likelihood detection in MIMO systems* (ISIT 2005, DOI 10.1109/ISIT.2005.1523742), and Taricco–Viterbo, *Performance of high-diversity multidimensional constellations* (IEEE Transactions on Information Theory, 1998, DOI 10.1109/18.681327). Their settings are related but not asserted to be identical to the frozen codebooks here.

### Next gate, not executed

The load-bearing missing bridge is **the actual held-out Sionna ray-traced H distribution**. To support a physical-channel mechanism, obtain a reproducible, versioned H sample from the existing notebook, freeze its scene/channel sampling protocol, and compare QPSK², rank-one, off-cone twin and an equally optimized unconstrained 16-symbol control on held-out matrices, all under matched compute, power, CSI, noise, symbol priors, seeds, and per-scene uncertainty. The primary test is out-of-sample SER and **incremental predictive value** of geometric features over standard Gram/eigenvalue/distance features. Retire any proposed special-geometry causal interpretation if the unconstrained matched control equals or exceeds it. Do not start a GPU job or upstream PR without that gate.

**Authorship:** human-led geometric research direction and evidence constraints; this reconstruction, synthetic experiment implementation, theorem proof and write-up are AI-assisted. No claim of independent human mathematical review.
