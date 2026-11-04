# Phase-frame covariance: what the null-cone experiment does (and does not) show

**Research checkpoint — 2026-11-04.** Fork-only analytical audit of [NVlabs/sionna-rt PR #82](https://github.com/NVlabs/sionna-rt/pull/82); **not** a request to merge a new physical theory. It preserves the later [matched-control correction](https://github.com/NVlabs/sionna-rt/pull/82#issuecomment-6491527495) that retired the claim of a *causal* advantage from the determinant constraint.

## Typed objects

- Transmitted dual-polarized symbol: column vector `x=(x1,x2) in C^2`.
- Channel in a fixed transmit/receive representation: `H in C^{r x 2}` (the notebook uses `r=2`).
- Symbol codebook: finite `C subset C^2`; received candidates: `H C`.
- Chosen realification: `X(x) = [[Re x1, Im x1], [Re x2, Im x2]]`.
- *Frame-sensitive* null form: `q(x) = det X(x) = Im(conj(x1)*x2)`.
- Independent transmit-component phase-frame transformation: `D = diag(exp(i*a), exp(i*b)) in U(1)^2`; coordinates `x' = D x`, channel `H' = H D^{-1}`.

The last line is a **joint coordinate/frame change**. If the physical channel `H` is held fixed while only the transmitted symbols are modified, that is a different, potentially physical codebook design.

## Proposition A — the null form is not phase-frame invariant [P: elementary derivation]

Writing `z = conj(x1)*x2`, a relative phase change gives

```text
z' = exp(i*(b-a)) * z
q(Dx) = Im(z') = sin(b-a)*Re(z) + cos(b-a)*Im(z).
```

For `x = (1,1)/sqrt(2)` and `D = diag(1,i)`, `q(x)=0` but `q(Dx)=1/2`. Thus the null cone defined by this **chosen** realification is not invariant under all independent component phase conventions. The degenerate cases `x1=0` or `x2=0` remain on the cone in every such phase frame.

The pair `(Re z, Im z)` is *covariant*: it rotates as a two-component real vector by the relative phase `b-a`. Its length `|z|` is invariant under this restricted group. The coherency matrix `Q(x)=x x^*` transports as `Q(Dx)=D Q(x) D^*`. These are typed, frame-aware observables; none by itself demonstrates a better modem.

## Proposition B — ML communication is invariant under a consistent phase-frame change [P: algebraic identity]

For every codeword `x in C`,

```text
H' x' = (H D^{-1})(D x) = H x.
```

Therefore for each pair of codewords `x,y`,

```text
||H' (x'-y')||_2^2 = ||H (x-y)||_2^2.
```

The received candidate list, minimum received distance, and ML decision for any fixed received observation are identical (with correspondingly indexed codewords). Since `D` is unitary, the transmit symbol energy is also preserved. With identical channel/noise sampling, the *entire* symbol-error-rate distribution is the same. No particular SNR, propagation environment or simulator is needed for this equality.

This is **representation covariance**, not an assertion that arbitrary physical polarization rotations have no effect. At fixed `H`, applying `D` to symbols alone generally changes `Hx` and may change SER. Nor is a general non-unitary change a fair energy-constrained physical comparison without accounting for power and noise transformations.

## Why this matters for PR #82

The executed tutorial reported an empirical rank-one constellation SER advantage over QPSK² in its tested channels. That result is **M** (bounded simulation evidence). The later pre-registered matched controls showed that the determinant constraint itself was not established as the cause; the causal attribution remains **R** (retired). These new elementary propositions do not contradict either result. They identify a further interpretability pitfall: a frame-sensitive determinant cannot, by itself, provide a coordinate-independent causal explanation for an invariant receiver statistic.

In a future, independently scoped experiment, a physical mechanism must be formulated in terms of specified channel and polarization frames, or a genuinely frame-covariant object. Received pairwise Gram geometry (`||H(x-y)||^2`) is a minimal example. A stronger geometric lift would need predictive value beyond a resource-matched unconstrained optimizer.

## Pre-registered, compute-light audit

- **Prediction:** `q(Dx)` changes under relative phase for a nondegenerate null-cone point, while `H' D x = H x` for any phase-frame `D`; ML decisions remain exactly the same when channel and codebook are transformed jointly.
- **Controls:** identity frame, global phase, nontrivial relative phase, zero-component degeneracies, original 16-point rank-one construction, QPSK², and *symbol-only* changes at fixed channel.
- **Metric:** maximum absolute difference of received candidates and pairwise squared distances, plus exact equality of ML decision indices under identical noisy observations. Analytical claim is exact; floating-point checks use tolerance ~`1e-13`.
- **Kill condition:** if joint transformations change received candidates beyond numerical tolerance under the declared implementation, the implementation or declared transformation is defective; do not claim a new channel effect.
- **Stop rule:** do not run a new high-cost Sionna experiment or add hypercomplex coordinates to explain a representation artifact. Any new physical hypothesis requires a separate, matched-control preregistration.
- **Promotion rule:** elementary identities are supported by the proof above; the included CPU-only numerical tests are a regression check, **not** independent physical validation.

Run locally with NumPy (no Sionna RT, Mitsuba, or GPU needed):

```bash
python -m unittest discover -s research/tests -p 'test_phase_frame_covariance.py' -v
```

**Local verification (2026-11-04):** 8 tests passed on an isolated Python/NumPy environment. This does not verify NVIDIA upstream CI or the original ray-tracing notebook.

## Evidence ledger

| Claim | Status | Evidence |
|---|---|---|
| `q(x)=Im(conj(x1)*x2)` and not generally phase-frame invariant | P | determinant expansion and explicit `(1,1)/sqrt(2)` counterexample |
| Joint `x'=Dx, H'=HD^-1` leaves received candidates, pairwise distances, and indexed ML decisions unchanged | P | matrix identity, fixed-noise reasoning |
| CPU-only regression of identities for the notebook codebook and QPSK² | M (local test) | `research/tests/test_phase_frame_covariance.py` |
| Original codebook beats QPSK² for the notebook's declared simulated channels | M (prior artifact) | PR #82 notebook; not rerun here |
| Null-cone determinant term causes the reported SER gain | R | later matched optimizer controls documented on PR #82 |
| Any novel physical advantage from phase-frame geometry alone | H / not established | requires physical model and matched controls |

**Provenance:** Conceptual direction is human-led; derivation, experimental design, and regression implementation are AI-assisted. This audit was composed outside the local Copilot 6G checkout, which contains other unpushed test branches; it makes no claim to reproduce their results.
