# Resource-aware moving-frame audit — no new wireless-performance claim

**5 November 2026 · CPU only · fork research branch · no changes to NVlabs or PR #82.**

**Scope first:** one rounded notebook channel, the previously frozen 75 **synthetic** Gram cells, **0/16 original Munich receiver matrices**, and six existing 16-symbol codebooks. This is a **coordinate and physical-constraint reproducibility check**, not a new channel dataset, a new geometric theorem, or a reinstatement of the retired null-cone/Bargmann performance claims. The concise scientific [technical note](./TECHNICAL_NOTE_SCOPE_AND_NEGATIVE_RESULT.md) remains the publishable kernel; this file is a software engineering appendix.

## Finding: a moving frame must move the physical constraint as well

The dual-port input channel Gram G, codewords x and physical port projectors E_a belong to one typed problem. Under a passive input coordinate change U in U(2),

\[
G' = UGU^\dagger,\qquad x'=Ux,\qquad E'_a=UE_aU^\dagger.
\]

The fixed physical null test is also part of the representation. With Q=[[0,-i],[i,0]], define \(q_Q(x)=x^\dagger Qx/2\). In moved coordinates use Q'=UQU^\dagger. The numerical value of q is unchanged, even when its *untransported coordinate expression* appears different.

For any codeword pair Δ, \(Δ'^\dagger G'Δ'=Δ^\dagger GΔ\). Thus **all 120 squared pair distances, all nearest-competitor and all-pair Gaussian error bounds, all strict decisions and abstentions are invariant**. This is elementary linear algebra, not new radio physics.

The prior equal-port beam family is optimized under physical constraints \(x^\dagger E_0x=x^\dagger E_1x=1/2\) for every symbol. If software transforms G but calls the source helper `equal_port_beam(G')` without transforming the resource basis, it *changes the feasible family* rather than representing the same physical design. Consequently, apparent decision changes cannot be attributed to a genuine channel advantage.

## Frozen test: 75 grids, six codebooks, two successive moving frames

The [protocol](./FRAME_RESOURCE_COVARIANCE_PROTOCOL.md) was committed **before** the CPU execution. Source SHA256 is
`16ad7ca1eda73d9c6a73776555be2f93333d248ad44d11a32ac9be3b05d2f4c6`.

- Frame 1: SU(2) mixing angle 31 degrees, complex phase 0.71 rad.
- Frame 2: *another* SU(2) mixing angle 17 degrees, phase -0.49 rad, composed after frame 1.
- Every case: six existing codebooks including fixed rank-one, QPSK², off-cone circular 16-PSK and three CSI-based references; N0=0.01.
- Exact metric: original Gaussian error intervals [L,U], min squared received distance, physical port powers and q; preserve all **75×5 strict rank-one-vs-comparator decisions**.
- Negative control: rebuild *only* the equal-coordinate-port 16-PSK control in the new basis without transporting E0/E1; record violations and changed classifications, but **do not** call it physically equivalent.

| Metric | Correctly transported object | Improper fixed-coordinate-port control |
|---|---:|---:|
| Strict decisions retained | **375/375**, unchanged | **14/75 changed** after first frame; **18/75 changed** after second |
| Maximum absolute bound change | **1.67e-15** | Up to **0.06588** first frame, **0.07294** after second |
| Maximum squared-distance change | **2.22e-15** | Comparison is a different codebook, not an invariance claim |
| Physical per-symbol port-power preservation | Better than **5.56e-16** | Violated in **75/75** cases at each frame |
| Maximum physical per-port power violation | — | **0.44147** first frame; **0.48882** after second |

**Important classification detail:** the 14 and 18 changes may be transitions **to or from abstention**, not necessarily reversals of two established strict rankings. They are not physical effects. The synthetic grid has repeated physical G at isotropic/axis cells and is not a sample of radio propagation.

The single printed H also passes an independently transformed **receiver** frame W together with transmit U: \(H'=WHU^\dagger\), \(C'=UC\). The largest direct-H squared-distance or interval deviation is **2.66e-15**. This says only that the original conditional Gaussian model is represented consistently.

## Verification and evidence boundaries

- [Executable audit](./experiments/frame_resource_audit.py): SHA-256 input verification, source-bound codebooks, port and null-form transforms, all 120 distances and strict-bound decisions, deliberate negative control, and the printed-channel receiver-frame check.
- [Ten regression tests](./tests/test_frame_resource_audit.py): **10/10 passed locally**, including a separately coded direct-H pair-distance oracle, two-step transport composition, physical beam-feasibility checks, stale source rejection and historical printed-H bounds.
- [Full compact result JSON](./FRAME_RESOURCE_AUDIT_RESULTS_2026-11-05.json): holds quantitative limits and negative-control counts. All three are committed on this fork-only branch. Source and tests were fetched back and compared against the locally executed **Git blob SHA**; the local formal audit manifest passed its schema validator with zero errors/warnings.

The tests and code are AI-authored and correlated in methodology, despite the separate direct-H implementation and metamorphic checks. This is a bounded software correctness result, **not independent external peer review**.

### Reproduce without installing Sionna

```bash
python research/experiments/gram_decision_sweep.py --out /tmp/sionna_gram.json
python research/experiments/frame_resource_audit.py --source /tmp/sionna_gram.json --out /tmp/frame_resource.json
SIONNA_GRAM_SWEEP=/tmp/sionna_gram.json python -m unittest discover -s research/tests -p 'test_frame_resource_audit.py' -v
```

Dependencies: ordinary CPU NumPy and SciPy only. No new ray tracing or Hugging Face/ZeroGPU/Work expenditure. The audit **refuses** any input that does not exactly match the source's frozen 75-case SHA256.

## Interpretation and stop

- **P (conditional algebra):** joint channel, codebook, physical-port and null-operator covariance under passive U(2) frame changes; coherent white-AWGN error-bound invariance.
- **M (bounded synthetic computation):** all 375 comparisons preserved; false comparison changes caused by the intentionally wrong resource-basis recomputation.
- **R (invalid representation interpretation):** calling the moved-coordinate equal-port reoptimizer physically equivalent without transporting the physical port projectors.
- **Still R:** geometric null-cone causality and three-site Bargmann incremental prediction advantage for the prior synthetic target.
- **Open empirical gap:** no real 16-receiver Munich channels were recovered. The previously attempted CPU Sionna regeneration was blocked by package resolution.

**Next smallest step:** keep this audit as a repeatable acceptance gate for any future channel/sample adapter. Do not add holonomy or geometric layers. Open the physics line only when a verified source-matched 16-channel dataset is available; test the existing codebooks with a matched unconstrained baseline before any engineering-advantage claim.

No upstream comments, submissions or PR changes.
