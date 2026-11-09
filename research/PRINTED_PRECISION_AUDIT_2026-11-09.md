# Independent numerical recheck of the single printed Munich channel

**9 November 2026 · CPU-only · no new geometry, no model fitting, no NVlabs edits.**

**Scope:** exactly ONE PR #82 notebook-printed 2×2 receiver H (real and imaginary parts printed to eight decimal places), the frozen rank-one and QPSK² source constellations, and the already disclosed **post-hoc adversarial** circular off-null-cone 16-PSK control. Also check one synthetic balanced channel H=I/sqrt(2). **0/16 original receiver matrices are archived;** no result is promoted to an unobserved receiver or propagation distribution.

## Frozen test, source and implementation

- [Frozen protocol](./PRECISION_RECHECK_PREREG_2026-11-09.md) was committed before numerical evaluation.
- Notebook source revision: `57435845d3395b2d5e167b0893035e15ffbdaa0d`; notebook Git blob `10499baff91d408e5c64be080e0b0872fe81010e`.
- Historical bound ledger Git blob `f691364eed61adf928b772c3f6255d1c1f36270f`.
- [Independent implementation](./experiments/independent_printed_precision.py): Python mpmath 1.3.0 at 80 decimal digits; source-printed H is parsed as decimal strings; all received pair distances are calculated by *direct complex H×Delta multiplication*, not via prior project routines or Gram shortcuts. The normal-tail special function is cross-checked by separate Gaussian quadrature at t=0,0.5,2,4.
- [Six CPU regressions](./tests/test_independent_printed_precision.py) test source codebooks, energy parity, original null functional, Q integration, all 12 historical L/U values, strict gaps and rounding tolerance.
- Local full 80-digit output: `printed_precision_results.json`, SHA256 `0bb9d8296f8602646839512984d25c0dfa5e04c4882a3f81e213b47c949ff7af`; reproducible by the committed implementation. The full output is *not* an additional physical channel dataset.

Each symbol has unit energy and each design gives exactly 1/2 **average** energy to each physical TX port. The fixed circular control has Im(conj(x0)x1)=1/2 per symbol. The noise model remains white proper complex Gaussian `CN(0,N0 I₂)` with `N0=0.01`.

## High-precision conditional bounds

For 16 equiprobable symbols and coherent ML, compute the exact-real pairwise error event probabilities `p_ij = Q(||H(x_i-x_j)||/sqrt(2N0))`. The average over transmitted symbols of the maximum competing PEP is a lower bound L; the average sum of all 15 competing PEPs is a union upper bound U. These are **probability bounds**, not SER point estimates or confidence intervals.

| Fixed codebook | Printed H: L | Printed H: U | Balanced H: L | Balanced H: U |
|---|---:|---:|---:|---:|
| Circular 16-PSK, off cone | 0.025844113327 | 0.051823521836 | 0.025534279763 | 0.051198388993 |
| Source rank-one | 0.068087394256 | 0.106462722629 | 0.000064900908 | 0.000129801823 |
| Source QPSK² | 0.165171905715 | 0.414657762631 | 0.000000286652 | 0.000001146611 |

**Printed-channel strict gaps, direction lower SER:**

- `L(rank)-U(circular) = 0.016263872420245002609…`;
- `L(QPSK²)-U(rank) = 0.058709183085954243350…`.

**Balanced negative-control strict gaps:**

- `L(rank)-U(QPSK²) = 0.0000637542966149169501…`;
- `L(circular)-U(rank) = 0.0254044779407005281525…`.

Maximum absolute drift against the original 64-bit results was `2.934×10⁻¹⁶`, below the frozen `5×10⁻¹¹` tolerance.

## Rounding robustness (an analytic allowance; not formal interval software)

If each of the **eight real coordinates** of the complex H was rounded to the nearest 1e-8, then

`||E||_F <= sqrt(8)*5e-9 = 1.414213562...e-8.`

Because unit-energy symbols satisfy `||x_i-x_j||<=2` and Q is globally `1/sqrt(2π)`-Lipschitz,

`|Delta p_ij| <= epsilon = 7.978845608...e-8.`

A symbolwise max changes by at most epsilon; a sum over its 15 competitors by at most 15epsilon. Consequently any strict `L(B)-U(A)` separation on the printed channel loses **at most 16epsilon ≈ 1.2766152973e-6**. After this allowance, the smaller printed comparison gap is still approximately **0.016262595804947718**; the other is approximately **0.058707906470656959**.

This is an analytic perturbation statement *conditional on nearest-decimal rounding*. The 80-digit mpmath special-function evaluation is a **high-precision numerical check**, not a formally verified directed-rounding interval certificate. The empirical source/origin issue cannot be solved by numerical precision.

## Clean-checkout engineering evidence and stop

The [fork-only research CPU workflow](../.github/workflows/research-cpu-regression.yml) pins mpmath 1.3.0 alongside the existing Python/NumPy/SciPy/scikit-learn dependencies. The clean-checkout run verifies the frozen 75-cell SHA, runs **181 unit tests**, and independently executes the 80-digit script.

**P** conditional Gaussian event inequalities, reverse ordering on balanced synthetic H, and the rounding Lipschitz allowance.

**M** reproduced numerical inequalities on exactly one rounded source-printed channel, no extra receiver data. Same collaborator authored the tests, so this is **not** independent scientific replication.

**R** universal null-cone superiority and claimed incremental predictive value of Bargmann features remain retired. The off-cone baseline was found after seeing the one printed H.

**Next scientific gate:** authenticate the remaining 16 receiver scene channels (none presently available) and repeat a preregistered fixed-design comparison. Until then, retain the [short technical note](./TECHNICAL_NOTE_SCOPE_AND_NEGATIVE_RESULT.md) as the scientific claim and stop geometry expansion.
