# Frozen precision audit — 9 November 2026

**Source:** PR #82 source notebook at `jesusvilela/sionna-rt@57435845d3395b2d5e167b0893035e15ffbdaa0d`, notebook blob `10499baff91d408e5c64be080e0b0872fe81010e`. **Comparison:** existing fixed printed-channel bounds in `research/experiments/printed_channel_control.py` (blob `cd908949b8c15512203734f824e6915fdefe8370`). New work is fork-only, CPU-only, and does not change PR #82.

## Object and narrow claim

For exactly the one **eight-decimal-place printed** complex 2×2 H in the notebook, the three already specified, unit-energy, 16-symbol fixed codebooks (rank-one, QPSK-squared and off-cone circular 16-PSK) and proper circular complex AWGN `N0=0.01`:

(1) Independent, arbitrary-precision direct-channel computation of all 120 unique pair distances and 16 symbolwise pairwise errors reproduces the existing Gaussian lower and union upper SER bounds, without calling the project implementations or using the Gram shortcut.

(2) Strict bound separation, **not merely comparisons of central estimates**, implies conditional ML-SER ordering: printed H = circular < rank-one < QPSK-squared; negative-control balanced H=I/sqrt(2) = reverse.

(3) For the printed H, every real and imaginary scalar was rounded to eight decimal places. Assuming conventional nearest rounding (each coordinate error ≤ 5e-9), derive a uniform deterministic perturbation allowance and check that both strict ordering gaps remain positive. This is a *conditional* robustness statement, not evidence of other receivers.

## Frozen computation and decisions

- Use Python 3 CPU and independent `mpmath` complex arithmetic at **80 decimal digits**, parsing the H literals directly as **decimal strings**. Construct both source codebooks independently from their published formulas; construct the existing circular 16-PSK adversarial baseline. No training, optimization, seed sweeping or channel fitting.
- For each ordered `i,j`, evaluate `d_ij=||H(x_i-x_j)||` by direct 2-complex-vector multiplication, and `p_ij=erfc(d_ij/(2 sqrt(N0)))/2`. For each design set `L=mean_i max_{j != i} p_ij`, `U=mean_i sum_{j != i} p_ij`. Re-evaluate the exact same formulas on the balanced diagonal negative-control H.
- Primary numerical gate: each previously published L and U differs from the 80-digit computation by **at most 5e-11 absolute**.
- Primary classification gate: for printed H, `U_circular < L_rank` and `U_rank < L_qpsk`; for balanced control `U_qpsk < L_rank` and `U_rank < L_circular`. If any fail, correct the existing note rather than add another model.
- Printed-rounding guarantee: `||E||_F ≤ sqrt(8)*5e-9`, `||x_i - x_j|| ≤ 2`, and Gaussian Q is 1/sqrt(2pi)-Lipschitz. Thus each PEP changes by ≤ `eps = 2||E||F/sqrt(4pi N0)`, per-row max changes by ≤ eps and per-row sum changes by ≤ 15 eps. Require printed gaps > **16 eps**, with no tuning.
- Confirm the three codebooks' per-symbol norm is 1, mean port powers are 0.5 each, and the circular baseline has `Im(conj(x0)*x1)=0.5`.
- Stop once the single printed H and balanced negative-control order are verified or falsified. No further geometry, no HF/GPU, no estimated Munich receiver data, no upstream changes or promotional claims.

## Evidence boundary

This is a **separate computational implementation** of the same conditional mathematical model, not independent experimental observation nor a formally directed-rounding interval proof. mpmath high precision does not itself authenticate H or prove implementation correctness. Numerical strict gaps combined with the *analytical* Lipschitz perturbation bound support a substantial, but conditional, rounding-robust decision. Nothing is concluded about the unobserved other 15 Munich receivers or general radio propagation.
