"""Optional CPU-only PyTorch portability audit; not part of research CPU CI.

Reimplements the tiny inv_cholesky arithmetic path from NVlabs/sionna at
src/sionna/phy/utils/linalg.py (source SHA cbc53d3e8791834ebc8cdc8001441867c997320d).
No production Sionna import, radio scene, GPU data, or field calibration.
Needs torch, numpy and mpmath installed explicitly by the operator.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import numpy as np
import mpmath as mp
import torch


def reference_inv_cholesky(s):
    """Exact structural reproduction of the upstream torch arithmetic path."""
    chol = torch.linalg.cholesky_ex(s, check_errors=False)[0]
    n = chol.shape[-1]
    identity = torch.eye(n, dtype=chol.dtype, device=chol.device)
    identity = identity.expand(*chol.shape[:-2], n, n)
    return torch.linalg.solve_triangular(chol, identity, upper=False)


def normalized_inv_cholesky(s):
    """A slower diagnostic candidate, NOT a production recommendation."""
    scale = torch.amax(torch.abs(s), dim=(-2, -1), keepdim=True)
    if s.is_complex():
        scaled = torch.complex(s.real / scale, s.imag / scale)
    else:
        scaled = s / scale
    chol = torch.linalg.cholesky_ex(scaled, check_errors=False)[0]
    n = chol.shape[-1]
    identity = torch.eye(n, dtype=chol.dtype, device=chol.device)
    identity = identity.expand(*chol.shape[:-2], n, n)
    inv_unit = torch.linalg.solve_triangular(chol, identity, upper=False)
    return inv_unit / torch.sqrt(scale)


def gram(inv_fn, h, s):
    z = inv_fn(s) @ h
    return (z.mH @ z).numpy()


def binary_float_oracle(h, s):
    """100-digit direct inversion of exact binary64 inputs, independent of Cholesky."""
    h = h.numpy()
    s = s.numpy()
    with mp.workdps(100):
        def c(z):
            return mp.mpc(mp.mpf(float(z.real)), mp.mpf(float(z.imag)))

        hm = mp.matrix([[c(v) for v in row] for row in h])
        sm = mp.matrix([[c(v) for v in row] for row in s])
        g = hm.transpose_conj() * (sm ** -1) * hm
        return np.array([[complex(g[i, j]) for j in range(2)] for i in range(2)])


def probe():
    s0 = torch.tensor(
        [[1 + 0j, .25 + .1j], [.25 - .1j, 1.5 + 0j]],
        dtype=torch.complex128,
    )
    h0 = torch.linalg.cholesky(s0) @ torch.diag(
        torch.tensor([1 + 0j, .5 + 0j], dtype=torch.complex128)
    )
    scales = (1., 1e-20, 1e20, 1e-200, 1e-300, 1e-309,
              1e-320, float(np.nextafter(0., 1.)))
    rows = []
    for scale in scales:
        s = scale * s0
        h = np.sqrt(scale) * h0
        exact = binary_float_oracle(h, s)
        original = gram(reference_inv_cholesky, h, s)
        normalized = gram(normalized_inv_cholesky, h, s)
        old_error = float(np.max(np.abs(original - exact)))
        new_error = float(np.max(np.abs(normalized - exact)))
        rows.append({"scale": scale, "existing_max_abs_error": old_error,
                     "normalized_max_abs_error": new_error})
        if not (np.isfinite(new_error) and new_error < 1e-12):
            raise AssertionError(("normalized implementation lost represented-input fidelity", scale, new_error))
        if scale == 1e-320 and not old_error > 1e-6:
            raise AssertionError(("previously reproduced precision defect no longer manifests", old_error))
        if scale == 1e-309 and not old_error < 1e-12:
            raise AssertionError(("unexpected change to existing well-handled subnormal case", old_error))

    w = torch.tensor([[1.1 + .3j, .4 - .1j], [.2 + .05j, .8 - .2j]],
                     dtype=torch.complex128)
    g0 = gram(reference_inv_cholesky, h0, s0)
    transported = gram(reference_inv_cholesky, w @ h0, w @ s0 @ w.mH)
    wrong = gram(reference_inv_cholesky, w @ h0, s0)
    covariant_error = float(np.max(np.abs(g0 - transported)))
    untransported_error = float(np.linalg.norm(g0 - wrong, ord=2))
    if covariant_error >= 1e-12 or untransported_error <= .1:
        raise AssertionError(("receiver-frame controls failed", covariant_error, untransported_error))
    return {
        "source_upstream_file_sha": "cbc53d3e8791834ebc8cdc8001441867c997320d",
        "torch_version": torch.__version__,
        "device": "CPU", "scales": rows,
        "valid_receiver_frame_max_abs_error": covariant_error,
        "untransported_noise_operator_norm_error": untransported_error,
        "claim_scope": "finite precision conditioning only; no Sionna simulation or 6G physics",
    }


def compile_probe():
    """Checks graph capture using the eager backend, not compiled CUDA or Inductor."""
    summary = {}
    for dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128):
        rtype = torch.float32 if dtype in (torch.float32, torch.complex64) else torch.float64
        a = torch.randn(8, 4, 4, dtype=rtype)
        if dtype.is_complex():
            a = torch.complex(a, torch.randn(8, 4, 4, dtype=rtype))
        a = a.to(dtype)
        a = a @ a.mH + .4 * torch.eye(4, dtype=dtype)
        direct = normalized_inv_cholesky(a)
        compiled = torch.compile(normalized_inv_cholesky, backend="eager", fullgraph=True)
        candidate = compiled(a)
        difference = float(torch.max(torch.abs(direct - candidate)))
        if difference > 1e-12:
            raise AssertionError(("compiled graph differs from direct path", str(dtype), difference))
        summary[str(dtype)] = difference
    return summary


def benchmark_probe():
    """Informational local CPU timing, not a comparative performance certification."""
    torch.set_num_threads(1)
    results = {}
    torch.manual_seed(2)
    for count in (128, 1024):
        a = torch.randn(count, 2, 2, dtype=torch.complex128)
        s = a @ a.mH + torch.eye(2, dtype=torch.complex128)
        pair = {}
        for label, fn in (("existing", reference_inv_cholesky),
                          ("normalized", normalized_inv_cholesky)):
            for _ in range(20):
                fn(s)
            samples = []
            for _ in range(4):
                started = time.perf_counter()
                for _ in range(80):
                    fn(s)
                samples.append(1000. * (time.perf_counter() - started) / 80.)
            pair[label] = {"median_ms_per_call": statistics.median(samples),
                           "four_runs_ms_per_call": samples}
        results[str(count)] = pair
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compile", action="store_true", help="run fullgraph eager torch.compile")
    p.add_argument("--benchmark", action="store_true", help="run matched local CPU benchmark")
    args = p.parse_args()
    torch.manual_seed(0)
    result = probe()
    if args.compile:
        result["fullgraph_eager"] = compile_probe()
    if args.benchmark:
        result["cpu_benchmark"] = benchmark_probe()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
