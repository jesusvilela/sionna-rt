"""Matched, CPU-only post-discovery study of covariance scale sensitivity.

Run alongside torch_whitening_portability.py. This is a deterministic exploratory
experiment, not physical receiver data, and excludes numerically encoded
non-positive-definite inputs before algorithm comparison.
"""
import argparse
import json
import numpy as np
import torch

from torch_whitening_portability import (
    binary_float_oracle,
    gram,
    normalized_inv_cholesky,
    reference_inv_cholesky,
)

SEED = 20261112
SCALES = (1., 1e-200, 1e-309, 1e-320, float(np.nextafter(0., 1.)))


def represented_positive_definite(s):
    # Test the *represented* complex128 matrix, without float64 det underflow.
    if not np.array_equal(s, s.conj().T):
        return False
    a = np.longdouble(s[0, 0].real)
    d = np.longdouble(s[1, 1].real)
    x = np.longdouble(s[0, 1].real)
    y = np.longdouble(s[0, 1].imag)
    return bool(a > 0 and a * d - x * x - y * y > 0)


def sample_matrices():
    rng = np.random.default_rng(SEED)
    pairs = []
    for i in range(64):
        b = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        s = b @ b.conj().T + 0.8 * np.eye(2)
        s /= np.max(np.abs(s))
        s = (s + s.conj().T) / 2
        h = np.linalg.cholesky(s) @ np.diag([1., 0.5])
        pairs.append((s, h))
    return pairs


def safe_error(fn, h, s, exact):
    try:
        g = gram(fn, torch.from_numpy(h), torch.from_numpy(s))
        value = float(np.max(np.abs(g - exact)))
        return value if np.isfinite(value) else None
    except (RuntimeError, ValueError, FloatingPointError):
        return None


def summarize_errors(values):
    finite = np.asarray([x for x in values if x is not None], dtype=float)
    return {
        "finite": int(len(finite)),
        "nonfinite_or_exception": len(values) - len(finite),
        "median": float(np.median(finite)) if len(finite) else None,
        "max": float(np.max(finite)) if len(finite) else None,
        "above_1e-8": int(np.sum(finite > 1e-8)),
    }


def run():
    torch.set_num_threads(1)
    samples = sample_matrices()  # same 64 well-scaled shapes for *every* scale
    result = {}
    for scale in SCALES:
        old = []
        new = []
        omitted = []
        for index, (s0, h0) in enumerate(samples):
            s = np.asarray(scale * s0, dtype=np.complex128)
            h = np.asarray(np.sqrt(scale) * h0, dtype=np.complex128)
            if not represented_positive_definite(s):
                omitted.append(index)
                continue
            exact = binary_float_oracle(torch.from_numpy(h), torch.from_numpy(s))
            old.append(safe_error(reference_inv_cholesky, h, s, exact))
            new.append(safe_error(normalized_inv_cholesky, h, s, exact))
        a = summarize_errors(old)
        b = summarize_errors(new)
        better = sum((x is not None and y is not None and y < x) for x, y in zip(old, new))
        worse = sum((x is not None and y is not None and y > x) for x, y in zip(old, new))
        ties = sum((x is not None and y is not None and x == y) for x, y in zip(old, new))
        result[str(scale)] = {
            "valid_represented_spd": len(old),
            "excluded_non_spd_indices": omitted,
            "existing": a,
            "normalized": b,
            "normalized_better": better,
            "existing_better": worse,
            "ties": ties,
        }
    return {
        "protocol": "post-discovery matched synthetic 2x2 complex128 covariance shapes",
        "generator": "NumPy PCG64 seed 20261112; 64 fixed covariance shapes",
        "version": {"torch": torch.__version__, "numpy": np.__version__},
        "hardware": "CPU only; torch.set_num_threads(1)",
        "oracle": "100 decimal-digit direct inverse of actually represented complex128 inputs",
        "no_real_munich_data": True,
        "results": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data = run()
    with open(args.out, 'x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    print(json.dumps(data['results'], indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
