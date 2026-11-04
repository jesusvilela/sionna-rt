"""Export the 16 source-matched Munich dual-polarization channels, if available.

Safe default: dependency check, *no* ray tracing or package installation.
CPU regeneration is opt-in and preserves the notebook geometry exactly.
No GPU/Hugging Face backend, no modifications to PR #82.

Modes:
    --mode check
    --mode from-npz --input existing_notebook_channels.npz
    --mode regenerate-default      # notebook PathSolver(), seed 42 DEFAULT; NOT deterministic
    --mode regenerate-deterministic # PathSolver(deterministic=True), seed 42
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys

import numpy as np

SOURCE_REPO = "jesusvilela/sionna-rt"
SOURCE_HEAD = "57435845d3395b2d5e167b0893035e15ffbdaa0d"
SOURCE_FILE = "tutorials/Null-Cone-Constellations.ipynb"
RT_VERSION = "2.2.0"
CPU_VARIANT = "llvm_ad_mono_polarized"
NUM_RX = 16
MAX_DEPTH = 5
SOURCE_SEED_DEFAULT = 42
TX_POSITION = (8.5, 21.0, 27.0)
PRINTED_H0 = np.array([
    [0.92982936 + 0.33584663j, 0.01006189 + 0.00676795j],
    [-0.06026739 - 0.00641686j, -0.13674566 - 0.01022593j],
], dtype=np.complex128)


def receiver_positions() -> np.ndarray:
    """Exact notebook order: for each of eight y, x=25 then x=35."""
    return np.array([(x, y, 1.5)
                     for y in np.linspace(15.0, 60.0, 8)
                     for x in (25.0, 35.0)], dtype=np.float64)


def required_deps() -> dict[str, bool]:
    """Non-importing availability probe: never installs or imports ray tracing."""
    return {key: importlib.util.find_spec(key) is not None
            for key in ("sionna", "mitsuba", "drjit")}


def channel_arrays_from_paths(coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror notebook reshape/sum, preserve all 16 indices and alive mask."""
    a = np.asarray(coefficients)
    if a.size == 0 or a.size % (NUM_RX * 2 * 2):
        raise ValueError("CIR array size incompatible with 16x2x2 path coefficients")
    raw = np.asarray(a.reshape(NUM_RX, 2, 2, -1).sum(-1), dtype=np.complex128)
    if not np.isfinite(raw).all():
        raise ValueError("CIR contains NaN or Inf")
    norm = np.linalg.norm(raw, axis=(1, 2))
    alive = norm > 1e-12
    normalized = np.zeros_like(raw)
    normalized[alive] = raw[alive] / norm[alive, None, None]
    return raw, normalized, alive


def validate_normalized_h(H: np.ndarray, require_all_alive: bool = False) -> np.ndarray:
    """Validate a 16x2x2 array; zero channels may be retained if masked later."""
    H = np.asarray(H)
    if H.shape != (NUM_RX, 2, 2) or not np.issubdtype(H.dtype, np.complexfloating):
        raise ValueError("expected 16x2x2 *complex* H with original receiver ordering")
    if not np.isfinite(H).all():
        raise ValueError("H contains NaN/Inf")
    norms = np.linalg.norm(H, axis=(1, 2))
    alive = norms > 1e-12
    if require_all_alive and not alive.all():
        raise ValueError("require all 16 receiver matrices alive")
    if np.any(np.abs(norms[alive] - 1.0) > 5e-7):
        raise ValueError("H is not normalized as in PR #82")
    return alive


def printed_first_channel_error(H: np.ndarray, alive: np.ndarray) -> float | None:
    """Optional diagnostic only: original stochastic solver need not reproduce H0."""
    if not bool(alive[0]):
        return None
    return float(np.linalg.norm(H[0] - PRINTED_H0 / np.linalg.norm(PRINTED_H0)))


def make_scene(rt):
    """Recreate TX/16 RX and cross-polarized isotropic array setup, no optimization."""
    scene = rt.load_scene(rt.scene.munich)
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="VH")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="VH")
    scene.add(rt.Transmitter("tx", position=list(TX_POSITION)))
    for idx, pos in enumerate(receiver_positions()):
        scene.add(rt.Receiver(f"rx{idx}", position=pos.tolist()))
    return scene


def solve_scene(rt, scene, deterministic: bool) -> np.ndarray:
    """One solver call; the notebook default and deterministic alternative are distinct."""
    solver = rt.PathSolver(deterministic=True) if deterministic else rt.PathSolver()
    if deterministic:
        paths = solver(scene, max_depth=MAX_DEPTH, seed=SOURCE_SEED_DEFAULT)
    else:
        # Exact notebook call uses seed=42 by default; fixed seed NOT deterministic.
        paths = solver(scene, max_depth=MAX_DEPTH)
    a, _ = paths.cir(out_type="numpy")
    return np.asarray(a)


def regenerate(deterministic: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Opt-in only: CPU LLVM Sionna v2.2.0; no install and no GPU fallback."""
    missing = [k for k, present in required_deps().items() if not present]
    if missing:
        raise RuntimeError("CPU regeneration unavailable; missing: " + ", ".join(missing)
                           + ". No installation attempted. Use --mode from-npz with a saved H array.")
    import mitsuba as mi  # pylint: disable=import-outside-toplevel
    mi.set_variant(CPU_VARIANT)
    import sionna.rt as rt  # pylint: disable=import-outside-toplevel
    version = getattr(rt, "__version__", "unknown")
    if version != RT_VERSION:
        raise RuntimeError(f"Sionna RT version {version} != pinned {RT_VERSION}; refusing untracked drift")
    scene = make_scene(rt)
    a = solve_scene(rt, scene, deterministic)
    raw, H, alive = channel_arrays_from_paths(a)
    meta = {
        "mode": "regenerate-deterministic" if deterministic else "regenerate-default",
        "rt_version": version,
        "mitsuba_variant": mi.variant(),
        "solver_deterministic": bool(deterministic),
        "solver_seed": SOURCE_SEED_DEFAULT,
        "solver_max_depth": MAX_DEPTH,
        "source_match": ("source scene/array/path defaults, but CPU backend and stochastic execution"
                         if not deterministic else
                         "source scene/array/path settings with changed deterministic candidate generator"),
        "scene_frequency_hz": float(scene.frequency),
        "cir_original_array_shape": list(np.asarray(a).shape),
    }
    return raw, H, alive, meta


def load_existing_npz(input_path: Path) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, dict]:
    """Load an archived 16x2x2 H from a live PR #82 notebook; no pickle."""
    with np.load(input_path, allow_pickle=False) as archive:
        if "H" not in archive.files and "H_all_normalized" not in archive.files:
            raise ValueError("npz must contain H or H_all_normalized array")
        key = "H_all_normalized" if "H_all_normalized" in archive.files else "H"
        H = np.asarray(archive[key], dtype=np.complex128)
        raw = np.asarray(archive["H_raw"], dtype=np.complex128) if "H_raw" in archive.files else None
        provided_alive = np.asarray(archive["alive"], dtype=bool) if "alive" in archive.files else None
    alive = validate_normalized_h(H)
    if provided_alive is not None and not np.array_equal(alive, provided_alive):
        raise ValueError("saved alive mask inconsistent with H")
    if raw is not None:
        if raw.shape != H.shape or not np.isfinite(raw).all():
            raise ValueError("bad H_raw shape or values")
        norms = np.linalg.norm(raw, axis=(1, 2))
        if not np.array_equal(norms > 1e-12, alive):
            raise ValueError("raw and normalized alive masks differ")
        np.testing.assert_allclose(raw[alive] / norms[alive, None, None], H[alive], atol=5e-7)
    return raw, H, alive, {"mode": "from-npz", "input_name": input_path.name,
                           "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                           "input_key": key, "rt_version": None, "solver_deterministic": None,
                           "source_match": "user-provided saved H; provenance must be checked separately"}


def archive_export(out_path: Path, H: np.ndarray, alive: np.ndarray,
                   meta: dict, raw: np.ndarray | None = None) -> dict:
    """One archive and a self-describing manifest, preserving receiver positions."""
    alive_derived = validate_normalized_h(H)
    if not np.array_equal(alive_derived, alive):
        raise ValueError("alive does not match normalized channels")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path.with_suffix(".manifest.json")
    if out_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite archive or manifest: {out_path}")
    arrays = {"H": H[alive], "H_all_normalized": H, "alive": alive,
              "rx_positions_xyz_m": receiver_positions(),
              "tx_position_xyz_m": np.array(TX_POSITION)}
    if raw is not None:
        arrays["H_raw"] = np.asarray(raw)
    np.savez_compressed(out_path, **arrays)
    manifest = {
        "artifact": out_path.name,
        "archive_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "H_all_sha256": hashlib.sha256(np.ascontiguousarray(H).tobytes()).hexdigest(),
        "source_repo": SOURCE_REPO, "source_commit": SOURCE_HEAD, "source_file": SOURCE_FILE,
        "num_receiver_positions": NUM_RX, "num_alive": int(alive.sum()),
        "all_16_available": bool(alive.all()), "receiver_order": "y outer, x inner (25,35)",
        "tx_position_m": TX_POSITION, "rx_positions_xyz_m": receiver_positions().tolist(),
        "transmitter_array": "1x1 iso VH", "receiver_array": "1x1 iso VH",
        "max_depth": MAX_DEPTH, "alive_threshold": 1e-12,
        "reduction": "reshape(16,2,2,-1).sum(-1), then alive-filter and Frobenius normalize",
        "raw_available": raw is not None,
        "printed_H0_normalized_fro_error": printed_first_channel_error(H, alive),
        "printed_channel_match_is_required": False,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "numpy": np.__version__,
        **meta,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "from-npz", "regenerate-default",
                                          "regenerate-deterministic"), default="check")
    parser.add_argument("--input", type=Path, help="saved npz with 16x2x2 H")
    parser.add_argument("--out", type=Path, default=Path("munich_16_channels.npz"))
    args = parser.parse_args(argv)
    if args.mode == "check":
        print(json.dumps({"cpu_variant": CPU_VARIANT, "required_version": RT_VERSION,
                          "dependencies": required_deps(), "raytracing_started": False}, indent=2))
        return 0
    if args.mode == "from-npz":
        if args.input is None:
            parser.error("--mode from-npz requires --input")
        if args.input.resolve() == args.out.resolve():
            parser.error("input cannot be overwritten by output")
        raw, H, alive, meta = load_existing_npz(args.input)
    else:
        if args.input is not None:
            parser.error("--input applies only to from-npz")
        raw, H, alive, meta = regenerate(args.mode == "regenerate-deterministic")
    manifest = archive_export(args.out, H, alive, meta, raw)
    print(json.dumps({"archive": str(args.out), "manifest": str(args.out.with_suffix('.manifest.json')),
                      "alive": manifest["num_alive"], "H_sha256": manifest["H_all_sha256"],
                      "source_reproduced_exactly": False}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
