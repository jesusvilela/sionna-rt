"""Fail-closed structural and evidence-origin gate for 16 Munich channel arrays.

Checksum + metadata are integrity checks, NOT independent provenance proofs.
Every accepted archive is diagnostic-only until external scene/runtime capture
and provenance corroboration; synthetic and imported archives need opt-in.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

SOURCE_COMMIT = '57435845d3395b2d5e167b0893035e15ffbdaa0d'
CPU_VARIANT = 'llvm_ad_mono_polarized'
SOURCE_FRAME = 'transmitter_VH_shared_port_basis'
CPU_MODES = frozenset((
    'regenerate-default', 'regenerate-deterministic',
    'exact_notebook_best_effort', 'deterministic_cpu',
    'deterministic_cpu_reduced_config',
))


def receiver_positions():
    return np.array([(x,y,1.5) for y in np.linspace(15.,60.,8)
                     for x in (25.,35.)], dtype=float)


def _exact_complex_array(archive, name, shape):
    if name not in archive.files:
        raise ValueError(f'archive missing {name}')
    arr = np.asarray(archive[name])
    if arr.shape != shape or not np.issubdtype(arr.dtype, np.complexfloating):
        raise ValueError(f'{name} must have shape {shape} and complex dtype')
    if not np.isfinite(arr).all():
        raise ValueError(f'{name} has nonfinite entries')
    return np.asarray(arr, dtype=np.complex128)


def _classify(meta, allow_synthetic_fixture, allow_unverified_import):
    mode = meta.get('mode')
    if mode == 'test-fixture':
        if not allow_synthetic_fixture:
            raise ValueError('synthetic fixture requires explicit allow_synthetic_fixture')
        return 'synthetic_fixture_only'
    if mode == 'from-npz':
        if not allow_unverified_import:
            raise ValueError('from-npz provenance unverified; explicit opt-in required')
        return 'unverified_import_only'
    if mode not in CPU_MODES:
        raise ValueError(f'unsupported provenance mode: {mode!r}')
    backend = meta.get('CPU_variant', meta.get('mitsuba_variant'))
    if backend != CPU_VARIANT:
        raise ValueError('candidate requires explicit CPU llvm_ad_mono_polarized backend')
    if 'rt_version' in meta and meta['rt_version'] != '2.2.0':
        raise ValueError('candidate Sionna RT version mismatch')
    pins = meta.get('package_pins')
    if pins is not None and pins.get('sionna-rt',{}).get('version') != '2.2.0':
        raise ValueError('candidate resolved Sionna version mismatch')
    cfg = meta.get('config',{})
    if mode == 'deterministic_cpu_reduced_config':
        if cfg.get('max_depth') != 2 or cfg.get('samples_per_src') != 20000 or cfg.get('deterministic') is not True:
            raise ValueError('reduced CPU config is not the frozen reduced specification')
    elif mode in ('deterministic_cpu','exact_notebook_best_effort'):
        if cfg.get('max_depth') != 5 or bool(cfg.get('deterministic')) != (mode=='deterministic_cpu'):
            raise ValueError('CPU config mislabelled as full notebook mode')
    else:
        if meta.get('solver_max_depth') != 5 or bool(meta.get('solver_deterministic')) != (mode=='regenerate-deterministic'):
            raise ValueError('exporter CPU solver config mismatch')
    return 'cpu_regeneration_candidate_unverified'


def load_munich_archive(path, *, allow_synthetic_fixture=False,
                        allow_unverified_import=False):
    """Return (normalized H, manifest, diagnostic assessment), never 'original H'."""
    path = Path(path)
    manifest_path = path.with_suffix('.manifest.json')
    if not manifest_path.is_file():
        raise ValueError('missing archive manifest')
    meta = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not isinstance(meta,dict):
        raise ValueError('bad manifest object')
    if meta.get('source_commit') != SOURCE_COMMIT or meta.get('num_receiver_positions') != 16 or meta.get('all_16_available') is not True:
        raise ValueError('source or receiver count provenance mismatch')
    if 'num_alive' in meta and meta['num_alive'] != 16:
        raise ValueError('reported alive count disagrees with all-alive claim')
    if 'source_repo' in meta and meta['source_repo'] != 'jesusvilela/sionna-rt':
        raise ValueError('unexpected source repository')
    if 'source_file' in meta and meta['source_file'] != 'tutorials/Null-Cone-Constellations.ipynb':
        raise ValueError('unexpected source notebook')
    if 'frame_contract' in meta and meta['frame_contract'] != SOURCE_FRAME:
        raise ValueError('transmitter port frame is not source VH')
    declared = _classify(meta, allow_synthetic_fixture, allow_unverified_import)
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta.get('archive_sha256'):
        raise ValueError('archive SHA256 mismatch')
    with np.load(path,allow_pickle=False) as d:
        H = _exact_complex_array(d,'H_all_normalized',(16,2,2))
        compact = _exact_complex_array(d,'H',(16,2,2))
        mask=np.asarray(d['alive']) if 'alive' in d.files else None
        if mask is None or mask.shape!=(16,) or mask.dtype.kind != 'b' or not mask.all():
            raise ValueError('16-channel alive mask must be exactly boolean and all true')
        if 'rx_positions_xyz_m' not in d.files or 'tx_position_xyz_m' not in d.files:
            raise ValueError('missing source coordinate arrays')
        rx=np.asarray(d['rx_positions_xyz_m'],float)
        tx=np.asarray(d['tx_position_xyz_m'],float)
        if rx.shape!=(16,3) or not np.isfinite(rx).all() or not np.allclose(rx,receiver_positions(),rtol=0,atol=1e-9):
            raise ValueError('receiver coordinates/order mismatch')
        if tx.shape!=(3,) or not np.allclose(tx,[8.5,21.,27.],rtol=0,atol=1e-9):
            raise ValueError('transmitter position mismatch')
        if not np.allclose(compact,H,rtol=0,atol=1e-12):
            raise ValueError('H and H_all_normalized disagree')
        norms=np.linalg.norm(H,axis=(1,2))
        if not np.allclose(norms,1.,rtol=0,atol=5e-7):
            raise ValueError('H is not Frobenius-normalized')
        if meta.get('H_all_sha256') is not None:
            if hashlib.sha256(np.ascontiguousarray(H).tobytes()).hexdigest() != meta['H_all_sha256']:
                raise ValueError('normalized H content hash mismatch')
        raw=_exact_complex_array(d,'H_raw',(16,2,2)) if 'H_raw' in d.files else None
        if declared == 'cpu_regeneration_candidate_unverified' and raw is None:
            raise ValueError('CPU candidate requires raw CIR-derived H')
        if raw is not None:
            raw_norm=np.linalg.norm(raw,axis=(1,2))
            if np.any(raw_norm <= 1e-12):
                raise ValueError('raw H implies a dead receiver')
            if not np.allclose(raw/raw_norm[:,None,None],H,rtol=0,atol=5e-7):
                raise ValueError('raw-versus-normalized H mismatch')
    assessment = {
        'classification':declared,
        'mode':meta['mode'],
        'frame':meta.get('frame_contract',SOURCE_FRAME),
        'frame_authority':'source-basis declaration; not an independently measured calibration',
        'origin_authenticated':False,
        'receiver_count':16,
        'qualification':'diagnostic candidates only; no claim of original Munich matrices',
    }
    return H,meta,assessment
