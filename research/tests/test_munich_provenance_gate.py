"""Adversarial structural tests. Synthetic arrays are never treated as Munich data."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from munich_provenance_gate import load_munich_archive,receiver_positions,SOURCE_COMMIT,SOURCE_FRAME


def fixture(root, mode='test-fixture', *, variant=True, raw=True, frame=None):
    root=Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng=np.random.default_rng(2398)
    H=rng.standard_normal((16,2,2))+1j*rng.standard_normal((16,2,2))
    H=H/np.linalg.norm(H,axis=(1,2))[:,None,None]
    out=root/'channels.npz'
    arrays={'H_all_normalized':H, 'H':H,'alive':np.ones(16,dtype=bool),
            'rx_positions_xyz_m':receiver_positions(),
            'tx_position_xyz_m':np.array([8.5,21.,27.])}
    if raw:arrays['H_raw']=2*H
    np.savez_compressed(out,**arrays)
    meta={'mode':mode,'source_commit':SOURCE_COMMIT,'num_receiver_positions':16,
          'all_16_available':True,'num_alive':16,'archive_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
          'frame_contract':SOURCE_FRAME if frame is None else frame,
          'H_all_sha256':hashlib.sha256(np.ascontiguousarray(H).tobytes()).hexdigest()}
    if variant:
        meta['CPU_variant']='llvm_ad_mono_polarized'
        meta['package_pins']={'sionna-rt':{'version':'2.2.0'}}
        meta['config']={'max_depth':2 if mode=='deterministic_cpu_reduced_config' else 5,
                        'samples_per_src':20000 if mode=='deterministic_cpu_reduced_config' else 1000000,
                        'deterministic':mode not in ('exact_notebook_best_effort',)}
    (root/'channels.manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    return out,H,meta


class GateTests(unittest.TestCase):
    def test_checksum_valid_synthetic_fixture_is_refused_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            path,H,_=fixture(d)
            with self.assertRaisesRegex(ValueError,'synthetic fixture'):
                load_munich_archive(path)
            loaded,_,result=load_munich_archive(path,allow_synthetic_fixture=True)
            np.testing.assert_allclose(loaded,H)
            self.assertEqual(result['classification'],'synthetic_fixture_only')
            self.assertFalse(result['origin_authenticated'])

    def test_manual_npz_import_requires_opt_in(self):
        with tempfile.TemporaryDirectory() as d:
            path,_,_=fixture(d,mode='from-npz',variant=False,raw=False)
            with self.assertRaisesRegex(ValueError,'unverified'):
                load_munich_archive(path)
            _,_,result=load_munich_archive(path,allow_unverified_import=True)
            self.assertEqual(result['classification'],'unverified_import_only')
            self.assertFalse(result['origin_authenticated'])

    def test_runner_candidate_remains_explicitly_unverified(self):
        for mode in ('deterministic_cpu','exact_notebook_best_effort','deterministic_cpu_reduced_config'):
            with tempfile.TemporaryDirectory() as d:
                path,H,_=fixture(d,mode=mode)
                loaded,_,status=load_munich_archive(path)
                np.testing.assert_array_equal(loaded,H)
                self.assertEqual(status['classification'],'cpu_regeneration_candidate_unverified')
                self.assertFalse(status['origin_authenticated'])

    def test_exporter_style_regeneration_candidate(self):
        for mode,det in [('regenerate-default',False),('regenerate-deterministic',True)]:
            with tempfile.TemporaryDirectory() as d:
                path,H,meta=fixture(d,mode=mode,variant=False)
                meta['mitsuba_variant']='llvm_ad_mono_polarized'
                meta['rt_version']='2.2.0';meta['solver_max_depth']=5
                meta['solver_deterministic']=det
                path.with_suffix('.manifest.json').write_text(json.dumps(meta))
                np.testing.assert_array_equal(load_munich_archive(path)[0],H)

    def test_candidates_must_contain_raw_data(self):
        with tempfile.TemporaryDirectory() as d:
            path,_,_=fixture(d,mode='deterministic_cpu',raw=False)
            with self.assertRaisesRegex(ValueError,'raw CIR-derived'):
                load_munich_archive(path)

    def test_source_commit_and_coordinate_order(self):
        with tempfile.TemporaryDirectory() as d:
            path,_,meta=fixture(d)
            meta['source_commit']='0'*40
            path.with_suffix('.manifest.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'source'):
                load_munich_archive(path,allow_synthetic_fixture=True)
        with tempfile.TemporaryDirectory() as d:
            path,_,meta=fixture(d)
            with np.load(path) as x:arrays={k:x[k] for k in x.files}
            arrays['rx_positions_xyz_m']=arrays['rx_positions_xyz_m'][::-1]
            np.savez_compressed(path,**arrays)
            meta['archive_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_suffix('.manifest.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'coordinates/order'):
                load_munich_archive(path,allow_synthetic_fixture=True)

    def test_frame_is_not_silently_relabelled(self):
        with tempfile.TemporaryDirectory() as d:
            path,_,_=fixture(d,frame='arbitrary_U2_unknown')
            with self.assertRaisesRegex(ValueError,'frame'):
                load_munich_archive(path,allow_synthetic_fixture=True)

    def test_self_consistent_checksum_cannot_replace_raw_consistency(self):
        with tempfile.TemporaryDirectory() as d:
            path,_,meta=fixture(d,mode='deterministic_cpu')
            with np.load(path) as x:arrays={k:x[k] for k in x.files}
            arrays['H_raw']=arrays['H_raw']@np.array([[0,1],[-1,0]])
            np.savez_compressed(path,**arrays)
            meta['archive_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_suffix('.manifest.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'raw-versus-normalized'):
                load_munich_archive(path)

    def test_rejects_mismatched_compact_full_array(self):
        with tempfile.TemporaryDirectory() as d:
            path,_,meta=fixture(d)
            with np.load(path) as x:arrays={k:x[k] for k in x.files}
            arrays['H']=arrays['H'][::-1]
            np.savez_compressed(path,**arrays)
            meta['archive_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_suffix('.manifest.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'disagree'):
                load_munich_archive(path,allow_synthetic_fixture=True)

    def test_rejects_tampered_bytes_even_if_npz_still_opens(self):
        with tempfile.TemporaryDirectory() as d:
            path,_,_=fixture(d)
            with open(path,'ab') as f:f.write(b'tamper')
            with self.assertRaisesRegex(ValueError,'SHA256'):
                load_munich_archive(path,allow_synthetic_fixture=True)

    def test_reduced_label_requires_exact_reduced_configuration(self):
        with tempfile.TemporaryDirectory() as d:
            path,_,meta=fixture(d,mode='deterministic_cpu_reduced_config')
            meta['config']['max_depth']=5
            path.with_suffix('.manifest.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'reduced CPU config'):
                load_munich_archive(path)

    def test_claims_cannot_be_upgraded_by_consistent_fabrication(self):
        # Even fully consistent raw, normalized H, geometry and checksum do not
        # prove physical origin. This is a designed trust boundary, not a solver test.
        with tempfile.TemporaryDirectory() as d:
            path,_,_=fixture(d,mode='deterministic_cpu')
            _,_,assessment=load_munich_archive(path)
            self.assertEqual(assessment['qualification'],'diagnostic candidates only; no claim of original Munich matrices')
            self.assertFalse(assessment['origin_authenticated'])

if __name__=='__main__': unittest.main()
