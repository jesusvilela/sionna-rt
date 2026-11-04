"""Offline regression for every exported operation. No Sionna imports or RT jobs."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest import TestCase, main, mock

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import export_munich_16 as exp


class ExportTests(TestCase):
    def test_positions_exact_source_order(self):
        p=exp.receiver_positions()
        self.assertEqual(p.shape,(16,3))
        np.testing.assert_allclose(p[0],[25.,15.,1.5])
        np.testing.assert_allclose(p[1],[35.,15.,1.5])
        np.testing.assert_allclose(p[-1],[35.,60.,1.5])
        self.assertEqual(len(np.unique(p,axis=0)),16)

    def test_dependency_probe_and_safe_default(self):
        avail=exp.required_deps()
        self.assertEqual(set(avail),{'sionna','mitsuba','drjit'})
        self.assertEqual(set(type(x) for x in avail.values()),{bool})
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(exp.main(['--mode','check']),0)
        status=json.loads(out.getvalue())
        self.assertFalse(status['raytracing_started'])
        self.assertEqual(status['required_version'],'2.2.0')

    def test_cir_reduction_normalization_and_alive_mask(self):
        # Synthetic CIR: path a0 contributes identity, path a1 identical;
        # index 3 intentionally dead. This does not simulate propagation.
        a=np.zeros((16,2,2,2),complex)
        a[:,0,0,0]=1j
        a[:,1,1,1]=2.
        a[3]=0
        raw,H,alive=exp.channel_arrays_from_paths(a)
        self.assertEqual(raw.shape,(16,2,2))
        self.assertEqual(int(alive.sum()),15)
        np.testing.assert_allclose(np.linalg.norm(H[alive],axis=(1,2)),1.,atol=1e-14)
        np.testing.assert_allclose(H[3],0.)
        self.assertFalse(exp.validate_normalized_h(H)[3])
        with self.assertRaises(ValueError):exp.validate_normalized_h(H,True)
        with self.assertRaises(ValueError):exp.channel_arrays_from_paths(np.ones(10))
        with self.assertRaises(ValueError):exp.channel_arrays_from_paths(np.full((16,2,2,1),np.nan))

    def test_normalized_data_rejects_bad_shapes_values_and_scales(self):
        H=np.tile(np.eye(2,dtype=complex)[None,:,:]/np.sqrt(2),(16,1,1))
        self.assertTrue(exp.validate_normalized_h(H,True).all())
        with self.assertRaises(ValueError):exp.validate_normalized_h(H[:15])
        with self.assertRaises(ValueError):exp.validate_normalized_h(H*2)
        Hbad=H.copy();Hbad[0,0,0]=np.nan
        with self.assertRaises(ValueError):exp.validate_normalized_h(Hbad)
        with self.assertRaises(ValueError):exp.validate_normalized_h(np.ones((16,2,2)))

    def test_printed_reference_diagnostic(self):
        H=np.tile(np.eye(2,dtype=complex)[None,:,:]/np.sqrt(2),(16,1,1))
        H[0]=exp.PRINTED_H0/np.linalg.norm(exp.PRINTED_H0)
        alive=np.ones(16,dtype=bool)
        self.assertLess(exp.printed_first_channel_error(H,alive),1e-14)
        alive[0]=False
        self.assertIsNone(exp.printed_first_channel_error(H,alive))

    def test_source_scene_stub_is_exact_and_ordered(self):
        class FakeScene:
            def __init__(self):self.devices=[]
            def add(self,device):self.devices.append(device)
        class FakeRT:
            scene=type('Scenes',(),{'munich':'munich-marker'})
            def __init__(self):self.r=FakeScene()
            def load_scene(self,name):
                assert name=='munich-marker'
                return self.r
            @staticmethod
            def PlanarArray(**kw):return ('array',kw)
            @staticmethod
            def Transmitter(name,position):return ('tx',name,position)
            @staticmethod
            def Receiver(name,position):return ('rx',name,position)
        rt=FakeRT();scene=exp.make_scene(rt)
        self.assertIs(scene,rt.r)
        self.assertEqual(scene.tx_array[1],{'num_rows':1,'num_cols':1,'pattern':'iso','polarization':'VH'})
        self.assertEqual(scene.rx_array[1],{'num_rows':1,'num_cols':1,'pattern':'iso','polarization':'VH'})
        self.assertEqual(len(scene.devices),17)
        self.assertEqual(scene.devices[0],('tx','tx',[8.5,21.,27.]))
        self.assertEqual(scene.devices[1],('rx','rx0',[25.,15.,1.5]))
        self.assertEqual(scene.devices[-1],('rx','rx15',[35.,60.,1.5]))

    def test_path_solver_modes_are_never_silently_conflated(self):
        class MockPaths:
            def cir(self,**kw):
                assert kw=={'out_type':'numpy'}
                return np.zeros((16,2,2,1),complex),'delays'
        class FakeRT:
            def __init__(self):self.calls=[]
            def PathSolver(self,**init):
                self.calls.append(('constructor',init))
                def run(scene,**kw):
                    self.calls.append(('call',kw))
                    return MockPaths()
                return run
        rt=FakeRT()
        for deterministic in (False,True):
            a=exp.solve_scene(rt,'scene',deterministic)
            self.assertEqual(a.shape,(16,2,2,1))
        self.assertEqual(rt.calls[0],('constructor',{}))
        self.assertEqual(rt.calls[1],('call',{'max_depth':5}))
        self.assertEqual(rt.calls[2],('constructor',{'deterministic':True}))
        self.assertEqual(rt.calls[3],('call',{'max_depth':5,'seed':42}))

    def test_missing_deps_fail_fast_with_no_install(self):
        with mock.patch.object(exp,'required_deps',return_value={'sionna':False,'mitsuba':False,'drjit':False}):
            with self.assertRaisesRegex(RuntimeError,'No installation attempted'):
                exp.regenerate(False)
            with self.assertRaisesRegex(RuntimeError,'No installation attempted'):
                exp.regenerate(True)

    def test_from_npz_preserves_source_order_and_raw_invariants(self):
        H=np.tile(np.eye(2,dtype=complex)[None,:,:]/np.sqrt(2),(16,1,1))
        raw=H*2
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.npz'
            np.savez_compressed(path,H=H,H_raw=raw,alive=np.ones(16,bool))
            received_raw,received_H,alive,meta=exp.load_existing_npz(path)
            np.testing.assert_array_equal(received_raw,raw)
            np.testing.assert_array_equal(received_H,H)
            self.assertTrue(alive.all())
            self.assertEqual(meta['mode'],'from-npz')
            self.assertEqual(len(meta['input_sha256']),64)
            wrong=Path(tmp)/'wrong.npz'
            np.savez_compressed(wrong,H=H,alive=np.zeros(16,bool))
            with self.assertRaises(ValueError):exp.load_existing_npz(wrong)
            wrongraw=Path(tmp)/'wrongraw.npz'
            np.savez_compressed(wrongraw,H=H,H_raw=H*3,alive=np.ones(16,bool))
            # Normalizing *raw* still recovers H: this is valid provenance.
            self.assertTrue(exp.load_existing_npz(wrongraw)[2].all())

    def test_archive_and_manifest_roundtrip_with_checksum(self):
        H=np.tile(np.eye(2,dtype=complex)[None,:,:]/np.sqrt(2),(16,1,1))
        alive=np.ones(16,bool)
        with tempfile.TemporaryDirectory() as tmp:
            dst=Path(tmp)/'channels.npz'
            meta=exp.archive_export(dst,H,alive,{'mode':'from-npz'},raw=2*H)
            self.assertTrue(meta['all_16_available'])
            self.assertEqual(meta['num_alive'],16)
            self.assertEqual(meta['source_commit'],exp.SOURCE_HEAD)
            self.assertEqual(len(meta['archive_sha256']),64)
            with np.load(dst,allow_pickle=False) as p:
                np.testing.assert_array_equal(p['H_all_normalized'],H)
                np.testing.assert_array_equal(p['H'],H)
                np.testing.assert_array_equal(p['rx_positions_xyz_m'],exp.receiver_positions())
            self.assertEqual(json.loads(dst.with_suffix('.manifest.json').read_text())['archive_sha256'],meta['archive_sha256'])
            with self.assertRaises(FileExistsError):exp.archive_export(dst,H,alive,{})
            # Atomic safety: manifest-only collision must not write archive.
            second=Path(tmp)/'second.npz';second.with_suffix('.manifest.json').write_text('{}')
            with self.assertRaises(FileExistsError):exp.archive_export(second,H,alive,{})
            self.assertFalse(second.exists())

    def test_main_from_npz_and_rejected_overwrite(self):
        H=np.tile(np.eye(2,dtype=complex)[None,:,:]/np.sqrt(2),(16,1,1))
        with tempfile.TemporaryDirectory() as tmp:
            inp=Path(tmp)/'source.npz';out=Path(tmp)/'out.npz'
            np.savez_compressed(inp,H=H)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(exp.main(['--mode','from-npz','--input',str(inp),'--out',str(out)]),0)
            self.assertTrue(out.is_file())
            self.assertTrue(out.with_suffix('.manifest.json').is_file())
            with self.assertRaises(FileExistsError):
                exp.main(['--mode','from-npz','--input',str(inp),'--out',str(out)])


if __name__=='__main__':main()
