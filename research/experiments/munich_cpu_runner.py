"""Budgeted Sionna RT 2.2.0 CPU-only Munich regeneration runner.

This runner NEVER installs dependencies, selects a GPU, or writes to NVlabs.
The exact and deterministic modes are distinct experiments, not original H.
Run --mode both --wall-seconds 60 --memory-mib 4096; logs are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

import numpy as np

HEAD = '57435845d3395b2d5e167b0893035e15ffbdaa0d'
CPU_VARIANT='llvm_ad_mono_polarized'
SIONNA_PIN='2.2.0'
SOURCE_SEED=42
MODES=('exact_notebook_best_effort','deterministic_cpu','deterministic_cpu_reduced_config')
PRINTED = np.array([[.92982936+.33584663j,.01006189+.00676795j],
                    [-.06026739-.00641686j,-.13674566-.01022593j]],complex)


def pins():
    out={}
    for package,mod in [('sionna-rt','sionna'),('mitsuba','mitsuba'),('drjit','drjit')]:
        try: version=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: version=None
        out[package]={'installed':importlib.util.find_spec(mod) is not None,
                      'version':version}
    return out


def receiver_positions():
    return np.array([[x,y,1.5] for y in np.linspace(15.,60.,8) for x in (25.,35.)])


def config(mode):
    if mode not in MODES:raise ValueError('invalid CPU configuration')
    return {'max_depth':2 if mode==MODES[2] else 5,
            'samples_per_src':20000 if mode==MODES[2] else 1000000,
            'deterministic':mode!=MODES[0],
            'seed':SOURCE_SEED,'synthetic_array':True,'cpu_variant':CPU_VARIANT,
            'source_match':mode==MODES[0]}


def scene_and_paths(rt,mode):
    """Exact notebook scene; explicit CPU mode only; solver parameters frozen."""
    cfg=config(mode)
    scene=rt.load_scene(rt.scene.munich)
    scene.tx_array=rt.PlanarArray(num_rows=1,num_cols=1,pattern='iso',polarization='VH')
    scene.rx_array=rt.PlanarArray(num_rows=1,num_cols=1,pattern='iso',polarization='VH')
    scene.add(rt.Transmitter('tx',position=[8.5,21.0,27.0]))
    for i,p in enumerate(receiver_positions()):
        scene.add(rt.Receiver(f'rx{i}',position=p.tolist()))
    if mode==MODES[0]:
        paths=rt.PathSolver()(scene,max_depth=5)
    elif mode==MODES[1]:
        paths=rt.PathSolver(deterministic=True)(scene,max_depth=5,seed=42)
    else:
        paths=rt.PathSolver(deterministic=True)(scene,max_depth=2,samples_per_src=20000,seed=42)
    a,_=paths.cir(out_type='numpy')
    return np.asarray(a),scene


def cir_to_channels(a):
    x=np.asarray(a)
    if x.size<=0 or x.size%(16*4):raise ValueError('CIR incompatible with 16 receivers, two ports each')
    raw=x.reshape(16,2,2,-1).sum(-1)
    if not np.isfinite(raw).all():raise ValueError('nonfinite CIR')
    norms=np.linalg.norm(raw,axis=(1,2))
    alive=norms>1e-12
    H=np.zeros_like(raw,dtype=np.complex128)
    H[alive]=raw[alive]/norms[alive,None,None]
    return raw,H,alive


def worker(mode,out_dir):
    # Fail clearly on missing dependencies BEFORE initializing ray tracing.
    dependencies=pins()
    missing=[k for k,v in dependencies.items() if not v['installed']]
    if missing:raise RuntimeError('dependencies unavailable (no pip/network): '+','.join(missing))
    if dependencies['sionna-rt']['version']!=SIONNA_PIN:
        raise RuntimeError('Sionna pin mismatch: '+str(dependencies['sionna-rt']['version']))
    import mitsuba as mi
    mi.set_variant(CPU_VARIANT)
    import sionna.rt as rt
    if mi.variant()!=CPU_VARIANT:raise AssertionError('CPU variant not selected')
    a,scene=scene_and_paths(rt,mode)
    raw,H,alive=cir_to_channels(a)
    return save_archive(Path(out_dir)/f'{mode}.npz',raw,H,alive,mode,
                        dependencies,mi.variant(),float(scene.frequency),list(a.shape))



def save_archive(out,raw,H,alive,mode,dependencies,cpu_variant,scene_frequency,cir_shape):
    """Write exact schema accepted by the previous 16-receiver Gram importer."""
    if cpu_variant != CPU_VARIANT:raise ValueError('non-CPU variant refused')
    out=Path(out)
    if out.exists() or out.with_suffix('.manifest.json').exists():
        raise FileExistsError('refusing to overwrite archive or manifest')
    H=np.asarray(H,complex);raw=np.asarray(raw,complex);alive=np.asarray(alive,bool)
    if H.shape!=(16,2,2) or raw.shape!=(16,2,2) or alive.shape!=(16,):
        raise ValueError('expected 16x2x2 channels and 16 alive flags')
    if not np.isfinite(H).all() or not np.isfinite(raw).all():raise ValueError('nonfinite channels')
    if not np.allclose(np.linalg.norm(H[alive],axis=(1,2)),1,atol=5e-7):
        raise ValueError('alive H must be normalized')
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,H=H[alive],H_all_normalized=H,H_raw=raw,
                        alive=alive,rx_positions_xyz_m=receiver_positions(),
                        tx_position_xyz_m=np.array([8.5,21.,27.]))
    sha=hashlib.sha256(out.read_bytes()).hexdigest()
    meta={'mode':mode,'config':config(mode),'num_receiver_positions':16,
          'num_alive':int(alive.sum()),'all_16_available':bool(alive.all()),
          'source_commit':HEAD,'package_pins':dependencies,'CPU_variant':cpu_variant,
          'frame_contract':'transmitter_VH_shared_port_basis',
          'scene_frequency_hz':scene_frequency,'cir_shape':cir_shape,
          'output':out.name,'archive_sha256':sha,
          'first_channel_printed_fro_diff':float(np.linalg.norm(H[0]-PRINTED/np.linalg.norm(PRINTED))) if alive[0] else None,
          'provenance':'CPU regeneration, NOT source-original CUDA matrices'}
    out.with_suffix('.manifest.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n')
    return meta


def compare_outputs(a,b):
    """Compare per-receiver channel and receiver-side unitary invariant Gram."""
    A=np.asarray(a,dtype=complex);B=np.asarray(b,dtype=complex)
    if A.shape!=(16,2,2) or B.shape!=(16,2,2):raise ValueError('16x2x2 required')
    Hdist=np.linalg.norm(A-B,axis=(1,2))
    GA=A.conj().transpose(0,2,1)@A
    GB=B.conj().transpose(0,2,1)@B
    Gdist=np.linalg.norm(GA-GB,axis=(1,2))
    return {'per_receiver_Fro_H':Hdist.tolist(),'per_receiver_Fro_Gram':Gdist.tolist(),
            'mean_Fro_H':float(Hdist.mean()),'max_Fro_H':float(Hdist.max()),
            'mean_Fro_Gram':float(Gdist.mean()),'max_Fro_Gram':float(Gdist.max()),
            'note':'Gram differences are left-unitary invariant; direct H differences are not'}


def _child_memory_cap(max_mib):
    limit=int(max_mib)*1024*1024
    resource.setrlimit(resource.RLIMIT_AS,(limit,limit))


def run_mode(mode,out_dir,wall_seconds,memory_mib,script_path=None):
    """Hard deadline via subprocess session isolation, plus 4096-MiB VA cap."""
    if wall_seconds<=0 or memory_mib<=0:raise ValueError('positive budget required')
    if script_path is None:script_path=Path(__file__).resolve()
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    stdout=out_dir/f'{mode}.log'
    cmd=[sys.executable,str(script_path),'--worker',mode,'--out-dir',str(out_dir)]
    started=time.monotonic()
    with stdout.open('w') as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,
                              start_new_session=True,preexec_fn=lambda:_child_memory_cap(memory_mib))
        try:
            exit_code=proc.wait(timeout=wall_seconds)
            status='succeeded' if exit_code==0 else 'failed'
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=2)
            exit_code=proc.returncode;status='wall_timeout'
    result={'mode':mode,'status':status,'exit_code':exit_code,
            'elapsed_wall_seconds':round(time.monotonic()-started,3),
            'wall_budget_seconds':wall_seconds,'memory_limit_mib':memory_mib,
            'stdout_log':stdout.name,'output_exists':(out_dir/f'{mode}.npz').exists()}
    (out_dir/f'{mode}.status.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result



def load_verified_output(out_dir,mode,attempt):
    """Never compare stale partial output from an unsuccessful current run."""
    if attempt['status']!='succeeded':return None
    path=Path(out_dir)/f'{mode}.npz'
    manifest=path.with_suffix('.manifest.json')
    if not path.is_file() or not manifest.is_file():
        raise ValueError('successful process did not produce a complete artifact')
    info=json.loads(manifest.read_text())
    if info.get('mode')!=mode or info.get('source_commit')!=HEAD or info.get('CPU_variant')!=CPU_VARIANT:
        raise ValueError('source, mode, or CPU provenance mismatch')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=info.get('archive_sha256'):
        raise ValueError('archive checksum mismatch')
    with np.load(path,allow_pickle=False) as d:
        H=np.asarray(d['H_all_normalized'],complex)
        alive=np.asarray(d['alive'],bool)
    if H.shape!=(16,2,2) or alive.shape!=(16,) or not np.isfinite(H).all():
        raise ValueError('bad channel shape/mask/values')
    if not np.allclose(np.linalg.norm(H[alive],axis=(1,2)),1,atol=5e-7):
        raise ValueError('un-normalized living channels')
    return H


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('both','reduced')+MODES,default='both')
    p.add_argument('--out-dir',type=Path,default=Path('munich_cpu_attempts'))
    p.add_argument('--wall-seconds',type=int,default=60)
    p.add_argument('--memory-mib',type=int,default=4096)
    p.add_argument('--worker',choices=MODES,help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.worker:
        result=worker(a.worker,a.out_dir)
        print(json.dumps(result,sort_keys=True))
        return
    if a.mode=='both':modes=MODES[:2]
    elif a.mode=='reduced':modes=MODES[2:]
    else:modes=(a.mode,)
    report={'dependencies_before':pins(),'source_commit':HEAD,
            'requested':list(modes),'attempts':[]}
    for mode in modes:
        result=run_mode(mode,a.out_dir,a.wall_seconds,a.memory_mib)
        report['attempts'].append(result)
    statuses={entry['mode']:entry for entry in report['attempts']}
    if all(mode in statuses for mode in MODES[:2]):
        exact=load_verified_output(a.out_dir,MODES[0],statuses[MODES[0]])
        det=load_verified_output(a.out_dir,MODES[1],statuses[MODES[1]])
        if exact is not None and det is not None:
            report['comparison']=compare_outputs(exact,det)
        else:
            report['comparison']={'status':'not_measurable','reason':'both current successful and verified channel arrays required'}
    else:
        report['comparison']={'status':'not_measurable','reason':'both configuration modes were not run together'}
    a.out_dir.mkdir(parents=True,exist_ok=True)
    (a.out_dir/'attempt_summary.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
