"""ADSceneGraph adapter using the existing nuPlan reconstruction implementation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from .common import ADSCENEGRAPH_ROOT, ARTIFACT_ROOT, NEXUS_ROOT, PROJECT_ROOT, require_local_output, runtime_environment, write_json
from .manifest import load_manifest


def existing_adapter():
    if str(NEXUS_ROOT) not in sys.path:
        sys.path.insert(0,str(NEXUS_ROOT))
    path=NEXUS_ROOT/'scripts/pipeline/prepare_nuplan_3dgs.py'
    spec=importlib.util.spec_from_file_location('existing_nuplan_adscenegraph_adapter',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_dataset(manifest_path,run_dir,cameras,iterations):
    from omegaconf import OmegaConf
    data=load_manifest(manifest_path)
    if not cameras or any(str(c) not in data['calibrations'] for c in cameras):
        raise ValueError('Requested reconstruction cameras are not available')
    adapter=existing_adapter()
    scene=run_dir/'processed'/data['scene_id']
    for name in ('images','intrinsics','extrinsics','ego_pose','camera_ego_pose','instances','lidar'):
        (scene/name).mkdir(parents=True,exist_ok=True)
    for camera in cameras:
        cal=data['calibrations'][str(camera)]
        K=np.asarray(cal['intrinsics'])
        np.savetxt(scene/'intrinsics'/f'{camera}.txt',np.r_[K[0,0],K[1,1],K[0,2],K[1,2],cal['distortion']])
        np.savetxt(scene/'extrinsics'/f'{camera}.txt',cal['camera_to_ego'])
    lidar_tasks=[]
    for frame in data['frames']:
        index=frame['frame_index']
        np.savetxt(scene/'ego_pose'/f'{index:03d}.txt',frame['ego_to_world'])
        for image in frame['images']:
            c=image['camera_id']
            if c not in cameras:
                continue
            source=Path(image['image_path'])
            if not source.is_file():
                raise FileNotFoundError(source)
            adapter.link(source,scene/'images'/f'{index:03d}_{c}.jpg')
            np.savetxt(scene/'camera_ego_pose'/f'{index:03d}_{c}.txt',image['ego_to_world'])
        lidar_tasks.append((Path(frame['lidar_path']),scene/'lidar'/f'{index:03d}.bin'))
    write_json(scene/'instances/instances_info.json',data['instances_info'])
    write_json(scene/'instances/frame_instances.json',data['frame_instances'])
    with ThreadPoolExecutor(max_workers=2) as pool:
        counts=list(pool.map(adapter.convert_lidar,lidar_tasks))
    config_path=run_dir/'train_config.yaml'
    adapter.training_config(scene,cameras,config_path)
    config=OmegaConf.load(config_path)
    config.trainer.optim.num_iters=iterations
    config.logging.vis_freq=max(iterations,1)
    config.logging.saveckpt_freq=max(iterations,1)
    config.logging.print_freq=min(100,max(iterations,1))
    config.render.fps=data['fps']
    # A first clip should not generate a million random scene points before
    # the real calibrated initialization is verified.
    config.model.Background.init.from_lidar.num_samples=150000
    config.model.Background.init.near_randoms=10000
    config.model.Background.init.far_randoms=10000
    OmegaConf.save(config,config_path)
    write_json(scene/'scene_manifest.json',data)
    return {'scene_path':str(scene),'config_path':str(config_path),'point_counts':counts,'frame_count':data['frame_count']}


def finalize_existing(run_dir):
    """Render a completed native checkpoint; never retrain it during recovery."""
    run_dir=require_local_output(run_dir)
    state=json.loads((run_dir/'run.json').read_text())
    training_dir=run_dir/'training/nuplan/scene'
    checkpoint=training_dir/'checkpoint_final.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    def valid_videos():
        import cv2
        matches=[]
        expected=load_manifest(state['scene_manifest_path'])
        for path in (training_dir/'videos_eval').glob('full_set_*_rgbs.mp4'):
            capture=cv2.VideoCapture(str(path))
            count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps=capture.get(cv2.CAP_PROP_FPS)
            capture.release()
            if count==expected['frame_count'] and abs(fps-expected['fps'])<1e-3:
                matches.append(path)
        return matches
    choices=valid_videos()
    if not choices:
        command=[sys.executable,str(ADSCENEGRAPH_ROOT/'tools/eval.py'),'--resume_from',str(checkpoint),
                 '--full_eval_align_drivestudio']
        if state.get('error'):
            state['previous_failure']={'type':state.get('error_type'),'message':state['error']}
        state.update(status='rendering',render_command=command,checkpoint_path=str(checkpoint))
        write_json(run_dir/'run.json',state)
        log=run_dir/'render.log'
        with log.open('w') as stream:
            result=subprocess.run(command,cwd=ADSCENEGRAPH_ROOT,env=runtime_environment(),stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            state.update(status='error',error_type='RenderError',error=log.read_text(errors='replace')[-2500:])
            write_json(run_dir/'run.json',state)
            raise RuntimeError(f'ADSceneGraph checkpoint rendering failed; see {log}')
        choices=valid_videos()
    if not choices:
        raise RuntimeError(f'No rendered RGB video after checkpoint evaluation: {training_dir}')
    state.pop('error',None)
    state.pop('error_type',None)
    state.update(status='success',checkpoint_path=str(checkpoint),rendered_video=str(sorted(choices)[-1]),
        output_dir=str(training_dir),reconstruction_log=str(run_dir/'reconstruction.log'),render_log=str(run_dir/'render.log'),
        native_scene_format='adscenegraph_checkpoint',
        limitation='Native checkpoint and annotations are provided; DGGT edit/PLY consumers need a separate backend adapter.')
    write_json(run_dir/'run.json',state)
    return state


def reconstruct(payload):
    manifest_path=Path(payload['scene_manifest_path']).expanduser().resolve()
    scene=load_manifest(manifest_path)
    video=Path(payload.get('video_path') or scene['video_path']).expanduser().resolve()
    if not video.is_file() or hashlib.sha256(video.read_bytes()).hexdigest()!=scene['video_sha256']:
        raise ValueError('Reconstruction video and scene annotations are not the same exported clip')
    if payload.get('num_frames',scene['frame_count'])!=scene['frame_count']:
        raise ValueError('For ADSceneGraph, num_frames must match the annotated clip; do not truncate its annotations separately')
    iterations=payload.get('iterations',3000)
    if isinstance(iterations,bool) or not isinstance(iterations,int) or not 1<=iterations<=30000:
        raise ValueError('iterations must be an integer in 1..30000')
    cameras=payload.get('cameras',[0])
    if (not isinstance(cameras,list) or any(isinstance(c,bool) or not isinstance(c,int) or not 0<=c<8 for c in cameras)
        or len(cameras)!=len(set(cameras))):
        raise ValueError('cameras must be a list of distinct camera IDs')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    run_dir=require_local_output(payload.get('output_dir') or ARTIFACT_ROOT/'reconstruction'/f'run_{stamp}')
    if run_dir.exists():
        raise FileExistsError(f'Reconstruction output already exists: {run_dir}')
    run_dir.mkdir(parents=True)
    state={'backend':'adscenegraph','scene_id':scene['scene_id'],'scene_manifest_path':str(manifest_path),
        'source_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'video_path':str(video),'iterations':iterations,'cameras':cameras,'run_dir':str(run_dir),'status':'preparing'}
    write_json(run_dir/'run.json',state)
    try:
        state.update(prepare_dataset(manifest_path,run_dir,cameras,iterations),status='prepared')
        write_json(run_dir/'run.json',state)
        if payload.get('prepare_only'):
            return state
        training_dir=run_dir/'training'/'nuplan'/'scene'
        command=[sys.executable,str(ADSCENEGRAPH_ROOT/'tools/train.py'),'--config_file',state['config_path'],
                 '--output_root',str(run_dir/'training'),'--project','nuplan','--run_name','scene']
        state.update(status='training',command=command)
        write_json(run_dir/'run.json',state)
        env=runtime_environment()
        log=run_dir/'reconstruction.log'
        with log.open('w') as stream:
            completed=subprocess.run(command,cwd=ADSCENEGRAPH_ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f'ADSceneGraph failed ({completed.returncode}); log: {log}\n'+log.read_text(errors='replace')[-2500:])
        return finalize_existing(run_dir)
    except Exception as exc:
        if (run_dir/'run.json').is_file():
            state.update(json.loads((run_dir/'run.json').read_text()))
        state.update(status='error',error_type=type(exc).__name__,error=str(exc))
        write_json(run_dir/'run.json',state)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-manifest',type=Path)
    parser.add_argument('--iterations',type=int,default=3000)
    parser.add_argument('--cameras',type=int,nargs='+',default=[0])
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    payload={'scene_manifest_path':str(args.scene_manifest),'iterations':args.iterations,'cameras':args.cameras,'prepare_only':args.prepare_only} if args.scene_manifest else json.load(sys.stdin)
    try:
        with redirect_stdout(sys.stderr):
            result=reconstruct(payload)
        print(json.dumps({'status':'success','tool':'reconstruction','result':result},ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({'status':'error','error':{'type':type(exc).__name__,'message':str(exc)}},ensure_ascii=False))
        raise SystemExit(1)


if __name__=='__main__':
    main()
