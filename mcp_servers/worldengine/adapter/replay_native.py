"""Execute native WE replay under the simulator environment, without AlgEngine."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import pickle
import runpy
import sys
import subprocess
import numpy as np


def prepare(request):
    with Path(request['scenario_pkl']).open('rb') as source:
        scenes=pickle.load(source)
    key=request['scene_id']
    if not isinstance(scenes,dict) or key not in scenes:
        raise ValueError('Requested scene key is absent from the native package: '+key)
    scene=deepcopy(scenes[key])
    cameras=scene.get('cameras',{})
    selected=request.get('cameras')
    if selected:
        unknown=set(selected)-set(cameras)
        if unknown:raise ValueError('Unknown replay cameras: '+', '.join(sorted(unknown)))
        scene['cameras']={name:cameras[name] for name in selected}
    if not scene.get('cameras'):raise ValueError('Native scene has no cameras')
    for camera in scene['cameras'].values():
        for field in ('sensor2ego_rotation','sensor2ego_translation'):
            if field in camera:camera[field]=np.asarray(camera[field],dtype=float)
    # DataManager reads camera metadata separately from the render manager.
    original = scene.get('metadata', {}).get('openscene_data_infos_dict', {})
    for frame in original.values():
        if isinstance(frame, dict) and 'cams' in frame:
            frame['cams'] = {name: value for name, value in frame['cams'].items() if name in scene['cameras']}
    metadata=scene.get('metadata',{})
    for frame in metadata.get('openscene_data_infos_dict',{}).values():
        frame['cams']={name:camera for name,camera in frame.get('cams',{}).items() if name in scene['cameras']}
    if 'synthetic_scene_info' in metadata:
        for frame in metadata['synthetic_scene_info'].get('frames',[]):
            allowed={name.lower() for name in scene['cameras']}
            frame['camera_dict']={name:camera for name,camera in frame.get('camera_dict',{}).items() if name.lower() in allowed}
    length=int(scene.get('log_length',scene.get('length',0)))
    if length<2:raise ValueError('Replay requires at least two native scene states')
    staged=Path(request['staged_scenario']);staged.parent.mkdir(parents=True,exist_ok=True)
    with staged.open('wb') as target:pickle.dump({key:scene},target,protocol=4)
    output=Path(request['output_dir'])
    if output.exists():raise FileExistsError('Replay output must be a new directory')
    history=min(4,length-1)
    arguments=[
        'data_file_path='+str(staged), 'asset_folder_path='+request['asset_folder_path'],
        'output_dir='+str(output),'data_output_dir='+str(output/'openscene_format'),
        'ego_policy=trajectory_policy','ego_controller=log_play_controller',
        'ego_navigation=trajectory_navigation','ego_client=base_client',
        'agent_policy=trajectory_policy','agent_controller=log_play_controller',
        'agent_navigation=trajectory_navigation','use_planner_actions=false',
        'with_render_manager=true','with_metric_manager=false','with_dense_reward_manager=false',
        'visualize_video=false','save_data=true','enable_resume=false',
        'hydra.job.chdir=false',
        'num_history='+str(history),'num_future='+str(length-history),
        'distributed_mode=SINGLE_NODE',
    ]
    return scene,arguments


def export_videos(output,scene,ffmpeg='ffmpeg'):
    """Encode WE-rendered camera images, without the fixed eight-camera visualizer."""
    import cv2
    import re
    import subprocess
    metas=sorted((output/'openscene_format/meta_datas').glob('*.pkl'))
    if len(metas)!=1:raise RuntimeError('Expected one replay metadata file')
    with metas[0].open('rb') as source:frames=pickle.load(source)
    if not isinstance(frames,list) or len(frames)!=int(scene.get('log_length',scene.get('length',0))):
        raise RuntimeError('Native replay did not export the full requested time window')
    videos=output/'videos';videos.mkdir(exist_ok=False)
    encoded={}
    for camera in scene['cameras']:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',camera):raise ValueError('Invalid camera name')
        sequence=videos/camera;sequence.mkdir()
        for index,frame in enumerate(frames):
            name=frame['cams'][camera]['data_path']
            source=(output/'openscene_format/sensor_blobs'/name).resolve()
            allowed=(output/'openscene_format/sensor_blobs').resolve()
            if not source.is_relative_to(allowed) or not source.is_file():
                raise ValueError('WE rendered camera image is missing or outside its output directory')
            image=cv2.imread(str(source))
            if image is None:raise RuntimeError('WE camera image cannot be decoded')
            (sequence/f'{index:06d}.jpg').symlink_to(source)
        step=float(scene.get('sample_rate',10))*.05
        if step<=0:raise ValueError('Native sample interval must be positive')
        target=videos/(camera+'.mp4')
        completed=subprocess.run([ffmpeg,'-v','error','-nostdin','-framerate',str(1/step),
            '-i',str(sequence/'%06d.jpg'),'-c:v','libx264','-pix_fmt','yuv420p',
            '-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-movflags','+faststart',str(target)],
            capture_output=True,text=True,timeout=180)
        if completed.returncode:raise RuntimeError('WE replay video export failed for '+camera)
        encoded[camera]=len(frames)
    return encoded


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args();request=json.loads(args.request.read_text())
    scene,arguments=prepare(request)
    output=Path(request['output_dir']);output.mkdir(parents=True,exist_ok=False)
    os.chdir(output)
    root=Path(request['worldengine_root']);os.environ['WORLDENGINE_ROOT']=str(root)
    sim=root/'projects/SimEngine';sys.path.insert(0,str(sim))
    if request.get('render_python'):
        os.environ['WE_NATIVE_RENDER_REQUEST']=str(args.request.resolve())
        from worldengine.manager.agent_manager import BaseAgentManager
        BaseAgentManager.STATIC_THRESHOLD=-1.
        arguments.append('renderer._target_=replay_gpu_bridge.NativeRenderBridge')
    elif scene.get('metadata',{}).get('native_export'):
        from replay_pose import install
        install()
    entry=sim/'worldengine/runner/run_simulation.py'
    sys.argv=[str(entry),*arguments]
    runpy.run_path(str(entry),run_name='__main__')
    # WE may be configured to continue after per-scene failure; reports must be inspected.
    reports=[]
    for path in Path(request['output_dir']).rglob('runner_report_*.json'):
        rows=json.loads(path.read_text())
        if not isinstance(rows,list) or not rows:
            raise RuntimeError('Malformed native runner report')
        for item in rows:
            if not isinstance(item,dict) or item.get('succeeded') is not True:
                raise RuntimeError('Native WE reported replay failure')
            reports.append(item)
    if not reports:raise RuntimeError('No native runner completion report was produced')
    video_frames=export_videos(output,scene,request.get('ffmpeg','ffmpeg'))
    Path(request['report_path']).write_text(json.dumps(dict(status='completed',scene_id=request['scene_id'],
        frame_count=scene.get('log_length',scene.get('length',0)),cameras=list(scene['cameras']),videos=video_frames,runner_reports=reports),indent=2))

if __name__=='__main__':main()
