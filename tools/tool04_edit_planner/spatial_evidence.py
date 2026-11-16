"""Offline BEV/trajectory reader, isolated from the Agent's lightweight environment."""
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]
CORE = PROJECT/'mcp_servers/scene_editing/core/adversarial'


def read(scene, output, track_id=None, max_samples=24):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    command=[str(PROJECT/'environments/scene_editing/bin/python'),'-B',str(Path(__file__).resolve()),
        '--scene',str(scene),'--output',str(output),'--max-samples',str(max_samples)]
    if track_id is not None:command+=['--track-id',track_id]
    try:
        env=dict(os.environ,PYTHONPATH=str(PROJECT)+os.pathsep+str(CORE)+os.pathsep+os.environ.get('PYTHONPATH',''))
        with (output/'worker.log').open('w') as log:
            result=subprocess.run(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=45)
        path=output/'spatial-evidence.json'
        if result.returncode or not path.is_file():
            return dict(status='unavailable',reason='Offline geometry reader failed; see worker.log',report_path=str(path))
        return json.loads(path.read_text())
    except (OSError,subprocess.TimeoutExpired) as error:
        return dict(status='unavailable',reason=type(error).__name__,report_path=str(output/'spatial-evidence.json'))


def build(scene, output, track_id=None, max_samples=24):
    import numpy as np
    sys.path.insert(0,str(CORE))
    from run_ego_reference import ego_path_from_frames, observed_objects_from_frames
    from ego_road_checks import load_official_drivable
    from decision_scene_view import render_overview, actor_view
    scene=Path(scene);output=Path(output)
    frames=json.loads((scene/'metadata/frames.json').read_text())
    if len(frames)<2 or any(not all(key in frame for key in ('ego2global','lidar2ego')) for frame in frames):
        raise ValueError('Recorded transforms and at least two frames are required for a coherent BEV')
    times,ego,yaw=ego_path_from_frames(frames);actors=observed_objects_from_frames(frames)
    roster={}
    for index,objects in enumerate(actors):
        for actor in objects:
            if actor['label']!='vehicle':continue
            row=roster.setdefault(actor['track'],dict(track_id=actor['track'],diagram_label=actor['track'][:6],observed_frames=0))
            row['observed_frames']+=1
            distance=float(np.linalg.norm([actor['x']-ego[index,0],actor['y']-ego[index,1]]))
            row['closest_recorded_ego_center_distance_m']=round(min(distance,row.get('closest_recorded_ego_center_distance_m',float('inf'))),2)
    if track_id is not None and track_id!='ego' and track_id not in roster:
        raise ValueError('Unknown vehicle track ID; use the supplied catalog')
    road_path=scene/'metadata/map_drivable_official.json'
    road=load_official_drivable(road_path) if road_path.is_file() else None
    target=None if track_id=='ego' else track_id
    diagram_actors=[[actor for actor in objects if actor['label']=='vehicle'] for objects in actors]
    path=render_overview(times,ego,yaw,diagram_actors,road,output/'scene-overview.png',target_track=target)
    visible=actor_view(times,ego,diagram_actors,target)
    report=dict(status='available',image_path=str(path),
        coordinate_frame='initial ego rear axle: x forward, y left, metres; yaw radians',
        scope='Original recorded scene geometry and ego reference, not an edited scene or policy rollout. Vehicles are shown for readability; only the listed vehicles are labelled. Nonvehicles and omitted vehicles still exist; use cameras and full evidence for obstacles.',
        map_available=road is not None,diagram_vehicle_labels=[dict(track_id=row['track_id'],diagram_label=row['diagram_label']) for row in visible['actors']],
        vehicle_catalog=list(roster.values()),
        ego_summary=dict(duration_s=round(float(times[-1]),3),recorded_path_length_m=round(float(np.linalg.norm(np.diff(ego,axis=0),axis=1).sum()),2)),
        report_path=str(output/'spatial-evidence.json'))
    if track_id is not None:
        indices=np.linspace(0,len(times)-1,min(max_samples,len(times)),dtype=int)
        points=[]
        for index in indices:
            if track_id=='ego':position=ego[index];angle=yaw[index]
            else:
                actor=next((item for item in actors[index] if item['track']==track_id),None)
                if actor is None:
                    points.append(dict(time_s=round(float(times[index]),3),observed=False));continue
                position=[actor['x'],actor['y']];angle=actor['yaw']
            points.append(dict(time_s=round(float(times[index]),3),observed=True,
                center_xy_m=[round(float(value),2) for value in position],heading_rad=round(float(angle),3),
                ego_rear_axle_xy_m=[round(float(value),2) for value in ego[index]]))
        report['trajectory']=dict(track_id=track_id,points=points,source_frame_count=len(times),
            max_samples=max_samples,position_reference='ego_rear_axle' if track_id=='ego' else 'actor_body_center',
            scope='Sampled observations; missing samples stay unknown. No missing pose is fabricated.')
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--scene',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--track-id');parser.add_argument('--max-samples',type=int,default=24)
    args=parser.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    try:report=build(args.scene,output,args.track_id,args.max_samples)
    except (OSError,ValueError,KeyError,TypeError) as error:
        report=dict(status='unavailable',reason=str(error),error_type=type(error).__name__)
    (output/'spatial-evidence.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
