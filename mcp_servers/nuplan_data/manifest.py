"""Versioned scene handoff and frame-aligned retrieval clip annotations."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from .common import write_json


def load_manifest(path):
    path=Path(path).expanduser().resolve()
    data=json.loads(path.read_text())
    if data.get('schema_version')!=1 or data.get('dataset')!='nuplan' or data.get('status')!='prepared':
        raise ValueError('Expected prepared nuPlan scene_manifest schema_version=1')
    frames=data.get('frames') or []
    fps=data.get('fps')
    if not isinstance(fps,(int,float)) or not math.isfinite(fps) or fps<=0:
        raise ValueError('Invalid scene FPS')
    if not frames or len(frames)!=data.get('frame_count'):
        raise ValueError('Scene frame count mismatch')
    for index,frame in enumerate(frames):
        if frame['frame_index']!=index:
            raise ValueError('Scene frames must be contiguous and zero based')
        if index and frame['timestamp_us']<=frames[index-1]['timestamp_us']:
            raise ValueError('LiDAR timestamps must increase strictly')
        ids=[im['camera_id'] for im in frame['images']]
        if 0 not in ids or len(ids)!=len(set(ids)):
            raise ValueError('Missing front camera or repeated camera IDs')
        for image in frame['images']:
            if str(image['camera_id']) not in data['calibrations']:
                raise ValueError('Image has no matching calibration')
        for instance in data['frame_instances'][str(index)]:
            record=data['instances_info'][str(instance)]
            if index not in record['frame_annotations']['frame_idx']:
                raise ValueError('Instance identity/frame annotations disagree')
    for record in data['instances_info'].values():
        annotations=record['frame_annotations']
        lengths=[len(annotations[k]) for k in ('frame_idx','obj_to_world','box_size')]
        if len(set(lengths))!=1 or any(i<0 or i>=len(frames) for i in annotations['frame_idx']):
            raise ValueError('Invalid instance annotation lengths or frame indices')
        if any(not math.isfinite(v) or v<=0 for box in annotations['box_size'] for v in box):
            raise ValueError('Instance dimensions must be finite positive meters')
    return data


def index_metadata(video):
    path=Path(video).parent/'scene_manifest.json'
    if not path.is_file():
        return {}
    data=load_manifest(path)
    if Path(data['video_path']).resolve()!=Path(video).resolve():
        raise ValueError('Video does not match adjacent scene manifest')
    if hashlib.sha256(Path(video).read_bytes()).hexdigest()!=data['video_sha256']:
        raise ValueError('Source video changed after scene annotations were prepared')
    return {'dataset':'nuplan','scene_id':data['scene_id'],'scene_manifest_path':str(path.resolve()),
        'source_video_sha256':data['video_sha256'],'scene_manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def export_clip_manifest(source, video_path, start_seconds, end_seconds, output):
    data=load_manifest(source)
    fps=data['fps']
    start=int(round(start_seconds*fps))
    end=int(round(end_seconds*fps))
    if abs(start-start_seconds*fps)>1e-4 or abs(end-end_seconds*fps)>1e-4:
        raise ValueError('Annotated nuPlan clips require frame-aligned crop boundaries')
    if not 0<=start<end<=len(data['frames']):
        raise ValueError('Crop interval outside annotated scene')
    result=copy.deepcopy(data)
    result['frames']=result['frames'][start:end]
    for index,frame in enumerate(result['frames']):
        frame['source_frame_index']=frame['frame_index']
        frame['frame_index']=index
        frame['video_timestamp_seconds']=index/fps
    result['instances_info']={}
    result['frame_instances']={str(i):[] for i in range(end-start)}
    for key,record in data['instances_info'].items():
        annotations=record['frame_annotations']
        keep=[j for j,index in enumerate(annotations['frame_idx']) if start<=index<end]
        if not keep:
            continue
        new=copy.deepcopy(record)
        new['frame_annotations']={
            'frame_idx':[annotations['frame_idx'][j]-start for j in keep],
            'obj_to_world':[annotations['obj_to_world'][j] for j in keep],
            'box_size':[annotations['box_size'][j] for j in keep]}
        result['instances_info'][key]=new
        for index in new['frame_annotations']['frame_idx']:
            result['frame_instances'][str(index)].append(int(key))
    # The ADSceneGraph loader allocates a dense [frames, instances] tensor.
    # A crop can remove lower-numbered tracks, so compact local IDs while
    # keeping genuine track tokens and original instance IDs for provenance.
    local_ids={old:i for i,old in enumerate(sorted(result['instances_info'],key=int))}
    result['instances_info']={str(local_ids[key]):{**record,'source_instance_id':record.get('source_instance_id',int(key))}
        for key,record in result['instances_info'].items()}
    result['frame_instances']={key:[local_ids[str(i)] for i in ids] for key,ids in result['frame_instances'].items()}
    video=Path(video_path).resolve()
    result.update(video_path=str(video),video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
        frame_count=end-start,duration_seconds=(end-start)/fps,
        start_timestamp_us=result['frames'][0]['grid_timestamp_us'],
        end_timestamp_us_exclusive=int(result['frames'][-1]['grid_timestamp_us']+1e6/fps),
        source_scene_manifest_path=str(Path(source).resolve()),
        clip={'start_seconds':start_seconds,'end_seconds_exclusive':end_seconds,
              'source_start_frame':start,'source_end_frame_exclusive':end})
    result['reconstruction_backends']['adscenegraph']['scene_manifest_path']=str(Path(output).resolve())
    result['reconstruction_backends']['vision_only']['video_path']=str(video)
    write_json(output,result)
    load_manifest(output)
    return result
