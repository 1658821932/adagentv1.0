"""Prepare original nuPlan RGB clips and aligned reconstruction annotations."""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from .common import ARTIFACT_ROOT, DATA_ROOT, RAW_ROOT, nuplan_reader, require_local_output, write_json
from .manifest import load_manifest,index_metadata


def nearest(rows, times, target, tolerance=50000):
    i = bisect_left(times, target)
    candidates = [j for j in (i-1,i) if 0<=j<len(times)]
    if not candidates:
        raise ValueError('Missing camera frames in requested interval')
    j = min(candidates,key=lambda j:(abs(times[j]-target),times[j]))
    if abs(times[j]-target)>tolerance:
        raise ValueError(f'Camera/LiDAR timestamp mismatch: {times[j]-target} us')
    return rows[j]


def prepare_scene(selection, *, data_root=DATA_ROOT):
    nup = nuplan_reader()
    scene_id = selection['scene_id']
    directory = require_local_output(Path(data_root)/'scenes'/scene_id)
    manifest_path = directory/'scene_manifest.json'
    video_path = directory/'front.mp4'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get('status')=='prepared' and old['identity']==selection and video_path.is_file():
            load_manifest(manifest_path)
            index_metadata(video_path)
            return {'scene_id':scene_id,'scene_manifest_path':str(manifest_path),'video_path':str(video_path),'status':'reused'}
        raise ValueError(f'Conflicting or incomplete scene manifest: {manifest_path}')
    locator = nup.SensorLocator([RAW_ROOT/'sensor_blobs_full', RAW_ROOT/'sensor_blobs_mini'])
    calibrations, frames, instance_info = {}, [], {}
    with nup.connect(selection['database']) as db:
        cals = nup.cameras(db)
        # Use the front camera clock as the video sampling grid; LiDAR poses
        # and boxes remain tied to their nearest genuine LiDAR timestamps.
        first_image = nup.nearest_image(db,cals['CAM_F0']['token'],selection['start_timestamp_us'])
        start = int(first_image['image_timestamp'])
        targets = start+np.rint(np.arange(selection['frame_count'])*1e6/selection['fps']).astype(np.int64)
        lidars = nup.nearest_frames(db,targets)
        image_rows = defaultdict(list)
        for row in db.execute(
            'SELECT im.token AS image_token, im.camera_token, im.filename_jpg, im.timestamp AS image_timestamp, ep.* '
            'FROM image im JOIN ego_pose ep ON ep.token=im.ego_pose_token '
            'WHERE im.timestamp BETWEEN ? AND ? ORDER BY im.timestamp',
            (int(targets[0])-100000,int(targets[-1])+100000)):
            image_rows[row['camera_token']].append(dict(row))
        image_times = {token:[int(r['image_timestamp']) for r in rows] for token,rows in image_rows.items()}
        objects_by_lidar = defaultdict(list)
        tokens = [r['lidar_token'] for r in lidars]
        placeholders = ','.join('?' for _ in tokens)
        for row in db.execute(
            'SELECT lb.*,c.name AS category_name FROM lidar_box lb JOIN track t ON t.token=lb.track_token '
            'JOIN category c ON c.token=t.category_token WHERE lb.lidar_pc_token IN ('+placeholders+')',tokens):
            item = dict(row)
            item['track_token'] = item['track_token'].hex()
            objects_by_lidar[item['lidar_pc_token']].append(item)
        track_tokens = sorted({o['track_token'] for rows in objects_by_lidar.values() for o in rows if o['category_name'] in ('vehicle','pedestrian','bicycle')})
        ids = {token:i for i,token in enumerate(track_tokens)}
        frame_instances = {}
        for channel,cal in cals.items():
            calibrations[str(cal['id'])] = {'channel':channel,'intrinsics':cal['intrinsics'].tolist(),
                'distortion':cal['distortion'].tolist(),'camera_to_ego':cal['extrinsics'].tolist(),
                'width':int(cal['width']),'height':int(cal['height'])}
        used_images = defaultdict(set)
        for index,(target,lidar) in enumerate(zip(targets,lidars)):
            lidar_path = locator.resolve(lidar['filename'])
            if lidar_path is None:
                raise FileNotFoundError(lidar['filename'])
            images = []
            for channel,cal in cals.items():
                token = cal['token']
                image = nearest(image_rows[token],image_times[token],int(target))
                image_path = locator.resolve(image['filename_jpg'])
                if image_path is None:
                    raise FileNotFoundError(image['filename_jpg'])
                if image['image_token'] in used_images[channel]:
                    raise ValueError(f'Repeated source image in {channel}; timestamps cannot be represented as unique video frames')
                used_images[channel].add(image['image_token'])
                images.append({'camera_id':cal['id'],'channel':channel,'timestamp_us':int(image['image_timestamp']),
                    'image_token':image['image_token'].hex(),'image_path':str(image_path),
                    'ego_to_world':nup.ego_pose(image).tolist()})
            frames.append({'frame_index':index,'video_timestamp_seconds':index/selection['fps'],
                'grid_timestamp_us':int(target),'timestamp_us':int(lidar['lidar_timestamp']),
                'lidar_token':lidar['lidar_token'].hex(),'lidar_path':str(lidar_path),
                'ego_to_world':nup.ego_pose(lidar).tolist(),'images':images})
            frame_instances[str(index)] = []
            for obj in sorted(objects_by_lidar[lidar['lidar_token']],key=lambda o:o['track_token']):
                if obj['track_token'] not in ids:
                    continue
                instance_id = ids[obj['track_token']]
                record = instance_info.setdefault(str(instance_id),{'id':obj['track_token'],'class_name':obj['category_name'],
                    'frame_annotations':{'frame_idx':[],'obj_to_world':[],'box_size':[]}})
                ann = record['frame_annotations']
                ann['frame_idx'].append(index)
                ann['obj_to_world'].append(nup.box_pose(obj).tolist())
                ann['box_size'].append([float(obj['length']),float(obj['width']),float(obj['height'])])
                frame_instances[str(index)].append(instance_id)
    directory.mkdir(parents=True,exist_ok=True)
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory(prefix='video-',dir=directory) as temporary:
        temp = Path(temporary)
        for frame in frames:
            front = next(im for im in frame['images'] if im['camera_id']==0)
            (temp/f"{frame['frame_index']:06d}.jpg").symlink_to(front['image_path'])
        temporary_video = directory/'front.tmp.mp4'
        command = [ffmpeg,'-nostdin','-hide_banner','-loglevel','error','-y','-framerate',str(selection['fps']),
            '-i',str(temp/'%06d.jpg'),'-frames:v',str(len(frames)),'-vf','scale=960:540',
            '-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-threads','2',str(temporary_video)]
        result = subprocess.run(command,capture_output=True,text=True)
        if result.returncode:
            raise RuntimeError(result.stderr[-2000:])
        temporary_video.replace(video_path)
    max_camera_offset = max(abs(im['timestamp_us']-f['timestamp_us']) for f in frames for im in f['images'])
    manifest = {'schema_version':1,'dataset':'nuplan','status':'prepared','identity':selection,
        'scene_id':scene_id,'video_path':str(video_path),'fps':selection['fps'],'frame_count':len(frames),
        'video_size':[960,540],'video_camera_id':0,'duration_seconds':len(frames)/selection['fps'],
        'start_timestamp_us':int(targets[0]),'end_timestamp_us_exclusive':int(targets[0]+len(frames)*1e6/selection['fps']),
        'coordinate_system':{'world':'nuPlan global ENU/map frame','ego':'forward-left-up','camera':'OpenCV right-down-forward','length_unit':'meter','timestamp_unit':'microsecond'},
        'map_reference':{'root':str(RAW_ROOT/'maps'),'location':selection['city'],'version':'nuplan-maps-v1.0'},
        'calibrations':calibrations,'frames':frames,'instances_info':instance_info,'frame_instances':frame_instances,
        'annotation_provenance':{'calibrations':'original nuPlan DB','ego_poses':'original ego_pose table at sensor exposure/scan times',
            'instances':'original track/category/lidar_box tables','boxes_timestamp':'nearest genuine lidar scan, not interpolated camera-time ground truth',
            'semantic_masks':'not supplied; related reconstruction supervision must be disabled',
            'smpl':'not supplied; SMPL reconstruction branch disabled'},
        'validation':{'all_images_exist':True,'all_lidar_exists':True,'unique_images_per_camera':True,
            'max_camera_lidar_offset_us':max_camera_offset,'num_instances':len(instance_info)},
        'video_sha256':hashlib.sha256(video_path.read_bytes()).hexdigest(),
        'reconstruction_backends':{'adscenegraph':{'supported':True,'scene_manifest_path':str(manifest_path)},
            'vision_only':{'video_path':str(video_path),'annotation_tool_required':True}}}
    if max_camera_offset>100000:
        raise ValueError(f'Camera/LiDAR offset exceeds valid combined tolerances: {max_camera_offset}')
    validation_path=directory/'scene_manifest.check.json'
    write_json(validation_path,manifest)
    load_manifest(validation_path)
    validation_path.replace(manifest_path)
    return {'scene_id':scene_id,'scene_manifest_path':str(manifest_path),'video_path':str(video_path),'status':'prepared','validation':manifest['validation']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,default=ARTIFACT_ROOT/'selection_300.json')
    parser.add_argument('--limit',type=int,default=5)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    scenes=json.loads(args.selection.read_text())['scenes']
    if args.limit>0:
        scenes=scenes[:args.limit]
    records=[]
    report_path=ARTIFACT_ROOT/f'preparation_{len(scenes)}.json'
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(prepare_scene,row):row for row in scenes}
        for future in as_completed(futures):
            row=futures[future]
            try:
                record=future.result()
            except Exception as exc:
                record={'scene_id':row['scene_id'],'status':'error','error_type':type(exc).__name__,'error':str(exc)}
            records.append(record)
            write_json(report_path,{'requested':len(scenes),'completed':len(records),'records':records})
            print(json.dumps(record,ensure_ascii=False),flush=True)
    if any(r['status']=='error' for r in records):
        raise SystemExit('Some scenes failed; see preparation report')


if __name__=='__main__':
    main()
