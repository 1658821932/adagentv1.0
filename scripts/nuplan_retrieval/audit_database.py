"""Audit final official scenes, video/annotation identity and caption coverage."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from mcp_servers.nuplan_data.common import ARTIFACT_ROOT,DATA_ROOT,write_json
from mcp_servers.nuplan_data.manifest import load_manifest

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--require-complete-index',action='store_true')
args=parser.parse_args()
selection=json.loads((ARTIFACT_ROOT/'selection_final_300.json').read_text())
official=json.loads((ARTIFACT_ROOT/'official_navtrain_catalog.json').read_text())
members={item['name'] for item in official['members']}
source_audit_path=ARTIFACT_ROOT/'source_reference_audit.json'
source_audit=json.loads(source_audit_path.read_text()) if source_audit_path.is_file() else None
if args.require_complete_index:
    assert source_audit and source_audit['status']=='passed', 'Raw source reference audit is required'
    assert source_audit['selection_sha256']==hashlib.sha256((ARTIFACT_ROOT/'selection_final_300.json').read_bytes()).hexdigest()
    assert source_audit['camera_references_checked']==240000 and source_audit['lidar_references_checked']==30000
    assert source_audit['database_references_checked']==300 and not source_audit['errors']
dbpath=DATA_ROOT/'runtime/.sentrysearch/kimi_caption.sqlite3'
with sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True) as db:
    db.row_factory=sqlite3.Row
    indexed={row['source_file']:dict(row) for row in db.execute('SELECT * FROM chunks')}
records=[]
errors=[]
expected=set()
for row in selection['scenes']:
    try:
        path=DATA_ROOT/'scenes'/row['scene_id']/'scene_manifest.json'
        data=load_manifest(path)
        assert data['identity']==row,'Selection and prepared scene identities differ'
        assert row['official_config_member'] in members
        assert row['official_revision']==selection['official_revision']
        assert len(data['frames'])==100 and len(data['calibrations'])==8
        assert set(data['calibrations'])=={str(i) for i in range(8)}
        assert Path(row['database']).stem==row['log_name']
        assert Path(data['map_reference']['root']).is_dir()
        camera_times=[[] for _ in range(8)]
        max_offset=0
        for frame in data['frames']:
            assert {image['camera_id'] for image in frame['images']}==set(range(8))
            for image in frame['images']:
                camera_times[image['camera_id']].append(image['timestamp_us'])
                max_offset=max(max_offset,abs(image['timestamp_us']-frame['timestamp_us']))
                pose=np.asarray(image['ego_to_world'])
                assert pose.shape==(4,4) and np.isfinite(pose).all()
                assert np.allclose(pose[3],[0,0,0,1])
        assert all(np.all(np.diff(times)>0) for times in camera_times)
        for calibration in data['calibrations'].values():
            pose=np.asarray(calibration['camera_to_ego']);K=np.asarray(calibration['intrinsics'])
            assert pose.shape==(4,4) and np.isfinite(pose).all()
            assert abs(np.linalg.det(pose[:3,:3])-1)<1e-6
            assert K.shape==(3,3) and K[0,0]>0 and K[1,1]>0
        video=Path(data['video_path']).resolve()
        digest=hashlib.sha256(video.read_bytes()).hexdigest()
        assert digest==data['video_sha256']
        cap=cv2.VideoCapture(str(video))
        assert cap.isOpened()
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        assert count==100 and abs(fps-10)<1e-4 and (width,height)==(960,540)
        expected.add(str(video))
        index=indexed.get(str(video))
        if index:
            meta=json.loads(index['metadata'])
            assert index['video_sha256']==digest
            assert meta['scene_id']==data['scene_id']
            assert meta['scene_manifest_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({'scene_id':data['scene_id'],'city':row['city'],'frames':count,
            'cameras':8,'instances':len(data['instances_info']),'indexed':index is not None,
            'max_camera_lidar_offset_us':max_offset})
    except Exception as exc:
        errors.append({'scene_id':row['scene_id'],'type':type(exc).__name__,'error':str(exc)})
missing=sorted(expected-set(indexed))
unexpected=sorted(set(indexed)-expected)
report={'status':'passed' if not errors and (not args.require_complete_index or not missing and not unexpected) else 'error',
    'official_revision':selection['official_revision'],'selected_scene_count':len(selection['scenes']),
    'validated_scene_count':len(records),'city_counts':dict(Counter(row['city'] for row in records)),
    'video_frames':sum(row['frames'] for row in records),'camera_image_references':sum(row['frames']*row['cameras'] for row in records),
    'selected_lidar_references':sum(row['frames'] for row in records),
    'total_video_seconds':sum(row['frames']/10 for row in records),
    'indexed_scene_count':sum(row['indexed'] for row in records),'errors':errors,
    'missing_index_videos':missing,'unexpected_index_videos':unexpected,
    'records':records,'require_complete_index':args.require_complete_index,
    'source_reference_audit':str(source_audit_path),'source_reference_audit_passed':bool(source_audit and source_audit['status']=='passed')}
write_json(ARTIFACT_ROOT/'database_audit.json',report)
print(json.dumps({k:v for k,v in report.items() if k not in ('records','missing_index_videos')},ensure_ascii=False,indent=2))
if report['status']!='passed':
    raise SystemExit(1)
