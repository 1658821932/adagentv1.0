import copy
import hashlib
import json
from pathlib import Path
import tempfile

import pytest

from mcp_servers.nuplan_data.manifest import export_clip_manifest, index_metadata, load_manifest
from mcp_servers.sentry_video_search.config import SearchConfig


def scene_fixture(directory):
    video=directory/'front.mp4'
    video.write_bytes(b'unit-test-placeholder-not-an-acceptance-video')
    frames=[{'frame_index':i,'timestamp_us':1000000+i*100000,'grid_timestamp_us':1000000+i*100000,
             'images':[{'camera_id':0,'image_path':f'{i}.jpg'}]} for i in range(6)]
    pose=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    data={'schema_version':1,'dataset':'nuplan','status':'prepared','scene_id':'official-scene',
          'fps':10.,'frame_count':6,'frames':frames,'video_path':str(video),'video_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),
          'calibrations':{'0':{'camera_to_ego':pose}},
          'frame_instances':{str(i):[0] if i<2 else [7] for i in range(6)},
          'instances_info':{
              '0':{'id':'track-outside-crop','frame_annotations':{'frame_idx':[0,1],'obj_to_world':[pose]*2,'box_size':[[4,2,1]]*2}},
              '7':{'id':'track-retained','frame_annotations':{'frame_idx':[2,3,4,5],'obj_to_world':[pose]*4,'box_size':[[5,2,2]]*4}}},
          'reconstruction_backends':{'adscenegraph':{},'vision_only':{}}}
    path=directory/'scene_manifest.json'
    path.write_text(json.dumps(data))
    return path,data


def test_crop_preserves_world_geometry_original_time_and_track_identity(tmp_path):
    path,original=scene_fixture(tmp_path)
    output=tmp_path/'crop.json'
    scene=export_clip_manifest(path,tmp_path/'front.mp4',.2,.5,output)
    assert scene['frame_count']==3
    assert scene['start_timestamp_us']==1200000
    assert scene['end_timestamp_us_exclusive']==1500000
    assert [f['source_frame_index'] for f in scene['frames']]==[2,3,4]
    assert [f['video_timestamp_seconds'] for f in scene['frames']]==[0,.1,.2]
    assert set(scene['instances_info'])=={'0'}
    retained=scene['instances_info']['0']
    assert retained['id']=='track-retained'
    assert retained['source_instance_id']==7
    assert retained['frame_annotations']['frame_idx']==[0,1,2]
    assert retained['frame_annotations']['obj_to_world']==original['instances_info']['7']['frame_annotations']['obj_to_world'][:3]
    assert scene['frame_instances']=={'0':[0],'1':[0],'2':[0]}
    assert scene['calibrations']==original['calibrations']
    load_manifest(output)


@pytest.mark.parametrize('start,end',[(.15,.5),(-.1,.4),(.2,.8),(.4,.4)])
def test_invalid_or_unaligned_crop_is_rejected(tmp_path,start,end):
    path,_=scene_fixture(tmp_path)
    with pytest.raises(ValueError):
        export_clip_manifest(path,tmp_path/'front.mp4',start,end,tmp_path/'crop.json')


def test_wrong_video_cannot_inherit_another_scenes_annotations(tmp_path):
    scene_fixture(tmp_path)
    other=tmp_path/'unrelated.mp4'
    other.write_bytes(b'unrelated')
    with pytest.raises(ValueError,match='does not match'):
        index_metadata(other)


def test_index_records_scene_and_exact_video_identity(tmp_path):
    _,data=scene_fixture(tmp_path)
    metadata=index_metadata(tmp_path/'front.mp4')
    assert metadata['dataset']=='nuplan'
    assert metadata['scene_id']==data['scene_id']
    assert metadata['source_video_sha256']==data['video_sha256']
    assert metadata['scene_manifest_sha256']


def test_changed_source_video_cannot_keep_old_annotation_identity(tmp_path):
    scene_fixture(tmp_path)
    (tmp_path/'front.mp4').write_bytes(b'different content')
    with pytest.raises(ValueError,match='Source video changed'):
        index_metadata(tmp_path/'front.mp4')


@pytest.mark.parametrize('corruption',['timestamp','identity','dimensions'])
def test_inconsistent_annotation_handoff_is_rejected(tmp_path,corruption):
    path,data=scene_fixture(tmp_path)
    if corruption=='timestamp':
        data['frames'][2]['timestamp_us']=data['frames'][1]['timestamp_us']
    elif corruption=='identity':
        data['frame_instances']['2']=[0]
    else:
        data['instances_info']['7']['frame_annotations']['box_size'][0]=[-5,2,2]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_manifest(path)


def test_nuplan_library_uses_independent_index_and_keeps_sources_in_place():
    default=SearchConfig.for_library('default')
    waymo=SearchConfig.for_library('waymo')
    nuplan=SearchConfig.for_library('nuplan')
    assert nuplan.runtime_home!=default.runtime_home
    assert nuplan.runtime_home!=waymo.runtime_home
    assert nuplan.import_source_videos is False
    assert nuplan.chunk_duration==10
