"""Rendering height comes from reconstruction even when UTM starts underground."""
from datetime import date
from pathlib import Path
from types import SimpleNamespace
import json
import pickle
import sys

import numpy as np
import pytest

CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'
if str(CORE) not in sys.path:sys.path.insert(0,str(CORE))
from reconstruction_calibration import ReconstructionCalibration, load_reconstruction_calibration, recorded_pose_at
from road_height import RoadHeightMap, ego_height_reference
from mcp_servers.worldengine.adapter.replay_pose import shift_ego_calibration


def reference():
    poses=np.tile(np.eye(4),(3,1,1));poses[:,0,3]=[0,1,2]
    angle=.04;poses[:,:3,:3]=[[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]]
    poses[:,2,3]=[-1.65,-1.55,-1.45]
    return ReconstructionCalibration(poses,[100,200,19],{'source':'offline measured reconstruction'})


class Road:
    def try_height_at(self,x,y):return -2+.1*x+.2*y
    def height_with_local_plane(self,x,y):return {'height':self.try_height_at(x,y)}


def test_recorded_comparison_uses_native_reference_when_road_is_unknown():
    class MissingRoad:
        def height_with_local_plane(self,*args):raise ValueError('No surrounding road measurements')
    value=reference()
    pose,ground=recorded_pose_at(value,[1,0],.3,MissingRoad())
    assert pose[2,3]==pytest.approx(value.poses[1,2,3])
    assert ground['height'] is None and ground['road_height_status']=='unavailable'
    with pytest.raises(ValueError,match='road measurements'):
        value.pose_at([1,0],.3,MissingRoad())


def test_height_anchor_is_independent_of_bad_frame_zero_and_later_utm_drift():
    calibration=reference();frames=[]
    for i in range(3):
        p=np.eye(4);p[:3,3]=[100+i,200,14-i*2]
        frames.append({'ego2global':p.tolist()})
    result=ego_height_reference(frames,Road(),[100,200,19],calibration=calibration)
    np.testing.assert_allclose(result['profile_m'],[.35,.35,.35])
    assert result['reference_m']==pytest.approx(.35)
    for f in frames:f['ego2global'][2][3]-=100
    after=ego_height_reference(frames,Road(),[100,200,19],calibration=calibration)
    assert after==result


def test_first_frame_height_hole_uses_surrounding_measured_plane():
    axis = np.linspace(-4., 4., 41)
    x, y = np.meshgrid(axis, axis)
    heights = -2. + .1*x + .2*y
    heights[(abs(x) <= .9) & (abs(y) <= .9)] = np.nan
    road = RoadHeightMap(heights, np.eye(2), [4., 4.], 5.)
    with pytest.raises(ValueError, match='insufficient surrounding measurements'):
        road.height_with_local_plane(0., 0., radius_m=.75)
    ground = road.height_with_local_plane(0., 0.)
    assert ground['source'] == 'enclosed_local_plane'
    assert ground['height'] == pytest.approx(-2.)
    pose = np.eye(4)
    pose[:3, 3] = [100., 200., 1.]  # An unrelated UTM height.
    result = ego_height_reference([{'ego2global': pose}], road, [100., 200., 19.],
                                  calibration=reference())
    assert result['reference_m'] == pytest.approx(.35)
    with pytest.raises(ValueError, match='outside the road height map'):
        road.height_with_local_plane(4.1, 0.)


def test_edited_pose_keeps_requested_xy_yaw_and_measured_road_clearance():
    calibration=reference();before=calibration.poses.copy()
    xy=[1.2,2.];pose=calibration.pose_at(xy,.7,Road())
    np.testing.assert_allclose(pose[:2,3],xy)
    assert np.arctan2(pose[1,0],pose[0,0])==pytest.approx(.7)
    assert pose[2,3]-Road().try_height_at(*xy)==pytest.approx(.35)
    assert pose[2,0]==pytest.approx(-np.sin(.04))
    np.testing.assert_array_equal(calibration.poses,before)
    native=calibration.pose_at(xy,.7)
    assert native[2,3]==pytest.approx(before[1,2,3])


def write_video(tmp_path,duplicate=False):
    calibration=reference();rows=[dict(token=f't{i}',timestamp=1000000+i*100000,
        ego2global=pose,skipped=i==0) for i,pose in enumerate(calibration.poses)]
    video=dict(video_token='asset-0',global2world_translation=[100,200,19],
        date=date(2021,10,5),frame_infos=rows)
    videos={'asset-0':video}
    if duplicate:videos['another']=dict(video,video_token='asset-1')
    path=tmp_path/'source_asset/video_scene_dict.pkl';path.parent.mkdir()
    path.write_bytes(pickle.dumps(videos,protocol=4))
    return path


def test_loader_binds_scene_tokens_retains_full_block_and_does_not_extrapolate_time(tmp_path):
    write_video(tmp_path)
    frames=[{'token':'t1'},{'token':'after-reconstruction'}]
    value=load_reconstruction_calibration(tmp_path,frames,asset_id='asset')
    assert len(value.poses)==3  # Official reference includes poses with skipped image flags.
    assert value.provenance['exact_scene_token_count']==1
    assert value.provenance['scene_tokens_without_exact_reference']==1
    assert value.nearest_index([50,0])==2
    with pytest.raises(ValueError,match='asset'):
        load_reconstruction_calibration(tmp_path,frames,asset_id='different')
    with pytest.raises(ValueError,match='origins differ'):value.verify_origin([100,200,0])


def test_missing_or_ambiguous_reference_does_not_fall_back_to_utm(tmp_path):
    with pytest.raises(ValueError,match='unavailable'):
        load_reconstruction_calibration(tmp_path,[{'token':'t1'}])
    write_video(tmp_path,duplicate=True)
    with pytest.raises(ValueError,match='Ambiguous'):
        load_reconstruction_calibration(tmp_path,[{'token':'t1'}],asset_id='asset')


def test_native_adapter_does_not_subtract_reconstruction_origin_twice():
    value=reference();translation=value.poses[:,:3,3].copy()
    renderer=SimpleNamespace(recon2global_translation=value.recon_translation,
        engine=SimpleNamespace(current_scene={'metadata':{'native_export':{
            'ego_calibration_frame':'reconstruction','reconstruction_calibration':value.provenance}}}))
    calibration={'ego':{'translation':translation,'rotation':'preserved'}}
    assert shift_ego_calibration(renderer,calibration) is calibration
    np.testing.assert_array_equal(calibration['ego']['translation'],translation)
    renderer.recon2global_translation=np.zeros(3)
    with pytest.raises(ValueError,match='asset origin'):shift_ego_calibration(renderer,calibration)
    renderer.engine.current_scene['metadata']['native_export']['ego_calibration_frame']='global'
    with pytest.raises(ValueError,match='re-export'):shift_ego_calibration(renderer,calibration)


def test_original_scenes_keep_their_official_renderer_reference():
    calibration={'ego':{'translation':np.array([[1.,2.,3.]])}}
    renderer=SimpleNamespace(engine=SimpleNamespace(current_scene={'metadata':{}}))
    assert shift_ego_calibration(renderer,calibration) is calibration


def test_invalid_pose_matrix_and_unknown_globals_are_rejected(tmp_path):
    bad=np.eye(4);bad[0,0]=2
    with pytest.raises(ValueError,match='rigid rotations'):
        ReconstructionCalibration([bad],[0,0,0],{})
    path=tmp_path/'source_asset/video_scene_dict.pkl';path.parent.mkdir()
    path.write_bytes(pickle.dumps({'bad':Path('not scene data')}))
    with pytest.raises(ValueError,match='Unsupported scene data global'):
        load_reconstruction_calibration(tmp_path,[{'token':'t1'}])
