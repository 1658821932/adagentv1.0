"""Shared history generation, actor coordinates and causal VAD takeover."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'
sys.path.insert(0,str(CORE))
from ego_history import generate,reproject_frames
from run_ego_reference import ego_path_from_frames,observed_objects_from_frames
import acceptance_policy as policy


def frames():
    result=[]
    for i in range(24):
        pose=np.eye(4);pose[:2,3]=[100.+2.5*i,200.]
        result.append(dict(timestamp=1000000+500000*i,token=str(i),map_location='map',
            ego2global=pose.tolist(),ego2global_rotation=[1.,0.,0.,0.],lidar2ego=np.eye(4).tolist(),
            anns=dict(track_tokens=['parked'],gt_names=['vehicle'],
                gt_boxes=[[50-2.5*i,8.,0.,4.,2.,1.5,0.]],gt_velocity_3d=[[0.,0.,0.]])))
    return result


def test_speed_history_accelerates_without_moving_any_other_actor():
    original=frames();source=json.dumps(original)
    xy,yaw,evidence=generate(original,[{'time_s':1.5,'speed_mps':8.}],policy.for_new_run())
    assert xy[3,0]==pytest.approx(9.75,abs=1e-5)
    assert evidence['continuous_reference_velocity_xy_mps'][3][0]==pytest.approx(8.,abs=1e-5)
    assert 7.<evidence['delivered_takeover_speed_mps']<=8.
    assert np.array_equal(xy[0],[0.,0.])
    edited=reproject_frames(original,xy,yaw)
    before=observed_objects_from_frames(original);after=observed_objects_from_frames(edited)
    for a,b in zip(before,after):
        assert a[0]['x']==pytest.approx(b[0]['x'],abs=1e-8)
        assert a[0]['y']==pytest.approx(b[0]['y'],abs=1e-8)
        assert a[0]['yaw']==pytest.approx(b[0]['yaw'],abs=1e-8)
    assert json.dumps(original)==source


@pytest.mark.parametrize('targets',[
    [{'time_s':1.,'speed_mps':6.}],
    [{'time_s':1.5,'speed_mps':-1.}],
    [{'time_s':.2,'speed_mps':14.},{'time_s':1.5,'speed_mps':14.}],
    [{'time_s':2.,'speed_mps':6.}],
])
def test_invalid_or_unreachable_history_is_rejected(targets):
    with pytest.raises(ValueError):generate(frames(),targets,policy.for_new_run())


def test_multiple_speed_targets_keep_initial_pose_and_continuity():
    xy,yaw,evidence=generate(frames(),[{'time_s':.8,'speed_mps':6.},
        {'time_s':1.5,'speed_mps':8.}],policy.for_new_run())
    assert evidence['requested_takeover_speed_mps']==8.
    assert np.isfinite(xy).all() and np.isfinite(yaw).all()
    assert np.max(np.linalg.norm(np.diff(xy,axis=0),axis=1))<4.1


def test_endpoint_route_freezes_fixed_start_and_reproduces_saved_controls(tmp_path):
    from ego_history import prepare, verify

    source = tmp_path/'source'
    metadata = source/'metadata'
    metadata.mkdir(parents=True)
    original = frames()
    (metadata/'frames.json').write_text(json.dumps(original))
    road = {'features': {'lane': {'polygon': [[-10, -20], [100, -20], [100, 20], [-10, 20]]}}}
    (metadata/'map_drivable_official.json').write_text(json.dumps(road))
    route = dict(end_xy_m=[7.65, -.28], end_yaw_rad=-.08, end_speed_mps=5.3,
                 via_xy_m=[3.8, -.07], via_time_s=.75)
    prepared = prepare(source, tmp_path/'prepared', None, policy.for_new_run(), route=route)
    assert verify(prepared)
    manifest = json.loads((prepared/'ego-history.json').read_text())
    assert manifest['mode'] == 'endpoint_route'
    assert manifest['takeover_xy_m'] == pytest.approx(route['end_xy_m'], abs=.12)
    assert manifest['takeover_yaw_rad'] == pytest.approx(route['end_yaw_rad'], abs=.08)
    assert manifest['continuous_takeover_speed_mps'] == pytest.approx(route['end_speed_mps'], abs=.25)
    edited = json.loads((prepared/'metadata/frames.json').read_text())
    assert edited[0]['ego2global'] == original[0]['ego2global']
    times, _, _ = ego_path_from_frames(original)
    assert [row['ego2global'] for row, t in zip(edited, times) if t >= manifest['reference_rejoin_s']] == [
        row['ego2global'] for row, t in zip(original, times) if t >= manifest['reference_rejoin_s']]


def test_endpoint_route_declines_a_far_endpoint(tmp_path):
    from ego_history_route import generate as generate_route

    with pytest.raises(ValueError, match='reachable in 1.5s'):
        generate_route(tmp_path, frames(), {'end_xy_m': [40., 0.]}, policy.for_new_run())


def test_initial_heading_alignment_does_not_create_first_step_yaw_spike(monkeypatch):
    from ego_speed_schedule import Route
    original=Route.drive
    def offset_drive(self,query,law):
        positions,velocities,angles=original(self,query,law)
        return positions,velocities,np.asarray(angles)-.044
    monkeypatch.setattr(Route,'drive',offset_drive)
    _,yaw,_=generate(frames(),[{'time_s':1.5,'speed_mps':8.}],policy.for_new_run())
    assert yaw[0]==pytest.approx(0.,abs=1e-12)
    assert np.max(np.abs(np.diff(yaw[:4])))/.5<.1


def test_prepared_history_survives_real_attack_freeze_and_keeps_source_immutable(tmp_path):
    from ego_history import prepare,attach,source_scene,verify,clone
    from attack_editor import write_trajectory
    from freeze_attack_scene import freeze
    original=tmp_path/'source';(original/'metadata').mkdir(parents=True)
    source=frames();raw=json.dumps(source)
    (original/'metadata/frames.json').write_text(raw)
    road={'features':{'lane':{'polygon':[[-10,-20],[100,-20],[100,20],[-10,20]]}}}
    (original/'metadata/map_drivable_official.json').write_text(json.dumps(road))
    prepared=prepare(original,tmp_path/'prepared',[{'time_s':1.5,'speed_mps':8.}],policy.for_new_run())
    assert source_scene(prepared)==original and verify(prepared)
    t,_,_=ego_path_from_frames(source)
    attack=tmp_path/'attack';candidate=attack/'candidates/candidate_001';candidate.mkdir(parents=True)
    write_trajectory(candidate/'trajectory.csv',t,np.tile([50.,8.],(len(t),1)),np.zeros(len(t)),[4.,2.,1.5])
    frozen=tmp_path/'frozen'
    freeze(prepared,attack,'candidate_001','parked',frozen)
    attach(prepared,frozen)
    assert (frozen/'ego-history.json').is_file()
    actual=json.loads((frozen/'metadata/frames.json').read_text())
    expected=json.loads((prepared/'metadata/frames.json').read_text())
    assert [f['ego2global'] for f in actual]==[f['ego2global'] for f in expected]
    assert (original/'metadata/frames.json').read_text()==raw
    assert prepare(prepared,tmp_path/'keep',None,policy.for_new_run())==prepared
    assert prepare(prepared,tmp_path/'reset',[],policy.for_new_run())==original
    imported=clone(prepared,tmp_path/'resumed/history')
    assert verify(imported)
    assert json.loads((imported/'ego-history.json').read_text())['prepared_scene']==str(imported)


def test_expert_future_cannot_change_shared_history_or_takeover_velocity():
    from native_scene_export import build
    from reconstruction_calibration import ReconstructionCalibration
    source=frames()
    for f in source:
        f.update(scene_token='s',lidar2ego_rotation=[1.,0.,0.,0.],lidar2ego_translation=[0.,0.,0.],cams={})
    t,xy,yaw=ego_path_from_frames(source)
    poses=np.array([f['ego2global'] for f in source]);poses[:,:2,3]-=[100.,200.]
    calibration=ReconstructionCalibration(poses,[100.,200.,0.],{'source':'offline'})
    attack,_=build(source,{},'s','asset',reconstruction=calibration)
    changed=xy.copy();changed[t>1.5,0]-=.2
    expert,_=build(source,{},'s','asset',expert=(t,changed,yaw),reconstruction=calibration)
    for key in ('position','velocity','heading'):
        assert np.array_equal(attack['object_track']['ego']['state'][key][:4],expert['object_track']['ego']['state'][key][:4])
    assert attack['metadata']['native_export']['navigation_reference_body_xy_m']==expert['metadata']['native_export']['navigation_reference_body_xy_m']
    changed[2,0]+=.1
    with pytest.raises(ValueError,match='shared 0–1.5s history'):
        build(source,{},'s','asset',expert=(t,changed,yaw),reconstruction=calibration)


def test_native_navigation_uses_frozen_task_only_for_ego():
    import ast
    from collections import deque
    from types import SimpleNamespace
    path=CORE.parents[2]/'worldengine/core/projects/SimEngine/worldengine/components/agents/navigation/trajectory_navigation.py'
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef))
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    fixed=[[1.,0.],[12.,0.],[40.,0.]]
    old=np.array([[0.,0.],[2.,0.]])
    env=dict(np=np,deque=deque,parse_full_trajectory=lambda track:(old.copy(),np.ones(2)),
        CenterLane=lambda positions,width:positions,
        get_engine=lambda:SimpleNamespace(current_scene={'metadata':{'native_export':{'navigation_reference_body_xy_m':fixed}}}))
    exec(compile(ast.Module(body=[init],type_ignores=[]),str(path),'exec'),env)
    for name in ('ego','other'):
        navigation=SimpleNamespace(ROUTE_WIDTH=2.,get_navigation_info_dim=lambda:2)
        env['__init__'](navigation,SimpleNamespace(name=name,object_track={}))
        assert np.array_equal(navigation.full_traj,np.asarray(fixed) if name=='ego' else old)
