"""Collision evidence must describe actual target contact, not PDM failure."""
import math
from types import SimpleNamespace
import sys
import json
from unittest.mock import Mock
import pytest

from mcp_servers.worldengine.adapter.collision_report import body_contact,summarize_frames,install_recorder


def actor(x,y=0,yaw=0,velocity=None):
    return dict(box=[x,y,4.,2.,yaw],velocity=velocity)


def frame(step,target_x=10,other=False):
    actors={'ego':actor(0,velocity=[2,0])}
    if target_x is not None:actors['target']=actor(target_x,velocity=[1,0])
    if other:actors['other']=actor(0)
    return dict(scene_id='scene',step=step,time_s=step*.5,actors=actors)


@pytest.mark.parametrize('x,contact,gap',[(5,False,1),(4.001,False,.001),(4,True,0),(3,True,0),(-3,True,0)])
def test_exact_body_geometry_has_no_collision_distance_buffer(x,contact,gap):
    actual,distance=body_contact(actor(0)['box'],actor(x)['box'])
    assert actual is contact
    assert distance==pytest.approx(gap)


def test_rotated_body_and_diagonal_separation():
    assert body_contact(actor(0)['box'],actor(0,2.5,math.pi/2)['box'])[0]
    contact,distance=body_contact(actor(0)['box'],actor(5,3)['box'])
    assert contact is False and distance==pytest.approx(math.sqrt(2))


def test_target_contact_after_takeover_and_closing_speed():
    data=[frame(i,3 if i==4 else 10) for i in range(6)]
    report=summarize_frames(data,'target',3)
    assert report['target_collision_detected'] is True
    assert report['first_target_collision_time_s']==2.0
    assert report['closest_approach']['closing_speed_mps']==1
    assert report['policy_takeover_time_s']==1.5


def test_pre_takeover_contact_is_not_policy_failure():
    report=summarize_frames([frame(i,3 if i==3 else 10) for i in range(6)],'target',3)
    assert report['initial_history_target_contact'] is True
    assert report['target_collision_detected'] is False


def test_other_actor_contact_does_not_satisfy_target_goal():
    report=summarize_frames([frame(i,10,other=i==4) for i in range(6)],'target',3)
    assert report['any_collision_detected'] is True
    assert report['target_collision_detected'] is False
    assert report['first_collision']['track_id']=='other'


def test_missing_target_is_unknown_not_avoidance():
    report=summarize_frames([frame(i,None if i==4 else 10) for i in range(6)],'target',3)
    assert report['status']=='incomplete_target_coverage'
    assert report['target_collision_detected'] is None


def test_interpolated_contact_cannot_be_promoted_to_observed_contact():
    data=[frame(i,-10 if i<5 else 10) for i in range(6)]
    report=summarize_frames(data,'target',3)
    assert report['target_collision_detected'] is False
    assert report['possible_interpolated_target_contact'] is True


def test_capture_uses_actual_body_centres_before_renderer_mutation(tmp_path,monkeypatch):
    class Manager:
        def render(self):
            agent.current_position[0]=999
            return 'image'
    agent=SimpleNamespace(current_position=[12,3],length=4,width=2,current_heading=.2,
        policy=SimpleNamespace(is_current_step_valid=True),current_velocity=[2,0])
    monkeypatch.setitem(sys.modules,'worldengine',SimpleNamespace())
    monkeypatch.setitem(sys.modules,'worldengine.manager',SimpleNamespace())
    monkeypatch.setitem(sys.modules,'worldengine.manager.render_manager',SimpleNamespace(RenderManager=Manager))
    path=tmp_path/'states.jsonl'
    install_recorder(path)
    manager=Manager()
    manager.engine=SimpleNamespace(agent_manager=SimpleNamespace(all_agents={'ego':agent}),
        current_scene={'id':'scene'},episode_step=4,sim_time_interval=.5)
    assert manager.render()=='image'
    recorded=json.loads(path.read_text())
    assert recorded['actors']['ego']['box']==[12,3,4,2,.2]
    assert recorded['time_s']==2


def test_client_requires_and_preserves_actual_collision_report(tmp_path,monkeypatch):
    from mcp_servers.worldengine.adapter import client
    from subprocess import CompletedProcess
    result_dir=tmp_path/'result'
    report_dir=result_dir/'WE_output';report_dir.mkdir(parents=True)
    report=summarize_frames([frame(i,3 if i==4 else 10) for i in range(6)],'target',3)
    (report_dir/'native_collision_report.json').write_text(json.dumps(report))
    payload=dict(status='success',result=dict(result_dir=str(result_dir),reports=[dict(succeeded=True)]))
    monkeypatch.setenv('ADAGENT_RUNTIME_ROOT',str(tmp_path/'runtime'))
    monkeypatch.setattr(client,'AGENT_PYTHON',__import__('pathlib').Path(sys.executable))
    monkeypatch.setattr(client,'run_logged',Mock(return_value=CompletedProcess([],0,json.dumps(payload),'')))
    cfg=tmp_path/'cfg.py';cfg.touch()
    weights=tmp_path/'weights.pth';weights.touch()
    args=dict(config_path=str(cfg),checkpoint_path=str(weights),model_name='policy',data_type='scene',
        max_frame_per_scene=6,collision_target_track='target')
    assert client.run_policy(args)['collision_report']['target_collision_detected'] is True
    args['collision_target_track']='other'
    with pytest.raises(RuntimeError,match='different attack target'):client.run_policy(args)


def test_main_context_keeps_compact_measured_collision_feedback():
    from agent.context import summarize
    report=summarize_frames([frame(i) for i in range(6)],'target',3)
    summary=summarize('policy_runner',dict(result_dir='/result',collision_report=report))
    assert summary['collision_report']['target_collision_detected'] is False
    assert summary['collision_report']['closest_approach']['closing_speed_mps']==1
