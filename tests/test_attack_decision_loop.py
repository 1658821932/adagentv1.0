"""Offline regressions for useful views, exact time boundaries and cumulative search."""
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'
if str(CORE) not in sys.path: sys.path.insert(0,str(CORE))

from action_validation import prevalidate_action
from attack_objective import validate_objective
from attack_budget import BudgetedArchive, AttackBudgetExhausted, constructor_budget_path
from attack_configuration import same_scene_constraints
from decision_scene_view import actor_view, render_overview


@pytest.mark.parametrize('start,hold',[(0.,11.5),(1.6,9.9)])
def test_native_endpoint_alignment_preserves_a_valid_whole_window_hold(start,hold):
    args=dict(parent_candidate_id='seed_000',max_candidates=1,reason='stop',
        objective=dict(interaction='either',interaction_window_s=[1.5,7.5],keyframes=[
            dict(time_window_s=[start,11.5],hold_duration_s=hold,speed_mps=[0.,.05])]))
    times=np.array([0.,1.5,7.5,11.499938])
    prevalidate_action('construct_attack_candidates',args)
    validate_objective(args,times,1.5,edit_start_s=0.)
    frame=args['objective']['keyframes'][0]
    assert frame['time_window_s'][1]==times[-1]
    assert frame['hold_duration_s']==pytest.approx(times[-1]-start)


def test_real_overlong_hold_is_rejected_and_float_subtraction_noise_is_not():
    def arguments(hold):return dict(parent_candidate_id='seed_000',max_candidates=1,reason='fixture',
        objective=dict(interaction='either',interaction_window_s=[1.5,7.5],keyframes=[
            dict(time_window_s=[1.1,1.3],hold_duration_s=hold,speed_mps=[0.,.1])]))
    prevalidate_action('construct_attack_candidates',arguments(.2))
    with pytest.raises(ValueError,match='hold_duration'):
        prevalidate_action('construct_attack_candidates',arguments(.3))


@pytest.mark.parametrize('scene_end,expected_end', [(7.500465, 7.5), (7.499535, 7.499535)])
def test_attack_constructor_keeps_scoring_window_inside_both_bounds(monkeypatch, scene_end, expected_end):
    import optimized_attack_pool

    captured = []

    def reached_search(editor, arguments):
        captured.append(arguments['objective']['interaction_window_s'])
        raise RuntimeError('reached constructor search')

    monkeypatch.setattr(optimized_attack_pool, 'ControlSearch', reached_search)
    times = np.array([0., 1.5, 2., 2.5, scene_end])
    args = dict(parent_candidate_id='seed_000', max_candidates=2, reason='fixture',
        objective=dict(interaction_window_s=[1.5, 7.5], keyframes=[
            dict(time_window_s=[1.6, 2.4], hold_duration_s=.3, speed_mps=[0., 1.])]))
    editor = SimpleNamespace(constraints={'event_window_s': [1.5, 7.5]})
    validate_objective(args, times, 1.5, edit_start_s=0.)
    with pytest.raises(RuntimeError, match='reached constructor search'):
        optimized_attack_pool.construct_pool(editor, args)
    assert captured == [[1.5, expected_end]]

    args['objective']['interaction_window_s'] = [1.5, 7.501]
    with pytest.raises(ValueError, match='inside configured event window'):
        optimized_attack_pool.construct_pool(editor, args)


def test_freeze_accepts_the_same_rounded_event_boundary_as_geometry():
    from attack_timing import contact_after_observation
    assert contact_after_observation([0.,1.499938,7.5],[1],1.5,time_tolerance_s=.001)
    assert not contact_after_observation([0.,1.498,7.5],[1],1.5,time_tolerance_s=.001)
    assert not contact_after_observation([0.,1.5,7.5],[1],1.5)  # archived strict behavior


def test_scope_budget_survives_new_output_directory_and_reserves_final_review(tmp_path,monkeypatch):
    monkeypatch.setenv('ADVERSARIAL_ATTACK_SESSION',str(tmp_path/'session'))
    observed=[]
    class Archive:
        def complete(self,*args,**kwargs):observed.append(kwargs);return '', 'offline'
    first=BudgetedArchive(Archive(),tmp_path/'first',max_requests=4)
    first.complete('edit_01',[],total_seconds=30,no_output_seconds=60)
    second=BudgetedArchive(Archive(),tmp_path/'second',max_requests=4)
    second.complete('edit_02',[],total_seconds=30)
    assert second.context()['requests']==2
    with pytest.raises(AttackBudgetExhausted): second.complete('fresh_search',[])
    second.closing=True
    second.complete('review',[]);second.complete('submit',[])
    with pytest.raises(AttackBudgetExhausted): second.complete('again',[])
    assert all(row['no_output_seconds']<=row['total_seconds'] for row in observed)
    assert constructor_budget_path(tmp_path/'first')==constructor_budget_path(tmp_path/'second')


def test_api_time_is_shared_and_inflight_interruption_is_charged(tmp_path,monkeypatch):
    monkeypatch.setenv('ADVERSARIAL_ATTACK_SESSION',str(tmp_path))
    (tmp_path/'api-budget.json').write_text(json.dumps(dict(requests=1,api_seconds=4.,
        active_request=dict(timeout=20,started_at=100.,label='interrupted'))))
    import attack_budget
    monkeypatch.setattr(attack_budget.time,'time',lambda:1000.)
    archive=BudgetedArchive(SimpleNamespace(complete=lambda *a,**k:('', 'fixture')),tmp_path,max_api_seconds=120)
    archive.complete('next',[],total_seconds=50)
    assert archive.context()['api_seconds']>=24.
    assert archive.context()['api_seconds']<25.


def test_budget_changes_allow_reuse_but_physics_and_task_changes_do_not():
    prior=dict(attack_stage_timeout_s=420,official_margin_m=.1,task_contract={'objective':'fixed'})
    assert same_scene_constraints(prior,dict(prior,attack_stage_timeout_s=600))
    assert not same_scene_constraints(prior,dict(prior,official_margin_m=0.))
    assert not same_scene_constraints(prior,dict(prior,task_contract={'objective':'changed'}))


def test_actor_view_keeps_future_conflicts_and_leaves_authoritative_data_unchanged():
    times=np.array([0.,1.,2.]);ego=np.zeros((3,2))
    actors=[[dict(track='near',x=3.,y=0.,yaw=0.),dict(track='crossing',x=x,y=0.,yaw=0.)] for x in (100.,40.,.5)]
    before=deepcopy(actors)
    view=actor_view(times,ego,actors,limit=1)
    assert view['actors'][0]['track_id']=='crossing'
    assert view['omitted_actor_count']==1
    assert actors==before


def test_bev_uses_geometry_only_and_writes_a_readable_local_figure(tmp_path):
    from PIL import Image
    times=np.array([0.,1.,2.]);ego=np.array([[0.,0.],[1.,0.],[2.,0.]])
    actors=[[dict(track='target',x=10.,y=0.,yaw=0.,length=4.,width=2.)] for _ in times]
    path=render_overview(times,ego,np.zeros(3),actors,None,tmp_path/'bev.png','target')
    with Image.open(path) as image:assert image.width>image.height>=400


def test_initial_small_gap_is_not_a_collision_and_late_contact_is_rejected():
    import acceptance_policy as policy
    from run_ego_reference import EGO_FRONT
    from vehicle_trajectory import screen_attack
    from ego_road_checks import OfficialDrivableMap
    from shapely.geometry import box
    times=np.arange(0.,8.01,.1);yaw=np.zeros(len(times))
    ego=np.column_stack([np.maximum(times-2,0)*.2,np.zeros(len(times))])
    xy=np.tile([EGO_FRONT+3.,0.],(len(times),1))
    target=dict(track_id='target',size_m=[4.,2.,1.5])
    road=OfficialDrivableMap([box(-100,-20,200,20)],{})
    constraints=dict(acceptance_policy=policy.for_new_run(),attack_prefix_lock_s=0.,event_window_s=[1.5,7.5])
    result=screen_attack(times,ego,yaw,[[] for _ in times],target,xy,yaw,times,xy,yaw,road,constraints)
    assert result['checks']['no_contact_before_explicit_event_start']
    assert result['checks']['target_contact_timing']
    assert result['reference_screen_passed']
    # Same physical approach, but move the requested event deadline earlier.
    late=screen_attack(times,ego,yaw,[[] for _ in times],target,xy,yaw,times,xy,yaw,road,
        dict(constraints,event_window_s=[1.5,5.]))
    assert not late['checks']['target_contact_timing']


@pytest.mark.parametrize('preference_mode,caller_contract,expected',[(True,False,True),(False,False,False),(True,True,False)])
def test_generated_keyframes_are_preferences_but_caller_contract_remains_hard(preference_mode,caller_contract,expected):
    from attack_editor import AttackEditor
    times=np.arange(0.,8.01,.1);ego=np.column_stack([times,np.zeros(len(times))])
    xy=np.tile([12.,0.],(len(times),1));yaw=np.zeros(len(times))
    objective=dict(interaction='contact',interaction_window_s=[1.5,7.5],keyframes=[
        dict(time_window_s=[5.,6.],hold_duration_s=.2,speed_mps=[50.,60.])])
    constraints=dict(generated_objectives_are_preferences=preference_mode,event_window_s=[1.5,7.5])
    if caller_contract:constraints['task_contract']=dict(objective=deepcopy(objective))
    editor=SimpleNamespace(times=times,dense_times=times,ego=ego,ego_yaw=yaw,
        target={'size_m':[4.,2.,1.5]},constraints=constraints)
    args=dict(parent_candidate_id='seed_000',objective=objective,max_candidates=1,reason='fixture')
    result=AttackEditor._goal_result(editor,dict(first_contact_time_s=6.5,native_replay_first_contact_time_s=6.5),xy,yaw,xy,yaw,args)
    assert bool(result['goal_met']) is expected
    if preference_mode and not caller_contract:
        assert result['dense']['preferences_met'] is False
        assert result['dense']['acceptance_basis']=='configured_event_and_declared_interaction'
        from agent_tool_view import compact_action_feedback
        feedback=compact_action_feedback(dict(construction_goal=result))
        assert feedback['optimization_preferences']['met'] is False
        assert feedback['optimization_preferences']['first_missed_interval']['actual_ranges']['speed_mps'][1]<.001


def test_same_runtime_sibling_checkpoint_can_be_reused_without_new_selection(tmp_path,monkeypatch):
    import run_attack_workflow as workflow
    from attack_configuration import normalize_constraints
    runtime=tmp_path/'runtime';previous=runtime/'old/output/attack';output=runtime/'new/output/attack'
    scene=tmp_path/'scene';(scene/'metadata').mkdir(parents=True)
    frames=[dict(timestamp=1000000+i*500000,ego2global=[[1,0,0,i],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) for i in range(24)]
    (scene/'metadata/frames.json').write_text(json.dumps(frames))
    (previous/'object_selection').mkdir(parents=True);output.parent.mkdir(parents=True)
    seed=previous/'object_selection/seed.csv';seed.write_text('fixture')
    config=normalize_constraints(dict(event_window_s=[1.5,7.5]))
    (previous/'input.json').write_text(json.dumps(dict(scene=str(scene),constraints=config,instruction='fixture',assets=[],required_operation=None,required_track=None)))
    selection=dict(status='completed',seed_trajectory_path=str(seed),target={'track_id':'target'},instruction='fixture')
    (previous/'object_selection/output.json').write_text(json.dumps(selection))
    monkeypatch.setenv('ADVERSARIAL_EXPERIMENTS_ROOT',str(output.parent))
    monkeypatch.setenv('ADVERSARIAL_CONTINUATION_ROOT',str(runtime))
    monkeypatch.delenv('ADVERSARIAL_ATTACK_SESSION',raising=False)
    monkeypatch.setattr(workflow,'load_llm_profile',lambda:dict(api_key='offline',base_url='https://example.invalid'))
    monkeypatch.setattr(workflow,'OpenAI',lambda **kwargs:SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(workflow,'StreamingRequestArchive',lambda *args:SimpleNamespace())
    monkeypatch.setattr(workflow,'initial_state_precheck',lambda editor:None)
    class Editor:
        def __init__(self,scene,selection,folder,config):
            self.run_dir=folder;folder.mkdir();self.pdm_probe=None;self.policy_probe=None
            self.target=selection['target'];self.nodes={};self.history=[]
    monkeypatch.setattr(workflow,'AttackEditor',Editor)
    def edit(editor,*args,**kwargs):
        (editor.run_dir/'output.json').write_text(json.dumps(dict(stop_reason='action_budget_exhausted')))
        return None
    monkeypatch.setattr(workflow,'edit_object',edit)
    result=workflow.run(scene,output,'fixture',constraints=config,continue_from=previous)
    assert result['status']=='not_submitted'
    assert (output/'object_selection/seed_trajectory.csv').read_text()=='fixture'


@pytest.mark.parametrize('distance',[1.,5.,30.])
def test_new_attack_mode_measures_proximity_without_an_interaction_class(distance):
    from attack_constructor import acceptance_goal_evidence
    times=np.arange(0.,8.01,.1);ego=np.column_stack([times,np.zeros(len(times))])
    xy=ego+np.array([distance,0.]);yaw=np.zeros(len(times))
    objective=dict(keyframes=[dict(time_window_s=[5.,6.],hold_duration_s=.2,speed_mps=[50.,60.])])
    result=acceptance_goal_evidence(times,xy,ego,yaw,{'objective':objective},None,
        target_yaw=yaw,target_size=[4.,2.,1.5],
        constraints=dict(agent_decides_interaction=True,event_window_s=[1.5,7.5]))
    assert result['goal_met']
    assert result['interaction_assessment']=='measurements_only_no_contact_or_clearance_class_gate'
    assert not result['preferences_met']  # unachieved Agent preference is still reported


def test_new_attack_constructor_accepts_keyframes_without_interaction_field():
    args=dict(parent_candidate_id='seed_000',objective={'keyframes':[]},max_candidates=1,reason='review')
    validate_objective(args,np.arange(0.,8.01,.5),1.5,edit_start_s=0.)
    assert 'interaction' not in args['objective']
    prevalidate_action('edit_route',{'parent_candidate_id':'seed_000'},
        SimpleNamespace(constraints={'agent_decides_interaction':True}))


def test_constructor_checks_timed_absolute_waypoints_even_with_advisory_keyframes():
    from attack_constructor import acceptance_goal_evidence

    times = np.arange(0., 4.01, .25)
    xy = np.column_stack((np.zeros(len(times)), times * 2.))
    yaw = np.full(len(times), np.pi / 2)
    ego = np.column_stack((np.full(len(times), 30.), times))
    args = dict(parent_candidate_id='seed_000', max_candidates=1, reason='crossing fixture',
        objective=dict(waypoints=[
            dict(position_m=[0., 0.], time_s=0., speed_mps=2., yaw_rad=np.pi/2),
            dict(position_m=[0., 3.], time_window_s=[1.4, 1.6], speed_mps=2.),
            dict(position_m=[0., 7.24], time_s=3.62, speed_mps=2., position_tolerance_m=.01),
        ]))
    validate_objective(args, times, 1.5, edit_start_s=0.)

    def check():
        return acceptance_goal_evidence(times, xy, ego, yaw, args, None,
            target_yaw=yaw, target_size=[4., 2., 1.5],
            constraints=dict(agent_decides_interaction=True, event_window_s=[1.5, 4.]))

    evidence = check()
    assert evidence['goal_met']
    reports = evidence['requested_waypoints']
    assert reports[2]['evaluated_time_s'] == 3.62
    assert reports[2]['actual']['position_m'] == pytest.approx([0., 7.24])
    args['objective']['waypoints'][1]['speed_mps'] = 4.
    rejected = check()
    assert not rejected['goal_met'] and rejected['failed_checks'] == ['requested_waypoint_2']
    assert rejected['requested_waypoints'][1]['errors']['speed_error_mps'] == pytest.approx(2.)


@pytest.mark.parametrize('bad_point', [
    dict(position_m=[2., 3.], time_s=1., time_window_s=[1., 2.]),
    dict(position_m=[2., 3.], time_window_s=[0., 2.]),
    dict(position_m=[2., 3.], time_s=2., speed_mps=-1.),
    dict(position_m=[2., 3.], time_s=2., speed_tolerance_mps=.1),
])
def test_absolute_waypoint_input_rejects_conflicting_fields(bad_point):
    args = dict(parent_candidate_id='seed_000', max_candidates=1, reason='fixture',
        objective=dict(waypoints=[dict(position_m=[0., 0.], time_s=0.), bad_point]))
    with pytest.raises(ValueError, match='waypoint 2'):
        validate_objective(args, np.arange(0., 4.01, .25), 1.5, edit_start_s=0.)
