import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'
if str(CORE) not in sys.path:sys.path.insert(0,str(CORE))

import optimized_attack_pool as pool_module
from optimized_attack_pool import finalist_order
from attack_editor import AttackEditor, write_trajectory
from initial_pose_edit import adjusted_arrays,adjusted_seed,search_specs
from vehicle_control_search import event_window_guidance,motion_guidance,release_starts,integrate_controls,ControlSearch,refinement_groups


def test_event_window_guidance_rewards_contact_only_inside_window():
    times=np.linspace(0.,10.,101)
    gap=np.full(len(times),5.)
    gap[50]=-.1
    residuals,metrics=event_window_guidance(times,gap,[1.5,7.5])
    assert residuals==[0.,0.,0.]
    assert metrics['window_reference_contact'] is True
    assert metrics['first_reference_contact_time_s']==5.


def test_event_window_guidance_penalizes_distant_and_late_contact():
    times=np.linspace(0.,10.,101)
    gap=np.full(len(times),5.)
    far,metrics=event_window_guidance(times,gap,[1.5,7.5])
    assert far==[12.5,0.,0.] and metrics['window_reference_contact'] is False
    gap[85]=-.1
    late,metrics=event_window_guidance(times,gap,[1.5,7.5])
    assert late==[12.5,4.,0.]
    assert metrics['late_contact_seconds']==1.


def test_contact_required_guidance_uses_stronger_collision_cost():
    times=np.linspace(0.,10.,101);gap=np.full(len(times),5.)
    residuals,metrics=event_window_guidance(times,gap,[1.5,7.5],required=True)
    assert residuals==[50.,0.,.08]
    assert metrics['window_penetration_m']==0.
    gap[60]=-.1
    residuals,metrics=event_window_guidance(times,gap,[1.5,7.5],required=True)
    assert residuals==[0.,0.,0.] and metrics['window_penetration_m']==.1


def test_reference_contact_is_mandatory_when_enabled():
    editor=NS(constraints=dict(require_reference_contact=True,event_window_s=[1.5,7.5],acceptance_policy=None))
    result=dict(checks=dict(official_road=True),reference_screen_passed=True,failed_checks=[],
        first_contact_time_s=None,native_replay_first_contact_time_s=None,min_ego_clearance_m=2.)
    failed=AttackEditor._apply_contact_requirement(editor,result)
    assert failed['reference_screen_passed'] is False
    assert failed['checks']['reference_contact'] is False and failed['failed_checks']==['reference_contact']
    passing=dict(result,first_contact_time_s=5.9,native_replay_first_contact_time_s=6.,checks=dict(official_road=True),failed_checks=[])
    passed=AttackEditor._apply_contact_requirement(editor,passing)
    assert passed['reference_screen_passed'] is True and passed['checks']['reference_contact'] is True


def test_motion_guidance_has_fixed_residual_dimension():
    native=np.linspace(0.,2.,5)
    dense=np.linspace(0.,2.,41)
    xy=np.column_stack([dense,np.zeros(len(dense))])
    editor=NS(times=native,seed_xy=np.column_stack([native,np.zeros(len(native))]),seed_yaw=np.zeros(len(native)),
        edit_start_s=0.,constraints={},target={'size_m':[4.5,2.,1.6]})
    residuals,metrics=motion_guidance(editor,dense,xy,np.zeros(len(dense)))
    assert len(residuals)==10
    assert metrics['motion_guidance_violation_count']==0


def record(identifier,cost,**metrics):
    return cost,np.asarray([cost]),dict(attempt_id=identifier,guidance_cost=cost,search_metrics=metrics)


def test_finalists_keep_window_contact_and_motion_diverse_records():
    initial=record('attempt_001',100.)
    cheap=record('attempt_002',1.,window_reference_contact=False,window_min_signed_gap_m=8.,
        late_contact_seconds=0.,motion_guidance_violation_count=4)
    contact=record('attempt_003',20.,window_reference_contact=True,window_min_signed_gap_m=-.1,
        late_contact_seconds=0.,motion_guidance_violation_count=1)
    ordered=finalist_order([cheap,initial,contact])
    assert [item[2]['attempt_id'] for item in ordered]==['attempt_001','attempt_003','attempt_002']


def test_finalists_retain_safe_contact_even_when_preference_cost_is_higher():
    initial=record('attempt_001',100.)
    unsafe=record('attempt_002',1.,window_reference_contact=True,other_overlap_count=8)
    safe=record('attempt_003',80.,window_reference_contact=True,geometry_motion_guidance_feasible=True)
    ordered=finalist_order([unsafe,safe,initial])
    assert [item[2]['attempt_id'] for item in ordered][:2]==['attempt_001','attempt_003']


def test_release_controls_preserve_approach_and_reaccelerate_continuously():
    native=np.arange(0.,10.01,.5)
    dense=np.arange(0.,10.01,.1)
    knots=np.arange(0.,10.01,.5)
    n=len(knots)-1
    values=np.r_[np.full(n,-3.),np.zeros(n)]
    guesses=list(release_starts(knots,np.arange(n),values,5.,8.))
    assert guesses and np.array_equal(values[:n],np.full(n,-3.))
    seed=np.column_stack([native*8.,np.zeros(len(native))])
    parent=dict(times_s=knots.tolist(),acceleration_mps2=[-3.]*len(knots),steering_rad=[0.]*len(knots))
    before,_=integrate_controls(native,seed,np.zeros(len(native)),dense,[4.5,2.,1.6],parent,0.)
    for guess in guesses:
        assert np.array_equal(guess[:6],values[:6])
        assert np.array_equal(guess[n:n+6],values[n:n+6])
        controls=dict(times_s=knots.tolist(),acceleration_mps2=[*guess[:n],guess[n-1]],
            steering_rad=[*(guess[n:]/10),guess[-1]/10])
        actual,state=integrate_controls(native,seed,np.zeros(len(native)),dense,[4.5,2.,1.6],controls,0.)
        assert np.allclose(actual[2][dense<=3.],before[2][dense<=3.])
        assert state[-1,0]>0. and np.isfinite(actual[2]).all()
        assert np.max(np.linalg.norm(np.diff(actual[2],axis=0),axis=1))<1.51


def test_route_seed_retimes_turn_by_progress_without_changing_acceleration():
    times=np.arange(0.,10.01,.5)
    x=8.*times
    xy=np.column_stack([x,.00005*x**3])
    yaw=np.arctan(.00015*x**2)
    search=ControlSearch.__new__(ControlSearch)
    search.editor=NS(times=times,seed_xy=xy,seed_yaw=yaw,constraints={},target={'size_m':[4.5,2.,1.6]})
    search.active_seed_xy=xy;search.active_seed_yaw=yaw;search.boundary=0.
    search.control_times=times;search.editable=np.arange(len(times)-1);search.n=len(search.editable)
    search.base=dict(times_s=times.tolist(),acceleration_mps2=[0.]*len(times),steering_rad=[0.]*len(times))
    search.lower=np.r_[np.full(search.n,-4.5),np.full(search.n,-5.5)]
    search.upper=np.r_[np.full(search.n,2.5),np.full(search.n,5.5)]
    cruise=np.zeros(2*search.n)
    brake=cruise.copy();brake[:search.n]=-3.
    fast=search.route_start(cruise)
    slow=search.route_start(brake)
    assert np.array_equal(slow[:search.n],brake[:search.n])
    # Same clock time but different distance into a tightening curve.
    assert slow[-1]<fast[-1] and slow[-1]>0.


def test_segment_refinement_preserves_command_bounds():
    n=16
    values=np.r_[[-4.5]*8+[2.5]*8,np.linspace(-1.,1.,n)]
    lower=np.r_[np.full(n,-4.5),np.full(n,-5.5)]
    upper=np.r_[np.full(n,2.5),np.full(n,5.5)]
    groups,bounds=refinement_groups(values,lower,upper,n)
    assert len(groups)<=8 and len(groups)>0
    for column in (0,1):
        actual=values.copy()
        for group,offset in zip(groups,bounds[:,column]):actual[group]+=offset
        assert np.all(actual>=lower) and np.all(actual<=upper)


def test_pose_search_reserves_budget_for_real_numerical_improvement():
    search=ControlSearch.__new__(ControlSearch)
    search.pose_specs=list(range(8));search.editor=NS(constraints={});search.arguments={}
    search.n=4;search.initial=np.full(8,.5)
    search.lower=np.full(8,-2.);search.upper=np.full(8,2.)
    search.activate=lambda spec:setattr(search,'active_pose',spec)
    search.starts=lambda:iter([search.initial.copy()])
    search.controls=lambda values:values.copy()
    search.simulate=lambda values:(values,None)
    search.last_metrics={}
    search.residual=lambda values,state:values
    evaluated=[];reserved=[]
    def evaluate(controls,arrays,state,cost):
        evaluated.append(cost)
        return dict(attempt_id=f'attempt_{len(evaluated):03d}')
    result=search.optimize(evaluate,60,reserve=lambda values:reserved.append(values.copy()))
    assert len(evaluated)==len(reserved)<=60
    assert len(result)>8 and min(evaluated)<1e-6
    assert {r[2]['initial_pose_adjustment'] for r in result}==set(range(8))


def pose_editor():
    seed=np.column_stack([np.arange(5,dtype=float),np.zeros(5)])
    return NS(seed_xy=seed,seed_yaw=np.zeros(5),ego=seed.copy(),ego_yaw=np.full(5,.3),
        times=np.arange(5,dtype=float)*.5,constraints={})


def test_initial_pose_translation_and_yaw_are_replayable():
    editor=pose_editor()
    xy,yaw,record=adjusted_seed(editor,{},dict(position_offset_m=dict(forward_m=-5.,left_m=1.),
        yaw_mode='offset',yaw_offset_rad=.2))
    assert np.allclose(xy[0],[-5.,1.]) and np.allclose(yaw,.2)
    direction=xy[1]-xy[0]
    assert np.isclose(np.arctan2(direction[1],direction[0]),.2)
    assert record['adjusted_pose']['x_m']==-5. and record['yaw_mode']=='offset'
    assert np.isclose(record['translation_m'],np.sqrt(26.))


def test_route_aligned_yaw_uses_new_position_and_cumulative_bounds_apply():
    editor=pose_editor()
    xy,yaw,record=adjusted_seed(editor,{},dict(position_offset_m=dict(forward_m=-2.,left_m=0.),
        yaw_mode='route_aligned'))
    assert np.allclose(yaw,.3) and record['yaw_mode']=='route_aligned'
    parent=dict(initial_seed_xy=xy,initial_seed_yaw=yaw,initial_pose=record)
    import pytest
    with pytest.raises(ValueError,match='Cumulative initial pose yaw change'):
        adjusted_seed(editor,parent,dict(position_offset_m=dict(forward_m=0.,left_m=0.),
            yaw_mode='offset',yaw_offset_rad=1.4))


def test_constructor_pose_search_is_bounded_and_optional():
    args=dict(initial_pose_search=dict(forward_offsets_m=[0.,-10.],left_offsets_m=[0.,1.],
        yaw_mode='preserve'))
    specs=search_specs(args)
    assert len(specs)==4 and specs[0]['position_offset_m']['forward_m']==0.
    assert search_specs({})==[None]
    import pytest
    with pytest.raises(ValueError,match='at most 6'):
        search_specs(dict(initial_pose_search=dict(forward_offsets_m=[1,2,3,4],left_offsets_m=[1,2])))
    with pytest.raises(ValueError,match='cannot preserve'):
        search_specs(dict(initial_pose_search=dict(),preserve_outside_window=True))


def test_stationary_added_actor_pose_adjustment_is_rigid():
    editor=pose_editor();editor.constraints={'stationary_target':True}
    parent=dict(initial_seed_xy=editor.seed_xy,initial_seed_yaw=editor.seed_yaw,initial_pose=None,
        xy=editor.seed_xy.copy(),yaw=editor.seed_yaw.copy(),dense_xy=editor.seed_xy.copy(),dense_yaw=editor.seed_yaw.copy())
    spec=dict(position_offset_m=dict(forward_m=-2.,left_m=1.),yaw_mode='offset',yaw_offset_rad=.25)
    xy,yaw,dense_xy,dense_yaw,initial_seed=adjusted_arrays(editor,parent,spec)
    direction=xy[1]-xy[0]
    assert np.isclose(np.arctan2(direction[1],direction[0]),.25)
    assert np.allclose(xy,dense_xy) and np.allclose(yaw,dense_yaw)
    assert initial_seed[2]['yaw_mode']=='offset'


def make_trajectory_editor(root, stationary=False):
    scene = root / 'scene'
    metadata = scene / 'metadata'
    metadata.mkdir(parents=True)
    times = np.arange(0., 8.01, .5)
    frames = [dict(timestamp=int(1_000_000 + t * 1_000_000),
                   ego2global=np.eye(4).tolist(), lidar2ego=np.eye(4).tolist(),
                   anns=dict(track_tokens=[], gt_names=[], gt_boxes=[])) for t in times]
    (metadata / 'frames.json').write_text(json.dumps(frames))
    road = dict(features=dict(lane=dict(polygon=[[-10, -20], [100, -20], [100, 20], [-10, 20]])))
    (metadata / 'map_drivable_official.json').write_text(json.dumps(road))
    seed = root / 'seed.csv'
    xy = np.column_stack([15. + times * (0. if stationary else 2.), np.zeros(len(times))])
    size = [4., 2., 1.5]
    write_trajectory(seed, times, xy, np.zeros(len(times)), size)
    selection = dict(target=dict(operation='edit_existing', track_id='vehicle', size_m=size),
                     seed_trajectory_path=str(seed))
    return AttackEditor(scene, selection, root / 'editor',
                        dict(attack_prefix_lock_s=0., event_window_s=[1.5, 7.5],
                             stationary_target=stationary, vehicle_motion_model='bicycle.v1'))


@pytest.mark.parametrize('stationary', [True, False])
def test_initial_pose_tool_creates_and_archives_the_candidate(tmp_path, stationary):
    editor = make_trajectory_editor(tmp_path, stationary=stationary)
    args = dict(parent_candidate_id='seed_000', position_offset_m=dict(forward_m=0., left_m=1.),
                yaw_mode='route_aligned', reason='Move the test trajectory left')
    result = editor.adjust_initial_pose(args)
    node = editor.nodes[result['candidate_id']]
    np.testing.assert_allclose(node['xy'][0], [15., 1.], atol=1e-8)
    np.testing.assert_allclose(node['initial_seed_xy'][0], node['xy'][0], atol=1e-8)
    assert node['initial_pose']['yaw_mode'] == 'route_aligned'
    path = editor.run_dir / 'candidates' / result['candidate_id'] / 'operation.json'
    assert json.loads(path.read_text())['operation'] == dict(tool='adjust_initial_pose', input=args)


def test_control_point_construction_uses_original_seed_without_parent(tmp_path):
    editor = make_trajectory_editor(tmp_path)
    points = [dict(time_s=0., x_m=15., y_m=0.), dict(time_s=8., x_m=31., y_m=0.)]
    result = editor.construct(dict(points=points, end_velocity_mps=[2., 0.], reason='Follow the seed'))
    node = editor.nodes[result['candidate_id']]
    assert node['parent_id'] is None
    np.testing.assert_allclose(node['initial_seed_xy'], editor.seed_xy)
    assert np.isfinite(node['result']['reference_tracking_error_m'])


def test_constructor_keeps_valid_pose_branches_when_others_fail_precheck(tmp_path, monkeypatch):
    import attack_editor
    import initial_pose_edit
    from agent_tool_view import compact_action_feedback

    editor = make_trajectory_editor(tmp_path)
    editor.seed_yaw[1] = .1
    editor.constraints['agent_decides_interaction'] = True
    zero_speed = dict(position_offset_m=dict(forward_m=0., left_m=0.), initial_speed_mps=0.)
    out_of_bounds = dict(position_offset_m=dict(forward_m=30., left_m=1.))
    eligible = dict(position_offset_m=dict(forward_m=1., left_m=0.))
    monkeypatch.setattr(initial_pose_edit, 'search_specs',
        lambda *args: [None, zero_speed, out_of_bounds, eligible])
    monkeypatch.setattr(attack_editor, 'MAX_INTERNAL_PER_CALL', 12)
    result = pool_module.construct_pool(editor, dict(parent_candidate_id='seed_000',
        objective=dict(interaction_window_s=[1.5, 7.5], keyframes=[]),
        max_candidates=1, reason='Mixed initial-state fixture'))

    assert result['candidates'] and result['status'] != 'interrupted'
    assert result['rejected_initial_poses'] == [
        dict(variant=2, reason='unsupported_initial_state: recorded steering exceeds generator range'),
        dict(variant=3, reason='Initial pose translation exceeds 30m'),
    ]
    compact = compact_action_feedback(compact_action_feedback(result))
    assert compact['initial_pose_filter']['rejected_count'] == 2
    pool = editor.run_dir/'construction_pools/pool_001'
    attempts = [json.loads(row) for row in (pool/'attempts.jsonl').read_text().splitlines()]
    assert len(attempts) == result['evaluations_spent']
    assert any(row['initial_pose_adjustment'] == eligible for row in attempts)
    assert not (pool/'pending-evaluation.json').exists()


def test_independent_review_replays_initial_pose_edits_across_ancestors(tmp_path):
    from review_attack_workflow import replay_initial_seed

    editor = make_trajectory_editor(tmp_path)
    first = editor.adjust_initial_pose(dict(parent_candidate_id='seed_000',
        position_offset_m=dict(forward_m=0., left_m=1.), yaw_mode='preserve',
        reason='Fixture initial pose change'))['candidate_id']
    controls = dict(times_s=[0., 8.], acceleration_mps2=[0., 0.], steering_rad=[0., 0.])
    second = editor.construct_controls(dict(parent_candidate_id=first, controls=controls,
        initial_pose_adjustment=dict(position_offset_m=dict(forward_m=1., left_m=0.),
            yaw_mode='offset', yaw_offset_rad=.1), reason='Fixture control seed change'))['candidate_id']
    third = editor.construct_controls(dict(parent_candidate_id=second, controls=controls,
        reason='Fixture inherited initial state'))['candidate_id']

    folder = editor.run_dir/'candidates'
    xy, yaw, provenance, operation = replay_initial_seed(folder/third, editor)
    np.testing.assert_allclose(xy, editor.nodes[third]['initial_seed_xy'])
    np.testing.assert_allclose(yaw, editor.nodes[third]['initial_seed_yaw'])
    np.testing.assert_allclose(xy[0], [16., 1.])
    assert provenance == ['adjust_initial_pose', 'construct_vehicle_controls']
    assert operation['input']['parent_candidate_id'] == second
    for key, expected in [('seed_000', editor.seed_xy), (first, editor.nodes[first]['initial_seed_xy'])]:
        actual, _, _, _ = replay_initial_seed(folder/key, editor)
        np.testing.assert_allclose(actual, expected)

    path = folder/second/'operation.json'
    record = json.loads(path.read_text())
    record['parent_candidate_id'] = 'seed_000'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='inconsistent parent candidate'):
        replay_initial_seed(folder/third, editor)


def test_initial_speed_change_is_encoded_in_seed_and_provenance():
    editor=pose_editor()
    xy,yaw,record=adjusted_seed(editor,{},dict(position_offset_m=dict(forward_m=0.,left_m=0.),
        yaw_mode='preserve',initial_speed_mps=1.5))
    assert np.allclose(xy[1]-xy[0],[.75,0.])
    assert record['initial_speed_mps']==1.5 and record['initial_speed_changed'] is True
    assert record['source_initial_speed_mps']==2.


def test_unified_search_mixes_corridor_pose_and_speed_without_agent_modes():
    editor=pose_editor()
    parent=dict(initial_seed_xy=editor.seed_xy,initial_seed_yaw=editor.seed_yaw,initial_pose=None,
        xy=editor.seed_xy.copy())
    editor.constraints['revision_feedback']=dict(evidence=dict(scene_feedback=dict(keyframes=[
        dict(time_s=1.5,ego_body_center_xy_m=[10.,5.],ego_heading_rad=.1)])))
    specs=search_specs({},editor,parent)
    assert any(spec and spec['position_offset_m']['left_m']==5. for spec in specs)
    assert any(spec and 'initial_speed_mps' in spec for spec in specs)
    assert len(specs)<=12


def adaptive_pool(monkeypatch,tmp_path,qualified_at):
    records=[]
    for index in range(1,9):
        record=dict(attempt_id=f'attempt_{index:03d}',controls=dict(index=index),
            guidance_cost=float(index),search_metrics={})
        records.append((float(index),np.asarray([float(index)]),record))
    class Search:
        active_pose=None
        rejected_initial_poses=[]
        def __init__(self,editor,arguments):pass
        def optimize(self,*args,**kwargs):return records
        def simulate(self,controls,initial_pose=None):
            xy=np.asarray([[controls['index'],0.],[controls['index'],0.]],dtype=float)
            return (xy,xy,xy,xy),None
    class Editor:
        def __init__(self):
            self.run_dir=tmp_path;self.constraints={};self.temporary=0;self.formal=[]
        def construct_controls(self,operation,persist=True):
            index=operation['controls']['index'];passed=index==qualified_at
            if persist:self.formal.append(index)
            else:self.temporary+=1
            return dict(candidate_id=f'candidate_{len(self.formal):03d}' if persist else 'temporary',
                reference_screen_passed=passed,pdm_probe_eligible=passed,failed_checks=[],
                construction_goal=dict(goal_met=passed,dense=dict(position_error_m=0.)))
    monkeypatch.setattr(pool_module,'ControlSearch',Search)
    return Editor()


def adaptive_arguments():
    return dict(parent_candidate_id='seed_000',objective=dict(interaction_window_s=[1.5,7.5],keyframes=[]),
        max_candidates=1,reason='adaptive strict screening fixture')


def test_strict_screening_continues_past_six_until_first_qualified(monkeypatch,tmp_path):
    editor=adaptive_pool(monkeypatch,tmp_path,7)
    result=pool_module.construct_pool(editor,adaptive_arguments())
    assert result['strict_finalist_count']==7
    assert result['strict_stop_reason']=='qualified_candidate_found_after_minimum'
    assert editor.temporary==7 and editor.formal==[7]
    assert result['candidates'][0]['candidate_id']=='candidate_001'


def test_strict_screening_covers_all_records_without_persisting_every_failure(monkeypatch,tmp_path):
    editor=adaptive_pool(monkeypatch,tmp_path,None)
    result=pool_module.construct_pool(editor,adaptive_arguments())
    assert result['strict_finalist_count']==8
    assert result['strict_stop_reason']=='all_optimization_records_screened'
    assert editor.temporary==8 and len(editor.formal)==1
    assert result['status']=='no_feasible_goal'
