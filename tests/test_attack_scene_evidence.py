"""Cheap hypotheses expose geometry without creating candidates or claiming policy success."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import box

import sys
CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'
if str(CORE) not in sys.path:sys.path.insert(0,str(CORE))
import acceptance_policy as policy
from attack_scene_evidence import conflict_context,reaction_screen,preview_options
from ego_road_checks import OfficialDrivableMap


def fixture():
    times=np.arange(0.,8.01,.5);ego=np.column_stack([4*times,np.zeros(len(times))]);yaw=np.zeros(len(times))
    road=OfficialDrivableMap([box(-100,-20,100,20)],{})
    config=dict(observation_end_s=1.5,event_window_s=[1.5,7.5],acceptance_policy=policy.for_new_run())
    return times,ego,yaw,[[] for _ in times],road,[dict(asset_id='car',size_m=[4.,2.,1.5])],config


def added(identifier,x,y=0.):
    return dict(option_id=identifier,operation='add_vehicle',asset_id='car',x_m=x,y_m=y,yaw_rad=0.,speed_mps=0.)


def test_preview_preserves_order_and_distinguishes_prefix_and_post_takeover_contact():
    args=fixture();before=args[1].copy()
    result=preview_options([added('far',60),added('near',15),added('too_early',5)],*args)
    assert [row['option_id'] for row in result['options']]==['far','near','too_early']
    assert result['ranked'] is result['candidate_created'] is False
    far,near,early=result['options']
    assert not far['prefix_target_contact'] and not near['prefix_target_contact']
    assert early['prefix_target_contact']
    assert far['reactions']['hypotheses'][0]['first_sampled_contact_s'] is None
    assert near['reactions']['hypotheses'][0]['first_sampled_contact_s'] is not None
    assert 'policy_attack_success' not in json.dumps(result)
    from attack_context_view import feedback_view
    assert feedback_view(result)['options']==result['options']
    assert len(json.dumps(feedback_view(result)))<9000
    np.testing.assert_array_equal(args[1],before)


def test_explicit_locks_and_invalid_rows_are_reported_without_mutation():
    args=fixture()
    result=preview_options([added('locked',15)],*args,required_operation='edit_existing')
    assert result['options'][0]['status']=='invalid'
    with pytest.raises(ValueError,match='1..3'):preview_options([added(str(i),20) for i in range(4)],*args)
    with pytest.raises(ValueError,match='distinct'):preview_options([added('same',15),added('same',20)],*args)
    assert preview_options([added('off_road',20,30)],*args)['options'][0]['road_passed'] is False


def test_conflict_positions_are_rear_axle_and_do_not_claim_visibility():
    times,ego,yaw,*_,config=fixture()
    result=conflict_context(times,ego,yaw,config)
    assert result['coordinate_frame']=='initial_ego' and result['ego_reference_point']=='rear_axle'
    assert result['takeover_speed_mps']==4
    assert result['reference_samples'][0]['rear_axle_xy_m']==[6.,0.]
    assert 'not visibility' in result['scope']


def test_route_exhaustion_is_explicit_instead_of_inventing_policy_stopping():
    times=np.arange(0.,4.1,.5);xy=np.array([[0,0],[4,0],[8,0],[12,0],[13,0],[14,0],[15,0],[16,0],[17,0]],float)
    result=reaction_screen(times,xy,np.zeros(len(times)),np.tile([40.,0.],(len(times),1)),np.zeros(len(times)),
        [4.,2.,1.5],dict(observation_end_s=1.5,event_window_s=[1.5,4.]))
    coast=next(row for row in result['hypotheses'] if row['hypothesis']=='constant_takeover_speed')
    assert coast['route_exhausted'] and coast['coverage_status']=='truncated_at_known_route_end'


def test_vad_states_convert_world_axes_body_centres_to_scene_rear_axle(tmp_path):
    from vad_scene_feedback import feedback
    # Initial scene heading is90deg; native WE axes translate the first rear axle.
    angle=np.pi/2;origin=np.eye(4);origin[:2,:2]=[[0,-1],[1,0]]
    states=[dict(step=i,time_s=i*.5,actors={'ego':{'box':[0.,float(i)+1.461,5.176,2.297,angle],
        'velocity':[0.,2.]},'target':{'box':[0.,15.,4.,2.,angle]}}) for i in range(5)]
    path=tmp_path/'states.jsonl';path.write_text('\n'.join(json.dumps(row) for row in states))
    editor=SimpleNamespace(frames=[{'ego2global':origin.tolist()}],target={'track_id':'target'})
    result=feedback(editor,dict(states_path=str(path),closest_approach={'step':4}))
    assert result['coordinate_frame']=='initial_ego'
    np.testing.assert_allclose(result['full_precision_states'][0]['ego_rear_axle_xy_m'],[0.,0.],atol=1e-8)
    np.testing.assert_allclose(result['full_precision_states'][-1]['target_body_center_xy_m'],[15.,0.],atol=1e-8)
    assert len(result['keyframes'])<=6
    assert feedback(editor,{})['status']=='unavailable'
