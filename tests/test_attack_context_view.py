"""Readable stage-specific views retain actionable evidence without full report duplication."""
from copy import deepcopy
import json
from pathlib import Path
import sys

CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'
sys.path.insert(0,str(CORE))
from attack_context_view import editing_view,selection_view


def test_edit_view_keeps_all_candidate_identities_actual_vad_and_hard_contract():
    precise=.12345678901234
    contract={'keyframes':[{'time_s':3.,'x_m':precise}]}
    raw=dict(instruction='original user requirement',target={'track_id':'car'},
        constraints={'task_contract':contract,'event_window_s':[1.5,7.5],
            '_fixed_ego_history':{'takeover':{'speed_mps':3.}},
            'policy_source_scene_ref':{'huge_source_catalog':['entry']*1000},
            'vadv2_policy_config':{'checkpoint_path':'/archive/weights'},
            'acceptance_policy':{'max_speed_mps':15.,'attack_lateral_slip_mps':.3}},
        candidates=[dict(candidate_id='candidate_'+str(i),reference_screen_passed=bool(i%2),
            failed_checks=['motion'] if i%2==0 else [],
            optimization_preferences={'rows':[{'value':precise}]*100}) for i in range(10)],
        timeline=[{'x':i} for i in range(100)],
        last_tool_result={'status':'inspection_ready','candidate_id':'candidate_3','path':'/figure.png','metrics':{'rows':['detail']*1000}},
        vadv2_self_test={'enabled':True,'calls':1,'remaining':2,
            'latest_feedback':{'candidate_id':'candidate_1','target_collision_detected':False,
                'min_target_clearance_m':precise,'policy_behavior_summary':{'stopped_detected':True}},
            'candidates':{'candidate_1':{'target_collision_detected':False,'min_target_clearance_m':precise}}},
        full_scene_evidence_path='/archive/scene-evidence.json')
    before=deepcopy(raw)
    detail={'candidate_id':'candidate_3','construction_goal_input':{'keyframes':contract['keyframes']},
        'failure_measurements':{'actual_speed_mps':precise},'task_failed_checks':['keyframe_1'],
        'diagnostic':{'suggested_changes':['Lengthen braking interval']},
        'motion_violations':[{'time_s':i,'actual_acceleration_mps2':8.12345678} for i in range(5)]}
    viewed=[]
    result=editing_view(raw,lambda key:(viewed.append(key) or detail))
    assert raw==before and viewed==['candidate_3']
    assert [row['candidate_id'] for row in result['candidate_index']]==[row['candidate_id'] for row in raw['candidates']]
    assert result['constraints']['task_contract']==contract
    assert result['fixed_ego_history']['takeover']['speed_mps']==3.
    assert result['vadv2_self_test']['latest_feedback']['min_target_clearance_m']==precise
    assert result['vadv2_self_test']['latest_feedback']['policy_behavior_summary']['stopped_detected']
    assert result['active_candidate_details']['failure_measurements']['actual_speed_mps']==precise
    assert result['active_candidate_details']['motion_violations'][0]['actual_acceleration_mps2']==8.12345678
    assert result['active_candidate_details']['motion_violations_omitted']==3
    assert result['active_candidate_details']['diagnostic']['suggested_changes']==['Lengthen braking interval']
    assert result['last_tool_result']['path']=='/figure.png'
    assert 'timeline' not in result and 'vadv2_policy_config' not in result['constraints']
    assert len(json.dumps(result))<len(json.dumps(raw))*.5


def test_selection_keeps_all_vehicle_ids_and_options_with_frozen_takeover():
    options=[{'option_id':'add','instruction':'Add a lead vehicle.'}]
    raw=dict(existing_vehicles=[{'track_id':str(i)} for i in range(20)],plan_options=options,
        constraints={'_fixed_ego_history':{'takeover':{'speed_mps':4.}},'plan_options':options},
        ego_timeline=[{'time_s':i*.5,'speed_to_next_mps':i} for i in range(24)],
        actors_at_start=['verbose']*1000)
    result=selection_view(raw)
    assert result['plan_options']==options and result['existing_vehicles']==raw['existing_vehicles']
    assert result['fixed_ego_history']['takeover']['speed_mps']==4.
    assert [row['time_s'] for row in result['ego_reference_samples']]==[0.,1.5,7.5,11.5]
    assert 'ego_timeline' not in result and 'actors_at_start' not in result


def test_constructor_feedback_keeps_failed_checks_and_measured_repair_limits():
    from attack_context_view import feedback_view
    raw=dict(status='invalid',error='target_unreachable',feasibility_limits={'minimum_transition_duration_s':1.23456789},
        decision_summary={'outcome':'rejected','guidance':'Lengthen the transition'},
        candidates=[{'candidate_id':'c1','reference_screen_passed':False,'failed_checks':['vehicle_motion']},
            {'candidate_id':'c2','reference_screen_passed':True,'failed_checks':[]}])
    result=feedback_view(raw)
    assert result['feasibility_limits']==raw['feasibility_limits']
    assert result['guidance']=='Lengthen the transition'
    assert result['candidates'][0]['failed_checks']==['vehicle_motion']
