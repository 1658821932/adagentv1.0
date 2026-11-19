"""Stage B model input: fixed takeover, candidate index and current evidence only.

Full reports, trajectories and precision remain in the editor and audit archives.
This view never changes eligibility, construction or collision measurements.
"""
from copy import deepcopy

from agent_tool_view import candidate_card, compact_action_feedback


def constraints_view(value):
    value=value or {}
    hidden={'acceptance_policy','plan_options','policy_source_scene_ref','vadv2_policy_config',
        'object_selection_feedback','pdm_probe_root','vadv2_probe_root','revision_feedback','pdm_reselection_feedback',
        'prepare_ego_history','ego_history_target_speed_mps','ego_history_route','ego_history_revision_count'}
    result={key:deepcopy(item) for key,item in value.items() if key not in hidden and not key.startswith('_')}
    policy=value.get('acceptance_policy') or {}
    result['physical_limits']={key:deepcopy(item) for key,item in policy.items()
        if key.startswith(('max_','attack_max_')) or key in ('official_margin_m','road_tolerance_m',
            'attack_lateral_slip_mps','attack_steering_rad','attack_steering_rate_radps','attack_reverse_tolerance_mps',
            'attack_stationary_yaw_rate_radps','attack_danger_start_s','attack_window_start_s','attack_window_end_s')}
    result['acceptance_requirements']={key:deepcopy(policy[key]) for key in
        ('policy_version','required_checks','edit_start_s','observation_end_s','history_policy') if key in policy}
    if 'revision_feedback' in value:
        review=value['revision_feedback']
        result['prior_review']={key:deepcopy(review[key]) for key in ('stage','outcome','guidance') if key in review}
        if isinstance(review.get('evidence'),dict):result['prior_review']['evidence']=feedback_view(review['evidence'])
    if isinstance(value.get('pdm_reselection_feedback'),dict):
        result['prior_pdm_feedback']=feedback_view(value['pdm_reselection_feedback'])
    return result


def feedback_view(value):
    if not isinstance(value,dict):return value
    result=compact_action_feedback(value)
    if value.get('candidate_id') is not None:
        result.update(candidate_details(value))
    for key in ('policy_attack_success','policy_evaluation','target_collision_detected','first_target_collision_time_s',
                'min_target_clearance_m','closest_approach','policy_behavior_summary','evaluation_window_s','scene_feedback',
                'initial_history_target_contact','possible_interpolated_target_contact','elapsed_s',
                'overlay_path','result_path','path','focus_time_s','guidance','recovery_hint','reason',
                'options','remaining_hypotheses','ranked','scope','code','operation','track_id','option_id',
                'initial_state','measurements','repeated','next_action',
                'min_body_clearance_m','peak_planar_acceleration_mps2','planar_acceleration_limit_mps2',
                'acceleration_limit_fraction','reserve_reference_m','feedback_path','connector_route'):
        if key in value:result[key]=deepcopy(value[key])
    if isinstance(value.get('candidates'),list):
        result['candidates']=[candidate_card(dict(row,
            reference_screen_passed=row.get('reference_screen_passed',row.get('reference_geometry_passed'))))
            for row in value['candidates'] if isinstance(row,dict)]
        for card, row in zip(result['candidates'], (row for row in value['candidates'] if isinstance(row, dict))):
            if 'connector_passage' in row:
                card['connector_passage'] = deepcopy(row['connector_passage'])
    metrics=value.get('metrics')
    if isinstance(metrics,dict):
        result['candidate']=candidate_details(metrics)
    return result


def candidate_details(value):
    result=candidate_card(value)
    for key in ('diagnostic','failure_diagnostics','speed_edit_summary','route_edit_summary',
                'acceleration_peak','motion_intent','first_contact_time_s','event_min_clearance_m',
                'task_failed_checks','failure_measurements','construction_goal_input'):
        if value.get(key) is not None:result[key]=deepcopy(value[key])
    for key in ('motion_violations','road_violations','road_surface_failures','other_contacts'):
        rows=value.get(key)
        if isinstance(rows,list) and rows:
            result[key]=deepcopy(rows[:2])
            if len(rows)>2:result[key+'_omitted']=len(rows)-2
    return result


def _probe_view(value):
    value=value or {}
    result={key:deepcopy(value[key]) for key in ('enabled','calls','remaining') if key in value}
    if value.get('latest_feedback') is not None:result['latest_feedback']=feedback_view(value['latest_feedback'])
    result['tested_candidates']=[dict(candidate_id=key,**{name:deepcopy(row[name]) for name in
        ('policy_attack_success','policy_evaluation','target_collision_detected','min_target_clearance_m') if name in row})
        for key,row in (value.get('candidates') or {}).items()]
    return result


def _fixed_history_view(constraints):
    state = constraints.get('_fixed_ego_history') or {}
    return {key: deepcopy(state[key]) for key in ('frozen', 'end_s', 'status', 'takeover') if key in state} or None


def selection_view(context):
    result=deepcopy(context)
    constraints=context.get('constraints') or {}
    result['constraints']=constraints_view(constraints)
    result['fixed_ego_history']=_fixed_history_view(constraints)
    result.pop('actors_at_start',None)
    timeline=context.get('ego_timeline') or []
    if timeline:
        result['ego_reference_samples']=[timeline[index] for index in sorted({0,len(timeline)-1,
            min(range(len(timeline)),key=lambda i:abs(timeline[i]['time_s']-1.5)),
            min(range(len(timeline)),key=lambda i:abs(timeline[i]['time_s']-7.5))})]
        result.pop('ego_timeline',None)
    result['future_scope']='Post-takeover reference motion is not a prediction of VAD actions.'
    return result


def editing_view(context,detail_provider=None):
    result=deepcopy(context)
    constraints=context.get('constraints') or {}
    result['constraints']=constraints_view(constraints)
    result['fixed_ego_history']=_fixed_history_view(constraints)
    result['vadv2_self_test']=_probe_view(context.get('vadv2_self_test'))
    feedback=context.get('last_tool_result') or {}
    active=feedback.get('candidate_id')
    if active is None:
        active=next((row.get('candidate_id') for row in feedback.get('candidates',[]) if isinstance(row,dict) and row.get('candidate_id')),None)
    if active is None:active=(context.get('vadv2_self_test',{}).get('latest_feedback') or {}).get('candidate_id')
    catalog=context.get('candidates') or []
    result['candidate_index']=[candidate_card(row) for row in catalog]
    result.pop('candidates',None)
    result.pop('timeline',None)
    result.pop('edit_history',None)
    result['last_tool_result']=feedback_view(feedback)
    if active is not None and detail_provider is not None:
        value=detail_provider(active)
        result['active_candidate_details']=candidate_details(value)
    result['active_candidate_id']=active
    result['evidence_scope']='Candidate index is unranked reference geometry. Only VAD probe feedback describes actual policy motion. Inspect any candidate for its image and detailed checks; complete evidence stays in archived files.'
    return result
