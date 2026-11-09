"""Decision summaries retain every candidate; detailed evidence is expanded once per ID."""
from agent_tool_view import candidate_card


CANDIDATE_FIELDS = (
    'source', 'status', 'submit_status', 'editing_status', 'full_window_status',
    'lane_markings', 'input_qualification', 'native_max_acceleration_mps2', 'min_other_clearance_m',
    'min_road_margin_m', 'min_road_margin_official_m', 'terminal_speed_mps',
    'terminal_speed_source', 'terminal_sample_interval_speed_mps',
    'terminal_target_clearance_rate_mps', 'progress_ratio_of_recorded',
    'comfort_reference_2_40_exceeded', 'clearance_periods', 'motion_timeline', 'motion_outcome', 'behavior_metrics', 'behavior_action_summary')
CURVE_FIELDS = ('schema_version', 'max_speed_mps', 'max_acceleration_mps2',
    'tangential_deceleration_mps2', 'target_speed_mps', 'transition_time_s',
    'transition_distance_m', 'hold_starts_at_s', 'reference_path_candidate_id',
    'route_edit_summary', 'speed_edit_summary', 'acceleration_peak', 'speed_law', 'peak_tangential_deceleration_mps2', 'terminal_velocity_xy_mps')


def candidate_summary(value):
    result = candidate_card(value)
    result.update({key: value[key] for key in CANDIDATE_FIELDS if key in value})
    curve = value.get('generation_curve')
    if isinstance(curve, dict):
        result['curve_metrics'] = {key: curve[key] for key in CURVE_FIELDS if key in curve}
    motion = value.get('motion_evidence') or {}
    result['direction'] = {key: motion[key] for key in ('max_reverse_speed_mps', 'reverse_distance_m') if key in motion}
    intent = value.get('motion_intent') or {}
    result['intent'] = {key: intent[key] for key in ('declared', 'passed', 'terminal_stopped', 'terminal_state_method', 'hold_matched', 'failure_kind') if key in intent}
    operation = value.get('operation') or {}
    result['operation_summary'] = {key: operation[key] for key in ('mode', 'edit_mode', 'start_time_s', 'deceleration_mps2', 'target_speed_mps', 'lateral_offset_m', 'end_behavior', 'operation', 'targets', 'braking_start_shift_s', 'preserve') if key in operation}
    result['collision_geometry'] = {key: item for key, item in (value.get('collision_geometry') or {}).items() if key != 'scope'}
    return result


def candidate_details(value):
    result = candidate_summary(value)
    for key in ('operation', 'actual_default_start_state', 'near_stop_indices', 'conflict_window',
                'target_forward_left_at_minimum_m', 'target_overlap_indices', 'acceleration_components'):
        if key in value:
            result[key] = value[key]
    curve = value.get('generation_curve')
    if isinstance(curve, dict):
        result['curve_start_and_targets'] = {key: curve[key] for key in
            ('actual_start_state', 'resolved_targets', 'scope', 'speed_profile_scope') if key in curve}
    return result


def compact_context(context):
    result = dict(context)
    catalog = context['candidate_catalog']
    lookup = {row['candidate_id']: row for row in catalog}
    for name in ('current', 'best'):
        value = context.get(name)
        if isinstance(value, dict):
            lookup[value['candidate_id']] = value
    history = context.get('edit_history') or []
    latest = context.get('latest_generated_candidate_id') or (history[-1]['candidate_id'] if history else None)
    current = context.get('current_edit_parent_id') or (context.get('current') or {}).get('candidate_id')
    active = [current, latest, context.get('pending_review_candidate_id')]
    result['active_candidate_details'] = {key: candidate_details(lookup[key]) for key in dict.fromkeys(active)
                                         if key in lookup}
    summaries = [candidate_summary(row) for row in catalog]
    columns = list(dict.fromkeys(key for row in summaries for key in row))
    result['candidate_catalog'] = dict(columns=columns, rows=[[row.get(key) for key in columns] for row in summaries])
    for name in ('current', 'best'):
        if isinstance(context.get(name), dict):
            result[name] = {'candidate_id': context[name]['candidate_id'], 'evidence': 'candidate_catalog and active_candidate_details'}
    result['current_edit_parent_id'] = current
    result['latest_generated_candidate_id'] = latest
    result['untried_initial_candidate_ids'] = [row['candidate_id'] for row in context.get('untried_initial_candidates', [])]
    result.pop('untried_initial_candidates', None)
    # Shared prompt and tool schemas already define these modes once.
    result.pop('new_edit_modes', None)
    result.pop('edit_modes', None)
    feedback = context.get('last_tool_result')
    if isinstance(feedback, dict):
        feedback = dict(feedback)
        for key, value in list(feedback.items()):
            if isinstance(value, dict) and 'candidate_id' in value:
                # Candidate details live in the catalog/active table; retain actual edit/error evidence.
                if key in ('current', 'best', 'candidate', 'recommended_parent_preview'):
                    feedback[key] = {'candidate_id': value['candidate_id']}
        result['last_tool_result'] = feedback
    result['context_policy'] = ('candidate_catalog is a columns/rows table; each row uses exactly the declared column order. All identities, failures, unknown checks and selection metrics are retained. '
        'Curve generation definitions and acceptance rules are shared; expanded evidence appears once per active ID. '
        'Inspection returns other candidate details and a reference diagram. Full numeric evidence remains archived. '
        'Current edit parent, latest generated candidate and numerical best are distinct; ranking never selects a final candidate.')
    if 'candidate_order_note' in context:
        result['context_policy']='Candidate rows retain generation order, all IDs and checks; no quality ranking. Timelines use frame_times_s. Details expand for active IDs; full numeric evidence is archived.'
    return result
