"""Cheap action preflight and structured invalid-result feedback."""


def prevalidate_action(name, arguments, editor=None):
    if name == 'construct_attack_candidates':
        for frame in (arguments.get('objective') or {}).get('keyframes', []):
            start, end = frame['time_window_s']
            duration = frame['hold_duration_s']
            if duration < .1 or duration > end-start:
                raise ValueError('hold_duration_s must be >=0.1s and fit the keyframe window')
    if name in {'edit_speed_profile', 'edit_route'} and editor is not None:
        parent = arguments.get('parent_candidate_id')
        if parent == 'seed_000' and not editor.nodes[parent]['result'].get('construction_goal_input'):
            raise ValueError('Choose a parent with a declared objective from construct_attack_candidates; seed_000 has no objective to inherit')


def invalid_action_result(name, arguments, error):
    message = str(error)
    hints = []
    if name == 'construct_attack_candidates':
        for frame in (arguments.get('objective') or {}).get('keyframes', []):
            start, end = frame['time_window_s']
            hints.append(dict(time_window_s=[start, end], maximum_hold_duration_s=end-start,
                requested_hold_duration_s=frame['hold_duration_s']))
    if name in {'edit_speed_profile', 'edit_route'} and arguments.get('parent_candidate_id') == 'seed_000':
        hints.append('Construct an objective-bearing parent before editing its speed or route')
    return dict(status='invalid', error=message,
        decision_summary=dict(schema_version='agent_decision.v1', outcome='invalid_request', tool=name,
            parent_candidate_id=arguments.get('parent_candidate_id'),
            primary_failure=dict(check='invalid_request', message=message),
            next_step='Correct only the stated parameters and retry the same objective; no candidate search is required for this correction.',
            parameter_hints=hints))
