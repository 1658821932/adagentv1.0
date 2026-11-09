"""Compact, actionable Agent views over unchanged editing evidence."""
import copy


_INTERNAL_ARGUMENTS = {
    'edit_mode', 'mode', 'speed_schedule_version', 'route_edit_version',
    'path_geometry_version', 'curve_state_version',
}


def _tool_name(arguments):
    return 'edit_speed_profile' if arguments.get('edit_mode') == 'speed_schedule' else 'edit_route'


def _call(arguments, *, preview):
    clean = {key: copy.deepcopy(value) for key, value in arguments.items()
             if key not in _INTERNAL_ARGUMENTS}
    clean['preview_only'] = preview
    return dict(tool=_tool_name(arguments), arguments=clean)


def _recommended_previews(result, arguments):
    error = str(result.get('error', ''))
    code = result.get('error_code') or error.split(':', 1)[0]
    recommendations = []

    def add(condition, **changes):
        call = _call(arguments, preview=True)
        call['arguments'].update(changes)
        recommendations.append(dict(condition=condition, **call))

    if code == 'route_exhausted':
        if arguments.get('edit_mode') == 'speed_schedule' and arguments.get('path_source', 'parent') == 'parent':
            add('Continue only when following the known recorded route matches the task; all checks still apply.',
                path_source='recorded_baseline')
        if arguments.get('edit_mode') == 'route_edit' and arguments.get('speed_policy', 'preserve_speed') == 'preserve_speed':
            add('Use only when preserving along-route progress is worth a possible total-speed increase.',
                speed_policy='preserve_progress')
    elif code == 'target_unreachable' and arguments.get('edit_mode') == 'speed_schedule':
        limits = result.get('feasibility_limits') or {}
        interval = limits.get('interval_s')
        minimum = limits.get('minimum_transition_duration_s')
        targets = copy.deepcopy(arguments.get('targets') or [])
        if interval and len(interval) == 2 and minimum is not None:
            for index, target in enumerate(targets):
                start = arguments.get('start_time_s') if index == 0 else targets[index-1]['time_s']
                if abs(start-interval[0]) > 1e-6 or abs(target['time_s']-interval[1]) > 1e-6:
                    continue
                delayed = start+minimum
                next_time = targets[index+1]['time_s'] if index+1 < len(targets) else limits.get('scene_end_s')
                if next_time is not None and delayed < next_time+1e-8:
                    targets[index]['time_s'] = delayed
                    add(f'Delay only this speed target to the measured minimum transition end ({delayed:.6f}s); later targets and all checks remain unchanged.',
                        targets=targets)
                break
    elif code == 'immutable_history':
        boundary = (result.get('feasibility_limits') or {}).get('earliest_editable_time_s')
        if boundary is not None and 'start_time_s' in arguments:
            add('Start at the earliest editable time only if preserving the requested later behavior still matches the task.',
                start_time_s=boundary)
    return recommendations


def _outcome(result):
    status = result.get('status')
    if status == 'invalid':
        return 'invalid_request'
    if status == 'preview':
        if result.get('feasible') is True:
            return 'preview_passed'
        if result.get('error_code') or result.get('failure_kind') == 'requested_goal_unreachable':
            return 'preview_infeasible'
        return 'preview_checks_failed'
    if status == 'not_feasible' or result.get('failure_kind') == 'requested_goal_unreachable':
        return 'request_infeasible'
    if result.get('candidate_id'):
        passed = result.get('reference_screen_passed')
        if passed is None:
            passed = status == 'reference_screen_passed'
        task = result.get('task_contract_passed')
        if task is None and isinstance(result.get('construction_goal'), dict):
            task = result['construction_goal'].get('goal_met')
        return 'candidate_checks_passed' if passed and task is not False else 'candidate_checks_failed'
    return status or 'completed'


def _next_step(outcome):
    return {
        'preview_passed': 'If this behavior is worth retaining, use create_candidate_call; creation still does not imply submission.',
        'preview_infeasible': 'Choose a conditional recommended preview, explicitly adjust the goal, or keep a passing fallback.',
        'preview_checks_failed': 'Read failed_checks and the referenced evidence before revising; a preview cannot be submitted.',
        'request_infeasible': 'Choose a conditional recommended preview, explicitly adjust the goal, or keep a passing fallback.',
        'candidate_checks_passed': 'Review the candidate and complete stage-specific inspection or PDM gates before submission.',
        'candidate_checks_failed': 'Use failed_checks to revise this candidate or choose another parent; it is not submission-eligible.',
        'invalid_request': 'Correct the stated request using the error and parameter hints; do not infer trajectory infeasibility.',
    }.get(outcome, 'Read the returned status and evidence before choosing the next action.')


def with_decision_summary(result, arguments):
    """Add a compact decision layer without changing raw feedback or acceptance."""
    enriched = dict(result)
    outcome = _outcome(result)
    failed = list(dict.fromkeys([
        *result.get('failed_checks', []), *result.get('task_failed_checks', []),
        *(result.get('diagnostic') or {}).get('failures', []),
    ]))
    summary = dict(
        schema_version='agent_decision.v1',
        outcome=outcome,
        tool=_tool_name(arguments),
        parent_candidate_id=arguments.get('parent_candidate_id'),
        candidate_id=result.get('candidate_id'),
        candidate_created=bool(result.get('candidate_created', bool(result.get('candidate_id')))),
        failed_checks=failed,
        next_step=_next_step(outcome),
        recommended_previews=_recommended_previews(result, arguments),
        evidence_fields=[key for key in (
            'error', 'feasibility_limits', 'checks', 'failure_diagnostics',
            'speed_edit_summary', 'route_edit_summary', 'motion_intent',
        ) if result.get(key) is not None],
    )
    if result.get('error_code') or result.get('error'):
        summary['primary_failure'] = dict(
            check=result.get('error_code') or (failed[0] if failed else str(result.get('error')).split(':', 1)[0]),
            message=result.get('error_summary') or result.get('error'),
        )
    elif failed:
        summary['primary_failure'] = dict(check=failed[0], evidence='failed_checks and evidence_fields')
    if outcome == 'preview_passed':
        summary['create_candidate_call'] = dict(
            condition='Create only if the previewed behavior is desired; stage-specific submission gates remain.',
            **_call(arguments, preview=False),
        )
    enriched['decision_summary'] = summary
    return enriched


def candidate_card(value):
    """Common candidate identity and eligibility fields for attack and ego views."""
    reference = value.get('reference_screen_passed')
    if reference is None and value.get('status') in ('reference_screen_passed', 'reference_screen_failed'):
        reference = value['status'] == 'reference_screen_passed'
    task = value.get('task_contract_passed')
    if task is None and isinstance(value.get('construction_goal'), dict):
        task = value['construction_goal'].get('goal_met')
    card = dict(
        candidate_id=value.get('candidate_id'),
        parent_candidate_id=value.get('parent_candidate_id', value.get('parent_id')),
        reference_screen_passed=reference,
        task_contract_passed=task,
        failed_checks=list(value.get('failed_checks', [])),
    )
    for key in (
        'source', 'pdm_probe_eligible', 'task_failed_checks', 'min_ego_clearance_m',
        'min_target_clearance_m', 'closest_time_s', 'minimum_clearance_time_s',
        'max_speed_mps', 'max_acceleration_mps2', 'official_margin_m',
    ):
        if key in value:
            card[key] = value[key]
    if 'coverage' in value:
        card['unknown_checks'] = [key for key, state in (value.get('coverage') or {}).items()
                                  if state not in ('passed', 'failed')]
    return card
