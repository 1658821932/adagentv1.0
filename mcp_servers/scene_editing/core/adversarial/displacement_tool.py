"""Sparse edits of a recorded trajectory in one fixed local frame."""
import numpy as np
from scipy.interpolate import BPoly, PchipInterpolator

from trajectory_checks import (screen_reference, screen_reference_relative, continuous_headings, phase_label,
    direction_consistency_screen, DIRECTION_CONSISTENCY_VERSION)
from trajectory_search_feedback import rectangle_overlap, entry_dynamics


KNOT_FIELDS = ('time_s', 'delta_forward_m', 'delta_left_m')
TAIL_FOLLOW_BLEND_SECONDS = 0.8


def follow_tail_blend_weight(s):
    """C2 smoothstep weight: 0 at start, 1 at end, zero first/second derivatives at both ends."""
    s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
    return 10*s**3 - 15*s**4 + 6*s**5


def interpolate_offsets(knots, times, endpoint_velocities=None):
    """C2 quintics through offsets; shared slopes, zero knot acceleration.

    PCHIP estimates interior slopes; endpoint offset velocity is zero. Equal
    adjacent offsets have zero slopes and remain constant on that interval.
    """
    if not isinstance(knots, list) or not 2 <= len(knots) <= 24:
        raise ValueError('Provide 2..24 time-stamped offset knots')
    if any(not isinstance(k, dict) or set(k) != set(KNOT_FIELDS) or
           any(type(k[f]) not in (int, float) or not np.isfinite(k[f]) for f in KNOT_FIELDS)
           for k in knots):
        raise ValueError('Each knot needs exactly three finite numeric fields: '+','.join(KNOT_FIELDS))
    values = np.array([[k[f] for f in KNOT_FIELDS] for k in knots], dtype=float)
    t, offsets = values[:, 0], values[:, 1:]
    if t[0] != times[0] or abs(t[-1]-times[-1]) > 1e-8 or np.any(np.diff(t) < .2-1e-8):
        raise ValueError('Knots must cover the sample window, strictly ordered and separated by >=0.2s')
    if np.any(offsets[0] != 0):
        raise ValueError('Initial offsets must both be zero; initial state cannot teleport')
    slopes = PchipInterpolator(t, offsets, axis=0).derivative()(t)
    slopes[[0, -1]] = 0
    if endpoint_velocities is not None:
        endpoint_velocities = np.asarray(endpoint_velocities, dtype=float)
        if endpoint_velocities.shape != (2, 2) or not np.isfinite(endpoint_velocities).all():
            raise ValueError('Endpoint offset velocities must be a finite 2x2 array')
        slopes[[0, -1]] = endpoint_velocities
    spline = BPoly.from_derivatives(t, [[p, v, np.zeros(2)] for p, v in zip(offsets, slopes)])
    return spline(times), spline


NORMAL_FIELDS = ('normal_before_until_s', 'normal_after_from_s')
NORMAL_POLICIES = ('return_to_recorded', 'constant_offset_after', 'follow_ego_path_after',
                   'follow_ego_path_after_c2')
FOLLOW_POLICIES = ('follow_ego_path_after', 'follow_ego_path_after_c2')
SUBMISSION_FIELDS = ('candidate_id', 'image_marker', 'intended_event', 'normal_before_until_s',
                     'normal_after_from_s', 'measured_risk', 'remaining_uncertainty')
OPTIONAL_SUBMISSION_FIELDS = ('event_complete_by_s',)
REFERENCE_SUBMISSION_FIELDS = ('candidate_id', 'event_complete_by_s', 'selection_reason', 'remaining_uncertainty')
RISK_EVIDENCE_FIELDS = ('minimum_center_distance_m', 'closest_time_s',
    'minimum_projected_body_gap_while_laterally_overlapping_m', 'projected_body_gap_time_s',
    'minimum_constant_velocity_ttc_s', 'minimum_ttc_time_s')


def check_normal_intervals(arguments, times, spline, minimum_duration,
                           policy='return_to_recorded', minimum_prefix_duration=None):
    if policy not in NORMAL_POLICIES:
        raise ValueError('Unknown normal interval policy')
    before = float(arguments['normal_before_until_s'])
    after = float(arguments['normal_after_from_s'])
    knots = arguments['knots']
    knot_times = np.array([k['time_s'] for k in knots], dtype=float)
    knot_offsets = np.array([[k['delta_forward_m'], k['delta_left_m']] for k in knots], dtype=float)
    offsets = spline(times)
    velocities = spline.derivative()(times)
    tail_offset = knot_offsets[-1] if policy in ('constant_offset_after', *FOLLOW_POLICIES) else np.zeros(2)
    expected_knots = np.zeros_like(knot_offsets)
    expected_samples = np.zeros_like(offsets)
    expected_knots[knot_times >= after-1e-8] = tail_offset
    expected_samples[times >= after-1e-8] = tail_offset
    violations = []

    def require(condition, metric, actual, requirement):
        if not condition:
            violations.append(dict(metric=metric, actual=actual, requirement=requirement))

    require(0 <= before < after <= times[-1], 'ordered_normal_intervals', [before, after],
            f'0 <= before < after <= {times[-1]}')
    prefix_min = minimum_duration if minimum_prefix_duration is None else minimum_prefix_duration
    require(before >= prefix_min, 'normal_before_duration_s', before,
            f'>={prefix_min}')
    latest_after = float(times[-1]-minimum_duration)
    require(times[-1]-after >= minimum_duration, 'normal_after_duration_s', times[-1]-after,
            f'>={minimum_duration} (normal_after_from_s <= {latest_after:.3f}s)')
    for name, value in [('normal_before_until_s', before), ('normal_after_from_s', after)]:
        require(bool(np.isclose(knot_times, value, rtol=0, atol=1e-8).any()), name, value,
                'must exactly match a knot time')
    if policy == 'follow_ego_path_after_c2':
        # New explicit contract only. Legacy policies retain their historical
        # non-grid boundary behavior so archived candidates keep their verdicts.
        for name, value in [('normal_before_until_s', before), ('normal_after_from_s', after)]:
            require(bool(np.isclose(times, value, rtol=0, atol=1e-8).any()), name+'_sample_time', value,
                    'must exactly match a 10 Hz sample time; a non-grid boundary is not silently shifted')
    in_normal_knots = (knot_times <= before+1e-8) | (knot_times >= after-1e-8)
    in_normal_samples = (times <= before+1e-8) | (times >= after-1e-8)
    knot_magnitudes = np.linalg.norm(knot_offsets, axis=1)
    offset_magnitudes = np.linalg.norm(offsets, axis=1)
    knot_deviations = np.linalg.norm(knot_offsets-expected_knots, axis=1)
    sample_deviations = np.linalg.norm(offsets-expected_samples, axis=1)
    velocity_magnitudes = np.linalg.norm(velocities, axis=1)
    knot_offset_max = float(knot_magnitudes[in_normal_knots].max()) if in_normal_knots.any() else 0.0
    offset_max = float(offset_magnitudes[in_normal_samples].max()) if in_normal_samples.any() else 0.0
    velocity_max = float(velocity_magnitudes[in_normal_samples].max()) if in_normal_samples.any() else 0.0
    knot_deviation_max = float(knot_deviations[in_normal_knots].max()) if in_normal_knots.any() else 0.0
    sample_deviation_max = float(sample_deviations[in_normal_samples].max()) if in_normal_samples.any() else 0.0
    if policy == 'return_to_recorded':
        require(knot_offset_max <= 1e-9, 'normal_interval_max_knot_offset_m', knot_offset_max, '<=1e-9')
        require(offset_max <= 1e-8, 'normal_interval_max_sampled_offset_m', offset_max, '<=1e-8')
    else:
        require(knot_deviation_max <= 1e-9, 'normal_interval_max_knot_deviation_m', knot_deviation_max, '<=1e-9')
        require(sample_deviation_max <= 1e-8, 'normal_interval_max_sampled_deviation_m', sample_deviation_max, '<=1e-8')
    require(velocity_max <= 1e-8, 'normal_interval_max_correction_velocity_mps', velocity_max, '<=1e-8')
    offending_knots = [dict(time_s=float(knot_times[i]), delta_forward_m=float(knot_offsets[i, 0]),
        delta_left_m=float(knot_offsets[i, 1]), expected_offset_m=expected_knots[i].tolist())
        for i in np.flatnonzero(in_normal_knots & (knot_deviations > 1e-9))]
    offending_indices = np.flatnonzero(in_normal_samples & ((sample_deviations > 1e-8) | (velocity_magnitudes > 1e-8)))
    offending_indices = offending_indices[np.argsort(-np.maximum(sample_deviations[offending_indices],
        velocity_magnitudes[offending_indices]))][:8]
    offending_samples = [dict(time_s=float(times[i]), correction_forward_m=float(offsets[i, 0]),
        correction_left_m=float(offsets[i, 1]), correction_velocity_forward_mps=float(velocities[i, 0]),
        correction_velocity_left_mps=float(velocities[i, 1])) for i in offending_indices]
    grid_hint = '; both boundaries must be exact 10 Hz sample times' if policy == 'follow_ego_path_after_c2' else ''
    repair_hint = (f'Choose ordered knot boundaries with prefix duration >= {prefix_min}s and tail duration >= {minimum_duration}s '
        f'(normal_after_from_s <= {latest_after:.3f}s){grid_hint}. Keep prefix knots at (0,0). '
        'Keep every tail knot equal to the same final offset, which may be nonzero, with zero correction velocity; '
        'do not catch up to the recorded position just to erase displacement.' if policy in ('constant_offset_after', *FOLLOW_POLICIES) else
        'Make both boundaries exact knot times; set every knot at or before the before-boundary and at or after the after-boundary to (0,0), with enough consecutive zero knots to cover both minimum-duration intervals. A zero final knot alone is insufficient.')
    meanings = {'constant_offset_after': 'Unedited prefix; constant displacement tail',
        'follow_ego_path_after': 'Unedited prefix; constant knot offsets, then Python holds the ego-relative lateral offset',
        'follow_ego_path_after_c2': 'Unedited prefix; C2 transition before the tail, then exact ego-relative lateral hold',
        'return_to_recorded': 'Unedited prefix and tail'}
    return dict(status='failed' if violations else 'passed', normal_before_until_s=before,
        normal_after_from_s=after, normal_after_latest_s=latest_after,
        sample_interval_s=float(np.median(np.diff(times))) if len(times) > 1 else None,
        minimum_duration_s=float(minimum_duration),
        minimum_prefix_duration_s=float(prefix_min), minimum_tail_duration_s=float(minimum_duration),
        normal_phase_preserved=not violations, policy=policy,
        tail_offset_m=tail_offset.tolist(),
        normal_phase_meaning=meanings[policy],
        violations=violations,
        offending_knots=offending_knots, offending_samples=offending_samples,
        repair_hint=repair_hint,
        measured=dict(max_abs_knot_offset_m=knot_offset_max, max_abs_sampled_offset_m=offset_max,
                      max_knot_normal_target_deviation_m=knot_deviation_max,
                      max_sampled_normal_target_deviation_m=sample_deviation_max,
                      max_abs_sampled_correction_velocity_mps=velocity_max),
        verification='Equal knots preserve constant offsets over complete normal intervals; samples and correction derivatives are also checked')


def apply_ego_path_follow(xy, times, ego, ego_heading, after):
    """Legacy exact hold at the declared boundary (frozen for old runs).

    This keeps the historical behavior: the ego-relative lateral offset is held
    constant from the first tail sample. It is intentionally not smoothed; old
    configurations and their archived evaluations retain this policy.
    """
    xy = np.array(xy, dtype=float, copy=True)
    delta = xy-ego
    cs, sn = np.cos(ego_heading), np.sin(ego_heading)
    forward = cs*delta[:, 0]+sn*delta[:, 1]
    left = -sn*delta[:, 0]+cs*delta[:, 1]
    tail = np.flatnonzero(times >= after-1e-8)
    if not len(tail):
        return xy, None
    hold = float(left[tail[0]])
    adjusted = np.column_stack([ego[tail, 0]+cs[tail]*forward[tail]-sn[tail]*hold,
                                ego[tail, 1]+sn[tail]*forward[tail]+cs[tail]*hold])
    change = np.linalg.norm(adjusted-xy[tail], axis=1)
    xy[tail] = adjusted
    return xy, dict(policy='follow_ego_path_after', applied_from_time_s=float(times[tail[0]]),
        hold_relative_left_m=hold, max_lateral_adjustment_m=float(change.max()),
        scope='Legacy exact hold from the boundary; no transition. The forward component remains '
              'knot-planned. Following the recorded ego path is not lane keeping.')


def apply_ego_path_follow_c2(xy, times, ego, ego_heading, after, before=None,
                             blend_seconds=TAIL_FOLLOW_BLEND_SECONDS):
    """Reach the ego-relative lateral hold at `after` through a C2 transition before it.

    The forward component stays as planned by the knots. A quintic transition starts
    shortly before the declared tail (but never inside the unedited prefix) and reaches
    a constant ego-relative lateral hold exactly at the boundary, cancelling the recorded
    drift. This keeps the declared normal tail itself free of a hidden transition while
    preserving position, velocity and acceleration at the hand-off. If there is no room
    for the transition, the hold is immediate and the normal check remains responsible
    for rejecting the too-short tail. The ego path remains a proxy, not a lane boundary.
    """
    xy = np.array(xy, dtype=float, copy=True)
    delta = xy-ego
    cs, sn = np.cos(ego_heading), np.sin(ego_heading)
    forward = cs*delta[:, 0]+sn*delta[:, 1]
    left = -sn*delta[:, 0]+cs*delta[:, 1]
    tail = np.flatnonzero(times >= after-1e-8)
    if not len(tail):
        return xy, None
    t0 = float(times[tail[0]])
    hold = float(left[tail[0]])
    transition_start = max(float(times[0]), t0-float(blend_seconds))
    if before is not None:
        transition_start = max(transition_start, float(before))
    start_index = int(np.searchsorted(times, transition_start-1e-8))
    start_index = min(max(start_index, 0), int(tail[0]))
    transition_start = float(times[start_index])
    duration = t0-transition_start
    left_out = left.copy()
    if duration > 1e-12 and start_index < int(tail[0]):
        weight = follow_tail_blend_weight((times[start_index:tail[0]]-transition_start)/duration)
        left_out[start_index:tail[0]] = left[start_index:tail[0]]+weight*(hold-left[start_index:tail[0]])
    left_out[tail] = hold
    changed = np.flatnonzero(times >= transition_start-1e-8)
    adjusted = np.column_stack([ego[changed, 0]+cs[changed]*forward[changed]-sn[changed]*left_out[changed],
                                ego[changed, 1]+sn[changed]*forward[changed]+cs[changed]*left_out[changed]])
    change = np.linalg.norm(adjusted-xy[changed], axis=1)
    xy[changed] = adjusted
    return xy, dict(policy='follow_ego_path_after_c2', applied_from_time_s=t0,
        hold_relative_left_m=hold, blend_duration_s=duration,
        transition_start_s=transition_start, transition_end_s=t0,
        blend_profile='quintic_C2_zero_endpoint_derivatives',
        max_lateral_adjustment_m=float(change.max()),
        scope='A C2 lateral transition before the boundary reaches the constant ego-relative hold '
              'at the boundary; the normal tail itself is exact hold. The transition is capped at '
              'normal_before_until_s so it cannot edit the unedited prefix. The forward component '
              'remains knot-planned. Following the recorded ego path is not lane keeping.')


def add_displacement_violation_context(violations, spline, normal):
    for violation in violations:
        local = violation.get('local_motion', {})
        start = local.get('interval_start_time_s', local.get('segment_start_time_s'))
        end = local.get('interval_end_time_s', local.get('segment_end_time_s'))
        if start is not None and end is not None:
            local['correction_offset_start_m'] = spline(start).tolist()
            local['correction_offset_end_m'] = spline(end).tolist()
            local['correction_velocity_start_mps'] = spline.derivative()(start).tolist()
            local['correction_velocity_end_mps'] = spline.derivative()(end).tolist()
        boundaries = []
        for field in NORMAL_FIELDS:
            boundary_time = float(normal[field])
            signed = float(violation['time_s']-boundary_time)
            boundaries.append(dict(boundary_field=field, boundary_time_s=boundary_time,
                signed_violation_time_minus_boundary_s=signed, distance_to_boundary_s=abs(signed),
                relation='before_boundary' if signed < 0 else 'after_boundary' if signed > 0 else 'at_boundary'))
        nearest = min(boundaries, key=lambda item: item['distance_to_boundary_s'])
        violation['normal_boundary_context'] = nearest
        violation['boundary_repair_hint'] = (
            f"Nearest declared normal boundary is {nearest['boundary_field']} at "
            f"{nearest['boundary_time_s']:.3f}s, {nearest['distance_to_boundary_s']:.3f}s from this violation. "
            'Inspect knots and recovery slopes around both times; do not change only the final knot.')


def sampled_risk_intervals(times, valid, ttc, closing):
    indices = np.flatnonzero(valid)
    if not len(indices):
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1)
    result = []
    for group in groups:
        minimum = int(group[np.argmin(ttc[group])])
        result.append(dict(start_time_s=float(times[group[0]]), end_time_s=float(times[group[-1]]),
            minimum_constant_velocity_ttc_s=float(ttc[minimum]), minimum_ttc_time_s=float(times[minimum]),
            maximum_forward_closing_speed_mps=float(closing[group].max())))
    return result


def validated_assessment(evaluation):
    assessment = dict(status='passed', numerical_screen=evaluation['numerical_screen'],
        normal_screen=evaluation.get('normal_screen'), risk_metrics=evaluation['risk_metrics'],
        endpoint_displacement_from_recorded_m=evaluation['endpoint_displacement_from_recorded_m'],
        event_success=None,
        assessment_scope='Reference samples, declared normal intervals and structured risk values checked; no CARLA execution or critical-event certification')
    if 'other_actor_screen' in evaluation:
        assessment['other_actor_screen']=evaluation['other_actor_screen']
    return assessment


def validate_displacement_submission(arguments, candidate, image_marker, tolerance=.02,
                                     intent_policy='exact', binding_policy='explicit'):
    if intent_policy not in {'exact', 'candidate_reference'}:
        raise ValueError('Unknown submission intent policy')
    if binding_policy not in {'explicit', 'candidate_reference'}:
        raise ValueError('Unknown submission binding policy')
    resolved = None
    if binding_policy == 'candidate_reference':
        if not isinstance(arguments, dict) or set(arguments) != set(REFERENCE_SUBMISSION_FIELDS):
            return dict(status='rejected', reason='Require candidate_id, event_complete_by_s, selection_reason and remaining_uncertainty; Python binds the evaluated facts')
        if candidate is None:
            return dict(status='rejected', reason='Unknown candidate')
        if not isinstance(arguments['selection_reason'], str) or not arguments['selection_reason'].strip():
            return dict(status='rejected', reason='Require a nonempty candidate selection reason')
        evaluation, parameters = candidate['evaluation'], candidate['parameters']
        if evaluation.get('status') != 'passed':
            return dict(status='rejected', reason='Candidate did not pass evaluation')
        resolved = dict(candidate_id=arguments['candidate_id'], image_marker=image_marker,
            event_complete_by_s=arguments['event_complete_by_s'],
            remaining_uncertainty=arguments['remaining_uncertainty'],
            measured_risk={key: evaluation['risk_metrics'][key] for key in RISK_EVIDENCE_FIELDS},
            **{key: parameters[key] for key in NORMAL_FIELDS})
        if intent_policy == 'exact':
            resolved['intended_event'] = evaluation['intent']
        arguments = resolved
    required = set(SUBMISSION_FIELDS)
    if intent_policy == 'candidate_reference':
        required.remove('intended_event')
    if not isinstance(arguments, dict) or not (required <= set(arguments)
            <= required|set(OPTIONAL_SUBMISSION_FIELDS)):
        return dict(status='rejected', reason='Require exactly the structured v2 submission fields')
    if candidate is None:
        return dict(status='rejected', reason='Unknown candidate')
    evaluation = candidate['evaluation']
    parameters = candidate['parameters']
    errors = []
    if 'event_complete_by_s' in arguments:
        event_complete = arguments['event_complete_by_s']
        window_end = max(float(k['time_s']) for k in parameters['knots'])
        if (type(event_complete) not in (int, float) or not np.isfinite(event_complete) or
                not parameters['normal_before_until_s'] < event_complete <= window_end+1e-8):
            errors.append(dict(field='event_complete_by_s', actual=event_complete,
                requirement=f'normal_before_until_s < event_complete_by_s <= {window_end}'))
    if evaluation['status'] != 'passed':
        errors.append(dict(field='candidate_id', reason='Candidate did not pass evaluation'))
    if arguments['candidate_id'] != evaluation.get('candidate_id'):
        errors.append(dict(field='candidate_id', actual=arguments['candidate_id'], expected=evaluation.get('candidate_id')))
    if arguments['image_marker'] != image_marker:
        errors.append(dict(field='image_marker', actual=arguments['image_marker'], expected=image_marker))
    if intent_policy == 'exact' and arguments['intended_event'] != evaluation['intent']:
        errors.append(dict(field='intended_event', reason='Submission intent differs from evaluated intent'))
    if not isinstance(arguments['remaining_uncertainty'], str) or not arguments['remaining_uncertainty'].strip():
        errors.append(dict(field='remaining_uncertainty', reason='Require nonempty uncertainty text'))
    for field in NORMAL_FIELDS:
        actual = arguments[field]
        expected = parameters[field]
        if type(actual) not in (int, float) or not np.isfinite(actual) or abs(actual-expected) > 1e-8:
            errors.append(dict(field=field, actual=actual, expected=expected))
    measured = arguments['measured_risk']
    if not isinstance(measured, dict) or set(measured) != set(RISK_EVIDENCE_FIELDS):
        errors.append(dict(field='measured_risk', reason='Require exactly the six returned risk values and times'))
    else:
        for field in RISK_EVIDENCE_FIELDS:
            actual, expected = measured[field], evaluation['risk_metrics'][field]
            if expected is None:
                if actual is not None:
                    errors.append(dict(field=field, actual=actual, expected=None))
            elif (type(actual) not in (int, float) or not np.isfinite(actual) or
                  abs(actual-expected) > tolerance):
                errors.append(dict(field=field, actual=actual, expected=expected, tolerance=tolerance))
    if errors:
        return dict(status='rejected', reason='Structured submission does not match the evaluated candidate', violations=errors)
    result = dict(status='accepted', candidate_id=arguments['candidate_id'],
                  canonical_intent=evaluation['intent'],
                  validated_assessment=validated_assessment(evaluation))
    if resolved is not None:
        result['resolved_submission'] = resolved
    return result


SOLVE_QUERY_FIELDS = ('time_s', 'desired_relative_forward_m', 'desired_relative_left_m')
MAX_SOLVE_QUERIES = 24
SOLVE_NOTE = ('A knot at a query time with these corrections places the target center at the desired ego-relative pose '
    'before the follow-tail transform; the interpolation passes through knot corrections exactly. When '
    'follow_ego_path_after_c2 applies, the C2 transition can adjust samples in its pre-boundary window, so the '
    'post-transform target at that timestamp may differ; evaluate_trajectory is the final check. Corrections are '
    'solved per timestamp: the recorded relative geometry drifts, so a correction solved for one time does not '
    'hold that pose at another time. The tail needs one constant correction: solve at a tail timestamp and reuse '
    'the same values at every tail knot. Neighboring knots still shape speed and acceleration; evaluate_trajectory '
    'remains the acceptance check.')


def _solve_displacement_query(query, times, baseline, ego, ego_heading):
    if not isinstance(query, dict) or set(query) != set(SOLVE_QUERY_FIELDS):
        return dict(status='invalid', error='Each query needs exactly: '+','.join(SOLVE_QUERY_FIELDS))
    for name in SOLVE_QUERY_FIELDS:
        value = query[name]
        if type(value) not in (int, float) or not np.isfinite(value):
            return dict(status='invalid', error=f'Require finite numeric {name}')
    time_s = float(query['time_s'])
    if not times[0] <= time_s <= times[-1]:
        return dict(status='invalid', error=f'time_s must lie within the sample window {times[0]}..{times[-1]}')
    ego_xy = np.array([np.interp(time_s, times, ego[:, i]) for i in [0, 1]])
    heading = float(np.interp(time_s, times, np.unwrap(ego_heading)))
    recorded = np.array([np.interp(time_s, times, baseline[:, i]) for i in [0, 1]])
    c, s = np.cos(heading), np.sin(heading)
    desired = ego_xy+np.array([[c, -s], [s, c]])@np.array([query['desired_relative_forward_m'],
                                                           query['desired_relative_left_m']])
    correction = desired-recorded
    delta = recorded-ego_xy
    return dict(status='ok', time_s=time_s,
        suggested_delta_forward_m=float(correction[0]), suggested_delta_left_m=float(correction[1]),
        baseline_relative_forward_m=float(c*delta[0]+s*delta[1]),
        baseline_relative_left_m=float(-s*delta[0]+c*delta[1]))


def solve_displacement_knot(arguments, times, baseline, ego, ego_heading):
    """Fixed-frame knot corrections placing the target at desired ego-relative poses pre-transform.

    Read-only helper: at each queried time it interpolates the recorded ego pose and
    the unedited target position, converts the desired ego-relative (forward, left)
    pose into the fixed frame, and subtracts the recorded target position. The
    follow-tail C2 transform can subsequently move samples in its pre-boundary
    window; evaluate_trajectory is the final check. Accepts a
    single query ({time_s, desired_relative_forward_m, desired_relative_left_m}) or a
    batch ({queries: [...]}) of up to MAX_SOLVE_QUERIES items solved in one call.
    """
    if not isinstance(arguments, dict):
        return dict(status='invalid', error='Require a query object or a queries list')
    if set(arguments) == {'queries'}:
        queries = arguments['queries']
        if not isinstance(queries, list) or not 1 <= len(queries) <= MAX_SOLVE_QUERIES:
            return dict(status='invalid', error=f'queries must be a list of 1..{MAX_SOLVE_QUERIES} knot queries')
        results = [_solve_displacement_query(q, times, baseline, ego, ego_heading) for q in queries]
        failed = [i for i, r in enumerate(results) if r['status'] != 'ok']
        batch = dict(status='ok' if not failed else 'invalid', results=results,
            failed_query_indices=failed, note=SOLVE_NOTE)
        if failed:
            batch['error'] = f'{len(failed)} of {len(results)} queries invalid; see failed_query_indices'
        return batch
    if set(arguments) == set(SOLVE_QUERY_FIELDS):
        result = _solve_displacement_query(arguments, times, baseline, ego, ego_heading)
        if result['status'] == 'ok':
            result['note'] = SOLVE_NOTE
        return result
    return dict(status='invalid',
        error='Provide either time_s/desired_relative_forward_m/desired_relative_left_m or a queries list')


def evaluate_displacements(arguments, times, baseline, ego, ego_heading, initial_velocity,
                           limits, target_length, target_width, ego_length=4.5, ego_width=2.,
                           min_normal_interval_s=None, normal_interval_policy='return_to_recorded',
                           min_normal_prefix_s=None, relative_baseline=None, other_actor_context=None,
                           direction_consistency=None, dwell_metric=None):
    if normal_interval_policy not in NORMAL_POLICIES:
        return dict(status='invalid', error='Unknown normal interval policy'), None
    if direction_consistency is not None and direction_consistency != DIRECTION_CONSISTENCY_VERSION:
        return dict(status='invalid', error='Unknown direction consistency version'), None
    if dwell_metric is not None:
        if (not isinstance(dwell_metric, dict)
                or any(type(dwell_metric.get(k)) not in (int, float) or not np.isfinite(dwell_metric.get(k))
                       for k in ('window_start_s', 'window_end_s', 'threshold_mps'))
                or not dwell_metric['window_start_s'] < dwell_metric['window_end_s']
                or dwell_metric['threshold_mps'] <= 0):
            return dict(status='invalid', error='dwell_metric requires finite window_start_s < window_end_s and a positive threshold_mps'), None
    if not isinstance(arguments, dict):
        return dict(status='invalid', error='Provide intent text and knots array'), None
    fields = set(arguments)
    v2 = fields == {'knots', 'intent', *NORMAL_FIELDS}
    if ((fields != {'knots', 'intent'} and not v2) or
            not isinstance(arguments.get('intent'), str) or not arguments['intent'].strip()):
        return dict(status='invalid', error='Provide intent, knots, and the v2 normal interval fields when applicable'), None
    if v2 and (any(type(arguments[f]) not in (int, float) or not np.isfinite(arguments[f]) for f in NORMAL_FIELDS) or
               min_normal_interval_s is None or min_normal_interval_s <= 0 or
               (min_normal_prefix_s is not None and (type(min_normal_prefix_s) not in (int, float) or
                not np.isfinite(min_normal_prefix_s) or min_normal_prefix_s < 0))):
        return dict(status='invalid', error='V2 requires finite normal boundaries and a positive configured minimum duration'), None
    try:
        offsets, spline = interpolate_offsets(arguments['knots'], times)
    except ValueError as error:
        return dict(status='invalid', error=str(error)), None
    xy = baseline + offsets
    follow_tail = None
    if v2 and normal_interval_policy in FOLLOW_POLICIES:
        if normal_interval_policy == 'follow_ego_path_after_c2':
            xy, follow_tail = apply_ego_path_follow_c2(
                xy, times, ego, ego_heading, float(arguments['normal_after_from_s']),
                before=float(arguments['normal_before_until_s']))
        else:
            xy, follow_tail = apply_ego_path_follow(
                xy, times, ego, ego_heading, float(arguments['normal_after_from_s']))
    direction_screen = None
    if direction_consistency is not None:
        direction_screen = direction_consistency_screen(times, xy, baseline)
    screen_limits = dict(limits)
    if v2:
        screen_limits.pop('max_endpoint_error_m', None)
    screen = screen_reference(np.column_stack([times, xy]), None if v2 else xy[-1], initial_velocity, screen_limits)
    relative_screen = None
    if relative_baseline is not None:
        relative_screen = screen_reference_relative(np.column_stack([times, xy]),
            np.column_stack([times, relative_baseline]), initial_velocity, screen_limits)
    normal = check_normal_intervals(arguments, times, spline, min_normal_interval_s,
        normal_interval_policy, minimum_prefix_duration=min_normal_prefix_s) if v2 else None
    if v2:
        add_displacement_violation_context(screen['violations'], spline, normal)
    delta = xy-ego
    cs, sn = np.cos(ego_heading), np.sin(ego_heading)
    relative = np.column_stack([cs*delta[:, 0]+sn*delta[:, 1], -sn*delta[:, 0]+cs*delta[:, 1]])
    target_heading = continuous_headings(xy, float(np.arctan2(*initial_velocity[::-1])))
    angle = target_heading-ego_heading
    half_x = (target_length*np.abs(np.cos(angle))+target_width*np.abs(np.sin(angle)))/2+ego_length/2
    half_y = (target_length*np.abs(np.sin(angle))+target_width*np.abs(np.cos(angle)))/2+ego_width/2
    signed_gap = np.abs(relative[:, 0])-half_x
    lateral_overlap = np.abs(relative[:, 1]) < half_y
    velocity = np.gradient(xy, times, axis=0)
    ego_velocity = np.gradient(ego, times, axis=0)
    correction_velocity = spline.derivative()(times)
    dv = velocity-ego_velocity
    closing = -(cs*dv[:, 0]+sn*dv[:, 1])
    ttc_valid = lateral_overlap & (relative[:, 0] > 0) & (signed_gap > 0) & (closing > .1)
    ttc = np.full(len(times), np.inf)
    ttc[ttc_valid] = signed_gap[ttc_valid]/closing[ttc_valid]
    nearest = int(np.linalg.norm(relative, axis=1).argmin())
    overlap_indices = np.flatnonzero(lateral_overlap)
    gap_index = int(overlap_indices[np.argmin(signed_gap[overlap_indices])]) if len(overlap_indices) else None
    ttc_index = int(ttc.argmin()) if np.isfinite(ttc).any() else None
    closing_index = int(np.flatnonzero(ttc_valid)[np.argmax(closing[ttc_valid])]) if ttc_valid.any() else None
    key_indices = set(range(0, len(times), 10)) | {len(times)-1, nearest}
    key_indices.update(i for i in [gap_index, ttc_index, closing_index] if i is not None)
    samples = []
    for i in sorted(key_indices):
        relation = ('ego_approaching_target' if closing[i] > .1 else
                    'target_pulling_away' if closing[i] < -.1 else 'approximately_matched_forward_speed')
        samples.append(dict(time_s=float(times[i]), longitudinal_center_gap_m=float(relative[i, 0]),
            lateral_center_offset_m=float(relative[i, 1]), speed_mps=float(np.linalg.norm(velocity[i])),
            ego_speed_mps=float(np.linalg.norm(ego_velocity[i])),
            correction_velocity_forward_mps=float(correction_velocity[i, 0]),
            correction_velocity_left_mps=float(correction_velocity[i, 1]),
            forward_closing_speed_mps=float(closing[i]), motion_relation=relation,
            estimated_longitudinal_body_gap_m=float(signed_gap[i]),
            constant_velocity_ttc_s=float(ttc[i]) if np.isfinite(ttc[i]) else None,
            projected_lateral_overlap=bool(lateral_overlap[i])))
    screen_for_status = relative_screen if relative_screen is not None else screen
    status = 'passed' if screen_for_status['status'] == 'passed' and (normal is None or normal['status'] == 'passed') else 'failed'
    final_offsets = xy-baseline
    encounter_index = gap_index if gap_index is not None else nearest
    overlap_blocked = bool(np.any(lateral_overlap & (signed_gap < 0)))
    contact = rectangle_overlap(relative, angle, target_length, target_width, ego_length, ego_width)
    contact_ids = np.flatnonzero(contact)
    contact_groups = np.split(contact_ids, np.flatnonzero(np.diff(contact_ids)>1)+1) if len(contact_ids) else []
    closure = None
    if gap_index is not None and not len(contact_ids) and signed_gap[gap_index] > 0:
        t1 = float(times[gap_index])
        gap_m = float(signed_gap[gap_index])
        t0 = float(arguments['normal_before_until_s']) if v2 else float(times[0])
        window = t1-t0
        in_window = (times >= t0) & (times <= t1)
        recorded_accel = np.gradient(np.gradient(baseline, times, axis=0), times, axis=0)
        recorded_peak = float(np.linalg.norm(recorded_accel[in_window], axis=1).max()) if in_window.any() else None
        estimate = None
        if window > 0:
            estimate = dict(
                assumed_profile='Cubic smoothstep forward correction with zero correction velocity at both window ends',
                estimated_peak_correction_speed_mps=float(1.5*gap_m/window),
                estimated_peak_correction_acceleration_mps2=float(6*gap_m/window**2),
                recorded_max_acceleration_in_window_mps2=recorded_peak,
                configured_max_acceleration_mps2=float(limits['max_acceleration_mps2']))
        closure = dict(reference_time_s=t1,
            remaining_forward_body_gap_m=gap_m,
            additional_forward_correction_m=-gap_m,
            edit_window_start_s=t0,
            edit_window_duration_s=float(window) if window > 0 else None,
            boundary_conditions=('Correction offset and correction velocity are zero at the edit window start; '
                'if the closure reaches into the constant tail, correction velocity must also return to zero '
                'at the tail boundary. A large constant offset by itself has zero correction velocity.'),
            smoothstep_estimate=estimate,
            scope=('The gap, additional correction and window are measured facts about this candidate; the '
                   'additional correction is relative to this candidate\'s correction at the reference time. '
                   'The smoothstep figures are an explicit-assumption 1-D estimate superimposed on recorded '
                   'motion, excluding lateral coupling and spline shape — not a feasibility certificate; any '
                   'concrete candidate must still pass the numerical screen. A null window means the closest '
                   'overlapped instant lies at or before the edit window start; closing there requires an '
                   'earlier edit start. An earlier edit start lengthens the window and reduces the required '
                   'correction acceleration quadratically.'))
    phase_attribution = None
    if v2:
        before, after = float(arguments['normal_before_until_s']), float(arguments['normal_after_from_s'])
        timed_metrics = dict(minimum_center_distance=float(times[nearest]),
            minimum_projected_body_gap=float(times[gap_index]) if gap_index is not None else None,
            minimum_constant_velocity_ttc=float(times[ttc_index]) if ttc_index is not None else None,
            maximum_forward_closing=float(times[closing_index]) if closing_index is not None else None)
        phase_attribution = dict(interaction_phase_s=[before, after],
            metric_phases={name: (phase_label(value, before, after) if value is not None else None)
                           for name, value in timed_metrics.items()},
            scope='Phase labels follow the declared normal boundaries; a closest, braking or closing '
                  'extremum stamped in the settled tail or unedited prefix is not evidence about the '
                  'interaction phase.')
    result = dict(status=status, numerical_screen=screen, normal_screen=normal, intent=arguments['intent'],
        planar_contact_proxy=dict(sampled_overlap=bool(contact.any()),
            first_overlap_time_s=float(times[contact_ids[0]]) if len(contact_ids) else None,
            intervals=[dict(start_s=float(times[g[0]]), end_s=float(times[g[-1]])) for g in contact_groups],
            scope='Separating-axis rectangle test on 10Hz reference samples with assumed dimensions; '
                  'not continuous collision detection, CARLA contact, or mesh geometry. After first '
                  'overlap, deeper reference penetration is geometric extrapolation, not physical motion; '
                  'executed contact is governed by CARLA physics from first contact onward.'),
        entry_dynamics=(entry_dynamics(times, relative, closing, signed_gap,
            float(arguments['normal_before_until_s']), float(arguments['normal_after_from_s']), ego_width/2) if v2 else None),
        risk_metrics=dict(minimum_center_distance_m=float(np.linalg.norm(relative[nearest])),
            closest_time_s=float(times[nearest]),
            minimum_projected_body_gap_while_laterally_overlapping_m=float(signed_gap[gap_index]) if gap_index is not None else None,
            projected_body_gap_time_s=float(times[gap_index]) if gap_index is not None else None,
            minimum_constant_velocity_ttc_s=float(ttc[ttc_index]) if ttc_index is not None else None,
            minimum_ttc_time_s=float(times[ttc_index]) if ttc_index is not None else None,
            maximum_forward_closing_speed_mps=float(closing[closing_index]) if closing_index is not None else None,
            maximum_forward_closing_time_s=float(times[closing_index]) if closing_index is not None else None,
            sampled_longitudinal_and_lateral_projection_overlap=overlap_blocked),
        closest_encounter=dict(
            basis=('minimum_projected_body_gap_while_laterally_overlapping' if gap_index is not None
                   else 'minimum_center_distance_without_lateral_overlap'),
            time_s=float(times[encounter_index]),
            center_distance_m=float(np.linalg.norm(relative[encounter_index])),
            projected_longitudinal_body_gap_m=float(signed_gap[encounter_index]),
            forward_closing_speed_mps=float(closing[encounter_index]),
            constant_velocity_ttc_s=float(ttc[encounter_index]) if np.isfinite(ttc[encounter_index]) else None,
            projected_lateral_overlap=bool(lateral_overlap[encounter_index])),
        longitudinal_closure_evidence=closure,
        limit_utilization={k: round(screen['metrics'][k]/screen['limits'][k], 4)
            for k in screen['limits'] if k in screen['metrics']},
        limit_utilization_scope='Measured value divided by its screening limit; the remaining headroom '
            'is adjustment room, not a danger rating.',
        projection_overlap_guidance=('Longitudinal and lateral projection overlap at some sampled instant; '
            'this alone does not prove collision — interpret it together with the trajectory geometry and '
            'the later CARLA checks.' if overlap_blocked else None),
        phase_attribution=phase_attribution,
        sampled_states=samples, sampled_risk_intervals=sampled_risk_intervals(times, ttc_valid, ttc, closing),
        risk_interval_scope='Contiguous 10Hz samples with lateral projection overlap, positive longitudinal gap and closing speed >0.1m/s; not continuous collision certification',
        max_displacement_from_recorded_m=float(np.linalg.norm(final_offsets, axis=1).max()),
        endpoint_displacement_from_recorded_m=float(np.linalg.norm(final_offsets[-1])),
        follow_ego_path_tail=follow_tail,
        event_success=None, event_assessment='Risk evidence only; no universal event threshold or 6-12m gap gate',
        geometry_assumptions=dict(target_length_m=target_length, target_width_m=target_width,
            ego_assumed_length_m=ego_length, ego_assumed_width_m=ego_width,
            limitation='Ego-axis projected rectangular extents, not exact polygon clearance or CARLA collision boxes; projection overlap alone does not prove collision'),
        evaluation_scope='Recorded-ego reference only; sampled at10Hz; no road-boundary or other-actor validation; endpoint is model-selected, not independent goal')
    if relative_screen is not None:
        result['relative_screen'] = relative_screen
    if other_actor_context is not None:
        from other_actor_checks import screen_other_actors
        other=screen_other_actors(times,xy,target_heading,target_length,target_width,
            other_actor_context,result['planar_contact_proxy']['first_overlap_time_s'])
        result['other_actor_screen']=other
        if other['status']!='passed':result['status']='failed'
        result['evaluation_scope']='Recorded-ego reference and observed other-vehicle rectangles checked at10Hz; '
        result['evaluation_scope']+='other-vehicle gate ends at first ego contact; missing tracks, roads and multi-body physics remain unvalidated'
    if direction_screen is not None:
        result['direction_screen']=direction_screen
        if direction_screen['status']=='failed':
            result['status']='failed'
        result['evaluation_scope']=result.get('evaluation_scope','')+'; direction consistency '+direction_screen['version']+' active'
    if dwell_metric is not None:
        dt = np.diff(times)
        seg_speed = np.linalg.norm(np.diff(xy, axis=0), axis=1)/dt
        seg_t = times[:-1]
        in_window = (seg_t >= dwell_metric['window_start_s']) & (seg_t < dwell_metric['window_end_s'])
        low = in_window & (seg_speed < dwell_metric['threshold_mps'])
        result['dwell_metric'] = dict(
            window_start_s=float(dwell_metric['window_start_s']),
            window_end_s=float(dwell_metric['window_end_s']),
            threshold_mps=float(dwell_metric['threshold_mps']),
            dwell_s=float(dt[low].sum()), low_speed_segments=int(low.sum()),
            window_segments=int(in_window.sum()),
            scope='Report-only task metric: cumulative segment time below the threshold within the window; not an acceptance gate')
    return result, xy
