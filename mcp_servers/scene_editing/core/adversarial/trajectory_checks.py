"""Numerical reference screening and forward-only waypoint selection.

These screens detect gross defects; they do not certify full vehicle dynamics.
"""
import acceptance_policy as policy
import numpy as np


LIMITS = dict(max_speed_mps=policy.ego_limits()['max_speed_mps'], max_acceleration_mps2=policy.ego_limits()['max_acceleration_mps2'],
              max_lateral_acceleration_mps2=policy.ego_limits()['max_lateral_acceleration_mps2'], max_yaw_rate_radps=policy.ATTACK_MAX_YAW_RATE_RADPS,
              max_initial_velocity_error_mps=1.0, max_endpoint_error_m=0.25)


def _series(values):
    """Shared per-sample metric series; the only sampling/mask definition."""
    t, xy = values[:, 0], values[:, 1:]
    dt = np.diff(t)
    velocity = np.diff(xy, axis=0) / dt[:, None]
    speed = np.linalg.norm(velocity, axis=1)
    acceleration = np.diff(velocity, axis=0) / ((dt[:-1]+dt[1:])/2)[:, None]
    middle_velocity = (velocity[:-1]+velocity[1:])/2
    moving = np.linalg.norm(middle_velocity, axis=1) > 1.0
    lateral = np.abs(middle_velocity[:, 0]*acceleration[:, 1]-middle_velocity[:, 1]*acceleration[:, 0]) / np.maximum(np.linalg.norm(middle_velocity, axis=1), 1e-9)
    angles = np.arctan2(velocity[:, 1], velocity[:, 0])
    delta = np.arctan2(np.sin(np.diff(angles)), np.cos(np.diff(angles)))
    yaw = np.abs(delta/((dt[:-1]+dt[1:])/2))
    turning = (speed[:-1] > 1.0) & (speed[1:] > 1.0)
    return dict(t=t, xy=xy, velocity=velocity, speed=speed, acceleration=acceleration,
        middle_velocity=middle_velocity, moving=moving, lateral=lateral, yaw=yaw, turning=turning,
        mid_t=t[1:-1], dt=dt)


def screen_reference(values, goal, initial_velocity, limits):
    s = _series(values)
    t, xy = s['t'], s['xy']
    dt, velocity, speed = s['dt'], s['velocity'], s['speed']
    acceleration, moving, lateral, yaw, turning = (s['acceleration'], s['moving'], s['lateral'], s['yaw'], s['turning'])
    metrics = dict(max_speed_mps=float(speed.max()),
        max_acceleration_mps2=float(np.linalg.norm(acceleration,axis=1).max()) if len(acceleration) else 0.0,
        max_lateral_acceleration_mps2=float(lateral[moving].max()) if moving.any() else 0.0,
        max_yaw_rate_radps=float(yaw[turning].max()) if turning.any() else 0.0,
        max_initial_velocity_error_mps=float(np.linalg.norm(velocity[0]-initial_velocity)))
    if goal is not None:
        metrics['max_endpoint_error_m']=float(np.linalg.norm(xy[-1]-goal))
    indices = dict(max_speed_mps=int(speed.argmax()),
        max_acceleration_mps2=int(np.linalg.norm(acceleration,axis=1).argmax()) if len(acceleration) else None,
        max_lateral_acceleration_mps2=int(np.flatnonzero(moving)[np.argmax(lateral[moving])]) if moving.any() else None,
        max_yaw_rate_radps=int(np.flatnonzero(turning)[np.argmax(yaw[turning])]) if turning.any() else None,
        max_initial_velocity_error_mps=0, max_endpoint_error_m=None)
    hints = dict(max_speed_mps='Reduce the relevant correction slope and re-evaluate the full horizon.',
        max_acceleration_mps2='Reduce the local velocity change around this time; reshape nearby correction knots and re-evaluate.',
        max_lateral_acceleration_mps2='Reduce lateral curvature around this time; reshape lateral correction slopes and re-evaluate.',
        max_yaw_rate_radps='Reduce the local heading-change rate and re-evaluate.',
        max_initial_velocity_error_mps='Preserve the recorded initial velocity by reshaping the first nonzero correction segment.',
        max_endpoint_error_m='Match the independent endpoint goal; this is separate from normal-interval preservation.')

    def violation(metric):
        i = indices[metric]
        item = dict(metric=metric, actual=metrics[metric], limit=limits[metric],
            excess=float(metrics[metric]-limits[metric]), repair_hint=hints[metric])
        if metric == 'max_endpoint_error_m':
            item.update(time_s=float(t[-1]), local_motion=dict(
                actual_endpoint_xy_m=xy[-1].tolist(), independent_goal_xy_m=np.asarray(goal).tolist(),
                error_vector_m=(xy[-1]-goal).tolist()))
        elif metric in {'max_acceleration_mps2', 'max_lateral_acceleration_mps2', 'max_yaw_rate_radps'}:
            item.update(time_s=float(t[i+1]), local_motion=dict(
                interval_start_time_s=float(t[i]), interval_end_time_s=float(t[i+2]),
                velocity_before_mps=velocity[i].tolist(), velocity_after_mps=velocity[i+1].tolist(),
                speed_before_mps=float(speed[i]), speed_after_mps=float(speed[i+1]),
                acceleration_vector_mps2=acceleration[i].tolist()))
        else:
            item.update(time_s=float((t[i]+t[i+1])/2), local_motion=dict(
                segment_start_time_s=float(t[i]), segment_end_time_s=float(t[i+1]),
                segment_start_xy_m=xy[i].tolist(), segment_end_xy_m=xy[i+1].tolist(),
                segment_velocity_mps=velocity[i].tolist(), segment_speed_mps=float(speed[i])))
            if metric == 'max_initial_velocity_error_mps':
                item['local_motion']['expected_initial_velocity_mps'] = np.asarray(initial_velocity).tolist()
        return item

    violations = [violation(k) for k in limits if k in metrics and metrics[k]>limits[k]]
    return dict(status='failed' if violations else 'passed', metrics=metrics, limits=limits,
                violations=violations, full_dynamic_feasibility_certified=False)


def violation_intervals(values, limits):
    """Every over-limit interval per metric, same masks and sampling as screen_reference.

    screen_reference reports only the peak sample per metric; planning needs every
    recorded-baseline interval so it can leave them editable. Returns
    {metric: [dict(start_s, end_s, peak, peak_time_s)]}, interval metrics only
    (speed/accel/lateral/yaw); empty dict when nothing exceeds.
    """
    s = _series(values)
    series = {
        'max_speed_mps': (s['speed'], np.ones(len(s['speed']), bool), (s['t'][:-1]+s['t'][1:])/2),
        'max_acceleration_mps2': (np.linalg.norm(s['acceleration'], axis=1), np.ones(len(s['t'])-2, bool), s['mid_t']),
        'max_lateral_acceleration_mps2': (s['lateral'], s['moving'], s['mid_t']),
        'max_yaw_rate_radps': (s['yaw'], s['turning'], s['mid_t']),
    }
    result = {}
    for metric, (values_, mask, times_) in series.items():
        if metric not in limits:
            continue
        over = (values_ > limits[metric]) & mask
        groups = np.split(np.flatnonzero(over), np.flatnonzero(np.diff(np.flatnonzero(over)) > 1)+1) if over.any() else []
        intervals = []
        for g in groups:
            peak = int(g[values_[g].argmax()])
            intervals.append(dict(start_s=float(times_[g[0]]), end_s=float(times_[g[-1]]),
                peak=round(float(values_[peak]), 4), peak_time_s=float(times_[peak]), samples=int(len(g))))
        if intervals:
            result[metric] = intervals
    return result


def relative_allowance(baseline_values, limits):
    """Per-sample allowance max(fixed limit, recorded baseline value at that sample).

    allowance(t) = max(fixed, recorded(t)) separately for total acceleration,
    lateral acceleration and yaw rate, under the official masks: a baseline sample
    that is masked out (low speed) grants no allowance. Inherits recorded anomalies
    without forcing repair; edit-added exceedances above the allowance still fail.
    """
    s = _series(baseline_values)
    accel = np.linalg.norm(s['acceleration'], axis=1)
    return dict(
        max_acceleration_mps2=np.maximum(limits['max_acceleration_mps2'], accel),
        max_lateral_acceleration_mps2=np.maximum(limits['max_lateral_acceleration_mps2'],
            np.where(s['moving'], s['lateral'], 0.0)),
        max_yaw_rate_radps=np.maximum(limits['max_yaw_rate_radps'],
            np.where(s['turning'], s['yaw'], 0.0)),
    )


def screen_reference_relative(values, baseline_values, initial_velocity, limits):
    """Per-time relative screen: no NEW exceedances beyond the recorded baseline.

    Acceptance verdict reads 'no new over-limit behaviour relative to the recording';
    absolute speed and initial-velocity gates still apply; physical feasibility is
    assessed separately. Absolute metrics are always reported unchanged.
    """
    absolute = screen_reference(values, None, initial_velocity, limits)
    allowance = relative_allowance(baseline_values, limits)
    s = _series(values)
    violations = []
    for metric, candidate, mask in (
            ('max_acceleration_mps2', np.linalg.norm(s['acceleration'], axis=1), np.ones(len(s['t'])-2, bool)),
            ('max_lateral_acceleration_mps2', s['lateral'], s['moving']),
            ('max_yaw_rate_radps', s['yaw'], s['turning'])):
        over = (candidate > allowance[metric] + 1e-8) & mask
        for i in np.flatnonzero(over):
            violations.append(dict(metric=metric, time_s=float(s['mid_t'][i]), actual=float(candidate[i]),
                allowance=float(allowance[metric][i]), excess=float(candidate[i]-allowance[metric][i]),
                repair_hint='New exceedance beyond the recorded baseline at this time; reshape the edit, not the record.'))
    for metric in ('max_speed_mps', 'max_initial_velocity_error_mps'):
        if absolute['metrics'][metric] > limits[metric]:
            violations.append(dict(metric=metric, actual=absolute['metrics'][metric], limit=limits[metric],
                excess=float(absolute['metrics'][metric]-limits[metric]),
                repair_hint='Absolute gate; relative allowance does not apply.'))
    return dict(status='failed' if violations else 'passed', violations=violations,
        absolute_metrics=absolute['metrics'], absolute_violations=absolute['violations'],
        acceptance='no_new_exceedance_vs_recorded_baseline; physical feasibility assessed separately')


def phase_label(time_s, before, after):
    """Attribute a timestamp to the declared normal/interaction phases."""
    if time_s <= before+1e-8:
        return 'unedited_prefix'
    if time_s >= after-1e-8:
        return 'settled_tail'
    return 'interaction'


def continuous_headings(xy, fallback):
    """Retain the last moving heading when consecutive reference points stop."""
    delta = np.diff(xy, axis=0)
    result = []; heading = float(fallback)
    for vector in delta:
        if np.linalg.norm(vector)>1e-6:
            heading = float(np.arctan2(vector[1],vector[0]))
        result.append(heading)
    return np.unwrap(result+[heading])


DIRECTION_CONSISTENCY_VERSION = 'recorded_progression.v1'


def direction_consistency_screen(times, xy, baseline, min_direction_speed_mps=0.5,
                                 step_tolerance_m=1e-6):
    """Signed-progress screen along the recorded baseline's travel direction (v1).

    Stopping is allowed; backward motion newly introduced by an edit is rejected.
    The trusted direction is the recorded baseline segment direction, held through
    low-speed gaps — never the candidate's own tangent and never world-axis
    monotonicity. Regression intervals are always reported for disclosure, and the
    aggregate metrics are present whether the screen passes or fails. Reversing
    intents are out of scope for v1. A baseline that never reaches the direction
    speed is reported not_applicable and does not gate.
    """
    times = np.asarray(times, dtype=float)
    xy = np.asarray(xy, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    dt = np.diff(times)
    base_steps = np.diff(baseline, axis=0)
    base_speed = np.linalg.norm(base_steps, axis=1)/dt
    direction = np.zeros_like(base_steps)
    last = None
    for i, step in enumerate(base_steps):
        if base_speed[i] >= min_direction_speed_mps:
            last = step/np.linalg.norm(step)
        if last is not None:
            direction[i] = last
    scope = ('Signed per-step progress along the recorded baseline segment direction; '
             'stopping allowed; backward motion newly introduced by the edit rejected; '
             'reversing intents out of scope for v1')
    if last is None:
        return dict(status='not_applicable', version=DIRECTION_CONSISTENCY_VERSION,
            reason='Recorded baseline never reaches %.3g m/s; no trusted travel direction' % min_direction_speed_mps,
            regression_intervals=[], total_regression_m=0.0, max_reverse_speed_mps=0.0,
            min_direction_speed_mps=min_direction_speed_mps, step_tolerance_m=step_tolerance_m,
            direction_source='Recorded baseline segment direction, held through low-speed segments; candidate tangent never used',
            scope=scope)
    # Backfill the leading region (before the recording first reaches the direction
    # speed) with the first valid direction — the initial orientation before motion.
    # Without this, regressions in that region would multiply a zero vector and
    # escape the screen.
    first = int(np.flatnonzero(base_speed >= min_direction_speed_mps)[0])
    leading_inferred = first
    if first:
        direction[:first] = base_steps[first]/np.linalg.norm(base_steps[first])
    progress = np.sum(np.diff(xy, axis=0)*direction, axis=1)
    reverse_speed = -progress/dt
    bad = np.flatnonzero(progress < -step_tolerance_m)
    groups = np.split(bad, np.flatnonzero(np.diff(bad) > 1)+1) if len(bad) else []
    intervals = [dict(start_time_s=float(times[g[0]]), end_time_s=float(times[g[-1]+1]),
        duration_s=float(times[g[-1]+1]-times[g[0]]),
        cumulative_regression_m=float(-progress[g].sum()),
        max_reverse_speed_mps=float(reverse_speed[g].max()),
        min_step_progress_m=float(progress[g].min())) for g in groups]
    return dict(status='failed' if intervals else 'passed', version=DIRECTION_CONSISTENCY_VERSION,
        regression_intervals=intervals,
        total_regression_m=float(-progress[bad].sum()) if len(bad) else 0.0,
        max_reverse_speed_mps=float(reverse_speed[bad].max()) if len(bad) else 0.0,
        leading_direction_inferred_segments=leading_inferred,
        min_direction_speed_mps=min_direction_speed_mps, step_tolerance_m=step_tolerance_m,
        direction_source=('Recorded baseline segment direction, held through low-speed segments; '
            'leading segments before the recording first reaches %.3g m/s use the first valid '
            'direction (initial orientation); candidate tangent never used' % min_direction_speed_mps),
        scope=scope)


def forward_reference_index(times, xy, t, actual, heading, speed, preview_seconds, min_distance_m):
    """Select within a bounded future window; never steer toward a point behind."""
    indices=np.flatnonzero((times>=t) & (times<=min(times[-1],t+preview_seconds)))
    delta=xy[indices]-actual
    ahead=delta@np.array([np.cos(heading),np.sin(heading)])>0.05
    forward=indices[ahead]
    if not len(forward):
        return None
    distances=np.linalg.norm(xy[forward]-actual,axis=1)
    desired=max(min_distance_m,0.5*speed)
    distant=forward[distances>=desired]
    return int(distant[0] if len(distant) else forward[-1])
