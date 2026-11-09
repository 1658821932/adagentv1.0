"""Offline P0 acceptance for an ego reference trajectory.

This module checks a proposed reference against a fixed, recorded attack-vehicle
trajectory.  A passing result is not evidence that CARLA executed the reference
or that actual avoidance succeeded.
"""
from numbers import Real

import acceptance_policy as policy
import numpy as np

from trajectory_search_feedback import rectangle_overlap


_LIMIT_NAMES = (
    'max_speed_mps', 'max_acceleration_mps2', 'max_lateral_acceleration_mps2',
    'max_yaw_rate_radps', 'max_jerk_mps3', 'min_progress_m', 'min_clearance_m',
)
_MOTION_NAMES = _LIMIT_NAMES[:5]


def _array(values, name, ndim, length=None):
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a numeric array') from exc
    if result.ndim != ndim or (ndim == 2 and result.shape[1] != 2):
        shape = '(n, 2)' if ndim == 2 else '(n,)'
        raise ValueError(f'{name} must have shape {shape}')
    if length is not None and len(result) != length:
        raise ValueError(f'{name} must have the same number of samples as times_s')
    if not np.isfinite(result).all():
        raise ValueError(f'{name} must contain only finite values')
    return result


def _number(value, name, positive=False, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite numeric scalar, not bool')
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f'{name} must be finite')
    if positive and result <= 0:
        raise ValueError(f'{name} must be positive')
    if nonnegative and result < 0:
        raise ValueError(f'{name} must be nonnegative')
    return result


def _validate_limits(limits):
    if not isinstance(limits, dict):
        raise ValueError('limits must be a dict')
    unknown = set(limits)-set(_LIMIT_NAMES)
    if unknown:
        raise ValueError(f'Unsupported limits: {sorted(unknown)}')
    result = {key: _number(value, f'limits[{key}]', nonnegative=True)
              for key, value in limits.items()}
    result.setdefault('min_clearance_m', 0.0)
    return result


def _intervals(times, indices):
    indices = np.asarray(indices, dtype=int)
    if not len(indices):
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1)
    return [dict(start_s=float(times[group[0]]), end_s=float(times[group[-1]]),
                 start_index=int(group[0]), end_index=int(group[-1]))
            for group in groups]


def _corners(xy, heading, length, width):
    c, s = np.cos(heading), np.sin(heading)
    longitudinal = np.array([c, s])*length/2
    lateral = np.array([-s, c])*width/2
    return np.array([xy-longitudinal-lateral, xy+longitudinal-lateral,
                     xy+longitudinal+lateral, xy-longitudinal+lateral])


def _edge_distance(first, second):
    """Exact distance between disjoint convex rectangle boundaries."""
    best = np.inf
    for points, other in ((first, second), (second, first)):
        for i in range(4):
            start = other[i]
            edge = other[(i+1) % 4]-start
            fraction = np.clip(((points-start)@edge)/(edge@edge), 0.0, 1.0)
            closest = start+fraction[:, None]*edge
            best = min(best, float(np.linalg.norm(points-closest, axis=1).min()))
    return best


def _geometry(times, ego_xy, ego_heading, attack_xy, attack_heading,
              ego_length, ego_width, attack_length, attack_width):
    delta = attack_xy-ego_xy
    c, s = np.cos(ego_heading), np.sin(ego_heading)
    relative = np.column_stack((c*delta[:, 0]+s*delta[:, 1],
                                -s*delta[:, 0]+c*delta[:, 1]))
    overlap = np.asarray(rectangle_overlap(
        relative, attack_heading-ego_heading, attack_length, attack_width,
        ego_length, ego_width), dtype=bool)
    center_distance = np.linalg.norm(delta, axis=1)
    clearance = np.zeros(len(times), dtype=float)
    for i in np.flatnonzero(~overlap):
        first = _corners(ego_xy[i], ego_heading[i], ego_length, ego_width)
        second = _corners(attack_xy[i], attack_heading[i], attack_length, attack_width)
        clearance[i] = _edge_distance(first, second)
    nearest_clearance = int(clearance.argmin())
    nearest_center = int(center_distance.argmin())
    return dict(
        has_overlap=bool(overlap.any()),
        overlap_intervals=_intervals(times, np.flatnonzero(overlap)),
        minimum_clearance_m=float(clearance[nearest_clearance]),
        minimum_clearance_time_s=float(times[nearest_clearance]),
        minimum_clearance_index=nearest_clearance,
        minimum_center_distance_m=float(center_distance[nearest_center]),
        minimum_center_distance_time_s=float(times[nearest_center]),
        minimum_center_distance_index=nearest_center,
    ), clearance


def _motion(times, xy, heading, limits):
    dt = np.diff(times)
    velocity = np.diff(xy, axis=0)/dt[:, None]
    velocity_times = (times[:-1]+times[1:])/2
    speed = np.linalg.norm(velocity, axis=1)
    unwrapped_heading = np.unwrap(heading)
    yaw_rate = np.abs(np.diff(unwrapped_heading)/dt)

    if len(velocity) >= 2:
        acceleration = np.diff(velocity, axis=0)/np.diff(velocity_times)[:, None]
        acceleration_times = times[1:-1]
        middle_velocity = (velocity[:-1]+velocity[1:])/2
        middle_speed = np.linalg.norm(middle_velocity, axis=1)
        lateral = np.abs(middle_velocity[:, 0]*acceleration[:, 1]
                         -middle_velocity[:, 1]*acceleration[:, 0])
        lateral = lateral/np.maximum(middle_speed, 1e-12)
    else:
        acceleration = np.empty((0, 2))
        acceleration_times = np.empty(0)
        lateral = np.empty(0)
    if len(acceleration) >= 2:
        jerk = np.diff(acceleration, axis=0)/np.diff(acceleration_times)[:, None]
        jerk_times = acceleration_times[1:]
    else:
        jerk = np.empty((0, 2))
        jerk_times = np.empty(0)

    series = {
        'max_speed_mps': (speed, velocity_times, times[:-1], times[1:]),
        'max_acceleration_mps2': (np.linalg.norm(acceleration, axis=1),
            acceleration_times, times[:-2], times[2:]),
        'max_lateral_acceleration_mps2': (lateral,
            acceleration_times, times[:-2], times[2:]),
        'max_yaw_rate_radps': (yaw_rate, velocity_times, times[:-1], times[1:]),
        'max_jerk_mps3': (np.linalg.norm(jerk, axis=1),
            jerk_times, acceleration_times[:-1], acceleration_times[1:]),
    }
    peaks = {}
    peak_times = {}
    violation_intervals = {}
    violations = []
    for name, (values, sample_times, starts, ends) in series.items():
        if len(values):
            peak_index = int(values.argmax())
            peaks[name] = float(values[peak_index])
            peak_times[name] = float(sample_times[peak_index])
        else:
            peaks[name] = 0.0
            peak_times[name] = None
        if name not in limits:
            continue
        over = np.flatnonzero(values > limits[name])
        intervals = []
        if len(over):
            groups = np.split(over, np.flatnonzero(np.diff(over) > 1)+1)
            for group in groups:
                local_peak = int(group[values[group].argmax()])
                intervals.append(dict(
                    start_s=float(starts[group[0]]), end_s=float(ends[group[-1]]),
                    start_index=int(group[0]), end_index=int(group[-1]),
                    peak=float(values[local_peak]),
                    peak_time_s=float(sample_times[local_peak]),
                    limit=float(limits[name])))
            violation_intervals[name] = intervals
            violations.extend(dict(metric=name, **item) for item in intervals)
    return dict(status='failed' if violations else 'passed', peaks=peaks,
                peak_times_s=peak_times, violation_intervals=violation_intervals,
                violations=violations,
                method='Finite differences on the supplied nonuniform time grid; headings unwrapped before yaw differentiation')


def _progress_direction(ego_heading, baseline_xy, baseline_heading):
    if baseline_xy is not None and len(baseline_xy) >= 2:
        steps = np.diff(baseline_xy, axis=0)
        lengths = np.linalg.norm(steps, axis=1)
        moving = np.flatnonzero(lengths > 1e-9)
        if len(moving):
            step = steps[moving[0]]
            return step/lengths[moving[0]], 'first nonzero baseline displacement'
        heading = baseline_heading[0]
        return np.array([np.cos(heading), np.sin(heading)]), 'initial baseline heading (stationary baseline)'
    heading = ego_heading[0]
    return np.array([np.cos(heading), np.sin(heading)]), 'initial ego heading (no baseline)'


def evaluate_reference(times_s, ego_xy, ego_heading_rad, attack_xy, attack_heading_rad,
                       ego_length_m, ego_width_m, attack_length_m, attack_width_m,
                       limits, baseline_ego_xy=None, baseline_ego_heading_rad=None, task_policy=None):
    """Evaluate reference geometry, sampled dynamics, progress and baseline validity.

    Baseline overlap is diagnostic; both collision and near-miss scenarios are valid.  ``min_progress_m`` uses net progress
    along the original travel direction.  Omitted motion/progress limits disable
    only their corresponding gates.  Clearance defaults to zero, but any sampled
    oriented-rectangle overlap is always forbidden.
    """
    times = _array(times_s, 'times_s', 1)
    if not len(times):
        raise ValueError('times_s must contain at least one sample')
    if len(times) >= 2:
        dt = np.diff(times)
        if not np.isfinite(dt).all() or np.any(dt <= 0):
            raise ValueError('times_s must be strictly increasing with finite intervals')
    n = len(times)
    ego = _array(ego_xy, 'ego_xy', 2, n)
    ego_heading = _array(ego_heading_rad, 'ego_heading_rad', 1, n)
    attack = _array(attack_xy, 'attack_xy', 2, n)
    attack_heading = _array(attack_heading_rad, 'attack_heading_rad', 1, n)
    ego_length = _number(ego_length_m, 'ego_length_m', positive=True)
    ego_width = _number(ego_width_m, 'ego_width_m', positive=True)
    attack_length = _number(attack_length_m, 'attack_length_m', positive=True)
    attack_width = _number(attack_width_m, 'attack_width_m', positive=True)
    checked_limits = _validate_limits(limits)
    if (baseline_ego_xy is None) != (baseline_ego_heading_rad is None):
        raise ValueError('baseline_ego_xy and baseline_ego_heading_rad must be supplied together')
    baseline = baseline_heading = None
    if baseline_ego_xy is not None:
        baseline = _array(baseline_ego_xy, 'baseline_ego_xy', 2, n)
        baseline_heading = _array(baseline_ego_heading_rad, 'baseline_ego_heading_rad', 1, n)

    geometry, clearance = _geometry(times, ego, ego_heading, attack, attack_heading,
        ego_length, ego_width, attack_length, attack_width)
    clearance_limit = checked_limits['min_clearance_m']
    too_close = clearance < clearance_limit
    clearance_intervals = _intervals(times, np.flatnonzero(too_close))
    collision_failed = geometry['has_overlap'] or bool(too_close.any())
    collision_check = dict(status='failed' if collision_failed else 'passed',
        **geometry, min_clearance_limit_m=clearance_limit,
        clearance_violation_intervals=clearance_intervals,
        scope='All samples in the full window; touching counts as overlap; clearance is exact oriented-rectangle boundary distance')

    motion_check = _motion(times, ego, ego_heading, checked_limits)
    direction, direction_source = _progress_direction(ego_heading, baseline, baseline_heading)
    progress = float((ego[-1]-ego[0])@direction)
    required = checked_limits.get('min_progress_m', 0.)
    if 'min_progress_m' in checked_limits:
        progress_failed = progress < checked_limits['min_progress_m']
        progress_status = 'failed' if progress_failed else 'passed'
    else:
        progress_failed = False
        progress_status = 'not_checked'
        if task_policy is not None and baseline is not None and len(times)>1:
            effective = policy.resolve(task_policy)
            recorded_progress = float(np.linalg.norm(np.diff(baseline,axis=0),axis=1).sum())
            progress_ok, reason = policy.task_progress(times, ego, recorded_progress, effective)
            required = (effective['minimum_progress_fraction'] or 0.)*recorded_progress
            progress_failed = not progress_ok
            progress_status = 'failed' if progress_failed else 'passed'
    progress_check = dict(status=progress_status, progress_m=progress,
        minimum_progress_m=checked_limits.get('min_progress_m'),
        minimum_cumulative_traveled_m=required if task_policy and 'min_progress_m' not in checked_limits else None,
        policy=('explicit_net_progress_override' if 'min_progress_m' in checked_limits else 'task_contract' if task_policy else 'diagnostic_only'),
        cumulative_traveled_m=float(np.linalg.norm(np.diff(ego, axis=0),axis=1).sum()),
        direction_xy=direction.tolist(), direction_source=direction_source,
        definition='Net displacement from first to last ego sample projected onto the original travel direction')

    baseline_check = None
    baseline_has_overlap = None
    if baseline is not None:
        baseline_geometry, _ = _geometry(times, baseline, baseline_heading, attack,
            attack_heading, ego_length, ego_width, attack_length, attack_width)
        baseline_has_overlap = baseline_geometry['has_overlap']
        baseline_check = dict(status='overlap_observed' if baseline_has_overlap else 'no_overlap_observed',
            baseline_has_overlap=baseline_has_overlap,
            overlap_intervals=baseline_geometry['overlap_intervals'],
            minimum_center_distance_m=baseline_geometry['minimum_center_distance_m'],
            minimum_center_distance_time_s=baseline_geometry['minimum_center_distance_time_s'],
            scope='Baseline overlap diagnostic only; does not gate candidate acceptance')

    passed = not (collision_failed or motion_check['status'] == 'failed'
                  or progress_failed)
    violations = []
    if geometry['has_overlap']:
        violations.append(dict(metric='ego_attack_overlap',
            intervals=geometry['overlap_intervals']))
    if clearance_intervals:
        violations.append(dict(metric='min_clearance_m',
            limit=clearance_limit, intervals=clearance_intervals))
    violations.extend(motion_check['violations'])
    if progress_failed:
        violations.append(dict(metric='min_progress_m' if 'min_progress_m' in checked_limits else 'minimum_cumulative_traveled_m',
            actual=progress if 'min_progress_m' in checked_limits else float(np.linalg.norm(np.diff(ego,axis=0),axis=1).sum()),
            limit=checked_limits.get('min_progress_m', required)))

    return dict(
        acceptance_policy=policy.resolve(task_policy), effective_limits=checked_limits,
        status='reference_passed' if passed else 'reference_failed',
        reference_passed=passed,
        avoidance_status='undetermined',
        road_status='not_strictly_validated',
        carla_validated=False,
        validation_scope='P0 offline reference acceptance only; CARLA execution and actual avoidance have not been validated',
        limits=checked_limits,
        collision_check=collision_check,
        motion_check=motion_check,
        progress_check=progress_check,
        baseline_has_overlap=baseline_has_overlap,
        baseline_check=baseline_check,
        violations=violations,
    )
