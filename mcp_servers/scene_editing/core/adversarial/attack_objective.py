"""Structured, time-windowed actor objectives shared by search and acceptance."""
import acceptance_policy as policy
import numpy as np
import shapely

from run_ego_reference import EGO_FRONT, EGO_REAR, EGO_WIDTH


def validate_objective(arguments, times, prefix):
    required = {'parent_candidate_id', 'objective', 'max_candidates', 'reason'}
    if not required <= set(arguments) or set(arguments) - required - {'edit_window_s', 'preserve_outside_window'}:
        raise ValueError('Provide parent_candidate_id, objective, max_candidates and reason')
    if type(arguments['max_candidates']) is not int or not 1 <= arguments['max_candidates'] <= 3:
        raise ValueError('max_candidates must be 1..3')
    if not isinstance(arguments['reason'], str) or not arguments['reason'].strip():
        raise ValueError('A nonempty reason is required')
    objective = arguments['objective']
    if not isinstance(objective, dict) or 'interaction' not in objective or set(objective) - {'interaction', 'interaction_window_s', 'keyframes'}:
        raise ValueError('objective requires interaction; interaction_window_s and keyframes are optional')
    objective.setdefault('interaction_window_s', [float(np.nextafter(prefix, np.inf)), float(times[-1])])
    objective.setdefault('keyframes', [])
    if objective['interaction'] not in ('contact', 'near_miss', 'either', 'allowed'):
        raise ValueError('interaction must be contact, near_miss, either or allowed')

    # LLMs round native timestamps (e.g. 11.499809 -> 11.5). Normalize only
    # time endpoints within 1 ms; position/heading/speed limits remain exact.
    windows = [objective.get('interaction_window_s'), arguments.get('edit_window_s')]
    if isinstance(objective.get('keyframes'), list):
        windows += [frame.get('time_window_s') for frame in objective['keyframes'] if isinstance(frame, dict)]
    for window in windows:
        if isinstance(window, list) and len(window) == 2:
            for index in (0, 1):
                if type(window[index]) in (int, float) and abs(window[index]-times[-1]) <= .001:
                    window[index] = float(times[-1])

    def interval(value, name, minimum=None, maximum=None):
        if not isinstance(value, list) or len(value) != 2 or any(type(v) not in (float, int) for v in value) or not np.isfinite(value).all() or value[0] > value[1]:
            raise ValueError(name + ' must be an ordered finite interval')
        if minimum is not None and value[0] < minimum or maximum is not None and value[1] > maximum:
            raise ValueError(name + ' is outside its allowed range')

    interval(objective['interaction_window_s'], 'interaction_window_s', float(np.nextafter(prefix, np.inf)), float(times[-1]))
    keyframes = objective['keyframes']
    if not isinstance(keyframes, list) or not 0 <= len(keyframes) <= 6:
        raise ValueError('Provide 0..6 keyframes')
    for frame in keyframes:
        if not isinstance(frame, dict) or not {'time_window_s', 'hold_duration_s'} <= set(frame) or set(frame) - {'time_window_s', 'hold_duration_s', 'relative_position', 'heading_difference_rad', 'speed_mps'}:
            raise ValueError('Invalid keyframe fields')
        interval(frame['time_window_s'], 'keyframe time_window_s', prefix, float(times[-1]))
        hold = frame['hold_duration_s']
        if type(hold) not in (int, float) or not np.isfinite(hold) or hold < .1 or hold > frame['time_window_s'][1] - frame['time_window_s'][0] + 1e-9:
            raise ValueError('hold_duration_s must be >=0.1s and fit the keyframe window')
        if not set(frame) & {'relative_position', 'heading_difference_rad', 'speed_mps'}:
            raise ValueError('A keyframe needs a position, heading or speed constraint')
        if 'relative_position' in frame:
            position = frame['relative_position']
            if not isinstance(position, dict) or not position or set(position) - {'forward_m', 'left_m'}:
                raise ValueError('relative_position uses forward_m and/or left_m intervals')
            for name, value in position.items():
                interval(value, name)
        if 'heading_difference_rad' in frame:
            interval(frame['heading_difference_rad'], 'heading_difference_rad', -np.pi, np.pi)
        if 'speed_mps' in frame:
            interval(frame['speed_mps'], 'speed_mps', 0.)
    if 'edit_window_s' in arguments:
        interval(arguments['edit_window_s'], 'edit_window_s', prefix, float(times[-1]))
        if arguments['edit_window_s'][1] - arguments['edit_window_s'][0] < .5:
            raise ValueError('edit_window_s must be at least 0.5s')
    if 'preserve_outside_window' in arguments and type(arguments['preserve_outside_window']) is not bool:
        raise ValueError('preserve_outside_window must be boolean')
    if arguments.get('preserve_outside_window') and 'edit_window_s' not in arguments:
        raise ValueError('preserve_outside_window requires edit_window_s')


def bodies(xy, yaw, size):
    corners = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * np.asarray(size[:2]) / 2
    c, s = np.cos(yaw), np.sin(yaw)
    points = xy[:, None, :] + np.stack([c[:, None] * corners[:, 0] - s[:, None] * corners[:, 1],
                                       s[:, None] * corners[:, 0] + c[:, None] * corners[:, 1]], axis=-1)
    return shapely.polygons(points)


def interval_error(values, interval):
    return np.maximum(interval[0] - values, 0.) + np.maximum(values - interval[1], 0.)


def relative_state(times, xy, yaw, ego, ego_yaw):
    # Both positions below are BODY centers; source ego is rear-axle based.
    ego_center = ego + (EGO_FRONT-EGO_REAR)/2 * np.column_stack([np.cos(ego_yaw), np.sin(ego_yaw)])
    delta = xy - ego_center
    forward = delta[:, 0] * np.cos(ego_yaw) + delta[:, 1] * np.sin(ego_yaw)
    left = -delta[:, 0] * np.sin(ego_yaw) + delta[:, 1] * np.cos(ego_yaw)
    heading = np.arctan2(np.sin(yaw-ego_yaw), np.cos(yaw-ego_yaw))
    speed = np.linalg.norm(np.gradient(xy, times, axis=0), axis=1)
    return forward, left, heading, speed


def frame_errors(frame, state):
    forward, left, heading, speed = state
    errors = []
    for name, values in (('forward_m', forward), ('left_m', left)):
        if name in frame.get('relative_position', {}):
            errors.append(interval_error(values, frame['relative_position'][name]))
    if 'heading_difference_rad' in frame:
        errors.append(2 * interval_error(heading, frame['heading_difference_rad']))
    if 'speed_mps' in frame:
        errors.append(interval_error(speed, frame['speed_mps']))
    return np.column_stack(errors)


def best_hold(times, window, duration, errors):
    choices = []
    for start in np.flatnonzero((times >= window[0]-.001) & (times <= window[1]+.001)):
        end = int(np.searchsorted(times, times[start] + duration - .001))
        if end < len(times) and times[end] <= window[1] + .001:
            choices.append((float(np.mean(errors[start:end+1]**2)), start, end))
    if not choices:
        raise ValueError('No sampled hold interval fits this window; widen the window')
    return min(choices)


def objective_evidence(times, xy, yaw, ego, ego_yaw, size, objective):
    state = relative_state(times, xy, yaw, ego, ego_yaw)
    reports = []
    residual = 0.
    for frame in objective['keyframes']:
        errors = frame_errors(frame, state)
        cost, start, end = best_hold(times, frame['time_window_s'], frame['hold_duration_s'], errors)
        passed = bool(np.max(errors[start:end+1]) <= 1e-6)
        index = (start + end)//2
        reports.append(dict(passed=passed, evaluated_interval_s=[float(times[start]), float(times[end])],
            max_constraint_error=float(errors[start:end+1].max()),
            requested=frame,
            actual_ranges={name: [float(values[start:end+1].min()), float(values[start:end+1].max())]
                           for name, values in zip(('forward_m', 'left_m', 'heading_difference_rad', 'speed_mps'), state)},
            actual=dict(forward_m=float(state[0][index]), left_m=float(state[1][index]),
                        heading_difference_rad=float(state[2][index]), speed_mps=float(state[3][index]))))
        residual += cost
    target = bodies(xy, yaw, size)
    center = ego + (EGO_FRONT-EGO_REAR)/2 * np.column_stack([np.cos(ego_yaw), np.sin(ego_yaw)])
    own = bodies(center, ego_yaw, [EGO_FRONT+EGO_REAR, EGO_WIDTH])
    gaps = shapely.distance(target, own)
    contacts = np.flatnonzero(shapely.area(shapely.intersection(target, own)) > 1e-9)
    lower, upper = objective['interaction_window_s']
    mask = (times >= lower) & (times <= upper)
    if not mask.any():
        raise ValueError('Interaction window has no samples')
    if objective['interaction'] == 'allowed':
        interaction_ok, interaction_cost = True, 0.
    elif objective['interaction'] in ('contact', 'either'):
        interaction_ok = bool(len(contacts) and lower <= times[contacts[0]] <= upper)
        interaction_cost = float(gaps[mask].min())
        if objective['interaction'] == 'either' and not len(contacts):
            interaction_ok = bool(gaps.min() >= policy.ATTACK_NEAR_MISS_MIN_GAP_M and gaps[mask].min() <= policy.ATTACK_EVENT_MAX_GAP_M)
            interaction_cost = float(max(0., gaps[mask].min()-policy.ATTACK_EVENT_MAX_GAP_M))
    else:
        interaction_ok = bool(not len(contacts) and gaps.min() >= policy.ATTACK_NEAR_MISS_MIN_GAP_M and gaps[mask].min() <= policy.ATTACK_EVENT_MAX_GAP_M)
        interaction_cost = float(max(0., policy.ATTACK_NEAR_MISS_MIN_GAP_M-gaps.min()) + max(0., gaps[mask].min()-policy.ATTACK_EVENT_MAX_GAP_M))
    failed = [f'keyframe_{i+1}' for i, report in enumerate(reports) if not report['passed']]
    if not interaction_ok:
        failed.append('requested_interaction')
    index = int(np.argmin(gaps + np.where(mask, 0., 1e6)))
    return dict(goal_met=not failed, failed_checks=failed, keyframes=reports,
        objective_cost=float(residual + interaction_cost**2 + (0 if interaction_ok else 1)),
        measured_time_s=float(times[index]), interaction_mode=objective['interaction'],
        first_contact_time_s=None if not len(contacts) else float(times[contacts[0]]),
        min_clearance_m=float(gaps.min()),
        scope='Time-windowed body-center relative position, heading and speed; sampled reference geometry')


def goal_cost(evidence):
    return evidence.get('objective_cost', evidence.get('position_error_m', 0.))
