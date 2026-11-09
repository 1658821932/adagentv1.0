"""Goal-directed candidate search using the original displacement curve families."""
import numpy as np

def _knots_smoothstep(p, needed, start_time_s=0., end_time_s=19.3, prefix_end_s=1., time_decimals=1):
    t1 = float(p['t0']+p['dt'])
    if time_decimals is not None:
        t1 = round(t1, time_decimals)
    return [dict(time_s=start_time_s, delta_forward_m=0.0, delta_left_m=0.0),
            dict(time_s=prefix_end_s, delta_forward_m=0.0, delta_left_m=0.0),
            dict(time_s=p['t0'], delta_forward_m=0.0, delta_left_m=0.0),
            dict(time_s=t1, delta_forward_m=float(needed[0]*p['scale']),
                 delta_left_m=float(needed[1]*p['scale'])),
            dict(time_s=p['tail_from'], delta_forward_m=float(needed[0]*p['scale']),
                 delta_left_m=float(needed[1]*p['scale'])),
            dict(time_s=end_time_s, delta_forward_m=float(needed[0]*p['scale']),
                 delta_left_m=float(needed[1]*p['scale']))]

def _knots_two_phase(p, needed, start_time_s=0., end_time_s=19.3, prefix_end_s=1., time_decimals=1):
    full_f, full_l = float(needed[0]*p['scale']), float(needed[1]*p['scale'])
    prep_f = float(full_f*p['prep_fraction'])
    prep_end = float(p['prep_start_s']+p['prep_duration_s'])
    lat_end = float(p['t0']+p['dt'])
    if time_decimals is not None:
        prep_end, lat_end = round(prep_end, time_decimals), round(lat_end, time_decimals)
    return [dict(time_s=start_time_s, delta_forward_m=0.0, delta_left_m=0.0),
            dict(time_s=prefix_end_s, delta_forward_m=0.0, delta_left_m=0.0),
            dict(time_s=p['prep_start_s'], delta_forward_m=0.0, delta_left_m=0.0),
            dict(time_s=prep_end, delta_forward_m=prep_f, delta_left_m=0.0),
            dict(time_s=p['t0'], delta_forward_m=prep_f, delta_left_m=0.0),
            dict(time_s=lat_end, delta_forward_m=full_f, delta_left_m=full_l),
            dict(time_s=p['tail_from'], delta_forward_m=full_f, delta_left_m=full_l),
            dict(time_s=end_time_s, delta_forward_m=full_f, delta_left_m=full_l)]
from vehicle_trajectory import sample_xy

MAX_INTERNAL_PER_CALL = 200
MAX_INTERNAL_PER_RUN = 1000


def validate_goal(arguments, times, prefix_end):
    if 'objective' in arguments:
        from attack_objective import validate_objective
        return validate_objective(arguments, times, prefix_end)
    if 'interaction_mode' in arguments:
        expected = {'parent_candidate_id', 'time_window_s', 'interaction_mode', 'entry_side',
                    'hold_duration_s', 'max_candidates', 'reason'}
        if set(arguments) != expected:
            raise ValueError('Invalid behavior-constructor arguments')
        window = arguments['time_window_s']
        if not isinstance(window,list) or len(window)!=2 or any(type(v) not in (int,float) or not np.isfinite(v) for v in window):
            raise ValueError('time_window_s needs two finite times')
        if not prefix_end < window[0] < window[1] <= times[-1]:
            raise ValueError('Behavior window must be after observation and inside the scene')
        if arguments['interaction_mode'] not in ('near_miss','contact') or arguments['entry_side'] not in ('left','right','either'):
            raise ValueError('Choose near_miss/contact and left/right/either')
        hold=arguments['hold_duration_s']
        if type(hold) not in (int,float) or not np.isfinite(hold) or not .3<=hold<=2. or hold>window[1]-window[0]:
            raise ValueError('Hold duration must be 0.3..2s and fit inside the window')
        if type(arguments['max_candidates']) is not int or not 1<=arguments['max_candidates']<=3:
            raise ValueError('max_candidates must be 1..3')
        if not isinstance(arguments['reason'],str) or not arguments['reason'].strip():
            raise ValueError('A nonempty reason is required')
        return
    expected = {'parent_candidate_id', 'goal_time_s', 'time_window_s',
                'desired_relative_forward_m', 'desired_relative_left_m',
                'position_tolerance_m', 'max_candidates', 'reason'}
    if set(arguments) != expected:
        raise ValueError('Invalid active-constructor arguments')
    for key in ('goal_time_s', 'desired_relative_forward_m', 'desired_relative_left_m', 'position_tolerance_m'):
        if type(arguments[key]) not in (int, float) or not np.isfinite(arguments[key]):
            raise ValueError(f'{key} must be finite')
    window = arguments['time_window_s']
    if not isinstance(window, list) or len(window) != 2 or any(
            type(value) not in (int, float) or not np.isfinite(value) for value in window):
        raise ValueError('time_window_s must contain two finite times')
    if not prefix_end < window[0] <= arguments['goal_time_s'] <= window[1] <= times[-1]:
        raise ValueError('Goal window must be after observation, inside the scene, and contain goal_time_s')
    if arguments['goal_time_s'] - prefix_end < .8:
        raise ValueError('Allow at least 0.8s after the observation prefix for construction')
    if arguments['position_tolerance_m'] <= 0:
        raise ValueError('position_tolerance_m must be positive')
    if type(arguments['max_candidates']) is not int or not 1 <= arguments['max_candidates'] <= 3:
        raise ValueError('max_candidates must be an integer in [1,3]')
    if not isinstance(arguments['reason'], str) or not arguments['reason'].strip():
        raise ValueError('A nonempty construction reason is required')


def _all_specs(times, ego, ego_yaw, parent_xy, arguments, prefix_end):
    """Search onset, interaction time, displacement scale and phase allocation.

    Geometry is generic: a requested relative pose supplies a displacement target.
    No lane-change, braking or crossing trajectory is selected by its name.
    """
    lower, upper = arguments['time_window_s']
    event_times = list(dict.fromkeys([arguments['goal_time_s'],
        lower + .25 * (upper - lower), lower + .75 * (upper - lower)]))
    desired = np.array([arguments['desired_relative_forward_m'], arguments['desired_relative_left_m']])
    for scale in (1., .85, 1.15):
        for event_time in event_times:
            span = event_time - prefix_end
            if span < .8:
                continue
            angle = float(np.interp(event_time, times, np.unwrap(ego_yaw)))
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            destination = sample_xy(times, ego, [event_time])[0] + desired @ rotation.T
            needed = destination - sample_xy(times, parent_xy, [event_time])[0]
            for onset_fraction in (0., .15, .3):
                start = prefix_end if onset_fraction == 0 else max(prefix_end + .2, prefix_end + span * onset_fraction)
                duration = event_time - start
                tail = event_time
                for family, prep_fraction in (('smoothstep', 0.), ('two_phase', .6), ('two_phase', .85)):
                    params = dict(family=family, t0=start, dt=duration, scale=scale, tail_from=tail)
                    if family == 'two_phase':
                        prep_duration = max(.2, duration * .4)
                        lateral_start = max(start + duration * .55, start + prep_duration + .2)
                        if event_time - lateral_start < .2:
                            continue
                        params.update(prep_start_s=start, prep_duration_s=prep_duration,
                            prep_fraction=prep_fraction, t0=lateral_start, dt=event_time-lateral_start)
                    factory = _knots_smoothstep if family == 'smoothstep' else _knots_two_phase
                    raw = factory(params, needed, start_time_s=float(times[0]),
                        end_time_s=float(times[-1]), prefix_end_s=prefix_end, time_decimals=None)
                    knots = []
                    for knot in raw:
                        if knots and abs(knot['time_s'] - knots[-1]['time_s']) < 1e-8:
                            if not np.allclose([knot['delta_forward_m'], knot['delta_left_m']],
                                    [knots[-1]['delta_forward_m'], knots[-1]['delta_left_m']], atol=1e-9):
                                raise ValueError('Coincident constructor knots disagree')
                            continue
                        knots.append(knot)
                    if np.any(np.diff([knot['time_s'] for knot in knots]) < .2 - 1e-8):
                        continue
                    parent_end_velocity = (parent_xy[-1] - parent_xy[-2]) / (times[-1] - times[-2])
                    event_position = sample_xy(times, parent_xy, [event_time])[0] + needed * scale
                    for terminal_fraction in (0., .5, 1.):
                        final_velocity = parent_end_velocity * terminal_fraction
                        terminal_knots = [dict(knot) for knot in knots]
                        end_offset = event_position + final_velocity * (times[-1] - event_time) - parent_xy[-1]
                        terminal_knots[-1].update(delta_forward_m=float(end_offset[0]),
                                                  delta_left_m=float(end_offset[1]))
                        yield dict(parameters=dict(params, terminal_speed_fraction=terminal_fraction),
                            interaction_time_s=float(event_time), knots=terminal_knots,
                            endpoint_velocities=[[0., 0.], (final_velocity - parent_end_velocity).tolist()])


def trial_specs(times, ego, ego_yaw, parent_xy, arguments, prefix_end, search_round=0, feedback=None, parent_parameters=None):
    if 'interaction_mode' in arguments:
        from vehicle_motion import motion_specs
        yield from motion_specs(times,prefix_end,arguments,search_round=search_round,feedback=feedback,parent_parameters=parent_parameters)
        return
    specs = list(_all_specs(times, ego, ego_yaw, parent_xy, arguments, prefix_end))
    # Spread the bounded budget over all parameter ranges instead of exhausting
    # it in the first displacement scale or terminal-speed group.
    indices = np.linspace(0, len(specs) - 1, min(len(specs), MAX_INTERNAL_PER_CALL), dtype=int)
    for index in indices:
        yield specs[int(index)]


def goal_evidence(times, xy, ego, ego_yaw, arguments, first_contact_time_s, target_yaw=None, target_size=None, task_contract=None):
    if 'objective' in arguments:
        from attack_task_contract import plan_evidence
        return plan_evidence(times, xy, target_yaw, ego, ego_yaw, target_size, arguments['objective'], task_contract)
    if 'interaction_mode' in arguments:
        from attack_behavior import behavior_evidence
        return behavior_evidence(times,xy,target_yaw,ego,ego_yaw,target_size,arguments)
    lower, upper = arguments['time_window_s']
    mask = (times >= lower - 1e-9) & (times <= upper + 1e-9)
    if not mask.any():
        raise ValueError('Goal window contains no dense evaluation samples; widen it')
    relative = xy - ego
    forward = relative[:, 0] * np.cos(ego_yaw) + relative[:, 1] * np.sin(ego_yaw)
    left = -relative[:, 0] * np.sin(ego_yaw) + relative[:, 1] * np.cos(ego_yaw)
    error = np.hypot(forward - arguments['desired_relative_forward_m'],
                     left - arguments['desired_relative_left_m'])
    indices = np.flatnonzero(mask)
    index = int(indices[np.argmin(error[mask])])
    contact_ok = first_contact_time_s is None or lower <= first_contact_time_s <= upper
    return dict(goal_met=bool(error[index] <= arguments['position_tolerance_m'] and contact_ok),
        position_goal_met=bool(error[index] <= arguments['position_tolerance_m']),
        first_contact_in_window_or_absent=bool(contact_ok),
        position_error_m=float(error[index]), measured_time_s=float(times[index]),
        actual_relative_forward_m=float(forward[index]), actual_relative_left_m=float(left[index]),
        requested_relative_forward_m=arguments['desired_relative_forward_m'],
        requested_relative_left_m=arguments['desired_relative_left_m'],
        scope='Relative target-center position in recorded ego rear-axle frame, not complete maneuver certification')
