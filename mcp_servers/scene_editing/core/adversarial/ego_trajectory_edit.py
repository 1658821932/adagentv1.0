"""Explicit target replacement and forward-path progress editing for ego trajectories."""
import copy
import numpy as np
from scipy.interpolate import CubicHermiteSpline


INTENTS = ('unrestricted', 'forward_stop', 'forward', 'reverse')


def motion_evidence(times, xy, yaw):
    angles = np.unwrap(yaw)
    dt = np.diff(times)
    heading = (angles[:-1]+angles[1:])/2
    signed = np.sum(np.diff(xy, axis=0)/dt[:, None] *
        np.column_stack([np.cos(heading), np.sin(heading)]), axis=1)
    reverse = signed < -1e-6  # Numerical tolerance, not an allowed maneuver speed.
    return dict(max_reverse_speed_mps=float(max(0., -signed.min())),
        reverse_distance_m=float(np.sum(-signed[reverse]*dt[reverse])),
        reverse_intervals=[dict(start_s=float(times[i]), end_s=float(times[i+1]),
                                signed_speed_mps=float(signed[i])) for i in np.flatnonzero(reverse)],
        signed_speed_mps=signed.tolist(), scope='Body-heading projection of sampled displacements')


def intent_evidence(times, xy, yaw, intent, start_s, end_behavior=None, end_s=None, *, terminal_state=None):
    if intent not in INTENTS:
        raise ValueError('Unknown motion_intent')
    metrics = motion_evidence(times, xy, yaw)
    speed = np.array(metrics['signed_speed_mps'])
    # Include the interval crossing the edit start; history itself is not constrained.
    mask = times[1:] > start_s+1e-8
    direction_ok = True
    if intent in ('forward', 'forward_stop'):
        direction_ok = bool(np.all(speed[mask] >= -1e-6))
    elif intent == 'reverse':
        direction_ok = bool(np.all(speed[mask] <= 1e-6))
    terminal_stopped = bool(np.linalg.norm(xy[-1]-xy[-2]) <= 1e-6 and
        abs(np.arctan2(np.sin(yaw[-1]-yaw[-2]), np.cos(yaw[-1]-yaw[-2]))) <= 1e-6)
    terminal_method = 'last_sample_interval'
    if terminal_state is not None:
        terminal_stopped = bool(terminal_state['speed_mps'] <= 1e-6)
        terminal_method = 'reconstructed_generation_curve'
    hold_ok = True
    if end_behavior == 'hold':
        mask = times >= end_s-1e-8
        if mask.any():
            hold_ok = bool(np.max(np.linalg.norm(xy[mask]-xy[mask][0], axis=1)) <= 1e-6 and
                np.max(np.abs(np.unwrap(yaw[mask])-yaw[mask][0])) <= 1e-6)
    passed = bool(direction_ok and hold_ok and (intent != 'forward_stop' or terminal_stopped))
    return dict(passed=passed, declared=intent, direction_matched=bool(direction_ok),
        terminal_stopped=terminal_stopped, terminal_state_method=terminal_method, hold_matched=hold_ok, start_s=float(start_s),
        reverse_distance_m=metrics['reverse_distance_m'],
        max_reverse_speed_mps=metrics['max_reverse_speed_mps'],
        reverse_intervals=metrics['reverse_intervals'],
        failure_kind=None if passed else 'intent_mismatch',
        scope='Declared edit semantics, separate from research motion limits')


def apply_intent(result, times, xy, yaw, operation):
    """Use the same declared-intent check in editing, recovery and file review."""
    if operation.get('edit_mode', 'delta_parent') == 'delta_parent':
        return result
    if (result.get('generation_curve') or {}).get('edit_contract_version')=='trajectory_edit.v2':
        from edit_feedback import apply_edit_contract
        return apply_edit_contract(result,result['generation_curve'])
    if operation.get('edit_mode') == 'route_edit':
        curve=result['generation_curve'];summary=curve['route_edit_summary']
        matched=summary['target_offset_error_m']<=1e-4
        result['motion_intent']=dict(passed=matched,declared='lateral_route_edit',direction_matched=True,
            terminal_stopped=bool(np.linalg.norm(curve['terminal_velocity_xy_mps'])<1e-6),
            scope=summary['scope'])
        for key in ('checks','full_window_checks','editing_checks'):
            if isinstance(result.get(key),dict):result[key]['motion_intent']=matched
        if not matched:result.update(status='reference_screen_failed',submit_status='rejected')
        return result
    if operation.get('edit_mode') == 'speed_schedule':
        from ego_speed_schedule import apply_schedule_intent
        return apply_schedule_intent(result,times,xy,yaw)
    if operation.get('edit_mode') == 'speed_profile':
        operation = result['generation_curve']['resolved_speed_request']
    start = operation.get('start_time_s', operation.get('window_s', [3.])[0])
    if operation.get('edit_mode') == 'brake_profile':
        end = result['generation_curve']['transition_time_s']
    else:
        end = operation['points'][-1]['time_s'] if operation['end_behavior']=='hold' else operation['window_s'][1]
    end = _endpoint(end, times)
    curve = result.get('generation_curve') or {}
    terminal_state = None
    if curve.get('hold_starts_at_s') is not None and curve['hold_starts_at_s'] <= times[-1]+1e-8:
        terminal_state = dict(speed_mps=0.)  # Recomputed curve has an explicit zero-velocity hold.
    evidence = intent_evidence(times, xy, yaw, operation['motion_intent'], start,
                               operation['end_behavior'], end, terminal_state=terminal_state)
    result['motion_intent'] = evidence
    for key in ('checks', 'full_window_checks', 'editing_checks'):
        if isinstance(result.get(key), dict):
            result[key]['motion_intent'] = evidence['passed']
    for key in ('coverage', 'full_window_coverage', 'editing_coverage'):
        if isinstance(result.get(key), dict):
            result[key]['motion_intent'] = 'passed' if evidence['passed'] else 'failed'
    if not evidence['passed']:
        result.update(status='reference_screen_failed', full_window_status='reference_screen_failed',
            editing_status='reference_screen_failed', submit_status='rejected')
    return result


def _finite_number(value, label):
    if type(value) not in (float, int) or not np.isfinite(value):
        raise ValueError(label+' must be a finite number')
    return float(value)


def _endpoint(value, times):
    value = _finite_number(value, 'time_s')
    return float(times[-1]) if abs(value-times[-1]) <= .001 else value


def start_state(times, xy, yaw, time):
    """Use a completed left interval, allowing native timestamp rounding within 1 ms."""
    index = int(np.searchsorted(times, time+.001, side='right'))-1
    end = max(1,min(len(times)-1,index))
    begin = end-1
    velocity = (xy[end]-xy[begin])/(times[end]-times[begin])
    angle = float(np.interp(time,times,np.unwrap(yaw)))
    direction = np.array([np.cos(angle),np.sin(angle)])
    return dict(time_s=float(time),xy_m=[float(np.interp(time,times,xy[:,k])) for k in range(2)],
        yaw_rad=angle,velocity_xy_mps=velocity.tolist(),signed_speed_mps=float(velocity@direction),
        lateral_speed_mps=float(velocity@np.array([-direction[1],direction[0]])),
        source_interval_s=[float(times[begin]),float(times[end])],
        method='completed_left_interval' if index>=1 else 'first_interval_fallback',
        reference_point='ego_rear_axle',time_rounding_s=.001)


def _speed_at(times, xy, time):
    return np.asarray(start_state(times,xy,np.zeros(len(times)),time)['velocity_xy_mps'])


def curve_evidence(spline, state, scope):
    """Exact speed-norm and acceleration-norm maxima for each cubic polynomial piece."""
    max_speed = max_accel = 0.
    for index, duration in enumerate(np.diff(spline.x)):
        coefficients = spline.c[:,index]
        if coefficients.ndim==1: coefficients = coefficients[:,None]
        velocity = [np.polynomial.Polynomial([c[2],2*c[1],3*c[0]]) for c in coefficients.T]
        speed_squared = sum((v*v for v in velocity),np.polynomial.Polynomial([0.]))
        roots = speed_squared.deriv().roots()
        queries = [0.,float(duration)]+[float(v.real) for v in roots if abs(v.imag)<1e-8 and 0<v.real<duration]
        max_speed = max(max_speed,float(np.sqrt(max(0.,max(speed_squared(np.asarray(queries)))))))
        for t in (0.,duration):
            max_accel = max(max_accel,float(np.linalg.norm([v.deriv()(t) for v in velocity])))
    return dict(schema_version='generated_curve.v1',actual_start_state=state,
        max_speed_mps=max_speed,max_acceleration_mps2=max_accel,scope=scope)


def apply_curve_checks(result, evidence):
    if evidence is None: return result
    effective = result['acceptance_policy']
    speed_ok = evidence['max_speed_mps']<=effective['max_speed_mps']+1e-8
    acceleration_ok = evidence['max_acceleration_mps2']<=effective['max_acceleration_mps2']+1e-8
    result['generation_curve'] = evidence
    for key in ('speed','acceleration'):
        field = 'max_'+key+('_mps' if key=='speed' else '_mps2')
        result['native_'+field] = result[field]
        result[field] = max(result[field],evidence[field])
    for checks_key,coverage_key in (('checks','coverage'),('editing_checks','editing_coverage'),('full_window_checks','full_window_coverage')):
        checks = result.get(checks_key)
        if checks is None: continue
        checks['speed'] = checks['speed'] and speed_ok
        checks['acceleration'] = checks['acceleration'] and acceleration_ok
        checks['generated_curve_motion'] = speed_ok and acceleration_ok
        coverage = result[coverage_key]
        for key in ('speed','acceleration','generated_curve_motion'):coverage[key] = 'passed' if checks[key] else 'failed'
    if not (speed_ok and acceleration_ok):
        result.update(status='reference_screen_failed',editing_status='reference_screen_failed',
            full_window_status='reference_screen_failed',submit_status='rejected')
    return result


def replace_positions(times, parent_xy, parent_yaw, anchor_xy, anchor_yaw, arguments, boundary, *, with_details=False):
    """Replace timed target positions; anchor only defines coordinates, never accumulation."""
    mode = arguments['edit_mode']
    start, end = [_endpoint(value, times) for value in arguments['window_s']]
    if not boundary <= start < end <= times[-1]:
        raise ValueError('Replacement window must start after observation and fit the scene')
    behavior = arguments['end_behavior']
    if behavior not in ('hold', 'reconnect'):
        raise ValueError('end_behavior must be hold or reconnect')
    points = arguments['points']
    if not isinstance(points, list) or not 1 <= len(points) <= 5:
        raise ValueError('Provide 1..5 replacement points')
    knot_times = [start]
    position = [np.array([np.interp(start, times, parent_xy[:, k]) for k in range(2)])]
    headings = [float(np.interp(start, times, np.unwrap(parent_yaw)))]
    for point in points:
        expected = {'time_s','forward_m','left_m','yaw_rad'} if mode == 'set_offsets' else {'time_s','x_m','y_m','yaw_rad'}
        if set(point) != expected:
            raise ValueError('Invalid replacement point fields')
        t = _endpoint(point['time_s'], times)
        if not knot_times[-1] < t <= end:
            raise ValueError('Replacement point times must strictly increase after window start')
        angle = _finite_number(point['yaw_rad'], 'yaw_rad')
        if mode == 'set_offsets':
            forward = _finite_number(point['forward_m'], 'forward_m')
            left = _finite_number(point['left_m'], 'left_m')
            if abs(forward)>2. or abs(left)>2. or abs(angle)>.35:
                raise ValueError('Fixed-anchor offsets use existing +/-2m, +/-0.35rad limits')
            heading = float(np.interp(t, times, np.unwrap(anchor_yaw)))
            xy = np.array([np.interp(t, times, anchor_xy[:, k]) for k in range(2)])
            xy += [np.cos(heading)*forward-np.sin(heading)*left,
                   np.sin(heading)*forward+np.cos(heading)*left]
            angle += heading
        else:
            xy = np.array([_finite_number(point[key], key) for key in ('x_m','y_m')])
        knot_times.append(t); position.append(xy); headings.append(angle)
    if behavior == 'reconnect':
        if knot_times[-1] >= end:
            raise ValueError('Reconnect needs a free final boundary after the last target point')
        knot_times.append(end)
        position.append(np.array([np.interp(end, times, parent_xy[:, k]) for k in range(2)]))
        headings.append(float(np.interp(end, times, np.unwrap(parent_yaw))))
    elif abs(knot_times[-1]-end)>1e-8:
        raise ValueError('Hold replacement final point must equal window end')
    position = np.asarray(position)
    derivatives = np.gradient(position, knot_times, axis=0)
    derivatives[0] = _speed_at(times, parent_xy, start)
    derivatives[-1] = 0. if behavior == 'hold' else _speed_at(times, parent_xy, end)
    spline = CubicHermiteSpline(knot_times, position, derivatives)
    headings = np.unwrap(headings)
    rates = np.gradient(headings, knot_times)
    parent_rates = np.gradient(np.unwrap(parent_yaw), times)
    rates[0] = np.interp(start, times, parent_rates)
    rates[-1] = 0. if behavior == 'hold' else np.interp(end, times, parent_rates)
    yaw_spline = CubicHermiteSpline(knot_times, headings, rates)
    xy, yaw = parent_xy.copy(), parent_yaw.copy()
    inside = (times>start)&(times<=end)
    xy[inside], yaw[inside] = spline(times[inside]), yaw_spline(times[inside])
    if behavior == 'hold':
        xy[times>end], yaw[times>end] = position[-1], headings[-1]
    details = curve_evidence(spline,start_state(times,parent_xy,parent_yaw,start),
        'Exact planar cubic speed/acceleration extrema; yaw/body and road checks remain sampled.')
    details['resolved_targets'] = [dict(time_s=float(t),xy_m=pos.tolist(),yaw_rad=float(angle))
        for t,pos,angle in zip(knot_times,position,headings)]
    details['hold_starts_at_s'] = float(end) if behavior=='hold' else None
    return (xy,yaw,details) if with_details else (xy,yaw)


def path_progress(times, parent_xy, parent_yaw, path_xy, path_yaw, arguments, boundary, *, with_details=False):
    """Interpolate signed arc progress with inherited start speed and explicit end hold."""
    if arguments['end_behavior'] != 'hold':
        raise ValueError('Path progress currently requires explicit final hold')
    start = _endpoint(arguments['start_time_s'], times)
    if not boundary <= start < times[-1]:
        raise ValueError('Progress edit starts after observation and inside the scene')
    if motion_evidence(times, path_xy, path_yaw)['reverse_distance_m'] > 1e-6:
        raise ValueError('Choose a forward reference path; a reversing parent is not a forward path')
    distance = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    keep = np.r_[True, distance>1e-8]
    path_xy, path_yaw = path_xy[keep], np.unwrap(path_yaw)[keep]
    if len(path_xy)<2:
        raise ValueError('Reference path has no usable forward length')
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path_xy, axis=0), axis=1))]
    origin = np.array([np.interp(start, times, parent_xy[:, k]) for k in range(2)])
    delta = np.diff(path_xy, axis=0)
    fractions = (np.sum((origin-path_xy[:-1])*delta, axis=1)/np.sum(delta**2,axis=1)).clip(0.,1.)
    projection = path_xy[:-1]+fractions[:,None]*delta
    index = int(np.argmin(np.linalg.norm(projection-origin,axis=1)))
    if np.linalg.norm(projection[index]-origin)>1e-5:
        raise ValueError('Edit start is not on the chosen reference path; choose a matching path')
    origin_arc = arc[index]+fractions[index]*(arc[index+1]-arc[index])
    heading = float(np.interp(origin_arc, arc, path_yaw))
    angle = float(np.interp(start, times, np.unwrap(parent_yaw)))
    if abs(np.arctan2(np.sin(angle-heading),np.cos(angle-heading)))>1e-5:
        raise ValueError('Start heading must match the reference path')
    velocity = _speed_at(times,parent_xy,start)
    initial_speed = float(velocity @ np.array([np.cos(angle),np.sin(angle)]))
    points = arguments['points']
    if not isinstance(points,list) or not 1<=len(points)<=5:
        raise ValueError('Provide 1..5 progress targets')
    knots, values, speeds = [start], [origin_arc], [initial_speed]
    for point in points:
        if set(point)!={'time_s','distance_from_start_m','speed_mps'}:
            raise ValueError('Progress points require time_s, distance_from_start_m and signed speed_mps')
        t = _endpoint(point['time_s'], times)
        value = origin_arc+_finite_number(point['distance_from_start_m'],'distance_from_start_m')
        speed = _finite_number(point['speed_mps'],'speed_mps')
        if not knots[-1]<t<=times[-1]:
            raise ValueError('Progress times must strictly increase inside scene')
        if not arc[0]<=value<=arc[-1]:
            raise ValueError('Progress target leaves observed reference path; no path extrapolation')
        knots.append(t); values.append(value); speeds.append(speed)
    if abs(speeds[-1])>1e-8:
        raise ValueError('Final hold requires zero final signed speed')
    spline = CubicHermiteSpline(knots,values,speeds)
    # Check exact scalar extrema and velocity extrema as well as sampling.
    extrema = spline.derivative().roots(extrapolate=False)
    queries = np.unique(np.r_[knots,extrema[np.isfinite(extrema)]])
    if np.min(spline(queries))<arc[0]-1e-6 or np.max(spline(queries))>arc[-1]+1e-6:
        raise ValueError('Progress interpolation leaves available path')
    intent = arguments['motion_intent']
    speed_extrema = spline.derivative(2).roots(extrapolate=False)
    values_v = spline.derivative()(np.unique(np.r_[knots,speed_extrema[np.isfinite(speed_extrema)]]))
    if intent in ('forward','forward_stop') and np.min(values_v)<-1e-6:
        raise ValueError(f'intent_mismatch: cubic progress needs reverse (min speed {values_v.min():.3f}m/s); inherited speed={initial_speed:.3f}m/s. Increase stop distance or shorten stopping time, then recheck acceleration.')
    if intent=='reverse' and np.max(values_v)>1e-6:
        raise ValueError('intent_mismatch: reverse requires a stopped or reversing start, not a moving-forward state')
    xy,yaw = parent_xy.copy(),parent_yaw.copy()
    mask = times>start
    s = spline(np.minimum(times[mask],knots[-1]))
    xy[mask] = np.column_stack([np.interp(s,arc,path_xy[:,k]) for k in range(2)])
    yaw[mask] = np.interp(s,arc,path_yaw)
    details = curve_evidence(spline,start_state(times,parent_xy,parent_yaw,start),
        'Exact scalar arc-progress speed/tangential acceleration extrema; polyline curvature dynamics not certified.')
    details['resolved_targets'] = [dict(time_s=float(t),distance_from_start_m=float(value-origin_arc),
        speed_mps=float(speed),xy_m=[float(np.interp(value,arc,path_xy[:,k])) for k in range(2)])
        for t,value,speed in zip(knots,values,speeds)]
    details['hold_starts_at_s'] = float(knots[-1])
    return (xy,yaw,details) if with_details else (xy,yaw)


def inherited_brake_state(times,xy,yaw,time,parent_curve=None):
    """A defined parent curve, rather than its coarser CSV, owns its exact start state."""
    continuous = (parent_curve or {}).get('continuous_state')
    if continuous is not None and continuous['times_s'][0] <= time <= times[-1]+1e-8:
        from scipy.interpolate import PPoly
        knots = np.asarray(continuous['times_s'])
        spline = PPoly(np.asarray(continuous['xy_coefficients']), knots)
        query = min(time, float(knots[-1]))
        position = spline(query)
        velocity = spline(query, 1) if time <= knots[-1] else np.zeros(2)
        if time > knots[-1] and parent_curve.get('hold_starts_at_s') is None:
            return start_state(times,xy,yaw,time)
        if 'yaw_coefficients' in continuous:
            angle=float(PPoly(np.asarray(continuous['yaw_coefficients']),knots)(query))
        else:
            angle = float(np.arctan2(velocity[1],velocity[0])) if np.linalg.norm(velocity)>1e-8 else float(np.interp(time,times,np.unwrap(yaw)))
        direction = np.array([np.cos(angle),np.sin(angle)])
        return dict(time_s=float(time),xy_m=position.tolist(),yaw_rad=angle,velocity_xy_mps=velocity.tolist(),
            signed_speed_mps=float(velocity@direction),lateral_speed_mps=float(velocity@np.array([-direction[1],direction[0]])),
            source_interval_s=[float(knots[0]),float(knots[-1])],method='continuous_parent_curve',
            reference_point='ego_rear_axle',time_rounding_s=.001)
    actual=(parent_curve or {}).get('actual_start_state')
    if actual is not None and abs(actual['time_s']-time)<=1e-9:
        return copy.deepcopy(actual)
    return start_state(times,xy,yaw,time)


def modern_edit(times, lookup, operation, boundary, *, parent_curve=None, acceptance=None, parent_context=None):
    parent_xy,parent_yaw = lookup(operation['parent_candidate_id'])
    if operation['edit_mode']=='route_edit':
        if operation.get('route_edit_version','state_continuity.v2') != 'state_continuity.v2':
            raise ValueError('Unsupported route edit version; create a new task with current tools')
        from ego_route_edit import generate
        return generate(times,lookup,operation,boundary,parent_curve=parent_curve,acceptance=acceptance,parent_context=parent_context)
    if operation['edit_mode']=='speed_schedule':
        if operation.get('speed_schedule_version','state_continuity.v4') != 'state_continuity.v4':
            raise ValueError('Unsupported speed schedule version; create a new task with current tools')
        from ego_speed_schedule import generate
        return generate(times,lookup,operation,boundary,parent_curve=parent_curve,acceptance=acceptance,parent_context=parent_context)
    raise ValueError('Only current route_edit and speed_schedule operations are supported')


def braking_parameters(state, arguments, scene_end):
    """Resolve a constant tangential deceleration without changing supplied targets."""
    initial = float(np.linalg.norm(state['velocity_xy_mps']))
    target = _finite_number(arguments['target_speed_mps'],'target_speed_mps')
    limit = _finite_number(arguments['deceleration_mps2'],'deceleration_mps2')
    if not 0 <= target < initial or limit <= 0:
        raise ValueError('Braking requires 0 <= target_speed_mps < actual start speed and positive deceleration')
    shape_factor = 1.5 if arguments.get('speed_law') == 'smoothstep.v1' else 1.
    duration = shape_factor*(initial-target)/limit
    if 'transition_time_s' in arguments:
        duration = _finite_number(arguments['transition_time_s'],'transition_time_s')-state['time_s']
    if 'transition_distance_m' in arguments:
        distance_duration = 2*_finite_number(arguments['transition_distance_m'],'transition_distance_m')/(initial+target)
        if 'transition_time_s' in arguments and abs(distance_duration-duration)>1e-6:
            raise ValueError('incompatible_targets: time, distance and target speed disagree')
        duration = distance_duration
    if duration <= 0 or shape_factor*(initial-target)/duration > limit+1e-8:
        raise ValueError('incompatible_targets: braking target requires more than requested deceleration')
    if state['time_s']+duration > scene_end+1e-8:
        raise ValueError(f'target_unreachable: earliest target time {state["time_s"]+duration:.3f}s exceeds scene end {scene_end:.3f}s; required peak tangential deceleration at least {shape_factor*(initial-target)/(scene_end-state["time_s"]):.3f} m/s^2')
    behavior = 'hold' if target == 0 else 'cruise'
    if arguments['end_behavior'] != behavior:
        raise ValueError('Zero target requires hold; positive target requires cruise')
    if arguments['motion_intent'] not in (('forward_stop','forward') if target == 0 else ('forward',)):
        raise ValueError('Braking profile requires forward intent, or forward_stop with zero target')
    return duration,(initial-target)/duration,target


def brake_along_path(times, parent_xy, parent_yaw, path_xy, path_yaw, arguments, boundary, *, with_details=False, parent_curve=None):
    """Build a planar cubic curve from a consistent braking law and smooth path offset.

    The returned sampled poses and exact polynomial motion maxima share this curve.
    Road/contact/body-heading checks retain their existing sampled scope.
    """
    start = _endpoint(arguments['start_time_s'],times)
    if not boundary <= start < times[-1]:
        raise ValueError('Braking must start after observation and inside the scene')
    state = inherited_brake_state(times,parent_xy,parent_yaw,start,parent_curve)
    duration,deceleration,target = braking_parameters(state,arguments,times[-1])
    if motion_evidence(np.arange(len(path_xy),dtype=float),path_xy,path_yaw)['reverse_distance_m']>1e-6:
        raise ValueError('Braking requires a forward reference path')
    distance = np.linalg.norm(np.diff(path_xy,axis=0),axis=1)
    keep = np.r_[True,distance>1e-8]
    path_xy,path_yaw = path_xy[keep],np.unwrap(path_yaw)[keep]
    if len(path_xy)<2: raise ValueError('Reference path has no usable forward length')
    arc = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path_xy,axis=0),axis=1))]
    origin = np.asarray(state['xy_m'])
    delta = np.diff(path_xy,axis=0)
    fractions = (np.sum((origin-path_xy[:-1])*delta,axis=1)/np.sum(delta**2,axis=1)).clip(0.,1.)
    projection = path_xy[:-1]+fractions[:,None]*delta
    index = int(np.argmin(np.linalg.norm(projection-origin,axis=1)))
    if np.linalg.norm(projection[index]-origin)>1e-5:
        raise ValueError(f'path_mismatch: edit start is {np.linalg.norm(projection[index]-origin):.6f}m from reference; use adjust_trajectory_speed to preserve the parent route or request a checked connection to recorded_baseline')
    origin_arc = arc[index]+fractions[index]*(arc[index+1]-arc[index])
    heading = float(np.interp(origin_arc,arc,path_yaw))
    if abs(np.arctan2(np.sin(heading-state['yaw_rad']),np.cos(heading-state['yaw_rad'])))>1e-5:
        raise ValueError('Reference path heading must match inherited start heading')
    # Local geometry starts exactly at inherited pose, with its actual velocity tangent.
    ahead = arc>origin_arc+1e-8
    local_arc = np.r_[0.,arc[ahead]-origin_arc]
    positions = np.vstack([origin,path_xy[ahead]])
    headings = np.r_[state['yaw_rad'],path_yaw[ahead]]
    tangent = np.column_stack([np.cos(headings),np.sin(headings)])
    initial_speed = float(np.linalg.norm(state['velocity_xy_mps']))
    tangent[0] = np.asarray(state['velocity_xy_mps'])/initial_speed
    geometry_version = arguments.get('path_geometry_version', 'merge_micro_knots.v1')
    if geometry_version != 'merge_micro_knots.v1':
        raise ValueError('Unknown braking path geometry version')
    merge_distance = max(1e-8, initial_speed*state['time_rounding_s'])
    selected = [0]
    for i in range(1, len(local_arc)):
        if local_arc[i]-local_arc[selected[-1]] > merge_distance:
            selected.append(i)
    if len(selected) < 2:
        raise ValueError('No reference geometry beyond the exact braking start')
    local_arc,positions,tangent = local_arc[selected],positions[selected],tangent[selected]
    path = CubicHermiteSpline(local_arc,positions,tangent)
    offset = _finite_number(arguments.get('lateral_offset_m',0.),'lateral_offset_m')
    lateral_duration = _finite_number(arguments.get('lateral_duration_s',duration),'lateral_duration_s')
    if lateral_duration <= 0 or (offset and lateral_duration > duration+1e-8 and target==0):
        raise ValueError('Lateral transition must finish before a zero-speed hold')
    end = start+duration
    horizon = end if target==0 else float(times[-1])
    knots = np.unique(np.r_[np.linspace(start,horizon,max(2,int(np.ceil((horizon-start)/.025))+1)),
        end, start+min(lateral_duration,horizon-start)])
    from scipy.integrate import solve_ivp

    def components(time, progress):
        elapsed=time-start
        u=float(np.clip(elapsed/lateral_duration,0.,1.))
        shift=offset*(10*u**3-15*u**4+6*u**5)
        shift_rate=offset*30*u**2*(1-u)**2/lateral_duration
        first=path(progress,1);second=path(progress,2)
        length=float(np.linalg.norm(first))
        if length<1e-8: raise ValueError('Reference geometry has a degenerate tangent')
        normal=np.array([-first[1],first[0]])/length
        derivative=np.array([-second[1],second[0]])/length-normal*float(first@second)/length**2
        along=first+derivative*shift
        cross=normal*shift_rate
        if arguments.get('speed_law') == 'smoothstep.v1':
            phase=float(np.clip(elapsed/duration,0.,1.))
            speed=initial_speed-(initial_speed-target)*(3*phase**2-2*phase**3)
        else:
            speed=max(target,initial_speed-deceleration*elapsed)
        square=float(along@along)
        dot=float(along@cross)
        discriminant=dot*dot+square*(speed*speed-float(cross@cross))
        if discriminant < -1e-9 or square<1e-12:
            raise ValueError('incompatible_targets: lateral transition exceeds available braking speed')
        rate=(-dot+np.sqrt(max(0.,discriminant)))/square
        if rate < -1e-8:
            raise ValueError('incompatible_targets: lateral transition requires reverse reference progress')
        return max(0.,rate),shift,normal,along,cross

    def derivative(time, value):
        return [components(time,float(value[0]))[0]]

    solution=solve_ivp(derivative,(start,horizon),[0.],t_eval=knots,rtol=1e-9,atol=1e-11,max_step=.025)
    if not solution.success: raise ValueError('Braking path integration failed')
    progress=solution.y[0]
    if progress[-1] > local_arc[-1]+1e-8:
        raise ValueError('Braking/cruise leaves available reference path; no extrapolation')
    position=[];velocity=[];directions=[]
    for time,value in zip(knots,progress):
        rate,shift,normal,along,cross=components(time,float(value))
        position.append(path(value)+normal*shift)
        velocity.append(along*rate+cross)
        directions.append(along)
    position=np.asarray(position);velocity=np.asarray(velocity);directions=np.asarray(directions)
    velocity[0] = state['velocity_xy_mps']
    if target==0: velocity[-1]=0.
    spline = CubicHermiteSpline(knots,position,velocity)
    xy,yaw = parent_xy.copy(),parent_yaw.copy()
    mask = times>start
    queries = np.minimum(times[mask],horizon)
    xy[mask] = spline(queries)
    moving_knots = np.linalg.norm(velocity,axis=1)>1e-8
    directions[moving_knots] = velocity[moving_knots]
    angles = np.unwrap(np.arctan2(directions[:,1],directions[:,0]))
    actual_velocity = spline(queries,1)
    moving = np.linalg.norm(actual_velocity,axis=1)>1e-8
    result_angles = np.interp(queries,knots,angles)
    result_angles[moving] = np.arctan2(actual_velocity[moving,1],actual_velocity[moving,0])
    yaw[mask] = np.unwrap(result_angles)
    from scipy.integrate import quad
    travel = sum(quad(lambda t:float(np.linalg.norm(spline(t,1))),left,min(right,end),
        epsabs=1e-10,epsrel=1e-9)[0] for left,right in zip(knots[:-1],knots[1:]) if left<end)
    if 'transition_distance_m' in arguments and abs(travel-arguments['transition_distance_m'])>1e-5:
        raise ValueError('incompatible_targets: delivered curve travel differs from requested distance beyond numerical construction tolerance')
    details = curve_evidence(spline,state,'Exact planar generated cubic speed/acceleration extrema, including path curvature and lateral transition. Heading, road and contacts remain sampled; no physical execution.')
    if parent_curve is not None and parent_curve.get('actual_start_state',{}).get('time_s')==start:
        details['start_state_authority']='reconstructed_parent_generation_curve_at_matching_start'
    details['path_geometry_version'] = geometry_version
    details.update(profile='constant_tangential_deceleration_then_hold_or_cruise',
        tangential_deceleration_mps2=deceleration,target_speed_mps=target,
        transition_time_s=end,transition_distance_m=float(initial_speed*duration-.5*deceleration*duration**2),
        actual_transition_travel_m=float(travel),distance_construction_tolerance_m=1e-5,
        transition_xy_m=spline(end).tolist(),hold_starts_at_s=end if target==0 else None,
        reference_path_candidate_id=arguments['path_candidate_id'],lateral_offset_m=offset,
        resolved_targets=[dict(time_s=float(t),xy_m=spline(t).tolist(),speed_mps=float(np.linalg.norm(spline(t,1)))) for t in np.unique([start,end,horizon])],
        generated_piece_count=len(knots)-1,
        reference_progress_at_transition_m=float(np.interp(end,knots,progress)),
        speed_profile_scope='Analytic planar speed law mapped onto curved/offset path by numerical integration; actual delivered cubic extrema checked separately.',
        integration_rtol=1e-9,integration_atol=1e-11,
        construction_step_s=.025,limits_source='Requested tangential deceleration; planar maxima screened against unchanged acceptance_policy')
    if arguments.get('curve_state_version') == 'continuous.v1':
        details['continuous_state'] = dict(version='continuous.v1',times_s=spline.x.tolist(),xy_coefficients=spline.c.tolist())
        details['terminal_velocity_xy_mps'] = spline(horizon,1).tolist()
        details['speed_law'] = arguments.get('speed_law','constant_deceleration')
        if arguments.get('speed_law')=='smoothstep.v1':
            details['profile']='smooth_deceleration_then_hold_or_cruise'
        details['peak_tangential_deceleration_mps2'] = deceleration*(1.5 if arguments.get('speed_law')=='smoothstep.v1' else 1.)
        details['lateral_duration_s'] = lateral_duration
    return (xy,yaw,details) if with_details else (xy,yaw)


def verify_braking_candidate(times, baseline_xy, baseline_yaw, xy, yaw, operation, boundary):
    """Reconstruct constructor curves from the recorded baseline, never trust a peak label."""
    expected_xy,expected_yaw,curve = brake_along_path(times,baseline_xy,baseline_yaw,
        baseline_xy,baseline_yaw,operation,boundary,with_details=True)
    if not np.allclose(xy,expected_xy,atol=1e-9,rtol=0) or not np.allclose(yaw,expected_yaw,atol=1e-9,rtol=0):
        raise ValueError('Constructor braking curve differs from saved trajectory')
    return curve
