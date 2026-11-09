"""Bounded reference candidates for the observed 2 Hz target scene, including reverse."""
import math
import acceptance_policy as policy
import numpy as np
from ego_road_checks import check_drivable_area, check_drivable_area_official
from run_ego_reference import DEFAULT_TARGET_TRACK, evaluate, smoothstep


def construct(times, xy, yaw, measured_forward_speed, mode, start_index, amount_m, duration_s):
    times = np.asarray(times, float); xy = np.asarray(xy, float); yaw = np.asarray(yaw, float)
    if mode not in {'retime', 'reverse'}:
        raise ValueError('mode must be retime or reverse')
    if isinstance(start_index, bool) or not isinstance(start_index, int) or not 0 <= start_index < len(times)-1:
        raise ValueError('start_index must identify an observed nonterminal frame')
    if isinstance(amount_m, bool) or not isinstance(amount_m, (int, float)) or not math.isfinite(amount_m) or not 0 < amount_m <= 4:
        raise ValueError('amount must be finite in (0,4]')
    if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)) or not math.isfinite(duration_s) or not 2 <= duration_s <= 8:
        raise ValueError('duration must be finite in [2,8] seconds')
    start = max(3., float(times[start_index])); phase = smoothstep((times-start)/duration_s)
    if mode == 'retime':
        if duration_s <= 1.875*amount_m:
            raise ValueError('retime duration is too short for monotonic progress')
        st = times - amount_m*phase
        if np.any(np.diff(st) <= 0) or st.min() < -1e-9:
            raise ValueError('retime must preserve monotonic source time')
        out_xy = np.column_stack([np.interp(st, times, xy[:, k]) for k in range(2)])
        out_yaw = np.interp(st, times, yaw)
    else:
        if abs(float(measured_forward_speed[start_index])) > 0.5:
            raise ValueError('reverse start requires observed forward speed <= 0.5 m/s')
        if start + duration_s > times[-1] + 1e-6:
            raise ValueError('reverse maneuver must fit in the observed window')
        out_xy = xy.copy(); out_yaw = yaw.copy(); after = np.arange(len(times)) >= start_index
        heading = yaw[start_index]; direction = np.array([np.cos(heading), np.sin(heading)])
        out_xy[after] = xy[start_index] - amount_m*phase[after,None]*direction
        out_yaw[after] = heading
    return out_xy, out_yaw


def _assess_full_window(times, baseline_xy, baseline_yaw, actors, candidate_xy, candidate_yaw, mode,
        target_track=DEFAULT_TARGET_TRACK, road=None, road_official=None, official_min_margin_m=policy.OFFICIAL_MARGIN_M, allow_safe_stop=False, no_cross_lines=None, acceptance=None):
    effective = policy.resolve(acceptance)
    if acceptance is None:
        effective['official_margin_m'] = official_min_margin_m
        effective.update(allow_safe_stop=allow_safe_stop, minimum_progress_fraction=policy.MIN_PROGRESS_FRACTION)
    official_min_margin_m = effective['official_margin_m']
    v = evaluate(times, candidate_xy, candidate_yaw, actors, times, target_track=target_track)
    dt = np.diff(times); delta = np.diff(candidate_xy, axis=0); motion = delta/dt[:,None]
    midpoint = (candidate_yaw[:-1] + candidate_yaw[1:])/2
    signed = np.sum(motion*np.column_stack([np.cos(midpoint),np.sin(midpoint)]),axis=1)
    lateral = np.sum(motion*np.column_stack([-np.sin(midpoint),np.cos(midpoint)]),axis=1)
    max_reverse = float(max(0., -signed.min()))
    other = v['min_other_clearance_m']
    others_observed = any(v['other_observed_counts'])
    checks = dict(target_clearance=not v['target_overlap_indices'],
                  observed_others=not others_observed or (other is not None and other > 0.),
                  speed=v['max_speed_mps'] <= effective['max_speed_mps'], acceleration=v['max_acceleration_mps2'] <= effective['max_acceleration_mps2'],
                  reverse_speed=max_reverse <= effective['max_reverse_speed_mps'], lateral_motion=float(np.abs(lateral).max()) <= effective['max_lateral_speed_mps'])
    # Observed actors are interpolated only across adjacent available samples.
    from vehicle_trajectory import actor_observation_series, actor_pose_at_time
    dense_times = np.linspace(times[0], times[-1], int(np.ceil((times[-1]-times[0])/effective['max_check_step_s']))+1)
    dense_xy = np.column_stack([np.interp(dense_times, times, candidate_xy[:, k]) for k in range(2)])
    dense_yaw = np.interp(dense_times, times, np.unwrap(candidate_yaw))
    from run_ego_reference import polygon, EGO_FRONT, EGO_REAR, EGO_WIDTH
    contacts = []
    for track, observed, poses in actor_observation_series(times, actors, '__none__'):
        for i, time in enumerate(dense_times):
            pose = actor_pose_at_time(observed, poses, time, 1.1*float(np.median(np.diff(times))), scene_times=times)
            if pose is None:
                continue
            center = dense_xy[i]+(EGO_FRONT-EGO_REAR)/2*np.array([np.cos(dense_yaw[i]), np.sin(dense_yaw[i])])
            if polygon(*center, dense_yaw[i], EGO_FRONT+EGO_REAR, EGO_WIDTH).intersects(polygon(*pose[:2], pose[2], *pose[3:5])):
                contacts.append(dict(time_s=float(time), track=track))
    checks['interpolated_contacts'] = not contacts
    if effective['road_speed_limit_mps'] is not None:
        checks['road_speed_rule'] = v['max_speed_mps'] <= effective['road_speed_limit_mps']
    prefix = times <= effective['observation_end_s']+1e-8
    checks['observation_prefix'] = not prefix.any() or bool(np.max(np.linalg.norm(candidate_xy[prefix]-baseline_xy[prefix], axis=1)) <= policy.POSE_TOLERANCE and
        np.max(np.abs(np.arctan2(np.sin(candidate_yaw[prefix]-baseline_yaw[prefix]), np.cos(candidate_yaw[prefix]-baseline_yaw[prefix])))) <= policy.POSE_TOLERANCE)
    if mode == 'reverse': checks['reverse_observed_in_candidate'] = bool(np.any(signed < -0.05))
    road_result = check_drivable_area(times, candidate_xy, candidate_yaw, road, tolerance_m=effective['road_tolerance_m'], max_step_s=effective['max_check_step_s']) if road is not None else None
    if road_result is not None:
        checks['drivable_area'] = road_result['passed']
    road_official_result = None
    if road_official is not None:
        road_official_result = check_drivable_area_official(
            times, candidate_xy, candidate_yaw, road_official, min_margin_m=official_min_margin_m)
        checks['drivable_area_official'] = road_official_result['passed']
    from run_ego_reference import progress_with_safe_stop
    progress_ok, progress_reason = policy.task_progress(times, candidate_xy,
        float(np.linalg.norm(np.diff(baseline_xy, axis=0), axis=1).sum()), effective)
    checks['progress'] = progress_ok
    lane_markings = None
    if no_cross_lines is not None:
        from ego_lane_markings import check_no_cross_lines
        lane_markings = check_no_cross_lines(times, candidate_xy, candidate_yaw, no_cross_lines)
        checks['no_cross_lines'] = lane_markings['passed']
    progress = v['progress_m']
    road_status = 'not_strictly_validated' if road_result is None else 'passed' if road_result['passed'] else 'failed'
    road_scope = 'road not validated' if road_result is None else 'whole-body road checks densified to 0.1 s; not continuous certification'
    coverage = {key: ('passed' if value else 'failed') for key,value in checks.items()}
    for key, available in [('drivable_area',road is not None),('drivable_area_official',road_official is not None),('no_cross_lines',no_cross_lines is not None)]:
        if not available:
            coverage[key] = 'unknown'
    if effective['road_speed_limit_mps'] is None:
        coverage['road_speed_rule'] = 'unknown'
    unknown = [key for key in effective['required_checks'] if coverage.get(key,'unknown')=='unknown']
    submit_status = 'rejected' if not all(checks.values()) else 'incomplete' if unknown else 'reference_only'
    acceleration = np.diff(motion,axis=0)/((dt[:-1]+dt[1:])/2)[:,None]
    accel_heading = candidate_yaw[1:-1]
    longitudinal = np.sum(acceleration*np.column_stack([np.cos(accel_heading),np.sin(accel_heading)]),axis=1)
    transverse = np.sum(acceleration*np.column_stack([-np.sin(accel_heading),np.cos(accel_heading)]),axis=1)
    acceleration_components = dict(max_longitudinal_acceleration_mps2=float(max(0.,longitudinal.max())),
        max_braking_deceleration_mps2=float(max(0.,-longitudinal.min())),
        max_lateral_acceleration_mps2=float(abs(transverse).max()),
        scope='Native finite differences projected onto body heading; diagnostics, not a joint tire/friction capability model')
    return dict(acceleration_components=acceleration_components, coverage=coverage, missing_required_checks=unknown, submit_status=submit_status,
        status='reference_screen_passed' if all(checks.values()) else 'reference_screen_failed', checks=checks,
        min_target_clearance_m=v['min_target_clearance_m'], min_other_clearance_m=other,
        target_overlap_indices=v['target_overlap_indices'], max_speed_mps=v['max_speed_mps'],
        max_acceleration_mps2=v['max_acceleration_mps2'], max_reverse_speed_mps=max_reverse,
        max_lateral_speed_mps=float(np.abs(lateral).max()), traveled_m=progress,
        acceptance_policy=effective, progress_reason=progress_reason, lane_markings=lane_markings,
        lane_compliance='verified_no_cross_lines' if no_cross_lines is not None else 'unknown', signed_speed_mps=signed.tolist(), road_status=road_status, road=road_result,
        road_official=road_official_result,
        interpolated_contacts=contacts, rear_coverage='partial_observations_only', physical_execution='not_run',
        scope=f'2 Hz sampled reference geometry; {road_scope}; no continuous collision certification; traveled/progress metrics are cumulative path length, not net forward displacement')


# Preserve the full-window evidence and classify immutable history separately.


def assess(times, baseline_xy, baseline_yaw, actors, candidate_xy, candidate_yaw, mode,
           target_track=DEFAULT_TARGET_TRACK, road=None, road_official=None,
           official_min_margin_m=policy.OFFICIAL_MARGIN_M, allow_safe_stop=False,
           no_cross_lines=None, acceptance=None):
    options = dict(target_track=target_track, road=road, road_official=road_official,
        official_min_margin_m=official_min_margin_m, allow_safe_stop=allow_safe_stop,
        no_cross_lines=no_cross_lines, acceptance=acceptance)
    result = _assess_full_window(times, baseline_xy, baseline_yaw, actors, candidate_xy, candidate_yaw, mode, **options)
    effective = result['acceptance_policy']
    boundary = effective['observation_end_s']
    result['full_window_coverage'] = dict(result['coverage'])
    result['full_window_checks'] = dict(result['checks'])
    result['full_window_status'] = result['status']
    history = np.flatnonzero(times <= boundary+1e-8)
    start = max(0, int(np.searchsorted(times, boundary))-1)
    def phase(indices):
        if len(indices) < 3:
            return None
        local = dict(effective, minimum_progress_fraction=None, required_checks=[])
        return _assess_full_window(times[indices], baseline_xy[indices], baseline_yaw[indices],
            [actors[i] for i in indices], candidate_xy[indices], candidate_yaw[indices], mode,
            **dict(options, acceptance=local))
    prior = phase(history)
    future = phase(np.arange(start, len(times)))
    historical_failures = [] if prior is None else [k for k,v in prior['checks'].items() if not v and k != 'progress']
    history_contact = any(k in historical_failures for k in ('target_clearance','observed_others','interpolated_contacts'))
    result['input_qualification'] = dict(status='input_ineligible' if history_contact else
        'input_anomaly' if historical_failures else 'eligible' if prior is not None else 'unknown',
        failed_checks=historical_failures, interval_s=[float(times[0]), min(boundary,float(times[-1]))])
    result['editing_coverage'] = None if future is None else future['coverage']
    result['editing_checks'] = None if future is None else future['checks']
    result['editing_status'] = 'unknown' if future is None else future['status']
    if effective['history_policy']=='report_existing' and future is not None:
        result['checks'] = dict(future['checks'], observation_prefix=result['checks']['observation_prefix'],
                                progress=result['checks']['progress'])
        if history_contact:
            result['checks']['input_eligible'] = False
        result['status'] = 'reference_screen_passed' if all(result['checks'].values()) else 'reference_screen_failed'
    if not all(result['checks'].values()):
        result['submit_status'] = 'rejected'
    elif result['missing_required_checks']:
        result['submit_status'] = 'incomplete'
    else:
        result['submit_status'] = 'reference_only_with_input_anomaly' if historical_failures else 'reference_only'
    return result
