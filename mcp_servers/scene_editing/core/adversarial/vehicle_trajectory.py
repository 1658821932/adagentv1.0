"""Actor-independent construction and reference checks on a native scene timeline."""
import acceptance_policy as policy
import numpy as np
import shapely

from displacement_tool import interpolate_offsets
from ego_road_checks import check_drivable_area_official
from run_ego_reference import EGO_FRONT, EGO_REAR, EGO_WIDTH, polygon
from trajectory_checks import screen_reference


def sample_xy(times, xy, query):
    return np.column_stack([np.interp(query, times, xy[:, axis]) for axis in range(2)])


def attack_time_grid(times, boundary=3., step=.1):
    """Keep native samples and the exact observation boundary in the check grid."""
    dense = np.linspace(times[0], times[-1], int(np.ceil((times[-1]-times[0])/step))+1)
    extra = [boundary] if times[0] <= boundary <= times[-1] else []
    return np.unique(np.r_[times, dense, extra])


def motion_heading(times, xy, initial_yaw):
    velocity = np.gradient(xy, times, axis=0)
    heading = np.full(len(times), float(initial_yaw))
    for index in range(1, len(times)):
        if np.linalg.norm(velocity[index]) > .1:
            heading[index] = np.arctan2(velocity[index, 1], velocity[index, 0])
        else:
            heading[index] = heading[index - 1]
    return np.unwrap(heading)


def construct_trajectory(times, seed_xy, seed_yaw, points, end_velocity_mps, immutable_until_s=3.):
    """Reuse the displacement constructor's C2 interpolation without a fixed frame rate.

    Points are absolute target-center positions in the initial-ego local frame.
    Initial position and velocity are inherited from the selected object seed.
    """
    if not isinstance(points, list) or not 2 <= len(points) <= 16:
        raise ValueError('Provide 2..16 absolute control points covering the full scene')
    values = []
    for point in points:
        if not isinstance(point, dict) or set(point) != {'time_s', 'x_m', 'y_m'}:
            raise ValueError('Each control point needs time_s, x_m and y_m')
        row = [point[key] for key in ('time_s', 'x_m', 'y_m')]
        if any(type(value) not in (int, float) for value in row) or not np.isfinite(row).all():
            raise ValueError('Control points must contain finite numeric values')
        values.append(row)
    values = np.asarray(values, dtype=float)
    if abs(values[0, 0] - immutable_until_s) <= .001:
        values[0, 0] = immutable_until_s
    if values[0, 0] >= immutable_until_s:
        values = np.vstack([np.r_[times[0], seed_xy[0]], values])
    for index, boundary in ((0, times[0]), (-1, times[-1])):
        if abs(values[index, 0] - boundary) <= .001:
            values[index, 0] = boundary
    if values[0, 0] != times[0] or values[-1, 0] != times[-1] or np.any(np.diff(values[:, 0]) < .2 - 1e-8):
        raise ValueError('Control points must cover the whole window and be ordered at least 0.2s apart')
    seed_at_points = sample_xy(times, seed_xy, values[:, 0])
    if np.linalg.norm(values[0, 1:] - seed_xy[0]) > 1e-6:
        raise ValueError('Initial position must equal the selected object initial position')
    if not times[0] < immutable_until_s < times[-1]:
        raise ValueError('Immutable prefix must end inside the scene')
    prefix = values[:, 0] <= immutable_until_s + 1e-9
    if np.any(np.linalg.norm(values[prefix, 1:] - seed_at_points[prefix], axis=1) > .02):
        raise ValueError('Control points may not change the immutable observation prefix')
    # Small transcription/rounding differences in prefix points do not become
    # edits. This interval comes directly from the selected motion seed.
    boundary = sample_xy(times, seed_xy, np.array([immutable_until_s]))[0]
    values = np.vstack([np.r_[times[0], seed_xy[0]],
                        np.r_[immutable_until_s, boundary], values[~prefix]])
    seed_at_points = sample_xy(times, seed_xy, values[:, 0])
    offsets = values[:, 1:] - seed_at_points
    offsets[0] = 0.
    knots = [dict(time_s=float(row[0]), delta_forward_m=float(delta[0]),
                  delta_left_m=float(delta[1])) for row, delta in zip(values, offsets)]
    end_velocity = np.asarray(end_velocity_mps, dtype=float)
    if end_velocity.shape != (2,) or not np.isfinite(end_velocity).all():
        raise ValueError('end_velocity_mps must contain finite local x/y velocity')
    seed_end_velocity = (seed_xy[-1] - seed_xy[-2]) / (times[-1] - times[-2])
    native_offsets, spline = interpolate_offsets(
        knots, times, endpoint_velocities=np.array([[0., 0.], end_velocity - seed_end_velocity]))
    # Include native observations and the exact immutable boundary in geometry checks.
    dense_times = attack_time_grid(times, immutable_until_s)
    dense_xy = sample_xy(times, seed_xy, dense_times) + spline(dense_times)
    dense_yaw = motion_heading(dense_times, dense_xy, seed_yaw[0])
    xy = seed_xy + native_offsets
    yaw = np.interp(times, dense_times, dense_yaw)
    return xy, yaw, dense_times, dense_xy, dense_yaw


def actor_observation_series(times, actors, excluded):
    tracks = sorted({actor['track'] for frame in actors for actor in frame} - {excluded})
    series = []
    for track in tracks:
        observations = [(time, actor) for time, frame in zip(times, actors)
                        for actor in frame if actor['track'] == track]
        observed_times = np.array([item[0] for item in observations])
        poses = np.array([[item[1][key] for key in ('x', 'y', 'yaw', 'length', 'width')]
                          for item in observations], dtype=float)
        poses[:, 2] = np.unwrap(poses[:, 2])
        series.append((track, observed_times, poses))
    return series


def actor_pose_at_time(observed, poses, time, max_gap_s, scene_times=None):
    exact = np.flatnonzero(abs(observed - time) <= 1e-6)
    if len(exact):
        return poses[int(exact[0])]
    right = int(np.searchsorted(observed, time))
    if right == 0 or right == len(observed) or observed[right] - observed[right - 1] > max_gap_s:
        return None
    if scene_times is not None and np.any((scene_times > observed[right-1]+1e-6) & (scene_times < observed[right]-1e-6)):
        return None
    weight = (time - observed[right - 1]) / (observed[right] - observed[right - 1])
    return poses[right - 1] * (1 - weight) + poses[right] * weight



def motion_samples(times, dense_times, dense_xy, boundary):
    """Differentiate observed history at native cadence; retain every edited sample.

    Densifying piecewise-linear recorded positions creates artificial acceleration
    impulses at native knots. This changes only the derivative grid in the fixed
    prefix. Geometry uses the original complete dense grid, and every point after
    the boundary remains in the motion grid, including microsecond-spaced points.
    """
    prefix = times[times < boundary]
    boundary_sample = [boundary] if times[0] <= boundary <= times[-1] else []
    query = np.unique(np.r_[prefix, boundary_sample, dense_times[dense_times > boundary]])
    return np.column_stack([query, sample_xy(dense_times, dense_xy, query)])


def screen_attack(times, ego_xy, ego_yaw, actors, target, seed_xy, seed_yaw,
                  dense_times, dense_xy, dense_yaw, official_road, constraints, motion_times=None, motion_xy=None, geometry_cache=None):
    """Check native observations plus 0.1s interpolation; missing actor samples stay unknown."""
    effective = policy.resolve(constraints.get('acceptance_policy'))
    prefix_end = float(constraints.get('observation_end_s', policy.OBSERVATION_END_S))
    values = (motion_samples(times, dense_times, dense_xy, prefix_end) if motion_times is None
              else np.column_stack([motion_times, motion_xy]))
    initial_velocity = (seed_xy[1] - seed_xy[0]) / (times[1] - times[0])
    limits = dict(max_speed_mps=effective['max_speed_mps'], max_acceleration_mps2=effective['max_acceleration_mps2'],
                  max_lateral_acceleration_mps2=effective['attack_max_lateral_acceleration_mps2'], max_yaw_rate_radps=policy.ATTACK_MAX_YAW_RATE_RADPS,
                  max_initial_velocity_error_mps=policy.ATTACK_MAX_INITIAL_SPEED_ERROR_MPS)
    motion = screen_reference(values, dense_xy[-1], initial_velocity, limits)
    motion['sampling_scope'] = ('Native recorded prefix plus boundary; all dense edited samples'
                                if motion_times is None else 'Explicit native replay trajectory')
    prefix = dense_times <= prefix_end + 1e-9
    prefix_error = float(np.linalg.norm(dense_xy[prefix] - sample_xy(times, seed_xy, dense_times[prefix]), axis=1).max())
    static = np.array_equal(dense_xy, np.tile(dense_xy[0], (len(dense_xy), 1))) and np.array_equal(dense_yaw, np.full(len(dense_yaw), dense_yaw[0]))
    road = check_drivable_area_official(dense_times[:1] if static else dense_times,
        dense_xy[:1] if static else dense_xy, dense_yaw[:1] if static else dense_yaw, official_road,
        front=target['size_m'][0] / 2, rear=target['size_m'][0] / 2,
        width=target['size_m'][1], min_margin_m=float(constraints.get('official_margin_m', policy.OFFICIAL_MARGIN_M)))
    # Cache only immutable scene geometry, scoped to this editor and exact grid.
    # No raw-data hashes and no cache reuse between scene instances.
    cache = geometry_cache if geometry_cache is not None else {}
    if not np.array_equal(cache.get('times'), dense_times):
        cache.clear()
        cache['times'] = dense_times.copy()
        ego_dense = sample_xy(times, ego_xy, dense_times)
        ego_angles = np.interp(dense_times, times, np.unwrap(ego_yaw))
        center = ego_dense + (EGO_FRONT-EGO_REAR)/2*np.column_stack([np.cos(ego_angles), np.sin(ego_angles)])
        from attack_objective import bodies
        cache['ego_bodies'] = bodies(center, ego_angles, [EGO_FRONT+EGO_REAR, EGO_WIDTH])
        cache['actors'] = []
        for track, observed, poses in actor_observation_series(times, actors, target['track_id']):
            valid, shapes, states = [], [], []
            for index, time in enumerate(dense_times):
                pose = actor_pose_at_time(observed, poses, time, 1.1*np.median(np.diff(times)), scene_times=times)
                if pose is not None:
                    valid.append(index)
                    states.append(pose)
                    shapes.append(polygon(*pose[:2], pose[2], pose[3], pose[4]))
            cache['actors'].append((track, np.asarray(valid, dtype=int), shapes, states))
    from attack_objective import bodies
    target_shapes = bodies(dense_xy[:1], dense_yaw[:1], target['size_m']) if static else bodies(dense_xy, dense_yaw, target['size_m'])
    if static:
        target_shapes = np.repeat(target_shapes, len(dense_times))
    clearances = shapely.distance(target_shapes, cache['ego_bodies'])
    contacts = np.flatnonzero(shapely.area(shapely.intersection(target_shapes, cache['ego_bodies'])) > 1e-9).tolist()
    other_contacts = []
    checked_pairs = missing_pairs = 0
    for track, valid, shapes, states in cache['actors']:
        checked_pairs += len(valid)
        missing_pairs += len(dense_times)-len(valid)
        if not len(valid):
            continue
        overlaps = shapely.area(shapely.intersection(target_shapes[valid], shapes)) > 1e-9
        for local_index in np.flatnonzero(overlaps):
            index = valid[local_index]
            pose = states[local_index]
            other_contacts.append(dict(time_s=float(dense_times[index]), track_id=track,
                other_xy_m=pose[:2].tolist(), other_yaw_rad=float(pose[2]),
                other_size_m=pose[3:].tolist(), target_xy_m=dense_xy[index].tolist()))
    event_window = constraints.get('event_window_s', [prefix_end, float(times[-1])])
    event_mask = (dense_times > max(prefix_end, event_window[0])) & (dense_times <= event_window[1] + .001)
    if not event_mask.any():
        raise ValueError('Event window must include time after the observation period')
    event_gap = float(np.min(np.asarray(clearances)[event_mask]))
    travelled = float(np.linalg.norm(np.diff(dense_xy, axis=0), axis=1).sum())
    checks = dict(initial_position=np.linalg.norm(dense_xy[0] - seed_xy[0]) <= 1e-6,
                  observation_prefix=prefix_error <= 1e-5,
                  no_contact_during_observation=all(dense_times[index] > prefix_end for index in contacts),
                  motion=motion['status'] == 'passed', official_road=road['passed'],
                  observed_others=not other_contacts,
                  risk_in_event_window=event_gap <= float(constraints.get('max_ego_clearance_m', policy.ATTACK_EVENT_MAX_GAP_M)),
                  required_motion=travelled >= float(constraints.get('min_travel_m', 0.)))
    if constraints.get('stationary_target', False):
        checks['stationary_target'] = bool(np.max(np.linalg.norm(dense_xy-seed_xy[0], axis=1)) <= 1e-5 and
            np.max(np.abs(np.arctan2(np.sin(dense_yaw-seed_yaw[0]), np.cos(dense_yaw-seed_yaw[0])))) <= 1e-5)
    if constraints.get('require_contact', False):
        checks['contact_in_event_window'] = any(event_mask[index] for index in contacts)
    checks = {key:bool(value) for key, value in checks.items()}
    minimum = int(np.argmin(clearances))
    velocity = np.diff(dense_xy, axis=0) / np.diff(dense_times)[:, None]
    yaw_rate = abs(np.diff(dense_yaw) / np.diff(dense_times))
    moving = np.linalg.norm(velocity, axis=1) > 1.
    min_radius = float(np.min(np.linalg.norm(velocity[moving], axis=1) / np.maximum(yaw_rate[moving], 1e-9))) if moving.any() else None
    post_mask = dense_times > prefix_end
    post_index = int(np.flatnonzero(post_mask)[np.argmin(clearances[post_mask])])
    first = contacts[0] if contacts else None
    native_mask = np.isin(dense_times, times)
    native_contacts = [index for index in contacts if native_mask[index]]
    prefix_gap = float(np.min(clearances[prefix]))
    feedback = dict(placement_valid=all(checks.values()), checks=checks,
        first_positive_overlap_s=None if first is None else float(dense_times[first]),
        last_clear_sample_before_overlap_s=None if first is None or first == 0 else float(dense_times[first-1]),
        overlap_bracket_width_s=None if first is None or first == 0 else float(dense_times[first]-dense_times[first-1]),
        native_first_positive_overlap_s=None if not native_contacts else float(dense_times[native_contacts[0]]),
        observation_min_clearance_m=prefix_gap, post_observation_min_clearance_m=float(clearances[post_index]),
        post_observation_closest_time_s=float(dense_times[post_index]),
        zero_distance_sample_count=int(np.count_nonzero(clearances <= 1e-9)),
        positive_area_overlap_sample_count=len(contacts),
        first_other_contact=min(other_contacts, key=lambda item:item['time_s']) if other_contacts else None,
        checked_actor_pairs=checked_pairs, missing_actor_pairs=missing_pairs,
        lane_evidence=dict(status='unknown', reason='Drivable coverage does not identify a lane or certify lane-line compliance'),
        road_evidence=dict(passed=road['passed'], worst_corner_margin_m=road['worst_corner_margin_m'],
            map_source=road['map_source'], unique_pose_check=static, violation_count=len(road['violations']),
            violations=road['violations'][:8]), native_sample_count=int(native_mask.sum()),
        interpolated_sample_count=int((~native_mask).sum()),
        max_sample_interval_s=float(np.diff(dense_times).max()),
        temporal_scope='Sampled positive-area overlap; linear observation interpolation; earlier between-sample events and physical collision remain unverified')
    return dict(geometry_feedback=feedback, acceptance_policy=effective, effective_constraints=constraints, reference_screen_passed=all(checks.values()), checks=checks,
        failed_checks=[key for key, passed in checks.items() if not passed],
        min_ego_clearance_m=float(min(clearances)), event_min_clearance_m=event_gap,
        closest_time_s=float(dense_times[minimum]), first_contact_time_s=None if not contacts else float(dense_times[contacts[0]]),
        max_speed_mps=motion['metrics']['max_speed_mps'],
        max_acceleration_mps2=motion['metrics']['max_acceleration_mps2'],
        min_turn_radius_proxy_m=min_radius, travelled_m=travelled,
        observation_prefix_max_error_m=prefix_error, motion=motion, official_road=road,
        other_contacts=other_contacts, checked_actor_pairs=checked_pairs, missing_actor_pairs=missing_pairs,
        interpolation='Target C2 construction or parent edit; ego and observed others interpolated in time; gaps remain unknown',
        dense_sample_interval_max_s=float(np.diff(dense_times).max()), physical_collision='undetermined')
