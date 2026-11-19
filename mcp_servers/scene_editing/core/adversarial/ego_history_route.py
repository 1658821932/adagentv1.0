"""Fixed-start, short ego prefix fitting with the shared bicycle integrator."""

from copy import deepcopy

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import least_squares

from ego_history import TAKEOVER_S, check_history_actors
from ego_trajectory_edit import start_state
from run_ego_reference import EGO_FRONT, EGO_REAR, EGO_WIDTH, ego_path_from_frames
from vehicle_control_search import integrate_controls, wrap


EGO_SIZE = (EGO_FRONT + EGO_REAR, EGO_WIDTH)
MODEL_CENTER_OFFSET = .3 * EGO_SIZE[0]
KNOTS = np.linspace(0., TAKEOVER_S, 4)
PREFIX_TIMES = np.linspace(0., TAKEOVER_S, 61)
LIMITS = {'end_position_m': .12, 'end_yaw_rad': .08, 'end_speed_mps': .25, 'via_position_m': .18}


def _request(route):
    optional = {'end_yaw_rad', 'end_speed_mps', 'via_xy_m', 'via_time_s', 'connector_id'}
    if not isinstance(route, dict) or 'end_xy_m' not in route or set(route) - optional - {'end_xy_m'}:
        raise ValueError('ego_history_route needs end_xy_m and accepts end_yaw_rad, end_speed_mps, via_xy_m, via_time_s and connector_id')
    def finite(v):
        return type(v) in (int, float) and np.isfinite(v)

    clean = {key: value for key, value in route.items() if value is not None}
    for key in ('end_xy_m', 'via_xy_m'):
        if key == 'end_xy_m' or key in clean:
            value = clean.get(key)
            if not isinstance(value, list) or len(value) != 2 or not all(map(finite, value)):
                raise ValueError(f'{key} needs two finite scene-local rear-axle coordinates')
    for key in ('end_yaw_rad', 'end_speed_mps', 'via_time_s'):
        if key in clean and not finite(clean[key]):
            raise ValueError(f'{key} must be finite')
    if 'end_speed_mps' in clean and clean['end_speed_mps'] < 0.:
        raise ValueError('end_speed_mps must be nonnegative')
    if 'via_time_s' in clean and ('via_xy_m' not in clean or not 0. < clean['via_time_s'] < TAKEOVER_S):
        raise ValueError('via_time_s requires via_xy_m and must be strictly within the 1.5s history')
    if 'connector_id' in clean and (not isinstance(clean['connector_id'], str) or not clean['connector_id'].strip()):
        raise ValueError('connector_id must be a nonempty scene map token')
    return clean


def _headings(yaw):
    return np.column_stack([np.cos(yaw), np.sin(yaw)])


def _continue_reference(times, recorded_xy, recorded_yaw, end_xy, end_yaw, end_speed):
    """Create a C1 bridge from the prefix to a later recorded reference state."""
    future = np.flatnonzero(times >= TAKEOVER_S + 1.25)
    if not len(future):
        raise ValueError('Need at least 1.25s of source reference after the ego history to rejoin continuously')
    index = future[0]
    rejoin_s = float(times[index])
    reference = start_state(times, recorded_xy, recorded_yaw, rejoin_s)
    boundary_yaw = np.array([end_yaw, recorded_yaw[index]])
    speed = [end_speed, max(0., reference['signed_speed_mps'])]
    bridge = CubicHermiteSpline([TAKEOVER_S, rejoin_s], [end_xy, recorded_xy[index]],
                               _headings(boundary_yaw) * np.asarray(speed)[:, None], axis=0)
    return bridge, rejoin_s


def generate(scene, frames, route, acceptance, *, controls=None):
    """Return the frozen history, dense prefix samples and reproducible control evidence."""
    route = _request(route)
    times, xy, yaw = ego_path_from_frames(frames)
    if times[-1] < TAKEOVER_S:
        raise ValueError('Source does not cover the 1.5s takeover')
    index = int(np.searchsorted(times, TAKEOVER_S + .001, side='right')) - 1
    if index < 1:
        raise ValueError('Ego route history needs a completed native frame interval before takeover')
    initial = start_state(times, xy, yaw, 0.)
    v0 = initial['signed_speed_mps']
    if v0 < 0.:
        raise ValueError('Ego route history supports forward motion only')
    maximum_speed = min(acceptance['max_speed_mps'], acceptance['generator_max_speed_mps'])
    if v0 > maximum_speed or route.get('end_speed_mps', 0.) > maximum_speed:
        raise ValueError(f'Ego start/target speed exceeds the {maximum_speed:.2f} m/s generator limit')
    goal = np.asarray(route['end_xy_m'], dtype=float)
    max_distance = v0 * TAKEOVER_S + .5 * min(2.5, acceptance['max_acceleration_mps2']) * TAKEOVER_S**2
    required_distance = float(np.linalg.norm(goal - xy[0]))
    if required_distance > max_distance + .35:
        raise ValueError(f'Ego endpoint is {required_distance:.2f}m away; at most about {max_distance:.2f}m is reachable in 1.5s')

    prefix_grid = PREFIX_TIMES
    native_prefix = np.r_[times[times < TAKEOVER_S - .001], TAKEOVER_S]
    seed_yaw = np.interp(native_prefix, times, yaw)
    seed_xy = np.column_stack([np.interp(native_prefix, times, component) for component in xy.T])
    model_center = seed_xy + MODEL_CENTER_OFFSET * _headings(seed_yaw)
    # The shared model estimates the initial state from a body-center interval.
    # Supply the exact observed rear-axle velocity plus its rigid-body yaw term.
    left = np.array([-np.sin(seed_yaw[0]), np.cos(seed_yaw[0])])
    model_center[1] = (model_center[0] + native_prefix[1]*np.asarray(initial['velocity_xy_mps'])
                       + MODEL_CENTER_OFFSET*float(wrap(seed_yaw[1] - seed_yaw[0]))*left)
    via = np.asarray(route['via_xy_m']) if 'via_xy_m' in route else None
    lane = None
    lane_points = None
    if 'connector_id' in route:
        from connector_route import connector_centerlines
        lane, lines = connector_centerlines(scene, frames, route['connector_id'])
        lane_points = np.vstack(lines)

    def simulate(commands):
        arrays, state = integrate_controls(native_prefix, model_center, seed_yaw, prefix_grid, EGO_SIZE,
            commands, boundary=0., acceptance=acceptance)
        orientations = arrays[3]
        positions = arrays[2] - MODEL_CENTER_OFFSET * _headings(orientations)
        return positions, orientations, state

    def errors(positions, orientations, state):
        residual = {'end_position_m': positions[-1] - goal}
        if 'end_yaw_rad' in route:
            residual['end_yaw_rad'] = wrap(orientations[-1] - route['end_yaw_rad'])
        if 'end_speed_mps' in route:
            residual['end_speed_mps'] = state[-1, 0] - route['end_speed_mps']
        if via is not None:
            if 'via_time_s' in route:
                mid = np.array([np.interp(route['via_time_s'], prefix_grid, component) for component in positions.T])
            else:
                mid = positions[1:-1][np.linalg.norm(positions[1:-1] - via, axis=1).argmin()]
            residual['via_position_m'] = mid - via
        return residual

    def commands(parameters):
        return dict(times_s=KNOTS.tolist(), acceleration_mps2=np.r_[parameters[:3], parameters[2]].tolist(),
                    steering_rad=np.r_[parameters[3:], parameters[-1]].tolist())

    if controls is None:
        direction = goal - xy[0]
        heading_change = float(wrap(route.get('end_yaw_rad', 2 * np.arctan2(direction[1], direction[0]) - yaw[0]) - yaw[0]))
        turn = np.clip(np.arctan(.6 * EGO_SIZE[0] * heading_change / max(required_distance, .4)), -.48, .48)
        accel = np.clip((route.get('end_speed_mps', 2 * required_distance / TAKEOVER_S - v0) - v0) / TAKEOVER_S, -4., 2.)

        def loss(parameters):
            positions, orientations, state = simulate(commands(parameters))
            values = errors(positions, orientations, state)
            residual = [np.atleast_1d(value) / LIMITS[key] for key, value in values.items()]
            residual.extend((.035 * np.diff(parameters[:3]), .06 * np.diff(parameters[3:])))
            if lane_points is not None:
                distances = np.linalg.norm(positions[::10, None] - lane_points, axis=2).min(axis=1)
                residual.append(.08 * distances)
            return np.concatenate(residual)

        result = least_squares(loss, np.r_[np.full(3, accel), np.full(3, turn)],
            bounds=([-4.5]*3 + [-.55]*3, [2.5]*3 + [.55]*3), max_nfev=24, ftol=1e-4)
        controls = commands(result.x)

    positions, angles, state = simulate(controls)
    error_norms = {key: float(np.linalg.norm(np.atleast_1d(value)))
                   for key, value in errors(positions, angles, state).items()}
    missed = [f'{key} error {error:.3f} > {LIMITS[key]:.3f}' for key, error in error_norms.items() if error > LIMITS[key]]
    if missed:
        raise ValueError('Ego route target could not be fitted in 1.5s: ' + ', '.join(missed))

    speed = state[:, 0]
    yaw_rate = np.gradient(angles, prefix_grid)
    velocity = np.gradient(positions, prefix_grid, axis=0)
    acceleration = np.gradient(velocity, prefix_grid, axis=0)
    heading = _headings(angles)
    cross = heading[:, 0]*acceleration[:, 1] - heading[:, 1]*acceleration[:, 0]
    lateral = np.maximum(abs(speed * yaw_rate), abs(cross))
    planar = np.maximum(np.hypot(state[:, 1], lateral), np.linalg.norm(acceleration, axis=1))
    metrics = dict(peak_speed_mps=float(speed.max()), peak_acceleration_mps2=float(planar.max()),
        peak_lateral_acceleration_mps2=float(lateral.max()), peak_yaw_rate_radps=float(abs(yaw_rate).max()))
    for key, limit in zip(metrics, (maximum_speed, acceptance['max_acceleration_mps2'],
                         acceptance['max_lateral_acceleration_mps2'], acceptance['max_yaw_rate_radps'])):
        if metrics[key] > limit + 1e-6:
            raise ValueError(f'Ego route exceeds {key}: {metrics[key]:.3f} > {limit:.3f}')
    check_history_actors(frames, times, prefix_grid, positions, angles)

    end_speed = float(speed[-1])
    bridge, rejoin_s = _continue_reference(times, xy, yaw, positions[-1], angles[-1], end_speed)
    native_xy, native_yaw = xy.copy(), yaw.copy()
    prefix = times <= TAKEOVER_S
    native_xy[prefix] = np.column_stack([np.interp(times[prefix], prefix_grid, axis) for axis in positions.T])
    native_yaw[prefix] = np.interp(times[prefix], prefix_grid, angles)
    following = (times > TAKEOVER_S) & (times < rejoin_s)
    native_xy[following] = bridge(times[following])
    bridge_velocity = bridge(times[following], 1)
    native_yaw[following] = np.unwrap(np.r_[angles[-1], np.arctan2(bridge_velocity[:, 1], bridge_velocity[:, 0])])[1:]
    native_xy[0], native_yaw[0] = xy[0], yaw[0]
    centers = native_xy + (EGO_FRONT - EGO_REAR) / 2 * _headings(native_yaw)
    native_step = round(float(np.median(np.diff(times))) / .05) * .05
    report = dict(mode='endpoint_route', route=deepcopy(route), connector=lane, controls=controls,
        targets=[], takeover_s=TAKEOVER_S, takeover_xy_m=positions[-1].tolist(),
        takeover_yaw_rad=float(angles[-1]), continuous_takeover_speed_mps=end_speed,
        requested_takeover_speed_mps=route.get('end_speed_mps'),
        delivered_takeover_speed_mps=float(np.linalg.norm(centers[index] - centers[index-1]) / native_step),
        errors=error_norms, limits=LIMITS, motion=metrics, reference_rejoin_s=rejoin_s,
        reference_continuation='C1 rear-axle bridge to the recorded post-takeover reference')
    dense = (prefix_grid, positions, angles)
    return native_xy, native_yaw, report, dense
