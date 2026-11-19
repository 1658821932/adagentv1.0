"""Scene-local body-center waypoints for continuous vehicle control search."""

import math

import numpy as np

NUMBER = {'type': 'number'}
PAIR = {'type': 'array', 'items': NUMBER, 'minItems': 2, 'maxItems': 2}
TOLERANCES = {'position_tolerance_m': .5, 'speed_tolerance_mps': .5, 'yaw_tolerance_rad': .25}
FIELDS = {'position_m', 'time_s', 'time_window_s', 'speed_mps', 'yaw_rad', *TOLERANCES}
SCHEMA = {
    'type': 'array', 'minItems': 2, 'maxItems': 6,
    'items': {
        'type': 'object', 'required': ['position_m'], 'additionalProperties': False,
        'properties': {key: dict(NUMBER) for key in sorted(FIELDS - {'position_m', 'time_window_s'})}
        | {'position_m': dict(PAIR, description='Vehicle body center [x, y] in the scene frame, in metres.'),
           'time_window_s': dict(PAIR, description='Arrival interval in scene seconds. Supply this or time_s.')},
    },
}


def time_bounds(waypoint):
    return (waypoint['time_s'], waypoint['time_s']) if 'time_s' in waypoint else tuple(waypoint['time_window_s'])


def validate_waypoints(waypoints, lower, upper):
    def finite(value):
        return type(value) in (float, int) and math.isfinite(value)

    def pair(value):
        return isinstance(value, list) and len(value) == 2 and all(map(finite, value))

    if not isinstance(waypoints, list) or not 2 <= len(waypoints) <= 6:
        raise ValueError('Provide 2..6 waypoints including start and end')
    previous = -math.inf
    for index, waypoint in enumerate(waypoints, 1):
        label = f'waypoint {index}'
        if (not isinstance(waypoint, dict) or set(waypoint) - FIELDS
                or not pair(waypoint.get('position_m'))):
            raise ValueError(f'{label} requires finite position_m=[x, y] and supported fields')
        if ('time_s' in waypoint) == ('time_window_s' in waypoint):
            raise ValueError(f'{label} requires exactly one of time_s and time_window_s')
        if ('time_s' in waypoint and not finite(waypoint['time_s']) or
                'time_window_s' in waypoint and not pair(waypoint['time_window_s'])):
            raise ValueError(f'{label} requires finite scene times')
        start, end = time_bounds(waypoint)
        if abs(end - upper) <= .001 and end > upper:
            end = float(upper)
            if 'time_s' in waypoint:
                waypoint['time_s'] = start = end
            else:
                waypoint['time_window_s'][1] = end
        if not lower <= start <= end <= upper or start <= previous:
            raise ValueError(f'{label} times must be inside the editable horizon and strictly ordered')
        previous = end
        for key in ('speed_mps', 'yaw_rad', *TOLERANCES):
            if key not in waypoint:
                continue
            value = waypoint[key]
            if (not finite(value) or key == 'speed_mps' and value < 0
                    or key in TOLERANCES and value <= 0):
                raise ValueError(f'{label} has invalid {key}')
        for target, tolerance in (('speed_mps', 'speed_tolerance_mps'), ('yaw_rad', 'yaw_tolerance_rad')):
            if tolerance in waypoint and target not in waypoint:
                raise ValueError(f'{label} requires {target} with {tolerance}')


def measure_waypoints(times, xy, yaw, speed, waypoints):
    """Choose the best checked time in each ordered window, or use its exact time."""
    reports, residuals = [], []
    angles = np.unwrap(yaw)
    for waypoint in waypoints:
        start, end = time_bounds(waypoint)
        query = np.r_[start, times[(times > start) & (times < end)], end] if end > start else np.array([start])
        positions = np.column_stack([np.interp(query, times, axis) for axis in xy.T])
        speeds = np.interp(query, times, speed)
        headings = np.interp(query, times, angles)
        measurements = {'position_error_m': np.linalg.norm(positions - waypoint['position_m'], axis=1)}
        tolerances = {'position_error_m': waypoint.get('position_tolerance_m', TOLERANCES['position_tolerance_m'])}
        if 'speed_mps' in waypoint:
            measurements['speed_error_mps'] = np.abs(speeds - waypoint['speed_mps'])
            tolerances['speed_error_mps'] = waypoint.get('speed_tolerance_mps', TOLERANCES['speed_tolerance_mps'])
        if 'yaw_rad' in waypoint:
            difference = headings - waypoint['yaw_rad']
            measurements['yaw_error_rad'] = np.abs(np.arctan2(np.sin(difference), np.cos(difference)))
            tolerances['yaw_error_rad'] = waypoint.get('yaw_tolerance_rad', TOLERANCES['yaw_tolerance_rad'])
        errors = np.column_stack([np.maximum(values - tolerances[key], 0.) for key, values in measurements.items()])
        costs = np.einsum('ij,ij->i', errors, errors)
        index = int(np.argmin(costs))
        residuals.extend(errors[index])
        reports.append(dict(passed=bool(costs[index] <= 1e-12), evaluated_time_s=float(query[index]),
            requested=waypoint, actual=dict(position_m=positions[index].tolist(),
                speed_mps=float(speeds[index]), yaw_rad=float(headings[index])),
            errors={key: float(values[index]) for key, values in measurements.items()}, tolerances=tolerances))
    return reports, np.asarray(residuals)
