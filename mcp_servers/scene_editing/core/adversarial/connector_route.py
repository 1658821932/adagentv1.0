"""Turn a scene-map lane connector into timed body-center waypoints."""

import json
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from ego_road_checks import first_frame_yaw


FIELDS = ('connector_id', 'approach_m', 'exit_m', 'speed_mps')
SCHEMA = dict(type='object', properties={
    'connector_id': dict(type='string', description='Lane connector token from this scene map.'),
    'approach_m': dict(type='number', exclusiveMinimum=0, description='Start this far before the connector along its entry lane.'),
    'exit_m': dict(type='number', exclusiveMinimum=0, description='End this far beyond the connector along its exit lane.'),
    'speed_mps': dict(type='number', exclusiveMinimum=0, description='Nominal speed used to schedule the waypoints.'),
}, required=list(FIELDS), additionalProperties=False)


def connector_catalog(scene):
    path = Path(scene) / 'metadata/map_features.json'
    if not path.is_file():
        return {}, []
    features = json.loads(path.read_text())
    connectors = []
    for token, feature in features.items():
        if feature.get('type') not in ('LANE_SURFACE_UNSTRUCTURE', 'LANE_CONNECTOR'):
            continue
        incoming = feature.get('entry_lanes', [])
        outgoing = feature.get('exit_lanes', [])
        if len(incoming) != 1 or len(outgoing) != 1:
            continue
        lane_ids = (str(incoming[0]), token, str(outgoing[0]))
        if all(features.get(lane_id, {}).get('polyline') for lane_id in lane_ids):
            connectors.append(dict(connector_id=token, entry_lane=lane_ids[0], exit_lane=lane_ids[2]))
    return features, connectors


def nearby_connectors(scene, frames, ego_xy, limit=6):
    features, connectors = connector_catalog(scene)
    if not connectors:
        return []
    yaw0 = first_frame_yaw(frames)
    c, s = np.cos(yaw0), np.sin(yaw0)
    reference = ego_xy @ np.array([[c, s], [-s, c]])
    nearby = []
    for connector in connectors:
        points = np.asarray(features[connector['connector_id']]['polyline'], dtype=float)
        if points.ndim == 2 and points.shape[1] == 2 and np.isfinite(points).all():
            distance = float(np.linalg.norm(points[:, None] - reference, axis=2).min())
            nearby.append(dict(**connector, distance_to_ego_reference_m=round(distance, 1)))
    return sorted(nearby, key=lambda row: row['distance_to_ego_reference_m'])[:limit]


def connector_centerlines(scene, frames, connector_id):
    features, connectors = connector_catalog(scene)
    match = next((item for item in connectors if item['connector_id'] == connector_id), None)
    if match is None:
        raise ValueError(f'Connector {connector_id} has no usable centerline and unique entry/exit lanes in this scene map. Use explicit waypoints.')
    lane_ids = (match['entry_lane'], connector_id, match['exit_lane'])
    yaw0 = first_frame_yaw(frames)
    c, s = np.cos(yaw0), np.sin(yaw0)
    rotation = np.array([[c, -s], [s, c]])
    lines = []
    for lane_id in lane_ids:
        points = np.asarray(features[lane_id]['polyline'], dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError(f'Lane {lane_id} has an unusable centerline; use explicit waypoints')
        lines.append(points @ rotation)
    return match, lines


def plan_connector_route(scene, frames, route, start_s, horizon_s):
    if not isinstance(route, dict) or set(route) != set(FIELDS):
        raise ValueError('connector_route requires connector_id, approach_m, exit_m and speed_mps')
    connector_id = route['connector_id']
    if not isinstance(connector_id, str) or not connector_id.strip():
        raise ValueError('connector_id must be a nonempty scene map token')
    approach, departure, speed = (route[field] for field in FIELDS[1:])
    if any(type(value) not in (int, float) or not np.isfinite(value) or value <= 0
           for value in (approach, departure, speed)):
        raise ValueError('Connector distances and nominal speed must be finite positive numbers')

    match, lines = connector_centerlines(scene, frames, connector_id)
    line = np.vstack(lines)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    entry = float(arc[len(lines[0])])
    exit_ = float(arc[len(lines[0]) + len(lines[1]) - 1])
    if entry >= exit_:
        raise ValueError(f'Connector {connector_id} has an empty centerline; use explicit waypoints')
    if approach > arc[len(lines[0])-1] or departure > arc[-1] - arc[len(lines[0])+len(lines[1])]:
        raise ValueError('Requested approach or exit distance exceeds the mapped entry/exit lane')
    begin, end = entry - approach, exit_ + departure
    duration = (end - begin) / speed
    if duration > horizon_s - start_s + .001:
        raise ValueError(f'Connector route takes {duration:.2f}s at the requested speed; only {horizon_s-start_s:.2f}s is editable. Adjust the route distances/speed or use explicit waypoints.')

    def position(distance):
        return np.array([np.interp(distance, arc, component) for component in line.T])

    length = exit_ - entry
    stations = [begin, entry - min(1., approach / 2), entry + length*.18,
                entry + length*.44, entry + length*.72, end]
    times = [float(min(horizon_s, start_s + (distance - begin)/speed)) for distance in stations]
    waypoints = []
    for index, (distance, time_s) in enumerate(zip(stations, times)):
        dx, dy = position(distance+.35) - position(distance-.35)
        waypoint = dict(position_m=position(distance).tolist(), speed_mps=float(speed),
            yaw_rad=float(np.arctan2(dy, dx)), position_tolerance_m=.85,
            speed_tolerance_mps=.75, yaw_tolerance_rad=.33)
        if index == 0:
            waypoint.update(time_s=time_s, position_tolerance_m=.22)
        else:
            gap = min(time_s - times[index-1], times[index+1] - time_s if index+1 < len(times) else np.inf)
            slack = min(.42, gap/3)
            waypoint['time_window_s'] = [float(time_s-slack), float(min(horizon_s, time_s+slack))]
        waypoints.append(waypoint)
    return dict(**match, approach_m=approach, exit_m=departure, speed_mps=speed,
        connector_length_m=round(length, 2),
        nominal_connector_entry_s=round(start_s + approach / speed, 2),
        nominal_connector_exit_s=round(start_s + (approach + length) / speed, 2),
        nominal_route_end_s=round(start_s + duration, 2), waypoints=waypoints)


def connector_passage(scene, plan, times, positions):
    """Report native-frame lane evidence without changing candidate eligibility."""
    from shapely import covers, points, union_all

    path = Path(scene) / 'metadata/map_drivable_official.json'
    if not path.is_file():
        return dict(status='official_lane_polygons_unavailable')
    lane_ids = [plan['entry_lane'], plan['connector_id'], plan['exit_lane']]
    features = json.loads(path.read_text())['features'].values()
    samples = points(positions)
    passages = []
    for lane_id in lane_ids:
        shapes = [Polygon(row['polygon'], holes=row.get('holes')).buffer(0)
                  for row in features if str(row.get('token')) == lane_id]
        matched = times[covers(union_all(shapes), samples)] if shapes else []
        passages.append(dict(lane_id=lane_id, polygon_available=bool(shapes),
            first_time_s=round(float(matched[0]), 3) if len(matched) else None,
            last_time_s=round(float(matched[-1]), 3) if len(matched) else None))
    first_times = [row['first_time_s'] for row in passages]
    in_order = all(time is not None for time in first_times) and first_times[0] < first_times[1] < first_times[2]
    return dict(status='sampled', passes_in_order=in_order, lane_passages=passages)
