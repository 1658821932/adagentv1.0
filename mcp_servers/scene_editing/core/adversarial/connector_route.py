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
    allowed = set(FIELDS) | {'entry_lane', 'exit_lane'}
    if not isinstance(route, dict) or not {'connector_id', 'speed_mps'}.issubset(route) or not set(route).issubset(allowed):
        raise ValueError('connector_route requires connector_id and speed_mps (optional approach_m, exit_m)')
    connector_id = route['connector_id']
    if not isinstance(connector_id, str) or not connector_id.strip():
        raise ValueError('connector_id must be a nonempty scene map token')
    approach = route.get('approach_m', 20.0)
    departure = route.get('exit_m', 15.0)
    speed = route['speed_mps']
    if any(type(value) not in (int, float) or not np.isfinite(value) or value <= 0
           for value in (approach, departure, speed)):
        raise ValueError('Connector distances and nominal speed must be finite positive numbers')

    match, lines = connector_centerlines(scene, frames, connector_id)
    if 'entry_lane' in route and str(route['entry_lane']) != str(match.get('entry_lane')):
        raise ValueError(f"connector_route entry_lane {route['entry_lane']!r} != mapped {match.get('entry_lane')!r}")
    if 'exit_lane' in route and str(route['exit_lane']) != str(match.get('exit_lane')):
        raise ValueError(f"connector_route exit_lane {route['exit_lane']!r} != mapped {match.get('exit_lane')!r}")
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


def lane_centerline_seed(scene, frames, x_m, y_m, yaw_rad, speed_mps, times, max_snap_m=4.0, max_angle_rad=1.05, goal_xy=None, vehicle_length_m=4.5):
    """Build a natural lane-following seed trajectory aligned to the scene's real lane centerline.

    Returns (seed_xy, seed_yaw, aligned_start_xy, aligned_start_yaw) or None if no matching lane is found.
    """
    path = Path(scene) / 'metadata/map_features.json'
    if not path.is_file() or speed_mps <= 0.0:
        return None
    features = json.loads(path.read_text())
    yaw0 = first_frame_yaw(frames)
    c, s = np.cos(yaw0), np.sin(yaw0)
    rotation = np.array([[c, -s], [s, c]])

    lanes = {}
    for token, feat in features.items():
        if 'LANE' not in str(feat.get('type', '')).upper():
            continue
        pl = feat.get('polyline')
        if not pl or len(pl) < 2:
            continue
        pts = np.asarray(pl, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or not np.isfinite(pts).all():
            continue
        local_pts = pts @ rotation
        # Drop consecutive duplicates
        keep = np.r_[True, np.linalg.norm(np.diff(local_pts, axis=0), axis=1) > 1e-4]
        local_pts = local_pts[keep]
        if len(local_pts) >= 2:
            is_street = str(feat.get('type', '')).upper() == 'LANE_SURFACE_STREET'
            lanes[str(token)] = (local_pts, [str(x) for x in (feat.get('exit_lanes') or [])], is_street)

    if not lanes:
        return None

    target_pt = np.array([float(x_m), float(y_m)], dtype=float)
    best = None
    for token, (pts, _, is_street) in lanes.items():
        seg = np.diff(pts, axis=0)
        seg_len2 = np.sum(seg * seg, axis=1)
        for i in range(len(seg)):
            if seg_len2[i] <= 1e-8:
                continue
            t = float(np.clip(np.dot(target_pt - pts[i], seg[i]) / seg_len2[i], 0.0, 1.0))
            proj = pts[i] + t * seg[i]
            dist = float(np.linalg.norm(target_pt - proj))
            if dist > max_snap_m:
                continue
            seg_yaw = float(np.arctan2(seg[i, 1], seg[i, 0]))
            ang = abs(float(np.arctan2(np.sin(seg_yaw - yaw_rad), np.cos(seg_yaw - yaw_rad))))
            if ang > max_angle_rad:
                continue
            score = dist + 6.0 * ang + (0.0 if is_street else 0.8)
            if best is None or score < best[0]:
                best = (score, token, i, proj, seg_yaw)

    if best is None:
        return None

    _, cur_token, seg_idx, start_proj, start_yaw = best
    pts, exits, _ = lanes[cur_token]
    chain = [start_proj]
    for p in pts[seg_idx + 1:]:
        if np.linalg.norm(p - chain[-1]) > 1e-4:
            chain.append(p)

    needed_m = float((times[-1] - times[0]) * speed_mps) + 10.0
    visited = {cur_token}

    def chain_length(arr):
        if len(arr) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(np.asarray(arr), axis=0), axis=1)))

    while chain_length(chain) < needed_m and exits:
        last_dir = chain[-1] - chain[-2] if len(chain) >= 2 else np.array([np.cos(start_yaw), np.sin(start_yaw)])
        last_yaw = float(np.arctan2(last_dir[1], last_dir[0]))
        nxt_best = None
        goal_arr = np.asarray(goal_xy, dtype=float) if goal_xy is not None else None
        for cand_tok in exits:
            if cand_tok in visited or cand_tok not in lanes:
                continue
            npts, nexits, _ = lanes[cand_tok]
            ndir = npts[1] - npts[0]
            nyaw = float(np.arctan2(ndir[1], ndir[0]))
            ang = abs(float(np.arctan2(np.sin(nyaw - last_yaw), np.cos(nyaw - last_yaw))))
            gap = float(np.linalg.norm(npts[0] - chain[-1]))
            if gap > 5.0:
                continue
            if goal_arr is not None:
                # Prefer the branch whose endpoint/progress moves closer to goal_xy
                d_start = float(np.linalg.norm(npts[0] - goal_arr))
                d_end = float(np.linalg.norm(npts[-1] - goal_arr))
                progress = d_start - d_end  # positive when moving toward goal
                sc = -progress + 0.3 * ang + 0.2 * gap
            else:
                sc = ang + 0.2 * gap
            if nxt_best is None or sc < nxt_best[0]:
                nxt_best = (sc, cand_tok, npts, nexits)
        if nxt_best is None:
            break
        _, cur_token, npts, exits = nxt_best
        visited.add(cur_token)
        for p in npts:
            if np.linalg.norm(p - chain[-1]) > 1e-4:
                chain.append(p)

    line = np.asarray(chain, dtype=float)
    if len(line) < 2:
        return None
    seg = np.diff(line, axis=0)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(seg, axis=1))]
    total = float(arc[-1])
    if total < 1e-3:
        return None

    # Propagate a zero-lateral-slip kinematic bicycle (pure-pursuit) along centerline
    # so even sharp intersection turns satisfy bicycle.v1 rear-axle slip & steering limits.
    end_dir = seg[-1] / max(float(np.linalg.norm(seg[-1])), 1e-9)
    wb = 0.6 * float(vehicle_length_m)
    half_wb = 0.5 * wb
    t_arr = np.asarray(times, dtype=float)
    t_rel = t_arr - float(t_arr[0])

    def _point_at_arc(s_val):
        if s_val <= 0.0:
            return line[0].copy()
        if s_val <= total:
            j = int(np.searchsorted(arc, s_val, side='right')) - 1
            j = max(0, min(j, len(seg) - 1))
            denom = max(float(arc[j + 1] - arc[j]), 1e-9)
            return line[j] + ((s_val - arc[j]) / denom) * seg[j]
        return line[-1] + (s_val - total) * end_dir

    yaw = float(start_yaw)
    rx = float(line[0, 0] - half_wb * np.cos(yaw))
    ry = float(line[0, 1] - half_wb * np.sin(yaw))
    lookahead = max(3.5, 0.55 * float(speed_mps))
    max_steer = 0.45

    seed_xy = np.zeros((len(t_arr), 2), dtype=float)
    seed_yaw = np.zeros(len(t_arr), dtype=float)
    seed_xy[0] = [rx + half_wb * np.cos(yaw), ry + half_wb * np.sin(yaw)]
    seed_yaw[0] = yaw

    s_cursor = 0.0
    cur_t = 0.0
    dt_sub = 0.025
    for k in range(1, len(t_arr)):
        target_t = float(t_rel[k])
        n_steps = max(1, int(np.ceil((target_t - cur_t) / dt_sub)))
        h = (target_t - cur_t) / n_steps
        for _ in range(n_steps):
            s_cursor += float(speed_mps) * h
            goal_pt = _point_at_arc(s_cursor + lookahead)
            dx, dy = float(goal_pt[0] - rx), float(goal_pt[1] - ry)
            dist = max(float(np.hypot(dx, dy)), 1e-3)
            alpha = float(np.arctan2(np.sin(np.arctan2(dy, dx) - yaw), np.cos(np.arctan2(dy, dx) - yaw)))
            delta = float(np.clip(np.arctan2(2.0 * wb * np.sin(alpha), dist), -max_steer, max_steer))
            rx += float(speed_mps) * np.cos(yaw) * h
            ry += float(speed_mps) * np.sin(yaw) * h
            yaw += (float(speed_mps) / wb) * np.tan(delta) * h
        cur_t = target_t
        seed_xy[k] = [rx + half_wb * np.cos(yaw), ry + half_wb * np.sin(yaw)]
        seed_yaw[k] = yaw

    seed_yaw = np.unwrap(seed_yaw)
    return seed_xy, seed_yaw, seed_xy[0].copy(), float(seed_yaw[0])
