"""Hard drivable-area checks for rear-axle ego trajectories against map_features products.

Coordinate convention (verified against the WorldEngine map_features exporter):
map_features.json polygons are normalized by translation only (world frame minus
first-frame ego center). Trajectory-local coordinates rotate by first-frame ego
yaw on top of that (see scripts/run_ego_reference.py scene_from_frames), so a
map polygon enters the trajectory-local frame as polygon @ R(yaw0) with
R(yaw0) = [[cos, -sin], [sin, cos]]. yaw0 is the first-frame ego yaw from
metadata/frames.json ego2global.

Drivable surface is the union of lane/lane_connector polygons only; sidewalks,
crosswalks and road lines are never drivable. The whole ego body outline is
checked, not the center. Signed road margin is +distance to the road boundary
when the body is covered, minus the penetration depth (farthest body vertex
outside the road) when it is not. A sample passes when margin >= -tolerance_m;
tolerance_m is an explicit small pass/fail slack (default 0.05 m) for map
seam floating-point cracks, not a road widening.

Use result['passed'] directly as checks['drivable_area'].
"""
import csv
import json
from pathlib import Path

import acceptance_policy as policy
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep

EGO_FRONT, EGO_REAR, EGO_WIDTH = 4.049, 1.127, 2.297
DRIVABLE_LANE_TYPES = ('LANE_SURFACE_STREET', 'LANE_SURFACE_HIGHWAY', 'LANE_SURFACE_UNSTRUCTURE', 'LANE_CONNECTOR')


def _as_json(source):
    if isinstance(source, (str, Path)):
        return json.loads(Path(source).read_text())
    return source


def first_frame_yaw(frames):
    """First-frame ego yaw (rad) from metadata/frames.json entries."""
    frames = _as_json(frames)
    yaw = np.unwrap(np.arctan2([f['ego2global'][1][0] for f in frames], [f['ego2global'][0][0] for f in frames]))
    return float(yaw[0])


def load_scene_road(input_dir, frames):
    metadata = Path(input_dir)/'metadata'
    reference = metadata/'map_reference.json'
    return load_road_geometry(metadata/'map_features.json', first_frame_yaw(frames),
        map_reference=reference if reference.is_file() else None)


class RoadGeometry:
    """Drivable-area union in trajectory-local coordinates plus map provenance."""

    def __init__(self, geometry, map_source=None, map_location=None, drivable_types=DRIVABLE_LANE_TYPES, polygon_count=0):
        self.geometry = geometry
        self.boundary = geometry.boundary
        self.prepared = prep(geometry)
        self.map_source = map_source
        self.map_location = map_location
        self.drivable_types = tuple(drivable_types)
        self.polygon_count = int(polygon_count)


def load_road_geometry(map_features, yaw0=0.0, map_reference=None, drivable_types=DRIVABLE_LANE_TYPES):
    """Build the drivable-area union from map_features (path or loaded dict).

    Polygons are rotated by R(yaw0) into the trajectory-local frame; yaw0 must be
    first_frame_yaw(frames) when map_features is the translate-only export.
    """
    map_source = str(map_features) if isinstance(map_features, (str, Path)) else None
    features = _as_json(map_features)
    rotation = np.array([[np.cos(yaw0), -np.sin(yaw0)], [np.sin(yaw0), np.cos(yaw0)]])
    polygons, used = [], set()
    for feature in features.values():
        ftype = feature.get('type')
        if ftype not in drivable_types or 'polygon' not in feature:
            continue
        used.add(ftype)
        points = np.asarray(feature['polygon'], dtype=float)
        if points.ndim != 2 or points.shape[0] < 3 or not np.isfinite(points).all():
            continue
        polygon = Polygon(points @ rotation)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty:
            polygons.append(polygon)
    if not polygons:
        raise ValueError('No drivable lane polygons found in map features')
    map_location = None
    if map_reference is not None:
        map_location = _as_json(map_reference).get('map_location')
    return RoadGeometry(unary_union(polygons), map_source=map_source, map_location=map_location,
        drivable_types=tuple(sorted(used)), polygon_count=len(polygons))


def ego_body_polygon(x, y, yaw, front=EGO_FRONT, rear=EGO_REAR, width=EGO_WIDTH):
    """Ego outline polygon; (x, y) is the rear axle, front extends along yaw."""
    c, s = np.cos(yaw), np.sin(yaw)
    half = (front + rear) / 2
    corners = np.array([[-rear, -width / 2], [front, -width / 2], [front, width / 2], [-rear, width / 2]])
    return Polygon(corners @ np.array([[c, s], [-s, c]]) + [x, y])


def signed_road_margin(body, road):
    """Signed distance from an ego outline to the drivable area: positive inside."""
    if road.prepared.covers(body):
        return float(road.boundary.distance(body))
    depth = 0.0
    outside = body.difference(road.geometry)
    for part in ([outside] if outside.geom_type == 'Polygon' else outside.geoms):
        if part.is_empty:
            continue
        for px, py in part.exterior.coords:
            depth = max(depth, road.geometry.distance(Point(px, py)))
    return -float(depth)


def densify_trajectory(times, xy, yaw, max_step_s=0.1):
    """Insert samples so adjacent observations are at most max_step_s apart.

    Linear xy interpolation on yaw-unwrapped angles; returns dense times, xy, yaw
    and the observation frame each dense sample belongs to.
    """
    times = np.asarray(times, dtype=float)
    xy = np.asarray(xy, dtype=float)
    yaw = np.unwrap(np.asarray(yaw, dtype=float))
    if times.ndim != 1 or xy.ndim != 2 or xy.shape != (len(times), 2) or yaw.shape != (len(times),):
        raise ValueError('times, xy (n,2) and yaw (n,) must align')
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError('times must be strictly increasing with at least two samples')
    if max_step_s <= 0:
        raise ValueError('max_step_s must be positive')
    dense_t, dense_xy, dense_yaw, obs_frames = [times[0]], [xy[0]], [yaw[0]], [0]
    for i in range(1, len(times)):
        count = max(1, int(np.ceil((times[i] - times[i - 1]) / max_step_s)))
        for k in range(1, count + 1):
            u = k / count
            dense_t.append(times[i - 1] + (times[i] - times[i - 1]) * u)
            dense_xy.append(xy[i - 1] + (xy[i] - xy[i - 1]) * u)
            dense_yaw.append(yaw[i - 1] + (yaw[i] - yaw[i - 1]) * u)
            obs_frames.append(i - 1 if k < count else i)
    return (np.array(dense_t), np.array(dense_xy), np.array(dense_yaw),
        np.array(obs_frames, dtype=int))


def check_drivable_area(times, xy, yaw, road, yaw0=None, max_step_s=policy.MAX_CHECK_STEP_S, tolerance_m=policy.ROAD_TOLERANCE_M,
        front=EGO_FRONT, rear=EGO_REAR, width=EGO_WIDTH):
    """Check a rear-axle ego trajectory against the drivable area.

    road is a RoadGeometry from load_road_geometry or a map_features source
    (path/loaded dict) together with yaw0. Samples are densified to max_step_s.
    Returns a JSON-friendly dict; result['passed'] is the drivable-area boolean.
    """
    if not isinstance(road, RoadGeometry):
        if yaw0 is None:
            raise ValueError('yaw0 is required when road is given as map features')
        road = load_road_geometry(road, yaw0)
    dense_t, dense_xy, dense_yaw, obs_frames = densify_trajectory(times, xy, yaw, max_step_s)
    margins = []
    failing = []
    for i, (x, y) in enumerate(dense_xy):
        margin = signed_road_margin(ego_body_polygon(x, y, dense_yaw[i], front, rear, width), road)
        margins.append(margin)
        if margin < -tolerance_m:
            failing.append({'dense_index': i, 'time_s': float(dense_t[i]), 'obs_frame': int(obs_frames[i]),
                'margin_m': float(margin), 'x': float(x), 'y': float(y), 'yaw': float(dense_yaw[i])})
    return {
        'passed': not failing,
        'min_margin_m': float(min(margins)),
        'tolerance_m': float(tolerance_m),
        'max_step_s': float(max_step_s),
        'n_observations': int(len(times)),
        'n_samples': int(len(dense_t)),
        'time_range': [float(dense_t[0]), float(dense_t[-1])],
        'failing_obs_frames': sorted({f['obs_frame'] for f in failing}),
        'failing_samples': failing,
        'margins_m': margins,
        'map_source': road.map_source,
        'map_location': road.map_location,
        'drivable_types': list(road.drivable_types),
        'drivable_polygon_count': road.polygon_count,
        'ego_geometry': {'front_m': front, 'rear_m': rear, 'width_m': width},
    }


def read_trajectory_csv(path):
    """Read time_s, ego_x_m, ego_y_m, ego_yaw_rad from a trajectory.csv; returns times, xy, yaw."""
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    required = {'time_s', 'ego_x_m', 'ego_y_m', 'ego_yaw_rad'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f'{path} lacks columns {sorted(required)}')
    times = np.array([float(r['time_s']) for r in rows])
    xy = np.array([[float(r['ego_x_m']), float(r['ego_y_m'])] for r in rows])
    yaw = np.array([float(r['ego_yaw_rad']) for r in rows])
    return times, xy, yaw


# ------------------------------------------------------------------ official nuPlan map

class OfficialDrivableMap:
    """Official nuPlan drivable polygons (trajectory-local frame) exported by
    scripts/export_official_drivable_map.py, matching the WorldEngine DAC geometry."""

    def __init__(self, polygons, meta):
        self.polygons = polygons
        self.prepared = [prep(p) for p in polygons]
        self.union = unary_union(polygons)
        self.meta = meta


def load_official_drivable(path):
    """Load the official-map export produced by export_official_drivable_map.py."""
    payload = _as_json(path)
    polygons = []
    for feature in payload['features'].values():
        points = np.asarray(feature['polygon'], dtype=float)
        if points.ndim != 2 or points.shape[0] < 3 or not np.isfinite(points).all():
            continue
        poly = Polygon(points)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_empty:
            polygons.append(poly)
    if not polygons:
        raise ValueError('No polygons in official drivable export')
    return OfficialDrivableMap(polygons, {k: v for k, v in payload.items() if k != 'features'})


def ego_corners_xy(x, y, yaw, front=EGO_FRONT, rear=EGO_REAR, width=EGO_WIDTH):
    """Four body corners (FL, RL, RR, FR); (x, y) is the rear axle."""
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[front, width / 2], [-rear, width / 2], [-rear, -width / 2], [front, -width / 2]])
    return local @ np.array([[c, s], [-s, c]]) + [x, y]


def check_drivable_area_official(times, xy, yaw, official, front=EGO_FRONT, rear=EGO_REAR,
                                 width=EGO_WIDTH, min_margin_m=0.0):
    """WorldEngine/PDMS DAC semantics on the official nuPlan map, plus an optional
    positive margin requirement.

    At every observed frame, each of the four ego corners must be strictly inside
    (boundary-exclusive) at least one drivable polygon; with min_margin_m > 0 each corner
    must additionally be at least that far inside the drivable union (absorbing map-seam
    numerics, inter-frame interpolation and downstream tracking error). Boundary points
    count as outside, matching shapely.vectorized.contains in the WorldEngine scorer.
    worst_corner_margin_m follows the same sign convention as signed_road_margin:
    positive means the worst corner is inside the union with that distance to the
    boundary, negative means outside by that amount.
    """
    violations = []
    corner_margin_min = float('inf')
    for i in range(len(times)):
        for corner_idx, corner in enumerate(ego_corners_xy(xy[i, 0], xy[i, 1], yaw[i], front, rear, width)):
            point = Point(corner)
            covered = official.union.covers(point)
            boundary_distance = official.union.boundary.distance(point) if covered else 0.0
            corner_margin_min = min(corner_margin_min,
                                    boundary_distance if covered else -official.union.distance(point))
            contained = any(p.contains(point) for p in official.prepared)
            if contained and boundary_distance >= min_margin_m:
                continue
            reason = 'not_contained' if not contained else 'margin_below'
            violations.append({'time_s': float(times[i]), 'frame': i, 'corner': corner_idx,
                               'x': float(corner[0]), 'y': float(corner[1]),
                               'reason': reason,
                               'distance_outside_m': float(official.union.distance(point)),
                               'boundary_distance_m': float(boundary_distance)})
    return {
        'passed': not violations,
        'violations': violations,
        'n_frames': int(len(times)),
        'min_margin_m': float(min_margin_m),
        'worst_corner_margin_m': float(corner_margin_min),
        'map_source': 'official nuPlan DB (WorldEngine PDMDrivableMap export)',
        'map_meta': official.meta,
    }
