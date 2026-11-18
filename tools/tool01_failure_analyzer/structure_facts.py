"""Extract failure facts from official scores, recorded motion and map geometry.

Motion and road labels use explicit thresholds; ego_cause uses templates.
Run with the simulator interpreter:
    python structure_facts.py --archive <dir> --token <token>
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import pickle
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.pdm_scores import score_file

STOPPED_MPS = 0.3
LANE_HALF_WIDTH_M = 3.5
ADJACENT_MAX_M = 9.0
BODY_DISTANCE_LIMIT_M = 5.0
BODY_DISTANCE_TOLERANCE_M = 1e-6
TURN_DEG = 25.0
LANE_CHANGE_LATERAL_M = 2.0
INTERSECTION_RANGE_M = 40.0
CURVE_DEG = 8.0
DEFAULT_MAPS_ROOT = "data/navtrain_ep_per1/maps"


class FactsError(RuntimeError):
    """A required input is missing or inconsistent; the caller must pick another scene."""


# ------------------------------------------------------------------ scene input
def scene_pickle(archive, token):
    root = os.path.join(archive, "WE_output/openscene_format/meta_datas")
    candidates = [p for p in sorted(glob.glob(os.path.join(root, "*.pkl")))
                  if os.path.basename(p)[:-4] == token or os.path.basename(p)[:-4].endswith(token)]
    if not candidates:
        raise FactsError(f"no scene pickle for token {token} in {root}")
    if len(candidates) > 1:
        raise FactsError(f"ambiguous scene pickle for token {token}: {[os.path.basename(p) for p in candidates]}")
    with open(candidates[0], "rb") as stream:
        frames = pickle.load(stream)
    if not isinstance(frames, list) or len(frames) < 2:
        raise FactsError(f"scene pickle holds no frame list: {candidates[0]}")
    return frames, os.path.basename(candidates[0])[:-4]


def ego_facts(frames):
    times, positions, yaw = [], [], []
    for frame in frames:
        matrix = frame.get("ego2global")
        if matrix is None:
            continue
        times.append((frame["timestamp"] - frames[0]["timestamp"]) / 1e6)
        positions.append((float(matrix[0][3]), float(matrix[1][3])))
        yaw.append(math.atan2(matrix[1][0], matrix[0][0]))
    if len(times) < 2:
        raise FactsError("scene has fewer than two ego poses")
    origin_x, origin_y, yaw0 = positions[0][0], positions[0][1], yaw[0]
    lateral = []
    for x, y in positions:
        dx, dy = x - origin_x, y - origin_y
        lateral.append(-math.sin(yaw0) * dx + math.cos(yaw0) * dy)
    unwrapped = [yaw[0]]
    for value in yaw[1:]:
        delta = value - unwrapped[-1]
        while delta > math.pi:
            delta -= 2 * math.pi
        while delta < -math.pi:
            delta += 2 * math.pi
        unwrapped.append(unwrapped[-1] + delta)
    turn_deg = math.degrees(unwrapped[-1] - unwrapped[0])
    if abs(turn_deg) >= TURN_DEG:
        action = "turn_right" if turn_deg < 0 else "turn_left"
    elif abs(lateral[-1]) >= LANE_CHANGE_LATERAL_M:
        action = "lane_change_right" if lateral[-1] < 0 else "lane_change_left"
    else:
        action = "straight"
    return dict(times=times, positions=positions, yaw=yaw, action=action, turn_deg=turn_deg)


# ------------------------------------------------------------------ actors
def ego_heading(frame, index):
    """World heading of the ego pose that belongs to this very frame."""
    tolerance = 1e-6
    matrix = frame.get("ego2global")
    if matrix is None:
        raise FactsError(f"frame {index} has no ego2global pose")
    try:
        pose = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError) as error:
        raise FactsError(f"frame {index} ego2global is not numeric") from error
    if pose.shape != (4, 4):
        raise FactsError(f"frame {index} ego2global must be 4x4, got {pose.shape}")
    if not np.isfinite(pose).all():
        raise FactsError(f"frame {index} ego2global holds non-finite values")
    rotation = pose[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=tolerance, rtol=0.0):
        raise FactsError(f"frame {index} ego2global rotation is not orthonormal")
    if abs(float(np.linalg.det(rotation)) - 1.0) > tolerance:
        raise FactsError(f"frame {index} ego2global rotation is not a proper rotation")
    return math.atan2(float(rotation[1][0]), float(rotation[0][0]))


def actor_sequences(frames):
    tracks = {}
    for index, frame in enumerate(frames):
        heading = ego_heading(frame, index)
        anns = frame.get("anns") or {}
        tokens = [str(t) for t in anns.get("track_tokens", [])]
        names = [str(n) for n in anns.get("gt_names", [])]
        boxes = np.asarray(anns.get("gt_boxes", []), dtype=float)
        velocities = np.asarray(anns.get("gt_velocity_3d", []), dtype=float)
        for position, track in enumerate(tokens):
            if names[position] not in ("vehicle", "bus", "truck"):
                continue
            if boxes.ndim != 2 or boxes.shape[0] <= position or boxes.shape[1] < 7:
                raise FactsError(f"frame {index} has no usable box for {track}")
            box = boxes[position]
            if not all(math.isfinite(value) for value in (box[0], box[1], box[3], box[4], box[6])):
                raise FactsError(f"frame {index} box for {track} has non-finite planar geometry")
            if box[3] <= 0.0 or box[4] <= 0.0:
                raise FactsError(f"frame {index} box for {track} has unusable size {box[3]}x{box[4]}")
            if velocities.ndim != 2 or velocities.shape[0] <= position or velocities.shape[1] < 2:
                raise FactsError(f"frame {index} has no usable velocity for {track}")
            planar_x, planar_y = velocities[position][0], velocities[position][1]
            if not (math.isfinite(planar_x) and math.isfinite(planar_y)):
                raise FactsError(f"frame {index} velocity for {track} has non-finite planar components")
            speed = float(math.hypot(planar_x, planar_y))
            tracks.setdefault(track, []).append(dict(index=index, x=float(box[0]), y=float(box[1]),
                                                     length=float(box[3]), width=float(box[4]),
                                                     yaw=float(box[6]), world_yaw=float(box[6]) + heading,
                                                     speed=speed))
    return tracks


def worldengine_root():
    """WorldEngine checkout that provides the ego vehicle parameters."""
    configured = os.environ.get("ADAGENT_WORLDENGINE_ROOT")
    root = Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parents[2].parent / "WorldEngine"
    if not (root / "projects/SimEngine/worldengine").is_dir():
        raise FactsError(f"cannot locate the WorldEngine checkout for ego geometry: {root}")
    return root


_EGO_GEOMETRY = None


def ego_geometry():
    """Ego length, width and rear-axle-to-centre shift, taken from the scorer's parameters."""
    global _EGO_GEOMETRY
    if _EGO_GEOMETRY is None:
        sys.path.insert(0, str(worldengine_root() / "projects/SimEngine"))
        from worldengine.components.agents.vehicle_model.pacifica_vehicle import get_pacifica_parameters
        parameters = get_pacifica_parameters()
        _EGO_GEOMETRY = (float(parameters.length), float(parameters.width),
                         float((parameters.front_length - parameters.rear_length) / 2.0))
    return _EGO_GEOMETRY


def body_polygon(center_x, center_y, heading, length, width):
    """Oriented footprint of one body in world coordinates."""
    from shapely.geometry import Polygon

    if not all(math.isfinite(value) for value in (center_x, center_y, heading, length, width)):
        raise FactsError(f"unusable body geometry: centre=({center_x}, {center_y}) heading={heading} "
                         f"size={length}x{width}")
    if length <= 0.0 or width <= 0.0:
        raise FactsError(f"unusable body size {length}x{width}")
    cos, sin = math.cos(heading), math.sin(heading)
    half_length, half_width = length / 2.0, width / 2.0
    corners = ((half_length, half_width), (half_length, -half_width),
               (-half_length, -half_width), (-half_length, half_width))
    return Polygon([(center_x + corner_x * cos - corner_y * sin,
                     center_y + corner_x * sin + corner_y * cos) for corner_x, corner_y in corners])


def ego_body(frame, index):
    """Ego footprint; box positions are measured from the rear axle of this frame."""
    heading = ego_heading(frame, index)
    pose = np.asarray(frame["ego2global"], dtype=float)
    length, width, shift = ego_geometry()
    return body_polygon(float(pose[0][3]) + math.cos(heading) * shift,
                        float(pose[1][3]) + math.sin(heading) * shift,
                        heading, length, width)


def actor_body(observation, frame, index):
    heading = ego_heading(frame, index)
    pose = np.asarray(frame["ego2global"], dtype=float)
    cos, sin = math.cos(heading), math.sin(heading)
    return body_polygon(float(pose[0][3]) + observation["x"] * cos - observation["y"] * sin,
                        float(pose[1][3]) + observation["x"] * sin + observation["y"] * cos,
                        observation["world_yaw"], observation["length"], observation["width"])


def conflicting_track(tracks, step, frames):
    """Closest vehicle whose body is within the distance limit of the ego body at that frame.

    The result is the analysis target of this rule, not an officially identified
    collision or cause object.
    """
    if isinstance(step, bool) or not isinstance(step, int):
        raise FactsError(f"violation frame must be an integer, got {step!r}")
    if not 0 <= step < len(frames):
        raise FactsError(f"violation frame {step} is outside the recorded frames 0..{len(frames) - 1}")
    frame = frames[step]
    ego = ego_body(frame, step)
    chosen = []
    for name, sequence in tracks.items():
        observation = next((item for item in sequence if item["index"] == step), None)
        if observation is None:
            continue
        distance = ego.distance(actor_body(observation, frame, step))
        if not math.isfinite(distance):
            raise FactsError(f"body distance between the ego and {name} at frame {step} is not finite")
        if distance <= BODY_DISTANCE_LIMIT_M + BODY_DISTANCE_TOLERANCE_M:
            chosen.append((distance, str(name), sequence))
    if not chosen:
        return None, None
    chosen.sort(key=lambda entry: (entry[0], entry[1]))
    return chosen[0][1], chosen[0][2]


def classify_origin(first):
    x, y = first["x"], first["y"]
    relative = math.degrees((first["yaw"] + math.pi) % (2 * math.pi) - math.pi)
    aligned = abs(relative) <= 60.0 or abs(relative) >= 120.0
    if abs(y) < LANE_HALF_WIDTH_M:
        if x < 0:
            return "behind_same_lane"
        return "ahead_same_lane" if abs(relative) <= 60.0 else "oncoming"
    side = "left" if y > 0 else "right"
    if abs(y) <= ADJACENT_MAX_M and aligned:
        return side + "_adjacent"
    return side + "_arm"


def classify_action(sequence, origin):
    speeds = []
    for item in sequence:
        speed = item.get("speed")
        if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed):
            raise FactsError(f"observation at frame {item.get('index')} has no usable speed: {speed!r}")
        speeds.append(float(speed))
    if not speeds:
        raise FactsError("actor sequence holds no observation")
    if max(speeds) <= STOPPED_MPS:
        return "stop"
    unwrapped = [sequence[0]["world_yaw"]]
    for item in sequence[1:]:
        delta = item["world_yaw"] - unwrapped[-1]
        while delta > math.pi:
            delta -= 2 * math.pi
        while delta < -math.pi:
            delta += 2 * math.pi
        unwrapped.append(unwrapped[-1] + delta)
    heading = math.degrees(unwrapped[-1] - unwrapped[0])
    if heading <= -TURN_DEG:
        return "turn_right"
    if heading >= TURN_DEG:
        return "turn_left"
    if speeds[0] - min(speeds) >= 2.0:
        return "brake"
    if origin.endswith("_adjacent"):
        lateral = [abs(i["y"]) for i in sequence]
        if lateral[0] - min(lateral) >= 1.0:
            if min(lateral) <= LANE_HALF_WIDTH_M:
                return "cut_in_left" if origin.startswith("left") else "cut_in_right"
            return "squeeze_left" if origin.startswith("left") else "squeeze_right"
    return "straight"


# ------------------------------------------------------------------ outcome
def outcome(archive, token):
    try:
        table = score_file(archive)
    except FileNotFoundError as error:
        raise FactsError(f"official PDM score table is missing for {archive}: {error}") from error
    rows = {}
    with table.open(newline="") as stream:
        for row in csv.DictReader(stream):
            name = (row.get("token") or "").strip()
            if name and name != "overall_average":
                rows[name] = row
    match = [name for name in rows if name == token or name.endswith(token)]
    if len(match) != 1:
        raise FactsError(f"expected exactly one PDM row for token {token} in {table.name}, "
                         f"found {len(match)}")
    row = rows[match[0]]

    def number(key):
        try:
            return float(row[key])
        except (KeyError, TypeError, ValueError):
            raise FactsError(f"PDM row for {token} in {table.name} has no usable {key}") from None

    collision = number("no_at_fault_collisions") <= 0.0
    departure = number("drivable_area_compliance") <= 0.0
    score = number("score")
    violation = number("first_violation_step")
    if not math.isfinite(violation):
        raise FactsError(f"PDM row for {token} in {table.name} has no first_violation_step")
    if not float(violation).is_integer():
        raise FactsError(f"PDM row for {token} in {table.name} has a non-integer "
                         f"first_violation_step={violation}")
    step = int(violation)
    if collision and departure:
        kind = "collision_and_departure"
    elif collision:
        kind = "collision"
    elif departure:
        kind = "road_departure"
    elif score > 0.0:
        raise FactsError(f"scene {token} is not a zero-score failure (score={score}); pick a failure scene")
    else:
        raise FactsError(f"scene {token} scores zero with neither collision nor road departure")
    return kind, match[0], step


# ------------------------------------------------------------------ map
def map_file(map_location):
    root = os.path.join(os.environ.get("ADAGENT_MAPS_ROOT") or _default_maps_root(), map_location)
    if not os.path.isdir(root):
        raise FactsError(f"no nuPlan map for map_location {map_location} under {root}")
    versions = sorted(v for v in os.listdir(root) if os.path.isfile(os.path.join(root, v, "map.gpkg")))
    if len(versions) != 1:
        raise FactsError(f"expected exactly one map version for {map_location}, found {versions}")
    return os.path.join(root, versions[0], "map.gpkg")


def _default_maps_root():
    return os.path.join(PROJECT_ROOT, DEFAULT_MAPS_ROOT)


def utm_to_lonlat(easting, northing, bounds):
    from pyproj import Transformer
    centre_lon = (bounds[0] + bounds[2]) / 2.0
    zone = int((centre_lon + 180.0) / 6.0) + 1
    for datum in (26900, 32600):
        transformer = Transformer.from_crs(f"EPSG:{datum + zone}", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(easting, northing)
        if bounds[0] <= lon <= bounds[2] and bounds[1] <= lat <= bounds[3]:
            return lon, lat
    raise FactsError(f"ego position ({easting:.0f},{northing:.0f}) falls outside the map extent")


def metric_crs(map_crs, longitude, latitude):
    """Metric frame for map distances: the map's own when it is metric, else the local UTM zone."""
    from pyproj import CRS

    if map_crs is None:
        raise FactsError("map has no coordinate reference system")
    crs = CRS.from_user_input(map_crs)
    if crs.is_projected:
        unit = crs.axis_info[0].unit_name if crs.axis_info else ""
        if "metre" not in unit and "meter" not in unit:
            raise FactsError(f"map CRS {crs.name} is projected but not metric (unit {unit})")
        return crs
    if not crs.is_geographic:
        raise FactsError(f"map CRS {crs.name} is neither geographic nor projected")
    zone = int((longitude + 180.0) // 6.0) + 1
    return CRS.from_dict({"proj": "utm", "zone": zone, "south": latitude < 0.0})


def map_layer(gpd, path, layer):
    """Read one map layer and refuse anything that would distort geometry counts."""
    frame = gpd.read_file(path, layer=layer, engine="pyogrio", fid_as_index=True)
    if frame.empty:
        raise FactsError(f"map layer {layer} has no features")
    for name, mask in (("missing", frame.geometry.isna()),
                       ("empty", frame.geometry.is_empty),
                       ("invalid", ~frame.geometry.is_valid)):
        if bool(mask.any()):
            raise FactsError(f"map layer {layer} has {int(mask.sum())} {name} geometries")
    return frame


def map_context(map_location, conflict_xy, turn_deg):
    import geopandas as gpd
    from pyproj import Transformer
    from shapely.geometry import Point

    path = map_file(map_location)
    intersections = map_layer(gpd, path, "intersections")
    segments = map_layer(gpd, path, "road_segments")
    if intersections.crs is None:
        raise FactsError("map intersections have no coordinate reference system")
    if segments.crs is None:
        raise FactsError("map road segments have no coordinate reference system")
    extent = intersections.to_crs("EPSG:4326").total_bounds
    lonlat = utm_to_lonlat(conflict_xy[0], conflict_xy[1], extent)
    metric = metric_crs(intersections.crs, *lonlat)
    point = Point(*Transformer.from_crs("EPSG:4326", metric, always_xy=True).transform(*lonlat))
    intersections = intersections.to_crs(metric)
    segments = segments.to_crs(metric)
    distance = intersections.geometry.distance(point)
    if not np.isfinite(distance.to_numpy(dtype=float)).all():
        raise FactsError(f"map {map_location} has intersections without a usable distance to the ego")
    index = distance.idxmin()
    if float(distance.loc[index]) > INTERSECTION_RANGE_M:
        return "curve" if abs(turn_deg) >= CURVE_DEG else "straight_road"
    polygon = intersections.geometry.loc[index]
    arms = int(segments.geometry.intersects(polygon).sum())
    if arms >= 4:
        return "cross_intersection"
    if arms == 3:
        return "t_junction"
    # Zero, one or two touching road segments use the same straight/curve rule.
    return "curve" if abs(turn_deg) >= CURVE_DEG else "straight_road"


TURN_TYPE_ACTIONS = {"0": None, "1": "turn_left", "2": "turn_right"}


def map_ego_action(map_location, frames, ego):
    """Prefer the real map connector the ego traverses over raw lateral displacement.

    World-frame lateral offset cannot distinguish a lane change from a missed turn on a
    curving road. When the recorded path enters a connected lane group, use the official
    connector turn type; return None when the map cannot establish a turn.
    """
    import geopandas as gpd
    from pyproj import Transformer
    from shapely.geometry import Point

    try:
        path = map_file(map_location)
        lanes = map_layer(gpd, path, "lanes_polygons")
        connectors = map_layer(gpd, path, "lane_connectors")
        if lanes.crs is None:
            return None
        extent = lanes.to_crs("EPSG:4326").total_bounds
        positions = ego["positions"]
        lonlat = utm_to_lonlat(positions[0][0], positions[0][1], extent)
        metric = metric_crs(lanes.crs, *lonlat)
        to_metric = Transformer.from_crs("EPSG:4326", metric, always_xy=True)
        points = [Point(*to_metric.transform(*utm_to_lonlat(x, y, extent)))
                  for x, y in positions]
        lanes = lanes.to_crs(metric)
        connectors = connectors.to_crs(metric)

        def nearest_group(point):
            distance = lanes.geometry.distance(point)
            if distance.empty:
                return None
            return str(lanes.loc[distance.idxmin(), "lane_group_fid"])

        start_group = nearest_group(points[0])
        if start_group is None:
            return None
        route_groups = {nearest_group(point) for point in points[1:]}
        route_groups.discard(None)
        route_groups.discard(start_group)
        candidates = connectors[connectors["exit_lane_group_fid"].astype(str) == start_group]
        candidates = candidates[candidates["entry_lane_group_fid"].astype(str).isin(route_groups)]
        if candidates.empty:
            return None
        candidates = candidates.copy()
        candidates["_distance"] = [min(row.geometry.distance(point) for point in points)
                                   for _, row in candidates.iterrows()]
        row = candidates.sort_values("_distance").iloc[0]
        turn_type = str(row.get("turn_type_fid"))
        action = TURN_TYPE_ACTIONS.get(turn_type)
        if action:
            return action
        coords = list(row.geometry.coords)
        if len(coords) < 3:
            return None
        start_tangent = math.atan2(coords[1][1] - coords[0][1], coords[1][0] - coords[0][0])
        end_tangent = math.atan2(coords[-1][1] - coords[-2][1], coords[-1][0] - coords[-2][0])
        delta = (math.degrees(end_tangent - start_tangent) + 180.0) % 360.0 - 180.0
        if abs(delta) < TURN_DEG:
            return None
        return "turn_left" if delta > 0 else "turn_right"
    except FactsError:
        return None
    except (OSError, ValueError, TypeError, KeyError):
        return None

# ------------------------------------------------------------------ derive
def interaction(attack, ego_action):
    origin = attack["other_origin"]
    if origin == "none":
        return "single_vehicle"
    if ego_action in ("lane_change_left", "lane_change_right"):
        return "lane_change"
    if origin in ("ahead_same_lane", "behind_same_lane"):
        return "rear_end"
    if origin in ("left_arm", "right_arm", "oncoming"):
        return "crossing"
    return "side_swipe"


def ego_cause(interaction_kind, attack, ego_action):
    place = {"straight_road": "直路上", "t_junction": "丁字路口", "cross_intersection": "十字路口",
             "curve": "弯道上", "merge": "汇流处"}[attack["map_context"]]
    motion = {"straight": "直行", "turn_left": "左转", "turn_right": "右转",
              "lane_change_left": "向左变道", "lane_change_right": "向右变道"}[ego_action]
    if attack["other_origin"] == "none":
        return f"首次违规时刻{BODY_DISTANCE_LIMIT_M:g}米内无目标车辆"
    origin = {"ahead_same_lane": "同车道前方", "behind_same_lane": "同车道后方",
              "left_adjacent": "左相邻车道", "right_adjacent": "右相邻车道",
              "oncoming": "对向车道", "left_arm": "左侧支路", "right_arm": "右侧支路"}[attack["other_origin"]]
    action = {"straight": "直行", "turn_left": "左转", "turn_right": "右转",
              "cut_in_left": "从左侧切入", "cut_in_right": "从右侧切入",
              "brake": "急刹", "stop": "停止", "squeeze_left": "从左侧贴近挤压",
              "squeeze_right": "从右侧贴近挤压"}[attack["other_action"]]
    if interaction_kind == "rear_end":
        return f"在{place}{motion}时，没有对{origin}{action}的车辆及时减速"
    if interaction_kind == "lane_change":
        return f"在{place}{motion}时，没有发现{origin}{action}的车辆"
    return f"在{place}{motion}时，没有对从{origin}{action}过来的车辆让行"


def frame_images(archive, frames):
    """Frame index -> camera -> absolute image path, taken from the scene metadata."""
    base = os.path.join(archive, "WE_output/openscene_format/sensor_blobs")
    rows = []
    for index, frame in enumerate(frames):
        entry = {"index": index,
                 "time_s": round((frame["timestamp"] - frames[0]["timestamp"]) / 1e6, 3),
                 "images": {}}
        for camera, data in (frame.get("cams") or {}).items():
            relative = (data or {}).get("data_path")
            if not relative:
                raise FactsError(f"frame {index} has no {camera} data_path")
            path = os.path.join(base, relative)
            if not os.path.isfile(path):
                raise FactsError(f"missing image for {camera} frame {index}: {path}")
            entry["images"][camera] = path
        rows.append(entry)
    return rows


def scene_time(frames):
    stamp = datetime.fromtimestamp(frames[0]["timestamp"] / 1e6, tz=timezone.utc)
    return stamp.isoformat(timespec="seconds")


# ------------------------------------------------------------------ entry
def facts(archive, token):
    frames, full_token = scene_pickle(archive, token)
    ego = ego_facts(frames)
    outcome_kind, _, violation_step = outcome(archive, full_token)
    tracks = actor_sequences(frames)
    _, sequence = conflicting_track(tracks, violation_step, frames)
    conflict_xy = ego["positions"][violation_step]
    if sequence is None:
        attack = {"other_origin": "none", "other_action": "none"}
    else:
        first = sequence[0]
        origin = classify_origin(first)
        action = classify_action(sequence, origin)
        attack = {"other_origin": origin, "other_action": action}
    attack["map_context"] = map_context(frames[0]["map_location"], conflict_xy, ego["turn_deg"])
    map_action = map_ego_action(frames[0]["map_location"], frames, ego)
    ego_action = map_action or ego["action"]
    kind = interaction(attack, ego_action)
    failure = {"outcome": outcome_kind, "interaction": kind,
               "ego_cause": ego_cause(kind, attack, ego_action)}
    if map_action and map_action != ego["action"]:
        failure["recorded_ego_action"] = ego["action"]
    return {
        "status": "ok",
        "token": full_token,
        "violation_step": violation_step,
        "map_location": frames[0]["map_location"],
        "failure": failure,
        "attack": {"map_context": attack["map_context"], "ego_action": ego_action,
                   "other_origin": attack["other_origin"], "other_action": attack["other_action"]},
        "environment": {"time": scene_time(frames), "weather": None, "extra": None},
        "frame_images": frame_images(archive, frames),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    try:
        payload = facts(args.archive, args.token)
    except FactsError as error:
        json.dump({"status": "error", "reason": str(error)}, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        raise SystemExit(2)
    json.dump(payload, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
