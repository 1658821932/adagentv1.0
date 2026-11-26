"""Extract deterministic failure facts, kinematics, trajectories and BEV image.

Run with the simulator/GIS interpreter:
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
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.pdm_scores import score_file

NEARBY_RADIUS_M = 30.0
COLLISION_CONTACT_M = 0.1
DEFAULT_MAPS_ROOT = "data/navtrain_ep_per1/maps"
EGO_HISTORY_COLOR = "#ffb74d"
EGO_COLOR = "#f57c00"
OTHER_COLOR = "#1976d2"
VIOLATION_EDGE = "#ff1744"


class FactsError(RuntimeError):
    """A required input is missing or inconsistent; no fallback is permitted."""


# ------------------------------------------------------------------ scene input
def scene_pickle(archive, token):
    root = os.path.join(archive, "WE_output/openscene_format/meta_datas")
    candidates = [
        p for p in sorted(glob.glob(os.path.join(root, "*.pkl")))
        if os.path.basename(p)[:-4] == token or os.path.basename(p)[:-4].endswith(token)
    ]
    if not candidates:
        raise FactsError(f"no scene pickle for token {token} in {root}")
    if len(candidates) > 1:
        raise FactsError(f"ambiguous scene pickle for token {token}: {[os.path.basename(p) for p in candidates]}")
    with open(candidates[0], "rb") as stream:
        frames = pickle.load(stream)
    if not isinstance(frames, list) or len(frames) < 2:
        raise FactsError(f"scene pickle holds no frame list: {candidates[0]}")
    return frames, os.path.basename(candidates[0])[:-4]


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
        for position, track in enumerate(tokens):
            if names[position] not in ("vehicle", "bus", "truck"):
                continue
            if boxes.ndim != 2 or boxes.shape[0] <= position or boxes.shape[1] < 7:
                raise FactsError(f"frame {index} has no usable box for {track}")
            box = boxes[position]
            if not all(math.isfinite(value) for value in (box[0], box[1], box[3], box[4], box[5], box[6])):
                raise FactsError(f"frame {index} box for {track} has non-finite geometry")
            if box[3] <= 0.0 or box[4] <= 0.0 or box[5] <= 0.0:
                raise FactsError(f"frame {index} box for {track} has unusable size {box[3]}x{box[4]}x{box[5]}")
            tracks.setdefault(track, []).append({
                "index": index,
                "label": names[position],
                "x": float(box[0]),
                "y": float(box[1]),
                "length": float(box[3]),
                "width": float(box[4]),
                "height": float(box[5]),
                "yaw": float(box[6]),
                "world_yaw": float(box[6]) + heading,
            })
    return tracks


def worldengine_root():
    configured = os.environ.get("ADAGENT_WORLDENGINE_ROOT")
    root = Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parents[2].parent / "WorldEngine"
    if not (root / "projects/SimEngine/worldengine").is_dir():
        raise FactsError(f"cannot locate the WorldEngine checkout for ego geometry: {root}")
    return root


_EGO_GEOMETRY = None


def ego_geometry():
    global _EGO_GEOMETRY
    if _EGO_GEOMETRY is None:
        sys.path.insert(0, str(worldengine_root() / "projects/SimEngine"))
        from worldengine.components.agents.vehicle_model.pacifica_vehicle import get_pacifica_parameters
        parameters = get_pacifica_parameters()
        _EGO_GEOMETRY = (
            float(parameters.length),
            float(parameters.width),
            float((parameters.front_length - parameters.rear_length) / 2.0),
        )
    return _EGO_GEOMETRY


def _point_on_segment(point, start, end, tolerance=1e-9):
    px, py = point
    ax, ay = start
    bx, by = end
    if abs((px - ax) * (by - ay) - (py - ay) * (bx - ax)) > tolerance:
        return False
    return (min(ax, bx) - tolerance <= px <= max(ax, bx) + tolerance
            and min(ay, by) - tolerance <= py <= max(ay, by) + tolerance)


def _segment_distance(first, second):
    a, b = first
    c, d = second

    def point_segment(point, start, end):
        dx, dy = end[0] - start[0], end[1] - start[1]
        if dx == 0.0 and dy == 0.0:
            return math.hypot(point[0] - start[0], point[1] - start[1])
        t = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / (dx * dx + dy * dy)))
        return math.hypot(point[0] - (start[0] + t * dx), point[1] - (start[1] + t * dy))

    def orient(p1, p2, p3):
        val = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        return 0 if abs(val) <= 1e-9 else (1 if val > 0 else -1)

    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    if (o1 != o2 and o3 != o4) or (o1 == 0 and _point_on_segment(c, a, b)) or (o2 == 0 and _point_on_segment(d, a, b)) or (o3 == 0 and _point_on_segment(a, c, d)) or (o4 == 0 and _point_on_segment(b, c, d)):
        return 0.0
    return min(point_segment(a, c, d), point_segment(b, c, d), point_segment(c, a, b), point_segment(d, a, b))


class _OrientedRectangle:
    def __init__(self, points):
        self.points = tuple((float(x), float(y)) for x, y in points)

    def contains(self, point):
        signs = []
        for idx, start in enumerate(self.points):
            end = self.points[(idx + 1) % len(self.points)]
            if _point_on_segment(point, start, end):
                return True
            signs.append((end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0]))
        return all(v >= 0.0 for v in signs) or all(v <= 0.0 for v in signs)

    def distance(self, other):
        if any(other.contains(p) for p in self.points) or any(self.contains(p) for p in other.points):
            return 0.0
        dists = []
        for i, p1 in enumerate(self.points):
            p2 = self.points[(i + 1) % len(self.points)]
            for j, q1 in enumerate(other.points):
                q2 = other.points[(j + 1) % len(other.points)]
                dists.append(_segment_distance((p1, p2), (q1, q2)))
        return min(dists)


def body_polygon(center_x, center_y, heading, length, width):
    if not all(math.isfinite(v) for v in (center_x, center_y, heading, length, width)) or length <= 0.0 or width <= 0.0:
        raise FactsError(f"unusable body geometry: centre=({center_x}, {center_y}) size={length}x{width}")
    cos, sin = math.cos(heading), math.sin(heading)
    hl, hw = length / 2.0, width / 2.0
    corners = ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))
    return _OrientedRectangle([(center_x + cx * cos - cy * sin, center_y + cx * sin + cy * cos) for cx, cy in corners])


def ego_body(frame, index):
    heading = ego_heading(frame, index)
    pose = np.asarray(frame["ego2global"], dtype=float)
    length, width, shift = ego_geometry()
    return body_polygon(
        float(pose[0][3]) + math.cos(heading) * shift,
        float(pose[1][3]) + math.sin(heading) * shift,
        heading, length, width,
    )


def actor_body(observation, frame, index):
    heading = ego_heading(frame, index)
    pose = np.asarray(frame["ego2global"], dtype=float)
    cos, sin = math.cos(heading), math.sin(heading)
    return body_polygon(
        float(pose[0][3]) + observation["x"] * cos - observation["y"] * sin,
        float(pose[1][3]) + observation["x"] * sin + observation["y"] * cos,
        observation["world_yaw"], observation["length"], observation["width"],
    )


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
        raise FactsError(f"expected exactly one PDM row for token {token} in {table.name}, found {len(match)}")
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
    if not math.isfinite(violation) or not float(violation).is_integer():
        raise FactsError(f"PDM row for {token} has invalid first_violation_step={violation}")
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
def _default_maps_root():
    return os.path.join(PROJECT_ROOT, DEFAULT_MAPS_ROOT)


def map_file(map_location):
    root = os.path.join(os.environ.get("ADAGENT_MAPS_ROOT") or _default_maps_root(), map_location)
    if not os.path.isdir(root):
        raise FactsError(f"no nuPlan map for map_location {map_location} under {root}")
    versions = sorted(v for v in os.listdir(root) if os.path.isfile(os.path.join(root, v, "map.gpkg")))
    if len(versions) != 1:
        raise FactsError(f"expected exactly one map version for {map_location}, found {versions}")
    return os.path.join(root, versions[0], "map.gpkg")


def detect_utm_crs(easting, northing, bounds):
    from pyproj import Transformer
    centre_lon = (bounds[0] + bounds[2]) / 2.0
    zone = int((centre_lon + 180.0) / 6.0) + 1
    for datum in (26900, 32600):
        epsg = f"EPSG:{datum + zone}"
        transformer = Transformer.from_crs(epsg, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(easting, northing)
        if bounds[0] <= lon <= bounds[2] and bounds[1] <= lat <= bounds[3]:
            return epsg, (lon, lat)
    raise FactsError(f"ego position ({easting:.0f},{northing:.0f}) falls outside the map extent")


def utm_to_lonlat(easting, northing, bounds):
    _, lonlat = detect_utm_crs(easting, northing, bounds)
    return lonlat


def metric_crs(map_crs, longitude, latitude):
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
    frame = gpd.read_file(path, layer=layer, engine="pyogrio", fid_as_index=True)
    if frame.empty:
        raise FactsError(f"map layer {layer} has no features")
    for name, mask in (("missing", frame.geometry.isna()), ("empty", frame.geometry.is_empty), ("invalid", ~frame.geometry.is_valid)):
        if bool(mask.any()):
            raise FactsError(f"map layer {layer} has {int(mask.sum())} {name} geometries")
    return frame


# ------------------------------------------------------------------ derive
def frame_images(archive, frames):
    base = os.path.join(archive, "WE_output/openscene_format/sensor_blobs")
    rows = []
    for index, frame in enumerate(frames):
        entry = {"index": index, "time_s": round((frame["timestamp"] - frames[0]["timestamp"]) / 1e6, 3), "images": {}}
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
    return datetime.fromtimestamp(frames[0]["timestamp"] / 1e6, tz=timezone.utc).isoformat(timespec="seconds")


def _kinematics(frame_indices, times, points, yaws, violation):
    keep = [i for i, f in enumerate(frame_indices) if f <= violation]
    if not keep:
        raise FactsError("no observations at or before violation frame")
    pts = [points[i] for i in keep]
    ts = [times[i] for i in keep]
    ys = [yaws[i] for i in keep]
    speeds, disp = [], 0.0
    for i in range(1, len(pts)):
        d = math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        dt = ts[i] - ts[i - 1]
        disp += d
        if dt > 1e-6:
            speeds.append(d / dt)
    unwrapped = [ys[0]]
    for y in ys[1:]:
        delta = (y - unwrapped[-1] + math.pi) % (2 * math.pi) - math.pi
        unwrapped.append(unwrapped[-1] + delta)
    yaw_deg = math.degrees(unwrapped[-1] - unwrapped[0])
    tail_idx = max(0, len(unwrapped) - 3)
    tail_yaw_deg = math.degrees(unwrapped[-1] - unwrapped[tail_idx])
    max_v = max(speeds) if speeds else 0.0
    return {
        "speed_start_mps": round(speeds[0], 2) if speeds else 0.0,
        "speed_violation_mps": round(speeds[-1], 2) if speeds else 0.0,
        "displacement_m": round(disp, 2),
        "yaw_change_deg": round(yaw_deg, 1),
        "tail_yaw_deg": round(tail_yaw_deg, 1),
        "is_stationary": bool(disp < 0.5 and max_v < 0.3),
    }


def _bearing_text(dx, dy):
    if abs(dx) >= abs(dy) * 1.5:
        return "正前方" if dx >= 0 else "正后方"
    if abs(dy) >= abs(dx) * 1.5:
        return "左侧" if dy >= 0 else "右侧"
    return ("左" if dy >= 0 else "右") + ("前" if dx >= 0 else "后") + "方"


def _load_3dgs_coverage_grid(sample_root: Path):
    """Load 3DGS road_height_map valid mask + UTM world grid in pure numpy (~0.02s)."""
    import re
    cfg_path = sample_root / "asset_config.yaml"
    rhm_dir = None
    for sub in ("source_asset/road_height_map", "asset/road_height_map"):
        cand = sample_root / sub
        if (cand / "road_height_map.npy").is_file() and (cand / "sim2.json").is_file():
            rhm_dir = cand
            break
    if rhm_dir is None or not cfg_path.is_file():
        return None
    block = cfg_path.read_text().split("road_block:", 1)
    nums = re.findall(r"-\s+([-0-9.]+)", block[1])[:4] if len(block) == 2 else []
    if len(nums) < 4:
        return None
    x0, y0, x1, y1 = (float(v) for v in nums)
    recon_xy = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=float)
    heights = np.load(rhm_dir / "road_height_map.npy")
    sim2 = json.loads((rhm_dir / "sim2.json").read_text())
    rot = np.asarray(sim2["R"], dtype=float)
    trans = np.asarray(sim2["t"], dtype=float)
    scale = float(sim2["s"])
    valid = np.isfinite(heights)
    if not valid.any():
        return None
    rows, cols = heights.shape
    py, px = np.mgrid[0:rows, 0:cols]
    pts = np.stack([px, py], axis=-1) / scale - trans
    xy_world = pts @ rot + recon_xy
    return xy_world[..., 0], xy_world[..., 1], valid


def render_scene_bev(
    frames,
    tracks,
    *,
    end_frame: int,
    violation_frame: int | None = None,
    margin_m: float = 35.0,
    ahead_only: bool = False,
    radius_m: float = NEARBY_RADIUS_M,
    dpi: int = 120,
    out_bev_path: Path | None = None,
    out_traj_path: Path | None = None,
    title: str | None = None,
    sample_root: Path | None = None,
    highlight_placeable_lanes: bool = False,
):
    """Unified BEV renderer for both Tool 01 (simulation) and Tool 04 (real dataset)."""
    try:
        import geopandas as gpd
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.lines import Line2D
        from matplotlib.patches import FancyArrowPatch, Patch, Polygon as PlotPolygon
        from pyproj import Transformer
    except ImportError as error:
        raise FactsError(f"BEV rendering requires GIS/matplotlib: {type(error).__name__}") from None

    if not 0 <= end_frame < len(frames):
        raise FactsError(f"end_frame {end_frame} is outside 0..{len(frames) - 1}")
    ref_frame = violation_frame if violation_frame is not None else end_frame
    indices = list(range(0, end_frame + 1))

    map_loc = next((str(f.get("map_location")) for f in frames if f.get("map_location")), "")
    if not map_loc:
        raise FactsError("scene frames carry no map_location")
    path = map_file(map_loc)
    layers = {name: map_layer(gpd, path, name) for name in ("lanes_polygons", "baseline_paths", "road_segments", "intersections")}
    extent = layers["lanes_polygons"].to_crs("EPSG:4326").total_bounds
    pose0 = np.asarray(frames[0]["ego2global"], dtype=float)
    utm_epsg, lonlat = detect_utm_crs(float(pose0[0, 3]), float(pose0[1, 3]), extent)
    metric = metric_crs(layers["lanes_polygons"].crs, *lonlat)
    utm_to_metric = Transformer.from_crs(utm_epsg, metric, always_xy=True)
    layers = {name: frame.to_crs(metric) for name, frame in layers.items()}

    def to_metric(x, y):
        return utm_to_metric.transform(float(x), float(y))

    def to_heading(x, y, yaw):
        x0, y0 = to_metric(x, y)
        x1, y1 = to_metric(x + math.cos(yaw), y + math.sin(yaw))
        return x0, y0, math.atan2(y1 - y0, x1 - x0)

    def shape_of(poly):
        return np.asarray([to_metric(x, y) for x, y in poly.points], dtype=float)

    def center_of(poly):
        return np.asarray(poly.points, dtype=float).mean(axis=0)

    def draw_body(axis, pts, yaw, color, label, *, alpha, zorder, show_label, edgecolor="black", linewidth=0.75):
        axis.add_patch(PlotPolygon(pts, closed=True, facecolor=color, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha, zorder=zorder))
        c = pts.mean(axis=0)
        axis.add_patch(FancyArrowPatch((c[0], c[1]), (c[0] + 4.0 * math.cos(yaw), c[1] + 4.0 * math.sin(yaw)),
                                       arrowstyle="-|>", mutation_scale=8.5, linewidth=0.75, color="black", zorder=zorder + 1))
        if show_label:
            axis.text(c[0], c[1] + 1.2, label, fontsize=7.0, color=edgecolor if edgecolor != "black" else color,
                      fontweight="bold", ha="center", va="bottom", zorder=zorder + 2)

    t0 = frames[0]["timestamp"]
    frame_times = [(f["timestamp"] - t0) / 1e6 for f in frames]

    nearby, behind_count = [], 0
    for track, observations in sorted(tracks.items()):
        rows = {item["index"]: item for item in observations if 0 <= item["index"] <= end_frame}
        if not rows:
            continue
        min_dist, viol_dist, ever_ahead = math.inf, None, False
        for idx, item in rows.items():
            d = ego_body(frames[idx], idx).distance(actor_body(item, frames[idx], idx))
            if math.isfinite(d):
                min_dist = min(min_dist, d)
                if violation_frame is not None and idx == violation_frame:
                    viol_dist = d
                if item["x"] > -5.0 and d <= radius_m:
                    ever_ahead = True
        if ahead_only:
            if ever_ahead:
                nearby.append((track, rows, min_dist, viol_dist))
            elif min_dist <= radius_m:
                behind_count += 1
        elif min_dist <= radius_m:
            nearby.append((track, rows, min_dist, viol_dist))

    figure = Figure(figsize=(12, 9), dpi=dpi)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    layers["lanes_polygons"].plot(ax=axis, facecolor="#edf1f5", edgecolor="#9ca8b6", linewidth=0.55, zorder=0)
    layers["road_segments"].plot(ax=axis, facecolor="none", edgecolor="#596573", linewidth=0.8, zorder=1)
    layers["baseline_paths"].plot(ax=axis, color="#d5a21c", linewidth=0.55, alpha=0.75, zorder=2)
    layers["intersections"].boundary.plot(ax=axis, color="black", linewidth=1.6, zorder=3)

    ego_points, ego_yaws, ego_traj, body_geometry = [], [], [], []
    label_frames = {0, 3, end_frame} | {i for i in range(0, end_frame + 1, 3)}
    if violation_frame is not None:
        label_frames.add(violation_frame)

    for idx in indices:
        ego_utm = ego_body(frames[idx], idx)
        ego_shape = shape_of(ego_utm)
        c = center_of(ego_utm)
        x, y, yaw = to_heading(c[0], c[1], ego_heading(frames[idx], idx))
        is_viol = violation_frame is not None and idx == violation_frame
        is_hist = idx < 3
        fill = EGO_HISTORY_COLOR if (is_hist and not is_viol) else EGO_COLOR
        edge = VIOLATION_EDGE if is_viol else ("#e65100" if idx == 3 else "black")
        lw = 2.2 if is_viol else (1.4 if idx == 3 else 0.75)
        draw_body(axis, ego_shape, yaw, fill, f"E{idx}",
                  alpha=0.82 if is_viol else (0.55 if is_hist else 0.70),
                  zorder=10 if is_viol else (8 if is_hist else 9),
                  show_label=idx in label_frames, edgecolor=edge, linewidth=lw)
        ego_points.append((x, y))
        ego_yaws.append(yaw)
        ego_traj.append({"frame": idx, "t_s": round(frame_times[idx], 2),
                         "x": round(float(x), 3), "y": round(float(y), 3), "yaw_rad": round(float(yaw), 4)})
        body_geometry.extend(ego_shape)

    ego_kin = _kinematics(indices, [frame_times[i] for i in indices], ego_points, ego_yaws, ref_frame)
    ref_pos = indices.index(ref_frame)
    ex, ey = ego_points[ref_pos]
    eyaw = ego_yaws[ref_pos]
    cos_y, sin_y = math.cos(-eyaw), math.sin(-eyaw)

    ego_arr = np.asarray(ego_points)
    axis.plot(ego_arr[:, 0], ego_arr[:, 1], color=EGO_COLOR, linewidth=2.5, zorder=6)

    actor_records, actor_trajs = [], {}
    for number, (track, rows, min_dist, viol_dist) in enumerate(nearby, start=1):
        obs_frames = sorted(rows)
        pts, yaws, traj = [], [], []
        for idx in obs_frames:
            item = rows[idx]
            actor_utm = actor_body(item, frames[idx], idx)
            actor_shape = shape_of(actor_utm)
            c = center_of(actor_utm)
            x, y, yaw = to_heading(c[0], c[1], item["world_yaw"])
            draw_body(axis, actor_shape, yaw, OTHER_COLOR, f"V{number}" if ahead_only else "",
                      alpha=0.35, zorder=7, show_label=(ahead_only and idx == obs_frames[0]))
            pts.append((x, y))
            yaws.append(yaw)
            traj.append({"frame": idx, "t_s": round(frame_times[idx], 2),
                         "x": round(float(x), 3), "y": round(float(y), 3), "yaw_rad": round(float(yaw), 4)})
            body_geometry.extend(actor_shape)
        kin = _kinematics(obs_frames, [frame_times[i] for i in obs_frames], pts, yaws, ref_frame) if any(f <= ref_frame for f in obs_frames) else None
        rel_yaw, bearing = None, None
        if ref_frame in obs_frames:
            v_idx = obs_frames.index(ref_frame)
            rel_yaw = round(math.degrees((yaws[v_idx] - eyaw + math.pi) % (2 * math.pi) - math.pi), 1)
            gx, gy = pts[v_idx][0] - ex, pts[v_idx][1] - ey
            bearing = _bearing_text(gx * cos_y - gy * sin_y, gx * sin_y + gy * cos_y)
        first_item = rows[obs_frames[0]]
        pts_arr = np.asarray(pts)
        axis.plot(pts_arr[:, 0], pts_arr[:, 1], color=OTHER_COLOR, linewidth=1.6, alpha=0.7, zorder=6)
        vid = f"V{number}"
        actor_records.append({
            "vehicle": vid,
            "track": track,
            "label": first_item["label"],
            "size_m": [round(first_item["length"], 2), round(first_item["width"], 2), round(first_item["height"], 2)],
            "frames_observed": obs_frames,
            "min_distance_m": round(float(min_dist), 2),
            "violation_distance_m": None if viol_dist is None else round(float(viol_dist), 2),
            "bearing_at_violation": bearing,
            "relative_heading_at_violation_deg": rel_yaw,
            "kinematics": kin,
        })
        actor_trajs[vid] = traj

    has_gs_coverage = False
    cov_affine = None
    if sample_root is not None:
        cov = _load_3dgs_coverage_grid(Path(sample_root))
        if cov is not None:
            ux, uy, valid = cov
            # Map UTM grid corners via local affine approximation of to_metric
            c00 = to_metric(float(ux[0, 0]), float(uy[0, 0]))
            c01 = to_metric(float(ux[0, -1]), float(uy[0, -1]))
            c10 = to_metric(float(ux[-1, 0]), float(uy[-1, 0]))
            rows, cols = valid.shape
            u = np.linspace(0.0, 1.0, cols)[None, :]
            v = np.linspace(0.0, 1.0, rows)[:, None]
            mx = c00[0] + u * (c01[0] - c00[0]) + v * (c10[0] - c00[0])
            my = c00[1] + u * (c01[1] - c00[1]) + v * (c10[1] - c00[1])
            axis.contourf(mx, my, valid.astype(float), levels=[0.5, 1.5], colors=["#66bb6a"], alpha=0.22, zorder=4)
            has_gs_coverage = True
            mat = np.array([[c01[0] - c00[0], c10[0] - c00[0]],
                            [c01[1] - c00[1], c10[1] - c00[1]]], dtype=float)
            if abs(float(np.linalg.det(mat))) > 1e-9:
                cov_affine = (np.array(c00, dtype=float), np.linalg.inv(mat), valid)

    geom = np.asarray(body_geometry)
    low, high = geom.min(axis=0) - margin_m, geom.max(axis=0) + margin_m

    placeable_lanes = []
    if highlight_placeable_lanes and cov_affine is not None:
        origin_c00, mat_inv, valid_grid = cov_affine
        g_rows, g_cols = valid_grid.shape
        ego_hist_arr = np.asarray(ego_points[:3], dtype=float)

        def _iter_lines(geom_obj):
            if geom_obj is None or geom_obj.is_empty:
                return
            gtype = geom_obj.geom_type
            if gtype == "LineString":
                yield geom_obj
            elif gtype == "MultiLineString":
                for sub in geom_obj.geoms:
                    yield sub

        for raw_geom in layers["baseline_paths"].geometry:
            for line in _iter_lines(raw_geom):
                bx0, by0, bx1, by1 = line.bounds
                if bx1 < low[0] or bx0 > high[0] or by1 < low[1] or by0 > high[1]:
                    continue
                length = float(line.length)
                if length < 3.0:
                    continue
                dists = np.arange(0.0, length, 1.0)
                if dists[-1] < length - 0.1:
                    dists = np.append(dists, length)
                pts = np.array([[p.x, p.y] for p in (line.interpolate(float(d)) for d in dists)], dtype=float)
                in_view = (
                    (pts[:, 0] >= low[0]) & (pts[:, 0] <= high[0])
                    & (pts[:, 1] >= low[1]) & (pts[:, 1] <= high[1])
                )
                d_ego = np.min(np.linalg.norm(pts[:, None, :] - ego_hist_arr[None, :, :], axis=2), axis=1)
                uv = (pts - origin_c00[None, :]) @ mat_inv.T
                ix = np.rint(uv[:, 0] * (g_cols - 1)).astype(int)
                iy = np.rint(uv[:, 1] * (g_rows - 1)).astype(int)
                in_grid = (ix >= 0) & (ix < g_cols) & (iy >= 0) & (iy < g_rows)
                has_h = np.zeros(len(pts), dtype=bool)
                has_h[in_grid] = valid_grid[iy[in_grid], ix[in_grid]]
                ok_mask = in_view & has_h
                if not np.any(ok_mask):
                    continue
                idx_start = None
                runs = []
                for k, flag in enumerate(ok_mask):
                    if flag and idx_start is None:
                        idx_start = k
                    elif not flag and idx_start is not None:
                        runs.append((idx_start, k))
                        idx_start = None
                if idx_start is not None:
                    runs.append((idx_start, len(ok_mask)))
                for r0, r1 in runs:
                    if r1 - r0 < 2:
                        continue
                    seg = pts[r0:r1]
                    seg_steps = np.linalg.norm(np.diff(seg, axis=0), axis=1)
                    seg_cum = np.concatenate([[0.0], np.cumsum(seg_steps)])
                    if seg_cum[-1] < 3.0:
                        continue
                    axis.plot(seg[:, 0], seg[:, 1], color="#00acc1", linewidth=2.6, alpha=0.88, zorder=5)
                    placeable_lanes.append(seg)
                    next_arrow = 5.0 if seg_cum[-1] >= 6.0 else seg_cum[-1] * 0.5
                    while next_arrow < seg_cum[-1] - 0.8:
                        ai = int(np.searchsorted(seg_cum, next_arrow))
                        ai = max(1, min(ai, len(seg) - 1))
                        p_prev, p_curr = seg[ai - 1], seg[ai]
                        dvec = p_curr - p_prev
                        dnorm = float(np.linalg.norm(dvec))
                        if dnorm > 1e-4:
                            dvec = dvec / dnorm * 1.2
                            axis.annotate(
                                "",
                                xy=(p_curr[0], p_curr[1]),
                                xytext=(p_curr[0] - dvec[0], p_curr[1] - dvec[1]),
                                arrowprops=dict(arrowstyle="-|>", color="#006064", lw=1.4, mutation_scale=11),
                                zorder=5.5,
                            )
                        next_arrow += 10.0
    from matplotlib.ticker import FixedLocator, FuncFormatter
    ox_m, oy_m = to_metric(float(pose0[0, 3]), float(pose0[1, 3]))
    axis.set_xlim(low[0], high[0])
    axis.set_ylim(low[1], high[1])
    axis.set_aspect("equal", adjustable="box")
    x_maj = ox_m + np.arange(math.floor((low[0] - ox_m) / 10.0) * 10.0, math.ceil((high[0] - ox_m) / 10.0) * 10.0 + 1.0, 10.0)
    y_maj = oy_m + np.arange(math.floor((low[1] - oy_m) / 10.0) * 10.0, math.ceil((high[1] - oy_m) / 10.0) * 10.0 + 1.0, 10.0)
    x_min = ox_m + np.arange(math.floor((low[0] - ox_m) / 5.0) * 5.0, math.ceil((high[0] - ox_m) / 5.0) * 5.0 + 1.0, 5.0)
    y_min = oy_m + np.arange(math.floor((low[1] - oy_m) / 5.0) * 5.0, math.ceil((high[1] - oy_m) / 5.0) * 5.0 + 1.0, 5.0)
    axis.xaxis.set_major_locator(FixedLocator(x_maj))
    axis.yaxis.set_major_locator(FixedLocator(y_maj))
    axis.xaxis.set_minor_locator(FixedLocator(x_min))
    axis.yaxis.set_minor_locator(FixedLocator(y_min))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v - ox_m))}"))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v - oy_m))}"))
    axis.grid(which="major", color="#78909c", alpha=0.45, linewidth=0.8)
    axis.grid(which="minor", color="#b0bec5", alpha=0.30, linestyle="--", linewidth=0.5)
    axis.set_xlabel("x from E0 (m, East)")
    axis.set_ylabel("y from E0 (m, North)")
    handles = [
        Line2D([0], [0], color=EGO_HISTORY_COLOR, linewidth=3.0, label="EGO history (E0..E2)"),
        Line2D([0], [0], color=EGO_COLOR, linewidth=3.0, label=f"EGO path (E3..E{end_frame})"),
    ]
    if violation_frame is not None:
        handles.append(Line2D([0], [0], color=VIOLATION_EDGE, linewidth=2.5, label=f"EGO violation (F{violation_frame})"))
    handles.append(Line2D([0], [0], color=OTHER_COLOR, linewidth=2.2, label=f"Other vehicles ({len(nearby)})"))
    if has_gs_coverage:
        handles.append(Patch(facecolor="#66bb6a", alpha=0.35, label="3DGS coverage"))
    if placeable_lanes:
        handles.append(Line2D([0], [0], color="#00acc1", linewidth=2.6, label="Placeable lane centerlines (->)"))
    axis.legend(handles=handles, loc="upper left", fontsize=7.0, framealpha=0.9)
    axis.set_title(title or f"BEV OVERVIEW E0..E{end_frame} (0..{frame_times[end_frame]:.1f}s): {len(nearby)} vehicles")
    figure.tight_layout()

    bev_path = out_bev_path
    traj_path = out_traj_path
    created = []
    try:
        if bev_path is None:
            with tempfile.NamedTemporaryFile(prefix="tool01-bev-", suffix=".png", delete=False) as stream:
                bev_path = Path(stream.name)
                created.append(bev_path)
        else:
            bev_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(bev_path, format="png", dpi=dpi)
        if traj_path is None:
            with tempfile.NamedTemporaryFile(prefix="tool01-traj-", suffix=".json", delete=False, mode="w", encoding="utf-8") as stream:
                traj_path = Path(stream.name)
                created.append(traj_path)
                json.dump({"EGO": ego_traj, **actor_trajs}, stream, ensure_ascii=False, indent=2)
        else:
            traj_path.parent.mkdir(parents=True, exist_ok=True)
            traj_path.write_text(json.dumps({"EGO": ego_traj, **actor_trajs}, ensure_ascii=False, indent=2))
    except OSError as error:
        for p in created:
            if p.exists():
                p.unlink()
        raise FactsError(f"failed to save BEV/trajectory artifacts: {error}") from None

    origin = np.array([ox_m, oy_m])
    return {
        **({
            "placeable_lanes": [(lane - origin).tolist() for lane in placeable_lanes],
            "ego_bev_xy": (np.asarray(ego_points) - origin).tolist(),
        } if highlight_placeable_lanes else {}),
        "ego_kinematics": ego_kin,
        "nearby_vehicles": actor_records,
        "behind_only_count": behind_count,
        "eval_duration_s": round(frame_times[end_frame], 2),
        "bev_image": str(bev_path),
        "trajectories_path": str(traj_path),
    }


def _render_bev_and_kinematics(frames, tracks, raw_outcome, violation):
    if not 0 <= violation < len(frames):
        raise FactsError(f"violation frame {violation} is outside 0..{len(frames) - 1}")
    if raw_outcome == "collision_and_departure":
        ego_v = ego_body(frames[violation], violation)
        contact = any(
            ego_v.distance(actor_body(item, frames[violation], violation)) <= COLLISION_CONTACT_M
            for obs in tracks.values()
            for item in obs if item["index"] == violation
        )
        resolved_outcome = "collision" if contact else "road_departure"
    else:
        resolved_outcome = raw_outcome

    end = violation if resolved_outcome == "collision" else min(violation + 1, len(frames) - 1)
    rendered = render_scene_bev(
        frames, tracks, end_frame=end, violation_frame=violation,
        margin_m=35.0, ahead_only=False, radius_m=NEARBY_RADIUS_M, dpi=120,
    )
    return (resolved_outcome, rendered["ego_kinematics"], rendered["nearby_vehicles"],
            rendered["bev_image"], rendered["trajectories_path"])


# ------------------------------------------------------------------ entry
def facts(archive, token):
    frames, full_token = scene_pickle(archive, token)
    raw_outcome, _, violation_step = outcome(archive, full_token)
    tracks = actor_sequences(frames)
    resolved_outcome, ego_kin, nearby_vehicles, bev_image, traj_path = _render_bev_and_kinematics(
        frames, tracks, raw_outcome, violation_step
    )
    return {
        "status": "ok",
        "token": full_token,
        "violation_step": violation_step,
        "map_location": frames[0]["map_location"],
        "outcome": resolved_outcome,
        "ego_kinematics": ego_kin,
        "nearby_vehicles": nearby_vehicles,
        "bev_image": bev_image,
        "trajectories_path": traj_path,
        "environment": {"time": scene_time(frames), "weather": None, "extra": None},
        "frame_images": frame_images(archive, frames),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive")
    parser.add_argument("--token")
    parser.add_argument("--frames-json", help="Render real recorded dataset frames.json BEV for Tool 04")
    parser.add_argument("--out-bev", type=Path)
    parser.add_argument("--out-traj", type=Path)
    parser.add_argument("--margin-m", type=float, default=35.0)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--highlight-placeable-lanes", action="store_true")
    args = parser.parse_args()
    try:
        if args.frames_json:
            frames = json.loads(Path(args.frames_json).read_text())
            if not isinstance(frames, list) or len(frames) < 2:
                raise FactsError(f"frames json holds fewer than 2 frames: {args.frames_json}")
            tracks = actor_sequences(frames)
            end_frame = min(args.max_frames - 1, len(frames) - 1)
            rendered = render_scene_bev(
                frames, tracks, end_frame=end_frame, violation_frame=None,
                margin_m=args.margin_m, ahead_only=True, radius_m=35.0, dpi=120,
                out_bev_path=args.out_bev, out_traj_path=args.out_traj,
                title=f"RETRIEVED REAL SCENE BEV: E0..E{end_frame} (0..{(frames[end_frame]['timestamp'] - frames[0]['timestamp'])/1e6:.1f}s)",
                sample_root=Path(args.frames_json).resolve().parents[1],
                highlight_placeable_lanes=bool(args.highlight_placeable_lanes),
            )
            payload = {"status": "ok", "frame_count": len(frames), "eval_frames": end_frame + 1, **rendered}
        else:
            if not args.archive or not args.token:
                raise FactsError("--archive and --token are required outside --frames-json mode")
            payload = facts(args.archive, args.token)
    except (FactsError, OSError, ValueError) as error:
        json.dump({"status": "error", "reason": str(error)}, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        raise SystemExit(2)
    json.dump(payload, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    from tools.tool01_failure_analyzer.structure_facts import main as canonical_main
    canonical_main()
