"""Map-based road, lane-keeping and relative-direction facts for tool01."""
from __future__ import annotations

import math


LANE_KEEP_TOLERANCE_M = 0.5
TURN_DEG = 25.0
CURVE_DEG = 8.0
INTERSECTION_RANGE_M = 40.0


def lane_keeping(offset_m):
    """Classify lateral position relative to a vehicle's own lane centerline."""
    if not isinstance(offset_m, (int, float)) or isinstance(offset_m, bool) or not math.isfinite(offset_m):
        return "unknown"
    if offset_m > LANE_KEEP_TOLERANCE_M:
        return "left_offset"
    if offset_m < -LANE_KEEP_TOLERANCE_M:
        return "right_offset"
    return "keep"


def direction_action(sequence, through=None):
    """Straight / left / right from heading only; speed changes never alter it."""
    selected = [item for item in sequence if through is None or item["index"] <= through]
    if not selected:
        return "unknown"
    unwrapped = [selected[0]["world_yaw"]]
    for item in selected[1:]:
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
    return "straight"


def classify_interaction(origin, same_direction, entered_ego_lane, ego_changed_lane, ahead_at_violation):
    """Resolve conflict type without treating any lateral offset as a cut-in."""
    if origin == "none":
        return "single_vehicle"
    if origin == "oncoming" or (same_direction is False and origin not in ("left_arm", "right_arm")):
        return "opposite_direction"
    if origin in ("left_arm", "right_arm"):
        return "crossing"
    if origin in ("ahead_same_lane", "behind_same_lane"):
        return "rear_end"
    if origin in ("left_adjacent", "right_adjacent"):
        if same_direction and entered_ego_lane and not ego_changed_lane and ahead_at_violation:
            return "cut_in"
        return "side_swipe"
    return None


def unavailable(reason):
    return {
        "available": False,
        "reason": reason,
        "map_context": None,
        "ego_lane_keeping": "unknown",
        "other_origin": None,
        "other_action": None,
        "other_lane_keeping": "unknown",
        "interaction": None,
        "evidence": {"reason": reason},
    }


def analyze(map_location, frames, ego, sequences, selected, violation):
    """Return lane-relative behavior at the violation, or an explicit fallback."""
    try:
        import geopandas as gpd
        from pyproj import Transformer
        from shapely.geometry import Point, Polygon
        from shapely.ops import nearest_points
    except ImportError as error:
        return unavailable(f"GIS dependencies unavailable: {type(error).__name__}")

    # Import lazily so structure_facts can call this module without a cycle.
    from . import structure_facts as sf

    try:
        path = sf.map_file(map_location)
        intersections = sf.map_layer(gpd, path, "intersections")
        lanes = sf.map_layer(gpd, path, "lanes_polygons")
        baselines = sf.map_layer(gpd, path, "baseline_paths")
        segments = sf.map_layer(gpd, path, "road_segments")
        connectors = sf.map_layer(gpd, path, "lane_connectors")
    except (OSError, ValueError, sf.FactsError) as error:
        return unavailable(f"map unavailable: {error}")

    if frames[0].get("ego2global") is None:
        return unavailable("scene has no ego pose")
    extent = intersections.to_crs("EPSG:4326").total_bounds
    lonlat = sf.utm_to_lonlat(ego["positions"][violation][0], ego["positions"][violation][1], extent)
    metric = sf.metric_crs(lanes.crs, *lonlat)
    to_metric = Transformer.from_crs("EPSG:4326", metric, always_xy=True)

    def metric_point(x, y):
        return Point(*to_metric.transform(*sf.utm_to_lonlat(x, y, extent)))

    lanes = lanes.to_crs(metric)
    baselines = baselines.to_crs(metric)
    intersections = intersections.to_crs(metric)
    segments = segments.to_crs(metric)
    connectors = connectors.to_crs(metric)
    lines = {}
    for _, row in baselines.iterrows():
        lane_fid = row.get("lane_fid")
        if lane_fid is None or isinstance(lane_fid, float) and not math.isfinite(lane_fid):
            continue
        line = row.geometry
        if not hasattr(line, "length") or line.is_empty:
            continue
        key = int(lane_fid)
        if key not in lines or line.length > lines[key].length:
            lines[key] = line

    def lane_at(point, headings=None):
        distances = lanes.geometry.distance(point)
        contained = lanes[distances <= 0.02]
        candidates = contained if len(contained) else lanes[distances <= 2.0]
        if candidates.empty:
            return None
        order = candidates.geometry.distance(point).sort_values().index
        row = candidates.loc[order[0]]
        line = lines.get(int(row["lane_fid"]))
        if line is None:
            return dict(group=int(row["lane_group_fid"]), lane=int(row["lane_fid"]),
                        line=None, offset=None, tangent=None)
        nearest = nearest_points(point, line)[1]
        station = line.project(nearest)
        span = min(0.5, max(line.length / 2.0, 1e-6))
        before = line.interpolate(max(0.0, station - span))
        after = line.interpolate(min(line.length, station + span))
        tangent = (after.x - before.x, after.y - before.y)
        norm = math.hypot(*tangent)
        if norm <= 0.0:
            return dict(group=int(row["lane_group_fid"]), lane=int(row["lane_fid"]),
                        line=line, offset=None, tangent=None)
        tangent = (tangent[0] / norm, tangent[1] / norm)
        delta = (point.x - nearest.x, point.y - nearest.y)
        offset = tangent[0] * delta[1] - tangent[1] * delta[0]
        return dict(group=int(row["lane_group_fid"]), lane=int(row["lane_fid"]),
                    line=line, offset=float(offset), tangent=tangent, point=nearest)

    def body(point, heading, length, width):
        cos, sin = math.cos(heading), math.sin(heading)
        half_length, half_width = length / 2.0, width / 2.0
        corners = ((half_length, half_width), (half_length, -half_width),
                   (-half_length, -half_width), (-half_length, half_width))
        return Polygon([(point.x + cx * cos - cy * sin, point.y + cx * sin + cy * cos)
                        for cx, cy in corners])

    group_segments = {}
    for _, row in segments.iterrows():
        raw = row.get("lane_group_fids")
        values = raw.split(",") if isinstance(raw, str) else raw if isinstance(raw, (list, tuple)) else []
        for value in values:
            try:
                group_segments[int(float(str(value).strip()))] = str(row.name)
            except (TypeError, ValueError):
                continue

    ego_length, ego_width, _ = sf.ego_geometry()
    home_point = metric_point(*ego["positions"][0])
    home = lane_at(home_point)
    violation_point = metric_point(*ego["positions"][violation])
    current = lane_at(violation_point)
    ego_polygon = body(violation_point, ego["yaw"][violation], ego_length, ego_width)
    ego_segment = group_segments.get(home["group"]) if home is not None else None

    def side_of(point_or_geometry):
        if home is None or home.get("line") is None or home.get("tangent") is None:
            return None
        if hasattr(point_or_geometry, "centroid"):
            point = point_or_geometry.centroid
        else:
            point = point_or_geometry
        nearest = nearest_points(point, home["line"])[1]
        tangent = home["tangent"]
        delta = (point.x - nearest.x, point.y - nearest.y)
        cross = tangent[0] * delta[1] - tangent[1] * delta[0]
        return "left" if cross > 0 else "right" if cross < 0 else None

    ego_lane_state = "unknown"
    if current is not None and current.get("offset") is not None:
        ego_lane_state = lane_keeping(current["offset"])
    overlapping_groups = []
    if home is not None:
        for group, rows in lanes.groupby("lane_group_fid"):
            if int(group) == home["group"]:
                continue
            # Lanes from a crossing road can intersect the ego body at a junction;
            # they do not describe lateral position in the ego's current road.
            if ego_segment is not None and group_segments.get(int(group)) != ego_segment:
                continue
            geometry = rows.geometry.unary_union
            overlap = ego_polygon.intersection(geometry).area
            if overlap <= 0.01:
                continue
            side = side_of(geometry)
            overlapping_groups.append((int(group), float(overlap), side))
        if overlapping_groups:
            overlapping_groups.sort(key=lambda item: (-item[1], item[0]))
            dominant_side = overlapping_groups[0][2]
            if dominant_side:
                ego_lane_state = f"{dominant_side}_offset"

    def curvature_at(lane):
        line = (lane or {}).get("line")
        if line is None or line.length < 2.0:
            return None
        station = line.project((lane or {}).get("point") or line.interpolate(line.length / 2.0))
        span = min(10.0, line.length / 3.0)
        first = line.interpolate(max(0.0, station - span))
        centre = line.interpolate(station)
        last = line.interpolate(min(line.length, station + span))
        first_angle = math.atan2(centre.y - first.y, centre.x - first.x)
        last_angle = math.atan2(last.y - centre.y, last.x - centre.x)
        delta = (last_angle - first_angle + math.pi) % (2 * math.pi) - math.pi
        return math.degrees(delta)

    map_context = "straight_road"
    nearest_intersection = float(intersections.geometry.distance(violation_point).min())
    if nearest_intersection <= INTERSECTION_RANGE_M:
        index = intersections.geometry.distance(violation_point).idxmin()
        polygon = intersections.geometry.loc[index]
        arms = int(segments.geometry.intersects(polygon).sum())
        if arms >= 4:
            map_context = "cross_intersection"
        elif arms == 3:
            map_context = "t_junction"
        else:
            curve = curvature_at(current)
            map_context = "curve" if curve is not None and abs(curve) >= CURVE_DEG else "straight_road"
    else:
        curve = curvature_at(current)
        map_context = "curve" if curve is not None and abs(curve) >= CURVE_DEG else "straight_road"

    other_origin = None
    other_action = None
    other_lane_state = "unknown"
    other_evidence = {}
    if selected and sequences.get(selected):
        sequence = sequences[selected]
        through = [item for item in sequence if item["index"] <= violation]
        first = through[0] if through else sequence[0]
        # Every actor point is interpreted against the ego pose of the same frame.
        located = []
        ego_lanes = {}
        for item in through:
            frame_index = item["index"]
            ego_center = ego["positions"][frame_index]
            heading = sf.ego_heading(frames[frame_index], frame_index)
            ego_lanes[frame_index] = lane_at(metric_point(*ego_center))
            world_x = ego_center[0] + item["x"] * math.cos(heading) - item["y"] * math.sin(heading)
            world_y = ego_center[1] + item["x"] * math.sin(heading) + item["y"] * math.cos(heading)
            located.append((item, lane_at(metric_point(world_x, world_y))))
        first_lane = located[0][1]
        conflict_first = next((item for item in sequence if item["index"] == violation), None)
        conflict_lane = next((lane for item, lane in located if item["index"] == violation), None)
        reference = home or current
        if first_lane and reference and first_lane.get("tangent") and reference.get("tangent"):
            dot = first_lane["tangent"][0] * reference["tangent"][0] + first_lane["tangent"][1] * reference["tangent"][1]
            if dot < -0.5:
                other_origin = "oncoming"
            elif first_lane["group"] == reference["group"]:
                longitudinal = first["x"]
                other_origin = "ahead_same_lane" if longitudinal >= 0 else "behind_same_lane"
            else:
                side = side_of(first_lane.get("point")) if first_lane.get("point") else None
                source_segment = group_segments.get(first_lane["group"])
                ego_segment = group_segments.get(reference["group"])
                connected = connectors[
                    ((connectors["exit_lane_group_fid"].astype(str) == str(first_lane["group"]))
                     & (connectors["entry_lane_group_fid"].astype(str) == str(reference["group"])))
                    | ((connectors["entry_lane_group_fid"].astype(str) == str(first_lane["group"]))
                       & (connectors["exit_lane_group_fid"].astype(str) == str(reference["group"])))
                ] if len(connectors) else connectors
                if source_segment is not None and ego_segment is not None and source_segment != ego_segment:
                    other_origin = (f"{side}_arm" if side else sf.classify_origin(first)) if len(connected) else sf.classify_origin(first)
                elif side:
                    other_origin = f"{side}_adjacent"
                else:
                    other_origin = sf.classify_origin(first)
            other_evidence["first_lane_group_fid"] = first_lane["group"]
            other_evidence["lane_direction_dot"] = round(dot, 4)
            other_evidence["observation_frame"] = first["index"]
        else:
            other_origin = sf.classify_origin(first)
        action = direction_action(sequence, violation)
        other_action = action if action in ("straight", "turn_left", "turn_right") else "straight"
        offset_lane = conflict_lane or first_lane
        if offset_lane and offset_lane.get("offset") is not None:
            other_lane_state = lane_keeping(offset_lane["offset"])
            other_evidence["lane_group_fid"] = offset_lane["group"]
            other_evidence["offset_m"] = round(offset_lane["offset"], 3)
        ego_current_group = current["group"] if current is not None else None
        target_entered_ego_lane = any(
            lane is not None and ego_lanes.get(item["index"]) is not None
            and lane["group"] == ego_lanes[item["index"]]["group"]
            for item, lane in located[1:]
        )
        ego_changed_lane = bool(home is not None and any(
            lane is not None and lane["group"] != home["group"] for lane in ego_lanes.values()))
        other_evidence["entered_ego_lane_before_violation"] = target_entered_ego_lane
        other_evidence["ego_changed_lane_before_violation"] = ego_changed_lane
        other_evidence["ahead_at_violation"] = bool(conflict_first and conflict_first["x"] >= 0)

    direction_dot = other_evidence.get("lane_direction_dot")
    same_direction = None if direction_dot is None else direction_dot > 0.5
    interaction = classify_interaction(
        other_origin, same_direction,
        other_evidence.get("entered_ego_lane_before_violation"),
        other_evidence.get("ego_changed_lane_before_violation"),
        other_evidence.get("ahead_at_violation"),
    )

    return {
        "available": True,
        "reason": None,
        "map_context": map_context,
        "ego_lane_keeping": ego_lane_state,
        "other_origin": other_origin,
        "other_action": other_action,
        "other_lane_keeping": other_lane_state,
        "interaction": interaction,
        "evidence": {
            "home_lane_group_fid": None if home is None else home["group"],
            "violation_lane_group_fid": None if current is None else current["group"],
            "ego_lane_offset_m": None if current is None else current.get("offset"),
            "ego_overlapping_lane_groups": overlapping_groups,
            "nearest_intersection_m": nearest_intersection,
            "other": other_evidence,
        },
    }
