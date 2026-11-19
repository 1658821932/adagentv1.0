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


def classify_interaction(origin, same_direction, entered_ego_lane, ego_changed_lane,
                         ahead_at_violation, collision_confirmed=True):
    """Resolve conflict type without turning a nearby vehicle into a collision."""
    if origin in (None, "none"):
        return "single_vehicle"
    if not collision_confirmed:
        return "near_miss"
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


def analyze(map_location, frames, ego, sequences, selected, violation, collision_confirmed=True):
    """Return measured lane-relative behavior; a target without a lane is an error."""
    from . import structure_facts as sf

    try:
        import geopandas as gpd
        from pyproj import Transformer
        from shapely.geometry import Point, Polygon
        from shapely.ops import nearest_points
    except ImportError as error:
        raise sf.FactsError(f"lane analysis requires GIS: {type(error).__name__}") from None

    try:
        path = sf.map_file(map_location)
        intersections = sf.map_layer(gpd, path, "intersections")
        lanes = sf.map_layer(gpd, path, "lanes_polygons")
        baselines = sf.map_layer(gpd, path, "baseline_paths")
        segments = sf.map_layer(gpd, path, "road_segments")
        connectors = sf.map_layer(gpd, path, "lane_connectors")
    except (OSError, ValueError, sf.FactsError) as error:
        raise sf.FactsError(f"lane analysis cannot read the map: {error}") from None

    if frames[0].get("ego2global") is None:
        raise sf.FactsError("lane analysis cannot run without the first ego pose")
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

    def lane_at(point):
        distances = lanes.geometry.distance(point)
        contained = lanes[distances <= 0.02]
        if contained.empty:
            return None
        order = contained.geometry.distance(point).sort_values().index
        row = contained.loc[order[0]]
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
    if home is None and selected is not None:
        raise sf.FactsError("ego vehicle is not inside an official lane at the first frame")
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
            if curve is None:
                raise sf.FactsError("ego lane curvature could not be measured at the junction")
            map_context = "curve" if abs(curve) >= CURVE_DEG else "straight_road"
    else:
        curve = curvature_at(current)
        if curve is None:
            raise sf.FactsError("ego lane curvature could not be measured")
        map_context = "curve" if abs(curve) >= CURVE_DEG else "straight_road"

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
        if first_lane is None:
            raise sf.FactsError(
                f"target vehicle {selected} is not inside an official lane at first observation frame {first['index']}")
        if conflict_lane is None:
            raise sf.FactsError(
                f"target vehicle {selected} is not inside an official lane at violation frame {violation}")
        if home is None or first_lane.get("tangent") is None or home.get("tangent") is None:
            raise sf.FactsError(
                f"target vehicle {selected} lacks a lane baseline needed to measure its direction")
        reference = home
        dot = first_lane["tangent"][0] * reference["tangent"][0] + first_lane["tangent"][1] * reference["tangent"][1]
        if dot < -0.5:
            other_origin = "oncoming"
        elif first_lane["group"] == reference["group"]:
            longitudinal = first["x"]
            other_origin = "ahead_same_lane" if longitudinal >= 0 else "behind_same_lane"
        else:
            side = side_of(first_lane["point"])
            if side is None:
                raise sf.FactsError(
                    f"target vehicle {selected} lane {first_lane['lane']} has no measurable left/right relation")
            source_segment = group_segments.get(first_lane["group"])
            ego_segment = group_segments.get(reference["group"])
            if source_segment is None or ego_segment is None:
                raise sf.FactsError(
                    f"target vehicle {selected} or ego lane lacks a road-segment binding")
            if source_segment != ego_segment:
                connected = connectors[
                    ((connectors["exit_lane_group_fid"].astype(str) == str(first_lane["group"]))
                     & (connectors["entry_lane_group_fid"].astype(str) == str(reference["group"])))
                    | ((connectors["entry_lane_group_fid"].astype(str) == str(first_lane["group"]))
                       & (connectors["exit_lane_group_fid"].astype(str) == str(reference["group"])))
                ]
                if connected.empty:
                    raise sf.FactsError(
                        f"target vehicle {selected} lane {first_lane['lane']} is not connected to ego lane")
                other_origin = f"{side}_arm"
            else:
                other_origin = f"{side}_adjacent"
        other_evidence["first_lane_group_fid"] = first_lane["group"]
        other_evidence["lane_direction_dot"] = round(dot, 4)
        other_evidence["observation_frame"] = first["index"]
        action = direction_action(sequence, violation)
        if action not in ("straight", "turn_left", "turn_right"):
            raise sf.FactsError(f"target vehicle {selected} has no measured direction")
        other_action = action
        if conflict_lane.get("offset") is None:
            raise sf.FactsError(f"target vehicle {selected} has no measured lane offset")
        other_lane_state = lane_keeping(conflict_lane["offset"])
        other_evidence["lane_group_fid"] = conflict_lane["group"]
        other_evidence["offset_m"] = round(conflict_lane["offset"], 3)
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
        collision_confirmed=collision_confirmed,
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
