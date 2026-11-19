"""Map-based road, lane-keeping and relative-direction facts for tool01."""
from __future__ import annotations

import math
from pathlib import Path


LANE_KEEP_TOLERANCE_M = 0.5
TURN_DEG = 25.0
CURVE_DEG = 8.0


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


def analyze(map_location, frames, ego, sequences, selected, violation, collision_confirmed=True, outcome_kind=None):
    """Resolve exact lane/intersection entities and measured conflict relations."""
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
            raise sf.FactsError(f"lane {int(row['lane_fid'])} has no baseline geometry")
        nearest = nearest_points(point, line)[1]
        station = line.project(nearest)
        span = min(0.5, max(line.length / 2.0, 1e-6))
        before = line.interpolate(max(0.0, station - span))
        after = line.interpolate(min(line.length, station + span))
        tangent = (after.x - before.x, after.y - before.y)
        norm = math.hypot(*tangent)
        if norm <= 0.0:
            raise sf.FactsError(f"lane {int(row['lane_fid'])} has no tangent at the vehicle")
        tangent = (tangent[0] / norm, tangent[1] / norm)
        delta = (point.x - nearest.x, point.y - nearest.y)
        offset = tangent[0] * delta[1] - tangent[1] * delta[0]
        return dict(group=int(row["lane_group_fid"]), lane=int(row["lane_fid"]), line=line,
                    offset=float(offset), tangent=tangent, point=nearest)

    def intersection_at(point):
        distances = intersections.geometry.distance(point)
        contained = intersections[distances <= 0.02]
        if contained.empty:
            return None
        return dict(fids=[str(value) for value in contained.index],
                    geometry=contained.geometry.unary_union)

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

    ego_entities = []
    for index in range(violation + 1):
        point = metric_point(*ego["positions"][index])
        lane = lane_at(point)
        intersection = intersection_at(point)
        if lane is None and intersection is None:
            if index != violation or outcome_kind not in ("road_departure", "collision_and_departure"):
                raise sf.FactsError(
                    f"ego vehicle is outside both official lanes and intersections at frame {index}")
        ego_entities.append(dict(frame=index, lane=lane, intersection=intersection))

    current = ego_entities[violation]["lane"]
    violation_intersection = ego_entities[violation]["intersection"]
    route_lanes = [entry["lane"] for entry in ego_entities if entry["lane"] is not None]
    home = route_lanes[0] if route_lanes else None
    last_lane = route_lanes[-1] if route_lanes else None
    ego_off_lane = current is None and violation_intersection is None

    def lane_curvature(lane):
        line = lane.get("line") if lane else None
        if line is None or line.length < 2.0:
            raise sf.FactsError("ego lane has no measurable baseline")
        coords = list(line.coords)
        headings = [math.atan2(coords[i][1] - coords[i-1][1], coords[i][0] - coords[i-1][0])
                    for i in range(1, len(coords))]
        if not headings:
            raise sf.FactsError("ego lane baseline has no direction")
        unwrapped = [headings[0]]
        for value in headings[1:]:
            delta = value - unwrapped[-1]
            while delta > math.pi: delta -= 2 * math.pi
            while delta < -math.pi: delta += 2 * math.pi
            unwrapped.append(unwrapped[-1] + delta)
        return math.degrees(unwrapped[-1] - unwrapped[0])

    lane_for_state = current or last_lane
    if violation_intersection is not None:
        polygon = violation_intersection["geometry"]
        arms = int(segments.geometry.intersects(polygon).sum())
        if arms == 3:
            map_context = "t_junction"
        elif arms >= 4:
            map_context = "cross_intersection"
        elif lane_for_state is not None:
            map_context = "curve" if abs(lane_curvature(lane_for_state)) >= sf.CURVE_DEG else "straight_road"
        else:
            raise sf.FactsError("ego is inside a non-junction intersection without a lane for shape analysis")
    else:
        if lane_for_state is None:
            raise sf.FactsError("ego is outside an intersection and has no route lane")
        map_context = "curve" if abs(lane_curvature(lane_for_state)) >= sf.CURVE_DEG else "straight_road"

    if current is not None:
        ego_lane_state = lane_keeping(current["offset"])
    elif ego_off_lane and last_lane is not None:
        violation_point = metric_point(*ego["positions"][violation])
        nearest = nearest_points(violation_point, last_lane["line"])[1]
        tangent = last_lane["tangent"]
        delta = (violation_point.x - nearest.x, violation_point.y - nearest.y)
        ego_lane_state = lane_keeping(tangent[0] * delta[1] - tangent[1] * delta[0])
    elif violation_intersection is not None:
        ego_lane_state = "unknown"
    else:
        raise sf.FactsError("ego lane keeping cannot be measured")

    def side_of(point_or_geometry, reference):
        if reference is None or reference.get("line") is None or reference.get("tangent") is None:
            raise sf.FactsError("ego reference lane has no baseline")
        point = point_or_geometry.centroid if hasattr(point_or_geometry, "centroid") else point_or_geometry
        nearest = nearest_points(point, reference["line"])[1]
        tangent = reference["tangent"]
        delta = (point.x - nearest.x, point.y - nearest.y)
        cross = tangent[0] * delta[1] - tangent[1] * delta[0]
        if cross == 0.0:
            raise sf.FactsError("target has no measurable left/right relation to ego lane")
        return "left" if cross > 0 else "right"

    overlapping_groups = []
    if home is not None:
        home_segment = group_segments.get(home["group"])
        if home_segment is None:
            raise sf.FactsError("ego lane has no road-segment binding")
        violation_point = metric_point(*ego["positions"][violation])
        ego_length, ego_width, _ = sf.ego_geometry()
        ego_polygon = body(violation_point, ego["yaw"][violation], ego_length, ego_width)
        for group, rows in lanes.groupby("lane_group_fid"):
            if int(group) == home["group"] or group_segments.get(int(group)) != home_segment:
                continue
            geometry = rows.geometry.unary_union
            overlap = ego_polygon.intersection(geometry).area
            if overlap > 0.01:
                side = side_of(geometry, home)
                overlapping_groups.append((int(group), float(overlap), side))
        if overlapping_groups:
            overlapping_groups.sort(key=lambda item: (-item[1], item[0]))
            ego_lane_state = overlapping_groups[0][2] + "_offset"

    other_origin = "none"
    other_action = "none"
    other_lane_state = "none"
    other_evidence = {}
    if selected and sequences.get(selected):
        sequence = sequences[selected]
        through = [item for item in sequence if item["index"] <= violation]
        if not through:
            raise sf.FactsError(f"target vehicle {selected} has no observations through violation")
        first = through[0]
        located = []
        unmapped = []
        for item in through:
            frame_index = item["index"]
            ego_center = ego["positions"][frame_index]
            heading = sf.ego_heading(frames[frame_index], frame_index)
            world_x = ego_center[0] + item["x"] * math.cos(heading) - item["y"] * math.sin(heading)
            world_y = ego_center[1] + item["x"] * math.sin(heading) + item["y"] * math.cos(heading)
            point = metric_point(world_x, world_y)
            lane = lane_at(point)
            intersection = intersection_at(point)
            if lane is None and intersection is None:
                unmapped.append(frame_index)
            located.append(dict(item=item, lane=lane, intersection=intersection, point=point))
        if unmapped:
            if collision_confirmed:
                raise sf.FactsError(
                    f"target vehicle {selected} is outside both official lanes and intersections "
                    f"at frames {unmapped}")
            # A non-collision nearest body outside the mapped road network is not a
            # valid interaction target. Keep the single-vehicle failure explicit.
            located = []
        if located:
            first_lane_entry = next((row for row in located if row["lane"] is not None), None)
            conflict_row = next((row for row in located if row["item"]["index"] == violation), None)
            if conflict_row is None:
                raise sf.FactsError(f"target vehicle {selected} has no observation at violation frame")
            conflict_lane = conflict_row["lane"]
            conflict_intersection = conflict_row["intersection"]
            reference = current or last_lane
            if reference is None:
                raise sf.FactsError(f"ego has no lane reference for target vehicle {selected}")
            ego_heading = sf.ego_heading(frames[violation], violation)
            relative_heading = ((conflict_row["item"]["world_yaw"] - ego_heading + math.pi)
                                % (2 * math.pi) - math.pi)
            heading_dot = math.cos(relative_heading)
            if heading_dot < 0.0:
                other_origin = "oncoming"
            elif conflict_lane is not None and conflict_lane["group"] == reference["group"]:
                other_origin = ("ahead_same_lane"
                                if conflict_row["item"]["x"] >= 0 else "behind_same_lane")
            elif conflict_lane is not None:
                side = side_of(conflict_lane["point"], reference)
                source_segment = group_segments.get(conflict_lane["group"])
                ego_segment = group_segments.get(reference["group"])
                if source_segment is None or ego_segment is None:
                    raise sf.FactsError("target or ego lane has no road-segment binding")
                if source_segment == ego_segment:
                    other_origin = f"{side}_adjacent"
                else:
                    connected = connectors[
                        ((connectors["exit_lane_group_fid"].astype(str) == str(conflict_lane["group"]))
                         & (connectors["entry_lane_group_fid"].astype(str) == str(reference["group"])))
                        | ((connectors["entry_lane_group_fid"].astype(str) == str(conflict_lane["group"]))
                           & (connectors["exit_lane_group_fid"].astype(str) == str(reference["group"])))
                    ]
                    if connected.empty:
                        raise sf.FactsError("target lane is not topologically connected to the ego lane")
                    other_origin = f"{side}_arm"
            elif conflict_intersection is not None:
                side = side_of(conflict_row["point"], reference)
                other_origin = f"{side}_arm"
            else:
                raise sf.FactsError("target relation cannot be measured from map entities")
            other_action = direction_action(sequence, violation)
            if other_action == "unknown":
                raise sf.FactsError(f"target vehicle {selected} has no measured direction")
            if conflict_lane is not None:
                other_lane_state = lane_keeping(conflict_lane["offset"])
            ego_lane_groups = [entry["lane"]["group"] for entry in ego_entities
                               if entry["lane"] is not None]
            target_lane_groups = [row["lane"]["group"] for row in located if row["lane"] is not None]
            target_entered_ego_lane = any(group in ego_lane_groups for group in target_lane_groups[1:])
            ego_changed_lane = any(group != reference["group"] for group in ego_lane_groups)
            first_lane = first_lane_entry["lane"] if first_lane_entry else None
            other_evidence = dict(
                first_lane_group_fid=None if first_lane is None else first_lane["group"],
                conflict_heading_dot=round(heading_dot, 6),
                observation_frame=first["index"],
                entered_ego_lane_before_violation=target_entered_ego_lane,
                ego_changed_lane_before_violation=ego_changed_lane,
                ahead_at_violation=bool(conflict_row["item"]["x"] >= 0),
                lane_group_fid=None if conflict_lane is None else conflict_lane["group"],
                offset_m=None if conflict_lane is None else round(conflict_lane["offset"], 3),
                conflict_intersection_fids=(None if conflict_intersection is None
                                            else conflict_intersection["fids"]),
            )

    same_direction = other_evidence.get("conflict_heading_dot", 0.0) > 0
    interaction = classify_interaction(
        other_origin, same_direction,
        other_evidence.get("entered_ego_lane_before_violation"),
        other_evidence.get("ego_changed_lane_before_violation"),
        other_evidence.get("ahead_at_violation"),
        collision_confirmed=collision_confirmed,
    )
    if interaction is None:
        raise sf.FactsError("interaction type could not be measured")

    # Render one self-contained BEV sequence for semantic verification. It is an
    # evidence artifact for VLM classification, not a second source of geometry.
    bev_path = None
    try:
        import tempfile
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.patches import Polygon as PlotPolygon, FancyArrowPatch
        ego_points = [metric_point(*point) for point in ego["positions"][:violation + 1]]
        target_rows = located if selected and located else []
        target_points = [row["point"] for row in target_rows]
        all_points = ego_points + target_points
        min_x = min(point.x for point in all_points); max_x = max(point.x for point in all_points)
        min_y = min(point.y for point in all_points); max_y = max(point.y for point in all_points)
        pad = max(8.0, 0.15 * max(max_x - min_x, max_y - min_y))
        figure = Figure(figsize=(12, 9), dpi=150)
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        lanes.plot(ax=axis, facecolor="#edf1f5", edgecolor="#657080", linewidth=0.8)
        baselines.plot(ax=axis, color="#d49a00", linewidth=0.55, alpha=0.75)
        intersections.boundary.plot(ax=axis, color="black", linewidth=1.8)
        ego_length, ego_width, _ = sf.ego_geometry()
        axis.plot([point.x for point in ego_points], [point.y for point in ego_points],
                  color="#1976d2", linewidth=3, label="EGO history")
        for index, point in enumerate(ego_points):
            heading = ego["yaw"][index]
            ego_shape = body(metric_point(*ego["positions"][index]), heading, ego_length, ego_width)
            color = "#d32f2f" if index == violation else "#00897b"
            axis.add_patch(PlotPolygon(list(ego_shape.exterior.coords), closed=True, facecolor=color,
                                       edgecolor="black", linewidth=0.8, alpha=0.72, zorder=8))
            axis.add_patch(FancyArrowPatch((point.x, point.y),
                                           (point.x + math.cos(heading) * 4.5,
                                            point.y + math.sin(heading) * 4.5),
                                           arrowstyle="-|>", mutation_scale=9, color="black",
                                           linewidth=0.9, zorder=10))
            axis.text(point.x, point.y + 1.6, str(index), fontsize=6.5, color=color, zorder=11)
        if target_rows:
            axis.plot([row["point"].x for row in target_rows],
                      [row["point"].y for row in target_rows],
                      color="#f57c00", linewidth=3, label="OTHER history")
            for row in target_rows:
                item = row["item"]
                actor = sf.actor_body(item, frames[item["index"]], item["index"])
                actor_polygon = Polygon([metric_point(x, y) for x, y in actor.points])
                color = "#d32f2f" if item["index"] == violation else "#f57c00"
                axis.add_patch(PlotPolygon(list(actor_polygon.exterior.coords), closed=True,
                                           facecolor=color, edgecolor="black", linewidth=0.8,
                                           alpha=0.72, zorder=9))
                center = actor_polygon.centroid
                axis.add_patch(FancyArrowPatch(
                    (center.x, center.y),
                    (center.x + math.cos(item["world_yaw"]) * 5,
                     center.y + math.sin(item["world_yaw"]) * 5),
                    arrowstyle="-|>", mutation_scale=9, color="black", linewidth=0.9,
                    zorder=11))
                axis.text(center.x, center.y + 1.6, str(item["index"]), fontsize=6.5,
                          color=color, zorder=12)
        axis.set_xlim(min_x - pad, max_x + pad)
        axis.set_ylim(min_y - pad, max_y + pad)
        axis.set_aspect("equal")
        axis.grid(alpha=0.12)
        axis.legend(loc="best")
        axis.set_title("BEV route evidence: ego, target, headings, lanes and violation frame")
        figure.tight_layout()
        with tempfile.NamedTemporaryFile(prefix="tool01-bev-", suffix=".png", delete=False) as stream:
            bev_path = Path(stream.name)
        figure.savefig(bev_path, format="png")
    except (ImportError, OSError, ValueError, TypeError, KeyError) as error:
        if bev_path is not None:
            bev_path.unlink()
        raise sf.FactsError(f"cannot render BEV evidence: {error}") from None

    return {
        "available": True,
        "reason": None,
        "map_context": map_context,
        "ego_lane_keeping": ego_lane_state,
        "other_origin": other_origin,
        "other_action": other_action,
        "other_lane_keeping": other_lane_state,
        "interaction": interaction,
        "bev_image": str(bev_path),
        "evidence": {
            "home_lane_group_fid": None if home is None else home["group"],
            "violation_lane_group_fid": None if current is None else current["group"],
            "ego_lane_offset_m": None if current is None else current["offset"],
            "ego_entities": [
                dict(frame=entry["frame"],
                     lane=None if entry["lane"] is None else dict(
                         lane_fid=entry["lane"]["lane"], lane_group_fid=entry["lane"]["group"]),
                     intersections=None if entry["intersection"] is None else entry["intersection"]["fids"])
                for entry in ego_entities
            ],
            "ego_overlapping_lane_groups": overlapping_groups,
            "other": other_evidence,
        },
    }
