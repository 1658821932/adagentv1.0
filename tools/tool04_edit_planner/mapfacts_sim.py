"""Raw nuPlan map facts for tool04 (simulator interpreter).

Joins follow the store's real semantics: ids are normalised through map_ids.key because
baseline columns are float (NULLs) while lane/connector ids are integers; a connector's
baseline is joined on the connector's own fid; and, per the nuPlan SDK, a lane connector
runs from its exit lane to its entry lane (incoming_edges=exit_lane_fid,
outgoing_edges=entry_lane_fid). Turn direction uses the two real end tangents, never a
single overall heading.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path("/inspire/hdd3/project/intelligent-driving-agent/public/workspace/ghc/adagent")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "projects" / "SimEngine"))

import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points, transform as transform_geometry

from tools.tool01_failure_analyzer.structure_facts import ego_heading as validate_ego_pose, map_file, metric_crs, utm_to_lonlat
from tools.tool04_edit_planner.map_ids import (arm_position, connected_intersection_regions,
                                               connector_paths_cross, key,
                                               origin_from_arm, tangent_deg, turn_from_tangents)

from tools.tool04_edit_planner.route_junction import route_junctions

UNKNOWN = dict(road="unknown", lanes="unknown", space="unknown")
JUNCTION_CONTEXTS = ("t_junction", "cross_intersection")
MIN_PLACEMENT_WIDTH_M = 3.0
CONFLICT_PROXIMITY_M = 4.0


def _coords(geometry):
    """Coordinates of a linear geometry, or of the largest ring of an areal one."""
    if geometry is None or geometry.is_empty:
        return []
    kind = geometry.geom_type
    if kind in ("LineString", "LinearRing"):
        return list(geometry.coords)
    if kind == "MultiLineString":
        geometry = max(geometry.geoms, key=lambda item: item.length)
        return list(geometry.coords)
    if kind == "Polygon":
        return list(geometry.exterior.coords)
    if kind == "MultiPolygon":
        geometry = max(geometry.geoms, key=lambda item: item.area)
        return list(geometry.exterior.coords)
    if hasattr(geometry, "coords"):
        return list(geometry.coords)
    return []


def _nonjunction_facts(context, attack, frames, ego_path, lanes, connectors, baselines,
                       group_connectors, from_map, map_location, path):
    """Map-only lane checks for straight_road, curve and merge targets."""
    route = LineString(ego_path) if len(ego_path) > 1 else None
    if route is None:
        return {"checks": dict(UNKNOWN), "facts": {"selection_policy": "no_junction_target",
                "map_location": map_location, "map_file": str(path), "target_junction_fid": None,
                "frame_target_distance": []},
                "reasons": ["The recorded route has fewer than two poses."],
                "evidence_limits": ["Lane continuity needs at least two ego poses."],
                "retrieval_suggestion": None}

    lanes = lanes.copy()
    baselines = baselines.copy()
    group_connectors = group_connectors.copy()
    for frame in (lanes, baselines, group_connectors):
        frame["geometry"] = frame.geometry.apply(lambda geometry: transform_geometry(from_map.transform, geometry))

    lane_direction = {}
    for _, row in baselines.iterrows():
        lane = key(row.get("lane_fid")); coords = _coords(row.geometry)
        if lane is not None and len(coords) >= 2:
            lane_direction[lane] = tangent_deg(coords[0], coords[-1])

    relevant = lanes[lanes.geometry.distance(route) <= 16.0].copy()
    lane_geometries, lane_widths, lane_groups = {}, {}, {}
    for _, row in relevant.iterrows():
        lane = key(row.get("lane_fid")); group = key(row.get("lane_group_fid"))
        if lane is None:
            continue
        lane_geometries[lane] = row.geometry if lane not in lane_geometries else lane_geometries[lane].union(row.geometry)
        lane_groups[lane] = group
        width = row.get("width")
        if isinstance(width, (int, float)) and width > 0:
            lane_widths.setdefault(lane, []).append(round(float(width), 2))
    if not lane_geometries:
        checks = dict(UNKNOWN)
        return {"checks": checks, "facts": {"selection_policy": "recorded_route_lane_corridor",
                "map_location": map_location, "map_file": str(path), "target_junction_fid": None,
                "route_frame_count": len(ego_path), "frame_target_distance": []},
                "reasons": ["No lane polygon is available near the recorded route; lanes stay unknown."],
                "evidence_limits": ["The map has no usable lane geometry in the recorded route corridor."],
                "retrieval_suggestion": None}

    nearest_lane = []
    for point in ego_path:
        lane = min(lane_geometries, key=lambda lane_id: lane_geometries[lane_id].distance(Point(point)))
        nearest_lane.append(lane if lane_geometries[lane].distance(Point(point)) <= 8.0 else None)
    observed = [lane for lane in nearest_lane if lane is not None]
    sequence = []
    for lane in observed:
        if not sequence or sequence[-1] != lane:
            sequence.append(lane)
    coverage = round(len(observed) / len(ego_path), 3) if ego_path else 0.0

    connector_pairs = {(key(row.get("exit_lane_fid")), key(row.get("entry_lane_fid")))
                       for _, row in connectors.iterrows()}
    group_pairs = {(key(row.get("from_lane_group_fid")), key(row.get("to_lane_group_fid")))
                   for _, row in group_connectors.iterrows()}
    continuity = []
    for first, second in zip(sequence, sequence[1:]):
        continuity.append({"from": first, "to": second,
                           "connected": (lane_groups.get(first) == lane_groups.get(second)
                                         or (first, second) in connector_pairs
                                         or (lane_groups.get(first), lane_groups.get(second)) in group_pairs)})

    origin = str(attack.get("other_origin") or "none")
    need_oncoming = origin == "oncoming"
    need_side = "left" if origin == "left_adjacent" else "right" if origin == "right_adjacent" else None
    candidates = []
    waypoints = [(point, frame) for point, frame in zip(ego_path, frames)]
    for lane, geometry in lane_geometries.items():
        direction = lane_direction.get(lane)
        if direction is None:
            continue
        route_point, lane_point = nearest_points(route, geometry)
        point_index = min(range(len(waypoints)), key=lambda i: Point(waypoints[i][0]).distance(route_point))
        matrix = frames[point_index]["ego2global"]
        heading = math.degrees(math.atan2(float(matrix[1][0]), float(matrix[0][0])))
        relative = (heading - direction + 180.0) % 360.0 - 180.0
        dot = math.cos(math.radians(relative))
        dx, dy = lane_point.x - route_point.x, lane_point.y - route_point.y
        local_left = -math.sin(math.radians(heading)) * dx + math.cos(math.radians(heading)) * dy
        side = "left" if local_left > 0 else "right" if local_left < 0 else None
        distance = round(route.distance(geometry), 3)
        if distance > 12.0:
            continue
        matches = ((need_oncoming and dot <= -0.5) or
                   (need_side is not None and dot >= 0.5 and side == need_side) or
                   (origin in ("ahead_same_lane", "behind_same_lane") and lane in sequence))
        if matches:
            candidates.append({"lane_fid": lane, "lane_group_fid": lane_groups.get(lane),
                               "side": side, "direction_dot": round(dot, 3),
                               "route_distance_m": distance,
                               "widths_m": lane_widths.get(lane, [])})

    checks = dict(UNKNOWN); reasons = []
    if coverage >= 0.5 and sequence:
        checks["road"] = "pass"
        reasons.append(f"The recorded ego route stays in mapped lanes for {coverage:.1%} of frames.")
    else:
        reasons.append("The recorded ego route does not have enough mapped lane coverage; road stays unknown.")

    continuity_known = all(item["connected"] for item in continuity)
    target_ok = True if origin in ("ahead_same_lane", "behind_same_lane", "none") else bool(candidates)
    if coverage >= 0.5 and continuity_known and target_ok:
        checks["lanes"] = "pass"
        reasons.append("Recorded lanes are connected and the required target lane is available."
                       if target_ok else "Recorded lanes are connected.")
    elif coverage >= 0.5 and continuity_known:
        reasons.append("The target vehicle requires a lane direction/side that is not proven by the map; lanes stays unknown.")
    else:
        reasons.append("Lane continuity or the required target lane is incomplete; lanes stays unknown.")

    space_lanes = set(sequence) | {item["lane_fid"] for item in candidates}
    widths = [width for lane in space_lanes for width in lane_widths.get(lane, [])]
    if widths and min(widths) >= MIN_PLACEMENT_WIDTH_M:
        checks["space"] = "pass"
        reasons.append(f"Official lane geometry provides at least {min(widths):.2f}m placement width.")
    elif widths:
        checks["space"] = "fail"
        reasons.append(f"The narrowest candidate lane is {min(widths):.2f}m, below the placement minimum.")

    return {"available": True, "checks": checks,
            "facts": {"map_location": map_location, "map_file": str(path),
                      "selection_policy": "recorded_route_lane_corridor",
                      "route_frame_count": len(ego_path), "target_junction_fid": None,
                      "ego_lane_sequence": sequence,
                      "ego_lane_group_sequence": [lane_groups.get(lane) for lane in sequence],
                      "ego_lane_coverage_fraction": coverage,
                      "ego_lane_continuity": continuity,
                      "target_lane_candidates": candidates, "frame_target_distance": []},
            "reasons": reasons,
            "evidence_limits": ["Non-junction lane checks use official lane polygons and baseline directions; "
                                "dynamic obstacles and occlusion remain outside map evidence."],
            "retrieval_suggestion": None}


def main():
    request = json.load(sys.stdin)
    frames = request["frames"]
    attack = request.get("attack") or {}
    map_location = next((str(frame.get("map_location")) for frame in frames if frame.get("map_location")), "")
    if not map_location:
        print(json.dumps({"available": False, "reason": "frames carry no map_location"}))
        return
    path = map_file(map_location)
    lanes = gpd.read_file(path, layer="lanes_polygons", engine="pyogrio")
    connectors = gpd.read_file(path, layer="lane_connectors", engine="pyogrio", fid_as_index=True)
    baselines = gpd.read_file(path, layer="baseline_paths", engine="pyogrio")
    junctions = gpd.read_file(path, layer="intersections", engine="pyogrio", fid_as_index=True)
    segments = gpd.read_file(path, layer="road_segments", engine="pyogrio", fid_as_index=True)
    connector_polygons = gpd.read_file(path, layer="gen_lane_connectors_scaled_width_polygons",
                                       engine="pyogrio")
    group_connectors = gpd.read_file(path, layer="lane_group_connectors", engine="pyogrio",
                                     fid_as_index=True)

    # A missing pose must not silently change route order or image frame indices.
    for index, frame in enumerate(frames):
        validate_ego_pose(frame, index)
    ego_utm = [(float(frame["ego2global"][0][3]), float(frame["ego2global"][1][3]))
               for frame in frames]
    if not ego_utm:
        print(json.dumps({"available": False, "reason": "frames carry no ego poses"}))
        return
    bounds = junctions.total_bounds
    metric = metric_crs(junctions.crs, *utm_to_lonlat(ego_utm[0][0], ego_utm[0][1], bounds))
    to_metric = Transformer.from_crs("EPSG:4326", metric, always_xy=True)
    from_map = Transformer.from_crs(junctions.crs, metric, always_xy=True)
    ego_path = [transform_geometry(to_metric.transform, Point(*utm_to_lonlat(x, y, bounds))).coords[0]
                for x, y in ego_utm]

    context = str(attack.get("map_context") or "")
    if context not in JUNCTION_CONTEXTS:
        print(json.dumps(_nonjunction_facts(context, attack, frames, ego_path, lanes, connectors,
                                            baselines, group_connectors, from_map, map_location, path),
                         ensure_ascii=False, default=str))
        return

    # Inspect the actual recorded route, not a radius around its end or a graph search.
    junctions = junctions.copy()
    junctions["geometry"] = junctions.geometry.apply(
        lambda geometry: transform_geometry(from_map.transform, geometry))
    route_line = LineString(ego_path)
    connector_map_frame = connectors.copy()
    connector_map_frame["geometry"] = connector_map_frame.geometry.apply(
        lambda geometry: transform_geometry(from_map.transform, geometry))
    route_connected_fids = {
        key(row["intersection_fid"]) for _, row in connector_map_frame.iterrows()
        if row.geometry is not None and row.geometry.distance(route_line) <= 10.0
    }
    route_candidates = route_junctions(ego_path, [
        (key(fid), row.geometry, bool(row.get("is_mini")))
        for fid, row in junctions.iterrows()], route_connected_fids=route_connected_fids)
    if request.get("list_junctions"):
        print(json.dumps({"available": True, "route_junctions": [
            {name: item[name] for name in ("intersection_fid", "entry_distance_m",
                                            "entry_frame_index")}
            for item in route_candidates
        ]}, ensure_ascii=False))
        return
    forced = key(request.get("force_junction_fid"))
    selected = next((item for item in route_candidates if item["intersection_fid"] == forced), None) \
        if forced is not None else (route_candidates[0] if route_candidates else None)
    if selected is None:
        print(json.dumps({"available": True, "checks": dict(UNKNOWN),
                          "facts": {"map_location": map_location, "map_file": str(path),
                                    "selection_policy": "first_real_junction_on_recorded_route",
                                    "target_junction_fid": None, "frame_target_distance": []},
                          "reasons": ["The recorded trajectory does not establish a real junction; "
                                      "no other road or distant junction is substituted."],
                          "evidence_limits": ["No junction was selected from the recorded trajectory."],
                          "retrieval_suggestion": None}))
        return
    target = selected["intersection_fid"]
    region_geometries = {key(fid): row.geometry for fid, row in junctions.iterrows()}
    region_fids = connected_intersection_regions(target, region_geometries)
    region_set = set(region_fids)
    junctions = junctions.loc[[key(fid) in region_set for fid in junctions.index]]
    connectors = connectors[[key(value) in region_set for value in connectors["intersection_fid"]]].copy()
    group_connectors = group_connectors[
        [key(value) in region_set for value in group_connectors["intersection_fid"]]].copy()
    relevant_lanes = {key(value) for column in ("exit_lane_fid", "entry_lane_fid")
                      for value in connectors[column]} - {None}
    lanes = lanes[[key(value) in relevant_lanes for value in lanes["lane_fid"]]].copy()
    relevant_connectors = {key(value) for value in connectors.index}
    connector_polygons = connector_polygons[
        [key(value) in relevant_connectors for value in connector_polygons["lane_connector_fid"]]].copy()
    lane_ids = {key(value) for value in lanes["lane_fid"]} - {None}
    connector_ids = {key(value) for value in connectors.index} - {None}
    base_rows = baselines[[key(value) in lane_ids or key(value) in connector_ids
                           for value in
                           baselines["lane_fid"].where(baselines["lane_fid"].notna(),
                                                       baselines["lane_connector_fid"])]]
    for frame in (lanes, connectors):
        frame["geometry"] = frame.geometry.apply(lambda g: transform_geometry(from_map.transform, g))
    baselines = base_rows.copy()
    baselines["geometry"] = baselines.geometry.apply(lambda g: transform_geometry(from_map.transform, g))
    connector_polygons = connector_polygons.copy()
    connector_polygons["geometry"] = connector_polygons.geometry.apply(
        lambda g: transform_geometry(from_map.transform, g))
    group_connectors = group_connectors.copy()
    group_connectors["geometry"] = group_connectors.geometry.apply(
        lambda g: transform_geometry(from_map.transform, g))
    segments = segments.copy()
    segments["geometry"] = segments.geometry.apply(
        lambda g: transform_geometry(from_map.transform, g))

    lane_direction, lane_baseline, connector_geometry, connector_tangents = {}, {}, {}, {}
    for _, row in baselines.iterrows():
        geometry = row.geometry
        coords = _coords(geometry)
        lane = key(row.get("lane_fid"))
        if lane is not None and len(coords) >= 2:
            lane_direction[lane] = tangent_deg(coords[0], coords[-1])
            lane_baseline[lane] = geometry
        connector = key(row.get("lane_connector_fid"))
        if connector is not None and len(coords) >= 2:
            connector_geometry[connector] = geometry
            connector_tangents[connector] = (tangent_deg(coords[0], coords[1]),
                                             tangent_deg(coords[-2], coords[-1]))

    lane_rows, lane_geometry = {}, {}
    for _, row in lanes.iterrows():
        lane = key(row["lane_fid"])
        if lane is None:
            continue
        geometry = row.geometry
        entry = lane_rows.setdefault(lane, dict(lane_fid=lane, lane_group_fid=key(row["lane_group_fid"]),
                                                widths=[], distance_m=math.inf,
                                                direction_deg=lane_direction.get(lane)))
        width = row.get("width")
        if isinstance(width, (int, float)) and width > 0:
            entry["widths"].append(round(float(width), 2))
        entry["distance_m"] = round(min(entry["distance_m"],
                                        min(geometry.distance(Point(point)) for point in ego_path)), 2)
        lane_geometry.setdefault(lane, geometry)

    connector_rows = []
    for fid, row in connectors.iterrows():
        connector = key(fid)
        exit_lane, entry_lane = key(row["exit_lane_fid"]), key(row["entry_lane_fid"])
        incoming = outgoing = None
        tangents = connector_tangents.get(connector)
        exit_geometry, entry_geometry = lane_geometry.get(exit_lane), lane_geometry.get(entry_lane)
        geometry = connector_geometry.get(connector)
        orientation = None
        if tangents and geometry is not None and exit_geometry is not None and entry_geometry is not None:
            # Verify the geometry order against the lanes it connects; never assume it.
            start_to_exit = exit_geometry.distance(Point(_coords(geometry)[0]))
            start_to_entry = entry_geometry.distance(Point(_coords(geometry)[0]))
            orientation = "exit_first" if start_to_exit <= start_to_entry else "entry_first"
            if orientation == "exit_first":
                incoming, outgoing = tangents[0], tangents[1]
            else:
                incoming, outgoing = tangents[1], tangents[0]
        turn = turn_from_tangents(incoming, outgoing)
        connector_rows.append(dict(connector_fid=connector, intersection_fid=key(row["intersection_fid"]),
                                   exit_lane_fid=exit_lane, entry_lane_fid=entry_lane,
                                   exit_lane_group_fid=key(row["exit_lane_group_fid"]),
                                   entry_lane_group_fid=key(row["entry_lane_group_fid"]),
                                   turn=turn, geometry_orientation=orientation,
                                   incoming_deg=incoming, outgoing_deg=outgoing))

    # Recover the connector actually traversed by consecutive ego poses. Candidate
    # connectors are not interchangeable: a left-turn connector cannot replace a
    # recorded right turn merely because it crosses another vehicle path.
    ego_lane_sequence = []
    for point in ego_path:
        if not lane_geometry:
            break
        lane = min(lane_geometry, key=lambda lane_id: lane_geometry[lane_id].distance(Point(point)))
        distance = lane_geometry[lane].distance(Point(point))
        if distance <= 8.0 and (not ego_lane_sequence or ego_lane_sequence[-1] != lane):
            ego_lane_sequence.append(lane)
    actual_connectors = []
    for start_lane, end_lane in zip(ego_lane_sequence, ego_lane_sequence[1:]):
        actual_connectors.extend(
            row["connector_fid"] for row in connector_rows
            if row["exit_lane_fid"] == start_lane and row["entry_lane_fid"] == end_lane)
    actual_connectors = sorted(set(actual_connectors))

    junction_rows = {}
    for fid, row in junctions.iterrows():
        junction = key(fid)
        centre = row.geometry.representative_point()
        # Physical branches come from road_segments polygons that touch this junction.
        touched = [segment for _, segment in segments.iterrows()
                   if segment.geometry is not None and segment.geometry.intersects(row.geometry)]
        branch_ids = sorted({key(value) for value, _ in segments.iterrows()
                             if segments.loc[value].geometry is not None
                             and segments.loc[value].geometry.intersects(row.geometry)} )
        branch_groups = sorted({key(group) for segment in touched
                                for group in str(segment.get("lane_group_fids") or "").split(",")
                                if key(group) is not None})
        junction_rows[junction] = dict(intersection_fid=junction,
                                       is_mini=bool(row.get("is_mini")) if row.get("is_mini") is not None else None,
                                       type_fid=key(row.get("intersection_type_fid")),
                                       area_m2=round(row.geometry.area, 1),
                                       distance_m=round(min(row.geometry.distance(Point(p)) for p in ego_path), 2),
                                       centre=(centre.x, centre.y), geometry=row.geometry,
                                       branches=branch_ids, branch_lane_groups=branch_groups)

    frame_roadblocks = {key(value) for frame in frames
                        for value in (frame.get("roadblock_ids") or [])} - {None}
    review_point = Point(selected["review_xy"])
    covering_group_connectors = {key(fid) for fid, row in group_connectors.iterrows()
                                 if row.geometry.covers(review_point)}
    route_hits = covering_group_connectors & frame_roadblocks
    bound_connectors = route_hits if route_hits else covering_group_connectors
    approach_groups = {key(group_connectors.loc[fid]["from_lane_group_fid"])
                       for fid in bound_connectors} - {None}
    # Prefer the ego's own lane at the exact route-entry frame. Group-connector
    # coverage can be empty for a junction region split away from the recorded lane.
    entry_point = Point(ego_path[selected["entry_frame_index"]])
    if lane_geometry:
        entry_lane = min(lane_geometry, key=lambda lane_id: lane_geometry[lane_id].distance(entry_point))
        if lane_geometry[entry_lane].distance(entry_point) <= 8.0:
            approach_groups = {lane_rows[entry_lane]["lane_group_fid"]} - {None}
    # Several possible entrances do not establish which one the ego uses.
    if len(approach_groups) != 1:
        approach_groups = set()
    ego_lane_group = next(iter(approach_groups), None)
    ego_binding = "recorded_route_junction_approach" if approach_groups else None
    ego_heading = None
    first_matrix = frames[selected["entry_frame_index"]].get("ego2global")
    if first_matrix:
        # Recorded heading at the selected junction approach.
        ego_heading = math.degrees(math.atan2(float(first_matrix[1][0]), float(first_matrix[0][0])))
    for row in connector_rows:
        centre = junction_rows.get(row["intersection_fid"], {}).get("centre")
        approach = lane_geometry.get(row["exit_lane_fid"])
        row["arm_side"] = None
        row["arm_origin"] = None
        if centre and approach is not None:
            point = approach.representative_point()
            position = arm_position(tangent_deg(centre, (point.x, point.y)), ego_heading)
            # The approach lane's own travel direction is the incoming direction here.
            incoming = row.get("incoming_deg")
            if incoming is None:
                incoming = lane_rows.get(row["exit_lane_fid"], {}).get("direction_deg")
            row["arm_side"] = position
            row["arm_origin"] = origin_from_arm(position, incoming, ego_heading)

    def complete_junction(_junction=None):
        rows = connector_rows
        return bool(rows) and all(row["turn"] is not None and row["arm_origin"] is not None for row in rows)

    arms = {}
    for row in connector_rows:
        if row["arm_side"] is not None:
            arms.setdefault(target, set()).add(row["arm_side"])
    cluster_geometry = junctions.geometry.unary_union
    cluster_branches = sorted({branch for item in junction_rows.values()
                               for branch in item["branches"]})
    cluster_groups = sorted({group for item in junction_rows.values()
                             for group in item["branch_lane_groups"]})

    facts = dict(
        crs=str(metric), map_location=map_location, map_file=str(path),
        selection_policy="first_real_junction_on_recorded_route",
        target_entry_distance_m=selected["entry_distance_m"],
        target_entry_frame_index=selected["entry_frame_index"],
        approach_lane_count=len(lane_rows),
        lane_count_by_group={str(group): sum(1 for item in lane_rows.values() if item["lane_group_fid"] == group)
                             for group in sorted({item["lane_group_fid"] for item in lane_rows.values()} - {None})},
        lane_widths_m=sorted({width for item in lane_rows.values() for width in item["widths"]})[:12],
        lanes_with_direction=sum(1 for item in lane_rows.values() if item["direction_deg"] is not None),
        ego_lane_group_fid=ego_lane_group, ego_binding=ego_binding,
        approach_group_connectors=sorted(covering_group_connectors),
        approach_route_hits=sorted(route_hits),
        roadblock_ids_from_frames=sorted(str(value) for value in frame_roadblocks)[:12],
        junction_count=len(region_fids),
        target_junction=dict(intersection_fid=target,
                             region_fids=region_fids,
                             physical_branches=len(cluster_branches),
                             branch_lane_groups=cluster_groups,
                             arms=sorted(arms.get(target, ())),
                             distance_m=round(min(cluster_geometry.distance(Point(p)) for p in ego_path), 2),
                             connectors_complete=complete_junction(target)),
        connector_count=len(connector_rows),
        connectors_with_turn=sum(1 for row in connector_rows if row["turn"] is not None),
        connectors=[{name: row[name] for name in ("connector_fid", "intersection_fid", "exit_lane_fid",
                                                  "entry_lane_fid", "turn", "arm_side", "arm_origin")}
                    for row in connector_rows[:20]],
    )

    ego_action = str(attack.get("ego_action") or "")
    origin = str(attack.get("other_origin") or "")
    other_action = str(attack.get("other_action") or "")
    checks = dict(UNKNOWN)
    reasons, limits = [], [
        "Connector direction follows the nuPlan SDK (exit lane -> entry lane) and is only used when "
        "the connector baseline geometry matches the lanes it joins; otherwise it stays unknown.",
        "Map polygons measure geometry and connection availability, not that a planned trajectory is feasible.",
        "Dynamic obstacles and occlusion are not represented by the map and remain unknown.",
    ]
    branches = len(cluster_branches)
    if branches == 0:
        reasons.append(f"The first recorded-route junction {target} has no measured physical road "
                       "segments, so road stays unknown rather than being called incompatible.")
    elif context == "t_junction":
        # A T target needs at least three arms. A four-arm cross is compatible when the
        # required approach and lane movements are present; the extra arm may stay empty.
        if branches >= 3:
            checks["road"] = "pass"
            reasons.append(f"Route junction {target} has {branches} physical road "
                           f"segments; it provides at least the three arms a t_junction needs "
                           "(a cross intersection may supply the extra arm).")
        else:
            checks["road"] = "fail"
            reasons.append(f"Route junction {target} has only {branches} physical "
                           "road segments; a t_junction target needs at least three.")
    else:
        if branches >= 4:
            checks["road"] = "pass"
            reasons.append(f"Route junction {target} has {branches} physical road "
                           "segments, supporting a cross_intersection.")
        else:
            checks["road"] = "fail"
            reasons.append(f"Route junction {target} has only {branches} physical "
                           "road segments; a cross_intersection target needs at least four.")

    want_ego = {"turn_left": "left", "turn_right": "right", "straight": "straight"}.get(ego_action)
    want_side = origin if origin in {"left_arm", "right_arm", "oncoming",
                                     "ahead_same_lane", "behind_same_lane"} else None
    same_lane_target = origin in {"ahead_same_lane", "behind_same_lane"}
    want_other = {"turn_left": "left", "turn_right": "right", "straight": "straight"}.get(other_action)
    if target is not None:
        rows = connector_rows
        complete = complete_junction(target)
        approach_rows = [row for row in rows if row["exit_lane_group_fid"] in approach_groups]
        approach_lane_ids = sorted({row["exit_lane_fid"] for row in approach_rows})
        actual_rows = [row for row in approach_rows if row["connector_fid"] in actual_connectors]
        other_rows = ([] if same_lane_target else
                      [row for row in rows
                       if row["arm_origin"] == want_side and row["turn"] == want_other])
        official_connectors = (connector_polygons.geometry.unary_union
                               if len(connector_polygons) else None)
        official_lanes = lanes.geometry.unary_union if len(lanes) else None
        pairs = []
        approach_paths = [(f"approach_straight:{lane}", lane_baseline.get(lane))
                          for lane in approach_lane_ids]
        turn_paths = [(f"turn:{row['turn']}:{row['connector_fid']}",
                       connector_geometry.get(row["connector_fid"])) for row in actual_rows]
        ego_paths = (approach_paths + turn_paths if want_ego == "straight"
                     else turn_paths + approach_paths)
        for ego_kind, ego_line in ego_paths:
            for other_row in other_rows:
                other_line = connector_geometry.get(other_row["connector_fid"])
                paths_cross = connector_paths_cross(ego_line, other_line)
                path_distance = (round(ego_line.distance(other_line), 3)
                                 if ego_line is not None and other_line is not None else None)
                ego_uses_lane = ego_kind.startswith("approach_straight:")
                on_official_network = None if official_connectors is None else bool(
                    ego_line is not None and other_line is not None
                    and (ego_uses_lane is False or official_lanes is not None)
                    and (ego_line.intersects(official_lanes.buffer(0.25)) if ego_uses_lane
                         else ego_line.intersects(official_connectors.buffer(0.25)))
                    and other_line.intersects(official_connectors.buffer(0.25)))
                interaction_compatible = bool(
                    on_official_network is True
                    and (paths_cross is True
                         or (path_distance is not None and path_distance <= CONFLICT_PROXIMITY_M)))
                pairs.append(dict(ego_path_kind=ego_kind,
                                  other_connector_fid=other_row["connector_fid"],
                                  other_turn=other_row["turn"],
                                  other_arm_origin=other_row["arm_origin"],
                                  other_entry_lane_fid=other_row["entry_lane_fid"],
                                  paths_cross=paths_cross,
                                  path_distance_m=path_distance,
                                  interaction_compatible=interaction_compatible,
                                  on_official_network=on_official_network))
        ego_movements = sorted({row["turn"] for row in approach_rows if row["turn"]})
        exact_ego_match = want_ego in ego_movements
        facts["ego_observed_movements"] = ego_movements
        facts["ego_exact_action_match"] = exact_ego_match
        facts["ego_approach_lane_fids"] = approach_lane_ids
        facts["ego_path_options"] = [
            {name: row[name] for name in ("connector_fid", "turn", "entry_lane_fid",
                                           "entry_lane_group_fid")}
            for row in actual_rows
        ]
        facts["ego_lane_sequence"] = ego_lane_sequence
        facts["ego_actual_connector_fids"] = actual_connectors
        facts["ego_unused_approach_connectors"] = [
            row["connector_fid"] for row in approach_rows if row["connector_fid"] not in actual_connectors]
        facts["target_other_connectors"] = [
            {name: row[name] for name in ("connector_fid", "turn", "arm_origin",
                                           "exit_lane_fid", "entry_lane_fid")}
            for row in other_rows
        ]
        facts["interaction_path_pairs"] = pairs
        interaction_pairs = [pair for pair in pairs if pair["interaction_compatible"]]
        unknown_pairs = [pair for pair in pairs if pair["paths_cross"] is None]
        if not approach_groups:
            reasons.append("The recorded-route ego approach could not be bound uniquely to one lane group; "
                           "lane feasibility stays unknown.")
        elif not complete:
            reasons.append(f"Lane movement evidence for physical junction {region_fids} is incomplete; "
                           "lanes stays unknown.")
        elif not approach_rows:
            checks["lanes"] = "fail"
            reasons.append(f"Complete connector data provides no traversable path from the recorded "
                           f"ego approach group(s) {sorted(approach_groups)}.")
        elif same_lane_target and approach_rows:
            checks["lanes"] = "pass"
            reasons.append(f"Target vehicle is {origin} on the ego approach lane; no junction-arm "
                           "connector is required. The lead edit uses the existing approach path.")
        elif origin != "none" and not other_rows:
            checks["lanes"] = "fail"
            reasons.append(f"Complete connector data has no {want_side} approach supporting "
                           f"{want_other or 'the target motion'} in physical junction {region_fids}.")
        elif origin == "none" and approach_rows:
            checks["lanes"] = "pass"
            reasons.append(f"The ego approach has an editable connected path through junction {region_fids}.")
        elif interaction_pairs:
            checks["lanes"] = "pass"
            chosen = interaction_pairs[0]
            relation = ("crossing" if chosen["paths_cross"] else
                        f"same conflict area ({chosen['path_distance_m']:.2f}m apart, editable lane offset)")
            reasons.append(
                f"Recorded ego path {chosen['ego_path_kind']} and target other connector "
                f"{chosen['other_connector_fid']} ({chosen['other_turn']}, {want_side}) form a "
                f"{relation}. The target interaction may occur before the ego follows its normal turn.")
        elif unknown_pairs:
            reasons.append("Connector paths exist but their pairwise crossing cannot be measured; "
                           "lanes stays unknown.")
        else:
            reasons.append("Connectors exist, but no ego/other path pair was proven to form the "
                           "target interaction from map geometry; lanes stays unknown.")

        space_lane_ids = set()
        for row in (other_rows or approach_rows):
            space_lane_ids.update(value for value in (row["exit_lane_fid"], row["entry_lane_fid"])
                                  if value is not None)
        widths = [width for lane in space_lane_ids
                  for width in lane_rows.get(lane, {}).get("widths", [])]
        if not space_lane_ids or not widths:
            reasons.append("No measured lane width is available for placement; space stays unknown.")
        elif min(widths) < MIN_PLACEMENT_WIDTH_M:
            checks["space"] = "fail"
            reasons.append(f"The narrowest candidate placement lane is {min(widths):.2f}m, below "
                           f"the {MIN_PLACEMENT_WIDTH_M:.2f}m minimum for a vehicle and clearance.")
        else:
            checks["space"] = "pass"
            reasons.append(f"Official drivable lane geometry provides at least {min(widths):.2f}m "
                           "placement width for the target interaction.")

    target_geometry = cluster_geometry
    frame_rows = []
    for index, point in enumerate(ego_path):
        distance = round(target_geometry.distance(Point(point)), 1) if target_geometry is not None else None
        frame_rows.append(dict(frame_index=index, distance_to_target_m=distance))
    facts["frame_target_distance"] = frame_rows
    facts["target_junction_fid"] = target
    for row in junction_rows.values():
        row.pop("geometry", None)
    print(json.dumps({"available": True, "facts": facts, "checks": checks, "reasons": reasons,
                      "evidence_limits": limits, "retrieval_suggestion": None},
                     ensure_ascii=False, default=str))


main()
