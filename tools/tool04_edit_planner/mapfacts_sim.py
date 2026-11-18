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
from shapely.geometry import Point
from shapely.ops import transform as transform_geometry

from tools.tool01_failure_analyzer.structure_facts import ego_heading as validate_ego_pose, map_file, metric_crs, utm_to_lonlat
from tools.tool04_edit_planner.map_ids import (arm_position, key, lane_fit,
                                               origin_from_arm, tangent_deg, turn_from_tangents)

from tools.tool04_edit_planner.route_junction import first_route_junction

UNKNOWN = dict(road="unknown", lanes="unknown", space="unknown")
JUNCTION_CONTEXTS = ("t_junction", "cross_intersection")


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
        # A straight road, curve or merge target has no junction requirement. Do not
        # substitute a nearby or later junction; visual review decides its road shape.
        print(json.dumps({"available": True, "checks": dict(UNKNOWN),
                          "facts": {"map_location": map_location, "map_file": str(path),
                                    "selection_policy": "no_junction_target",
                                    "route_frame_count": len(ego_path),
                                    "target_junction_fid": None, "frame_target_distance": []},
                          "reasons": [f"{context or 'The target'} does not require a junction; "
                                      "road, curve and merge evidence is left to visual review."],
                          "evidence_limits": ["No junction is selected for straight_road, curve or merge targets."],
                          "retrieval_suggestion": None}))
        return

    # Inspect the actual recorded route, not a radius around its end or a graph search.
    junctions = junctions.copy()
    junctions["geometry"] = junctions.geometry.apply(
        lambda geometry: transform_geometry(from_map.transform, geometry))
    selected = first_route_junction(ego_path, [
        (key(fid), row.geometry, bool(row.get("is_mini")))
        for fid, row in junctions.iterrows()])
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
    junctions = junctions.loc[[target]]
    connectors = connectors[[key(value) == target for value in connectors["intersection_fid"]]].copy()
    group_connectors = group_connectors[
        [key(value) == target for value in group_connectors["intersection_fid"]]].copy()
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

    lane_direction, connector_geometry, connector_tangents = {}, {}, {}
    for _, row in baselines.iterrows():
        geometry = row.geometry
        coords = _coords(geometry)
        lane = key(row.get("lane_fid"))
        if lane is not None and len(coords) >= 2:
            lane_direction[lane] = tangent_deg(coords[0], coords[-1])
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

    def complete_junction(junction):
        rows = [row for row in connector_rows if row["intersection_fid"] == junction]
        return bool(rows) and all(row["turn"] is not None and row["arm_origin"] is not None for row in rows)

    arms = {}
    for row in connector_rows:
        if row["arm_side"] is not None:
            arms.setdefault(row["intersection_fid"], set()).add(row["arm_side"])

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
        junction_count=len(junction_rows),
        target_junction=dict(intersection_fid=target,
                             physical_branches=len(junction_rows[target]["branches"]),
                             branch_lane_groups=junction_rows[target]["branch_lane_groups"],
                             arms=sorted(arms.get(target, ())),
                             distance_m=junction_rows[target]["distance_m"],
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
        "Dynamic obstacles and occlusion are not in the map and remain a visual judgement.",
    ]
    branches = len(junction_rows[target]["branches"])
    if branches == 0:
        reasons.append(f"The first recorded-route junction {target} has no measured physical road "
                       "segments, so road stays unknown rather than being called incompatible.")
    elif context == "t_junction":
        # A T target needs at least three arms. A four-arm cross is compatible when the
        # required approach and lane movements are present; the extra arm may stay empty.
        if branches >= 3:
            checks["road"] = "pass"
            reasons.append(f"First recorded-route junction {target} has {branches} physical road "
                           f"segments; it provides at least the three arms a t_junction needs "
                           "(a cross intersection may supply the extra arm).")
        else:
            checks["road"] = "fail"
            reasons.append(f"First recorded-route junction {target} has only {branches} physical "
                           "road segments; a t_junction target needs at least three.")
    else:
        if branches >= 4:
            checks["road"] = "pass"
            reasons.append(f"First recorded-route junction {target} has {branches} physical road "
                           "segments, supporting a cross_intersection.")
        else:
            checks["road"] = "fail"
            reasons.append(f"First recorded-route junction {target} has only {branches} physical "
                           "road segments; a cross_intersection target needs at least four.")

    want_ego = {"turn_left": "left", "turn_right": "right", "straight": "straight"}.get(ego_action)
    want_side = origin if origin in {"left_arm", "right_arm", "oncoming",
                                     "ahead_same_lane", "behind_same_lane"} else None
    want_other = {"turn_left": "left", "turn_right": "right", "straight": "straight"}.get(other_action)
    if target is not None:
        rows = [row for row in connector_rows if row["intersection_fid"] == target]
        complete = complete_junction(target)
        ego_ok, other_ok = lane_fit(approach_groups, rows,
                                    want_ego, want_side, want_other, complete)
        if complete and (ego_ok is False or other_ok is False):
            checks["lanes"] = "fail"
            missing = ([f"no {want_ego} connector from the recorded approach"] if ego_ok is False else []) + \
                      ([f"no {want_side} approach supporting {want_other or 'the motion'}"] if other_ok is False else [])
            reasons.append(f"First recorded-route junction {target}: " + "; ".join(missing) + ".")
        elif complete and ego_ok is True and (origin == "none" or other_ok is True):
            checks["lanes"] = "pass"
            reasons.append(f"The selected junction {target} supports the required lane movements.")
        else:
            reasons.append(f"Lane movement evidence for the fixed junction {target} is incomplete; "
                           "lanes stays unknown.")

    target_geometry = junction_rows.get(target, {}).get("geometry")
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
