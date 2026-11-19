"""Identifier normalisation and geometry helpers shared by the map reader.

The map store mixes integer ids with float (because of NULLs) and text ids, so every
join goes through one normaliser instead of ad-hoc str()/astype(str) conversions.
"""
from __future__ import annotations

import math


def key(value):
    """Comparable id: int when numeric-integral, text otherwise, None when absent."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int,)):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return int(value) if float(value).is_integer() else None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return text                     # keep genuine non-numeric ids untouched
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def turn_from_tangents(incoming_deg, outgoing_deg, straight_deg=30.0):
    """Turn kind from the two end tangents at the junction."""
    if incoming_deg is None or outgoing_deg is None:
        return None
    delta = (outgoing_deg - incoming_deg + 180.0) % 360.0 - 180.0
    if abs(delta) <= straight_deg:
        return "straight"
    return "left" if delta > 0 else "right"


def arm_position(bearing_deg, ego_heading_deg):
    """Physical position of an arm relative to the ego heading: ahead/behind/left/right."""
    if bearing_deg is None or ego_heading_deg is None:
        return None
    relative = (bearing_deg - ego_heading_deg + 180.0) % 360.0 - 180.0
    if abs(relative) <= 45.0:
        return "ahead"
    if abs(relative) >= 135.0:
        return "behind"
    return "left" if relative > 0 else "right"


def origin_from_arm(position, incoming_deg, ego_heading_deg):
    """Map an arm to tool01's other_origin semantics.

    Position is physical (ahead/behind/left/right of the ego heading). Facing traffic in
    front of the ego is oncoming, same-direction traffic on the ego's own path ahead or
    behind is the same-lane case, and side arms keep the arm identity. Anything the
    evidence cannot decide stays None instead of being guessed.
    """
    if position is None or incoming_deg is None or ego_heading_deg is None:
        return None
    delta = (incoming_deg - ego_heading_deg + 180.0) % 360.0 - 180.0
    against = abs(delta) >= 135.0          # traffic moving towards the ego
    along = abs(delta) <= 45.0             # traffic moving with the ego
    if position == "ahead":
        if against:
            return "oncoming"
        if along:
            return "ahead_same_lane"
        return None
    if position == "behind":
        return "behind_same_lane" if along else None
    if position == "left":
        return "left_arm"
    if position == "right":
        return "right_arm"
    return None


def tangent_deg(point_a, point_b):
    if point_a is None or point_b is None:
        return None
    dx, dy = point_b[0] - point_a[0], point_b[1] - point_a[1]
    if math.hypot(dx, dy) == 0:
        return None
    return math.degrees(math.atan2(dy, dx))


def lane_fit(approach_groups, rows, want_ego, want_side, want_other, complete):
    """Does this junction support the ego manoeuvre and the required other-vehicle approach?

    ``approach_groups`` is the set of lane groups the route actually reaches the junction
    from, so a multi-hop target is judged by its own entrance, not by the ego's start lane.
    Returns (ego_ok, other_ok); None means the evidence cannot decide.
    """
    if not complete:
        return (None, None)
    ego_ok = None
    if want_ego and approach_groups:
        ego_ok = any(row.get("turn") == want_ego and row.get("exit_lane_group_fid") in approach_groups
                     for row in rows)
    other_ok = None
    if want_side:
        side_rows = [row for row in rows if row.get("arm_origin") == want_side]
        other_ok = (any(row.get("turn") == want_other for row in side_rows)
                    if want_other else bool(side_rows))
    return (ego_ok, other_ok)


def connected_intersection_regions(target_fid, geometries, max_radius_m=30.0,
                                   touch_tolerance_m=0.25):
    """Return the topology-connected regions of one physical intersection.

    nuPlan can split a junction into many lane-region polygons. Starting at the
    recorded-route junction, include touching/near-touching regions within one
    bounded junction neighbourhood; do not walk into unrelated later junctions.
    """
    if target_fid not in geometries or geometries[target_fid] is None:
        return [target_fid]
    target = geometries[target_fid]
    centre = target.representative_point()
    eligible = {
        fid for fid, geometry in geometries.items()
        if geometry is not None and not geometry.is_empty
        and centre.distance(geometry.representative_point()) <= max_radius_m
    }
    selected = {target_fid}
    frontier = [target_fid]
    while frontier:
        current = frontier.pop()
        geometry = geometries[current]
        for fid in eligible - selected:
            if geometry.distance(geometries[fid]) <= touch_tolerance_m:
                selected.add(fid)
                frontier.append(fid)
    return sorted(selected)


def connector_paths_cross(first, second, endpoint_extension_m=6.0):
    """Whether two official connector centrelines form a crossing path pair.

    Connector baselines can end at their entry lanes before the centre of a large
    junction. Extend both endpoints along their terminal tangent by a bounded
    distance, then test for a proper crossing. Missing/invalid geometry is unknown.
    """
    from shapely.geometry import LineString

    if first is None or second is None or first.is_empty or second.is_empty:
        return None

    def extended(line):
        coords = list(line.coords)
        if len(coords) < 2 or endpoint_extension_m <= 0:
            return line

        def move(point, other):
            dx, dy = point[0] - other[0], point[1] - other[1]
            length = math.hypot(dx, dy)
            if length == 0:
                return point
            return (point[0] + dx / length * endpoint_extension_m,
                    point[1] + dy / length * endpoint_extension_m)

        return LineString([move(coords[0], coords[1]), *coords,
                           move(coords[-1], coords[-2])])

    return bool(extended(first).crosses(extended(second)))
