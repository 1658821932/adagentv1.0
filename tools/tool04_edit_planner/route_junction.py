"""Select only the first real junction entered by the recorded trajectory."""


def first_route_junction(points, junctions):
    """Return a route-ordered junction and its entry/review points, or None.

    ``junctions`` contains (id, polygon, is_mini). A boundary-only touch does not
    count as passing through a junction. Selection never depends on target shape.
    """
    from shapely.geometry import LineString, Point

    if not points:
        return None
    start = Point(points[0])
    route = LineString(points) if len(points) > 1 else start
    if route.length == 0:
        route = start

    def line_parts(geometry):
        if geometry.geom_type in ("LineString", "LinearRing"):
            yield geometry
        elif hasattr(geometry, "geoms"):
            for part in geometry.geoms:
                yield from line_parts(part)

    hits = []
    for fid, polygon, is_mini in junctions:
        if is_mini or polygon is None or polygon.is_empty:
            continue
        if polygon.contains(start):
            hits.append((0.0, str(fid), fid, start, start))
            continue
        if route.geom_type == "Point":
            continue
        for part in line_parts(route.intersection(polygon)):
            if part.length == 0 or not polygon.contains(part.interpolate(0.5, normalized=True)):
                continue
            ends = [Point(part.coords[0]), Point(part.coords[-1])]
            entry = min(ends, key=route.project)
            hits.append((route.project(entry), str(fid), fid, entry,
                         part.interpolate(0.5, normalized=True)))
    if not hits:
        return None
    distance, _, fid, entry, review = min(hits, key=lambda row: (row[0], row[1]))
    travelled, index = 0.0, 0
    for index in range(max(0, len(points) - 1)):
        length = Point(points[index]).distance(Point(points[index + 1]))
        if travelled + length >= distance:
            break
        travelled += length
    return dict(intersection_fid=fid, entry_distance_m=round(distance, 3),
                entry_frame_index=index, entry_xy=list(entry.coords[0]),
                review_xy=list(review.coords[0]))
