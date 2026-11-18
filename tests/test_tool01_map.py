"""Map distances must be measured in metres on a proper projection, not in degrees."""
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, Point, Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.tool01_failure_analyzer import structure_facts as facts  # noqa: E402

ORIGIN = (-79.96, 40.42)                      # inside the Pittsburgh map extent
METRIC = "EPSG:32617"                          # UTM zone 17N, metres
TO_METRIC = Transformer.from_crs("EPSG:4326", METRIC, always_xy=True)
TO_LONLAT = Transformer.from_crs(METRIC, "EPSG:4326", always_xy=True)
HALF_SIDE = 5.0


def lonlat_of(east_offset, north_offset):
    """Longitude/latitude of a point offset from ORIGIN by metres."""
    east, north = TO_METRIC.transform(*ORIGIN)
    return TO_LONLAT.transform(east + east_offset, north + north_offset)


def ego_of(east_offset, north_offset):
    """Ego position in the archive convention (UTM metres) for the same offset."""
    east, north = TO_METRIC.transform(*ORIGIN)
    return (east + east_offset, north + north_offset)


def square(east_offset, north_offset, half=HALF_SIDE):
    corners = [lonlat_of(east_offset + dx, north_offset + dy)
               for dx, dy in ((-half, -half), (half, -half), (half, half), (-half, half))]
    return Polygon(corners)


def arms(east_offset, north_offset, count, reach=10.0):
    lines = []
    for step in range(count):
        angle = math.pi * step / count
        dx, dy = reach * math.cos(angle), reach * math.sin(angle)
        lines.append(LineString([lonlat_of(east_offset - dx, north_offset - dy),
                                 lonlat_of(east_offset + dx, north_offset + dy)]))
    return lines


def layers(targets, crs="EPSG:4326"):
    """Intersections and road segments; a far background polygon keeps the map extent wide."""
    polygons, segments = [], []
    for east, north, count in targets:
        polygons.append(square(east, north))
        segments.extend(arms(east, north, count))
    for offset in (5000.0, -5000.0):                        # keeps the map extent wide
        polygons.append(square(0.0, offset, half=500.0))
        segments.extend(arms(0.0, offset, 1, reach=600.0))
    return {
        "intersections": gpd.GeoDataFrame(geometry=polygons, crs=crs),
        "road_segments": gpd.GeoDataFrame(geometry=segments, crs=crs),
    }


MAP = "us-pa-pittsburgh-hazelwood"


def context(map_layers, ego, turn_deg=0.0):
    with mock.patch.object(facts, "map_file", return_value="map.gpkg"), \
         mock.patch("geopandas.read_file", side_effect=lambda path, layer, **kwargs: map_layers[layer]):
        return facts.map_context(MAP, ego, turn_deg)


def corners_of(map_layers, ego):
    return context(map_layers, ego)


class MetricDistanceTests(unittest.TestCase):
    def test_same_distance_east_and_north(self):
        map_layers = layers([(0.0, 0.0, 3)])
        for east, north in ((30.0, 0.0), (0.0, 30.0), (-30.0, 0.0), (0.0, -30.0)):
            with self.subTest(east=east, north=north):
                self.assertEqual(corners_of(map_layers, ego_of(east, north)), "t_junction")

    def test_the_forty_metre_limit_in_both_directions(self):
        map_layers = layers([(0.0, 0.0, 3)])
        inside = 40.0 - HALF_SIDE - 3.0      # 32 m from the body: clearly inside
        outside = 40.0 + HALF_SIDE + 3.0     # 48 m from the body: clearly outside
        for east, north in ((inside, 0.0), (0.0, inside)):
            with self.subTest(direction="inside", east=east, north=north):
                self.assertEqual(corners_of(map_layers, ego_of(east, north)), "t_junction")
        for east, north in ((outside, 0.0), (0.0, outside)):
            with self.subTest(direction="outside", east=east, north=north):
                self.assertEqual(corners_of(map_layers, ego_of(east, north)), "straight_road")

    def test_metric_nearest_differs_from_degree_nearest(self):
        # A is 18 m east with four arms; B is 20 m north with three arms.
        # In degrees B looks closer, in metres A is closer.
        map_layers = layers([(18.0, 0.0, 4), (0.0, 20.0, 3)])
        ego_lonlat = Point(*lonlat_of(0.0, 0.0))
        degree_a = ego_lonlat.distance(square(18.0, 0.0))
        degree_b = ego_lonlat.distance(square(0.0, 20.0))
        self.assertLess(degree_b, degree_a)                     # the old rule preferred B
        self.assertLess(18.0 - HALF_SIDE, 20.0 - HALF_SIDE)      # metres prefer A
        self.assertEqual(corners_of(map_layers, ego_of(0.0, 0.0)), "cross_intersection")

    def test_far_intersection_keeps_the_straight_road_rule(self):
        map_layers = layers([(0.0, 120.0, 4)])
        self.assertEqual(corners_of(map_layers, ego_of(0.0, 0.0)), "straight_road")

    def test_curve_rule_is_unchanged_far_from_any_intersection(self):
        map_layers = layers([(0.0, 120.0, 4)])
        with mock.patch.object(facts, "map_file", return_value="map.gpkg"), \
             mock.patch("geopandas.read_file", side_effect=lambda path, layer, **kwargs: map_layers[layer]):
            self.assertEqual(facts.map_context("map", ego_of(0.0, 0.0), -30.0), "curve")


def projected_layers(targets):
    """Same map stored in metres (EPSG:32617), as a projected map would be."""
    east, north = TO_METRIC.transform(*ORIGIN)

    def square_at(east_offset, north_offset, half=HALF_SIDE):
        center_x, center_y = east + east_offset, north + north_offset
        return Polygon([(center_x + dx, center_y + dy)
                        for dx, dy in ((-half, -half), (half, -half), (half, half), (-half, half))])

    polygons, segments = [], []
    for east_offset, north_offset, count in targets:
        polygons.append(square_at(east_offset, north_offset))
        for step in range(count):
            angle = math.pi * step / count
            dx, dy = 10.0 * math.cos(angle), 10.0 * math.sin(angle)
            segments.append(LineString([(east + east_offset - dx, north + north_offset - dy),
                                        (east + east_offset + dx, north + north_offset + dy)]))
    for offset in (5000.0, -5000.0):
        polygons.append(square_at(0.0, offset, half=500.0))
        segments.append(LineString([(east, north + offset - 600.0), (east, north + offset + 600.0)]))
    return {
        "intersections": gpd.GeoDataFrame(geometry=polygons, crs=METRIC),
        "road_segments": gpd.GeoDataFrame(geometry=segments, crs=METRIC),
    }


class ProjectedMapTests(unittest.TestCase):
    """A map stored in metres must not be read as if its bounds were degrees."""

    def test_projected_map_keeps_metric_distances(self):
        map_layers = projected_layers([(0.0, 0.0, 3)])
        self.assertEqual(map_layers["intersections"].crs.to_epsg(), 32617)
        for east, north in ((30.0, 0.0), (0.0, 30.0), (0.0, -30.0)):
            with self.subTest(east=east, north=north):
                self.assertEqual(corners_of(map_layers, ego_of(east, north)), "t_junction")
        self.assertEqual(corners_of(map_layers, ego_of(48.0, 0.0)), "straight_road")

    def test_projected_map_crs_is_used_without_reprojection(self):
        self.assertEqual(facts.metric_crs(METRIC, -79.96, 40.42).to_epsg(), 32617)

    def test_projected_map_without_crs_raises_before_conversion(self):
        map_layers = projected_layers([(0.0, 0.0, 3)])
        map_layers["intersections"].crs = None
        with self.assertRaises(facts.FactsError):
            corners_of(map_layers, ego_of(30.0, 0.0))


class JunctionArmTests(unittest.TestCase):
    def assert_counts_use_bend_rule(self, count):
        map_layers = layers([(0.0, 0.0, count)])
        for turn, expected in ((0.0, "straight_road"), (7.99, "straight_road"),
                               (8.0, "curve"), (-8.0, "curve")):
            with self.subTest(count=count, turn=turn):
                self.assertEqual(context(map_layers, ego_of(30.0, 0.0), turn), expected)

    def test_zero_connected_segments_use_bend_rule(self):
        self.assert_counts_use_bend_rule(0)

    def test_one_connected_segment_uses_bend_rule(self):
        self.assert_counts_use_bend_rule(1)

    def test_two_segments_stay_a_bend(self):
        map_layers = layers([(0.0, 0.0, 2)])
        self.assertEqual(context(map_layers, ego_of(30.0, 0.0), 0.0), "straight_road")
        self.assertEqual(context(map_layers, ego_of(30.0, 0.0), -30.0), "curve")
        self.assertEqual(context(map_layers, ego_of(30.0, 0.0), 30.0), "curve")

    def test_three_and_more_segments_are_unchanged(self):
        self.assertEqual(context(layers([(0.0, 0.0, 3)]), ego_of(30.0, 0.0)), "t_junction")
        self.assertEqual(context(layers([(0.0, 0.0, 4)]), ego_of(30.0, 0.0)), "cross_intersection")
        self.assertEqual(context(layers([(0.0, 0.0, 5)]), ego_of(30.0, 0.0)), "cross_intersection")

    def test_no_intersection_within_the_range_keeps_the_bend_rules(self):
        map_layers = layers([(0.0, 120.0, 0)])
        self.assertEqual(context(map_layers, ego_of(0.0, 0.0), 0.0), "straight_road")
        self.assertEqual(context(map_layers, ego_of(0.0, 0.0), -30.0), "curve")


class LayerIntegrityTests(unittest.TestCase):
    def broken(self, layer, geometry):
        map_layers = layers([(0.0, 0.0, 3)])
        map_layers[layer] = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")
        return map_layers

    def test_empty_layers_raise(self):
        for layer in ("intersections", "road_segments"):
            with self.subTest(layer=layer):
                with self.assertRaises(facts.FactsError) as caught:
                    context(self.broken(layer, []), ego_of(30.0, 0.0))
                self.assertIn("no features", str(caught.exception))

    def test_missing_geometry_raises(self):
        map_layers = layers([(0.0, 0.0, 3)])
        geometries = list(map_layers["road_segments"].geometry) + [None]
        with self.assertRaises(facts.FactsError) as caught:
            context(self.broken("road_segments", geometries), ego_of(30.0, 0.0))
        self.assertIn("missing", str(caught.exception))

    def test_invalid_geometry_raises(self):
        bowtie = Polygon([lonlat_of(-5.0, -5.0), lonlat_of(5.0, -5.0),
                          lonlat_of(-5.0, 5.0), lonlat_of(5.0, 5.0)])
        self.assertFalse(bowtie.is_valid)
        for layer in ("intersections", "road_segments"):
            with self.subTest(layer=layer):
                with self.assertRaises(facts.FactsError) as caught:
                    context(self.broken(layer, [bowtie]), ego_of(30.0, 0.0))
                self.assertIn("invalid", str(caught.exception))


class CoordinateSystemTests(unittest.TestCase):
    def test_missing_map_crs_raises(self):
        map_layers = layers([(0.0, 0.0, 3)])
        map_layers["intersections"].crs = None
        self.assertIsNone(map_layers["intersections"].crs)
        with self.assertRaises(facts.FactsError):
            corners_of(map_layers, ego_of(0.0, 0.0))

    def test_non_metric_projected_crs_raises(self):
        map_layers = layers([(0.0, 0.0, 3)])
        map_layers["intersections"] = map_layers["intersections"].set_crs("EPSG:2227", allow_override=True)
        with self.assertRaises(facts.FactsError):
            corners_of(map_layers, ego_of(0.0, 0.0))

    def test_metric_crs_is_utms_zone_of_the_point(self):
        from pyproj import CRS

        utm = facts.metric_crs("EPSG:4326", -79.96, 40.42)
        self.assertEqual(utm.to_epsg(), CRS.from_dict({"proj": "utm", "zone": 17, "south": False}).to_epsg())
        self.assertEqual(facts.metric_crs("EPSG:26917", -79.96, 40.42).to_epsg(), 26917)
        with self.assertRaises(facts.FactsError):
            facts.metric_crs(None, -79.96, 40.42)


if __name__ == "__main__":
    unittest.main()