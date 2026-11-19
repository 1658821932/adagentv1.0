"""Tool01 accepts an ego route that starts inside a junction and enters a lane."""
from unittest import mock

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, Polygon
import pytest

from tools.tool01_failure_analyzer import lane_behavior
from tools.tool01_failure_analyzer import structure_facts as sf


def layers():
    intersection = gpd.GeoDataFrame(
        {"lane_group_fids": [""]},
        geometry=[Polygon([(0, -4), (4, -4), (4, 4), (0, 4)])],
        index=[77], crs="EPSG:4326")
    lanes = gpd.GeoDataFrame(
        {"lane_group_fid": [10], "lane_fid": [100]},
        geometry=[Polygon([(5, -2), (20, -2), (20, 2), (5, 2)])],
        index=[1], crs="EPSG:4326")
    baselines = gpd.GeoDataFrame(
        {"lane_fid": [100.0]},
        geometry=[LineString([(5, 0), (20, 0)])],
        index=[1], crs="EPSG:4326")
    segments = gpd.GeoDataFrame(
        {"lane_group_fids": ["10"]},
        geometry=[LineString([(5, 0), (20, 0)])],
        index=["segment-1"], crs="EPSG:4326")
    connectors = gpd.GeoDataFrame(
        {"exit_lane_group_fid": [11], "entry_lane_group_fid": [10],
         "turn_type_fid": [2], "intersection_fid": [77]},
        geometry=[LineString([(1, 0), (5, 0)])],
        index=[5], crs="EPSG:4326")
    return {
        "intersections": intersection,
        "lanes_polygons": lanes,
        "baseline_paths": baselines,
        "road_segments": segments,
        "lane_connectors": connectors,
    }


def test_route_can_start_in_intersection_then_enter_lane():
    map_layers = layers()
    frames = [{"ego2global": [[1, 0, 0, 0]]}] * 5
    ego = {
        "positions": [(1, 0), (2, 0), (3, 0), (6, 0), (12, 0)],
        "yaw": [0.0] * 5,
    }
    with mock.patch.object(sf, "map_file", return_value="map.gpkg"),          mock.patch.object(sf, "map_layer", side_effect=lambda _gpd, _path, layer, **_: map_layers[layer]),          mock.patch.object(sf, "metric_crs", side_effect=lambda crs, *_: CRS.from_epsg(4326)),          mock.patch.object(sf, "utm_to_lonlat", side_effect=lambda x, y, _bounds: (x, y)),          mock.patch.object(sf, "ego_geometry", return_value=(4.6, 1.9, 0.5)):
        result = lane_behavior.analyze("map", frames, ego, {}, None, 4, collision_confirmed=False)
    assert result["map_context"] == "straight_road"
    assert result["ego_lane_keeping"] == "keep"
    assert result["other_origin"] == "none"
    assert result["other_lane_keeping"] == "none"
    assert result["interaction"] == "single_vehicle"
