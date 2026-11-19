"""Connector expansion in the trajectory frame and native-frame route evidence."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

CORE = Path(__file__).resolve().parents[1] / 'mcp_servers/scene_editing/core/adversarial'
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from connector_route import connector_passage, plan_connector_route
from trajectory_waypoints import validate_waypoints


@pytest.fixture
def scene_route(tmp_path):
    metadata = tmp_path / 'metadata'
    metadata.mkdir()
    features = {
        'in': dict(type='LANE_SURFACE_STREET', polyline=[[0, 0], [0, 10]]),
        'turn': dict(type='LANE_SURFACE_UNSTRUCTURE', polyline=[[0, 10], [0, 14], [-4, 18]],
                     entry_lanes=['in'], exit_lanes=['out']),
        'out': dict(type='LANE_SURFACE_STREET', polyline=[[-4, 18], [-16, 18]]),
    }
    (metadata / 'map_features.json').write_text(json.dumps(features))
    lane_polygons = {
        'in': [[0, -1], [10, -1], [10, 1], [0, 1]],
        'turn': [[10, -1], [18.5, -1], [18.5, 4.5], [10, 4.5]],
        'out': [[17, 4], [19, 4], [19, 16], [17, 16]],
    }
    (metadata / 'map_drivable_official.json').write_text(json.dumps(dict(features={
        f'feature/{i}': dict(token=token, polygon=polygon) for i, (token, polygon) in enumerate(lane_polygons.items())
    })))
    frames = [dict(ego2global=[[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])]
    route = dict(connector_id='turn', approach_m=2., exit_m=2., speed_mps=2.)
    return tmp_path, frames, route


def test_connector_waypoints_use_scene_frame_and_legal_time_windows(scene_route):
    scene, frames, route = scene_route
    plan = plan_connector_route(scene, frames, route, 1., 8.)
    assert plan['entry_lane'] == 'in'
    assert plan['exit_lane'] == 'out'
    assert plan['waypoints'][0]['time_s'] == 1.
    assert plan['waypoints'][0]['position_m'] == pytest.approx([8., 0.])
    assert plan['waypoints'][0]['yaw_rad'] == pytest.approx(0.)
    assert plan['waypoints'][-1]['position_m'] == pytest.approx([18., 6.])
    assert plan['nominal_connector_entry_s'] == pytest.approx(2.)
    assert plan['nominal_connector_exit_s'] == pytest.approx(6.83)
    validate_waypoints(plan['waypoints'], 1., 8.)


def test_missing_connector_and_short_horizon_return_actionable_errors(scene_route):
    scene, frames, route = scene_route
    with pytest.raises(ValueError, match='explicit waypoints'):
        plan_connector_route(scene, frames, dict(route, connector_id='unknown'), 0., 8.)
    with pytest.raises(ValueError, match='only 4.00s is editable'):
        plan_connector_route(scene, frames, route, 1., 5.)


def test_connector_evidence_checks_order_using_official_scene_local_polygons(scene_route):
    scene, frames, route = scene_route
    plan = plan_connector_route(scene, frames, route, 0., 8.)
    times = np.array([0., 3., 7.])
    xy = np.array([[8., 0.], [13., 0.], [18., 6.]])
    passage = connector_passage(scene, plan, times, xy)
    assert passage['passes_in_order']
    assert [row['first_time_s'] for row in passage['lane_passages']] == [0., 3., 7.]
    assert not connector_passage(scene, plan, times, xy[::-1])['passes_in_order']
