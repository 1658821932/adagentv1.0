"""The real sample always reviews its first recorded-route junction."""
import json
from pathlib import Path
import pytest
from tools.tool04_edit_planner import mapfacts

SAMPLE = Path('data/navtrain_ep_per1/samples/5863c4eb96825e85/metadata/frames.json')


def read_facts(context):
    if mapfacts.python_path() is None or not SAMPLE.is_file():
        pytest.skip('simulator interpreter or sample scene unavailable')
    frames = json.loads(SAMPLE.read_text())
    result = mapfacts.collect(frames, dict(map_context=context, ego_action='turn_right',
                                          other_origin='right_arm', other_action='turn_left'))
    assert result['available'] is True, result
    return result


@pytest.fixture(scope='module')
def t_junction_facts():
    return read_facts('t_junction')


def test_first_recorded_junction_is_fixed_before_shape_check(t_junction_facts):
    facts = t_junction_facts['facts']
    assert facts['selection_policy'] == 'first_real_junction_on_recorded_route'
    assert facts['target_junction_fid'] == 45326
    assert facts['target_entry_distance_m'] == 0
    assert facts['target_junction']['physical_branches'] == 4


def test_t_junction_accepts_the_first_four_arm_cross_and_does_not_jump(t_junction_facts):
    assert t_junction_facts['checks']['road'] == 'pass'
    assert t_junction_facts['facts']['target_junction_fid'] == 45326
    assert t_junction_facts['facts']['target_junction']['physical_branches'] == 4


def test_real_lane_and_connector_directions_are_retained(t_junction_facts):
    facts = t_junction_facts['facts']
    assert facts['lanes_with_direction'] > 0
    assert facts['connectors_with_turn'] > 0
    assert facts['ego_lane_group_fid'] is not None


def test_side_view_target_stays_on_the_selected_recorded_junction(t_junction_facts):
    rows = t_junction_facts['facts']['frame_target_distance']
    assert rows and rows[0]['distance_to_target_m'] == 0


def test_different_required_shape_does_not_change_target():
    result = read_facts('cross_intersection')
    assert result['facts']['target_junction_fid'] == 45326
    assert result['checks']['road'] == 'pass'


def test_straight_or_curve_target_does_not_select_a_junction():
    result = read_facts('straight_road')
    facts = result['facts']
    assert facts['selection_policy'] == 'no_junction_target'
    assert facts['target_junction_fid'] is None
    assert result['checks'] == {'road': 'unknown', 'lanes': 'unknown', 'space': 'unknown'}
