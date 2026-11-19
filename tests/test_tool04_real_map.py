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
    assert facts['selection_policy'] == 'recorded_route_lane_corridor'
    assert facts['target_junction_fid'] is None
    assert result['checks'] == {'road': 'pass', 'lanes': 'unknown', 'space': 'pass'}


def test_straight_road_oncoming_lane_is_checked_from_route_lanes():
    sample = Path('data/navtrain_ep_per1/samples/947d66f0231f503b/metadata/frames.json')
    if mapfacts.python_path() is None or not sample.is_file():
        pytest.skip('simulator interpreter or straight-road sample unavailable')
    frames = json.loads(sample.read_text())
    result = mapfacts.collect(frames, dict(
        map_context='straight_road', ego_action='straight', ego_lane_keeping='left_offset',
        other_origin='oncoming', other_action='straight', other_lane_keeping='keep'))
    assert result['available'] is True, result
    assert result['facts']['selection_policy'] == 'recorded_route_lane_corridor'
    assert result['facts']['ego_lane_coverage_fraction'] >= 0.5
    assert result['facts']['target_lane_candidates']
    assert any(candidate['direction_dot'] <= -0.5
               for candidate in result['facts']['target_lane_candidates'])
    assert result['checks'] == {'road': 'pass', 'lanes': 'pass', 'space': 'pass'}


def test_recorded_right_turn_is_preserved_and_forms_the_target_interaction():
    sample = Path('data/navtrain_ep_per1/samples/6ea16dba8b16523b/metadata/frames.json')
    if mapfacts.python_path() is None or not sample.is_file():
        pytest.skip('simulator interpreter or ranked sample unavailable')
    frames = json.loads(sample.read_text())
    result = mapfacts.collect(frames, dict(
        map_context='t_junction', ego_action='turn_right', ego_lane_keeping='left_offset',
        other_origin='right_arm', other_action='turn_left', other_lane_keeping='left_offset'))
    assert result['available'] is True, result
    assert result['facts']['target_junction_fid'] == 17589
    assert result['facts']['ego_exact_action_match'] is True
    assert result['facts']['ego_actual_connector_fids'] == [18210]
    assert result['facts']['ego_path_options'] == [dict(
        connector_fid=18210, turn='right', entry_lane_fid=19766, entry_lane_group_fid=18934)]
    assert 18184 in result['facts']['ego_unused_approach_connectors']
    assert result['facts']['target_other_connectors']
    assert any(pair['interaction_compatible'] is True and pair['on_official_network'] is True
               for pair in result['facts']['interaction_path_pairs'])
    assert result['checks'] == {'road': 'pass', 'lanes': 'pass', 'space': 'pass'}


def test_later_route_junction_is_used_when_the_first_junction_lacks_the_right_arm():
    sample = Path('data/navtrain_ep_per1/samples/bb137ceaa889594b/metadata/frames.json')
    if mapfacts.python_path() is None or not sample.is_file():
        pytest.skip('simulator interpreter or ranked sample unavailable')
    frames = json.loads(sample.read_text())
    result = mapfacts.collect(frames, dict(
        map_context='t_junction', ego_action='turn_right', ego_lane_keeping='left_offset',
        other_origin='right_arm', other_action='turn_left', other_lane_keeping='left_offset'))
    assert result['available'] is True, result
    assert result['facts']['target_junction_fid'] == 47148
    assert [attempt['intersection_fid'] for attempt in result['facts']['junction_attempts']] == [46933, 47148]
    assert result['facts']['junction_attempts'][0]['checks']['lanes'] == 'fail'
    assert result['facts']['junction_attempts'][1]['checks']['lanes'] == 'pass'
    assert result['facts']['target_other_connectors'][0]['connector_fid'] == 49447
    assert result['checks'] == {'road': 'pass', 'lanes': 'pass', 'space': 'pass'}


def test_same_lane_lead_uses_the_ego_approach_lane_not_a_junction_arm_connector():
    sample = Path('data/navtrain_ep_per1/samples/5bb17fd2b64d503a/metadata/frames.json')
    if mapfacts.python_path() is None or not sample.is_file():
        pytest.skip('simulator interpreter or same-lane sample unavailable')
    frames = json.loads(sample.read_text())
    result = mapfacts.collect(frames, dict(
        map_context='cross_intersection', ego_action='straight', ego_lane_keeping='right_offset',
        other_origin='ahead_same_lane', other_action='straight', other_lane_keeping='keep'))
    assert result['available'] is True, result
    assert result['checks']['road'] == 'pass'
    assert result['checks']['lanes'] == 'pass'
    assert result['checks']['space'] == 'pass'
    assert result['facts']['target_other_connectors'] == []
    assert any('no junction-arm connector is required' in reason for reason in result['reasons'])
