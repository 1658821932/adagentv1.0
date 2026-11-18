"""Lane-keeping and direction are independent measurements."""
import math

from tools.tool01_failure_analyzer.lane_behavior import (
    classify_interaction,
    direction_action,
    lane_keeping,
)
from tools.tool01_failure_analyzer.language import ego_cause


def test_lane_keeping_has_three_semantic_values():
    assert lane_keeping(1.2) == "left_offset"
    assert lane_keeping(-1.2) == "right_offset"
    assert lane_keeping(0.2) == "keep"
    assert lane_keeping(None) == "unknown"


def test_action_uses_heading_not_speed():
    sequence = [
        {"index": index, "world_yaw": 0.0, "speed": 30.0 - index * 5.0}
        for index in range(3)
    ]
    assert direction_action(sequence) == "straight"
    sequence[-1]["world_yaw"] = math.radians(40.0)
    assert direction_action(sequence) == "turn_left"


def test_action_window_stops_before_post_collision_motion():
    sequence = [
        {"index": index, "world_yaw": math.radians(-15.0 * max(0, index - 2))}
        for index in range(5)
    ]
    assert direction_action(sequence, through=2) == "straight"
    assert direction_action(sequence, through=4) == "turn_right"


def test_cut_in_requires_same_direction_lane_entry_and_an_ahead_target():
    assert classify_interaction("left_adjacent", True, True, False, True) == "cut_in"
    assert classify_interaction("left_adjacent", True, False, False, True) == "side_swipe"
    assert classify_interaction("left_adjacent", True, True, False, False) == "side_swipe"
    assert classify_interaction("right_adjacent", False, True, False, True) == "opposite_direction"


def test_ego_lane_change_does_not_turn_the_other_vehicle_into_a_cut_in():
    assert classify_interaction("left_adjacent", True, True, True, True) == "side_swipe"


def test_no_target_cause_keeps_ego_motion_lane_state_and_outcome():
    attack = {"map_context": "straight_road", "ego_action": "straight", "ego_lane_keeping": "left_offset",
              "other_origin": "none", "other_action": "none", "other_lane_keeping": "none"}
    text = ego_cause("single_vehicle", attack, "straight", "left_offset", "road_departure")
    assert "直行" in text and "向左偏离" in text and "驶出可行驶道路" in text
