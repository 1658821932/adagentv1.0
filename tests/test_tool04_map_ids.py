"""Identifier normalisation and arm/Direction helpers used by the map reader."""
import pytest
from tools.tool04_edit_planner.map_ids import (arm_position, key, origin_from_arm,
                                               tangent_deg, turn_from_tangents)


@pytest.mark.parametrize("value,expected", [
    (46832, 46832), (46832.0, 46832), ("47592.0", 47592), ("  47592 ", 47592),
    ("abc123", "abc123"), (None, None), (float("nan"), None), (float("inf"), None),
    ("", None), ("nan", None), ("None", None), (True, None), (46832.5, None),
])
def test_key_normalises_float_and_text_ids_without_losing_text(value, expected):
    assert key(value) == expected


def test_lane_and_baseline_ids_join_after_normalisation():
    assert key(46832) == key(46832.0) == key("46832")


def test_turn_uses_both_end_tangents():
    assert turn_from_tangents(0.0, 5.0) == "straight"
    assert turn_from_tangents(0.0, 80.0) == "left"
    assert turn_from_tangents(0.0, -80.0) == "right"
    assert turn_from_tangents(170.0, -170.0) == "straight"
    assert turn_from_tangents(None, 90.0) is None


def test_arm_position_is_physical_not_same_opposite():
    assert arm_position(10.0, 0.0) == "ahead"
    assert arm_position(180.0, 0.0) == "behind"
    assert arm_position(90.0, 0.0) == "left"
    assert arm_position(-90.0, 0.0) == "right"
    assert arm_position(90.0, 90.0) == "ahead"


def test_facing_traffic_ahead_is_oncoming_and_same_direction_ahead_is_same_lane():
    # traffic coming towards the ego from the arm in front of it
    assert origin_from_arm("ahead", 180.0, 0.0) == "oncoming"
    # traffic moving with the ego from the arm in front of it
    assert origin_from_arm("ahead", 0.0, 0.0) == "ahead_same_lane"
    # traffic moving with the ego from behind
    assert origin_from_arm("behind", 0.0, 0.0) == "behind_same_lane"
    # side arms keep the arm identity, relative to the ego's travel direction
    assert origin_from_arm("left", 90.0, 0.0) == "left_arm"
    assert origin_from_arm("right", -90.0, 0.0) == "right_arm"
    # the same physical bucket must not be called "oncoming" just because the name says so
    assert origin_from_arm("ahead", 0.0, 0.0) != "oncoming"
    assert origin_from_arm("left", 90.0, 0.0) != "oncoming"


def test_undecidable_arms_stay_unknown():
    assert origin_from_arm("ahead", 90.0, 0.0) is None       # perpendicular crossing traffic
    assert origin_from_arm("behind", 180.0, 0.0) is None      # no rear-oncoming class in tool01
    assert origin_from_arm(None, 0.0, 0.0) is None
    assert origin_from_arm("ahead", None, 0.0) is None


def test_tangent_deg_matches_geometry():
    assert tangent_deg((0, 0), (1, 0)) == 0.0
    assert tangent_deg((0, 0), (0, 1)) == 90.0
    assert tangent_deg((0, 0), (0, 0)) is None
