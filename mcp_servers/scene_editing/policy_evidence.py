"""Shared decision rule for exact-scene VADv2 failure evidence."""

import hashlib

EVALUATION_WINDOW = (1.5, 7.5)
POLICY_EVALUATION_VERSION = 'vad_collision_or_drivable_area.v1'


def in_policy_window(time_s):
    start, end = EVALUATION_WINDOW
    return type(time_s) in (int, float) and start < time_s <= end


def sampled_target_collision(collision):
    return (collision.get('target_collision_detected') is True
            and collision.get('initial_history_target_contact') is False
            and in_policy_window(collision.get('first_target_collision_time_s')))


def confirmed_road_failure(road):
    return (road.get('status') == 'failed'
            and in_policy_window(road.get('first_violation_time_s')))


def has_policy_failure(evidence, road_map):
    """Recheck the saved decision, including map identity for a road-only failure."""
    if evidence.get('policy_attack_success') is not True:
        return False
    if sampled_target_collision(evidence.get('collision_report') or {}):
        return True
    evaluation = evidence.get('policy_evaluation') or {}
    road = evaluation.get('drivable_area') or {}
    return (evaluation.get('schema_version') == POLICY_EVALUATION_VERSION
            and confirmed_road_failure(road)
            and road_map.is_file()
            and road.get('map_sha256') == hashlib.sha256(road_map.read_bytes()).hexdigest())
