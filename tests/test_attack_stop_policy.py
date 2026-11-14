"""Repeated failure stops require both recurrence and lack of measured progress."""
from mcp_servers.scene_editing.core.adversarial.attack_stop_policy import repetition


def test_same_invalid_request_stops_quickly():
    previous = None
    previous = repetition(previous, 'construct_attack_candidates', dict(status='invalid', error='hold_duration_s must fit window'))
    assert previous[1] == 1
    previous = repetition(previous, 'construct_attack_candidates', dict(status='invalid', error='hold_duration_s must fit window'))
    assert previous[1] == 2
    assert previous[0][0] == 'invalid'


def test_candidate_cost_improvement_resets_repetition():
    first = dict(status='fallback', failed_checks=['dense.keyframe_2'], candidates=[dict(guidance_cost=4.0)])
    second = dict(status='fallback', failed_checks=['dense.keyframe_2'], candidates=[dict(guidance_cost=2.0)])
    third = dict(status='fallback', failed_checks=['dense.keyframe_2'], candidates=[dict(guidance_cost=2.1)])
    previous = repetition(None, 'construct_attack_candidates', first)
    previous = repetition(previous, 'construct_attack_candidates', second)
    assert previous[1] == 1
    previous = repetition(previous, 'construct_attack_candidates', third)
    assert previous[1] == 2


def test_unproductive_same_failure_reaches_limit_but_new_failure_resets():
    result = dict(status='fallback', failed_checks=['dense.keyframe_2'], candidates=[dict(guidance_cost=3.0)])
    previous = repetition(None, 'construct_attack_candidates', result)
    previous = repetition(previous, 'construct_attack_candidates', result)
    previous = repetition(previous, 'construct_attack_candidates', result)
    assert previous[1] == 3
    changed = repetition(previous, 'construct_attack_candidates', dict(status='invalid', error='different'))
    assert changed[1] == 1
