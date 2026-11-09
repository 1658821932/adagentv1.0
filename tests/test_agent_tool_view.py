"""Shared Agent views are actionable without changing editing evidence."""
from copy import deepcopy
import json
from pathlib import Path
import sys

CORE = Path(__file__).resolve().parents[1] / 'mcp_servers/scene_editing/core/adversarial'
sys.path.insert(0, str(CORE))

from agent_tool_view import candidate_card, with_decision_summary
from ego_context_view import candidate_summary
from route_edit_contract import EXAMPLE as ROUTE_EXAMPLE, tool as route_tool, validate as validate_route
from speed_profile_contract import EXAMPLE as SPEED_EXAMPLE, tool as speed_tool, validate as validate_speed


def speed_arguments():
    return dict(parent_candidate_id='candidate_001', edit_mode='speed_schedule',
        operation='set_speed_targets', start_time_s=7.5,
        targets=[dict(time_s=10.5, speed_mps=0.0)], preview_only=True,
        reason='Preview a later stop on the parent route.', speed_schedule_version='state_continuity.v4')


def test_minimal_examples_are_valid_and_embedded_in_tool_descriptions():
    validate_speed(deepcopy(SPEED_EXAMPLE))
    validate_route(deepcopy(ROUTE_EXAMPLE))
    assert json.dumps(SPEED_EXAMPLE, separators=(',', ':')) in speed_tool()['function']['description']
    assert json.dumps(ROUTE_EXAMPLE, separators=(',', ':')) in route_tool()['function']['description']
    assert SPEED_EXAMPLE['preview_only'] and ROUTE_EXAMPLE['preview_only']


def test_route_exhaustion_returns_complete_conditional_preview_without_mutating_request():
    arguments = speed_arguments()
    before = deepcopy(arguments)
    raw = dict(status='preview', candidate_created=False, feasible=False,
        error_code='route_exhausted', error='route_exhausted: no extrapolation',
        error_summary='The known route is too short.',
        feasibility_limits=dict(missing_route_length_m=2.43))
    result = with_decision_summary(raw, arguments)
    summary = result['decision_summary']
    assert arguments == before
    assert result['error_code'] == 'route_exhausted'
    assert summary['outcome'] == 'preview_infeasible'
    assert summary['primary_failure']['check'] == 'route_exhausted'
    assert summary['candidate_id'] is None
    recommendation = summary['recommended_previews'][0]
    assert recommendation['tool'] == 'edit_speed_profile'
    assert recommendation['arguments']['path_source'] == 'recorded_baseline'
    assert recommendation['arguments']['preview_only'] is True
    assert 'edit_mode' not in recommendation['arguments']
    assert 'speed_schedule_version' not in recommendation['arguments']
    assert 'condition' in recommendation
    validate_speed(recommendation['arguments'])


def test_acceleration_limit_becomes_a_complete_delayed_target_preview():
    arguments = speed_arguments()
    arguments['start_time_s'] = 5.5
    arguments['targets'] = [dict(time_s=6.5, speed_mps=0.8), dict(time_s=7.5, speed_mps=0.8)]
    before = deepcopy(arguments)
    raw = dict(status='preview', feasible=False, candidate_created=False,
        error_code='target_unreachable', error='target_unreachable: interval exceeds cap',
        feasibility_limits=dict(interval_s=[5.5, 6.5], minimum_transition_duration_s=1.025623291,
            scene_end_s=11.499597))
    result = with_decision_summary(raw, arguments)
    recommendation = result['decision_summary']['recommended_previews'][0]
    adjusted = recommendation['arguments']['targets']
    assert arguments == before
    assert adjusted[0]['time_s'] == arguments['start_time_s']+1.025623291
    assert adjusted[0]['speed_mps'] == 0.8 and adjusted[1] == arguments['targets'][1]
    validate_speed(recommendation['arguments'])


def test_passed_preview_has_an_exact_create_call_but_no_submission_claim():
    result = with_decision_summary(dict(status='preview', feasible=True,
        candidate_created=False, checks=dict(road=True)), speed_arguments())
    summary = result['decision_summary']
    assert summary['outcome'] == 'preview_passed'
    assert 'can_submit' not in summary
    call = summary['create_candidate_call']
    assert call['tool'] == 'edit_speed_profile'
    assert call['arguments']['preview_only'] is False
    assert call['arguments']['targets'] == speed_arguments()['targets']
    validate_speed(call['arguments'])


def test_created_candidate_summary_combines_reference_and_task_failures():
    result = with_decision_summary(dict(status='reference_screen_failed',
        candidate_id='candidate_002', reference_screen_passed=False,
        task_contract_passed=False, failed_checks=['continuous_motion'],
        task_failed_checks=['native_replay.keyframe_1']), speed_arguments())
    summary = result['decision_summary']
    assert summary['outcome'] == 'candidate_checks_failed'
    assert summary['candidate_created'] is True
    assert summary['failed_checks'] == ['continuous_motion', 'native_replay.keyframe_1']
    assert summary['primary_failure']['check'] == 'continuous_motion'


def test_route_failure_suggests_previewing_explicit_progress_policy():
    arguments = dict(parent_candidate_id='candidate_001', edit_mode='route_edit',
        start_time_s=3.0, left_offset_m=0.2, transition_duration_s=2.0,
        return_duration_s=2.0, end_behavior='return_to_route', reason='Avoid and return')
    result = with_decision_summary(dict(status='not_feasible', feasible=False,
        candidate_created=False, error_code='route_exhausted'), arguments)
    recommendation = result['decision_summary']['recommended_previews'][0]
    assert recommendation['tool'] == 'edit_route'
    assert recommendation['arguments']['speed_policy'] == 'preserve_progress'
    validate_route(recommendation['arguments'])


def test_attack_and_ego_candidate_cards_share_identity_eligibility_and_metrics():
    attack = candidate_card(dict(candidate_id='candidate_001', parent_candidate_id='seed_000',
        reference_screen_passed=True, task_contract_passed=False,
        failed_checks=['road_surface'], task_failed_checks=['keyframe_1'],
        min_ego_clearance_m=0.4, max_acceleration_mps2=3.0))
    ego = candidate_summary(dict(candidate_id='edit_001', parent_id='constructor_001',
        status='reference_screen_passed', failed_checks=[], coverage=dict(no_cross_lines='unknown'),
        min_target_clearance_m=1.2, max_acceleration_mps2=2.0))
    common = ('candidate_id', 'parent_candidate_id', 'reference_screen_passed',
              'task_contract_passed', 'failed_checks', 'max_acceleration_mps2')
    assert all(key in attack and key in ego for key in common)
    assert ego['parent_candidate_id'] == 'constructor_001'
    assert ego['reference_screen_passed'] is True
    assert ego['unknown_checks'] == ['no_cross_lines']
    assert attack['min_ego_clearance_m'] == 0.4
    assert ego['min_target_clearance_m'] == 1.2
