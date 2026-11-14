"""Action preflight and compact evidence feedback preserve actionable measurements."""
from types import SimpleNamespace
import pytest

from mcp_servers.scene_editing.core.adversarial.action_validation import prevalidate_action, invalid_action_result
from mcp_servers.scene_editing.core.adversarial.agent_tool_view import compact_action_feedback


def test_hold_duration_must_fit_its_window():
    arguments = dict(objective=dict(keyframes=[dict(time_window_s=[5., 6.], hold_duration_s=1.5)]))
    with pytest.raises(ValueError, match='fit the keyframe window'):
        prevalidate_action('construct_attack_candidates', arguments)
    result = invalid_action_result('construct_attack_candidates', arguments, 'invalid')
    assert result['decision_summary']['parameter_hints'][0]['maximum_hold_duration_s'] == 1.


def test_seed_edit_requires_an_objective_parent():
    editor = SimpleNamespace(nodes={'seed_000': {'result': {}}})
    with pytest.raises(ValueError, match='declared objective'):
        prevalidate_action('edit_speed_profile', dict(parent_candidate_id='seed_000'), editor)


def test_compact_feedback_keeps_failure_measurements_and_archive_not_raw_geometry():
    raw = {
        'candidate_id': 'candidate_008', 'status': 'fallback', 'archive_ref': '/tmp/turn_02',
        'geometry_feedback': {'large': 'raw evidence remains archived'},
        'construction_goal': {
            'dense': {
                'failed_checks': ['keyframe_2'],
                'keyframes': [{
                    'passed': False, 'evaluated_interval_s': [5., 5.5],
                    'max_constraint_error': .2, 'requested': {'speed_mps': [0, 1]},
                    'actual': {'speed_mps': 1.2}, 'actual_ranges': {'speed_mps': [1., 1.3]},
                }],
            },
            'native_replay': {'failed_checks': []},
        },
    }
    result = compact_action_feedback(raw)
    assert result['candidate_id'] == 'candidate_008'
    assert result['archive_ref'] == '/tmp/turn_02'
    assert result['task_failed_checks'] == ['dense.keyframe_2']
    assert result['failure_measurements'][0]['max_constraint_error'] == .2
    assert 'geometry_feedback' not in result
