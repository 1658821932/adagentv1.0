import json
import pytest
from pydantic import ValidationError
from mcp_servers.scene_editing.core.adversarial.business_outcome import bounded_outcome
from mcp_servers.scene_editing.adapter.schemas import AttackResult, EgoInput


def test_attack_refusal_preserves_actual_finish_reason(tmp_path):
    directory=tmp_path/'attack'
    archive=directory/'trajectory_editing/output.json'
    archive.parent.mkdir(parents=True)
    archive.write_text(json.dumps(dict(status='not_submitted',stop_reason='finished_without_submission')))
    call=directory/'trajectory_editing/tool_calls/turn_003/result.json'
    call.parent.mkdir(parents=True)
    call.write_text(json.dumps(dict(status='finished',reason='All tried merges failed heading alignment')))
    result=bounded_outcome('attack_editing',dict(status='not_submitted'),directory)
    assert result['reason']=='All tried merges failed heading alignment'
    assert AttackResult.model_validate(result).model_dump()['status']=='unable_to_complete'
    with pytest.raises(ValidationError):
        EgoInput.model_validate(dict(attack_result=result,target_effect='Stop safely'))


def test_execution_error_cannot_become_attack_refusal(tmp_path):
    archive=tmp_path/'trajectory_editing/output.json';archive.parent.mkdir()
    archive.write_text(json.dumps(dict(status='not_submitted',stop_reason='request_or_execution_failed')))
    with pytest.raises(RuntimeError,match='normally'):
        bounded_outcome('attack_editing',dict(status='not_submitted'),tmp_path)


def test_object_selection_can_decline_without_fake_trajectory(tmp_path):
    archive=tmp_path/'object_selection/output.json';archive.parent.mkdir()
    archive.write_text(json.dumps(dict(status='not_submitted',stop_reason='object_selection_budget_exhausted')))
    result=bounded_outcome('attack_editing',dict(status='not_submitted',search_stage='object_selection',
        reason='No initial placement passed the observed-prefix checks'),tmp_path)
    assert result['status']=='unable_to_complete'
    assert result['evidence_paths']==[str(archive)]
    assert 'frozen_scene_path' not in result


def test_nested_constructor_failures_are_reported(tmp_path):
    archive=tmp_path/'trajectory_editing/output.json';archive.parent.mkdir()
    archive.write_text(json.dumps(dict(status='not_submitted',stop_reason='action_budget_exhausted')))
    call=tmp_path/'trajectory_editing/tool_calls/turn_01/result.json';call.parent.mkdir(parents=True)
    call.write_text(json.dumps(dict(candidates=[dict(checks=dict(road_surface=False,motion=True),
         task_failed_checks=['native_replay.keyframe_2'])])))
    result=bounded_outcome('attack_editing',dict(status='not_submitted'),tmp_path)
    assert result['failed_requirements']==['native_replay.keyframe_2','road_surface']


def test_refusal_preserves_attempts_before_object_reselection(tmp_path):
    child=tmp_path/'object_revision_01'
    (tmp_path/'output.json').write_text(json.dumps(dict(followup_run=str(child))))
    archive=child/'trajectory_editing/output.json';archive.parent.mkdir(parents=True)
    archive.write_text(json.dumps(dict(status='not_submitted',stop_reason='finished_without_submission')))
    for directory in [tmp_path,child]:
        result=directory/'trajectory_editing/tool_calls/turn_01/result.json'
        result.parent.mkdir(parents=True);result.write_text(json.dumps(dict(status='no_feasible_goal')))
    result=bounded_outcome('attack_editing',dict(status='not_submitted'),child)
    assert len(result['attempts'])==2
    assert str(tmp_path/'output.json') in result['evidence_paths']
    assert result['failed_requirements_scope']=='observed_in_attempted_candidates'


def test_cumulative_constructor_exhaustion_reports_business_refusal_without_model_finish(tmp_path):
    archive=tmp_path/'trajectory_editing/output.json';archive.parent.mkdir()
    stop='constructor_budget_exhausted_without_qualified_candidate'
    reason='Construction budget exhausted without a trajectory meeting the fixed task.'
    archive.write_text(json.dumps(dict(status='not_submitted',stop_reason=stop,reason=reason,
        termination_evidence=dict(failed_checks=['road_surface','native_replay.keyframe_1'],
            constructor_budget=dict(evaluations=1000,remaining_evaluations=0)))))
    result=bounded_outcome('attack_editing',dict(status='not_submitted'),tmp_path)
    assert result['status']=='unable_to_complete'
    assert result['reason']==reason and result['stop_reason']==stop
    assert result['failed_requirements']==['native_replay.keyframe_1','road_surface']
    assert not result['global_infeasibility_proven']
    assert AttackResult.model_validate(result).model_dump()['status']=='unable_to_complete'
