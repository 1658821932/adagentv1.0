import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from mcp_servers.scene_editing.core.adversarial.business_outcome import bounded_outcome, attack_retry_policy
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


@pytest.mark.parametrize('stop', ['native_tool_protocol_failed','invalid_or_missing_tool_call',
    'model_stopped_without_submission','invalid_call_budget_exhausted',
    'repeated_invalid_tool_call','interrupted_by_operator'])
def test_protocol_or_interruption_cannot_become_defense_not_found(tmp_path,stop):
    (tmp_path/'summary.json').write_text(json.dumps(dict(status='not_submitted',stop_reason=stop)))
    with pytest.raises(RuntimeError,match='normally'):
        bounded_outcome('ego_editing',dict(status='not_submitted'),tmp_path)


def test_object_selection_can_decline_without_fake_trajectory(tmp_path):
    archive=tmp_path/'object_selection/output.json';archive.parent.mkdir()
    archive.write_text(json.dumps(dict(status='not_submitted',stop_reason='object_selection_budget_exhausted')))
    result=bounded_outcome('attack_editing',dict(status='not_submitted',search_stage='object_selection',
        reason='No initial placement passed the observed-prefix checks'),tmp_path)
    assert result['status']=='unable_to_complete'
    assert result['evidence_paths']==[str(archive)]
    assert 'frozen_scene_path' not in result
    assert 'reuse_completed_from' not in result['detail']
    assert result['detail']['recovery_action'] == 'restart_object_selection'


def test_nested_constructor_failures_are_reported(tmp_path):
    archive=tmp_path/'trajectory_editing/output.json';archive.parent.mkdir()
    archive.write_text(json.dumps(dict(status='not_submitted',stop_reason='action_budget_exhausted')))
    call=tmp_path/'trajectory_editing/tool_calls/turn_01/result.json';call.parent.mkdir(parents=True)
    call.write_text(json.dumps(dict(candidates=[dict(checks=dict(road_surface=False,motion=True),
         task_failed_checks=['native_replay.keyframe_2'])])))
    result=bounded_outcome('attack_editing',dict(status='not_submitted'),tmp_path)
    assert result['failed_requirements']==['native_replay.keyframe_2','road_surface']


def test_incomplete_attack_checkpoint_restarts_selection(tmp_path, monkeypatch):
    core = Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'
    monkeypatch.syspath_prepend(str(core))
    import editing_backend

    prior = tmp_path/'prior'
    selection = prior/'output/attack/object_selection'
    selection.mkdir(parents=True)
    (selection.parent/'input.json').write_text('{}')
    (selection/'output.json').write_text(json.dumps(dict(status='not_submitted')))
    output = tmp_path/'retry/output'
    output.mkdir(parents=True)

    class ReachedFreshSelection(Exception):
        pass

    def run_attack(*args, **kwargs):
        assert kwargs['continue_from'] is None
        raise ReachedFreshSelection

    monkeypatch.setattr(editing_backend, 'run_attack', run_attack)
    args = dict(operation='auto', reuse_completed_from=str(prior), vadv2_self_test=False,
        scene_ref=dict(sample_root=str(tmp_path/'scene')), instruction='Fixture retry', timeout=120)
    with pytest.raises(ReachedFreshSelection):
        editing_backend.attack_output(args, output)
    assert json.loads((output/'attack-reuse.json').read_text())['status'] == 'recompute'


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


@pytest.mark.parametrize('balance',[
    {'search_api_seconds_remaining':0,'api_seconds_remaining':90},
    {'search_requests_remaining':0,'requests_remaining':2},
    {'api_seconds_remaining':0},
    {'requests_remaining':0},
])
def test_timeout_cannot_authorize_retry_with_exhausted_search_and_no_candidate(tmp_path,balance):
    archive=tmp_path/'trajectory_editing/output.json';archive.parent.mkdir()
    archive.write_text(json.dumps(dict(status='not_submitted',stop_reason='request_timeout_exhausted',
        api_budget=balance,resumable_candidate_ids=[])))
    result=bounded_outcome('attack_editing',dict(status='not_submitted'),tmp_path)
    assert result['detail']['retry_allowed'] is False
    assert result['detail']['recovery_action']=='stop_and_report'
    assert result['detail']['api_budget']==balance


def test_partial_model_decision_is_not_a_transient_network_retry():
    detail=dict(resumable_candidate_ids=[],termination_evidence={'timeout_category':'model_decision_timeout'},
        api_budget={'search_api_seconds_remaining':200,'api_seconds_remaining':290})
    result=attack_retry_policy('request_timeout_exhausted',detail)
    assert result['retry_allowed'] is False
    detail['termination_evidence']['timeout_category']='no_output_timeout'
    assert attack_retry_policy('request_timeout_exhausted',detail)['retry_allowed'] is True


def test_saved_candidate_review_is_allowed_once_with_only_reserve_remaining():
    detail=dict(resumable_candidate_ids=['candidate_001'],termination_evidence={'timeout_category':'model_decision_timeout'},
        api_budget={'search_api_seconds_remaining':0,'api_seconds_remaining':90,'requests_remaining':2})
    result=attack_retry_policy('stage_timeout_exhausted',detail)
    assert result['retry_allowed'] is True and result['recovery_action']=='review_saved_candidates'
    detail['finalization_grace_used']=True
    assert attack_retry_policy('request_timeout_exhausted',detail)['retry_allowed'] is False


def test_timeout_evidence_reaches_main_agent_compact_view(tmp_path):
    from agent.context import summarize
    evidence=dict(timeout_category='model_decision_timeout',failure_kind='stream_wall_time_exceeded',output_received=True)
    archive=tmp_path/'trajectory_editing/output.json';archive.parent.mkdir()
    archive.write_text(json.dumps(dict(status='not_submitted',stop_reason='request_timeout_exhausted',
        termination_evidence=evidence,resumable_candidate_ids=[])))
    result=bounded_outcome('attack_editing',dict(status='not_submitted'),tmp_path)
    feedback=summarize('attack_vehicle_editor',result)['recovery_feedback']
    assert feedback['termination_evidence']==evidence
    assert feedback['retry_allowed'] is False and feedback['recovery_action']=='stop_and_report'


def test_finalization_timeout_reports_checked_candidate_separately_from_earlier_failures(tmp_path):
    from agent.context import summarize

    search = tmp_path/'trajectory_editing'
    search.mkdir()
    (search/'output.json').write_text(json.dumps(dict(status='not_submitted',
        stop_reason='stage_timeout_exhausted', reason='model deadline', finalization_grace_used=True,
        resumable_candidate_ids=['candidate_005','candidate_006'],
        ready_to_submit_candidate_ids=['candidate_006'])))
    earlier = search/'tool_calls'/'turn_01'
    earlier.mkdir(parents=True)
    (earlier/'result.json').write_text(json.dumps(dict(failed_checks=['official_road'])))
    outcome = bounded_outcome('attack_editing', dict(status='not_submitted'), tmp_path)
    compact = summarize('attack_vehicle_editor', outcome)

    assert compact['recovery_feedback']['ready_to_submit_candidate_ids']==['candidate_006']
    assert compact['recovery_feedback']['retry_allowed'] is False
    assert compact['failed_requirements']==['official_road']
    assert 'candidate_006' in compact['reason'] and 'none was submitted' in compact['reason']
