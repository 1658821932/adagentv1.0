"""Build bounded business outcomes from completed search archives only."""
import json
from pathlib import Path


def attack_retry_policy(stop, detail):
    """A timeout is not permission to reset cumulative search or review budgets."""
    budget=detail.get('api_budget') or {}
    qualified=bool(detail.get('resumable_candidate_ids'))
    exhausted=(stop.startswith('cumulative_') or stop=='constructor_budget_exhausted_without_qualified_candidate'
        or any(budget.get(key,float('inf'))<=0 for key in ('requests_remaining','api_seconds_remaining'))
        or not qualified and any(budget.get(key,float('inf'))<=0 for key in
            ('search_requests_remaining','search_api_seconds_remaining')))
    evidence=detail.get('termination_evidence') or {}
    if exhausted:
        return dict(retry_allowed=False,recovery_action='stop_and_report',
            recovery_hint='No search budget and qualified candidate combination remains. Reserved finalization time is not new search budget; do not restart 05 or switch objects/scenes to reset it.')
    if detail.get('finalization_grace_used'):
        return dict(retry_allowed=False,recovery_action='stop_and_report',
            recovery_hint='The bounded saved-candidate finalization opportunity was already used; do not repeat it in a new service run.')
    if evidence.get('timeout_category')=='model_decision_timeout' and not qualified:
        return dict(retry_allowed=False,recovery_action='stop_and_report',
            recovery_hint='The model kept producing partial output until the deadline, without a completed decision or qualified candidate. This is not zero-output network failure; do not repeat the same request unchanged.')
    if qualified:
        return dict(retry_allowed=True,recovery_action='review_saved_candidates',
            recovery_hint='Reuse the existing checkpoint to inspect/test/submit qualified saved candidates; keep the scene, task, thinking configuration and cumulative budgets.')
    if stop == 'required_initial_placement_rejected':
        return dict(retry_allowed=False, recovery_action='stop_and_report',
            recovery_hint='The explicitly locked actor fails its fixed initial road check. Repeating the same locked selection cannot change that pose.')
    if stop in {'object_selection_repeated_rejection', 'object_selection_budget_exhausted'}:
        return dict(retry_allowed=True, recovery_action='restart_object_selection',
            recovery_hint='No object checkpoint was completed. Omit reuse_completed_from; retry only with an available alternative that addresses the measured rejection, within the same cumulative budgets.')
    return dict(retry_allowed=True,recovery_action='inspect_failure_before_retry',
        recovery_hint='Inspect the timeout category and archive. Only a transient transport/no-output failure permits one unchanged retry; preserve cumulative budgets.')


def bounded_outcome(stage, detail, search_dir):
    directory = Path(search_dir)
    archive = directory / ((detail.get('search_stage','trajectory_editing')+'/output.json') if stage == 'attack_editing' else 'summary.json')
    stored = json.loads(archive.read_text())
    stop = detail.get('stop_reason') or stored.get('stop_reason')
    execution_stops = {'request_or_execution_failed', 'native_tool_protocol_failed',
        'invalid_or_missing_tool_call', 'invalid_tool_call', 'model_stopped_without_submission',
        'invalid_call_budget_exhausted', 'repeated_invalid_tool_call', 'interrupted_by_operator'}
    if detail.get('status') != 'not_submitted' or not stop or stop in execution_stops:
        raise RuntimeError('Search did not complete normally; cannot report business refusal')
    root = directory
    while (root.parent/'output.json').is_file():
        parent = json.loads((root.parent/'output.json').read_text())
        followup = parent.get('followup_run')
        if not followup or Path(followup).resolve() != root.resolve():
            break
        root = root.parent
    calls = sorted(root.rglob('tool_calls/turn_*/result.json'),key=lambda path:path.stat().st_mtime_ns)
    attempts = []
    reasons = []
    failed = set()
    def failures(value):
        if isinstance(value,dict):
            for key,child in value.items():
                if key in ('failed_checks','task_failed_checks') and isinstance(child,list):
                    failed.update(str(item) for item in child)
                if key == 'checks' and isinstance(child,dict):
                    failed.update(name for name,state in child.items() if state is False or state == 'failed')
                failures(child)
        elif isinstance(value,list):
            for child in value:
                failures(child)
    for path in calls:
        value = json.loads(path.read_text())
        call_path=path.with_name('call.json')
        call=json.loads(call_path.read_text()) if call_path.is_file() else {}
        action=call.get('function',{}).get('name')
        attempts.append(dict(evidence_path=str(path),action=action,status=value.get('status'),
                             candidate_id=value.get('candidate_id')))
        if value.get('status') == 'finished' and value.get('reason'):
            reasons.append(str(value['reason']))
        failures(value)
    failures(stored.get('termination_evidence'))
    failures(stored.get('rejected_initial_placements'))
    if not attempts:
        attempts = [dict(evidence_path=str(archive), stop_reason=stop)]
    reason = detail.get('reason') or stored.get('reason') or (reasons[-1] if reasons else 'No accepted candidate was submitted within the current tools and search budget.')
    detail=dict(detail,**{key:stored[key] for key in
        ('api_budget','constructor_budget','resumable_candidate_ids','termination_evidence',
         'ready_to_submit_candidate_ids',
         'finalization_grace_used','request_timeout_recovery','editing_timeout_recovery_used') if key in stored})
    if stage=='attack_editing':
        ready = detail.get('ready_to_submit_candidate_ids') or []
        if ready:
            reason = ('Candidates passed all submission checks but none was submitted: '
                      + ', '.join(ready) + '. ' + reason)
        selection = directory/'object_selection/output.json'
        detail.pop('reuse_completed_from', None)
        if selection.is_file() and json.loads(selection.read_text()).get('status') == 'completed':
            detail['reuse_completed_from'] = str(root.parents[1])
        detail.update(attack_retry_policy(stop,detail))
    else:
        detail['reuse_completed_from'] = str(root.parents[1])
        detail['retry_allowed']=not (stop.startswith('cumulative_') or stop=='constructor_budget_exhausted_without_qualified_candidate')
    return dict(schema_version='editing_stage.v1',stage=stage,
        status='unable_to_complete' if stage == 'attack_editing' else 'defense_not_found',
        mode='real',run_dir=str(directory.parent),reason=reason,stop_reason=stop,
        failed_requirements=sorted(failed),failed_requirements_scope='observed_in_attempted_candidates',
        attempts=attempts,evidence_paths=([str(root/'output.json')] if root!=directory else [])+[str(archive)],
        scope='current_tools_and_budget',global_infeasibility_proven=False,detail=detail)
