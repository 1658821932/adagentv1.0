"""Build bounded business outcomes from completed search archives only."""
import json
from pathlib import Path


def bounded_outcome(stage, detail, search_dir):
    directory = Path(search_dir)
    archive = directory / ((detail.get('search_stage','trajectory_editing')+'/output.json') if stage == 'attack_editing' else 'summary.json')
    stored = json.loads(archive.read_text())
    stop = detail.get('stop_reason') or stored.get('stop_reason')
    if detail.get('status') != 'not_submitted' or not stop or stop == 'request_or_execution_failed':
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
    if not attempts:
        attempts = [dict(evidence_path=str(archive), stop_reason=stop)]
    reason = detail.get('reason') or stored.get('reason') or (reasons[-1] if reasons else 'No accepted candidate was submitted within the current tools and search budget.')
    return dict(schema_version='editing_stage.v1',stage=stage,
        status='unable_to_complete' if stage == 'attack_editing' else 'defense_not_found',
        mode='real',run_dir=str(directory.parent),reason=reason,stop_reason=stop,
        failed_requirements=sorted(failed),failed_requirements_scope='observed_in_attempted_candidates',
        attempts=attempts,evidence_paths=([str(root/'output.json')] if root!=directory else [])+[str(archive)],
        scope='current_tools_and_budget',global_infeasibility_proven=False,detail=detail)
