"""Bounded acceptance through the registered AdAgent tools, including PDM reattack.

The host Agent already owns this transition in agent/prompt.md. This executable
provides the same transition for reproducible tool acceptance without asking an
extra model to schedule deterministic validation steps.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone

from mcp_servers.runtime import write_json
from mcp_servers.scene_editing.adapter.schemas import AttackInput
from tools.registry import call_tool


def utc():
    return datetime.now(timezone.utc).isoformat()


def body(result, tool):
    if result.get('status') != 'success' or result.get('tool') != tool:
        raise ValueError(f'{tool} did not return a successful tool envelope')
    return result['result']


def pdm_feedback(attack, pdm):
    """Pass measured outcome and former placement, never old model dialogue."""
    accepted = body(attack, 'attack_vehicle_editor')
    challenge = body(pdm, 'policy_runner')['pdm_challenge']
    if challenge.get('pdm_defense_succeeded') is not True:
        raise ValueError('Only a safely defended attack supplies PDM revision feedback')
    if any(challenge[name][key] for name in ('attack',)
           for key in ('contact_samples', 'road_violation_samples')):
        raise ValueError('PDM outcome contradicts its contact/road evidence')
    evidence = dict(previous_candidate_id=accepted['selected_candidate_id'],
        target_track=accepted['target_track'], pdm=deepcopy(challenge))
    # Read the selected attack's numeric identity/state, not archived text/images.
    selected = Path(accepted['submission_path']).parent.parent / 'object_selection/output.json'
    if selected.is_file():
        target = json.loads(selected.read_text())['target']
        evidence['previous_target'] = {key: target[key] for key in
            ('operation', 'track_id', 'size_m', 'initial_state', 'initial_speed_mps') if key in target}
    return dict(stage='pdm', outcome='pdm_defense_succeeded',
        guidance='The previous valid attack was defended safely by PDM. Search a more challenging '
        'alternative on the ORIGINAL scene while preserving the user instruction and every hard check. '
        'Use the measured prior placement/outcome; do not repeat it unchanged. '
        'Do not invent a narrower event window or new clearance requirement.', evidence=evidence)


def defense_revision_feedback(response, attack):
    """A completed bounded search may guide revision; execution failures never do."""
    if response.get('status') != 'defense_not_found':
        return None
    detail = response.get('detail') or {}
    stop_reason=response.get('stop_reason') or detail.get('stop_reason')
    if (detail.get('status') != 'not_submitted' or not stop_reason
            or stop_reason == 'request_or_execution_failed'):
        return None
    accepted = body(attack, 'attack_vehicle_editor')
    evidence = {key: detail.get(key) for key in ('stop_reason', 'model_actions',
        'edit_candidate_count', 'invalid_tool_calls', 'best_min_target_clearance_m')}
    evidence['stop_reason']=stop_reason
    evidence.update({key:deepcopy(response[key]) for key in
        ('reason','failed_requirements','evidence_paths','scope') if key in response})
    evidence.update(previous_attack_candidate_id=accepted['selected_candidate_id'],
        previous_attack_scope=str(Path(accepted['submission_path']).parent),target_track=accepted['target_track'],
        global_feasibility_proven=False)
    return dict(stage='defense', outcome='bounded_search_no_submitted_defense',
        guidance='The completed defense search did not submit an accepted trajectory within its budget. '
        'Search a less demanding alternative while retaining the user instruction and hard checks. '
        'Use defense witnesses and braking evidence; do not invent a new collision window or clearance threshold. '
        'This result is not a global infeasibility proof.', evidence=evidence)


def load_seed(directory, attack_input):
    directory = Path(directory)
    previous = AttackInput.model_validate(json.loads((directory/'attack-input.json').read_text())).model_dump()
    current = AttackInput.model_validate(attack_input).model_dump()
    if previous != current:
        raise ValueError('Seed attack input differs from the current task or acceptance policy')
    attack = json.loads((directory/'attack-result.json').read_text())
    pdm = json.loads((directory/'pdm-result.json').read_text())
    accepted = body(attack, 'attack_vehicle_editor')
    if accepted['source_scene_ref'] != current['scene_ref']:
        raise ValueError('Seed attack belongs to a different source scene')
    request = json.loads((directory/'pdm-input.json').read_text())
    expected = dict(challenge_baseline_scene_path=current['scene_ref']['sample_root'],
        challenge_attack_scene_path=accepted['frozen_scene_path'], challenge_target_track=accepted['target_track'],
        challenge_target_is_new=current['operation']=='add_vehicle',
        challenge_history_s=current['constraints']['no_ego_contact_until_s'])
    if bool(request.get('challenge_ignore_traffic_lights',False)) != current['pdm_ignore_traffic_lights']:
        raise ValueError('Seed PDM traffic-light mode differs from the current task')
    if any(request.get(key) != value for key, value in expected.items()):
        raise ValueError('Seed PDM request is bound to a different attack or timeline')
    pdm_feedback(attack, pdm)
    requests = sum(1 for path in (directory/'services').rglob('request.json')
                   if path.parent.name.startswith('attempt_'))
    return dict(directory=str(directory.resolve()), attack=attack, pdm=pdm, api_requests=requests)


def request_count(directory):
    return sum(1 for path in Path(directory).rglob('request.json') if path.parent.name.startswith('attempt_'))


def run(config, output, *, seed=None, tool_runner=call_tool):
    output = Path(output).resolve()
    base = AttackInput.model_validate(config['attack_input']).model_dump()
    budget = config['budget']
    for key in ('max_attack_attempts', 'max_api_requests', 'total_seconds'):
        if type(budget[key]) is not int or budget[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'input.json', config)
    state = dict(status='running', started_at=utc(), active_stage=None, attempts=[],
        api_requests=0, selected_attempt=None, budget=budget,
        budget_scope='Attempt limits are enforced before each attack; API/wall budgets checked between stages. Internal tool action/request limits remain active.')
    started = time.monotonic()
    feedback = None
    previous_runtime = os.environ.get('ADAGENT_RUNTIME_ROOT')
    os.environ['ADAGENT_RUNTIME_ROOT'] = str(output/'services')

    def save():
        state['elapsed_s'] = round(time.monotonic()-started,3)
        write_json(output/'state.json', state)

    def stop(status, reason):
        state.update(status=status, stop_reason=reason, active_stage=None, finished_at=utc())
        save()
        return state

    def check_budget():
        return (time.monotonic()-started < budget['total_seconds']
                and state['api_requests'] < budget['max_api_requests'])

    def stage(record, name, tool, arguments):
        state['active_stage'] = name
        entry = dict(stage=name, tool=tool, status='running', started_at=utc())
        record['stages'].append(entry)
        directory = output/f"attempt_{record['attempt']:03d}"
        write_json(directory/f'{name}-input.json', arguments)
        save()
        print(json.dumps(dict(attempt=record['attempt'],stage=name,status='started',time=utc())),flush=True)
        began = time.monotonic()
        before = request_count(output/'services')
        try:
            result = tool_runner(tool, arguments, mode='real')
        except Exception as error:
            result = dict(status='error',tool=tool,error=dict(type=type(error).__name__,message=str(error)[:500]))
        state['api_requests'] += request_count(output/'services')-before
        entry.update(status=result.get('status'),elapsed_s=round(time.monotonic()-began,3),finished_at=utc())
        record[name] = result
        write_json(directory/f'{name}-result.json',result)
        save()
        print(json.dumps(dict(attempt=record['attempt'],stage=name,status=result.get('status'),elapsed_s=entry['elapsed_s'])),flush=True)
        return result

    try:
        if seed:
            feedback = pdm_feedback(seed['attack'],seed['pdm'])
            state['attempts'].append(dict(attempt=1,status='pdm_defense_succeeded',reused=True,
                source=seed['directory'],stages=[],attack=seed['attack'],pdm=seed['pdm']))
            state['api_requests'] = seed['api_requests']
            write_json(output/'seed.json',seed)
            write_json(output/'initial-revision-feedback.json',feedback)
            save()
        while len(state['attempts']) < budget['max_attack_attempts']:
            if not check_budget():
                return stop('budget_exhausted','API request or wall-clock budget exhausted between stages')
            record = dict(attempt=len(state['attempts'])+1,status='running',stages=[])
            state['attempts'].append(record)
            arguments = deepcopy(base)
            if feedback is not None:
                arguments['revision_feedback'] = deepcopy(feedback)
            attack = stage(record,'attack','attack_vehicle_editor',arguments)
            if attack.get('status') != 'success':
                record['status']='attack_execution_failed'
                return stop('paused','Attack execution failed; inspect API/protocol/dependency errors')
            if attack['result'].get('status') == 'unable_to_complete':
                record['status']='attack_refused'
                state['refusal']=attack['result']
                return stop('unable_to_complete','Attack search declined submission within the current tools and budget')
            accepted = attack['result']
            if not check_budget():
                return stop('budget_exhausted','Budget exhausted after attack')
            pdm_arguments=dict(execution_mode='pdm_closed_loop',
                challenge_baseline_scene_path=base['scene_ref']['sample_root'],
                challenge_attack_scene_path=accepted['frozen_scene_path'],
                challenge_target_track=accepted['target_track'], challenge_target_is_new=base['operation']=='add_vehicle',
                challenge_history_s=base['constraints']['no_ego_contact_until_s'],
                challenge_ignore_traffic_lights=base['pdm_ignore_traffic_lights'],timeout=300)
            if accepted.get('pdm_probe_result_path'):
                pdm_arguments['challenge_probe_result_path']=accepted['pdm_probe_result_path']
            pdm = stage(record,'pdm','policy_runner',pdm_arguments)
            if pdm.get('status') != 'success':
                record['status']='pdm_execution_failed'
                return stop('paused','PDM execution/control validation failed, not scene infeasibility')
            challenge = pdm['result']['pdm_challenge']
            if challenge['pdm_defense_succeeded']:
                feedback = pdm_feedback(attack,pdm)
                record['status']='pdm_defense_succeeded'
                write_json(output/f"attempt_{record['attempt']:03d}/revision-feedback.json",feedback)
                save()
                continue
            if not check_budget():
                return stop('budget_exhausted','Budget exhausted after PDM')
            ego = stage(record,'ego','ego_trajectory_editor',dict(attack_result=attack,
                target_effect=config['target_effect'],allow_safe_stop=True,use_foci=config.get('use_foci',True),timeout=3600))
            if ego.get('status') != 'success':
                record['status']='defense_or_render_execution_failed'
                return stop('paused','Defense or render execution failed; preserve accepted trajectories for recovery')
            if ego['result'].get('status') == 'defense_not_found':
                revision=defense_revision_feedback(ego['result'],attack)
                if revision is None:
                    return stop('paused','Malformed defense business outcome; refusing automatic reattack')
                feedback=revision
                record['status']='bounded_defense_not_found'
                write_json(output/f"attempt_{record['attempt']:03d}/revision-feedback.json",feedback)
                save()
                continue
            if not check_budget():
                return stop('budget_exhausted','Budget exhausted after defense/render')
            record['status']='accepted'
            state['selected_attempt']=record['attempt']
            return stop('accepted','Attack, PDM challenge, submitted defense, independent checks, render completed; standalone visual review is removed')
        return stop('no_accepted_candidate','attack_attempt_budget_exhausted')
    except Exception as error:
        state['error']=dict(type=type(error).__name__,message=str(error)[:500])
        return stop('paused','Loop integration failure; inspect checkpoint before retrying')
    except KeyboardInterrupt:
        stop('interrupted','User or process interrupted validation')
        raise
    finally:
        if previous_runtime is None:
            os.environ.pop('ADAGENT_RUNTIME_ROOT',None)
        else:
            os.environ['ADAGENT_RUNTIME_ROOT']=previous_runtime


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed-case-directory',type=Path)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    seed=load_seed(args.seed_case_directory,config['attack_input']) if args.seed_case_directory else None
    result=run(config,args.output,seed=seed)
    print(json.dumps(dict(status=result['status'],stop_reason=result['stop_reason'],output=str(args.output)),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
