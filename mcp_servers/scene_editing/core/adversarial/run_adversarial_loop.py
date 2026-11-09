"""Bounded Agent attack generation with PDM challenge and understood ego defense.

Every attempt rebuilds model inputs from current artifacts. Only compact structured
review feedback crosses attempts; archived conversations, reasoning and images do not.
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import acceptance_policy as policy
from experiment_runtime import ROOT, atomic_json, no_credentials, utc, experiments_root, full_pipeline_root
from run_attack_workflow import run as run_attack
from run_ego_conflict_brief import run as run_defense_understanding
from run_ego_reference import ego_path_from_frames
from run_ego_visual_editor_agent import run as run_defense_edit


class LoopBudgetExceeded(RuntimeError):
    pass


def request_count(path):
    return len(list(Path(path).rglob('request.json'))) if Path(path).is_dir() else 0


def validate_config(config):
    if config.get('schema_version') != 'adversarial_loop.v1':
        raise ValueError('Expected adversarial_loop.v1')
    if not isinstance(config.get('instruction'), str) or not config['instruction'].strip():
        raise ValueError('A nonempty user instruction is required')
    if not Path(config.get('scene', '')).is_dir():
        raise ValueError('Scene directory does not exist')
    if config.get('required_operation') not in {None, 'add_vehicle', 'edit_existing'}:
        raise ValueError('Unknown required operation')
    budget = config.get('budget')
    required = {'max_attack_attempts', 'attack_max_turns', 'defense_max_turns',
                'defense_max_edit_candidates', 'defense_max_inspections',
                'defense_constructor_candidates', 'request_timeout_seconds',
                'max_tokens', 'max_api_requests', 'total_seconds'}
    if not isinstance(budget, dict) or set(budget) != required:
        raise ValueError('Budget must contain exactly the explicit loop limits')
    for key, value in budget.items():
        if type(value) not in (int, float) or value <= 0:
            raise ValueError('Budgets must be positive numbers: ' + key)
    for key in ('max_attack_attempts', 'attack_max_turns', 'defense_max_turns',
                'defense_max_edit_candidates', 'defense_max_inspections',
                'defense_constructor_candidates', 'max_api_requests'):
        if type(budget[key]) is not int:
            raise ValueError('Count budgets must be integers: ' + key)
    if config['constraints'].get('attack_difficulty', {}).get('mode', 'free') != 'free':
        raise ValueError('The Agent loop requires free attack generation, not a numerical difficulty branch')
    pdm = config.get('pdm', {})
    for key in ('python', 'worldengine', 'system_lib', 'map_root'):
        if not Path(pdm.get(key, '')).exists():
            raise ValueError('Missing PDM path: ' + key)
    render = config.get('render', {})
    if render.get('enabled'):
        for key in ('python', 'worldengine_dir', 'system_lib', 'include_root'):
            if not Path(render.get(key, '')).exists():
                raise ValueError('Missing renderer path: ' + key)
        if type(render.get('timeout_seconds', 1800)) not in (int, float) or render.get('timeout_seconds', 1800) <= 0:
            raise ValueError('Renderer timeout must be positive')
    attack_source = Path(config['resume_attack_from']).resolve() if config.get('resume_attack_from') else None
    if attack_source and (not attack_source.is_dir() or (experiments_root()).resolve() not in attack_source.parents):
        raise ValueError('Attack recovery source must be an existing experiments run')
    no_credentials(config)


def compact_attack(result):
    metrics = result.get('metrics') or {}
    return dict(status=result.get('status'), selected_candidate_id=result.get('selected_candidate_id'),
        stop_reason=result.get('stop_reason'), failed_checks=metrics.get('failed_checks'),
        first_contact_time_s=metrics.get('first_contact_time_s'),
        min_ego_clearance_m=metrics.get('min_ego_clearance_m'))


def compact_pdm(result):
    scenarios = result['results']
    return {name: dict(contact_samples=value['contact_samples'],
        min_gap_m=value['min_gap_m'], min_target_gap_m=value['min_target_gap_m'],
        road_violation_samples=value['road_violation_samples'],
        first_contact=value['first_contact'], planning_calls=value['planning_calls'])
        for name, value in scenarios.items()}


def compact_defense(result, brief):
    return dict(status=result.get('status'), stop_reason=result.get('stop_reason'),
        selected_candidate_id=result.get('selected_candidate_id'),
        edit_candidate_count=result.get('edit_candidate_count'),
        invalid_tool_calls=result.get('invalid_tool_calls'),
        best_min_target_clearance_m=result.get('best_min_target_clearance_m'),
        understanding_directions=[item['type'] for item in brief.get('directions', [])])


def run_command(command, log, timeout, env=None):
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open('x') as stream:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
            stderr=subprocess.STDOUT, timeout=timeout)
    if completed.returncode:
        raise RuntimeError('Stage subprocess failed; see ' + str(log))


def run_pdm(config, attack_scene, output, history_s, target_track, target_is_new):
    settings = config['pdm']
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(Path(settings['worldengine']).resolve()))
    env['LD_LIBRARY_PATH'] = str(Path(settings['system_lib']).resolve()) + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    command = [settings['python'], str(Path(__file__).resolve().parent/'run_pdm_closed_loop.py'),
        '--baseline', config['scene'], '--attack', str(attack_scene), '--output', str(output),
        '--map-root', settings['map_root'], '--history-s', str(history_s), '--target-track', target_track]
    if target_is_new:
        command.append('--target-is-new')
    run_command(command, Path(str(output) + '.log'), settings.get('timeout_seconds', 300), env)
    result = json.loads((output/'summary.json').read_text())
    if result.get('status') != 'completed':
        raise RuntimeError('PDM did not complete')
    observations = json.loads((output/'attack/observations.json').read_text())
    if not observations or not observations[0].get('target_in_pdm'):
        raise RuntimeError('Attack target did not enter the initial PDM observation')
    if any(row['observation_timestamp_us'] > row['max_history_time_us'] for row in observations):
        raise RuntimeError('PDM received a future observation')
    return result


def run_constructor(scene, target_track, output, effective, count, timeout):
    policy_path = output.parent/'effective-policy.json'
    atomic_json(policy_path, effective)
    run_command([sys.executable, str(Path(__file__).resolve().parent/'run_ego_reference.py'), '--input', str(scene),
        '--run', str(output), '--target-track', target_track, '--acceptance-policy', str(policy_path),
        '--max-candidates', str(count)], output.parent/'constructor.log', timeout)


def run_independent(scene, editor_run, candidate, target_track, output, timeout):
    run_command([sys.executable, str(Path(__file__).resolve().parent/'run_ego_independent_check.py'),
        '--input', str(scene), '--run', str(editor_run), '--candidate', candidate,
        '--target-track', target_track, '--output', str(output)], Path(str(output)+'.log'), timeout)
    result = json.loads(Path(output).read_text())
    if result.get('status') != 'passed':
        raise RuntimeError('Submitted defense failed independent recomputation')
    return result


def pipeline_video_dir(run_name, case_id=None):
    case_id = case_id or run_name.split('_', 1)[0]
    if Path(run_name).name != run_name or Path(case_id).name != case_id or not run_name or not case_id:
        raise ValueError('Pipeline video run and case must be simple directory names')
    return full_pipeline_root()/run_name/f'{case_id}_final'/'defense_manual'


def run_render(config, attack_result, attack_dir, defense_dir, candidate, output):
    settings = config.get('render', {})
    if not settings.get('enabled'):
        return {'status': 'not_run'}
    target = attack_result['target']
    command = [settings['python'], str(Path(__file__).resolve().parent/'render_ego_mtgs_three_view.py'),
        '--input', attack_result['frozen_scene_path'], '--parent-run', str(defense_dir),
        '--candidate', candidate, '--output', str(output), '--worldengine', settings['worldengine_dir'],
        '--target-track', target['track_id']]
    if target['operation'] == 'add_vehicle':
        command.extend(['--added-vehicle-run', str(attack_dir/'materialized')])
    else:
        command.extend(['--attack-run', str(attack_dir/'trajectory_editing'),
            '--attack-candidate', attack_result['selected_candidate_id'], '--attack-track', target['track_id'],
            '--original-scene', config['scene']])
    if settings.get('gsplat_extension'):
        command = [settings['python'], str(Path(__file__).resolve().parent/'render_runtime.py'),
            '--extension', settings['gsplat_extension'], *command[1:]]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
        TORCH_EXTENSIONS_DIR=str(Path(str(output)+'-extension-cache')))
    env['PATH'] = str(Path(settings['python']).parent)+os.pathsep+env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = settings['system_lib']+os.pathsep+env.get('LD_LIBRARY_PATH', '')
    if settings.get('include_root'):
        env['CPATH'] = settings['include_root']+'/python3.10:'+settings['include_root']+os.pathsep+env.get('CPATH', '')
    build_headers = Path(settings['python']).parent.parent/'build-headers'
    if build_headers.is_dir():
        env['CPATH'] = str(build_headers)+os.pathsep+env.get('CPATH', '')
    env.setdefault('CUDA_HOME', '/usr/local/cuda-12.1')
    env.setdefault('MAX_JOBS', '4')
    run_command(command, Path(str(output)+'.log'), settings.get('timeout_seconds', 1800), env)
    report = json.loads((output/'render-report.json').read_text())
    if report.get('status') != 'passed':
        raise RuntimeError('Rendering did not pass artifact checks')
    run_command([settings['python'], str(Path(__file__).resolve().parent/'compose_mtgs_three_view_grid.py'),
        '--input', attack_result['frozen_scene_path'], '--render-run', str(output), '--output', str(output),
        '--target-track', target['track_id']], output/'compose.log', settings.get('timeout_seconds', 1800), env)
    validation = json.loads((output/'manual-explained-validation.json').read_text())
    if validation.get('status') != 'passed':
        raise RuntimeError('Pipeline video did not pass artifact checks')
    report['manual_explained'] = 'manual-explained.mp4'
    report['pipeline_output_dir'] = str(output)
    return report


def run_loop(config, output):
    validate_config(config)
    output = Path(output).resolve()
    if (experiments_root()).resolve() not in output.parents or output.exists():
        raise ValueError('Loop output must be a new experiments directory')
    output.mkdir(parents=True)
    atomic_json(output/'input.json', config)
    budget = config['budget']
    started = time.monotonic()
    deadline = started + budget['total_seconds']
    state = dict(schema_version='adversarial_loop.v1', status='running', started_at=utc(),
        active_stage='attack', current_attempt=0, api_requests=0, attempts=[],
        selected_attempt=None, context_policy='fresh per-stage state; compact review feedback only; no archived dialogue or reasoning reuse')
    atomic_json(output/'state.json', state)

    def save():
        state['elapsed_s'] = time.monotonic()-started
        atomic_json(output/'state.json', state)
        atomic_json(output/'output.json', state)

    def check_budget():
        if time.monotonic() >= deadline:
            raise LoopBudgetExceeded('total_seconds')
        if state['api_requests'] >= budget['max_api_requests']:
            raise LoopBudgetExceeded('max_api_requests')

    def account(path):
        state['api_requests'] += request_count(path)
        save()
        if state['api_requests'] > budget['max_api_requests']:
            raise LoopBudgetExceeded('max_api_requests')

    feedback = None
    try:
        for attempt_index in range(1, budget['max_attack_attempts']+1):
            check_budget()
            attempt = output/f'attempt_{attempt_index:03d}'
            attempt.mkdir()
            record = dict(attempt=attempt_index, status='running', stages={})
            state.update(current_attempt=attempt_index, active_stage='attack')
            state['attempts'].append(record)
            save()
            constraints = copy.deepcopy(config['constraints'])
            if feedback is not None:
                constraints['revision_feedback'] = feedback
            attack_dir = attempt/'attack'
            try:
                attack_result = run_attack(config['scene'], attack_dir, config['instruction'],
                    config.get('assets', []), constraints, timeout_seconds=budget['request_timeout_seconds'],
                    max_tokens=budget['max_tokens'], max_turns=budget['attack_max_turns'],
                    required_operation=config.get('required_operation'), required_track=config.get('required_track'),
                    continue_from=config.get('resume_attack_from') if attempt_index == 1 else None)
            finally:
                account(attack_dir)
            while attack_result.get('status') == 'continued_in_object_revision':
                attack_dir = Path(attack_result['followup_run'])
                attack_result = json.loads((attack_dir/'output.json').read_text())
            record['attack'] = compact_attack(attack_result)
            record['attack_dir'] = str(attack_dir)
            if attack_result.get('status') != 'reference_candidate_submitted':
                from business_outcome import bounded_outcome
                refusal = bounded_outcome('attack_editing',attack_result,attack_dir)
                record.update(status='attack_refused',refusal=refusal)
                atomic_json(attempt/'output.json',record)
                state.update(status='unable_to_complete',active_stage='finished',
                             stop_reason='attack_search_declined_submission',refusal=refusal)
                save()
                return state
            attack_scene = Path(attack_result['frozen_scene_path'])
            target_track = attack_result['target']['track_id']
            effective = json.loads((attack_dir/'input.json').read_text())['constraints']['acceptance_policy']
            effective = policy.resolve(effective)
            state['active_stage'] = 'pdm'
            save()
            check_budget()
            pdm_result = run_pdm(config, attack_scene, attempt/'pdm', effective['observation_end_s'],
                target_track, attack_result['target']['operation'] == 'add_vehicle')
            record['pdm'] = compact_pdm(pdm_result)
            attack_pdm = pdm_result['results']['attack']
            pdm_safe = not attack_pdm['contact_samples'] and not attack_pdm['road_violation_samples']
            atomic_json(attempt/'output.json', record)
            if pdm_safe:
                record['status'] = 'pdm_defense_succeeded'
                feedback = dict(stage='pdm', outcome='pdm_defense_succeeded',
                    guidance='The previous attack was valid but PDM avoided it safely. Revise the attack to be more challenging while preserving the original user instruction and all hard checks.',
                    evidence=record['pdm']['attack'])
                atomic_json(attempt/'output.json', record)
                continue
            state['active_stage'] = 'defense_understanding'
            save()
            check_budget()
            understanding_dir = attempt/'defense_understanding'
            frames = json.loads((attack_scene/'metadata/frames.json').read_text())
            times, _, _ = ego_path_from_frames(frames)
            visual_frame = min(range(len(times)), key=lambda i: abs(times[i]-effective['observation_end_s']))
            try:
                run_defense_understanding(attack_scene, attack_result['attack_run'] if 'attack_run' in attack_result else attack_dir/'trajectory_editing',
                    attack_result['selected_candidate_id'], target_track, understanding_dir,
                    timeout_seconds=budget['request_timeout_seconds'], max_tokens=budget['max_tokens'],
                    acceptance=effective, image_scene=config.get('image_scene',config['scene']), visual_frame=visual_frame,
                    cameras=config.get('cameras'))
            finally:
                account(understanding_dir)
            understanding = json.loads((understanding_dir/'summary.json').read_text())
            if understanding.get('status') != 'completed':
                raise RuntimeError('Defense understanding did not produce a valid brief')
            brief = json.loads((understanding_dir/'conflict-brief.json').read_text())
            record['defense_understanding'] = dict(status='completed', directions=len(brief['directions']))
            state['active_stage'] = 'defense_constructor'
            save()
            check_budget()
            constructor_dir = attempt/'defense_constructor'
            run_constructor(attack_scene, target_track, constructor_dir, effective,
                budget['defense_constructor_candidates'], 300)
            foci_dir=None
            if config.get('use_foci',True):
                from defense_generation import prepare_initial_foci
                state['active_stage']='defense_foci'
                save()
                check_budget()
                foci_dir=prepare_initial_foci(attack_scene,constructor_dir,target_track,attempt/'defense_foci',
                    effective['observation_end_s'],float(times[-1]))
            state['active_stage'] = 'defense_edit'
            save()
            check_budget()
            defense_dir = attempt/'defense_agent'
            try:
                defense_result = run_defense_edit(attack_scene, constructor_dir, foci_dir, defense_dir,
                    timeout_seconds=budget['request_timeout_seconds'], max_turns=budget['defense_max_turns'],
                    max_edit_candidates=budget['defense_max_edit_candidates'],
                    max_inspections=budget['defense_max_inspections'], max_tokens=budget['max_tokens'],
                    target_track=target_track, conflict_brief=understanding_dir,controlled_generation=True,
                    target_effect=config.get('target_effect'),foci_enabled=config.get('use_foci',True))
            finally:
                account(defense_dir)
            record['defense'] = compact_defense(defense_result, brief)
            atomic_json(attempt/'output.json', record)
            if defense_result.get('status') != 'reference_candidate_submitted':
                record['status'] = 'defense_not_found'
                feedback = dict(stage='defense', outcome='defense_not_found_within_budget',
                    guidance='PDM failed but the defense Agent did not find an accepted defense within its budget. Generate a less demanding alternative while preserving the original user instruction. This is not a global infeasibility proof.',
                    evidence=record['defense'])
                atomic_json(attempt/'output.json', record)
                continue
            state['active_stage'] = 'defense_independent_check'
            save()
            check_budget()
            candidate = defense_result['selected_candidate_id']
            independent = run_independent(attack_scene, defense_dir, candidate, target_track,
                attempt/'defense-independent.json', 300)
            record['defense_independent'] = dict(status=independent['status'], candidate_id=candidate)
            state['active_stage'] = 'render'
            save()
            check_budget()
            render = run_render(config, attack_result, attack_dir, defense_dir, candidate,
                pipeline_video_dir(output.name, config.get('case_id')))
            record['render'] = dict(status=render['status'], output_dir=render['pipeline_output_dir'],
                manual_explained=render['manual_explained'])
            record['status'] = 'accepted'
            state.update(status='accepted', active_stage='finished', selected_attempt=attempt_index,
                selected_attack_scene=str(attack_scene), selected_defense_candidate=candidate)
            atomic_json(attempt/'output.json', record)
            save()
            return state
        state.update(status='no_accepted_candidate', active_stage='finished', stop_reason='attack_attempt_budget_exhausted')
        save()
        return state
    except LoopBudgetExceeded as error:
        failed_stage = state['active_stage']
        if state['attempts'] and state['attempts'][-1]['status'] == 'running':
            current = state['attempts'][-1]
            current.update(status='budget_exhausted', error_type=type(error).__name__)
            atomic_json(output/f"attempt_{current['attempt']:03d}"/'output.json', current)
        state.update(status='budget_exhausted', active_stage='finished', failed_stage=failed_stage, stop_reason=str(error))
        save()
        return state
    except KeyboardInterrupt:
        failed_stage = state['active_stage']
        if state['attempts'] and state['attempts'][-1]['status'] == 'running':
            current = state['attempts'][-1]
            current['status'] = 'interrupted'
            atomic_json(output/f"attempt_{current['attempt']:03d}"/'output.json', current)
        state.update(status='interrupted', active_stage='finished', failed_stage=failed_stage)
        save()
        raise
    except Exception as error:
        failed_stage = state['active_stage']
        if state['attempts'] and state['attempts'][-1]['status'] == 'running':
            current = state['attempts'][-1]
            current.update(status='failed', error_type=type(error).__name__)
            atomic_json(output/f"attempt_{current['attempt']:03d}"/'output.json', current)
        state.update(status='failed', active_stage='finished', failed_stage=failed_stage,
            error_type=type(error).__name__, stop_reason='stage_execution_failed')
        save()
        return state


def resume_defense(config, source, output):
    validate_config(config)
    source, output = Path(source).resolve(), Path(output).resolve()
    if (experiments_root()).resolve() not in output.parents or output.exists() or source == output:
        raise ValueError('Recovery output must be a new experiments directory')
    source_state = json.loads((source/'output.json').read_text())
    source_input = json.loads((source/'input.json').read_text())
    for key in ('scene', 'instruction', 'assets', 'constraints', 'required_operation'):
        if source_input.get(key) != config.get(key):
            raise ValueError('Recovery input differs from the original run: ' + key)
    attempt_index = source_state['current_attempt']
    attempt = source/f'attempt_{attempt_index:03d}'
    defense_dir = attempt/'defense_agent'
    previous = json.loads((defense_dir/'summary.json').read_text())
    if previous.get('stop_reason') != 'request_or_execution_failed' or previous.get('selected_candidate_id') is not None:
        raise ValueError('Only a request-failed, unsubmitted defense edit can be resumed')
    attack_dir = attempt/'attack'
    attack_result = json.loads((attack_dir/'output.json').read_text())
    constructor_dir = attempt/'defense_constructor'
    understanding_dir = attempt/'defense_understanding'
    output.mkdir(parents=True)
    budget = config['budget']
    result = dict(schema_version='adversarial_loop_recovery.v1', status='running', source_run=str(source),
        source_attempt=attempt_index, request_timeout_seconds=budget['request_timeout_seconds'],
        new_api_requests=0, selected_defense_candidate=None)
    atomic_json(output/'input.json', config)
    atomic_json(output/'output.json', result)
    before = request_count(defense_dir)
    editor_config=json.loads((defense_dir/'config.json').read_text())
    prior_foci=editor_config.get('foci_run')
    try:
        defense = run_defense_edit(attack_result['frozen_scene_path'], constructor_dir, prior_foci, defense_dir,
            timeout_seconds=budget['request_timeout_seconds'], max_turns=budget['defense_max_turns'],
            max_edit_candidates=budget['defense_max_edit_candidates'],
            max_inspections=budget['defense_max_inspections'], max_tokens=budget['max_tokens'],
            target_track=attack_result['target']['track_id'], conflict_brief=understanding_dir,
            resume=True, allow_timeout_change=True,
            controlled_generation=editor_config.get('controlled_generation',False),
            target_effect=editor_config.get('target_effect'),foci_enabled=editor_config.get('foci_enabled',True))
        result['new_api_requests'] = request_count(defense_dir)-before
        result['defense'] = compact_defense(defense, json.loads((understanding_dir/'conflict-brief.json').read_text()))
        if defense.get('status') != 'reference_candidate_submitted':
            result.update(status='no_accepted_candidate', stop_reason=defense.get('stop_reason'))
            return result
        candidate = defense['selected_candidate_id']
        run_independent(attack_result['frozen_scene_path'], defense_dir, candidate,
            attack_result['target']['track_id'], output/'defense-independent.json', 300)
        render = run_render(config, attack_result, attack_dir, defense_dir, candidate,
            pipeline_video_dir(source.name, config.get('case_id')))
        result.update(status='accepted', selected_defense_candidate=candidate, render=render['status'],
            render_output_dir=render['pipeline_output_dir'], manual_explained=render['manual_explained'])
        return result
    except Exception as error:
        result.update(status='failed', error_type=type(error).__name__)
        return result
    finally:
        result['new_api_requests'] = request_count(defense_dir)-before
        atomic_json(output/'output.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--request-timeout-seconds', type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.request_timeout_seconds is not None:
        config['budget']['request_timeout_seconds'] = args.request_timeout_seconds
    result = (resume_defense(config, args.resume_from, args.output) if args.resume_from
              else run_loop(config, args.output))
    print(json.dumps(dict(status=result['status'], selected_attempt=result.get('selected_attempt'),
        attempts=len(result.get('attempts', [])),
        api_requests=result.get('api_requests', result.get('new_api_requests'))), ensure_ascii=False))


if __name__ == '__main__':
    main()
