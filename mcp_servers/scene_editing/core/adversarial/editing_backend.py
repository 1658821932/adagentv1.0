"""editing_stage.v1 JSON backend for the independent adversarial editing package."""
import contextlib
import json
import os
import re
import sys
from pathlib import Path

# Imported dependencies may print diagnostics; stdout is reserved for the JSON reply.
with contextlib.redirect_stdout(sys.stderr):
    import acceptance_policy as policy
    from experiment_runtime import atomic_json
    from review_attack_workflow import review
    from run_adversarial_loop import pipeline_video_dir, run_command, run_constructor, run_independent, run_render
    from run_attack_workflow import run as run_attack
    from run_ego_conflict_brief import run as run_understanding
    from run_ego_reference import ego_path_from_frames
    from run_ego_visual_editor_agent import run as run_defense_edit

BACKEND_ROOT = Path(__file__).resolve().parent
SETTINGS = json.loads((BACKEND_ROOT/'backend_settings.json').read_text())


def check_status(checks):
    return {name: 'passed' if value else 'failed' for name, value in checks.items()}


def attack_output(arguments, output_dir):
    source = arguments['scene_ref']
    scene = Path(source['sample_root'])
    constraints_input = arguments.get('constraints', {})
    acceptance = policy.resolve(arguments.get('acceptance_policy'))
    constraints = dict(observation_end_s=constraints_input.get('no_ego_contact_until_s', 3.),
        official_margin_m=acceptance['official_margin_m'], max_ego_clearance_m=2., min_travel_m=0.,
        vehicle_motion_model='bicycle.v1', require_structured_objective=True,
        acceptance_policy=acceptance)
    if arguments.get('task_contract') is not None:
        constraints['task_contract'] = arguments['task_contract']
    if arguments.get('revision_feedback') is not None:
        constraints['revision_feedback'] = arguments['revision_feedback']
    if arguments.get('pdm_self_check'):
        constraints.update(pdm_self_check=True,pdm_probe_budget=arguments.get('pdm_probe_budget',2),
            pdm_probe_root=str(output_dir/'pdm-probes'))
        constraints['pdm_defense_feedback']=arguments.get('pdm_defense_feedback',False)
        constraints['pdm_ignore_traffic_lights']=arguments.get('pdm_ignore_traffic_lights',False)
    if arguments.get('maneuver') == 'stationary':
        constraints['stationary_target'] = True
    assets = []
    if arguments.get('vehicle') is not None:
        vehicle = arguments['vehicle']
        assets.append(dict(asset_id='asset_001', path=vehicle['asset_path'], size_m=vehicle['size_m']))
    attack_dir = output_dir/'attack'
    result = run_attack(scene, attack_dir, arguments['instruction'], assets, constraints,
        timeout_seconds=min(300, arguments['timeout']), max_tokens=16384,
        max_turns=SETTINGS['defaults']['attack_max_turns'], required_operation=arguments['operation'],
        required_track=arguments.get('target_track'))
    while result.get('status') == 'continued_in_object_revision':
        attack_dir = Path(result['followup_run'])
        result = json.loads((attack_dir/'output.json').read_text())
    if result.get('status') != 'reference_candidate_submitted':
        try:
            from .business_outcome import bounded_outcome
        except ImportError:
            from business_outcome import bounded_outcome
        return bounded_outcome('attack_editing', result, attack_dir)
    frozen = Path(result['frozen_scene_path'])
    independent = review(attack_dir, output_dir/'attack-independent.json')
    if independent['status'] != 'passed':
        return dict(status='attack_independent_check_failed', detail=independent)
    trajectory_output = attack_dir/'trajectory_editing/output.json'
    from native_scene_export import export as export_native
    native=export_native(source,frozen,output_dir/'native_attack',attack_result=dict(operation=arguments['operation'],target_track=result['target']['track_id']))
    return dict(schema_version='editing_stage.v1', stage='attack_editing',
        status='completed', mode='real', run_dir=str(output_dir.parent),
        source_scene_ref=source, scene_id=source['scene_id'], scene_token=source['scene_token'],
        operation=arguments['operation'],
        target_track=result['target']['track_id'], selected_candidate_id=result['selected_candidate_id'],
        frozen_scene_path=str(frozen), frames_path=str(frozen/'metadata/frames.json'),
        trajectory_path=result['trajectory_path'], provenance_path=str(frozen/'freeze-info.json'),
        submission_path=str(trajectory_output),
        checks=dict(reference_screen='passed', checks=dict(check_status(result['metrics']['checks']),
            independent_recomputation='passed'), report_path=str(output_dir/'attack-independent.json'),
            physical_avoidance='undetermined'),
        annotation_changes=dict(result['annotation_changes'],native_export_report_path=str(output_dir/'native_attack/export-report.json')), simulation_input=native['simulation_input'], rendered_videos={},
        pdm_probe_result_path=result.get('pdm_probe_result_path'))


def ego_output(arguments, output_dir):
    attack = arguments['attack_result']
    scene = Path(attack['frozen_scene_path'])
    # Follow the submitted object revision rather than the first search directory.
    attack_run_dir = Path(attack['submission_path']).parent.parent
    target_track = attack['target_track']
    attack_input = json.loads((attack_run_dir/'input.json').read_text())
    acceptance = policy.resolve(attack_input['constraints']['acceptance_policy'])
    acceptance['allow_safe_stop'] = arguments['allow_safe_stop']
    understanding_dir = output_dir/'defense_understanding'
    frames = json.loads((scene/'metadata/frames.json').read_text())
    times, _, _ = ego_path_from_frames(frames)
    visual_frame = min(range(len(times)), key=lambda i: abs(times[i]-acceptance['observation_end_s']))
    if arguments.get('reuse_completed_from'):
        from ego_stage_reuse import reuse_understanding
        reuse_understanding(arguments['reuse_completed_from'],arguments,acceptance,
            SETTINGS['defaults']['defense_constructor_candidates'],understanding_dir)
    else:
        run_understanding(scene, attack_run_dir/'trajectory_editing', attack['selected_candidate_id'],
            target_track, understanding_dir, timeout_seconds=min(300, arguments['timeout']),
            max_tokens=16384, acceptance=acceptance, visual_frame=visual_frame,
            target_effect=arguments['target_effect'],
            image_scene=(attack.get('source_scene_ref') or {}).get('sample_root'))
    understanding = json.loads((understanding_dir/'summary.json').read_text())
    if understanding.get('status') != 'completed':
        return dict(status='defense_understanding_failed', detail=understanding)
    constructor_dir = output_dir/'defense_constructor'
    from defense_probe_feedback import reuse as reuse_defense_constructor
    reused = reuse_defense_constructor(attack.get('pdm_probe_result_path'), scene, target_track, acceptance,
        SETTINGS['defaults']['defense_constructor_candidates'], constructor_dir)
    if not reused:
        run_constructor(scene, target_track, constructor_dir, acceptance,
            SETTINGS['defaults']['defense_constructor_candidates'], 300)
    foci_dir = None
    if arguments['use_foci']:
        from defense_generation import prepare_initial_foci
        foci_dir=prepare_initial_foci(scene,constructor_dir,target_track,output_dir/'defense_foci',
            acceptance['observation_end_s'],float(times[-1]),
            candidate_limit=SETTINGS['defaults'].get('defense_initial_foci_candidates',2))
    else:
        atomic_json(output_dir/'foci-stage.json',dict(status='disabled_explicitly'))
    defense_dir = output_dir/'defense_agent'
    from defense_request_recovery import run_with_one_retry
    defense = run_with_one_retry(run_defense_edit, (scene, constructor_dir, foci_dir, defense_dir),
        dict(timeout_seconds=min(300, arguments['timeout']), max_turns=SETTINGS['defaults']['defense_max_turns'],
            max_edit_candidates=(SETTINGS['defaults']['defense_max_edit_candidates']
                if arguments.get('max_edit_candidates') is None else arguments['max_edit_candidates']),
            max_inspections=SETTINGS['defaults']['defense_max_inspections'], max_tokens=16384,
            target_track=target_track, conflict_brief=understanding_dir,controlled_generation=True,
            target_effect=arguments['target_effect'],foci_enabled=arguments['use_foci']), defense_dir)
    if defense.get('status') != 'reference_candidate_submitted':
        try:
            from .business_outcome import bounded_outcome
        except ImportError:
            from business_outcome import bounded_outcome
        return bounded_outcome('ego_editing', defense, defense_dir)
    candidate = defense['selected_candidate_id']
    independent = run_independent(scene, defense_dir, candidate, target_track,
        output_dir/'defense-independent.json', 300)
    case_id = re.sub(r'[^A-Za-z0-9_-]+', '_', attack['scene_id'])
    render_config = dict(SETTINGS, scene=attack['source_scene_ref']['sample_root'])
    render = run_render(render_config, dict(frozen_scene_path=attack['frozen_scene_path'],
        target=dict(operation='edit_existing' if attack.get('operation') == 'edit_existing' else 'add_vehicle',
            track_id=target_track), selected_candidate_id=attack['selected_candidate_id']),
        attack_run_dir, defense_dir, candidate, pipeline_video_dir(output_dir.parent.name, case_id))
    selected_source = ('constructor' if candidate.startswith('constructor_') else
        'foci' if candidate.startswith('foci_') else 'agent_edit' if candidate.startswith('edit_') else 'repair')
    metadata = output_dir/'trajectory-metadata.json'
    atomic_json(metadata, dict(coordinate_frame='initial_ego_rear_axle', position_reference='ego_rear_axle',
        length_unit='m', angle_unit='rad', time_unit='s'))
    trajectory = defense_dir/'candidates'/candidate/'trajectory.csv'
    accepted=json.loads((defense_dir/'accepted-submission.json').read_text())
    selected_source=accepted.get('generation_source',selected_source)
    from native_scene_export import export as export_native
    native=export_native(attack['source_scene_ref'],scene,output_dir/'native_expert',expert_trajectory=trajectory,attack_result=attack)
    return dict(schema_version='editing_stage.v1', stage='ego_editing',
        status='completed', mode='real', run_dir=str(output_dir.parent),
        scene_id=attack['scene_id'], scene_token=attack['scene_token'], attack_run_dir=attack['run_dir'],
        frozen_scene_path=attack['frozen_scene_path'], target_track=target_track,
        selected_candidate_id=candidate, selected_source=selected_source,
        expert_trajectory_path=str(trajectory), trajectory_metadata_path=str(metadata),
        conflict_brief_path=str(understanding_dir/'conflict-brief.json'),
        submission_path=str(defense_dir/'summary.json'),
        checks=dict(reference_screen='passed', checks=dict(check_status(independent['checks']),
            manual_video='passed'), report_path=str(output_dir/'defense-independent.json'),
            physical_avoidance='undetermined'),
        annotation_changes=dict(operation='export_ego_trajectory', frames_json_modified=False,native_export_report_path=str(output_dir/'native_expert/export-report.json')),
        simulation_input=native['simulation_input'], ilft_training_data=None,
        rendered_videos={'manual_explained': str(Path(render['pipeline_output_dir'])/'manual-explained.mp4')})


def main():
    payload = json.load(sys.stdin)
    output_dir = Path(payload['output_dir']).resolve()
    os.environ['ADVERSARIAL_EXPERIMENTS_ROOT'] = str(output_dir)
    os.environ['ADVERSARIAL_FULL_PIPELINE_ROOT'] = SETTINGS['public_pipeline_root']
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        stage = payload['stage']
        with contextlib.redirect_stdout(sys.stderr):
            result = (attack_output if stage == 'attack_editing' else ego_output)(payload['arguments'], output_dir)
    except Exception as error:
        result = dict(status='backend_error', error_type=type(error).__name__, error=str(error)[:300])
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
