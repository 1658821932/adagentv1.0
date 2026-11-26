"""editing_stage.v1 JSON backend for the independent adversarial editing package."""
import contextlib
import json
import os
import re
from subprocess import TimeoutExpired
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
from backend_configuration import load_settings
SETTINGS = load_settings()


def check_status(checks):
    return {name: 'passed' if value else 'failed' for name, value in checks.items()}


def native_export_view(report):
    from native_scene_export import export_summary
    return export_summary(report)


def attack_output(arguments, output_dir):
    source = arguments['scene_ref']
    scene = Path(source['sample_root'])
    constraints_input = arguments.get('constraints', {})
    acceptance = policy.for_new_run(arguments.get('acceptance_policy'))
    if any(arguments.get(key) for key in ('pdm_self_check','pdm_defense_feedback','pdm_ignore_traffic_lights')):
        raise ValueError('Attack editing does not run PDM. Main Agent evaluates with policy_runner(native) after expert editing.')
    if acceptance['policy_version']=='reference_acceptance.v2':
        acceptance['attack_danger_start_s']=max(acceptance['attack_window_start_s'],
            float(constraints_input.get('no_ego_contact_until_s',acceptance['attack_danger_start_s'])))
    # 攻击车只锁起点（prefix_lock=0），危险必须落在显式窗口内；
    # 这与自车历史无关，自车的历史由 policy_runner 的 num_history 决定。
    target_window = constraints_input.get('target_window_s')
    if target_window is not None:
        w0, w1 = float(target_window[0]), float(target_window[1])
        min_w, max_w = float(acceptance['attack_window_start_s']), float(acceptance['attack_window_end_s'])
        if not (min_w <= w0 < w1 <= max_w):
            raise ValueError(f'constraints.target_window_s {target_window} must lie within [{min_w}, {max_w}]')
        event_window = [w0, w1]
    else:
        event_window = [acceptance['attack_window_start_s'], acceptance['attack_window_end_s']]
    explicit_policy = arguments.get('acceptance_policy') or {}
    attack_official_margin_m = (
        acceptance['official_margin_m']
        if isinstance(explicit_policy, dict) and 'official_margin_m' in explicit_policy
        else -0.25
    )
    acceptance = dict(acceptance, official_margin_m=attack_official_margin_m)
    constraints = dict(observation_end_s=acceptance['observation_end_s'],
        attack_prefix_lock_s=acceptance['attack_prefix_lock_s'],
        event_window_s=event_window,
        official_margin_m=attack_official_margin_m, max_ego_clearance_m=2., min_travel_m=0.,
        vehicle_motion_model='bicycle.v1', require_structured_objective=False,
        agent_decides_interaction=True,
        require_reference_contact=True,reference_contact_policy='target_body_contact_in_event_window.v1',
        acceptance_policy=acceptance)
    constraints['attack_stage_timeout_s'] = min(arguments.get('stage_timeout_s',3600),max(1,arguments['timeout']-60))
    constraints['generated_objectives_are_preferences'] = True
    constraints['prepare_ego_history']=arguments.get('prepare_ego_history',True)
    constraints['ego_history_target_speed_mps']=arguments.get('ego_history_target_speed_mps')
    if arguments.get('ego_history_route') is not None:
        constraints['ego_history_route'] = arguments['ego_history_route']
    required_operation=None if arguments['operation']=='auto' else arguments['operation']
    if arguments.get('plan_options'):
        constraints['plan_options']=arguments['plan_options']
    if arguments.get('task_contract') is not None:
        if 'interaction' in arguments['task_contract'].get('objective',{}):
            raise ValueError('Interaction categories are removed; task_contract may state explicit keyframes/corridor, not contact/near-miss classes')
        constraints['task_contract'] = arguments['task_contract']
    if arguments.get('revision_feedback') is not None:
        constraints['revision_feedback'] = arguments['revision_feedback']
    if arguments.get('maneuver') == 'stationary':
        constraints['stationary_target'] = True
    if required_operation == 'add_vehicle':
        # Added vehicles have no original annotation to stay anchored to; allow connector_route entry alignment.
        constraints['initial_pose_max_translation_m'] = 80.0
        constraints['initial_pose_max_yaw_change_rad'] = 3.141592653589793
        constraints['initial_speed_max_change_mps'] = 12.0
    if arguments.get('reference_kinematics') is not None:
        constraints['reference_kinematics'] = arguments['reference_kinematics']
    if arguments.get('reference_bev_image') is not None:
        constraints['reference_bev_image'] = arguments['reference_bev_image']
    if arguments.get('spatial_hint') is not None:
        constraints['spatial_hint'] = arguments['spatial_hint']
    assets = []
    if arguments.get('vehicle') is not None:
        vehicle = arguments['vehicle']
        assets.append(dict(asset_id='asset_001', path=vehicle['asset_path'], size_m=vehicle['size_m']))
    elif arguments.get('vehicle_catalog_class') is not None:
        from vehicle_asset_catalog import load_vehicle_assets
        assets = load_vehicle_assets(arguments['vehicle_catalog_class'])
    if required_operation == 'ego_only':
        from attack_ego_preparation import prepare_acceleration
        from native_scene_export import export as export_native
        ego_dir = output_dir / 'attack' / 'ego_history'
        route_args = {'route': constraints['ego_history_route']} if constraints.get('ego_history_route') is not None else {}
        preparation = prepare_acceleration(
            scene, ego_dir, acceptance,
            enabled=constraints['prepare_ego_history'],
            target_speed_mps=constraints.get('ego_history_target_speed_mps'),
            **route_args,
        )
        if preparation['status'] == 'unavailable':
            return dict(
                schema_version='editing_stage.v1', stage='attack_editing',
                status='unable_to_complete', mode='real', run_dir=str(output_dir.parent),
                reason=preparation['reason'], stop_reason='ego_history_unavailable',
                failed_requirements=['ego_history_prepare'], failed_requirements_scope='ego_history',
                attempts=preparation.get('attempts', []), evidence_paths=[preparation['report_path']],
                scope='single_vehicle_ego_only', global_infeasibility_proven=False,
                detail=preparation,
            )
        frozen = Path(preparation['scene'])
        if not frozen.resolve().is_relative_to(output_dir.resolve()):
            import shutil
            mat = output_dir / 'attack' / 'materialized' / 'frozen_scene'
            mat.mkdir(parents=True, exist_ok=True)
            for item in frozen.iterdir():
                dst = mat / item.name
                if item.name == 'metadata' and item.is_dir():
                    shutil.copytree(item, dst, dirs_exist_ok=True)
                elif not dst.exists():
                    dst.symlink_to(item.resolve())
            frozen = mat
        native = export_native(source, frozen, output_dir / 'native_attack',
                               attack_result=dict(operation='ego_only', target_track=None))
        shared_history = dict(preparation['shared_state'],
                              preparation_status=preparation['status'],
                              preparation_report_path=preparation['report_path'])
        history_path = frozen / 'ego-history.json'
        if history_path.is_file():
            history = json.loads(history_path.read_text())
            for key in ('takeover_s', 'targets', 'requested_takeover_speed_mps', 'delivered_takeover_speed_mps'):
                if key in history:
                    shared_history[key] = history[key]
            shared_history['manifest_path'] = str(history_path)
        return dict(
            schema_version='editing_stage.v1', stage='attack_editing',
            status='completed', mode='real', run_dir=str(output_dir.parent),
            source_scene_ref=source, scene_id=source['scene_id'], scene_token=source['scene_token'],
            operation='ego_only', plan_choice=None, shared_history=shared_history,
            target_track=None, selected_candidate_id='ego_only',
            frozen_scene_path=str(frozen), frames_path=str(frozen / 'metadata/frames.json'),
            trajectory_path=None, provenance_path=preparation['report_path'],
            submission_path=preparation['report_path'],
            checks=dict(reference_screen='passed',
                        checks={'ego_history_prepare': 'passed'},
                        report_path=preparation['report_path'],
                        physical_avoidance='undetermined'),
            annotation_changes=dict(operation='ego_only',
                                    frames_json_modified=bool(preparation['status'] == 'prepared'),
                                    native_export_report_path=str(output_dir / 'native_attack/export-report.json')),
            simulation_input=native['simulation_input'],
            native_export=native_export_view(native),
            rendered_videos={},
            pdm_probe_result_path=None,
        )
    attack_dir = output_dir/'attack'
    continue_from = None
    reuse_source = arguments.get('reuse_completed_from')
    if reuse_source:
        candidate_dir = Path(reuse_source).resolve() / 'output/attack'
        while (candidate_dir / 'output.json').is_file():
            try:
                prior_out = json.loads((candidate_dir / 'output.json').read_text())
            except Exception:
                break
            if prior_out.get('status') == 'continued_in_object_revision' and prior_out.get('followup_run'):
                candidate_dir = Path(prior_out['followup_run']).resolve()
                continue
            break
        reuse_reason = None
        try:
            prior_input_path = candidate_dir / 'input.json'
            prior_selection_path = candidate_dir / 'object_selection/output.json'
            if not prior_input_path.is_file():
                raise ValueError('Missing attack checkpoint metadata')
            prior_selection = json.loads(prior_selection_path.read_text()) if prior_selection_path.is_file() else {}
            if prior_selection.get('status') != 'completed':
                atomic_json(output_dir/'attack-reuse.json', dict(status='recompute',
                    reason='Prior object_selection did not complete', source=str(candidate_dir)))
            else:
                prior_input = json.loads(prior_input_path.read_text())
                from ego_history import source_scene
                if (source_scene(prior_input['scene']) != scene.resolve()
                        or prior_input.get('instruction') != arguments['instruction']
                        or prior_input.get('assets') != assets
                        or prior_input.get('required_operation') != required_operation
                        or prior_input.get('required_track') != arguments.get('target_track')):
                    raise ValueError('Prior attack scene, instruction, assets or target requirement differs')
                from attack_configuration import same_scene_constraints
                if not same_scene_constraints(prior_input.get('constraints',{}), constraints):
                    raise ValueError('Prior attack constraints differ')
                for key in ('object_reselection_count','object_reselection_reason','object_selection_feedback','ego_history_revision_count','expert_margin_revision_count'):
                    if key in prior_input.get('constraints',{}):constraints[key]=prior_input['constraints'][key]
                continue_from = candidate_dir
                atomic_json(output_dir/'attack-reuse.json', dict(status='reused', source=str(candidate_dir)))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            reuse_reason = str(exc)
            atomic_json(output_dir/'attack-reuse.json', dict(status='recompute', reason=reuse_reason, source=str(candidate_dir)))
            raise ValueError('Requested attack checkpoint is incompatible: '+reuse_reason) from exc
    result = run_attack(scene, attack_dir, arguments['instruction'], assets, constraints,
        timeout_seconds=min(300, arguments['timeout']), max_tokens=16384,
        max_turns=SETTINGS['defaults']['attack_max_turns'], required_operation=required_operation,
        required_track=arguments.get('target_track'), continue_from=continue_from)
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
    native=export_native(source,frozen,output_dir/'native_attack',attack_result=dict(operation=result['target']['operation'],target_track=result['target']['track_id']))
    policy_probe=None
    history_path=frozen/'ego-history.json'
    shared_history=None
    if history_path.is_file():
        history=json.loads(history_path.read_text())
        shared_history={key:history[key] for key in ('takeover_s','targets','requested_takeover_speed_mps','delivered_takeover_speed_mps')}
        if history.get('mode') == 'endpoint_route':
            shared_history.update({key: history[key] for key in ('mode', 'route', 'takeover_xy_m', 'takeover_yaw_rad',
                'continuous_takeover_speed_mps', 'errors', 'reference_rejoin_s')})
        shared_history['manifest_path']=str(history_path)
    preparation=result.get('ego_history_preparation')
    if preparation is not None:
        shared_history=dict(shared_history or {},**preparation['shared_state'],
            preparation_status=preparation['status'],preparation_report_path=preparation['report_path'])
    return dict(schema_version='editing_stage.v1', stage='attack_editing',
        status='completed', mode='real', run_dir=str(output_dir.parent),
        source_scene_ref=source, scene_id=source['scene_id'], scene_token=source['scene_token'],
        operation=result['target']['operation'],plan_choice=result.get('plan_choice'),shared_history=shared_history,
        target_track=result['target']['track_id'], selected_candidate_id=result['selected_candidate_id'],
        frozen_scene_path=str(frozen), frames_path=str(frozen/'metadata/frames.json'),
        trajectory_path=result['trajectory_path'], provenance_path=str(frozen/'freeze-info.json'),
        submission_path=str(trajectory_output),
        checks=dict(reference_screen='passed', checks=dict(check_status(result['metrics']['checks']),
            independent_recomputation='passed'), report_path=str(output_dir/'attack-independent.json'),
            physical_avoidance='undetermined'),
        annotation_changes=dict(result['annotation_changes'],native_export_report_path=str(output_dir/'native_attack/export-report.json')), simulation_input=native['simulation_input'], native_export=native_export_view(native), rendered_videos={},
        pdm_probe_result_path=result.get('pdm_probe_result_path'))


def ego_output(arguments, output_dir):
    if arguments.get('export_only'):
        from completed_expert_export import reexport
        return reexport(arguments, output_dir)
    attack = arguments['attack_result']
    scene = Path(attack['frozen_scene_path'])
    # Follow the submitted object revision rather than the first search directory.
    attack_run_dir = Path(attack['submission_path']).parent.parent
    target_track = attack['target_track']
    from shared_history import shared_history_end_index
    frames = json.loads((scene/'metadata/frames.json').read_text())
    times, rec_xy, rec_yaw = ego_path_from_frames(frames)
    visual_frame = shared_history_end_index(times)
    takeover_s = float(times[visual_frame])
    attack_input = json.loads((attack_run_dir/'input.json').read_text())
    acceptance = policy.for_new_run(attack_input['constraints']['acceptance_policy'])
    acceptance['edit_start_s'] = takeover_s
    acceptance['observation_end_s'] = takeover_s
    acceptance, require_route_progress = policy.resolve_ego_progress_policy(
        acceptance, target_effect=arguments['target_effect'],
        allow_safe_stop=arguments['allow_safe_stop'],
        require_route_progress=arguments.get('require_route_progress'))
    atomic_json(output_dir/'effective-policy.json', acceptance)
    understanding_dir = output_dir/'defense_understanding'
    from ego_stage_reuse import (binding as stage_binding, restore as restore_stage, save_completed,
        preparation_deadline, PreparationTimeout)
    count = SETTINGS['defaults']['defense_constructor_candidates']
    foci_count = SETTINGS['defaults'].get('defense_initial_foci_candidates', 2)
    preparation_seconds = SETTINGS['defaults'].get('defense_preparation_timeout_seconds', 300)
    def preparation_failure(stage, error):
        detail = dict(status='not_submitted', stop_reason='preparation_timeout',
            reason=str(error), preparation_stage=stage, timeout_seconds=preparation_seconds)
        stage_dir = output_dir/stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(stage_dir/'summary.json', detail)
        try:
            from business_outcome import bounded_outcome
        except ImportError:
            from business_outcome import bounded_outcome
        return bounded_outcome('ego_editing', detail, stage_dir)
    reuse_source = arguments.get('reuse_completed_from')
    reused_understanding = restore_stage('defense_understanding', reuse_source, arguments,
        acceptance, count, foci_count, output_dir)
    if not reused_understanding:
        try:
            with preparation_deadline('defense_understanding', preparation_seconds):
                understanding_binding = stage_binding('defense_understanding', arguments, acceptance, count, foci_count, output_dir)
                run_understanding(scene, attack_run_dir/'trajectory_editing', attack['selected_candidate_id'],
                    target_track, understanding_dir, timeout_seconds=min(300, arguments['timeout']),
                    max_tokens=16384, acceptance=acceptance, visual_frame=visual_frame,
                    target_effect=arguments['target_effect'],
                    image_scene=(attack.get('source_scene_ref') or {}).get('sample_root'))
                if json.loads((understanding_dir/'summary.json').read_text()).get('status') == 'completed':
                    save_completed('defense_understanding', arguments, acceptance, count, foci_count, output_dir, understanding_binding)
        except PreparationTimeout as error:
            return preparation_failure('defense_understanding', error)
    understanding = json.loads((understanding_dir/'summary.json').read_text())
    if understanding.get('status') != 'completed':
        return dict(status='defense_understanding_failed', detail=understanding)
    constructor_dir = output_dir/'defense_constructor'
    from defense_probe_feedback import reuse as reuse_defense_constructor
    reused = restore_stage('defense_constructor', reuse_source, arguments, acceptance, count, foci_count, output_dir)
    if not reused:
        try:
            with preparation_deadline('defense_constructor', preparation_seconds):
                constructor_start = stage_binding('defense_constructor', arguments, acceptance, count, foci_count, output_dir)
                reused = reuse_defense_constructor(attack.get('pdm_probe_result_path'), scene, target_track, acceptance,
                    count, constructor_dir)
                if not reused:
                    run_constructor(scene, target_track, constructor_dir, acceptance, count, preparation_seconds,
                        resume=(constructor_dir/'reference_screen').exists())
                save_completed('defense_constructor', arguments, acceptance, count, foci_count, output_dir, constructor_start)
        except PreparationTimeout as error:
            return preparation_failure('defense_constructor', error)
    foci_dir = None
    from defense_generation import should_prepare_initial_foci
    if should_prepare_initial_foci(arguments, default=SETTINGS['defaults'].get('defense_initial_foci_enabled', False)):
        from defense_generation import prepare_initial_foci
        reused_foci = restore_stage('defense_foci', reuse_source, arguments, acceptance, count, foci_count, output_dir)
        foci_stage_dir = output_dir/'defense_foci'
        if reused_foci:
            foci_state = json.loads((foci_stage_dir/'initial-stage.json').read_text())
            atomic_json(output_dir/'foci-stage.json', foci_state)
            foci_dir = foci_stage_dir if foci_state['status'] == 'completed' else None
        else:
            try:
                with preparation_deadline('defense_foci', preparation_seconds):
                    foci_start = stage_binding('defense_foci', arguments, acceptance, count, foci_count, output_dir)
                    foci_dir=prepare_initial_foci(scene,constructor_dir,target_track,foci_stage_dir,
                        policy.edit_start(acceptance),float(times[-1]), candidate_limit=foci_count)
                    foci_stage_dir.mkdir(parents=True, exist_ok=True)
                    atomic_json(foci_stage_dir/'initial-stage.json', json.loads((output_dir/'foci-stage.json').read_text()))
                    save_completed('defense_foci', arguments, acceptance, count, foci_count, output_dir, foci_start)
            except PreparationTimeout as error:
                return preparation_failure('defense_foci', error)
    elif arguments['use_foci']:
        atomic_json(output_dir/'foci-stage.json',dict(status='on_demand'))
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
    import numpy as np
    from ego_road_checks import read_trajectory_csv
    from native_scene_export import export as export_native, validate_expert_trajectory
    trajectory = defense_dir/'candidates'/candidate/'trajectory.csv'
    ex_t, ex_xy, ex_yaw = read_trajectory_csv(trajectory)
    validate_expert_trajectory(times, rec_xy, rec_yaw, (ex_t, ex_xy, ex_yaw))
    from expert_attack_feedback import collect as collect_expert_feedback
    expert_feedback=collect_expert_feedback(attack,defense_dir,candidate,
        output_dir/'defense-independent.json',acceptance,output_dir/'expert-attack-feedback.json')
    selected_source = ('constructor' if candidate.startswith('constructor_') else
        'foci' if candidate.startswith('foci_') else 'agent_edit' if candidate.startswith('edit_') else 'repair')
    metadata = output_dir/'trajectory-metadata.json'
    atomic_json(metadata, dict(coordinate_frame='initial_ego_rear_axle', position_reference='ego_rear_axle',
        length_unit='m', angle_unit='rad', time_unit='s'))
    accepted=json.loads((defense_dir/'accepted-submission.json').read_text())
    selected_source=accepted.get('generation_source',selected_source)
    native=export_native(attack['source_scene_ref'],scene,output_dir/'native_expert',expert_trajectory=trajectory,attack_result=attack)
    case_id = re.sub(r'[^A-Za-z0-9_-]+', '_', attack['scene_id'])
    render_config = dict(SETTINGS, scene=attack['source_scene_ref']['sample_root'])
    render_output = pipeline_video_dir(output_dir.parent.name, case_id)
    try:
        render = run_render(render_config, dict(frozen_scene_path=attack['frozen_scene_path'],
            target=dict(operation='edit_existing' if attack.get('operation') == 'edit_existing' else 'add_vehicle',
                track_id=target_track), selected_candidate_id=attack['selected_candidate_id']),
            attack_run_dir, defense_dir, candidate, render_output)
        render_status = dict(status='passed' if render.get('pipeline_output_dir') else 'not_run')
    except (OSError, RuntimeError, ValueError, TimeoutExpired) as error:
        # A missing visualization must not discard a reviewed and exported trajectory.
        render = {}
        render_status = dict(status='failed', error_type=type(error).__name__, reason=str(error)[:500])
        render_log = Path(str(render_output)+'.log')
        if render_log.is_file():
            render_status['log_path'] = str(render_log)
    atomic_json(output_dir/'render-status.json', render_status)
    ex_yaw = np.unwrap(ex_yaw)
    traveled_m = float(np.linalg.norm(np.diff(ex_xy, axis=0), axis=1).sum())
    terminal_speed_mps = float(np.linalg.norm(ex_xy[-1] - ex_xy[-2]) / (ex_t[-1] - ex_t[-2])) if len(ex_t) > 1 else 0.0
    rec_yaw = np.unwrap(rec_yaw)
    recorded_progress_m = float(np.linalg.norm(np.diff(rec_xy, axis=0), axis=1).sum())
    _, progress_reason = policy.task_progress(
        ex_t, ex_xy, recorded_progress_m, acceptance,
        baseline_stopped=policy.is_terminal_stopped(times, rec_xy, acceptance))
    expert_motion_summary = dict(
        traveled_m=round(traveled_m, 3),
        recorded_progress_m=round(recorded_progress_m, 3),
        progress_ratio_of_recorded=(None if recorded_progress_m <= 1e-6 else round(traveled_m / recorded_progress_m, 4)),
        yaw_change_deg=round(float(np.degrees(abs(ex_yaw[-1] - ex_yaw[0]))), 2),
        recorded_yaw_change_deg=round(float(np.degrees(abs(rec_yaw[-1] - rec_yaw[0]))), 2),
        terminal_speed_mps=round(terminal_speed_mps, 3),
        terminal_stopped=policy.is_terminal_stopped(ex_t, ex_xy, acceptance),
        progress_reason=progress_reason,
        allow_safe_stop=acceptance['allow_safe_stop'],
        minimum_progress_fraction=acceptance['minimum_progress_fraction'],
        require_route_progress=require_route_progress,
        task_goal_assessment='agent_review_required',
    )
    return dict(schema_version='editing_stage.v1', stage='ego_editing',
        status='completed', mode='real', run_dir=str(output_dir.parent),
        scene_id=attack['scene_id'], scene_token=attack['scene_token'], attack_run_dir=attack['run_dir'],
        frozen_scene_path=attack['frozen_scene_path'], target_track=target_track,
        selected_candidate_id=candidate, selected_source=selected_source,
        expert_trajectory_path=str(trajectory), trajectory_metadata_path=str(metadata),
        conflict_brief_path=str(understanding_dir/'conflict-brief.json'),
        submission_path=str(defense_dir/'summary.json'),
        checks=dict(reference_screen='passed', checks=check_status(independent['checks']),
            report_path=str(output_dir/'defense-independent.json'),
            physical_avoidance='undetermined'),
        annotation_changes=dict(operation='export_ego_trajectory', frames_json_modified=False,native_export_report_path=str(output_dir/'native_expert/export-report.json')),
        simulation_input=native['simulation_input'], native_export=native_export_view(native), expert_motion_summary=expert_motion_summary, expert_feedback=expert_feedback, ilft_training_data=None,
        render_status=render_status,
        rendered_videos=({'manual_explained': str(Path(render['pipeline_output_dir'])/'manual-explained.mp4')}
            if render.get('pipeline_output_dir') else {}))


def main():
    payload = json.load(sys.stdin)
    output_dir = Path(payload['output_dir']).resolve()
    os.environ['ADVERSARIAL_EXPERIMENTS_ROOT'] = str(output_dir)
    os.environ['ADVERSARIAL_CONTINUATION_ROOT'] = str(output_dir.parent.parent)
    # The host adapter supplies the public path; standalone runs stay in output.
    os.environ.setdefault('ADVERSARIAL_FULL_PIPELINE_ROOT', str(output_dir / 'rendered'))
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        stage = payload['stage']
        with contextlib.redirect_stdout(sys.stderr):
            result = (attack_output if stage == 'attack_editing' else ego_output)(payload['arguments'], output_dir)
    except Exception as error:
        result = dict(status='backend_error', error_type=type(error).__name__, error=str(error)[:300])
        if getattr(error, 'code', None) == 'editing_configuration_error':
            result.update(error_code=error.code, failure_stage='configuration', retryable=False,
                error_details=error.details,
                recovery_hint='Repair the backend configuration or conflicting arguments; changing the scene or attack instruction cannot fix this error. Do not retry unchanged.')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
