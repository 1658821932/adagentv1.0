"""Re-export an existing expert without invoking trajectory generation or models."""
import copy
import json
import shutil
from pathlib import Path


def _inside(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Completed expert artifact missing or outside prior output: ' + str(path))
    return path


def reexport(arguments, output_dir):
    from native_scene_export import export, export_summary
    from run_adversarial_loop import run_independent
    source = arguments.get('reuse_completed_from')
    if not source:
        raise ValueError('export_only requires reuse_completed_from with a completed expert')
    source = Path(source).resolve()
    prior_root = source/'output'
    prior = json.loads((prior_root/'result.json').read_text())
    old_args = json.loads((source/'input/stage_request.json').read_text())['arguments']
    attack = arguments['attack_result']
    if prior.get('status') != 'completed' or prior.get('stage') != 'ego_editing':
        raise ValueError('export_only requires a completed ego_editing result')
    for field, expected in [('scene_id', attack['scene_id']), ('scene_token', attack['scene_token']),
            ('attack_run_dir', attack['run_dir']), ('frozen_scene_path', attack['frozen_scene_path']),
            ('target_track', attack['target_track'])]:
        if prior.get(field) != expected:
            raise ValueError('export_only attack binding mismatch: ' + field)
    for key in ('target_effect', 'allow_safe_stop', 'require_route_progress'):
        if old_args.get(key) != arguments.get(key):
            raise ValueError('export_only task changed: ' + key)
    if prior.get('checks', {}).get('reference_screen') != 'passed':
        raise ValueError('export_only requires a previously accepted expert')
    trajectory = _inside(prior['expert_trajectory_path'], prior_root)
    candidate = prior['selected_candidate_id']
    if trajectory.parent.name != candidate or trajectory.parent.parent.name != 'candidates':
        raise ValueError('Completed trajectory does not match selected candidate')
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Recompute acceptance against the current frozen scene before exporting old poses.
    independent = run_independent(Path(attack['frozen_scene_path']), trajectory.parents[2],
        candidate, attack['target_track'], output_dir/'defense-independent.json', 300)
    result = copy.deepcopy(prior)
    for key, name in [('expert_trajectory_path', 'expert-trajectory.csv'),
            ('trajectory_metadata_path', 'trajectory-metadata.json'),
            ('conflict_brief_path', 'conflict-brief.json'), ('submission_path', 'expert-submission.json')]:
        original = _inside(prior[key], prior_root)
        destination = output_dir/name
        shutil.copyfile(original, destination)
        result[key] = str(destination)
    native = export(attack['source_scene_ref'], Path(attack['frozen_scene_path']),
        output_dir/'native_expert', expert_trajectory=Path(result['expert_trajectory_path']), attack_result=attack)
    result.update(run_dir=str(output_dir.parent), simulation_input=native['simulation_input'],
        native_export=dict(export_summary(native), reexported_from=str(source)),
        rendered_videos={}, render_status=dict(status='not_run'), ilft_training_data=None,
        checks=dict(reference_screen='passed', checks={k: 'passed' if v else 'failed'
            for k, v in independent['checks'].items()},
            report_path=str(output_dir/'defense-independent.json'), physical_avoidance='undetermined'))
    result['annotation_changes'] = dict(operation='reexport_existing_expert', frames_json_modified=False,
        native_export_report_path=native['report_path'])
    return result
