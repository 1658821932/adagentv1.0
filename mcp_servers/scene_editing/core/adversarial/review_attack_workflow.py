"""Recompute a completed two-stage attack from saved paths and frozen annotations."""
from workflow_timing import timed_operation

import argparse
import json
from pathlib import Path

import acceptance_policy as policy
import numpy as np

from attack_editor import read_seed
from attack_constructor import goal_evidence
from ego_road_checks import load_official_drivable
from experiment_runtime import atomic_json
from run_ego_reference import ego_path_from_frames, observed_objects_from_frames, scene_from_frames
from vehicle_trajectory import sample_xy, screen_attack


@timed_operation('attack.independent_check', 'run_dir')
def review(run_dir, output):
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing review')
    run_input = json.loads((run_dir / 'input.json').read_text())
    result = json.loads((run_dir / 'output.json').read_text())
    if result.get('status') != 'reference_candidate_submitted':
        raise ValueError('Review requires a completed and materialized attack')
    selection = json.loads((run_dir / 'object_selection/output.json').read_text())
    trajectory_output = json.loads((run_dir / 'trajectory_editing/output.json').read_text())
    source = Path(run_input['scene'])
    original = json.loads((source / 'metadata/frames.json').read_text())
    frozen = json.loads((Path(result['frozen_scene_path']) / 'metadata/frames.json').read_text())
    times, ego, ego_yaw = ego_path_from_frames(original)
    native_times, xy, yaw = read_seed(result['trajectory_path'])
    _, seed_xy, seed_yaw = read_seed(selection['seed_trajectory_path'])
    dense_times, dense_xy, dense_yaw = read_seed(Path(result['trajectory_path']).with_name('dense_trajectory.csv'))
    recovered_times, _, _, frozen_actors = scene_from_frames(frozen, result['target']['track_id'])
    recovered = [next(actor for actor in frame if actor['track'] == result['target']['track_id']) for frame in frozen_actors]
    recovered_xy = np.array([[actor['x'], actor['y']] for actor in recovered])
    recovered_yaw = np.array([actor['yaw'] for actor in recovered])
    road = load_official_drivable(source / 'metadata/map_drivable_official.json')
    recomputed = screen_attack(times, ego, ego_yaw, observed_objects_from_frames(original),
        result['target'], seed_xy, seed_yaw, dense_times, dense_xy, dense_yaw, road, run_input['constraints'])
    replay = screen_attack(times, ego, ego_yaw, observed_objects_from_frames(original),
        result['target'], seed_xy, seed_yaw, dense_times, sample_xy(times, xy, dense_times),
        np.interp(dense_times, times, np.unwrap(yaw)), road, run_input['constraints'], motion_times=times, motion_xy=xy)
    target_id = result['target']['track_id']
    others_preserved = True
    for original_frame, frozen_frame in zip(original, frozen):
        before, after = original_frame['anns'], frozen_frame['anns']
        for index, track in enumerate(before['track_tokens']):
            if str(track) == target_id:
                continue
            matches = [i for i, value in enumerate(after['track_tokens']) if value == track]
            if len(matches) != 1 or any(before[key][index] != after[key][matches[0]] for key in before):
                others_preserved = False
    checks = dict(stage_identity=selection['target']['track_id'] == target_id == trajectory_output['target_track'],
        candidate_identity=trajectory_output['selected_candidate_id'] == result['selected_candidate_id'],
        timestamps=(len(original) == len(frozen) == len(native_times) and
                    np.allclose(times, native_times, atol=1e-6, rtol=0) and
                    np.allclose(times, recovered_times, atol=1e-6, rtol=0)),
        original_ego_preserved=all(a['ego2global'] == b['ego2global'] for a, b in zip(original, frozen)),
        other_annotations_preserved=others_preserved,
        target_position_roundtrip=bool(np.max(abs(recovered_xy - xy)) < 1e-6),
        target_yaw_roundtrip=bool(np.max(abs(np.arctan2(np.sin(recovered_yaw - yaw), np.cos(recovered_yaw - yaw)))) < 1e-6),
        recomputed_dense_reference=recomputed['reference_screen_passed'],
        native_replay_geometry=replay['reference_screen_passed'])
    saved = json.loads(Path(result['trajectory_path']).with_name('evaluation.json').read_text())
    operation_path=Path(result['trajectory_path']).with_name('operation.json')
    if operation_path.is_file():
        operation=json.loads(operation_path.read_text())['operation']
        if isinstance(operation,dict) and operation.get('tool') in ('edit_speed_profile','edit_route'):
            from ego_trajectory_edit import modern_edit
            from scipy.interpolate import PPoly
            folder=Path(result['trajectory_path']).parent.parent
            def lookup(candidate_id):
                if candidate_id=='recorded_baseline':return seed_xy,seed_yaw
                if candidate_id=='seed_000':return seed_xy,seed_yaw
                _,x,a=read_seed(folder/candidate_id/'trajectory.csv');return x,a
            args=operation['input'];parent_curve_path=folder/args['parent_candidate_id']/'curve-check.json'
            parent_curve=json.loads(parent_curve_path.read_text()) if parent_curve_path.is_file() else None
            _,parent_xy,parent_yaw=read_seed(folder/args['parent_candidate_id']/'dense_trajectory.csv') if args['parent_candidate_id']!='seed_000' else (dense_times,sample_xy(times,seed_xy,dense_times),np.interp(dense_times,times,seed_yaw))
            expected_xy,expected_yaw,curve=modern_edit(times,lookup,args,run_input['constraints'].get('observation_end_s',3.),parent_curve=parent_curve,acceptance=run_input['constraints'].get('acceptance_policy'),
                parent_context=dict(reference_point='target_body_center',rear_axle_offset_m=.3*float(result['target']['size_m'][0]),samples=(dense_times,parent_xy,parent_yaw)))
            cont=curve['continuous_state'];spline=PPoly(np.asarray(cont['xy_coefficients']),np.asarray(cont['times_s']));angle=PPoly(np.asarray(cont['yaw_coefficients']),np.asarray(cont['times_s']))
            _,parent_xy,parent_yaw=read_seed(folder/args['parent_candidate_id']/'dense_trajectory.csv') if args['parent_candidate_id']!='seed_000' else (dense_times,sample_xy(times,seed_xy,dense_times),np.interp(dense_times,times,seed_yaw))
            expected_dense=parent_xy.copy();expected_angles=parent_yaw.copy();mask=dense_times>spline.x[0];expected_dense[mask]=spline(dense_times[mask]);expected_angles[mask]=angle(dense_times[mask])
            effective=policy.resolve(run_input['constraints'].get('acceptance_policy'))
            checks['continuous_operation_reconstruction']=bool(np.allclose(expected_xy,xy,atol=1e-8,rtol=0) and np.allclose(expected_yaw,yaw,atol=1e-8,rtol=0) and np.allclose(expected_dense,dense_xy,atol=1e-8,rtol=0) and np.allclose(expected_angles,dense_yaw,atol=1e-8,rtol=0))
            checks['continuous_motion']=curve['max_speed_mps']<=effective['max_speed_mps']+1e-8 and curve['max_acceleration_mps2']<=effective['max_acceleration_mps2']+1e-8
            if curve.get('edit_contract_version')=='trajectory_edit.v2':
                from edit_feedback import intent_checks
                checks['edit_intent']=intent_checks(curve)['passed']
    construction_goal = None
    if saved.get('construction_goal_input') is not None:
        arguments = saved['construction_goal_input']
        ego_dense = sample_xy(times, ego, dense_times)
        ego_angles = np.interp(dense_times, times, ego_yaw)
        dense_goal = goal_evidence(dense_times, dense_xy, ego_dense, ego_angles, arguments,
                                  recomputed['first_contact_time_s'],target_yaw=dense_yaw,target_size=result['target']['size_m'], task_contract=run_input['constraints'].get('task_contract'))
        native_goal = goal_evidence(dense_times, sample_xy(times, xy, dense_times), ego_dense,
            ego_angles, arguments, replay['first_contact_time_s'],
            target_yaw=np.interp(dense_times,times,np.unwrap(yaw)),target_size=result['target']['size_m'], task_contract=run_input['constraints'].get('task_contract'))
        checks['construction_goal'] = dense_goal['goal_met'] and native_goal['goal_met']
        construction_goal = dict(dense=dense_goal, native_replay=native_goal)
    if run_input['constraints'].get('vehicle_motion_model') == 'bicycle.v1':
        from vehicle_motion import check_bicycle
        checks['vehicle_motion']=check_bicycle(dense_times,dense_xy,dense_yaw,result['target']['size_m'])['passed']
    if run_input['constraints'].get('task_contract') or (run_input['constraints'].get('require_structured_objective') and not run_input['constraints'].get('stationary_target')):
        checks['structured_objective'] = 'objective' in saved.get('construction_goal_input', {})
    arguments = saved.get('construction_goal_input', {})
    if arguments.get('preserve_outside_window'):
        from vehicle_control_search import preservation_evidence
        parent = Path(result['trajectory_path']).parent.parent / arguments['parent_candidate_id'] / 'dense_trajectory.csv'
        parent_times, parent_xy, parent_yaw = read_seed(parent)
        checks['local_preservation'] = bool(np.allclose(parent_times, dense_times, atol=1e-8, rtol=0) and
            preservation_evidence(dense_times, dense_xy, dense_yaw, parent_xy, parent_yaw, arguments)['passed'])
    if result['target']['operation'] == 'add_vehicle':
        from road_height import RoadHeightMap, read_recon_translation
        surface = RoadHeightMap.load(source)
        origin = np.asarray(original[0]['ego2global'])
        angle = np.arctan2(origin[1, 0], origin[0, 0])
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        positions = xy @ rotation.T + origin[:2, 3] - read_recon_translation(source)[:2]
        surface_ok = True
        for position, heading in zip(positions, yaw):
            try:
                plane = surface.placement_plane(*position, *result['target']['size_m'][:2], angle+heading)
                surface_ok = surface_ok and plane['max_corner_residual_m'] <= policy.ROAD_FOOTPRINT_FLATNESS_M
            except ValueError:
                surface_ok = False
        checks['road_surface'] = bool(surface_ok)
    relative = xy - ego
    checks = {key:bool(value) for key, value in checks.items()}
    forward = relative[:, 0] * np.cos(ego_yaw) + relative[:, 1] * np.sin(ego_yaw)
    left = -relative[:, 0] * np.sin(ego_yaw) + relative[:, 1] * np.cos(ego_yaw)
    report = dict(status='passed' if all(checks.values()) else 'failed', checks=checks,
        candidate_id=result['selected_candidate_id'], target_track=target_id, construction_goal=construction_goal,
        frozen_scene_path=result['frozen_scene_path'],
        max_position_roundtrip_error_m=float(np.max(abs(recovered_xy - xy))), dense_reference=recomputed,
        native_replay_geometry={key:replay[key] for key in ('min_ego_clearance_m', 'first_contact_time_s',
            'other_contacts', 'checked_actor_pairs', 'missing_actor_pairs')},
        relative_to_ego_rear_by_frame=[dict(time_s=float(time), forward_m=float(f), left_m=float(l))
            for time, f, l in zip(times, forward, left)],
        scope='CSV and annotation recomputation using shared checkers; interpolation remains an assumption; no physical execution')
    atomic_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = review(args.run, args.output)
    print(json.dumps(dict(status=report['status'], checks=report['checks'])))


if __name__ == '__main__':
    main()
