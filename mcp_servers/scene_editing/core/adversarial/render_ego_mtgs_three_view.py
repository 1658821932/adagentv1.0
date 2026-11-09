"""Render corrected candidate and recorded ego poses with MTGS front/left/right cameras."""
from workflow_timing import timed_operation

import argparse
import json
import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from experiment_runtime import atomic_json, experiments_root, full_pipeline_root
from render_ego_mtgs import (ROOT, asset_from_first_ego, camera_intrinsics, edited_ego_matrix,
    instantiate_models, load_checkpoint, matrix_from_rotation_translation, render_frame)
from run_ego_foci_probe import read_track
from run_ego_reference import DEFAULT_TARGET_TRACK, scene_from_frames

CAMERAS = ('CAM_F0', 'CAM_L0', 'CAM_R0')


@timed_operation('render.ego_three_view', 'output')
def run(input_dir, parent_run, candidate_id, output, worldengine_dir, selected_frames=None,
        target_track=None, attack_run=None, attack_candidate=None, attack_track=None,
        original_scene=None, added_vehicle_run=None):
    if target_track is None:
        raise ValueError('target_track must identify a vehicle in the current scene')
    if attack_run is not None and not (attack_candidate and attack_track and original_scene):
        raise ValueError('attack_run requires attack_candidate, attack_track and original_scene')
    if attack_run is not None and added_vehicle_run is not None:
        raise ValueError('Choose either an edited vehicle or an added vehicle')
    input_dir = Path(input_dir).resolve(); parent_run = Path(parent_run).resolve(); output = Path(output).resolve()
    if not any(output.is_relative_to(base) for base in (experiments_root(), full_pipeline_root())):
        raise ValueError('Output must be under experiments/')
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing three-view render')
    constructor_track = parent_run/'reference_screen'/candidate_id/'trajectory.csv'
    agent_track = parent_run/'candidates'/candidate_id/'trajectory.csv'
    if constructor_track.is_file() == agent_track.is_file():
        raise ValueError('Require exactly one constructor or agent candidate layout')
    candidate_track = constructor_track if constructor_track.is_file() else agent_track
    checkpoints = list((input_dir/'source_asset/background').glob('*.ckpt'))
    if len(checkpoints) != 1:
        raise ValueError('Require exactly one MTGS checkpoint')
    checkpoint = checkpoints[0]
    frames = json.loads((input_dir/'metadata/frames.json').read_text())
    times, recorded_xy, recorded_yaw, objects = scene_from_frames(frames, target_track)
    candidate_xy, candidate_yaw = read_track(candidate_track, times)
    indices = list(range(len(frames))) if selected_frames is None else sorted(set(selected_frames))
    if not indices or indices != (selected_frames if selected_frames is not None else indices) \
            or any(i < 0 or i >= len(frames) for i in indices):
        raise ValueError('Frames must be unique, sorted and covered by the recording')
    asset, checkpoint_globals = load_checkpoint(checkpoint)
    recon_translation = np.asarray(asset['background']['config']['recon2world_translation'], float)
    asset_from_plan = asset_from_first_ego(frames[0], recon_translation)
    attack = None
    if attack_run is not None:
        import csv
        from render_attack_mtgs import TargetOverride, displacement_in_asset, grounded_displacement_in_asset
        attack_path = Path(attack_run).resolve()/'candidates'/attack_candidate/'trajectory.csv'
        attack_rows = list(csv.DictReader(attack_path.open()))
        if len(attack_rows) != len(frames) \
                or any(abs(float(r['time_s'])-t) > 1e-6 for r, t in zip(attack_rows, times)):
            raise ValueError('Attack candidate time grid mismatch')
        attack_xy = np.array([[float(r['target_x_m']), float(r['target_y_m'])] for r in attack_rows])
        attack_yaw = np.array([float(r['target_yaw_rad']) for r in attack_rows])
        original_frames = json.loads((Path(original_scene).resolve()/'metadata/frames.json').read_text())
        _, _, _, original_objects = scene_from_frames(original_frames, attack_track)
        recorded_attack_xy = np.array([[next(o for o in f if o['track'] == attack_track)[k]
                                        for k in ('x', 'y')] for f in original_objects])
        recorded_attack_yaw = np.array([next(o for o in f if o['track'] == attack_track)['yaw']
                                        for f in original_objects])
        attack = dict(key='rigid_object_'+attack_track, xy=attack_xy, yaw=attack_yaw,
                      recorded_xy=recorded_attack_xy, recorded_yaw=recorded_attack_yaw,
                      csv=str(attack_path), TargetOverride=TargetOverride,
                      displacement_in_asset=displacement_in_asset, grounded_displacement=grounded_displacement_in_asset)
    output.mkdir(parents=True)
    (output/'input').mkdir()
    shutil.copyfile(candidate_track, output/'input/candidate-trajectory.csv')
    if attack is not None:
        shutil.copyfile(attack['csv'], output/'input/attack-trajectory.csv')
    from road_height import RoadHeightMap, ego_height_reference
    road_map = RoadHeightMap.load(input_dir)
    height_ref = ego_height_reference(frames, road_map, recon_translation)
    config = dict(version='target_ego_mtgs_three_view_render.v1', input_dir=str(input_dir),
        parent_run=str(parent_run), candidate_id=candidate_id, target_track=target_track,
        checkpoint=str(checkpoint),
        checkpoint_globals=checkpoint_globals, worldengine_dir=str(worldengine_dir), cameras=list(CAMERAS),
        frame_indices=indices, sample_count=len(frames), fps=2.,
        camera_construction='Corrected first-ego transform; candidate planar x/y/yaw delta applied to recorded full ego pose, then cam_to_ego',
        z_pitch_roll_policy=('Ego origin z re-anchored to the scene road_height_map plus the frame-0 '
            'recorded height reference (UTM z drifts from the reconstructed road); recorded '
            'pitch/roll retained'),
        camera_height_reference_m=height_ref['reference_m'],
        camera_offset_profile_m=height_ref['profile_m'],
        image_size_policy='Read width and height from each original JPEG',
        actor_policy=('Added PLY follows frozen reference; others use native log replay' if added_vehicle_run is not None else
            'Unmodified checkpoint log replay' if attack is None else
            'Canonical target Gaussian placed at absolute candidate x/y/yaw and road height; source roll/pitch and appearance clamped to supported log times; other actors use native replay'),
        attack=(None if attack is None else dict(run=str(Path(attack_run).resolve()),
            candidate=attack_candidate, track=attack_track, original_scene=str(Path(original_scene).resolve()))),
        added_vehicle_run=None if added_vehicle_run is None else str(Path(added_vehicle_run).resolve()),
        physical_execution='not_run')
    atomic_json(output/'config.json', config)
    models = instantiate_models(asset, worldengine_dir)
    if added_vehicle_run is not None:
        import math
        added_run = Path(added_vehicle_run).resolve()
        info = json.loads((added_run/'summary.json').read_text())
        if (added_run/'frozen_scene').resolve() != input_dir or info['track_id'] != target_track:
            raise ValueError('Added vehicle must match the frozen scene and target track')
        anchor = next(obj for obj in objects[0] if obj['track'] == target_track)
        from render_added_vehicle_mtgs import added_vehicle_model
        initial_yaw = math.atan2(frames[0]['ego2global'][1][0], frames[0]['ego2global'][0][0])
        models['added_vehicle'], added_placement = added_vehicle_model(
            info, anchor, asset_from_plan, road_map, initial_yaw + anchor['yaw'])
        config['added_vehicle_placement'] = added_placement
        atomic_json(output/'config.json', config)
    if attack is not None:
        from render_asset_preflight import assess_replay
        source_model = models[attack['key']]
        static = source_model.static_in_log
        source_positions = source_model.log_trans.detach().cpu().numpy()
        source_times = (None if static else source_model.log_timestamps.detach().cpu().numpy()
                        + int(source_model.log_start_time))
        annotated = [(asset_from_plan @ np.array([*attack['recorded_xy'][i],0.,1.]))[:2] for i in indices]
        preflight = assess_replay(indices,[int(frames[i]['timestamp']) for i in indices],annotated,
                                  source_times,source_positions)
        atomic_json(output/'attack-asset-preflight.json',preflight)
        # A failed source replay must not drive the edited target. Use an explicit
        # absolute-pose renderer instead, with its own strict delivered-pose gate.
        preflight['role'] = 'source_replay_diagnostic_only'
        preflight['edited_pose_mode'] = 'absolute_annotation_pose.v1'
        atomic_json(output/'attack-asset-preflight.json',preflight)
    writers = {}
    for camera_id in CAMERAS:
        for folder in ('frames', 'recorded', 'compare'):
            (output/folder/camera_id).mkdir(parents=True)
        writers[camera_id] = {
            'candidate':imageio.get_writer(output/f'new-camera-{camera_id}.mp4', fps=2.,
                codec='libx264', macro_block_size=2),
            'recorded':imageio.get_writer(output/f'recorded-camera-{camera_id}.mp4', fps=2.,
                codec='libx264', macro_block_size=2),
        }
    per_frame = []; max_camera_readback = 0.
    try:
        for i in indices:
            frame = frames[i]
            if added_vehicle_run is not None:
                current = next(obj for obj in objects[i] if obj['track'] == target_track)
                models['added_vehicle'], added_placement = added_vehicle_model(
                    info, current, asset_from_plan, road_map, initial_yaw + current['yaw'])
            recorded_ego = np.asarray(frame['ego2global'], float)
            recorded_ego[:3, 3] -= recon_translation
            recorded_ground = road_map.height_with_local_plane(recorded_ego[0,3],recorded_ego[1,3])
            recorded_ego[2,3] = recorded_ground['height'] + height_ref['reference_m']
            new_ego = edited_ego_matrix(recorded_ego, recorded_xy[i], recorded_yaw[i],
                candidate_xy[i], candidate_yaw[i], asset_from_plan)
            candidate_asset_xy = (asset_from_plan @ np.array([*candidate_xy[i], 0., 1.]))[:2]
            candidate_ground = road_map.height_with_local_plane(*candidate_asset_xy)
            new_ego[2,3] = candidate_ground['height'] + height_ref['reference_m']
            recovered = np.linalg.inv(asset_from_plan) @ new_ego
            recovered_yaw = np.arctan2(recovered[1, 0], recovered[0, 0])
            if np.linalg.norm(recovered[:2, 3]-candidate_xy[i]) > 1e-8 \
                    or abs(recovered_yaw-candidate_yaw[i]) > 1e-8:
                raise ValueError(f'Edited ego recovery mismatch at frame {i}')
            row = dict(frame_index=i, time_s=float(times[i]), timestamp=int(frame['timestamp']),
                candidate_ego_xy=candidate_xy[i].tolist(), candidate_ego_yaw=float(candidate_yaw[i]),
                recovered_ego_xy=recovered[:2, 3].tolist(), recovered_ego_yaw=float(recovered_yaw),
                road_z_at_candidate_m=candidate_ground['height'],
                candidate_camera_ground=candidate_ground, recorded_camera_ground=recorded_ground,
                cameras={})
            if added_vehicle_run is not None:
                row['added_vehicle'] = dict(track=target_track, xy=[current['x'],current['y']],
                    yaw=current['yaw'], vehicle_placement=added_placement)
            if attack is not None:
                key = attack['key']
                if key not in models:
                    raise ValueError(f'Attack target {attack_track} missing from MTGS asset')
                base_model = models[key]
                from render_actor_pose import AbsoluteTargetOverride, grounded_absolute_pose
                target_index = list(map(str, frame['anns']['track_tokens'])).index(attack_track)
                box_height = float(frame['anns']['gt_boxes'][target_index][5])
                translation, heading, ground = grounded_absolute_pose(attack['xy'][i],attack['yaw'][i],
                    asset_from_plan,road_map,box_height)
                models[key] = AbsoluteTargetOverride(base_model,translation,heading,int(frame['timestamp']))
                diagnostic = next(item for item in preflight['frames'] if item['frame_index']==i)
                row['attack_override'] = dict(track=attack_track,target_ground_height=ground,
                    pose_mode='absolute_annotation_pose.v1', candidate_asset_position_m=translation.tolist(),
                    candidate_asset_yaw_rad=heading, source_replay_diagnostic=diagnostic)
            from render_actor_pose import covered_models
            frame_models, hidden = covered_models(models,int(frame['timestamp']))
            row['actors_outside_source_coverage'] = hidden
            for camera_id in CAMERAS:
                camera_info = frame['cams'][camera_id]
                sensor_to_lidar = matrix_from_rotation_translation(camera_info['sensor2lidar_rotation'],
                    camera_info['sensor2lidar_translation'])
                cam_to_ego = np.asarray(frame['lidar2ego'], float) @ sensor_to_lidar
                recorded_camera = recorded_ego @ cam_to_ego
                candidate_camera = new_ego @ cam_to_ego
                actual_path = input_dir/'original_images'/camera_info['data_path']
                if not actual_path.is_file():
                    raise ValueError(f'Missing original image: {actual_path}')
                with Image.open(actual_path) as actual_image:
                    width, height = actual_image.size
                    actual = np.asarray(actual_image.convert('RGB'))
                intrinsic, inverse_maps, width, height = camera_intrinsics(camera_info, width, height)
                timestamp = int(frame['timestamp'])
                original, original_error = render_frame(frame_models, recorded_camera, intrinsic, inverse_maps,
                    width, height, timestamp)
                edited, edited_error = render_frame(frame_models, candidate_camera, intrinsic, inverse_maps,
                    width, height, timestamp)
                max_camera_readback = max(max_camera_readback, original_error, edited_error)
                if max_camera_readback > 2e-4:
                    raise ValueError('Camera readback gate failed')
                Image.fromarray(original).save(output/'recorded'/camera_id/f'frame{i:03d}.png')
                Image.fromarray(edited).save(output/'frames'/camera_id/f'frame{i:03d}.png')
                Image.fromarray(np.concatenate([original, edited], axis=1)).save(
                    output/'compare'/camera_id/f'frame{i:03d}.png')
                writers[camera_id]['recorded'].append_data(original)
                writers[camera_id]['candidate'].append_data(edited)
                mae = float(np.abs(actual.astype(float)-original.astype(float)).mean()/255.)
                row['cameras'][camera_id] = dict(candidate_image=f'frames/{camera_id}/frame{i:03d}.png',
                    recorded_image=f'recorded/{camera_id}/frame{i:03d}.png',
                    image_size=[width, height], recorded_render_mae=mae)
            if attack is not None:
                row['attack_override']['edited_pose_readback'] = models[key].readback
                if models[key].readback is None:
                    raise ValueError('Target pose was not evaluated by the renderer')
                models[key] = base_model
            per_frame.append(row)
            print(json.dumps(row), flush=True)
    finally:
        for pair in writers.values():
            pair['candidate'].close(); pair['recorded'].close()
    decoded = {}
    for camera_id in CAMERAS:
        for prefix in ('new-camera', 'recorded-camera'):
            name = f'{prefix}-{camera_id}.mp4'
            reader = imageio.get_reader(output/name); metadata = reader.get_meta_data()
            count = sum(1 for _ in reader); reader.close()
            if count != len(indices) or abs(float(metadata['fps'])-2.) > 1e-6:
                raise ValueError(f'Video decode/fps mismatch: {name}')
            decoded[name] = count
    full_window = indices == list(range(len(frames)))
    report = dict(status='passed', full_window=full_window,
        renderer='WorldEngine MTGS portable models and gsplat', checkpoint=str(checkpoint),
        candidate_id=candidate_id, cameras=list(CAMERAS), rendered_frames=len(indices),
        rendered_frame_ids=indices, decoded_videos=decoded, fps=2.,
        camera_readback_max_abs_error=max_camera_readback,
        recovered_ego_max_error_policy='Every edited ego must recover to the candidate x/y/yaw within 1e-8 before rendering',
        actors=config['actor_policy'],
        attack_pose_mode=None if attack is None else 'absolute_annotation_pose.v1',
        attack_source_replay_diagnostic=None if attack is None else str(output/'attack-asset-preflight.json'),
        attack_pose_readback_max_error_m=None if attack is None else max(
            row['attack_override']['edited_pose_readback']['position_error_m'] for row in per_frame),
        physical_avoidance='undetermined', carla_executed=False,
        quality_evaluation='Recorded-camera MAE is diagnostic only; no aligned ground-truth score is claimed for edited views',
        per_frame=per_frame)
    atomic_json(output/'render-report.json', report)
    atomic_json(output/'artifact-check.json', dict(status='passed', full_window=full_window,
        rendered_frames=len(indices), cameras=list(CAMERAS), decoded_videos=decoded,
        camera_readback_max_abs_error=max_camera_readback, candidate_id=candidate_id,
        physical_execution='not_run'))
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--parent-run', type=Path, required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worldengine', type=Path,
        default=ROOT.parent/'ghc/WorldEngine/projects/SimEngine')
    parser.add_argument('--frames', type=int, nargs='+', default=None)
    parser.add_argument('--target-track', required=True,
        help='Interaction target track id in the current scene')
    parser.add_argument('--attack-run', type=Path, default=None,
        help='Optional attack run; its selected vehicle is rendered with the attack trajectory')
    parser.add_argument('--attack-candidate', default=None)
    parser.add_argument('--attack-track', default=None)
    parser.add_argument('--original-scene', type=Path, default=None,
        help='Original (pre-attack) scene dir; needed to compute the recorded target baseline')
    parser.add_argument('--added-vehicle-run', type=Path, default=None,
        help='Candidate directory containing the frozen added vehicle trajectory and its asset')
    args = parser.parse_args()
    report = run(args.input, args.parent_run, args.candidate, args.output, args.worldengine, args.frames,
        target_track=args.target_track, attack_run=args.attack_run,
        attack_candidate=args.attack_candidate, attack_track=args.attack_track,
        original_scene=args.original_scene, added_vehicle_run=args.added_vehicle_run)
    print('SUMMARY '+json.dumps({key:value for key, value in report.items() if key != 'per_frame'}), flush=True)


if __name__ == '__main__':
    main()
