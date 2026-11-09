"""Add a stationary or moving reference vehicle to a nuPlan/OpenScene scene copy.

The new track uses supplied per-frame poses, or the recorded ego's planar pose
at an explicitly chosen anchor frame. No physical execution or rendering is
implied by this operation. Time is the scene's original native timestamp grid.
"""
from workflow_timing import timed_operation

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import acceptance_policy as policy
import numpy as np

from experiment_runtime import ROOT, atomic_json, experiments_root
from run_ego_reference import scene_from_frames, evaluate, polygon
from ego_road_checks import check_drivable_area_official, load_official_drivable


@timed_operation('attack.materialize_added', 'output')
def add_vehicle(scene_dir, output, anchor_frame, track_id, size, center_z_lidar=None,
                           task_path=None, recon_translation=None, trajectory_xy=None, trajectory_yaw=None, official_min_margin_m=policy.OFFICIAL_MARGIN_M):
    scene_dir = Path(scene_dir).resolve()
    output = Path(output).resolve()
    if experiments_root() not in output.parents or output.exists():
        raise ValueError('Output must be a new directory under experiments')
    task = None
    if task_path is not None:
        task = json.loads(Path(task_path).read_text())
        if task.get('version') != 'attack_task.v1' or task.get('operation') != 'add_vehicle':
            raise ValueError('Task is not a prepared add_vehicle instruction')
        if Path(task['scene']).resolve() != scene_dir:
            raise ValueError('Task scene differs from the requested scene')
    frames = json.loads((scene_dir/'metadata/frames.json').read_text())
    if len(frames) < 3 or not 0 <= anchor_frame < len(frames):
        raise ValueError('Anchor frame must refer to a recorded scene frame')
    if not isinstance(track_id, str) or not track_id.strip():
        raise ValueError('New track ID must be a non-empty string')
    size = np.asarray(size, dtype=float)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError('Vehicle size must contain positive finite length, width, height')
    if center_z_lidar is not None and not math.isfinite(center_z_lidar):
        raise ValueError('Lidar-frame box-center height must be finite')
    required = {'gt_boxes', 'gt_names', 'gt_velocity_3d', 'instance_tokens', 'track_tokens'}
    for i, frame in enumerate(frames):
        anns = frame['anns']
        if set(anns) != required or len({len(anns[k]) for k in required}) != 1:
            raise ValueError(f'Frame {i}: unsupported or inconsistent annotation fields')
        if track_id in map(str, anns['track_tokens']) or track_id in map(str, anns['instance_tokens']):
            raise ValueError('New vehicle ID already exists')
    ego_world = np.asarray([frame['ego2global'] for frame in frames], dtype=float)
    global_yaw = np.unwrap(np.arctan2(ego_world[:, 1, 0], ego_world[:, 0, 0]))
    yaw0 = float(global_yaw[0]); origin = ego_world[0, :2, 3]
    rotation = np.array([[math.cos(yaw0), -math.sin(yaw0)], [math.sin(yaw0), math.cos(yaw0)]])
    ego_xy = (ego_world[:, :2, 3]-origin) @ rotation
    ego_yaw = global_yaw-yaw0
    # ego poses describe the rear axle; place the new vehicle center on its future path.
    target_xy = ego_xy[anchor_frame].copy()
    target_yaw = float(ego_yaw[anchor_frame])
    moving = trajectory_xy is not None or trajectory_yaw is not None
    if moving:
        positions = np.asarray(trajectory_xy, dtype=float)
        headings = np.asarray(trajectory_yaw, dtype=float)
        if positions.shape != (len(frames), 2) or headings.shape != (len(frames),) or not np.isfinite(positions).all() or not np.isfinite(headings).all():
            raise ValueError('Dynamic vehicle requires finite xy/yaw on every observed frame')
        if center_z_lidar is not None:
            raise ValueError('Dynamic vehicle requires road-derived height')
        target_xy = positions[anchor_frame]
        target_yaw = float(headings[anchor_frame])
    anchor_transform = ego_world[anchor_frame] @ np.asarray(frames[anchor_frame]['lidar2ego'], dtype=float)
    z_info = {'z_source': 'explicit_lidar_assumption' if center_z_lidar is not None
              else 'road_height_map'}
    if center_z_lidar is not None:
        # Explicit legacy override: height in the anchor-frame lidar coordinates.
        anchor_lidar = np.asarray(frames[anchor_frame]['lidar2ego'], dtype=float)
        anchor_xy_lidar = np.linalg.solve(anchor_lidar[:2, :2], -anchor_lidar[:2, 3])
        anchor_ground_point = anchor_transform @ np.r_[anchor_xy_lidar, center_z_lidar, 1.]
        target_global_z = float(anchor_ground_point[2])
    else:
        # Code-computed height: put the box bottom on the reconstruction's own
        # road surface (the visual ground truth), not on the drifting UTM z.
        from road_height import RoadHeightMap, read_recon_translation
        road_map = RoadHeightMap.load(scene_dir)
        recon = np.asarray(recon_translation, float) if recon_translation is not None \
            else read_recon_translation(scene_dir)
        anchor_asset_xy = target_xy @ rotation.T + origin - recon[:2]
        plane = road_map.placement_plane(*anchor_asset_xy, size[0], size[1],
                                         float(yaw0 + target_yaw))
        target_global_z = float(recon[2]+plane['z_center']+size[2]/2)
        z_info.update(road_z_asset_m=plane['z_center'],
                      bottom_clearance_m=0.0,
                      pitch_rad=plane['pitch'], roll_rad=plane['roll'],
                      footprint_flatness_m=plane['max_corner_residual_m'],
                      anchor_center_z_lidar_m=float(target_global_z-ego_world[anchor_frame][2, 3]),
                      utm_reference_z_m=float(ego_world[anchor_frame][2, 3]),
                      utm_vs_road_drift_m=float(ego_world[anchor_frame][2, 3]
                                                -(recon[2]+plane['z_center'])))
    if not moving:
        positions = np.repeat(target_xy[None, :], len(frames), axis=0)
        headings = np.full(len(frames), target_yaw)
    heights = np.full(len(frames), target_global_z)
    if moving:
        global_positions = positions @ rotation.T + origin
        planes = [road_map.placement_plane(*xy, size[0], size[1], float(yaw0+h))
                  for xy, h in zip(global_positions-recon[:2], headings)]
        if any(plane['max_corner_residual_m'] > policy.ROAD_FOOTPRINT_FLATNESS_M for plane in planes):
            raise ValueError('Dynamic vehicle crosses a non-planar road footprint')
        heights = np.array([recon[2]+plane['z_center']+size[2]/2 for plane in planes])
        target_global_z = float(heights[0])
        z_info = dict(z_source='road_height_map', bottom_clearance_m=0.,
                      road_z_asset_by_frame_m=[plane['z_center'] for plane in planes],
                      max_footprint_flatness_m=max(plane['max_corner_residual_m'] for plane in planes))
    copied = json.loads(json.dumps(frames))
    times = (np.asarray([f['timestamp'] for f in frames], dtype=float)-frames[0]['timestamp'])/1e6
    if np.any(np.diff(times) <= 0):
        raise ValueError('Scene timestamps must be strictly increasing')
    velocity = np.gradient(positions, times, axis=0) @ rotation.T
    for i, frame in enumerate(copied):
        transform = np.asarray(frame['lidar2ego'], dtype=float)
        c, s = math.cos(ego_yaw[i]), math.sin(ego_yaw[i])
        xy_ego = np.array([[c, s], [-s, c]]) @ (positions[i]-ego_xy[i])
        xy_lidar = np.linalg.solve(transform[:2, :2], xy_ego-transform[:2, 3])
        lidar_world = ego_world[i] @ transform
        if abs(lidar_world[2, 2]) < 1e-8:
            raise ValueError('Cannot recover lidar height from world height')
        box_z = (heights[i]-lidar_world[2, 3]-lidar_world[2, 0]*xy_lidar[0]
                 -lidar_world[2, 1]*xy_lidar[1])/lidar_world[2, 2]
        lidar_yaw = math.atan2(transform[1, 0], transform[0, 0])
        box = [float(xy_lidar[0]), float(xy_lidar[1]), float(box_z),
               *size.tolist(), float(headings[i]-ego_yaw[i]-lidar_yaw)]
        anns = frame['anns']
        for key, value in [('gt_boxes', box), ('gt_names', 'vehicle'),
                           ('gt_velocity_3d', [*(velocity[i] @ np.array([[math.cos(global_yaw[i]), -math.sin(global_yaw[i])], [math.sin(global_yaw[i]), math.cos(global_yaw[i])]])).tolist(), 0.]), ('instance_tokens', track_id),
                           ('track_tokens', track_id)]:
            anns[key].append(value)
    times2, recorded_ego, recorded_yaw, actors = scene_from_frames(copied, track_id)
    if not np.allclose(times, times2, rtol=0, atol=1e-9):
        raise ValueError('Timestamp grid changed')
    recovered_positions = np.array([[next(a for a in group if a['track'] == track_id)[key]
                           for key in ('x', 'y')] for group in actors])
    recovered_headings = np.array([next(a for a in group if a['track'] == track_id)['yaw'] for group in actors])
    if not np.allclose(recovered_positions, positions, rtol=0, atol=1e-9) or not np.allclose(recovered_headings, headings, rtol=0, atol=1e-9):
        raise ValueError('New vehicle annotation roundtrip failed')
    metrics = evaluate(times2, recorded_ego, recorded_yaw, actors, times2, track_id)
    other_overlaps = []
    for i, group in enumerate(actors):
        target = polygon(*positions[i], headings[i], size[0], size[1])
        for actor in group:
            if actor['track'] != track_id and target.intersection(
                    polygon(actor['x'], actor['y'], actor['yaw'], actor['length'], actor['width'])).area > 1e-9:
                other_overlaps.append({'frame':i, 'track':actor['track']})
    official_path = scene_dir/'metadata/map_drivable_official.json'
    road_result = None
    if official_path.is_file():
        road_result = check_drivable_area_official(times2, positions, headings,
            load_official_drivable(official_path), front=size[0]/2, rear=size[0]/2,
            width=size[1], min_margin_m=official_min_margin_m)
    output.mkdir(parents=True)
    scene_out = output/'frozen_scene'; metadata = scene_out/'metadata'; metadata.mkdir(parents=True)
    (metadata/'frames.json').write_text(json.dumps(copied))
    for name in ('cameras.json', 'map_features.json', 'map_reference.json', 'map_drivable_official.json', 'no_cross_lines.json', 'image_manifest.csv'):
        path = scene_dir/'metadata'/name
        if path.is_file():
            shutil.copyfile(path, metadata/name)
    for name in ('source_asset', 'original_images'):
        path = scene_dir/name
        if path.exists():
            (scene_out/name).symlink_to(path, target_is_directory=True)
    folder = output/'candidates/candidate_001'; folder.mkdir(parents=True)
    with (folder/'trajectory.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['frame_index', 'time_s', 'target_x_m', 'target_y_m', 'target_yaw_rad', 'target_length_m', 'target_width_m', 'height_m'])
        for i, t in enumerate(times):
            writer.writerow([i, t, *positions[i], headings[i], *size])
    summary = dict(version='added_vehicle_reference.v1', status='constructed_unvalidated',
        source_scene=str(scene_dir), task_path=str(Path(task_path).resolve()) if task_path else None,
        instruction=task['instruction'] if task else None, asset_path=task['asset_path'] if task else None,
        track_id=track_id, anchor_frame=anchor_frame,
        size_m=size.tolist(), anchor_center_z_lidar_m=center_z_lidar, target_global_z_m=target_global_z,
        height_placement=z_info,
        trajectory='moving reference track' if moving else 'stationary in plan-local frame with constant global height',
        candidate_id='candidate_001', min_ego_clearance_m=metrics['min_target_clearance_m'],
        ego_overlap_indices=metrics['target_overlap_indices'], other_observed_overlaps=other_overlaps,
        observed_frames=len(times), official_road=road_result, asset_alignment='not_checked',
        render='not_run', physical_execution='not_run',
        note=('Synthetic reference track. Height computed from the scene road_height_map; '
              'box bottom sits on the reconstructed road surface.'
              if center_z_lidar is None else
              'Synthetic reference track. Anchor-frame height is an explicit assumption, '
              'not a verified road surface.'))
    atomic_json(output/'summary.json', summary)
    atomic_json(scene_out/'freeze-info.json', summary)
    return summary




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--anchor-frame', type=int, required=True)
    parser.add_argument('--track-id', required=True)
    parser.add_argument('--size', type=float, nargs=3, required=True, metavar=('L', 'W', 'H'))
    parser.add_argument('--center-z-lidar', type=float, default=None,
                        help='Legacy explicit lidar-frame box-center height. Omit to compute '
                             'the height from the scene road_height_map (default).')
    parser.add_argument('--task', type=Path, help='Optional prepared add_vehicle task JSON')
    args = parser.parse_args()
    result = add_vehicle(args.scene, args.output, args.anchor_frame, args.track_id,
                                    args.size, args.center_z_lidar, args.task)
    print(json.dumps({k:result[k] for k in ('status', 'track_id', 'min_ego_clearance_m', 'ego_overlap_indices')}))


if __name__ == '__main__':
    main()
