"""Freeze an attack candidate into a modified scene pack for downstream ego defense.

Reads the original nuPlan/OpenScene scene and one attack candidate CSV (plan-local
frame: origin = first-frame ego rear axle, x = first-frame ego forward, as produced
by run_attack_reference.py / run_attack_agent.py), then writes a new scene
pack whose metadata/frames.json has the target vehicle's gt_boxes x/y and heading
replaced by the edited trajectory. Everything else (other actors, timestamps,
cameras, sizes, z) is preserved; gt_velocity_3d of the target is recomputed by
global-frame finite differences of the edited center.

Coordinate conventions (must match scripts/run_ego_reference.py scene_from_frames):
- plan xy -> global xy:  rear0 + R(yaw0) @ plan
- per-frame box xyz is in the LIDAR frame: box = inv(lidar2ego_i) @ inv(ego2global_i) @ global
- box[6] (heading) satisfies: plan_yaw = ego_yaw_plan_i + lidar2ego_yaw + box[6]

The output pack symlinks source_asset and original_images (read-only reuse) and copies
the remaining metadata files. A roundtrip check re-reads the frozen frames.json with
scene_from_frames and asserts the target track matches the candidate CSV.
"""
from workflow_timing import timed_operation

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np

from experiment_runtime import atomic_json
from attack_timing import contact_after_observation
from run_ego_reference import EGO_FRONT, EGO_REAR, EGO_WIDTH, polygon, scene_from_frames


def _read_candidate_csv(path, times):
    import csv
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(times):
        raise ValueError('Candidate row count differs from scene frame count')
    if any(abs(float(row['time_s']) - t) > 1e-6 for row, t in zip(rows, times)):
        raise ValueError('Candidate time grid differs from the recorded scene')
    columns = rows[0].keys()
    x_key, y_key, yaw_key = 'target_x_m', 'target_y_m', 'target_yaw_rad'
    if not {x_key, y_key, yaw_key}.issubset(columns):
        raise ValueError(f'Cannot locate target x/y/yaw columns in {sorted(columns)}')
    xy = np.array([[float(r[x_key]), float(r[y_key])] for r in rows], dtype=float)
    yaw = np.array([float(r[yaw_key]) for r in rows], dtype=float)
    if not np.isfinite(xy).all() or not np.isfinite(yaw).all():
        raise ValueError('Candidate contains non-finite values')
    return xy, yaw, (x_key, y_key, yaw_key)


@timed_operation('attack.freeze', 'output')
def freeze(scene_dir, attack_run, candidate_id, target_track, output):
    scene_dir = Path(scene_dir).resolve(); attack_run = Path(attack_run).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    frames = json.loads((scene_dir / 'metadata/frames.json').read_text())
    times, ego_xy, ego_yaw, objects = scene_from_frames(frames, target_track)
    csv_path = attack_run / 'candidates' / candidate_id / 'trajectory.csv'
    if not csv_path.is_file():
        raise ValueError(f'Missing attack candidate CSV: {csv_path}')
    plan_xy, plan_yaw, columns = _read_candidate_csv(csv_path, times)
    early_overlaps = []
    for i, frame in enumerate(objects):
        target = next(actor for actor in frame if actor['track'] == target_track)
        shift = (EGO_FRONT - EGO_REAR) / 2
        center = ego_xy[i] + shift * np.array([np.cos(ego_yaw[i]), np.sin(ego_yaw[i])])
        ego = polygon(*center, ego_yaw[i], EGO_FRONT + EGO_REAR, EGO_WIDTH)
        attack = polygon(*plan_xy[i], plan_yaw[i], target['length'], target['width'])
        if ego.intersection(attack).area > 1e-9:
            early_overlaps.append(i)
    if not contact_after_observation(times, early_overlaps):
        raise ValueError('Cannot freeze an attack with sampled ego contact at or before 3 seconds')

    ego_world = np.array([f['ego2global'] for f in frames], dtype=float)
    global_yaw = np.unwrap(np.arctan2(ego_world[:, 1, 0], ego_world[:, 0, 0]))
    yaw0 = float(global_yaw[0])
    rear0 = ego_world[0, :2, 3].copy()
    rot0 = np.array([[math.cos(yaw0), -math.sin(yaw0)], [math.sin(yaw0), math.cos(yaw0)]])
    global_xy = rear0 + plan_xy @ rot0.T

    # velocity: global-frame central differences, then ROTATED into each frame's ego
    # frame to match the dataset's gt_velocity_3d convention (rotation only, no
    # Galilean transform: a parked car keeps exactly zero velocity)
    velocity = np.zeros((len(times), 3))
    dt = np.diff(times)
    velocity[0, :2] = (global_xy[1] - global_xy[0]) / dt[0]
    velocity[-1, :2] = (global_xy[-1] - global_xy[-2]) / dt[-1]
    if len(times) > 2:
        velocity[1:-1, :2] = (global_xy[2:] - global_xy[:-2]) / (times[2:] - times[:-2])[:, None]
    for i in range(len(times)):
        c, s = math.cos(global_yaw[i]), math.sin(global_yaw[i])
        velocity[i, :2] = np.array([[c, s], [-s, c]]) @ velocity[i, :2]

    # plan-frame ego path, exactly as scene_from_frames computes it (row-vector convention)
    ego_xy_plan = (ego_world[:, :2, 3] - rear0) @ rot0

    frozen = json.loads(json.dumps(frames))  # deep copy via json round trip
    replaced = 0
    for i, frame in enumerate(frozen):
        lidar_to_ego = np.array(frame['lidar2ego'], dtype=float)
        lidar_yaw_offset = math.atan2(lidar_to_ego[1, 0], lidar_to_ego[0, 0])
        ego_yaw_plan = float(global_yaw[i] - yaw0)
        a = frame['anns']
        hits = [j for j, token in enumerate(a['track_tokens']) if str(token) == target_track]
        if len(hits) != 1:
            raise ValueError(f'Frame {i}: expected exactly one box for {target_track}')
        j = hits[0]
        box = np.asarray(a['gt_boxes'][j], dtype=float)
        # inverse of the scene_from_frames READ convention: object plan xy =
        # ego_xy_plan_i + R(ego_yaw_plan_i) @ (lidar2ego @ box)[:2]; ego pitch/roll never
        # enters the object read path, so the write must stay in the yaw-projected frame
        cs, sn = math.cos(ego_yaw_plan), math.sin(ego_yaw_plan)
        obj_ego_2d = np.array([[cs, sn], [-sn, cs]]) @ (plan_xy[i] - ego_xy_plan[i])
        lidar_rot = lidar_to_ego[:2, :2]
        box_2d = np.linalg.solve(lidar_rot, obj_ego_2d - lidar_to_ego[:2, 3])
        new_box = box.copy()
        new_box[0], new_box[1] = float(box_2d[0]), float(box_2d[1])
        new_box[6] = float(plan_yaw[i] - ego_yaw_plan - lidar_yaw_offset)
        a['gt_boxes'][j] = [float(v) for v in new_box]
        a['gt_velocity_3d'][j] = [float(v) for v in velocity[i]]
        replaced += 1

    output.mkdir(parents=True)
    metadata = output / 'metadata'
    metadata.mkdir()
    (metadata / 'frames.json').write_text(json.dumps(frozen))
    copied = []
    for name in ('cameras.json', 'map_features.json', 'map_reference.json',
                 'map_drivable_official.json', 'no_cross_lines.json', 'image_manifest.csv'):
        src = scene_dir / 'metadata' / name
        if src.is_file():
            shutil.copyfile(src, metadata / name)
            copied.append(name)
    linked = []
    for name in ('source_asset', 'original_images'):
        src = scene_dir / name
        if src.exists():
            (output / name).symlink_to(src, target_is_directory=True)
            linked.append(name)

    # roundtrip check: re-read the frozen pack and compare with the candidate CSV
    times2, _, _, objects = scene_from_frames(json.loads((metadata / 'frames.json').read_text()), target_track)
    rt_xy = np.array([[next(o for o in frame if o['track'] == target_track)[k]
                       for k in ('x', 'y')] for frame in objects])
    rt_yaw = np.array([next(o for o in frame if o['track'] == target_track)['yaw'] for frame in objects])
    err_xy = float(np.abs(rt_xy - plan_xy).max())
    err_yaw = float(np.abs(np.unwrap(rt_yaw) - np.unwrap(plan_yaw)).max())
    if err_xy > 1e-9 or err_yaw > 1e-9:
        raise ValueError(f'Roundtrip mismatch: xy {err_xy:.3e}, yaw {err_yaw:.3e}')

    atomic_json(output / 'freeze-info.json', dict(
        version='freeze_attack_scene.v1', source_scene=str(scene_dir), attack_run=str(attack_run),
        candidate_id=candidate_id, candidate_csv=str(csv_path), candidate_columns=columns,
        target_track=target_track, frames_replaced=replaced,
        edits='target gt_boxes x/y + heading replaced in the yaw-projected ego/lidar frame (inverse of '
              'the scene_from_frames read convention); recorded box z kept, so global height rides with '
              'ego pitch/roll like any recorded box; gt_velocity_3d replaced by central differences '
              'of the edited center rotated into each frame ego frame (dataset convention: rotation '
              'only, no Galilean term)',
        preserved='timestamps, cameras, all other actors, box z/dimensions, ego2global, map files',
        metadata_copied=copied, directory_symlinks=linked,
        roundtrip_max_abs_error_xy_m=err_xy, roundtrip_max_abs_error_yaw_rad=err_yaw,
        note='Counterfactual frozen attack scene for offline defense editing; other actors stay recorded '
             'and do not react; interpolated motion between 2 Hz samples is an assumption, not observation'))
    print(json.dumps(dict(output=str(output), replaced=replaced, roundtrip_xy=err_xy,
                          roundtrip_yaw=err_yaw), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--scene', type=Path, required=True, help='original scene directory (read-only)')
    parser.add_argument('--attack-run', type=Path, required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--target-track', required=True)
    parser.add_argument('--output', type=Path, required=True, help='new frozen scene pack directory')
    args = parser.parse_args()
    freeze(args.scene, args.attack_run, args.candidate, args.target_track, args.output)


if __name__ == '__main__':
    main()
