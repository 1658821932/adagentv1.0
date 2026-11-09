"""Render a PLY Gaussian vehicle along its frozen nuPlan reference trajectory.

The added track is a reference hypothesis in the frozen scene. PLY local +X is
assumed to be the vehicle heading, +Z up; its center is its bounding-box center.
"""
from workflow_timing import timed_operation

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData
import torch

from experiment_runtime import ROOT, atomic_json
from render_ego_mtgs import (asset_from_first_ego, camera_intrinsics, instantiate_models,
    load_checkpoint, matrix_from_rotation_translation, render_frame)
from run_ego_reference import scene_from_frames

CAMERAS = ('CAM_L0', 'CAM_F0', 'CAM_R0')


def read_ply(path):
    vertices = PlyData.read(str(path))['vertex']
    names = set(vertices.data.dtype.names)
    fields = (['x', 'y', 'z'] + [f'f_dc_{i}' for i in range(3)] +
              [f'f_rest_{i}' for i in range(45)] + ['opacity'] +
              [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)])
    if not set(fields) <= names or len(vertices) == 0:
        raise ValueError('PLY must contain non-empty degree-3 Gaussian attributes')
    values = np.column_stack([vertices[name] for name in fields]).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('PLY contains non-finite Gaussian parameters')
    means = values[:, :3]
    center = (means.min(axis=0)+means.max(axis=0))/2
    quats = values[:, 55:59]
    if np.any(np.linalg.norm(quats, axis=1) < 1e-8):
        raise ValueError('PLY contains invalid Gaussian rotations')
    return dict(means=means-center, center=center, dimensions=means.max(axis=0)-means.min(axis=0),
                dc=values[:, 3:6], rest=values[:, 6:51].reshape(-1, 3, 15).transpose(0, 2, 1),
                opacity=values[:, 51], scales=values[:, 52:55], quats=quats)


class AddedVehicle:
    def __init__(self, data, center_asset, rotation):
        from worldengine.render.mtgs.utils.gaussian_utils import quat_mult
        self.xyz = torch.as_tensor(data['means'], device='cuda')
        self.dc = torch.as_tensor(data['dc'], device='cuda')
        self.rest = torch.as_tensor(data['rest'].copy(), device='cuda')
        self.scales = torch.exp(torch.as_tensor(data['scales'], device='cuda'))
        self.opacities = torch.sigmoid(torch.as_tensor(data['opacity'], device='cuda'))
        local_q = torch.nn.functional.normalize(torch.as_tensor(data['quats'], device='cuda'), dim=1)
        r = torch.as_tensor(rotation_to_quat(rotation), dtype=torch.float32, device='cuda')
        self.quats = quat_mult(r.expand_as(local_q), local_q)
        self.center = torch.as_tensor(center_asset, dtype=torch.float32, device='cuda')
        self.rotation = torch.as_tensor(np.asarray(rotation, dtype=np.float32), device='cuda')
        self.means = self.xyz @ self.rotation.T + self.center

    def get_global_gaussians(self, timestamp=None):
        return dict(means=self.means, scales=self.scales, quats=self.quats, opacities=self.opacities)

    def get_gaussian_rgbs(self, camera_to_worlds, **kwargs):
        from gsplat.cuda._wrapper import spherical_harmonics
        view = self.means-camera_to_worlds[..., :3, 3]
        view = torch.nn.functional.normalize(view @ self.rotation, dim=-1)
        return torch.clamp(spherical_harmonics(3, view, torch.cat([self.dc[:, None, :], self.rest], dim=1))+.5, 0., 1.)


def rotation_to_quat(rotation):
    """3x3 rotation matrix to a quaternion in (w, x, y, z) order."""
    m = np.asarray(rotation, dtype=float)
    trace = m[0, 0]+m[1, 1]+m[2, 2]
    if trace > 0:
        s = math.sqrt(trace+1.)*2
        quat = [s/4, (m[2, 1]-m[1, 2])/s, (m[0, 2]-m[2, 0])/s, (m[1, 0]-m[0, 1])/s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.+m[0, 0]-m[1, 1]-m[2, 2])*2
        quat = [(m[2, 1]-m[1, 2])/s, s/4, (m[0, 1]+m[1, 0])/s, (m[0, 2]+m[2, 0])/s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.+m[1, 1]-m[0, 0]-m[2, 2])*2
        quat = [(m[0, 2]-m[2, 0])/s, (m[0, 1]+m[1, 0])/s, s/4, (m[1, 2]+m[2, 1])/s]
    else:
        s = math.sqrt(1.+m[2, 2]-m[0, 0]-m[1, 1])*2
        quat = [(m[1, 0]-m[0, 1])/s, (m[0, 2]+m[2, 0])/s, (m[1, 2]+m[2, 1])/s, s/4]
    return np.asarray(quat, dtype=float)


def added_vehicle_model(info, anchor, plan_to_asset, road_map, global_yaw, flatness_gate_m=0.10):
    """Build the AddedVehicle with its bounding-box bottom on the road surface.

    The road_height_map (reconstruction's own road, the visual ground truth)
    provides the center height and the pitch/roll of the fitted road plane under
    the footprint. Fails when the road is too non-planar under the footprint."""
    ply = read_ply(info['asset_path'])
    position = plan_to_asset @ np.array([anchor['x'], anchor['y'], 0., 1.])
    plane = road_map.placement_plane(position[0], position[1],
                                     info['size_m'][0], info['size_m'][1], global_yaw)
    if plane['max_corner_residual_m'] > flatness_gate_m:
        raise ValueError(f"Road under the added vehicle is not planar: "
                         f"corner residual {plane['max_corner_residual_m']:.3f} m")
    center = np.array([position[0], position[1], plane['z_center']+ply['dimensions'][2]/2])
    model = AddedVehicle(ply, center, plane['rotation'])
    placement = dict(road_z_asset_m=plane['z_center'], bottom_clearance_m=0.0,
                     pitch_rad=plane['pitch'], roll_rad=plane['roll'],
                     footprint_flatness_m=plane['max_corner_residual_m'],
                     target_asset_position=center.tolist(),
                     asset_dimensions_m=ply['dimensions'].tolist(),
                     asset_point_count=len(ply['means']))
    return model, placement


def run(scene, added_run, output, worldengine, frame_indices):
    scene = Path(scene).resolve(); added_run = Path(added_run).resolve(); output = Path(output).resolve()
    if (ROOT/'experiments').resolve() not in output.parents or output.exists():
        raise ValueError('Output must be a new experiments directory')
    info = json.loads((added_run/'summary.json').read_text())
    if Path(info['source_scene']).resolve() != scene or not info.get('asset_path'):
        raise ValueError('Added vehicle run must reference this scene and a PLY asset')
    frozen = json.loads((added_run/'frozen_scene/metadata/frames.json').read_text())
    times, _, _, objects = scene_from_frames(frozen, info['track_id'])
    if not frame_indices or sorted(set(frame_indices)) != frame_indices or min(frame_indices) < 0 or max(frame_indices) >= len(times):
        raise ValueError('Invalid frame indices')
    checkpoints = list((scene/'source_asset/background').glob('*.ckpt'))
    if len(checkpoints) != 1:
        raise ValueError('Expected exactly one scene checkpoint')
    asset, _ = load_checkpoint(checkpoints[0])
    recon = np.asarray(asset['background']['config']['recon2world_translation'], float)
    models = instantiate_models(asset, Path(worldengine).resolve())
    from road_height import RoadHeightMap, ego_height_reference
    road_map = RoadHeightMap.load(scene)
    height_ref = ego_height_reference(frozen, road_map, recon)
    anchor = next(a for a in objects[0] if a['track'] == info['track_id'])
    plan_to_asset = asset_from_first_ego(frozen[0], recon)
    initial_yaw = math.atan2(frozen[0]['ego2global'][1][0], frozen[0]['ego2global'][0][0])
    models['added_vehicle'], placement = added_vehicle_model(
        info, anchor, plan_to_asset, road_map, initial_yaw+anchor['yaw'])
    output.mkdir(parents=True)
    records = []
    for i in frame_indices:
        frame = frozen[i]; camera_data = {}
        current = next(a for a in objects[i] if a['track'] == info['track_id'])
        models['added_vehicle'], placement = added_vehicle_model(
            info, current, plan_to_asset, road_map, initial_yaw + current['yaw'])
        ego_matrix = np.asarray(frame['ego2global'], float).copy(); ego_matrix[:3, 3] -= recon
        ego_matrix[2, 3] = road_map.height_at(ego_matrix[0, 3], ego_matrix[1, 3]) \
            + height_ref['reference_m']
        for camera in CAMERAS:
            config = frame['cams'][camera]
            with Image.open(scene/'original_images'/config['data_path']) as image:
                width, height = image.size
            intrinsic, maps, width, height = camera_intrinsics(config, width, height)
            pose = ego_matrix @ np.asarray(frame['lidar2ego'], float) @ matrix_from_rotation_translation(
                config['sensor2lidar_rotation'], config['sensor2lidar_translation'])
            rgb, error = render_frame(models, pose, intrinsic, maps, width, height, int(frame['timestamp']))
            if error > 2e-4:
                raise ValueError('Camera pose readback failed')
            folder = output/'frames'/camera; folder.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(folder/f'frame{i:03d}.png')
            camera_data[camera] = dict(path=str((folder/f'frame{i:03d}.png').relative_to(output)), readback_error=error)
        records.append(dict(frame_index=i, time_s=float(times[i]),
                            target_xy_yaw=[current['x'], current['y'], current['yaw']],
                            vehicle_placement=placement,
                            ego_height_above_road_m=height_ref['reference_m'], cameras=camera_data))
        print(f'rendered frame {i}', flush=True)
    atomic_json(output/'render-report.json', dict(status='frames_rendered_needs_visual_check',
        scene=str(scene), added_run=str(added_run), asset=info['asset_path'], track_id=info['track_id'],
        vehicle_placement=placement,
        camera_z_policy=('ego origin z re-anchored to the scene road_height_map plus the frame-0 '
                         'recorded height reference; recorded pitch/roll retained'),
        camera_height_reference_m=height_ref['reference_m'],
        camera_offset_profile_m=height_ref['profile_m'],
        pose_assumptions='PLY local +X forward, +Z up; bounding-box bottom on the road_height_map surface',
        camera_order=list(CAMERAS), frames=records,
        physical_execution='not_run'))
    return records


@timed_operation('render.added_vehicle', None)
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--added-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worldengine', type=Path, required=True)
    parser.add_argument('--frames', type=int, nargs='+', required=True)
    args = parser.parse_args()
    run(args.scene, args.added_run, args.output, args.worldengine, args.frames)


if __name__ == '__main__':
    main()
