"""Render recorded and edited ego cameras from the target scene's WorldEngine MTGS asset."""
from workflow_timing import timed_operation

import argparse
import csv
import json
import math
import pickletools
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from experiment_runtime import atomic_json
from run_ego_foci_probe import read_track
from run_ego_reference import DEFAULT_TARGET_TRACK, scene_from_frames

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_CHECKPOINT_GLOBALS = {
    'collections.OrderedDict', 'numpy.core.multiarray._reconstruct', 'numpy.ndarray',
    'numpy.dtype', '_codecs.encode', 'torch._utils._rebuild_tensor_v2',
    'torch.FloatStorage', 'torch.LongStorage',
}


def validate_checkpoint_pickle(path):
    path = Path(path)
    if not zipfile.is_zipfile(path):
        raise ValueError('MTGS checkpoint is not a PyTorch zip archive')
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith('data.pkl')]
        if len(names) != 1:
            raise ValueError('Require exactly one MTGS data.pkl')
        operators = list(pickletools.genops(archive.read(names[0])))
    globals_found = {arg.replace(' ', '.') for op, arg, _ in operators if op.name == 'GLOBAL'}
    unexpected = globals_found-ALLOWED_CHECKPOINT_GLOBALS
    if unexpected:
        raise ValueError(f'Unexpected checkpoint globals: {sorted(unexpected)}')
    return sorted(globals_found)


def load_checkpoint(path):
    import torch
    globals_found = validate_checkpoint_pickle(path)
    return torch.load(path, map_location='cpu'), globals_found


def planar_matrix(x, y, yaw, z=0.):
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    matrix = np.eye(4)
    matrix[:2, :2] = [[c, -s], [s, c]]
    matrix[:3, 3] = [float(x), float(y), float(z)]
    return matrix


def asset_from_first_ego(first, recon_translation):
    ego0 = np.asarray(first['ego2global_translation'], float)-recon_translation
    yaw0 = math.atan2(first['ego2global'][1][0], first['ego2global'][0][0])
    return planar_matrix(ego0[0], ego0[1], yaw0, ego0[2])


def matrix_from_rotation_translation(rotation, translation):
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(rotation, float)
    matrix[:3, 3] = np.asarray(translation, float)
    return matrix


def edited_ego_matrix(recorded_ego, recorded_xy, recorded_yaw, candidate_xy, candidate_yaw,
        asset_from_plan):
    recorded_planar = planar_matrix(*recorded_xy, recorded_yaw)
    candidate_planar = planar_matrix(*candidate_xy, candidate_yaw)
    delta = asset_from_plan @ candidate_planar @ np.linalg.inv(recorded_planar) @ np.linalg.inv(asset_from_plan)
    return delta @ recorded_ego


def camera_intrinsics(camera, width, height):
    intrinsic = np.asarray(camera['cam_intrinsic'], float)
    distortion = np.asarray(camera['distortion'], float)
    new_intrinsic, _ = cv2.getOptimalNewCameraMatrix(intrinsic, distortion, (width, height), 1)
    inverse_maps = cv2.initInverseRectificationMap(intrinsic, distortion, None, new_intrinsic,
        (width, height), cv2.CV_32FC1)
    return new_intrinsic, inverse_maps, width, height


def instantiate_models(asset, worldengine_dir):
    import types
    worldengine_dir = Path(worldengine_dir).resolve()
    namespaces = {
        'worldengine':worldengine_dir/'worldengine',
        'worldengine.render':worldengine_dir/'worldengine/render',
        'worldengine.render.mtgs':worldengine_dir/'worldengine/render/mtgs',
        'worldengine.render.mtgs.utils':worldengine_dir/'worldengine/render/mtgs/utils',
        'worldengine.render.mtgs.gaussian_model':worldengine_dir/'worldengine/render/mtgs/gaussian_model',
    }
    for name, path in namespaces.items():
        if name not in sys.modules:
            module = types.ModuleType(name); module.__path__ = [str(path)]
            sys.modules[name] = module
    from worldengine.render.mtgs.utils.portable_utils import convert_to_attribute_dict
    from worldengine.render.mtgs.gaussian_model.vanilla_gaussian_splatting import VanillaPortableGaussianModel
    from worldengine.render.mtgs.gaussian_model.rigid_object import RigidPortableSubModel
    from worldengine.render.mtgs.gaussian_model.rigid_object_mirrored import MirroredRigidPortableSubModel
    classes = {'VanillaGaussianSplattingModel':VanillaPortableGaussianModel,
        'SkyboxGaussianSplattingModel':VanillaPortableGaussianModel,
        'RigidSubModel':RigidPortableSubModel, 'MirroredRigidSubModel':MirroredRigidPortableSubModel}
    models = {}
    for name, atom in asset.items():
        converted = convert_to_attribute_dict(atom)
        model_type = atom['config']['type']
        if model_type not in classes:
            raise ValueError(f'Unsupported MTGS model type: {model_type}')
        models[name] = classes[model_type](asset=converted, model_name=name).cuda().eval()
    return models


@timed_operation('render.camera_frame')
def render_frame(models, camera_to_world, intrinsic, inverse_maps, width, height, timestamp):
    import torch
    from gsplat.rendering import rasterization
    device = torch.device('cuda')
    camera = torch.as_tensor(camera_to_world, dtype=torch.float32, device=device)[None]
    viewmat = torch.eye(4, dtype=torch.float32, device=device)[None]
    viewmat[:, :3, :3] = camera[:, :3, :3].transpose(1, 2)
    viewmat[:, :3, 3] = -(camera[:, :3, :3].transpose(1, 2) @ camera[:, :3, 3:4]).squeeze(-1)
    collected = {key:[] for key in ('means', 'scales', 'quats', 'opacities')}
    rgbs = []
    with torch.no_grad():
        for model in models.values():
            gaussians = model.get_global_gaussians(timestamp=timestamp)
            if gaussians is None:
                continue
            for key in collected:
                collected[key].append(gaussians[key].to(device))
            rgbs.append(model.get_gaussian_rgbs(camera_to_worlds=camera, timestamp=timestamp,
                device=device).unsqueeze(0))
        values = {key:torch.cat(value, dim=0) for key, value in collected.items()}
        colors = torch.cat(rgbs, dim=1)
        render, alpha, _ = rasterization(means=values['means'], quats=values['quats'],
            scales=values['scales'], opacities=values['opacities'], colors=colors,
            viewmats=viewmat, Ks=torch.as_tensor(intrinsic, dtype=torch.float32, device=device)[None],
            width=width, height=height, tile_size=16, packed=False, near_plane=.01, far_plane=1e10,
            render_mode='RGB', sparse_grad=False, absgrad=True, rasterize_mode='antialiased', radius_clip=0.)
        rgb = render[0, ..., :3]+(1-alpha[0])*0.
        rgb = rgb.clamp(0, 1).cpu().numpy()
        camera_readback = torch.inverse(viewmat).cpu().numpy()[0]
    rgb = cv2.remap((rgb*255).astype(np.uint8), inverse_maps[0], inverse_maps[1], cv2.INTER_LINEAR)
    return rgb, float(np.abs(camera_readback-camera_to_world).max())


@timed_operation('render.ego', 'output')
def run(input_dir, parent_run, candidate_id, output, worldengine_dir, selected_frames=None,
        target_track=None):
    if target_track is None:
        raise ValueError('target_track must identify a vehicle in the current scene')
    input_dir = Path(input_dir).resolve(); parent_run = Path(parent_run).resolve(); output = Path(output).resolve()
    if not output.is_relative_to((ROOT/'experiments').resolve()):
        raise ValueError('Output must be under experiments/')
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing MTGS render')
    constructor_track = parent_run/'reference_screen'/candidate_id/'trajectory.csv'
    agent_track = parent_run/'candidates'/candidate_id/'trajectory.csv'
    if constructor_track.is_file() == agent_track.is_file():
        raise ValueError('Require exactly one constructor or agent candidate layout')
    candidate_track = constructor_track if constructor_track.is_file() else agent_track
    checkpoints = list((input_dir/'source_asset/background').glob('*.ckpt'))
    if len(checkpoints) != 1:
        raise ValueError('Require exactly one WorldEngine MTGS checkpoint')
    checkpoint = checkpoints[0]
    frames = json.loads((input_dir/'metadata/frames.json').read_text())
    times, recorded_xy, recorded_yaw, _ = scene_from_frames(frames, target_track)
    candidate_xy, candidate_yaw = read_track(candidate_track, times)
    indices = list(range(len(frames))) if selected_frames is None else sorted(set(selected_frames))
    if not indices or indices != (selected_frames if selected_frames is not None else indices) \
            or any(i < 0 or i >= len(frames) for i in indices):
        raise ValueError('Frames must be unique, sorted and covered by the recording')
    asset, checkpoint_globals = load_checkpoint(checkpoint)
    recon_translation = np.asarray(asset['background']['config']['recon2world_translation'], float)
    first = frames[0]
    asset_from_plan = asset_from_first_ego(first, recon_translation)
    output.mkdir(parents=True)
    (output/'input').mkdir()
    shutil.copyfile(candidate_track, output/'input/candidate-trajectory.csv')
    config = dict(version='target_ego_mtgs_render.v1', input_dir=str(input_dir), parent_run=str(parent_run),
        candidate_id=candidate_id, checkpoint=str(checkpoint), checkpoint_globals=checkpoint_globals,
        worldengine_dir=str(worldengine_dir), frame_indices=indices, sample_count=len(frames), fps=2.,
        camera_construction='asset_from_plan @ candidate_planar @ inverse(recorded_planar) @ inverse(asset_from_plan) @ recorded_ego @ cam_to_ego',
        z_pitch_roll_policy=('Ego origin z re-anchored to the scene road_height_map plus the frame-0 '
            'recorded height reference (UTM z drifts from the reconstructed road); recorded '
            'pitch/roll retained'),
        actor_policy='Background, skybox and all rigid actors use unmodified MTGS checkpoint log replay at exact timestamps',
        physical_execution='not_run', carla_executed=False)
    atomic_json(output/'config.json', config)
    models = instantiate_models(asset, worldengine_dir)
    from road_height import RoadHeightMap, ego_height_reference
    road_map = RoadHeightMap.load(input_dir)
    height_ref = ego_height_reference(frames, road_map, recon_translation)
    config['camera_height_reference_m'] = height_ref['reference_m']
    config['camera_offset_profile_m'] = height_ref['profile_m']
    atomic_json(output/'config.json', config)
    for folder in ('frames', 'recorded', 'compare'):
        (output/folder).mkdir()
    writer = imageio.get_writer(output/'new-camera.mp4', fps=2., codec='libx264', macro_block_size=2)
    recorded_writer = imageio.get_writer(output/'recorded-camera.mp4', fps=2., codec='libx264', macro_block_size=2)
    per_frame = []; max_camera_readback = 0.
    try:
        for i in indices:
            frame = frames[i]
            recorded_ego = np.asarray(frame['ego2global'], float)
            recorded_ego[:3, 3] -= recon_translation
            recorded_ego[2, 3] = road_map.height_at(recorded_ego[0, 3], recorded_ego[1, 3]) \
                + height_ref['reference_m']
            new_ego = edited_ego_matrix(recorded_ego, recorded_xy[i], recorded_yaw[i],
                candidate_xy[i], candidate_yaw[i], asset_from_plan)
            candidate_asset_xy = (asset_from_plan @ np.array([*candidate_xy[i], 0., 1.]))[:2]
            new_ego[2, 3] = road_map.height_at(*candidate_asset_xy) + height_ref['reference_m']
            camera_info = frame['cams']['CAM_F0']
            sensor_to_lidar = matrix_from_rotation_translation(camera_info['sensor2lidar_rotation'],
                camera_info['sensor2lidar_translation'])
            cam_to_ego = np.asarray(frame['lidar2ego'], float) @ sensor_to_lidar
            recorded_camera = recorded_ego @ cam_to_ego
            candidate_camera = new_ego @ cam_to_ego
            if not np.isfinite(candidate_camera).all():
                raise ValueError(f'Nonfinite candidate camera at frame {i}')
            actual_path = input_dir/'original_images'/camera_info['data_path']
            if not actual_path.is_file():
                raise ValueError(f'Missing original camera image: {actual_path}')
            with Image.open(actual_path) as actual_image:
                width, height = actual_image.size
            intrinsic, inverse_maps, width, height = camera_intrinsics(camera_info, width, height)
            timestamp = int(frame['timestamp'])
            original, original_error = render_frame(models, recorded_camera, intrinsic, inverse_maps,
                width, height, timestamp)
            edited, edited_error = render_frame(models, candidate_camera, intrinsic, inverse_maps,
                width, height, timestamp)
            max_camera_readback = max(max_camera_readback, original_error, edited_error)
            if max_camera_readback > 2e-4:
                raise ValueError('Camera matrix readback exceeds tolerance')
            original_path = output/'recorded'/f'frame{i:03d}.png'
            edited_path = output/'frames'/f'frame{i:03d}.png'
            Image.fromarray(original).save(original_path); Image.fromarray(edited).save(edited_path)
            comparison = np.concatenate([original, edited], axis=1)
            Image.fromarray(comparison).save(output/'compare'/f'frame{i:03d}.png')
            recorded_writer.append_data(original); writer.append_data(edited)
            actual_path = input_dir/'original_images'/camera_info['data_path']
            mae = None
            if actual_path.is_file():
                with Image.open(actual_path) as actual_image:
                    actual = np.asarray(actual_image.convert('RGB'))
                if actual.shape == original.shape:
                    mae = float(np.abs(actual.astype(float)-original.astype(float)).mean()/255.)
            per_frame.append(dict(frame_index=i, time_s=float(times[i]), timestamp=timestamp,
                candidate_camera_image=f'frames/frame{i:03d}.png', recorded_camera_image=f'recorded/frame{i:03d}.png',
                candidate_ego_xy=candidate_xy[i].tolist(), candidate_ego_yaw=float(candidate_yaw[i]),
                recorded_render_mae=mae))
            print(json.dumps(per_frame[-1]), flush=True)
    finally:
        writer.close(); recorded_writer.close()
    decoded = {}
    for name in ('new-camera.mp4', 'recorded-camera.mp4'):
        reader = imageio.get_reader(output/name)
        metadata = reader.get_meta_data(); count = sum(1 for _ in reader); reader.close()
        if count != len(indices) or abs(float(metadata['fps'])-2.) > 1e-6:
            raise ValueError(f'Video decode/fps mismatch: {name}')
        decoded[name] = count
    full_window = indices == list(range(len(frames)))
    report = dict(status='passed', full_window=full_window, renderer='WorldEngine MTGS portable Gaussian models and gsplat',
        checkpoint=str(checkpoint), checkpoint_step=30000, candidate_id=candidate_id,
        rendered_frames=len(indices), rendered_frame_ids=indices, decoded_videos=decoded, fps=2.,
        camera='CAM_F0; candidate planar ego x/y/yaw with recorded z/pitch/roll and per-frame camera extrinsics',
        camera_readback_max_abs_error=max_camera_readback,
        actors='Unmodified MTGS log replay; no actor trajectory overrides',
        quality_evaluation='Recorded-camera MAE is a same-camera diagnostic only; no aligned ground truth or PSNR is claimed for the edited camera',
        physical_avoidance='undetermined', carla_executed=False, per_frame=per_frame)
    atomic_json(output/'render-report.json', report)
    atomic_json(output/'artifact-check.json', dict(status='passed', full_window=full_window,
        rendered_frames=len(indices), decoded_videos=decoded, camera_readback_max_abs_error=max_camera_readback,
        candidate_id=candidate_id, physical_execution='not_run'))
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
    args = parser.parse_args()
    report = run(args.input, args.parent_run, args.candidate, args.output, args.worldengine, args.frames,
        target_track=args.target_track)
    print('SUMMARY '+json.dumps({key:value for key, value in report.items() if key != 'per_frame'}), flush=True)


if __name__ == '__main__':
    main()
