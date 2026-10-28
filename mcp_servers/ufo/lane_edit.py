"""Apply a single observed vehicle lane-change trajectory in native UFO units."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import shutil

from .actor_edit import actor_mask, actor_track_indices
from .offline_pipeline import write_json


def lane_progress(timestamp: float, start: float, end: float) -> float:
    if not all(math.isfinite(x) for x in (timestamp, start, end)) or end <= start:
        raise ValueError('lane-change timestamps must be finite and end > start')
    value = max(0., min(1., (timestamp - start) / (end - start)))
    return value ** 3 * (10 - 15 * value + 6 * value ** 2)


def view_consistent_delta(points, pose, intrinsics, left, pixels):
    """Move each observed surface sample by the same horizontal image offset.

    A rigid translation shears single-view depth outliers apart. This 2.5D mode
    preserves their projected silhouette from the recorded camera, retaining
    world height. It is not a rigid, metric, or novel-view vehicle model.
    """
    import numpy as np
    xyz = (points - pose[:3, 3]) @ pose[:3, :3]
    projected = xyz @ intrinsics.T
    if np.any(projected[:, 2] <= 1e-6):
        raise ValueError('Target contains samples behind the render camera')
    u = projected[:, 0] / projected[:, 2] + pixels
    normal = intrinsics[0][None, :] - u[:, None] * intrinsics[2][None, :]
    denominator = normal @ (left @ pose[:3, :3])
    if np.any(np.abs(denominator) < 1e-6):
        raise ValueError('Horizontal edit direction is parallel to target camera rays')
    offsets = -(normal * xyz).sum(1) / denominator
    delta = offsets[:, None] * left[None, :]
    after = (xyz + delta @ pose[:3, :3]) @ intrinsics.T
    error = float(np.max(np.abs(after[:, 0] / after[:, 2] - u)))
    if not np.isfinite(delta).all() or error > .02:
        raise ValueError(f'Projected target offset error: {error}')
    return delta.astype(np.float32), error


def inventory(workspace: Path) -> list[dict]:
    import numpy as np
    scene = workspace / 'reconstruction/000/pred100k'
    records = {}
    for path in sorted((scene / 'sources').glob('*.npz')):
        with np.load(path, allow_pickle=False) as source:
            actor = actor_mask(source)
            for ident in np.unique(source['tracks'][actor]):
                if ident <= 0:
                    continue
                mask = actor & (source['tracks'] == ident)
                y, x = np.where(mask)
                if len(x) < 20:
                    continue
                # Cityscapes car, truck, bus. Excludes pedestrian/bicycle tracks.
                vehicle_fraction = float(np.isin(source['classes'][mask], [13, 14, 15]).mean())
                if vehicle_fraction < .5:
                    continue
                record = records.setdefault(int(ident), {'track_id': int(ident), 'observations': []})
                record['observations'].append({'frame': int(path.stem), 'points': len(x),
                    'bbox': [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1],
                    'image_size': list(source['tracks'].shape),
                    'center_distance': abs(float(np.median(x)) / source['tracks'].shape[1] - .5),
                    'vehicle_fraction': vehicle_fraction,
                    'class_id': {13: 2, 14: 7, 15: 5}[int(np.bincount(source['classes'][mask].astype(int)).argmax())]
                    if int(np.bincount(source['classes'][mask].astype(int)).argmax()) in (13, 14, 15) else 2})
    result = []
    for record in records.values():
        observations = record['observations']
        record['visible_frames'] = len(observations)
        record['median_center_distance'] = float(np.median([row['center_distance'] for row in observations]))
        result.append(record)
    return sorted(result, key=lambda row: (-row['visible_frames'], row['median_center_distance'], row['track_id']))


def choose_target(candidates: list[dict], track_id: int | None, frames: int) -> dict:
    if track_id is not None:
        selected = next((item for item in candidates if item['track_id'] == track_id), None)
        if selected is None:
            raise ValueError(f'SAM track {track_id} is not a usable car/truck/bus; inspect targets.json')
    else:
        # Generic tool demonstration only; not request-conditioned semantic selection.
        selected = next((item for item in candidates if item['visible_frames'] >= max(3, math.ceil(frames * .8))), None)
        if selected is None:
            raise ValueError('No motor vehicle visible in >=80% of frames; inspect targets.json and specify --track-id')
    if selected['visible_frames'] < 3:
        raise ValueError('A lane-change test needs at least 3 target observations')
    return selected


def preview(workspace: Path, target: dict, output: Path) -> None:
    from PIL import Image, ImageDraw
    observations = target['observations']
    samples = [observations[index] for index in sorted({0, len(observations) // 2, len(observations) - 1})]
    canvas = Image.new('RGB', (520 * len(samples), 382), '#111827')
    for panel, row in enumerate(samples):
        image = Image.open(workspace / f'reconstruction/000/pred100k/reference/{row["frame"]:06d}.png').convert('RGB')
        draw = ImageDraw.Draw(image)
        draw.rectangle(row['bbox'], outline='#00ffff', width=3)
        canvas.paste(image, (panel * 520, 30))
        ImageDraw.Draw(canvas).text((panel * 520 + 8, 8), f'SAM {target["track_id"]} | frame {row["frame"]}', fill='white')
    canvas.save(output, quality=95)


def edit(workspace: Path, asset_root: Path, output_root: Path, new_name: str,
         *, track_id: int | None = None, direction: str = 'right',
         lateral_distance: float | None = None, start_fraction: float = .1,
         end_fraction: float = .9, targets_only: bool = False, vehicle_widths: float = 1.2,
         target_selection_path: Path | None = None) -> dict:
    import numpy as np
    import torch
    if direction not in ('left', 'right') or not 0 <= start_fraction < end_fraction <= 1:
        raise ValueError('Require left/right and 0 <= start_fraction < end_fraction <= 1')
    if lateral_distance is not None and (not math.isfinite(lateral_distance) or lateral_distance <= 0):
        raise ValueError('lateral_distance must be a positive native-unit distance')
    if not math.isfinite(vehicle_widths) or not 0 < vehicle_widths <= 2:
        raise ValueError('vehicle_widths must be in (0, 2]')
    if not new_name or new_name == workspace.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in new_name):
        raise ValueError('new_name must be a distinct ASCII asset ID')
    output_root.mkdir(parents=True, exist_ok=True)
    original = asset_root / 'assets' / workspace.name
    manifest = json.loads((original / 'ASSET.json').read_text())
    if manifest.get('version') != 'ufo-worldengine-time-v2' or manifest.get('asset_id') != workspace.name:
        raise ValueError('Expected the requested time-v2 asset')
    with np.load(original / 'cameras.npz', allow_pickle=False) as cameras:
        poses, times = cameras['C'], cameras['timestamps']
        intrinsics = cameras['K'] if lateral_distance is None else None
    candidates = inventory(workspace)
    write_json(output_root / 'targets.json', candidates)
    if targets_only:
        return {'targets': str(output_root / 'targets.json'), 'candidates': candidates}
    binding = None
    if target_selection_path:
        from .binding import bind_selected_track
        binding = bind_selected_track(workspace, target_selection_path)
        if track_id is not None and track_id != binding['sam_track_id']:
            raise ValueError('Explicit track differs from selected target')
        track_id = binding['sam_track_id']
    target = choose_target(candidates, track_id, len(poses))
    preview(workspace, target, output_root / 'target_preview.jpg')
    write_json(output_root / 'target_selection.json', {
        'selection': 'explicit SAM track' if track_id else 'most visible motor vehicle; image-center proximity breaks ties',
        'semantic_user_request_evaluated': bool(binding), 'binding': binding, **target})
    ident = target['track_id']
    destination = output_root / 'assets' / new_name
    parameters = {'track_id': ident, 'direction': direction, 'lateral_distance_override': lateral_distance,
                  'start_fraction': start_fraction, 'end_fraction': end_fraction,
                  'vehicle_widths': vehicle_widths, 'profile': 'view_consistent_v2' if lateral_distance is None else 'relative_rigid_v2'}
    if destination.exists():
        audit_file = destination / 'EDIT_AUDIT.json'
        if audit_file.is_file() and (destination / 'ASSET.json').is_file():
            existing = json.loads(audit_file.read_text())
            if existing.get('edit_parameters') == parameters:
                print(f'RESUME edited asset: {destination}', flush=True)
                return existing
        raise FileExistsError(f'Edited asset already exists; use a new scene name: {destination}')
    checkpoint = torch.load(manifest['asset_ckpt'], map_location='cpu', weights_only=False)
    original_files = checkpoint['observed_dynamic']['config']['actor_files']
    if len(original_files) != len(poses):
        raise ValueError('Actor file count differs from camera count')
    rotation = np.asarray(manifest['source_to_world_rotation'], dtype=np.float32)
    origin = np.asarray(manifest['source_origin'], dtype=np.float32)
    observed = {}
    widths = []
    for index in range(len(poses)):
        path = workspace / f'reconstruction/000/pred100k/sources/{index:06d}.npz'
        with np.load(path, allow_pickle=False) as source:
            mask = actor_mask(source) & (source['tracks'] == ident)
            if not mask.any():
                continue
            points = (source['means'][mask] - origin) @ rotation.T
        left = -poses[index][:3, 0].copy()
        left[2] = 0  # WorldEngine world Z is up; keep road height unchanged.
        left /= np.linalg.norm(left)
        lateral = (points - poses[index][:3, 3]) @ left
        widths.append(float(np.quantile(lateral, .9) - np.quantile(lateral, .1)))
        observed[index] = {'lateral': float(np.median(lateral)), 'left': left}
    valid_indices = sorted(observed)
    if len(valid_indices) < 3:
        raise ValueError('Target has fewer than 3 observed Gaussian frames')
    width = float(np.median(widths))
    distance = float(lateral_distance if lateral_distance is not None else vehicle_widths * width)
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError('Cannot estimate a positive lateral distance; set --lateral-distance-native')
    start = float(times[0] + (times[-1] - times[0]) * start_fraction)
    end = float(times[0] + (times[-1] - times[0]) * end_fraction)
    boxes = {row['frame']: row['bbox'] for row in target['observations']}
    destination.mkdir(parents=True)
    (destination / 'dynamic').mkdir()
    (destination / 'background').mkdir()
    shutil.copy2(original / 'cameras.npz', destination / 'cameras.npz')
    audit, files = [], []
    for index, actor_file in enumerate(original_files):
        actor = torch.load(actor_file, map_location='cpu', weights_only=False)
        with np.load(workspace / f'reconstruction/000/pred100k/sources/{index:06d}.npz', allow_pickle=False) as source:
            full_mask = actor_mask(source)
            if len(actor['means']) != int(full_mask.sum()):
                raise ValueError(f'Actor/source order mismatch at frame {index}')
            indices = np.flatnonzero((source['tracks'][full_mask] == ident))
            expected = (source['means'][full_mask][indices] - origin) @ rotation.T
        if len(indices) and not np.allclose(actor['means'][indices].numpy(), expected, rtol=1e-5, atol=1e-4):
            raise ValueError(f'Target/source geometry mismatch at frame {index}')
        delta = np.zeros(3, dtype=np.float32)
        requested = actual = 0.
        pixel_offset = projection_error = 0.
        if index in observed and times[index] > start:
            row = observed[index]
            progress = lane_progress(float(times[index]), start, end)
            requested = (distance if direction == 'left' else -distance) * progress
            if lateral_distance is None:
                nearby = [boxes[j][2] - boxes[j][0] for j in range(max(0, index-2), min(len(poses), index+3)) if j in boxes]
                pixel_offset = (1 if direction == 'right' else -1) * vehicle_widths * float(np.median(nearby)) * progress
                delta, projection_error = view_consistent_delta(expected, poses[index], intrinsics[index], row['left'], pixel_offset)
                requested = float((delta @ row['left']).mean())
            else:
                delta = row['left'] * requested
        before = actor['means'].clone()
        if len(indices):
            actor['means'][indices] += torch.tensor(delta, dtype=actor['means'].dtype)
            actual = float(((actor['means'][indices] - before[indices]).mean(0).numpy() @ observed[index]['left']))
        other = np.ones(len(before), dtype=bool)
        other[indices] = False
        if not torch.equal(actor['means'][other], before[other]) or not torch.equal(actor['means'][:, 2], before[:, 2]):
            raise ValueError('Lane edit modified another actor or target world height')
        error = abs(actual - requested)
        if error > max(1e-4, distance * 1e-5):
            raise ValueError(f'Lane offset numerical error at frame {index}: {error}')
        path = destination / 'dynamic' / f'{index:06d}.pt'
        torch.save(actor, path)
        files.append(str(path))
        mean_delta = delta.mean(0) if delta.ndim == 2 else delta
        edited_lateral = (float(np.median((actor['means'][indices].numpy() - poses[index][:3, 3]) @ observed[index]['left']))
                          if index in observed else None)
        audit.append({'frame': index, 'timestamp_seconds': float(times[index]), 'target_gaussians': len(indices),
                      'world_native_delta': mean_delta.tolist(), 'requested_left_offset_native': requested,
                      'actual_left_offset_native': actual, 'absolute_error_native': error,
                      'image_offset_pixels': pixel_offset, 'projection_error_pixels': projection_error,
                      'original_lateral_native': observed[index]['lateral'] if index in observed else None,
                      'edited_lateral_native': edited_lateral,
                      'other_actors_unchanged': True, 'world_height_unchanged': True})
    checkpoint['observed_dynamic']['config']['actor_files'] = files
    background_config = checkpoint['background']['config']
    for key, name in [('neural_shader_file', 'background_shader.pt'), ('neural_states_file', 'background_states.pt'),
                      ('sky_environment_file', 'sky_environment.json')]:
        if key in background_config:
            shutil.copy2(background_config[key], destination / 'background' / name)
            background_config[key] = str(destination / 'background' / name)
    asset_ckpt = destination / 'background' / f'{new_name}.ckpt'
    torch.save(checkpoint, asset_ckpt)
    edited_manifest = {**manifest, 'asset_id': new_name, 'asset_ckpt': str(asset_ckpt),
                       'camera_file': str(destination / 'cameras.npz'), 'edited_track_id': ident,
                       'edit_operation': 'native_lane_change', 'metric_calibrated': False}
    write_json(destination / 'ASSET.json', edited_manifest)
    report = {'asset_id': new_name, 'asset_root': str(output_root), 'asset_manifest': str(destination / 'ASSET.json'),
              'operation': 'lane_change', 'direction': direction, 'track_id': ident, 'units': 'native',
              'lateral_distance_native': distance, 'target_width_native': width,
              'distance_source': 'explicit relative native distance' if lateral_distance else 'per-frame image width; native distance is diagnostic only',
              'profile': parameters['profile'], 'target_binding': binding,
              'start_seconds': start, 'end_seconds': end, 'metric_calibrated': False,
              'edit_parameters': parameters,
              'target_preview': str(output_root / 'target_preview.jpg'), 'frames': audit,
              'limitation': '2.5D observed-camera visual edit, not a rigid metric vehicle; no lane/collision validation or background completion; newly exposed background can retain ghosts'}
    write_json(destination / 'EDIT_AUDIT.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace', 'asset-root', 'output-root', 'worldengine-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--new-name', required=True)
    parser.add_argument('--track-id', type=int)
    parser.add_argument('--direction', choices=['left', 'right'], default='right')
    parser.add_argument('--lateral-distance-native', type=float)
    parser.add_argument('--start-fraction', type=float, default=.1)
    parser.add_argument('--end-fraction', type=float, default=.9)
    parser.add_argument('--vehicle-widths', type=float, default=1.2)
    parser.add_argument('--target-selection-path', type=Path)
    parser.add_argument('--targets-only', action='store_true')
    args = parser.parse_args()
    report = edit(args.workspace.resolve(), args.asset_root.resolve(), args.output_root.resolve(), args.new_name,
                  track_id=args.track_id, direction=args.direction, lateral_distance=args.lateral_distance_native,
                  start_fraction=args.start_fraction, end_fraction=args.end_fraction, targets_only=args.targets_only,
                  vehicle_widths=args.vehicle_widths, target_selection_path=args.target_selection_path)
    if not args.targets_only:
        from .metric_edit import render_asset
        video = render_asset(args.output_root.resolve(), args.new_name,
                             args.output_root.resolve() / 'edited.mp4', worldengine_root=args.worldengine_root.resolve())
        report['rendered_video'], report['frame_count'] = video['rendered_video'], video['frames']
    write_json(args.output_root.resolve() / 'result.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
