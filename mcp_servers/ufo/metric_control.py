"""Rigid metric UFO editing and known-size Gaussian asset insertion.

All numerical work runs in ufoex. Calibration and edited assets are cached;
reconstruction is never rerun here. Rendering is an explicit final stage.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import uuid

from .metric_config import validate, resolve_paths
from .metric_scene import calibrate, digest, load_calibration, road_frame


def read_reconstruction(path: Path) -> dict:
    value = json.loads(path.read_text())
    result = value.get('result', value)
    if result.get('backend') != 'ufo_worldengine_time_v2' or result.get('frame_count') != 40:
        raise ValueError('Use the successful 40-frame UFO reconstruction result/envelope')
    return result


def build_edit(reconstruction: dict, config: dict, output_root: Path) -> dict:
    import numpy as np
    import torch
    from .actor_edit import actor_mask
    from .binding import bind_selected_track
    from .lane_edit import lane_progress
    from .metric_assets import FIELDS, load_asset, place_asset, validate_gaussians

    config = validate(config)
    workspace, asset_root = Path(reconstruction['workspace']).resolve(), Path(reconstruction['asset_root']).resolve()
    calibration_path = Path(config['calibration_path']).expanduser().resolve()
    calibration = load_calibration(calibration_path, workspace, asset_root)
    scale = calibration['meters_per_native_unit']
    original = asset_root / 'assets' / workspace.name
    manifest = json.loads((original / 'ASSET.json').read_text())
    with np.load(original / 'cameras.npz', allow_pickle=False) as cameras:
        poses, times = cameras['C'].copy(), cameras['timestamps'].copy()
    if len(times) != 40 or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError('Metric editing requires 40 increasing finite camera timestamps')
    translations = [op for op in config['operations'] if op['operation'] == 'translate']
    for op in translations:
        if op['start_seconds'] < times[0]-1e-6 or op['end_seconds'] > times[-1]+1e-6:
            raise ValueError('Requested trajectory exceeds the reconstructed video time span')
    start = min((op['start_seconds'] for op in translations), default=float(times[0]))
    reference_index = int(np.argmin(np.abs(times-start)))
    basis, origin = road_frame(poses[reference_index], calibration['road_plane'])
    binding = None
    if any(op.get('target_id') == 'selected' for op in translations):
        if config.get('target_selection_path'):
            binding = bind_selected_track(workspace, Path(config['target_selection_path']).expanduser().resolve())
            track_id = binding['sam_track_id']
        else:
            track_id = config['track_id']  # Explicit manual CLI ID, never an Agent guess.
            binding = {'sam_track_id': track_id, 'binding_source': 'explicit manual track_id'}
    assets, metadata = {}, {}
    for op in config['operations']:
        if op['operation'] == 'insert':
            local, info = load_asset(Path(op['asset_manifest_path']).expanduser())
            assets[op['object_id']] = place_asset(local, basis, origin, op['position_m'], op['yaw_degrees'], scale)
            metadata[op['object_id']] = info
    contract = {'algorithm': 'ufo-rigid-metric-v1', 'config': config,
                'calibration_sha256': digest(calibration_path), 'assets': metadata,
                'target_binding': binding}
    if config.get('road_geometry')=='planar_proxy':
        from . import road_proxy
        contract['road_proxy_code_sha256']=digest(Path(road_proxy.__file__))
    job_hash = hashlib.sha256(json.dumps(contract, sort_keys=True, allow_nan=False).encode()).hexdigest()
    name = f'{workspace.name}_metric_{job_hash[:12]}'
    output_root = output_root.resolve()
    destination = output_root / 'assets' / name
    audit_file = destination / 'EDIT_AUDIT.json'
    if destination.exists():
        if audit_file.is_file() and (destination / 'ASSET.json').is_file():
            existing = json.loads(audit_file.read_text())
            if existing.get('job_sha256') == job_hash and all(
                (destination / 'dynamic' / f'{i:06d}.pt').is_file() for i in range(len(times))):
                print(f'RESUME metric asset: {destination}', flush=True)
                return existing
        raise FileExistsError('Incomplete or incompatible edited asset exists; choose a new output_root')
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f'.building_{uuid.uuid4().hex}'
    (staging / 'dynamic').mkdir(parents=True)
    (staging / 'background').mkdir()
    try:
        checkpoint = torch.load(manifest['asset_ckpt'], map_location='cpu', weights_only=False)
        original_files = checkpoint['observed_dynamic']['config']['actor_files']
        if len(original_files) != len(times):
            raise ValueError('Actor file count differs from reconstruction cameras')
        rotation = np.asarray(manifest['source_to_world_rotation'], dtype=np.float64)
        source_origin = np.asarray(manifest['source_origin'], dtype=np.float64)
        rows, files = [], []
        for index, actor_file in enumerate(original_files):
            actor = torch.load(actor_file, map_location='cpu', weights_only=True)
            validate_gaussians(actor)
            original_count = len(actor['means'])
            if assets and any(key not in FIELDS and isinstance(value, torch.Tensor) and value.ndim
                              and len(value) == original_count for key, value in actor.items()):
                raise ValueError('Actor contains unsupported per-point attributes; refusing incomplete insertion')
            original_actor = {key: actor[key].clone() for key in FIELDS}
            targets = {}
            if binding:
                with np.load(workspace / f'reconstruction/000/pred100k/sources/{index:06d}.npz', allow_pickle=False) as source:
                    full = actor_mask(source)
                    if int(full.sum()) != original_count:
                        raise ValueError(f'Actor/source count mismatch at frame {index}')
                    selected = np.flatnonzero(source['tracks'][full] == track_id)
                    expected = (source['means'][full][selected]-source_origin) @ rotation.T
                if len(selected) and not np.allclose(actor['means'][selected].numpy(), expected, atol=1e-4, rtol=1e-5):
                    raise ValueError(f'Selected actor/source geometry mismatch at frame {index}')
                targets['selected'] = selected
            frame_audit = []
            for op in config['operations']:
                if op['operation'] == 'insert':
                    ident = op['object_id']
                    inserted = assets[ident]
                    reference_delta = np.zeros(3, dtype=np.float64)
                    if op.get('motion_reference') in {'ego_projected','ego_translation'}:
                        reference_delta = poses[index,:3,3].astype(np.float64)-poses[reference_index,:3,3]
                        if op['motion_reference'] == 'ego_projected':
                            reference_delta -= basis[:,2]*float(reference_delta @ basis[:,2])
                    begin, count = len(actor['means']), len(inserted['means'])
                    for key in FIELDS:
                        values = inserted[key]+reference_delta if key == 'means' else inserted[key]
                        actor[key] = torch.cat([actor[key], torch.as_tensor(values, dtype=actor[key].dtype)], dim=0)
                    targets[ident] = np.arange(begin, begin+count)
                    frame_audit.append({'operation': 'insert', 'object_id': ident, 'gaussians': count,
                                        'requested_position_m': op['position_m'], 'dimensions_m': metadata[ident]['dimensions_m'],
                                        'motion_reference': op.get('motion_reference','world_fixed'),
                                        'reference_offset_flu_m': (reference_delta @ basis*scale).tolist()})
                else:
                    ident = op['target_id']
                    selected = targets[ident]
                    progress = lane_progress(float(times[index]), op['start_seconds'], op['end_seconds'])
                    offset = np.asarray(op['offset_m'], dtype=np.float64)*progress
                    delta = basis @ offset/scale
                    actual, error = None, None
                    if len(selected):
                        before = actor['means'][selected].clone()
                        actor['means'][selected] += torch.as_tensor(delta, dtype=actor['means'].dtype)
                        applied = (actor['means'][selected].double()-before.double()).numpy()
                        actual_vectors = applied @ basis*scale
                        actual = np.mean(actual_vectors, axis=0)
                        error = float(np.max(np.abs(actual_vectors-offset)))
                        if error > .001:
                            raise ValueError(f'Metric translation numerical error >1 mm at frame {index}: {error}')
                    frame_audit.append({'operation': 'translate', 'target_id': ident, 'gaussians': len(selected),
                        'progress': progress, 'requested_offset_m': offset.tolist(),
                        'actual_offset_m': actual.tolist() if actual is not None else None,
                        'max_point_error_m': error, 'rigid_translation': True})
            keep = np.ones(original_count, dtype=bool)
            if binding:
                keep[targets['selected']] = False
            for key in FIELDS:
                if not torch.equal(actor[key][:original_count][keep], original_actor[key][keep]):
                    raise ValueError(f'Non-target original Gaussian {key} changed')
                if key != 'means' and not torch.equal(actor[key][:original_count], original_actor[key]):
                    raise ValueError(f'Translation changed original Gaussian {key}')
            torch.save(actor, staging / 'dynamic' / f'{index:06d}.pt')
            files.append(str(destination / 'dynamic' / f'{index:06d}.pt'))
            rows.append({'frame': index, 'timestamp_seconds': float(times[index]),
                         'original_non_targets_unchanged': True, 'operations': frame_audit})
        for op in translations:
            executed = [entry for row in rows for entry in row['operations']
                        if entry.get('operation') == 'translate' and entry['target_id'] == op['target_id'] and entry['gaussians']]
            if len(executed) < 3 or not any(entry['progress'] >= 1-1e-8 for entry in executed):
                raise ValueError(f"Target {op['target_id']} not sufficiently observed through its trajectory endpoint")
        checkpoint['observed_dynamic']['config']['actor_files'] = files
        road_repair=None
        if config.get('road_geometry')=='planar_proxy':
            from .road_proxy import repair_road
            with np.load(original/'cameras.npz',allow_pickle=False) as cameras:
                road_repair=repair_road(checkpoint['background']['state_dict'],workspace,
                                        poses,cameras['K'],times,calibration['road_plane'])
        background = checkpoint['background']['config']
        for key, filename in [('neural_shader_file', 'background_shader.pt'),
                              ('neural_states_file', 'background_states.pt'), ('sky_environment_file', 'sky_environment.json')]:
            if key in background:
                shutil.copy2(background[key], staging / 'background' / filename)
                background[key] = str(destination / 'background' / filename)
        ckpt = destination / 'background' / f'{name}.ckpt'
        torch.save(checkpoint, staging / 'background' / f'{name}.ckpt')
        shutil.copy2(original / 'cameras.npz', staging / 'cameras.npz')
        edited = {**manifest, 'asset_id': name, 'asset_ckpt': str(ckpt),
                  'camera_file': str(destination / 'cameras.npz'), 'edit_operation': 'metric_edit',
                  'metric_calibration': {'path': str(calibration_path), 'sha256': digest(calibration_path),
                                         'meters_per_native_unit': scale},
                  'inserted_objects': metadata, 'metric_calibrated': True,
                  'road_geometry_repair':road_repair,
                  'world_units': 'native; metric conversion through calibration, checkpoint not rescaled'}
        report = {'backend': 'ufo', 'operation': 'metric_edit', 'profile': 'rigid_metric',
                  'asset_id': name, 'asset_root': str(output_root), 'asset_manifest': str(destination / 'ASSET.json'),
                  'audit_path': str(destination / 'EDIT_AUDIT.json'), 'frame_count': len(times),
                  'render_complete': False,
                  'job_sha256': job_hash, 'units': 'meters', 'calibration_path': str(calibration_path),
                  'meters_per_native_unit': scale, 'reference_frame': 'road_at_edit_start',
                  'calibration_validation_scope': calibration.get('validation_scope'),
                  'reference_frame_index': reference_index, 'reference_timestamp_seconds': float(times[reference_index]),
                  'road_basis_flu_world': basis.tolist(), 'road_origin_native': origin.tolist(),
                  'target_binding': binding, 'inserted_objects': metadata, 'frames': rows,
                  'road_geometry_repair':road_repair,
                  'accuracy_note': calibration['accuracy_note'],
                  'limitations': ['Local planar road, not lane or collision validation.',
                                 'Observed original vehicles remain incomplete single-view surfaces; rigid translation can expose artifacts.',
                                 'No background inpainting, cast-shadow generation, or physical accuracy certification.']}
        (staging / 'ASSET.json').write_text(json.dumps(edited, indent=2, allow_nan=False))
        (staging / 'EDIT_AUDIT.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(destination)
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def execute(reconstruction: dict, config: dict, output_root: Path, *, render=True, worldengine_root: Path | None = None) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / '.metric_edit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = build_edit(reconstruction, config, output_root)
        job = output_root / result['asset_id']
        job.mkdir(exist_ok=True)
        (job / 'edit_config.json').write_text(json.dumps(config, indent=2, allow_nan=False))
        done = job / 'result.json'
        cached = json.loads(done.read_text()) if done.is_file() else {}
        if render:
            from .metric_edit import render_asset
            from .runtime import resources
            from .video_compare import compare_videos
            edited = job / 'edited.mp4'
            if not (cached.get('render_complete') and cached.get('job_sha256') == result['job_sha256'] and edited.is_file()
                    and edited.stat().st_size and cached.get('rendered_video_sha256') == digest(edited)):
                video = render_asset(output_root.resolve(), result['asset_id'], edited,
                                     worldengine_root=(worldengine_root or resources()['worldengine']).resolve())
                result['frame_count'] = video['frames']
            else:
                result['frame_count'] = cached['frame_count']
            result.update(rendered_video=str(edited), rendered_video_sha256=digest(edited), render_complete=True)
            # Preserve a completed GPU render even if comparison encoding fails.
            done.write_text(json.dumps(result, indent=2, allow_nan=False))
            comparison = job / 'before_after.mp4'
            compare_videos(Path(reconstruction['rendered_video']), edited, comparison)
            result.update(rendered_video=str(edited), comparison_video=str(comparison), render_complete=True)
        elif cached.get('render_complete') and cached.get('job_sha256') == result['job_sha256']:
            for key in ('rendered_video', 'comparison_video', 'render_complete', 'rendered_video_sha256'):
                if key in cached:
                    result[key] = cached[key]
        (job / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        result['result_path'] = str(job / 'result.json')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    calibration = sub.add_parser('calibrate', help='Estimate/cache scene scale and road plane; does not reconstruct')
    calibration.add_argument('--reconstruction-json', type=Path, required=True)
    calibration.add_argument('--output', type=Path, required=True)
    calibration.add_argument('--reference', type=Path, help='Known measured distance reference JSON; otherwise offline MoGe-2')
    calibration.add_argument('--road-plane', type=Path, help='Optional measured native world road-plane JSON')
    calibration.add_argument('--force', action='store_true')
    calibration.add_argument('--method', choices=['depth_fit','frontend_metric'], default='depth_fit',
                             help='Strict depth agreement, or verify already-established estimated metric frontend units')
    calibration.add_argument('--road-model',choices=['local_depth','motion_near_depth'],default='local_depth')
    editing = sub.add_parser('edit', help='Insert known-size assets and/or rigidly translate targets in meters')
    editing.add_argument('--reconstruction-json', type=Path, required=True)
    editing.add_argument('--config', type=Path, required=True)
    editing.add_argument('--output-root', type=Path)
    editing.add_argument('--assets-only', action='store_true', help='Build assets/audit on CPU without GPU rendering')
    inspection = sub.add_parser('inspect-asset', help='Validate asset units, axes, grounding and Gaussian fields on CPU')
    inspection.add_argument('--asset-manifest', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'inspect-asset':
        from .metric_assets import load_asset
        _, info = load_asset(args.asset_manifest.expanduser().resolve())
        print(json.dumps(info, ensure_ascii=False, allow_nan=False), flush=True)
        return
    reconstruction = read_reconstruction(args.reconstruction_json.resolve())
    if args.command == 'calibrate':
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = calibrate(Path(reconstruction['workspace']), Path(reconstruction['asset_root']), args.output,
                               reference=args.reference, road_plane=args.road_plane, force=args.force, method=args.method,
                               road_model=args.road_model)
    else:
        config = resolve_paths(json.loads(args.config.read_text()), args.config.resolve().parent)
        output = args.output_root or Path(reconstruction['workspace']) / 'metric_edits_v1'
        result = execute(reconstruction, config, output.resolve(), render=not args.assets_only)
    # Full per-frame measurements remain in EDIT_AUDIT.json, not in each LLM turn.
    summary = {key: value for key, value in result.items() if key != 'frames'}
    if summary.get('road_geometry_repair'):
        summary['road_geometry_repair']={key:value for key,value in summary['road_geometry_repair'].items() if key!='frames'}
    if 'frames' in result:
        errors = [op['max_point_error_m'] for row in result['frames'] for op in row['operations']
                  if op.get('max_point_error_m') is not None]
        summary['max_execution_error_m'] = max(errors, default=0.)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
