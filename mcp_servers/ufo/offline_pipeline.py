"""Run the reference time-v2 stages in ufoex with local weights only."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .runtime import resources, offline_env, python_path


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def preflight(video: Path, *, check_imports: bool = False) -> dict:
    started = time.monotonic()
    r = resources()
    if Path(sys.prefix).resolve() != r['env']:
        raise RuntimeError(f'Expected ufoex prefix {r["env"]}, got {sys.prefix}')
    from .service import REQUESTED_CHECKPOINT
    required = [video, python_path(), REQUESTED_CHECKPOINT,
                r['custom'] / 'run_config.json', r['source'] / 'ufo/models/archs/small.py',
                r['install'] / 'grounded_sam/checkpoints/sam2.1_hiera_large.pt',
                r['install'] / 'grounding_dino/config.json',
                r['install'] / 'segformer_assets/pytorch_model.bin',
                r['install'] / 'frontend/checkpoints/vggt_omega_1b_512.pt',
                r['install'] / 'frontend/checkpoints/moge-2-vitl/model.pt']
    for name in ('prepare_scene.py', 'build_scene100.py', 'assemble_scene.py'):
        required.append(r['legacy'] / name)
    for name in ('prepare_sam.py', 'prepare_adapter.py'):
        required.append(r['install'] / 'video_tools' / name)
    for name in ('export_ufo_background.py', 'package_ufo_asset.py', 'render_ufo_replay.py'):
        required.append(r['worldengine'] / 'scripts' / name)
    missing = [str(path) for path in required if not path.is_file()]
    dino = r['install'] / 'grounding_dino'
    if not any((dino / name).is_file() for name in ('model.safetensors', 'pytorch_model.bin')):
        missing.append(str(dino / '{model.safetensors,pytorch_model.bin}'))
    if missing:
        raise FileNotFoundError('Offline assets missing:\n' + '\n'.join(missing))
    sys.path[:0] = [str(r['source']), str(r['install'] / 'frontend/vendor/omega'),
                    str(r['install'] / 'frontend/vendor/moge')]
    # A spec lookup checks installation without running package __init__ or
    # loading large shared libraries from the shared filesystem.
    installed = {}
    for name in ('torch', 'torchvision', 'numpy', 'av', 'cv2', 'transformers', 'scipy',
                 'hydra', 'timm', 'gsplat', 'utils3d', 'sam2'):
        spec = importlib.util.find_spec(name)
        if spec is None:
            raise ModuleNotFoundError(f'Offline environment is missing {name}')
        installed[name] = spec.origin
    versions, import_seconds = {}, {}
    input_info = None
    if check_imports:
        import faulthandler
        for name in ('torch', 'torchvision', 'numpy', 'av', 'cv2', 'transformers', 'scipy',
                 'hydra', 'timm', 'gsplat', 'utils3d', 'utils3d.torch', 'sam2', 'sam2.build_sam',
                 'sam2.sam2_image_predictor', 'adagent_posefree.backend.worker_stage',
                 'vggt_omega.models', 'moge.model.v2', 'ufo.models.archs.small'):
            print('OFFLINE_IMPORT', name, flush=True)
            tick = time.monotonic()
            faulthandler.dump_traceback_later(60, repeat=True)
            try:
                module = importlib.import_module(name)
            finally:
                faulthandler.cancel_dump_traceback_later()
            versions[name] = getattr(module, '__version__', 'import OK')
            import_seconds[name] = round(time.monotonic() - tick, 3)
            print(f'OFFLINE_IMPORT_DONE {name} seconds={import_seconds[name]}', flush=True)
        import av
        av.codec.Codec('libx264', 'w')
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            input_info = {'width': stream.width, 'height': stream.height,
                          'rate': str(stream.average_rate)}
    return {'status': 'offline_preflight_passed', 'python': sys.executable,
            'environment': str(r['env']), 'input': str(video), 'video': input_info,
            'mode': 'full_imports' if check_imports else 'paths_only', 'module_locations': installed,
            'imports': versions, 'import_seconds': import_seconds,
            'seconds': round(time.monotonic() - started, 3),
            'local_assets_checked': len(required), 'gpu_inference_started': False}


def extract_input(video: Path, workspace: Path) -> None:
    import av
    av.codec.Codec('libx264', 'w')
    from fractions import Fraction
    folder = workspace / 'input'
    rgb = folder / 'rgb/video/0'
    rgb.mkdir(parents=True, exist_ok=True)
    stamps, shape = [], None
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate
        for index, frame in enumerate(container.decode(stream)):
            if index >= 40:
                break
            if frame.pts is None:
                raise ValueError('Video frame has no timestamp')
            timestamp = Fraction(frame.pts) * frame.time_base
            if stamps and timestamp <= stamps[-1]:
                raise ValueError('Video timestamps must increase')
            image = frame.to_image().convert('RGB')
            current_shape = (image.height, image.width)
            if shape is not None and current_shape != shape:
                raise ValueError('Video frame dimensions change')
            shape = current_shape
            image.save(rgb / f'{index:06d}.jpg', quality=100, subsampling=0)
            stamps.append(timestamp)
    if len(stamps) != 40:
        raise ValueError('Agent UFO test requires at least 40 video frames')
    timestamps = [float(value - stamps[0]) for value in stamps]
    fps = 39 / timestamps[-1]
    manifest = {'scene_name': 'video', 'source': str(video), 'num_frames': 40,
                'fps': fps, 'nominal_rate': str(rate), 'frame_timestamps': timestamps,
                'output_pts': [round(t * 1e6) for t in timestamps], 'output_time_base': [1, 1000000],
                'duration_seconds': timestamps[-1] + float(1 / rate), 'image_size': shape}
    scene = {'scene_name': 'video', 'camera_ids': ['0'], 'reference_camera': '0', 'fps': fps,
             'frames': [{'frame_id': i, 'timestamp': t, 'images': {'0': str(rgb / f'{i:06d}.jpg')}}
                        for i, t in enumerate(timestamps)]}
    write_json(folder / 'input_manifest.json', manifest)
    write_json(folder / 'camera_manifest.json', {'schema_version': 1, 'scenes': [scene]})
    write_json(folder / 'geometry_config.json', {'output_image_size': [352, 520]})


def run(video: Path, workspace: Path, gpu: int, *, resume: bool = False) -> None:
    import fcntl
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / '.offline_pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _run_locked(video, workspace, gpu, resume=resume)


def _run_locked(video: Path, workspace: Path, gpu: int, *, resume: bool = False) -> None:
    from .resume import check_resume_contract, archive_contract, stage_complete
    r = resources()
    python = str(python_path())
    env = offline_env(gpu)
    env['UFO_WORKSPACE'] = str(workspace)
    scripts = [r['legacy'] / name for name in ('prepare_scene.py', 'build_scene100.py', 'assemble_scene.py')]
    scripts += list(Path(__file__).parent.glob('*.py'))
    scripts += list((r['source'] / 'ufo').rglob('*.py'))
    scripts += [r['worldengine'] / 'scripts' / name for name in
                ('export_ufo_background.py', 'package_ufo_asset.py', 'render_ufo_replay.py')]
    from .service import REQUESTED_CHECKPOINT
    weight_stat = REQUESTED_CHECKPOINT.stat()
    contract = {'video': str(video), 'sha256': digest(video), 'python': python,
                'source': str(r['source']), 'frames': 40,
                'checkpoint': {'path': str(REQUESTED_CHECKPOINT), 'size': weight_stat.st_size,
                               'mtime_ns': weight_stat.st_mtime_ns},
                'code': {str(p): digest(p) for p in scripts}}
    contract_path = workspace / 'OFFLINE_CONTRACT.json'
    if contract_path.exists():
        previous = json.loads(contract_path.read_text())
        changed = check_resume_contract(previous, contract, allow_adapter_updates=resume)
        if changed:
            archive_contract(contract_path, previous, changed)
            print(f'RESUME adapter update: {len(changed)} files; preserving completed stage outputs', flush=True)
    write_json(contract_path, contract)
    log_root = workspace / 'logs/offline'
    log_root.mkdir(parents=True, exist_ok=True)

    def stage(name, command, expected):
        marker = log_root / f'{name}.done.json'
        if stage_complete(marker, expected):
            print('RESUME', name, flush=True)
            return
        print('START', name, flush=True)
        started = time.monotonic()
        log = log_root / f'{name}.log'
        with log.open('a') as output:
            result = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT)
        if result.returncode or not all(path.is_file() for path in expected):
            raise RuntimeError(f'{name} failed; see {log}\n' + log.read_text(errors='replace')[-4000:])
        elapsed = time.monotonic() - started
        write_json(marker, {'seconds': elapsed, 'python': python})
        print(f'DONE {name} seconds={elapsed:.3f}', flush=True)

    input_root = workspace / 'input'
    decoded_files = [input_root / name for name in ('input_manifest.json', 'camera_manifest.json', 'geometry_config.json')]
    decoded_files += [input_root / f'rgb/video/0/{i:06d}.jpg' for i in range(40)]
    if stage_complete(log_root / 'decode.done.json', decoded_files):
        print('RESUME decode', flush=True)
    else:
        print('START decode', flush=True)
        tick = time.monotonic()
        extract_input(video, workspace)
        elapsed = time.monotonic() - tick
        write_json(log_root / 'decode.done.json', {'frames': 40, 'seconds': elapsed})
        print(f'DONE decode seconds={elapsed:.3f}', flush=True)
    stage('cameras', [python, '-m', 'adagent_posefree', 'prepare', '--manifest',
                     str(input_root / 'camera_manifest.json'), '--config', str(input_root / 'geometry_config.json'),
                     '--num-gpus', '1', '--output', str(input_root / 'cameras')],
          [input_root / 'cameras/scenes/video/done.json', input_root / 'cameras/global_aligned/video/omega_pose_override.npz'])
    sam_outputs = [input_root / 'sam.done.json'] + [input_root / f'sam/video/0/mask_data/mask_{i:06d}.npy' for i in range(40)]
    if not stage_complete(log_root / 'sam.done.json', sam_outputs) and (input_root / 'sam').exists():
        if all(p.is_file() for p in sam_outputs):
            write_json(log_root / 'sam.done.json', {'recovered_completed_stage': True})
        else:
            (input_root / 'sam').rename(input_root / f'sam_incomplete_{time.time_ns()}')
    stage('sam', [python, str(r['install'] / 'video_tools/prepare_sam.py'), str(input_root)], sam_outputs)
    stage('adapter', [python, str(r['install'] / 'video_tools/prepare_adapter.py'), str(input_root)],
          [input_root / 'annotations.txt', input_root / 'scene.json', input_root / 'ufo_cameras/video/omega_pose_override.npz']
          + [input_root / f'data/datasets/waymo/video/images/{i:06d}_0.jpg' for i in range(40)])
    scene = workspace / 'reconstruction/000/pred100k'
    scene.mkdir(parents=True, exist_ok=True)
    bootstrap = [python, '-m', 'mcp_servers.ufo.stage_runner']
    stage('semantics', [*bootstrap, str(r['legacy'] / 'prepare_scene.py')],
          [workspace / 'reconstruction/000/PREPARED.json']
          + [workspace / f'reconstruction/000/masks/{i:06d}.npy' for i in range(40)])
    stage('ufo', [*bootstrap, str(r['legacy'] / 'build_scene100.py'), '--sid', '0'],
          [scene / 'BUILD.json', scene / 'EXPORT_VERIFIED.json']
          + [scene / f'sources/{i:06d}.npz' for i in range(40)]
          + [scene / f'reference/{i:06d}.png' for i in range(40)])
    build = json.loads((scene / 'BUILD.json').read_text())
    if (build.get('all_sources') != 40 or build.get('limit') != 0 or not build.get('weights_unchanged')
            or Path(build.get('checkpoint', '')) != REQUESTED_CHECKPOINT
            or not json.loads((scene / 'EXPORT_VERIFIED.json').read_text()).get('passed')):
        raise ValueError('UFO export failed the pinned checkpoint/frame/export contract')
    write_json(scene / 'OFFLINE_MODEL.json', {'source': str(r['source']), 'python': python,
                                             'checkpoint': str(REQUESTED_CHECKPOINT), 'offline': True})
    stage('pool', [*bootstrap, str(r['legacy'] / 'assemble_scene.py'), '--sid', '0', '--camera', 'pred100k'],
          [scene / 'global_blend.pt', scene / 'environment.json'] + [scene / f'frames/{i:06d}.pt' for i in range(40)])
    baseline = workspace / 'baselines/worldengine_time_v2'
    background = baseline / 'background'
    stage('background', [*bootstrap, str(r['worldengine'] / 'scripts/export_ufo_background.py'),
                         '--workspace', str(workspace), '--out', str(background)],
          [background / name for name in ('BG_DONE.json', 'background_shader.pt', 'background_states.pt')])
    asset_root = workspace / 'worldengine_asset'
    asset = asset_root / 'assets' / workspace.name
    stage('package', [python, str(r['worldengine'] / 'scripts/package_ufo_asset.py'),
                      '--sky', '--temporal-profile', 'width20-discrete', '--workspace', str(workspace),
                      '--background', str(background), '--out', str(asset_root)],
          [asset / 'ASSET.json', asset / 'cameras.npz', asset / 'background' / f'{workspace.name}.ckpt']
          + [asset / f'dynamic/{i:06d}.pt' for i in range(40)])
    output = workspace / 'output/worldengine_time_v2'
    stage('render', [python, str(r['worldengine'] / 'scripts/render_ufo_replay.py'), '--workspace', str(workspace),
                     '--asset-root', str(asset_root), '--out', str(output)],
          [output / name for name in ('DONE.json', 'reconstruction.mp4', 'comparison.mp4')])
    write_json(baseline / 'READY.json', {'version': 'ufo-worldengine-time-v2', 'offline': True,
                                        'asset': str(asset / 'ASSET.json'), 'python': python})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--check-imports', action='store_true', help='Also perform slow, exhaustive library imports')
    parser.add_argument('--resume', action='store_true', help='Reuse completed stages after adapter-only repairs')
    args = parser.parse_args()
    report = preflight(args.video.resolve(), check_imports=args.check_imports)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if not args.check_only:
        if args.workspace is None:
            parser.error('--workspace is required for reconstruction')
        run(args.video.resolve(), args.workspace.resolve(), args.gpu, resume=args.resume)


if __name__ == '__main__':
    main()
