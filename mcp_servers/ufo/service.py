"""Run the pinned UFO video entry and verify its native WorldEngine artifacts."""

from __future__ import annotations

import argparse
import json
import hashlib
import math
import os
import re
import subprocess
import uuid
from pathlib import Path

DEFAULT_ENTRY = Path(
    "/inspire/hdd3/project/intelligent-driving-agent/public/workspace/yx/"
    "ufo_video_entry_20261012"
)
SCENE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOGE_ROOT = Path(
    "/inspire/hdd3/project/intelligent-driving-agent/public/workspace/yx/"
    "moge_depth_full_20260922/source"
)
DEFAULT_MOGE_CHECKPOINT = Path(
    "/inspire/hdd3/project/intelligent-driving-agent/public/workspace/yx/"
    "adagent_posefree/checkpoints/moge-2-vitl/model.pt"
)
REQUESTED_MODEL_SOURCE = Path(
    "/inspire/hdd/global_user/guoluosong-253108120129/yx-ufo/"
    "e2_front352_20260930/source"
)
REFERENCE_MODEL_SOURCE = Path(
    "/inspire/hdd3/project/intelligent-driving-agent/public/workspace/yx/"
    "exufo_eval_meeting_20261008/source"
)
REQUESTED_CHECKPOINT = Path(
    "/inspire/hdd/global_user/guoluosong-253108120129/yx-ufo/"
    "e2_front352_20260930/runs/e2_front352_posefree_moge_full/"
    "waymo_2s_100k_seed17/checkpoints/ckpt_099999.pth"
)


def _verify_model_implementation() -> None:
    """The pinned launcher imports the older source path; require byte parity."""
    requested = REQUESTED_MODEL_SOURCE / "ufo"
    reference = REFERENCE_MODEL_SOURCE / "ufo"
    files = sorted(path.relative_to(requested) for path in requested.rglob("*.py"))
    if not files or not reference.is_dir():
        raise FileNotFoundError("requested or reference UFO model source is missing")
    for relative in files:
        expected = requested / relative
        actual = reference / relative
        if not actual.is_file() or hashlib.sha256(expected.read_bytes()).digest() != hashlib.sha256(actual.read_bytes()).digest():
            raise ValueError(f"reference UFO model differs from requested source: {relative}")
    extra = {path.relative_to(reference) for path in reference.rglob("*.py")} - set(files)
    if extra:
        raise ValueError("reference UFO model contains Python files absent from requested source")


def _first_40_frames(video: Path, root: Path, name: str) -> Path:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", str(video)],
        capture_output=True, text=True,
    )
    if probe.returncode:
        raise RuntimeError("cannot count UFO video frames: " + probe.stderr[-1000:])
    try:
        count = int(probe.stdout.strip())
    except ValueError as exc:
        raise ValueError("ffprobe did not report a numeric video frame count") from exc
    if count < 40:
        raise ValueError("UFO Agent reconstruction requires at least 40 video frames")
    if count == 40:
        return video
    destination = root / "_inputs" / f"{name}_first40.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-frames:v", "40",
         "-fps_mode", "passthrough", "-an", "-c:v", "libx264", "-crf", "18",
         str(destination)], capture_output=True, text=True,
    )
    if encoded.returncode or not destination.is_file():
        raise RuntimeError("cannot prepare first 40 UFO frames: " + encoded.stderr[-1000:])
    return destination


def reconstruct(video_path: str, *, scene_name: str | None = None, gpu: int = 0,
                entry_root: str | Path | None = None,
                output_root: str | Path | None = None, resume: bool = False) -> dict:
    """Run UFO without treating its native coordinates as metric coordinates."""
    video = Path(video_path).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"UFO input video not found: {video}")
    entry = Path(entry_root or os.environ.get("UFO_ENTRY_ROOT") or DEFAULT_ENTRY).expanduser().resolve()
    launcher = entry / "run_video.sh"
    if not launcher.is_file():
        raise FileNotFoundError(f"UFO launcher not found: {launcher}")
    if type(gpu) is not int or gpu < 0:
        raise ValueError("UFO gpu must be a non-negative integer")
    offline = os.environ.get('UFO_RUNTIME', 'offline') != 'legacy'
    if not offline:
        _verify_model_implementation()
    name = scene_name or f"adagent_{uuid.uuid4().hex[:16]}"
    if not SCENE_NAME.fullmatch(name):
        raise ValueError("UFO scene_name must contain only letters, digits, _ or -")
    default_root = PROJECT_ROOT / 'outputs/ufo_reconstructions' if offline else entry / 'workspaces'
    root = Path(output_root or os.environ.get("UFO_OUTPUT_ROOT") or default_root).expanduser().resolve()
    workspace = root / name
    # The offline pipeline decodes exactly 40 original frames itself.
    prepared_video = video if offline else _first_40_frames(video, root, name)
    if offline:
        from .runtime import python_path, offline_env
        command = [str(python_path()), '-m', 'mcp_servers.ufo.offline_pipeline',
                   '--video', str(prepared_video), '--workspace', str(workspace), '--gpu', str(gpu)]
        if resume:
            command.append('--resume')
        workspace.mkdir(parents=True, exist_ok=True)
        log_path = workspace / 'offline_reconstruction.log'
        print(f'UFO reconstruction log: {log_path}', flush=True)
        with log_path.open('a') as log:
            with subprocess.Popen(command, cwd=PROJECT_ROOT, env=offline_env(gpu),
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                returncode = process.wait()
        if returncode:
            raise RuntimeError(f'UFO offline reconstruction failed; see {log_path}\n'
                               + log_path.read_text(errors='replace')[-4000:])
        completed = subprocess.CompletedProcess(command, returncode)
    else:
        command = ["bash", str(launcher), "--baseline", "worldengine-time-v2",
                   "--video", str(prepared_video), "--name", name, "--gpu", str(gpu),
                   "--output-root", str(root)]
        completed = subprocess.run(command, cwd=entry, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(
            f"UFO reconstruction failed ({completed.returncode}): "
            + (completed.stderr or completed.stdout)[-4000:]
        )
    output = workspace / "output" / "worldengine_time_v2"
    asset = (workspace / 'worldengine_asset/assets' / name if offline else
             entry.parent / "WorldEngine" / "data" / "ufo_time_v2" / "assets" / name)
    required = {
        "ready": workspace / "baselines" / "worldengine_time_v2" / "READY.json",
        "done": output / "DONE.json",
        "rendered_video": output / "reconstruction.mp4",
        "asset_manifest": asset / "ASSET.json",
        "camera_file": asset / "cameras.npz",
        "model_build": workspace / "reconstruction" / "000" / "pred100k" / "BUILD.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError("UFO finished without required artifacts: " + ", ".join(missing))
    manifest = json.loads(required["asset_manifest"].read_text(encoding="utf-8"))
    done = json.loads(required["done"].read_text(encoding="utf-8"))
    build = json.loads(required["model_build"].read_text(encoding="utf-8"))
    if manifest.get("asset_id") != name or manifest.get("version") != "ufo-worldengine-time-v2":
        raise ValueError("UFO asset identity or version differs from requested reconstruction")
    if not done.get("actual_WorldEngine_renderer") or not done.get("actual_WorldEngine_asset_manager"):
        raise ValueError("UFO DONE.json does not verify native WorldEngine rendering")
    if not isinstance(done.get("frames"), int) or done["frames"] < 40:
        raise ValueError("UFO output has fewer than 40 verified frames")
    if (build.get("checkpoint_step") != 99999 or not build.get("weights_unchanged")
            or Path(build.get("checkpoint", "")) != REQUESTED_CHECKPOINT):
        raise ValueError("UFO reconstruction did not use the pinned MoGe-trained 100k checkpoint")
    # The pinned reference explicitly says scale1 is native reconstruction
    # units, with no external metre calibration. Never imply metric support.
    return {
        "backend": "ufo_worldengine_time_v2",
        "video_path": str(prepared_video),
        "source_video_path": str(video),
        "scene_name": name,
        "workspace": str(workspace),
        "output_dir": str(output),
        "rendered_video": str(required["rendered_video"]),
        "asset_manifest": str(required["asset_manifest"]),
        "camera_file": str(required["camera_file"]),
        "asset_root": str(asset.parent.parent),
        "frame_count": done["frames"],
        "checkpoint": build["checkpoint"],
        "model_source": str(REQUESTED_MODEL_SOURCE),
        "model_source_verified_by": ('explicit model source override in offline stage_runner' if offline else
                                     "byte-identical UFO Python package used by reference launcher"),
        "offline": offline,
        "metric_calibrated": False,
        "metric_calibration_reason": "UFO ASSET.json retains native reconstruction scale without external metre calibration",
    }


def _metric_offsets_from_config(config_path: Path, output_path: Path, frames: int,
                                expected_instance_id: int) -> None:
    from mcp_servers.dggt.adapter.coordinates import _interpolate_waypoints
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rigid = config.get("Nodes", {}).get("RigidNodes", {})
    if set(rigid) != {"trajectory_edit"}:
        raise ValueError("UFO metric edit currently supports one trajectory_edit operation")
    trajectory = rigid["trajectory_edit"]
    ids = trajectory.get("instance_id")
    groups = trajectory.get("waypoints")
    methods = trajectory.get("method", ["linear"])
    if not isinstance(ids, list) or len(ids) != 1 or not isinstance(groups, list) or len(groups) != 1:
        raise ValueError("UFO metric edit requires one selected instance and one waypoint group")
    if ids != [expected_instance_id]:
        raise ValueError("UFO metric edit changed the selected target")
    if not isinstance(methods, list) or len(methods) != 1 or methods[0] not in {"linear", "catmull-rom"}:
        raise ValueError("UFO metric edit requires one explicit interpolation method")
    values = _interpolate_waypoints(groups[0], methods[0], frames)
    if any(len(vector) != 3 or any(type(value) not in (int, float) or not math.isfinite(value) for value in vector)
           for vector in values):
        raise ValueError("UFO waypoints must be [forward,left,up] metre vectors")
    output_path.write_text(json.dumps({"units": "metres",
                                       "coordinate_system": "camera_local_forward_left_up",
                                       "offsets": values}, indent=2) + "\n", encoding="utf-8")


def edit_metric(reconstruction: dict, *, edit_config_path: str,
                target_selection_path: str, gpu: int = 0,
                output_asset_root: str | None = None) -> dict:
    """Calibrate and edit an existing UFO scene using one selected vehicle."""
    from tools.tool05_instance_discovery.selection import load_bound_instances
    if reconstruction.get("backend") != "ufo_worldengine_time_v2":
        raise ValueError("metric UFO edit requires a UFO reconstruction result")
    if not isinstance(edit_config_path, str) or not edit_config_path.strip():
        raise ValueError("metric UFO edit requires edit_config_path")
    if not isinstance(target_selection_path, str) or not target_selection_path.strip():
        raise ValueError("metric UFO edit requires target_selection_path")
    workspace = Path(reconstruction["workspace"]).resolve()
    asset_root = Path(reconstruction["asset_root"]).resolve()
    selected = Path(target_selection_path).resolve()
    config = Path(edit_config_path).resolve()
    if not selected.is_file() or not config.is_file():
        raise FileNotFoundError("UFO metric edit needs an existing target selection and edit config")
    if reconstruction.get("frame_count") != 40:
        raise ValueError("Agent UFO metric editing currently requires exactly 40 video frames")
    if type(gpu) is not int or gpu < 0:
        raise ValueError("UFO gpu must be a non-negative integer")
    binding, _ = load_bound_instances(selected)
    root = Path(output_asset_root or workspace / "metric_edits").resolve()
    name = f"{workspace.name}_metric_{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    offsets = root / f"{name}_offsets.json"
    _metric_offsets_from_config(config, offsets, 40, binding["selected_instance_id"])
    moge_root = Path(os.environ.get("UFO_MOGE_ROOT") or DEFAULT_MOGE_ROOT).resolve()
    moge_checkpoint = Path(os.environ.get("UFO_MOGE_CHECKPOINT") or DEFAULT_MOGE_CHECKPOINT).resolve()
    wrapper = moge_root / "python_offline.sh"
    worldengine = Path(os.environ.get("UFO_WORLDENGINE_ROOT") or DEFAULT_ENTRY.parent / "WorldEngine").resolve()
    command = [str(wrapper), "-m", "mcp_servers.ufo.metric_edit",
               "--workspace", str(workspace), "--asset-root", str(asset_root),
               "--output-asset-root", str(root), "--new-name", name,
               "--target-selection", str(selected), "--offsets-json", str(offsets),
               "--moge-source", str(moge_root / "vendor_moge"),
               "--moge-checkpoint", str(moge_checkpoint),
               "--worldengine-root", str(worldengine), "--gpu", str(gpu)]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError("UFO metric edit failed: " + (completed.stderr or completed.stdout)[-4000:])
    try:
        result = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("UFO metric edit did not report a result") from exc
    if not Path(result["rendered_video"]).is_file():
        raise RuntimeError("UFO metric edit did not produce a video")
    return result


def _metric_worker(reconstruction: dict, arguments: list[str], *, gpu: int, log_path: Path) -> dict:
    from .runtime import python_path, offline_env
    if reconstruction.get('backend') != 'ufo_worldengine_time_v2' or reconstruction.get('frame_count') != 40:
        raise ValueError('Metric control requires a successful 40-frame UFO reconstruction result')
    if type(gpu) is not int or gpu < 0:
        raise ValueError('gpu must be a nonnegative integer')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep each request immutable for concurrent jobs in the same workspace.
    request = log_path.parent / f'reconstruction_{uuid.uuid4().hex}.json'
    request.write_text(json.dumps(reconstruction, ensure_ascii=False))
    command = [str(python_path()), '-u', '-m', 'mcp_servers.ufo.metric_control', *arguments,
               '--reconstruction-json', str(request)]
    print(f'UFO metric log: {log_path}', flush=True)
    last = ''
    try:
        with log_path.open('a') as log:
            with subprocess.Popen(command, cwd=PROJECT_ROOT, env=offline_env(gpu), stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if line.strip():
                        last = line
                    if not line.startswith('{'):
                        print(line, end='', flush=True)
                code = process.wait()
        if code:
            raise RuntimeError(f'UFO metric control failed; see {log_path}\n' + log_path.read_text(errors='replace')[-4000:])
        return json.loads(last)
    finally:
        request.unlink(missing_ok=True)


def calibrate_metric_scene(reconstruction: dict, *, output_path: str | None = None,
                           reference_path: str | None = None, road_plane_path: str | None = None,
                           gpu: int = 0, method: str = 'depth_fit', road_model: str = 'local_depth') -> dict:
    workspace = Path(reconstruction['workspace']).resolve()
    default_name = 'metric_scene_frontend_v1.json' if method == 'frontend_metric' else 'metric_scene_v1.json'
    output = Path(output_path or workspace / default_name).expanduser().resolve()
    if method not in {'depth_fit','frontend_metric'}:
        raise ValueError('Metric calibration method must be depth_fit or frontend_metric')
    arguments = ['calibrate', '--output', str(output), '--method', method, '--road-model', road_model]
    if reference_path:
        arguments.extend(['--reference', str(Path(reference_path).expanduser().resolve())])
    if road_plane_path:
        arguments.extend(['--road-plane', str(Path(road_plane_path).expanduser().resolve())])
    result = _metric_worker(reconstruction, arguments, gpu=gpu, log_path=output.with_suffix('.log'))
    return {'calibration_path': str(output), 'meters_per_native_unit': result['meters_per_native_unit'],
            'scale_fit': result['scale_fit'], 'road_plane': result['road_plane'], 'accuracy_note': result['accuracy_note'],
            'validation_scope': result.get('validation_scope')}


def edit_metric_control(reconstruction: dict, *, edit_config_path: str,
                        target_selection_path: str | None = None, gpu: int = 0,
                        output_root: str | None = None, assets_only: bool = False) -> dict:
    from .metric_config import validate, resolve_paths
    config_path = Path(edit_config_path).expanduser().resolve()
    config = validate(resolve_paths(json.loads(config_path.read_text()), config_path.parent))
    if target_selection_path and (not config.get('target_selection_path') or
        Path(config['target_selection_path']).resolve() != Path(target_selection_path).expanduser().resolve()):
        raise ValueError('Editor target differs from metric config')
    root = Path(output_root or Path(reconstruction['workspace']) / 'metric_edits_v1').expanduser().resolve()
    arguments = ['edit', '--config', str(config_path), '--output-root', str(root)]
    if assets_only:
        arguments.append('--assets-only')
    log_path = root / f'editor_{hashlib.sha256(config_path.read_bytes()).hexdigest()[:12]}.log'
    return _metric_worker(reconstruction, arguments, gpu=gpu, log_path=log_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and verify the UFO reconstruction adapter")
    parser.add_argument("--video", required=True, help="Absolute path to a single front-view video")
    parser.add_argument("--scene-name", help="Unique output scene name")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--entry-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    result = reconstruct(args.video, scene_name=args.scene_name, gpu=args.gpu,
                         entry_root=args.entry_root, output_root=args.output_root, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def edit_lane_change(reconstruction: dict, *, track_id: int | None = None,
                     direction: str = 'right', lateral_distance_native: float | None = None,
                     gpu: int = 0, targets_only: bool = False, target_selection_path: str | None = None,
                     start_fraction: float = .1, end_fraction: float = .9, vehicle_widths: float = 1.2) -> dict:
    from .runtime import python_path, offline_env, resources
    if reconstruction.get('backend') != 'ufo_worldengine_time_v2' or reconstruction.get('frame_count') != 40:
        raise ValueError('UFO lane-change test requires a 40-frame UFO reconstruction result')
    if type(gpu) is not int or gpu < 0 or (track_id is not None and (type(track_id) is not int or track_id <= 0)):
        raise ValueError('gpu must be nonnegative and track_id must be a positive integer')
    if direction not in ('left', 'right'):
        raise ValueError('lane change direction must be left or right')
    workspace = Path(reconstruction['workspace']).resolve()
    import hashlib
    identity = {'profile': 'v2', 'track_id': track_id, 'direction': direction,
                'lateral_distance_native': lateral_distance_native, 'start_fraction': start_fraction,
                'end_fraction': end_fraction, 'vehicle_widths': vehicle_widths,
                'binding': Path(target_selection_path).read_text() if target_selection_path else None}
    suffix = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:10]
    root = workspace / f'lane_change_{direction}_v2_{suffix}'
    root.mkdir(parents=True, exist_ok=True)
    command = [str(python_path()), '-m', 'mcp_servers.ufo.lane_edit',
               '--workspace', str(workspace), '--asset-root', reconstruction['asset_root'],
               '--output-root', str(root), '--worldengine-root', str(resources()['worldengine']),
               '--new-name', f'{workspace.name}_lane_{direction}_v2_{suffix}', '--direction', direction,
               '--start-fraction', str(start_fraction), '--end-fraction', str(end_fraction),
               '--vehicle-widths', str(vehicle_widths)]
    if target_selection_path:
        command.extend(['--target-selection-path', str(Path(target_selection_path).resolve())])
    if track_id is not None:
        command.extend(['--track-id', str(track_id)])
    if lateral_distance_native is not None:
        if not math.isfinite(lateral_distance_native) or lateral_distance_native <= 0:
            raise ValueError('lateral_distance_native must be finite and positive')
        command.extend(['--lateral-distance-native', str(lateral_distance_native)])
    if targets_only:
        command.append('--targets-only')
    log_path = root / 'editor.log'
    print(f'UFO lane-change log: {log_path}', flush=True)
    with log_path.open('a') as log:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, env=offline_env(gpu),
                                   stdout=log, stderr=subprocess.STDOUT, text=True)
    if completed.returncode:
        raise RuntimeError(f'UFO lane-change failed; see {log_path}\n' + log_path.read_text(errors='replace')[-4000:])
    result = json.loads((root / 'result.json').read_text())
    if not targets_only:
        comparison = root / 'before_after.mp4'
        with log_path.open('a') as log:
            compared = subprocess.run([str(python_path()), '-m', 'mcp_servers.ufo.video_compare',
                                       '--before', reconstruction['rendered_video'],
                                       '--after', result['rendered_video'], '--out', str(comparison)],
                                      cwd=PROJECT_ROOT, env=offline_env(gpu), stdout=log, stderr=subprocess.STDOUT)
        if compared.returncode or not comparison.is_file():
            raise RuntimeError(f'UFO comparison video failed; see {log_path}\n'
                               + log_path.read_text(errors='replace')[-4000:])
        result['comparison_video'] = str(comparison)
        (root / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == "__main__":
    main()
