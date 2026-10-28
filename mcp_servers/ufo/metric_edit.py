"""Calibrate and edit one observed UFO vehicle in metric units.

Run in the server's dggt_data environment with the UFO/WorldEngine runtime.
The output is a separate asset; the source workspace and asset stay intact.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from fractions import Fraction
from pathlib import Path

from .actor_edit import actor_track_indices
from .binding import bind_selected_track
from .metric import fit_metres_per_native_unit, metric_offset_to_native_world


def moge_input_geometry(rgb, intrinsics):
    """Undo anisotropic video resizing before MoGe's square-pixel camera model.

    Pixel-grid resizing does not change camera-z. Predictions are resized back
    onto the exact source grid before pairing depths. Off-center/skew cameras
    need full rectification and are rejected rather than silently approximated.
    """
    import numpy as np
    from PIL import Image
    if intrinsics is None:
        return rgb, None
    K = np.asarray(intrinsics, dtype=np.float64)
    height, width = rgb.shape[:2]
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]) or abs(K[0, 1])+abs(K[1, 0]) > 1e-5:
        raise ValueError('Metric depth requires finite positive, unskewed pinhole intrinsics')
    if abs(K[0, 2]-width/2) > 1 or abs(K[1, 2]-height/2) > 1:
        raise ValueError('Off-center intrinsics require full image rectification before metric fitting')
    new_height = round(height*K[0, 0]/K[1, 1])
    if not 14 <= new_height <= 4096:
        raise ValueError('Unsupported aspect ratio for metric depth rectification')
    image = np.asarray(Image.fromarray(rgb).resize((width, new_height), Image.Resampling.BICUBIC))
    fov = math.degrees(2*math.atan(width/(2*K[0, 0])))
    return image, fov


def _load_offsets(path: Path, frames: int) -> list[tuple[float, float, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("units") != "metres" or payload.get("coordinate_system") != "camera_local_forward_left_up":
        raise ValueError("UFO offsets must explicitly use metres and camera_local_forward_left_up")
    offsets = payload.get("offsets")
    if not isinstance(offsets, list) or len(offsets) not in (1, frames):
        raise ValueError(f"UFO offsets must have 1 or {frames} vectors")
    vectors = []
    for item in offsets:
        if not isinstance(item, list) or len(item) != 3 or any(
            type(value) not in (int, float) or not math.isfinite(value) for value in item
        ):
            raise ValueError("UFO offset vectors must contain three finite numbers")
        vectors.append(tuple(map(float, item)))
    return vectors * frames if len(vectors) == 1 else vectors


def calibrate_workspace(workspace: Path, *, moge_source: Path, moge_checkpoint: Path,
                        sample_frames: int = 5, depth_cache: Path | None = None,
                        camera_file: Path | None = None, asset_root: Path | None = None) -> dict:
    """Fit the native scene scale to MoGe-2 metric camera-z on static pixels."""
    import numpy as np
    import torch
    from PIL import Image
    from .rendered_depth import RenderedDepthCache

    if asset_root is None:
        raise ValueError('Metric fitting requires asset_root for native rendered expected depth, not Gaussian-center depth')
    camera_file = camera_file or asset_root / 'assets' / workspace.name / 'cameras.npz'
    rendered_cache = RenderedDepthCache(workspace, asset_root,
        (depth_cache or workspace / 'metric_depth_cache') / 'rendered_native_v1')

    model_file = moge_source / "moge" / "model" / "v2.py"
    if not model_file.is_file() or not moge_checkpoint.is_file():
        raise FileNotFoundError("Pinned MoGe-2 source and metric checkpoint are required")
    sys.path.insert(0, str(moge_source))
    from moge.model.v2 import MoGeModel

    model = None  # Loading is unnecessary when verified metric depth caches exist.

    scene = workspace / "reconstruction" / "000" / "pred100k"
    build = json.loads((scene / "BUILD.json").read_text(encoding="utf-8"))
    count = int(build["all_sources"])
    if count < 40:
        raise ValueError("UFO metric edit requires at least 40 source frames")
    indices = sorted(set(round(i * (count - 1) / max(sample_frames - 1, 1))
                         for i in range(sample_frames)))
    # Sample every selected reconstruction window, not only five arbitrary frames.
    representatives = {}
    for frame, window in build.get('selected_window', {}).items():
        representatives.setdefault(int(window), int(frame))
    indices = sorted(set(indices) | set(representatives.values()))
    intrinsics = None
    if camera_file:
        with np.load(camera_file, allow_pickle=False) as cameras:
            intrinsics = cameras['K'].copy()
    all_pairs = []
    frame_scales = []
    for index in indices:
        with np.load(scene / "sources" / f"{index:06d}.npz", allow_pickle=False) as source:
            static = (~source["dynamic"]) & (source["coverage"] < .01)
            if 'classes' in source:
                static &= source['classes'] != 10  # Sky is not metric static support.
            classes = source['classes'].copy() if 'classes' in source else None
        rgb = np.asarray(Image.open(scene / "reference" / f"{index:06d}.png").convert("RGB"))
        if rgb.shape[:2] != static.shape:
            raise ValueError("MoGe RGB and UFO depth grids differ")
        metric_rgb, fov = moge_input_geometry(rgb, intrinsics[index] if intrinsics is not None else None)
        import hashlib
        stat = moge_checkpoint.stat()
        cache_key = hashlib.sha256(json.dumps({'rgb': hashlib.sha256(rgb.tobytes()).hexdigest(),
            'checkpoint': [str(moge_checkpoint.resolve()), stat.st_size, stat.st_mtime_ns],
            'model_source': hashlib.sha256(model_file.read_bytes()).hexdigest(), 'fov_x_degrees': fov,
            'intrinsics': intrinsics[index].tolist() if intrinsics is not None else None,
            'inference_grid': list(metric_rgb.shape[:2]), 'rectification': 'square_pixels_v1',
            'resolution_level': 9, 'depth_type': 'camera_z_meters'}, sort_keys=True).encode()).hexdigest()
        cached = depth_cache / f'{index:06d}_{cache_key}.npz' if depth_cache else None
        if cached and cached.is_file():
            with np.load(cached, allow_pickle=False) as prediction:
                metres, valid = prediction['depth'].copy(), prediction['mask'].copy()
            print(f'RESUME metric depth frame={index}', flush=True)
        else:
            if model is None:
                checkpoint = torch.load(moge_checkpoint, map_location='cpu', weights_only=True)
                if not checkpoint['model_config'].get('scale_head'):
                    raise ValueError('MoGe-2 checkpoint has no metric scale head')
                model = MoGeModel(**checkpoint['model_config'])
                model.load_state_dict(checkpoint['model'], strict=True)
                model = model.cuda().eval()
                del checkpoint
            tensor = torch.from_numpy(metric_rgb.copy()).permute(2, 0, 1).float().div_(255).cuda()
            with torch.inference_mode():
                prediction = model.infer(tensor, resolution_level=9, use_fp16=True,
                                         apply_mask=False, fov_x=fov)
            depth_tensor = prediction['depth'].float()
            mask_tensor = prediction['mask'].float()
            metres = torch.nn.functional.interpolate(depth_tensor[None, None], size=rgb.shape[:2], mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
            valid = torch.nn.functional.interpolate(mask_tensor[None, None], size=rgb.shape[:2], mode='nearest')[0, 0].bool().cpu().numpy()
            if cached:
                cached.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cached, depth=metres, mask=valid)
        native_depth, native_valid = rendered_cache.get(index, rgb.shape[:2])
        if metres.shape != native_depth.shape or valid.shape != native_depth.shape:
            raise ValueError('Cached/predicted metric depth is not on the UFO pixel grid')
        valid &= static & native_valid
        valid &= np.isfinite(native_depth) & np.isfinite(metres)
        valid &= (native_depth > .2) & (native_depth < 400) & (metres > .01)
        native = native_depth[::4, ::4][valid[::4, ::4]]
        metric = metres[::4, ::4][valid[::4, ::4]]
        pairs = list(zip(native.tolist(), metric.tolist()))
        try:
            frame_fit = fit_metres_per_native_unit(pairs, minimum_samples=100,
                                                   maximum_median_relative_error=.15)
        except ValueError as error:
            from .metric_diagnostics import depth_report, write_failure_report
            report = depth_report(native_depth[::4, ::4], metres[::4, ::4], valid[::4, ::4],
                                  classes[::4, ::4] if classes is not None else None)
            report['native_depth_definition'] = 'WorldEngine rendered expected camera_z (RGB+ED), not Gaussian centers'
            directory = (depth_cache / 'rendered_depth_diagnostics' if depth_cache else workspace / 'metric_calibration_diagnostics')
            diagnostic = write_failure_report(directory, index, report, error)
            rendered_cache.close()
            raise ValueError(f'Frame {index}: one global metric scale was rejected. {error}. '
                             f'Diagnostics: {diagnostic}. Cached depths are preserved; metric editing has not run.') from error
        frame_scales.append({"frame": index, **frame_fit})
        all_pairs.extend(pairs)
    try:
        fitted = fit_metres_per_native_unit(all_pairs, minimum_samples=500,
                                            maximum_median_relative_error=.15)
    except ValueError:
        rendered_cache.close()
        raise
    frame_ratios = [item["metres_per_native_unit"] / fitted["metres_per_native_unit"]
                    for item in frame_scales]
    if max(abs(ratio - 1) for ratio in frame_ratios) > .10:
        rendered_cache.close()
        raise ValueError("MoGe/UFO metre scale varies more than 10% across sampled frames")
    rendered_cache.close()
    return {**fitted, "source": "MoGe-2 metric camera_z aligned to WorldEngine rendered expected camera_z",
            'native_depth_definition': 'rendered_expected_camera_z_v1',
            "moge_source": str(moge_source), "moge_checkpoint": str(moge_checkpoint),
            "frame_fits": frame_scales, 'fov_source': 'UFO intrinsics with anisotropic resize correction' if camera_file else 'MoGe estimated',
            'sampled_windows': sorted(representatives), 'metric_depth_cache': str(depth_cache) if depth_cache else None}


def edit_asset(workspace: Path, *, asset_root: Path, output_asset_root: Path,
               new_name: str, track_id: int, offsets_path: Path,
               calibration: dict) -> dict:
    """Shift only the selected SAM track's observed Gaussian means."""
    import numpy as np
    import torch

    original = asset_root / "assets" / workspace.name
    manifest = json.loads((original / "ASSET.json").read_text(encoding="utf-8"))
    if manifest.get("asset_id") != workspace.name or manifest.get("version") != "ufo-worldengine-time-v2":
        raise ValueError("UFO asset identity/version differs from workspace")
    camera = np.load(original / "cameras.npz", allow_pickle=False)
    poses, times = camera["C"], camera["timestamps"]
    offsets = _load_offsets(offsets_path, len(poses))
    if new_name == workspace.name or not new_name or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for char in new_name
    ):
        raise ValueError("new_name must be a distinct ASCII asset ID")
    target = output_asset_root / "assets" / new_name
    if target.exists():
        raise FileExistsError(f"refusing to replace existing edited UFO asset: {target}")
    target.mkdir(parents=True)
    (target / "dynamic").mkdir()
    (target / "background").mkdir()
    shutil.copy2(original / "cameras.npz", target / "cameras.npz")
    scene = workspace / "reconstruction" / "000" / "pred100k"
    asset = torch.load(manifest["asset_ckpt"], map_location="cpu", weights_only=False)
    original_files = asset["observed_dynamic"]["config"]["actor_files"]
    if len(original_files) != len(poses):
        raise ValueError("UFO actor file count differs from camera poses")
    audit = []
    files = []
    for index, original_file in enumerate(original_files):
        actor = torch.load(original_file, map_location="cpu", weights_only=False)
        with np.load(scene / "sources" / f"{index:06d}.npz", allow_pickle=False) as source:
            try:
                selected, actor_count = actor_track_indices(source, track_id)
            except ValueError as exc:
                if "no actor Gaussians" not in str(exc):
                    raise
                selected = np.empty(0, dtype=np.int64)
                _, actor_count = _actor_count(source)
        if actor_count != len(actor["means"]):
            raise ValueError(f"UFO actor/source index mismatch at frame {index}")
        delta = metric_offset_to_native_world(offsets[index], poses[index].tolist(),
                                              calibration["metres_per_native_unit"])
        before = actor["means"][selected].clone()
        if len(selected):
            actor["means"][selected] += torch.tensor(delta, dtype=actor["means"].dtype)
        after = actor["means"][selected]
        actual_native = (after.mean(0) - before.mean(0)).tolist() if len(selected) else None
        actual_flu = None
        error_metres = None
        if actual_native is not None:
            camera_delta = poses[index][:3, :3].T @ np.asarray(actual_native)
            actual_flu = (float(camera_delta[2]), float(-camera_delta[0]),
                          float(-camera_delta[1]))
            actual_flu = tuple(value * calibration["metres_per_native_unit"]
                               for value in actual_flu)
            error_metres = max(abs(actual_flu[axis] - offsets[index][axis])
                               for axis in range(3))
            if error_metres > .01:
                raise ValueError(f"UFO metric displacement error at frame {index}: {error_metres} m")
        path = target / "dynamic" / f"{index:06d}.pt"
        torch.save(actor, path)
        files.append(str(path))
        audit.append({"frame": index, "timestamp_seconds": float(times[index]),
                      "target_gaussians": len(selected), "requested_flu_metres": offsets[index],
                      "actual_flu_metres": actual_flu, "max_error_metres": error_metres,
                      "world_native_delta": delta, "actual_world_native_delta": actual_native})
    if sum(row["target_gaussians"] > 0 for row in audit) < 3:
        raise ValueError("selected SAM track has fewer than 3 visible frames")
    asset["observed_dynamic"]["config"]["actor_files"] = files
    for name in ("background_shader.pt", "background_states.pt", "sky_environment.json"):
        source = original / "background" / name
        if source.is_file():
            shutil.copy2(source, target / "background" / name)
    background_config = asset["background"]["config"]
    for key, name in (("neural_shader_file", "background_shader.pt"),
                      ("neural_states_file", "background_states.pt"),
                      ("sky_environment_file", "sky_environment.json")):
        if key in background_config:
            background_config[key] = str(target / "background" / name)
    ckpt = target / "background" / f"{new_name}.ckpt"
    torch.save(asset, ckpt)
    edited = {**manifest, "asset_id": new_name, "asset_ckpt": str(ckpt),
              "camera_file": str(target / "cameras.npz"),
              "metric_calibration": calibration, "edited_track_id": track_id,
              "edit_offsets_path": str(offsets_path)}
    (target / "ASSET.json").write_text(json.dumps(edited, indent=2) + "\n", encoding="utf-8")
    report = {"asset_id": new_name, "asset_root": str(output_asset_root),
              "asset_manifest": str(target / "ASSET.json"),
              "metres_per_native_unit": calibration["metres_per_native_unit"],
              "track_id": track_id, "frames": audit}
    (target / "EDIT_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _actor_count(source):
    import numpy as np
    means, camera = source["means"], source["C"]
    depth = ((means - camera[:3, 3]) @ camera[:3, :3])[..., 2]
    valid = (np.isfinite(means).all(-1) & np.isfinite(source["scales"]).all(-1)
             & (depth > .2) & (depth < 400) & (source["opacities"] > .01))
    actor = valid & source["dynamic"] & (source["coverage"] < .5)
    return actor, int(actor.sum())


def _video_rate(timestamps, nominal_rate=None) -> Fraction:
    """Return a bounded AVRational while preserving per-frame timestamps."""
    times = [float(value) for value in timestamps]
    if len(times) < 2 or any(not math.isfinite(value) for value in times) or any(
        right <= left for left, right in zip(times, times[1:])
    ):
        raise ValueError('Video timestamps must contain at least two increasing finite values')
    value = nominal_rate if nominal_rate is not None else (len(times) - 1) / (times[-1] - times[0])
    try:
        rate = Fraction(str(value)).limit_denominator(1000000)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f'Invalid video frame rate: {value}') from error
    if rate <= 0 or rate.numerator > 2147483647 or rate.denominator > 2147483647:
        raise ValueError(f'Video frame rate exceeds the codec rational range: {rate}')
    return rate


def render_asset(asset_root: Path, asset_id: str, output_path: Path,
                 *, worldengine_root: Path) -> dict:
    """Replay edited actor files through the native WorldEngine renderer."""
    import av
    import numpy as np
    import torch

    sys.path[:0] = [str(worldengine_root / "vendor_runtime"),
                    str(worldengine_root / "projects" / "SimEngine")]
    from worldengine.render.mtgs.mtgs import MTGSRenderEngine

    asset = asset_root / "assets" / asset_id
    cameras = np.load(asset / "cameras.npz", allow_pickle=False)
    poses, intrinsics, times, ego = (cameras[key] for key in ("C", "K", "timestamps", "ego"))
    manifest = json.loads((asset / 'ASSET.json').read_text())
    rate = _video_rate(times, manifest.get('nominal_rate'))
    print(f'UFO_VIDEO_RATE {rate.numerator}/{rate.denominator}', flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    engine = MTGSRenderEngine(device="cuda", asset_folder_path=asset_root / "assets",
                              rasterize_mode="classic", near_plane=.2,
                              far_plane=400., packed=True)
    engine.reset(asset_id, {"metadata": {"digitaltwin_asset_id": asset_id,
                                          "digitaltwin_ego2globals": list(ego)}})
    render_started = time.monotonic()
    with av.open(str(output_path), 'w') as output:
        stream = output.add_stream("libx264", rate=rate)
        stream.width, stream.height, stream.pix_fmt = 520, 352, "yuv420p"
        stream.time_base = Fraction(1, 1_000_000)
        stream.codec_context.time_base = Fraction(1, 1_000_000)
        with torch.inference_mode():
            for index, timestamp in enumerate(times):
                engine.update_world(round(float(timestamp) * 1_000_000), {})
                result = engine.render_camera_to_world(poses[index], intrinsics[index], 520, 352)
                bgr = result["image"][0]
                if bgr.shape != (352, 520, 3) or bgr.dtype != np.uint8:
                    raise ValueError("WorldEngine edited render produced an invalid frame")
                frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")
                frame.pts = round(float(timestamp) * 1_000_000)
                frame.time_base = Fraction(1, 1_000_000)
                for packet in stream.encode(frame):
                    output.mux(packet)
                if index == 0 or (index+1) % 10 == 0 or index+1 == len(times):
                    print(f'UFO_RENDER_FRAME {index+1}/{len(times)} seconds={time.monotonic()-render_started:.2f}', flush=True)
        for packet in stream.encode():
            output.mux(packet)
    with av.open(str(output_path)) as check:
        frame_count = sum(1 for _ in check.decode(video=0))
    if frame_count != len(times):
        raise ValueError("edited UFO render frame count differs from asset")
    return {"rendered_video": str(output_path), "frames": frame_count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-asset-root", type=Path, required=True)
    parser.add_argument("--new-name", required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--track-id", type=int)
    target.add_argument("--target-selection", type=Path)
    parser.add_argument("--offsets-json", type=Path, required=True)
    parser.add_argument("--moge-source", type=Path, required=True)
    parser.add_argument("--moge-checkpoint", type=Path, required=True)
    parser.add_argument("--worldengine-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("--gpu must be non-negative")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    workspace = args.workspace.expanduser().resolve()
    binding = (bind_selected_track(workspace, args.target_selection.resolve())
               if args.target_selection else None)
    track_id = binding["sam_track_id"] if binding else args.track_id
    calibration = calibrate_workspace(workspace, moge_source=args.moge_source.resolve(),
                                      moge_checkpoint=args.moge_checkpoint.resolve(), asset_root=args.asset_root.resolve())
    report = edit_asset(workspace, asset_root=args.asset_root.resolve(),
                        output_asset_root=args.output_asset_root.resolve(),
                        new_name=args.new_name, track_id=track_id,
                        offsets_path=args.offsets_json.resolve(), calibration=calibration)
    if binding:
        report["target_binding"] = binding
    video = render_asset(args.output_asset_root.resolve(), args.new_name,
                         args.output_asset_root.resolve() / "output" / args.new_name / "edited.mp4",
                         worldengine_root=args.worldengine_root.resolve())
    print(json.dumps({**report, **video}, ensure_ascii=False))


if __name__ == "__main__":
    main()
