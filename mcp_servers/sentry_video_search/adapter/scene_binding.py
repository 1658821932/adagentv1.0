"""Resolve an indexed camera video to its dataset scene without loading pickles."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median

from mcp_servers.runtime import write_json


def _sidecar(video):
    video = Path(video).expanduser().absolute()
    for base in dict.fromkeys((video, video.resolve())):
        for path in (base.with_suffix(".scene.json"), Path(str(base) + ".scene.json")):
            if path.is_file():
                return path
    return None


def _existing(path):
    return str(path.absolute()) if path.exists() else None


def bind_scene(source_file, output_dir):
    """Require a scene binding; write ordered image paths to a manifest.

    Binding is explicit: never infer scene identity from a renamed search clip.
    Unlabelled or mismatched videos fail explicitly.
    """
    if not source_file:
        raise ValueError("Scene binding requires the original indexed source_file")
    sidecar = _sidecar(source_file)
    if sidecar is None:
        raise ValueError(f"No scene binding metadata for indexed video: {source_file}; register a .scene.json before indexing")
    record = json.loads(sidecar.read_text())
    if not record.get("sample_root") or not record.get("camera"):
        raise ValueError(f"Scene sidecar requires sample_root and camera: {sidecar}")
    sample = Path(record["sample_root"]).expanduser()
    sample = (sidecar.parent / sample).resolve() if not sample.is_absolute() else sample.resolve()
    metadata = sample / "metadata"
    manifest = json.loads((sample / "manifest.json").read_text())
    frames = json.loads((metadata / "frames.json").read_text())
    calibrations = json.loads((metadata / "cameras.json").read_text())
    token = manifest.get("anchor_token") or manifest.get("scene_token")
    if not token or not frames or not isinstance(calibrations, dict) or not calibrations:
        raise ValueError(f"Scene is missing identity, frames or camera metadata: {sample}")
    if record.get("scene_token") and record["scene_token"] != token:
        raise ValueError("Camera sidecar and sample scene token disagree")
    camera = record["camera"]
    if camera not in calibrations:
        raise ValueError(f"Unknown scene camera: {camera}")
    timestamps = [int(frame["timestamp"]) for frame in frames]
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("Scene frame timestamps must be strictly increasing")
    expected = set(calibrations)
    if any(set(frame.get("cams", {})) != expected for frame in frames):
        raise ValueError("Scene frames must describe the same cameras")
    image_root = sample / "original_images"
    all_cameras, camera_summary = {}, {}
    for name in calibrations:
        images = []
        for frame in frames:
            relative = Path(frame["cams"][name]["data_path"])
            image = relative if relative.is_absolute() else image_root / relative
            images.append({"timestamp_us": int(frame["timestamp"]),
                           "frame_token": frame.get("token"),
                           "image_path": str(image.absolute()), "available": image.is_file()})
        available = sum(item["available"] for item in images)
        video = Path(source_file).expanduser().absolute() if name == camera else None
        # Only bind additional videos when their sidecars confirm the same scene/camera.
        if video is None:
            for folder in dict.fromkeys((sidecar.parent, sample / "preview")):
                for candidate in sorted(folder.glob("*.scene.json")):
                    other = json.loads(candidate.read_text())
                    other_root = Path(other.get("sample_root", ".")).expanduser()
                    if not other_root.is_absolute():
                        other_root = candidate.parent / other_root
                    if (other.get("camera") == name and other_root.resolve() == sample
                            and other.get("scene_token", token) == token):
                        path = Path(other.get("video_path") or candidate.with_suffix("").with_suffix(".mp4"))
                        if not path.is_absolute():
                            path = candidate.parent / path
                        if path.is_file():
                            video = path
                            break
                if video is not None:
                    break
        summary = {"video_path": str(video) if video is not None and video.is_file() else None,
                   "frame_count": len(images), "available_images": available,
                   "images_complete": available == len(images)}
        camera_summary[name] = summary
        all_cameras[name] = {**summary, "calibration": calibrations[name], "frames": images}
    delta = median([b - a for a, b in zip(timestamps, timestamps[1:])]) if len(timestamps) > 1 else None
    reference = {
        "scene_id": manifest.get("scene_id", token), "scene_token": token,
        "sample_root": str(sample), "dataset": sample.parent.parent.name,
        "matched_camera": camera, "frame_count": len(frames),
        "fps": round(1_000_000 / delta, 6) if delta else None,
        "scenario_pkl": _existing(sample / "scenario/all_scenarios.pkl"),
        "frames_path": str(metadata / "frames.json"),
        "cameras_path": str(metadata / "cameras.json"),
        "asset_config_path": _existing(sample / "asset_config.yaml"),
        "asset_path": _existing(sample / "asset"),
        "runtime_asset_path": _existing(sample / "runtime_asset"),
        "maps_path": _existing(sample / "maps"),
        "cameras": camera_summary,
        "images_complete": all(c["images_complete"] for c in camera_summary.values()),
    }
    path = Path(output_dir) / "scene_binding.json"
    reference["manifest_path"] = str(path.absolute())
    write_json(path, {**reference, "source_file": str(Path(source_file).absolute()),
                      "timestamps_us": timestamps, "cameras": all_cameras})
    return reference


def register_scene_video(video_path, sample_root, camera):
    """Attach explicit scene identity before indexing a new camera video."""
    video, sample = Path(video_path).expanduser().resolve(), Path(sample_root).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    metadata = json.loads((sample / "metadata/cameras.json").read_text())
    if camera not in metadata:
        raise ValueError(f"Unknown scene camera: {camera}")
    manifest = json.loads((sample / "manifest.json").read_text())
    token = manifest.get("anchor_token") or manifest.get("scene_token")
    if not token:
        raise ValueError("Sample manifest requires anchor_token or scene_token")
    path = video.with_suffix(".scene.json")
    write_json(path, {"video_path": str(video), "camera": camera,
                      "scene_token": token, "sample_root": str(sample)})
    return path
