"""Scene frame budget: one resolver for every simulation entry point.

The budget used to fall back to a global constant, so a 24-frame scene could be
launched as a 40-frame rollout. The budget is now taken from the scene the
caller actually handed over, in decreasing order of authority.
"""
from __future__ import annotations

import json
from pathlib import Path

FALLBACK_FRAME_COUNT = 40
_SCENE_PATH_KEYS = ("scenario_data_folder_path", "scene_dir", "frozen_scene_path",
                    "scene_path", "attack_scene_path")


def _frame_count_from_json(path: Path) -> int | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if isinstance(data, list):
        return len(data) or None
    if not isinstance(data, dict):
        return None
    for key in ("frame_count", "frame_total"):
        value = data.get(key)
        if type(value) is int and value > 0:
            return value
    for key in ("frames", "timestamps", "samples"):
        value = data.get(key)
        if isinstance(value, list) and value:
            return len(value)
    nested = data.get("checks")
    if isinstance(nested, dict):
        return _frame_count_from_json_from_dict(nested)
    return None


def _frame_count_from_json_from_dict(data: dict) -> int | None:
    for key in ("frame_count", "frame_total"):
        value = data.get(key)
        if type(value) is int and value > 0:
            return value
    return None


def scene_frame_count(scene_dir) -> int | None:
    """Frames recorded in a scene directory produced by the editors."""
    if not scene_dir:
        return None
    root = Path(scene_dir).expanduser()
    if not root.is_dir():
        return None
    for relative in ("metadata/frames.json", "frames.json", "export-report.json"):
        candidate = root / relative
        if candidate.is_file():
            count = _frame_count_from_json(candidate)
            if count:
                return count
    frames_dir = root / "frames"
    if frames_dir.is_dir():
        count = len([entry for entry in frames_dir.iterdir() if entry.is_file()])
        if count:
            return count
    return None


def resolve_frame_count(input_data: dict, fallback: int = FALLBACK_FRAME_COUNT) -> int:
    """Total frames for this rollout, from the most authoritative source available."""
    explicit = input_data.get("max_frame_per_scene")
    if type(explicit) is int and explicit >= 5:
        return explicit
    history, future = input_data.get("num_history"), input_data.get("num_future")
    if type(history) is int and type(future) is int and history + future >= 5:
        return history + future
    declared = input_data.get("frame_count")
    if type(declared) is int and declared >= 5:
        return declared
    for key in _SCENE_PATH_KEYS:
        count = scene_frame_count(input_data.get(key))
        if count and count >= 5:
            return count
    return fallback
