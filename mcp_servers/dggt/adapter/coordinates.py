"""Coordinate conversion from Smart Edit local axes to DGGT world axes."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def _frames(camera_params_path: str | Path) -> list[dict[str, Any]]:
    path = Path(camera_params_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DGGT camera params not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("DGGT camera params must contain a non-empty frames array")
    return frames


def smart_edit_to_dggt_camera(vector: list[float]) -> list[float]:
    if len(vector) != 3:
        raise ValueError("Smart Edit vector must contain [forward, left, up]")
    forward, left, up = (float(value) for value in vector)
    # DGGT OpenCV camera axes are [right, down, forward].
    return [-left, -up, forward]


def smart_edit_to_dggt_world(vector: list[float], world_to_camera: list[list[float]], meters_to_dggt: float) -> list[float]:
    if meters_to_dggt <= 0:
        raise ValueError("meters_to_dggt must be positive")
    if len(world_to_camera) != 4 or any(len(row) != 4 for row in world_to_camera):
        raise ValueError("world_to_camera_opencv must be a 4x4 matrix")
    camera_vector = smart_edit_to_dggt_camera(vector)
    world_vector = [
        sum(float(world_to_camera[row][axis]) * camera_vector[row] for row in range(3)) * float(meters_to_dggt)
        for axis in range(3)
    ]
    # DGGT normalized scenes use world Y as the vertical axis. Smart Edit
    # trajectory edits are road-plane translations; camera pitch must not turn
    # a forward/lateral move into a height change that sinks the object.
    world_vector[1] = 0.0
    return world_vector


def _interpolate_waypoints(waypoints: list[list[float]], method: str, count: int) -> list[list[float]]:
    """Sample Smart Edit local control points once per rendered frame."""
    if not waypoints:
        raise ValueError("trajectory_edit requires at least one waypoint")
    if count <= 0:
        return []
    if len(waypoints) == 1:
        return [list(map(float, waypoints[0])) for _ in range(count)]
    positions = [index * (len(waypoints) - 1) / max(count - 1, 1) for index in range(count)]
    values: list[list[float]] = []
    for position in positions:
        segment = min(int(position), len(waypoints) - 2)
        local = position - segment
        if method == "linear":
            left = waypoints[segment]
            right = waypoints[segment + 1]
            values.append([float(left[axis] + (right[axis] - left[axis]) * local) for axis in range(3)])
            continue
        p0 = waypoints[max(segment - 1, 0)]
        p1 = waypoints[segment]
        p2 = waypoints[segment + 1]
        p3 = waypoints[min(segment + 2, len(waypoints) - 1)]
        values.append([
            float(0.5 * (
                2 * p1[axis]
                + (-p0[axis] + p2[axis]) * local
                + (2 * p0[axis] - 5 * p1[axis] + 4 * p2[axis] - p3[axis]) * local ** 2
                + (-p0[axis] + 3 * p1[axis] - 3 * p2[axis] + p3[axis]) * local ** 3
            ))
            for axis in range(3)
        ])
    return values


def convert_edit_config(
    config: dict[str, Any],
    camera_params_path: str | Path,
    meters_to_dggt: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert local Smart Edit offsets into per-frame DGGT world offsets."""
    frames = _frames(camera_params_path)
    rigid = config.get("Nodes", {}).get("RigidNodes", {})
    trajectory = rigid.get("trajectory_edit")
    if not isinstance(trajectory, dict):
        return copy.deepcopy(config), {"converted": False, "reason": "no trajectory_edit"}
    if meters_to_dggt is None:
        raise ValueError("trajectory_edit requires explicit meters_to_dggt")
    groups = trajectory.get("waypoints")
    if not isinstance(groups, list):
        raise ValueError("trajectory_edit.waypoints must be a list")
    converted = copy.deepcopy(config)
    converted_trajectory = converted["Nodes"]["RigidNodes"]["trajectory_edit"]
    converted_groups = []
    for group in groups:
        if not isinstance(group, list):
            raise ValueError("trajectory_edit waypoint groups must be lists")
        method = (trajectory.get("method") or ["linear"])[len(converted_groups)]
        local_values = _interpolate_waypoints(group, method, len(frames))
        world_values = []
        for local_value, frame in zip(local_values, frames):
            pose = frame.get("world_to_camera_opencv")
            if pose is None:
                raise ValueError(f"DGGT camera params frame {frame.get('frame_index')} has no world_to_camera_opencv")
            world_values.append(smart_edit_to_dggt_world(local_value, pose, meters_to_dggt))
        converted_groups.append(world_values)
    converted_trajectory["waypoints"] = converted_groups
    diagnostics = {
        "converted": True,
        "meters_to_dggt": float(meters_to_dggt),
        "pose_count": len(frames),
        "reference_frame": int(frames[0].get("frame_index", 0)),
        "conversion_mode": "per_frame_camera_pose",
        "smart_edit_axes": "[forward, left, up]",
        "dggt_camera_axes": "[right, down, forward]",
        "vertical_axis_locked": "world_y",
        "converted_waypoints": converted_groups,
    }
    return converted, diagnostics
