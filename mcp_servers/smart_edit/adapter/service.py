"""Smart Edit MCP adapter for executable edit configuration generation."""
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, SMART_EDIT_ROOT, SMART_EDIT_PYTHON

def generate_edit_config(input_data: dict[str, Any]) -> dict[str, Any]:
    edit_plan = input_data.get("edit_plan")
    output_path = input_data.get("output_path")
    if not edit_plan:
        raise ValueError("edit_config_generator requires edit_plan")
    if not isinstance(output_path, str) or not output_path.strip():
        raise ValueError("edit_config_generator requires output_path")
    if Path(output_path).suffix.lower() != ".json":
        raise ValueError("edit_config_generator output_path must use .json for DGGT")
    if not SMART_EDIT_ROOT.is_dir():
        raise FileNotFoundError(f"Smart Edit project not found: {SMART_EDIT_ROOT}")
    if not SMART_EDIT_PYTHON.is_file():
        raise FileNotFoundError(f"Smart Edit Python not found: {SMART_EDIT_PYTHON}")
    output_kind = input_data.get("output_kind", "rigid_config")
    if output_kind not in {"rigid_config", "ego_trajectory_offsets"}:
        raise ValueError("edit_config_generator output_kind must be rigid_config or ego_trajectory_offsets")
    if input_data.get("frame_count", 40) != 40:
        raise ValueError("Agent edit_config_generator requires frame_count=40")
    asset_scale = input_data.get("asset_scale")
    if asset_scale is not None and (isinstance(asset_scale, bool) or not isinstance(asset_scale, (int, float)) or asset_scale <= 0):
        raise ValueError("edit_config_generator asset_scale must be a positive number")
    payload = {"edit_request": edit_plan, "instance_info_path": input_data.get("instance_info_path"), "output_path": str(Path(output_path).expanduser().resolve()), "scene_info": input_data.get("scene_info"), "asset_path": input_data.get("asset_path"), "asset_scale": asset_scale, "output_kind": output_kind, "frame_count": 40}
    env = os.environ.copy()
    env["SMART_EDIT_ROOT"] = str(SMART_EDIT_ROOT)
    proc = subprocess.run([str(SMART_EDIT_PYTHON), "-m", "mcp_servers.smart_edit.adapter.client"], input=json.dumps(payload, ensure_ascii=False), capture_output=True, text=True, cwd=PROJECT_ROOT, env=env)
    try:
        response = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(proc.stderr.strip() or "Smart Edit client returned invalid JSON") from exc
    if proc.returncode or response.get("status") != "success":
        error = response.get("error") or {}
        raise RuntimeError(error.get("message") or proc.stderr.strip() or "Smart Edit edit failed")
    result = response["result"]
    saved_path = Path(result["saved_path"])
    if not saved_path.is_file():
        raise RuntimeError(f"Smart Edit did not create {saved_path}")
    config = json.loads(saved_path.read_text(encoding="utf-8"))
    if output_kind == "ego_trajectory_offsets":
        offsets = config.get("offsets")
        if config.get("coordinate_system") != "camera_local_forward_left_up" or not isinstance(offsets, list) or not offsets:
            raise ValueError("Smart Edit must produce a valid ego trajectory offsets JSON")
        if any(not isinstance(offset, list) or len(offset) != 3 or any(type(value) not in (int, float) or not math.isfinite(value) for value in offset) for offset in offsets):
            raise ValueError("Smart Edit ego offsets must be finite [forward, left, up] vectors")
        frame_count = input_data.get("frame_count")
        if frame_count is not None and len(offsets) not in (1, int(frame_count)):
            raise ValueError(f"Smart Edit ego offsets require 1 or {frame_count} vectors, got {len(offsets)}")
        result["output_kind"] = output_kind
        return result
    rigid = config.get("Nodes", {}).get("RigidNodes", {})
    if not isinstance(rigid, dict) or not rigid or set(config.get("Nodes", {})) != {"RigidNodes"}:
        raise ValueError("Smart Edit must produce DGGT RigidNodes operations")
    unsupported = set(rigid) - {"trajectory_edit", "rotate", "scale", "remove", "add", "replace"}
    if unsupported:
        raise ValueError(f"DGGT does not support RigidNodes operations: {sorted(unsupported)}")
    if any(operation in rigid for operation in ("trajectory_edit", "rotate", "scale", "remove", "replace")):
        instance_path = config.get("instance_info_path")
        if not isinstance(instance_path, str):
            raise ValueError("Smart Edit existing-instance edits require instance_info_path")
        instance_file = Path(instance_path)
        if not instance_file.is_absolute():
            instance_file = saved_path.parent / instance_file
        if not instance_file.is_file():
            raise FileNotFoundError(f"DGGT instance_info not found: {instance_file}")
        available_ids = {instance["instance_id"] for instance in json.loads(instance_file.read_text(encoding="utf-8")).get("instances", [])}
        for operation in ("trajectory_edit", "rotate", "scale", "remove", "replace"):
            params = rigid.get(operation, {})
            ids = params.get("target_id" if operation == "replace" else "instance_id", [])
            if any(instance_id not in available_ids for instance_id in ids):
                raise ValueError(f"DGGT {operation} references an instance absent from instance_info: {ids}")
    for operation in ("add", "replace"):
        if operation not in rigid:
            continue
        parameters = rigid[operation]
        scales = parameters.get("scale") if isinstance(parameters, dict) else None
        if not isinstance(scales, list) or len(scales) != 1 or isinstance(scales[0], bool) or not isinstance(scales[0], (int, float)) or scales[0] <= 0:
            raise ValueError(f"DGGT {operation} requires one explicit positive asset scale")
        if asset_scale is None:
            raise ValueError(f"DGGT {operation} requires an explicit asset_scale input")
        if scales[0] != asset_scale:
            raise ValueError("Smart Edit scale differs from the requested asset_scale")
        asset_path = input_data.get("asset_path")
        if not isinstance(asset_path, str) or not Path(asset_path).is_file():
            raise ValueError(f"DGGT {operation} requires an existing asset_path input")
        asset_key = "add_obj" if operation == "add" else "replace_obj"
        assets = parameters.get(asset_key, [])
        if len(assets) != 1:
            raise ValueError(f"DGGT {operation} requires one asset path")
        generated_asset = Path(assets[0])
        if not generated_asset.is_absolute():
            generated_asset = saved_path.parent / generated_asset
        if generated_asset.resolve() != Path(asset_path).resolve():
            raise ValueError("Smart Edit asset path differs from the requested asset_path")
    return result
