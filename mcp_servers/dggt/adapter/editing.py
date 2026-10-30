"""Run DGGT Mode 2 with optional instance edits and ego trajectory offsets."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from .coordinates import convert_edit_config
from mcp_servers.runtime import runtime_task, current_run

from .config import PROJECT_ROOT, AGENT_PYTHON, AGENT_FRAME_COUNT

@runtime_task("dggt", "edit_scene")
def edit_scene(input_data: dict) -> dict:
    if input_data.get("num_frames", AGENT_FRAME_COUNT) != AGENT_FRAME_COUNT:
        raise ValueError("Agent scene_editor requires num_frames=40")
    video_path = input_data.get("video_path")
    edit_config_path = input_data.get("edit_config_path")
    if not isinstance(video_path, str) or not video_path.strip():
        raise ValueError("scene_editor requires video_path")
    video = Path(video_path).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"scene_editor video not found: {video}")
    config = None
    if edit_config_path is not None:
        if not isinstance(edit_config_path, str) or not edit_config_path.strip():
            raise ValueError("edit_config_path must be a non-empty path")
        config = Path(edit_config_path).expanduser().resolve()
        if config.suffix.lower() != ".json":
            raise ValueError("scene_editor requires a DGGT JSON edit config")
        if not config.is_file():
            raise FileNotFoundError(f"scene_editor edit config not found: {config}")
    if not AGENT_PYTHON.is_file():
        raise RuntimeError("Agent environment is missing: environments/agent/.venv/bin/python")
    coordinate_system = input_data.get("coordinate_system", "smart_edit")
    meters_to_dggt = input_data.get("meters_to_dggt")
    camera_params_path = input_data.get("camera_params_path")
    ego_trajectory_offsets_path = input_data.get("ego_trajectory_offsets_path")
    if ego_trajectory_offsets_path is not None:
        if not isinstance(ego_trajectory_offsets_path, str) or not ego_trajectory_offsets_path.strip():
            raise ValueError("ego_trajectory_offsets_path must be a non-empty path")
        ego_trajectory_offsets_path = str(Path(ego_trajectory_offsets_path).expanduser().resolve())
        if not Path(ego_trajectory_offsets_path).is_file():
            raise FileNotFoundError(f"ego trajectory offsets not found: {ego_trajectory_offsets_path}")
    if config is None and ego_trajectory_offsets_path is None:
        raise ValueError("scene_editor requires edit_config_path or ego_trajectory_offsets_path")
    execution_config = config
    coordinate_diagnostics = {"converted": False, "coordinate_system": coordinate_system}
    if coordinate_system not in {"smart_edit", "dggt_world"}:
        raise ValueError("coordinate_system must be smart_edit or dggt_world")
    if config is not None and coordinate_system == "smart_edit":
        source_config = json.loads(config.read_text(encoding="utf-8"))
        if "trajectory_edit" in source_config.get("Nodes", {}).get("RigidNodes", {}):
            if not isinstance(camera_params_path, str) or not camera_params_path.strip():
                raise ValueError("scene_editor smart_edit trajectory requires camera_params_path")
            if isinstance(meters_to_dggt, bool) or not isinstance(meters_to_dggt, (int, float)) or meters_to_dggt <= 0:
                raise ValueError("scene_editor trajectory_edit requires explicit positive meters_to_dggt")
            converted_config, coordinate_diagnostics = convert_edit_config(source_config, camera_params_path, meters_to_dggt)
            execution_config = current_run("dggt").input / (config.stem + ".dggt.json")
            execution_config.write_text(json.dumps(converted_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    proc = subprocess.run(
        [str(AGENT_PYTHON), "-m", "mcp_servers.dggt.adapter.runner"],
        input=json.dumps({"video_path": str(video), "edit_config_path": str(execution_config) if execution_config else None, "ego_trajectory_offsets_path": ego_trajectory_offsets_path, "export_dynamic_frame_ply": input_data.get("export_dynamic_frame_ply", True), "num_frames": AGENT_FRAME_COUNT}, ensure_ascii=False),
        capture_output=True, text=True, cwd=PROJECT_ROOT, env=current_run("dggt").child_env(),
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(proc.stderr.strip() or "scene_editor returned invalid JSON") from exc
    if proc.returncode != 0 or payload.get("status") != "success":
        error = payload.get("error") or {}
        raise RuntimeError(error.get("message") or proc.stderr.strip() or "DGGT scene editing failed")
    result = dict(payload["result"])
    result["scene_config"] = str(config) if config else None
    result["edit_config_path"] = str(config) if config else None
    result["coordinate_conversion"] = coordinate_diagnostics
    if ego_trajectory_offsets_path is not None:
        result["ego_trajectory_offsets_path"] = ego_trajectory_offsets_path
    return result
