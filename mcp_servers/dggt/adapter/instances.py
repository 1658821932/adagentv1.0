"""DGGT YOLO + ByteTrack instance discovery."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from mcp_servers.runtime import runtime_task, current_run

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DGGT_ROOT = Path(os.environ.get("DGGT_ROOT", str(PROJECT_ROOT / "mcp_servers" / "dggt" / "core"))).expanduser().resolve()
DGGT_PYTHON = Path(os.environ.get("DGGT_PYTHON", str(PROJECT_ROOT / "environments" / "dggt" / ".conda" / "bin" / "python"))).expanduser().resolve()
AGENT_FRAME_COUNT = 40


def _dggt_yolo_backend(input_data: dict[str, Any]) -> dict[str, Any]:
    if input_data.get("sequence_length", AGENT_FRAME_COUNT) != AGENT_FRAME_COUNT:
        raise ValueError("Agent instance_discovery requires sequence_length=40")
    image_dir = input_data.get("image_dir")
    output_path = input_data.get("output_path")
    if not isinstance(image_dir, str) or not image_dir.strip():
        raise ValueError("dggt_yolo instance discovery requires image_dir")
    if not isinstance(output_path, str) or not output_path.strip():
        raise ValueError("dggt_yolo instance discovery requires output_path")
    image_dir_path = Path(image_dir).expanduser().resolve()
    requested_output = Path(output_path).expanduser().resolve()
    task = current_run("dggt")
    output = task.output / "instance_info.json"
    if not image_dir_path.is_dir():
        raise FileNotFoundError(f"DGGT image_dir not found: {image_dir_path}")
    if not DGGT_PYTHON.is_file():
        raise FileNotFoundError(f"DGGT Python not found: {DGGT_PYTHON}")
    command = [str(DGGT_PYTHON), str(DGGT_ROOT / "tools" / "generate_yolo_instance_info.py"), "--image_dir", str(image_dir_path), "--scene_name", str(input_data.get("scene_name", "000")), "--sequence_length", str(AGENT_FRAME_COUNT), "--start_idx", str(int(input_data.get("start_idx", 0))), "--weights", str(input_data.get("weights", "yolo11m.pt")), "--conf", str(float(input_data.get("conf", 0.25))), "--output", str(output)]
    if input_data.get("vis_dir"):
        command.extend(["--vis_dir", str(task.output / "visualizations")])
    env = os.environ.copy()
    env["PYTHONPATH"] = str(DGGT_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(command, capture_output=True, text=True, cwd=DGGT_ROOT, env=env)
    (task.logs / "instances.log").write_text(proc.stdout + "\n" + proc.stderr)
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "DGGT YOLO instance generation failed")
    if not output.is_file():
        raise RuntimeError(f"DGGT YOLO generator did not create {output}")
    result = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(result.get("instances"), list):
        raise ValueError("DGGT instance_info must contain an instances array")
    requested_output.parent.mkdir(parents=True, exist_ok=True)
    if output.resolve() != requested_output:
        shutil.copy2(output, requested_output)
    if input_data.get("vis_dir") and (task.output / "visualizations").is_dir():
        destination = Path(input_data["vis_dir"]).expanduser().resolve()
        if destination != (task.output / "visualizations").resolve():
            shutil.copytree(task.output / "visualizations", destination, dirs_exist_ok=True)
    return {"run_dir": str(task.root), "instance_info_path": str(output), **result}


@runtime_task("dggt", "instance_discovery")
def discover_instances(input_data: dict) -> dict:
    backend = str(input_data.get("backend") or "dggt_yolo").lower()
    if backend != "dggt_yolo":
        raise ValueError("instance_discovery only supports backend='dggt_yolo'")
    result = _dggt_yolo_backend(input_data)
    return result
