"""Adapter for real single-view DGGT Mode 2 reconstruction."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGENT_PYTHON = PROJECT_ROOT / "environments" / "agent" / ".venv" / "bin" / "python"
AGENT_FRAME_COUNT = 40


def reconstruct_video(input_data: dict) -> dict:
    backend = input_data.get("backend", "adscenegraph" if input_data.get("scene_manifest_path") else "dggt")
    if backend == "adscenegraph":
        if not input_data.get("scene_manifest_path"):
            raise ValueError("ADSceneGraph reconstruction requires scene_manifest_path from nuplan retrieval")
        from ...nuplan_data.common import RECONSTRUCTION_PYTHON
        python = RECONSTRUCTION_PYTHON
        if not python.is_file():
            raise RuntimeError(f"Existing ADSceneGraph environment not found: {python}")
        proc = subprocess.run(
            [str(python), "-m", "mcp_servers.nuplan_data.reconstruct"],
            input=json.dumps(input_data, ensure_ascii=False), capture_output=True, text=True, cwd=PROJECT_ROOT,
            env=os.environ | {"PYTHONDONTWRITEBYTECODE": "1"},
        )
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(proc.stderr[-3000:] or "ADSceneGraph returned invalid JSON") from exc
        if proc.returncode or payload.get("status") != "success":
            raise RuntimeError((payload.get("error") or {}).get("message") or proc.stderr[-3000:])
        return payload["result"]
    if backend != "dggt":
        raise ValueError("reconstruction backend must be dggt or adscenegraph")
    if input_data.get("num_frames", AGENT_FRAME_COUNT) != AGENT_FRAME_COUNT:
        raise ValueError("Agent reconstruction requires num_frames=40")
    video_path = input_data.get("video_path")
    if not isinstance(video_path, str) or not video_path.strip():
        raise ValueError("reconstruction requires a non-empty 'video_path' string")
    candidate = Path(video_path).expanduser().resolve()
    if not candidate.is_file():
        compact = "".join(candidate.name.split())
        for sibling in candidate.parent.iterdir() if candidate.parent.is_dir() else []:
            if "".join(sibling.name.split()) == compact:
                candidate = sibling
                break
    video_path = str(candidate)
    if not AGENT_PYTHON.is_file():
        raise RuntimeError(
            "Agent environment is missing: environments/agent/.venv/bin/python"
        )

    proc = subprocess.run(
        [str(AGENT_PYTHON), "-m", "mcp_servers.dggt.adapter.runner"],
        input=json.dumps({"video_path": video_path.strip(), "num_frames": AGENT_FRAME_COUNT,
                          "export_render_state": input_data.get("export_render_state", False)}, ensure_ascii=False),
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(proc.stderr.strip() or "reconstruction returned invalid JSON") from exc

    if proc.returncode != 0 or payload.get("status") != "success":
        error = payload.get("error") or {}
        message = error.get("message") or proc.stderr.strip() or "DGGT reconstruction failed"
        raise RuntimeError(message)

    return payload["result"]
