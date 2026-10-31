"""Adapter for real single-view DGGT Mode 2 reconstruction."""

from __future__ import annotations

from mcp_servers.runtime import runtime_task, current_run
import json
import subprocess
from mcp_servers.process import run_logged
from pathlib import Path


from .config import PROJECT_ROOT, AGENT_PYTHON, AGENT_FRAME_COUNT


@runtime_task("dggt", "reconstruction")
def reconstruct_video(input_data: dict) -> dict:
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

    proc = run_logged(
        [str(AGENT_PYTHON), "-m", "mcp_servers.dggt.adapter.runner"],
        input=json.dumps({"video_path": video_path.strip(), "num_frames": AGENT_FRAME_COUNT,
                          "export_render_state": input_data.get("export_render_state", False)}, ensure_ascii=False),
        cwd=PROJECT_ROOT,
        env=current_run("dggt").child_env(),
        log_path=current_run("dggt").logs / "reconstruction.log",
        json_stdout=True,
        timeout=int(input_data.get("timeout", 86400)),
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
