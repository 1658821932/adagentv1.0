"""Video retriever adapter that runs the component in its own virtual environment."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIDEO_RETRIEVER_PYTHON = PROJECT_ROOT / "environments" / "video_retriever" / ".venv" / "bin" / "python"


def run(input_data: dict) -> dict:
    requirement = input_data.get("requirement")
    if not isinstance(requirement, str) or not requirement.strip():
        raise ValueError("video_retriever requires a non-empty 'requirement' string")
    library = input_data.get("library", "default")
    if library not in {"default", "waymo"}:
        raise ValueError("video_retriever library must be 'default' or 'waymo'")
    if not VIDEO_RETRIEVER_PYTHON.is_file():
        raise RuntimeError(
            "video_retriever environment is missing. Create it with: "
            "python3 -m venv environments/video_retriever/.venv && "
            "environments/video_retriever/.venv/bin/pip install -r "
            "environments/video_retriever/requirements.txt"
        )

    proc = subprocess.run(
        [str(VIDEO_RETRIEVER_PYTHON), "-m", "mcp_servers.sentry_video_search.runner"],
        input=json.dumps({"requirement": requirement.strip(), "library": library}, ensure_ascii=False),
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(proc.stderr.strip() or "video_retriever returned invalid JSON") from exc

    if proc.returncode != 0 or payload.get("status") != "success":
        error = payload.get("error") or {}
        message = error.get("message") or proc.stderr.strip() or "video retrieval failed"
        raise RuntimeError(message)

    return {
        "status": "success",
        "tool": "video_retriever",
        "result": payload["result"],
    }
