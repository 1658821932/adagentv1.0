"""Video retriever adapter that runs the component in its own virtual environment."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VIDEO_RETRIEVER_PYTHON = PROJECT_ROOT / "environments" / "video_retriever" / ".venv" / "bin" / "python"


def retrieve_video(input_data: dict) -> dict:
    requirement = input_data.get("requirement")
    if not isinstance(requirement, str) or not requirement.strip():
        raise ValueError("video_retriever requires a non-empty 'requirement' string")
    config_path = PROJECT_ROOT / "configs" / "video_retriever.json"
    default_library = json.loads(config_path.read_text()).get("default_library", "default")
    library = input_data.get("library", default_library)
    if library not in {"default", "waymo", "nuplan"}:
        raise ValueError("video_retriever library must be 'default', 'waymo', or 'nuplan'")
    if not VIDEO_RETRIEVER_PYTHON.is_file():
        raise RuntimeError(
            "video_retriever environment is missing. Create it with: "
            "python3 -m venv environments/video_retriever/.venv && "
            "environments/video_retriever/.venv/bin/pip install -r "
            "mcp_servers/sentry_video_search/requirements.txt"
        )

    proc = subprocess.run(
        [str(VIDEO_RETRIEVER_PYTHON), "-m", "mcp_servers.sentry_video_search.adapter.runner"],
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

    return payload["result"]
