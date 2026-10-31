"""Video retriever adapter that runs the component in its own virtual environment."""

from __future__ import annotations

from mcp_servers.runtime import runtime_task, current_run
import json
import subprocess
from mcp_servers.process import run_logged
from pathlib import Path


from .config import PROJECT_ROOT, VIDEO_RETRIEVER_PYTHON

@runtime_task("sentry_video_search", "retrieval")
def retrieve_video(input_data: dict) -> dict:
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

    proc = run_logged(
        [str(VIDEO_RETRIEVER_PYTHON), "-m", "mcp_servers.sentry_video_search.adapter.runner"],
        input=json.dumps({"requirement": requirement.strip(), "library": library}, ensure_ascii=False),
        cwd=PROJECT_ROOT,
        env=current_run("sentry_video_search").child_env(),
        log_path=current_run("sentry_video_search").logs / "retrieval.log",
        json_stdout=True,
        timeout=int(input_data.get("timeout", 86400)),
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
