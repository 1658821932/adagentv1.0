"""Video retriever adapter that runs the component in its own virtual environment."""

from __future__ import annotations

from mcp_servers.runtime import runtime_task, current_run
import json
import subprocess
from mcp_servers.process import run_logged
from pathlib import Path


from .config import DEFAULT_RETRIEVAL_TIMEOUT_S, PROJECT_ROOT, VIDEO_RETRIEVER_PYTHON

def rejected_scene_tokens(agent_log_dir: str | None) -> list[str]:
    if not agent_log_dir:
        return []
    log_dir = Path(agent_log_dir)
    outputs = []
    initial_state = log_dir / "initial_state.json"
    if initial_state.is_file():
        history = json.loads(initial_state.read_text()).get("history", [])
        outputs.extend(entry["result"] for entry in history if entry.get("tool") == "edit_planner")
    outputs.extend(
        json.loads(path.read_text())
        for path in (log_dir / "tools").glob("step_*_edit_planner/output.json")
    )
    rejected = set()
    for output in outputs:
        result = output.get("result") or {}
        token = result.get("scene_token")
        if token and output.get("status") == "success" and result.get("suitability") in {
            "unsuitable", "needs_information",
        }:
            rejected.add(token)
    return sorted(rejected)


@runtime_task("sentry_video_search", "retrieval")
def retrieve_video(input_data: dict) -> dict:
    requirement = input_data.get("requirement")
    if not isinstance(requirement, str) or not requirement.strip():
        raise ValueError("video_retriever requires a non-empty 'requirement' string")
    library = input_data.get("library", "default")
    if library != "default":
        raise ValueError("video_retriever library must be 'default'")
    if not VIDEO_RETRIEVER_PYTHON.is_file():
        raise RuntimeError(
            "video_retriever environment is missing. Create it with: "
            "python3 -m venv environments/video_retriever/.venv && "
            "environments/video_retriever/.venv/bin/pip install -r "
            "mcp_servers/sentry_video_search/core/requirements.txt"
        )

    run = current_run("sentry_video_search")
    excluded = rejected_scene_tokens(run.manifest.get("agent_log_dir"))
    if excluded:
        run.update(excluded_scene_tokens=excluded)

    proc = run_logged(
        [str(VIDEO_RETRIEVER_PYTHON), "-m", "mcp_servers.sentry_video_search.adapter.runner"],
        input=json.dumps({
            "requirement": requirement.strip(),
            "library": library,
            "excluded_scene_tokens": excluded,
        }, ensure_ascii=False),
        cwd=PROJECT_ROOT,
        env=run.child_env(),
        log_path=run.logs / "retrieval.log",
        json_stdout=True,
        timeout=int(input_data.get("timeout", DEFAULT_RETRIEVAL_TIMEOUT_S)),
    )

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(proc.stderr.strip() or "video_retriever returned invalid JSON") from exc

    if proc.returncode != 0 or payload.get("status") != "success":
        error = payload.get("error") or {}
        message = error.get("message") or proc.stderr.strip() or "video retrieval failed"
        raise RuntimeError(message)

    result = payload["result"]
    if not result.get("source_file") or not result.get("scene_ref"):
        raise RuntimeError("video_retriever requires a bound multi-camera scene")
    return result
