"""Video retriever adapter that runs the component in its own virtual environment."""

from __future__ import annotations

import json
import os
from pathlib import Path

from common.scene_pool import EXCLUDE_ENV_VAR, load_excluded_scene_tokens, normalize_scene_token
from mcp_servers.process import run_logged
from mcp_servers.runtime import current_run, runtime_task

from .config import DEFAULT_RETRIEVAL_TIMEOUT_S, PROJECT_ROOT, VIDEO_RETRIEVER_PYTHON


def _token_from_step_input(step_dir: Path) -> str | None:
    in_path = step_dir / "input.json"
    if not in_path.is_file():
        return None
    try:
        inp = json.loads(in_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    ref = inp.get("scene_ref")
    if isinstance(ref, dict):
        return normalize_scene_token(ref.get("scene_token") or ref.get("scene_id"))
    return None


def rejected_scene_tokens(agent_log_dir: str | None) -> list[str]:
    """Return scene tokens rejected by Tool 04 or failed in Tool 05 within this single run."""
    if not agent_log_dir:
        return []
    log_dir = Path(agent_log_dir)
    rejected: set[str] = set()

    # 1. History from resumed parent runs (edit_planner rejections & attack_vehicle_editor failures)
    initial_state = log_dir / "initial_state.json"
    if initial_state.is_file():
        try:
            history = json.loads(initial_state.read_text(encoding="utf-8")).get("history", [])
        except (OSError, json.JSONDecodeError):
            history = []
        for entry in history:
            tool = entry.get("tool")
            output = entry.get("result") or {}
            res = output.get("result") or {}
            if tool == "edit_planner":
                tok = normalize_scene_token(res.get("scene_token"))
                if tok and output.get("status") == "success" and res.get("suitability") in {
                    "unsuitable", "needs_information",
                }:
                    rejected.add(tok)
            elif tool == "attack_vehicle_editor":
                if output.get("status") != "success" or res.get("status") != "completed":
                    tok = normalize_scene_token(res.get("scene_token") or res.get("scene_id"))
                    if tok:
                        rejected.add(tok)

    # 2. Current run step outputs
    tools_dir = log_dir / "tools"
    if tools_dir.is_dir():
        for path in tools_dir.glob("step_*_edit_planner/output.json"):
            try:
                output = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            res = output.get("result") or {}
            tok = normalize_scene_token(res.get("scene_token"))
            if tok and output.get("status") == "success" and res.get("suitability") in {
                "unsuitable", "needs_information",
            }:
                rejected.add(tok)

        for path in tools_dir.glob("step_*_attack_vehicle_editor/output.json"):
            try:
                output = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            res = output.get("result") or {}
            if output.get("status") != "success" or res.get("status") != "completed":
                tok = normalize_scene_token(res.get("scene_token") or res.get("scene_id")) or _token_from_step_input(path.parent)
                if tok:
                    rejected.add(tok)

    return sorted(rejected)


def collect_excluded_scene_tokens(input_data: dict, agent_log_dir: str | None) -> list[str]:
    """Merge single-run rejections, explicit Tool 03 input exclusions, and batch pool file."""
    excluded = set(rejected_scene_tokens(agent_log_dir))
    explicit = input_data.get("exclude_scene_tokens") or []
    if isinstance(explicit, (list, tuple, set)):
        for item in explicit:
            tok = normalize_scene_token(item)
            if tok:
                excluded.add(tok)
    batch_file = os.environ.get(EXCLUDE_ENV_VAR)
    if batch_file:
        excluded.update(load_excluded_scene_tokens(batch_file))
    return sorted(excluded)


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
    excluded = collect_excluded_scene_tokens(input_data, run.manifest.get("agent_log_dir"))
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
