"""Service layer: invoke worldengine_failure_analysis (wfa) scene analysis.

Wraps wfa's run.py CLI for a single scene token and returns the parsed
VLM forensic result. VLM credentials live in wfa/.secrets/vlm.json (0600),
which is intentionally NOT vendored into the open-source repo.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


from mcp_servers.process import run_logged
from mcp_servers.runtime import runtime_task, current_run

from .config import WFA_ROOT, DEFAULT_CONFIG, DEFAULT_CONFIG_EXAMPLE, DEFAULT_RUN_NAME, _resolve_python, _check_secret
from .schemas import WfaSceneResult

class WfaError(RuntimeError):
    """The native failure-analysis workflow failed."""


@runtime_task("worldengine_failure_analysis", "analyze_scene")
def analyze_scene(
    token: str,
    *,
    config_path: str | None = None,
    run_name: str = DEFAULT_RUN_NAME,
    resume: bool = True,
    timeout: int = 1800,
) -> WfaSceneResult:
    """Run wfa forensic analysis for one scene token and return the parsed result."""
    if not token or not token.strip():
        raise ValueError("failure_analyzer (wfa) requires a non-empty scene 'token'")
    secret_file = _check_secret()

    config = Path(config_path).expanduser().resolve() if config_path else DEFAULT_CONFIG
    if not config.is_file():
        if config == DEFAULT_CONFIG and DEFAULT_CONFIG_EXAMPLE.is_file():
            raise WfaError(
                f"wfa config not found: {config}. "
                f"Copy {DEFAULT_CONFIG_EXAMPLE} to {config} and fill in your paths."
            )
        raise WfaError(f"wfa config not found: {config}")

    task = current_run("worldengine_failure_analysis")
    run_py = WFA_ROOT / "run.py"
    cmd = [
        _resolve_python(), str(run_py), "scene",
        "--token", token.strip(),
        "--config", str(config),
        "--run-name", run_name,
        "--run-dir", str(task.output),
    ]
    if resume:
        cmd.append("--resume")

    proc = run_logged(
        cmd, cwd=str(WFA_ROOT), log_path=task.logs / "analysis.log", timeout=timeout, env={**task.child_env(), "WFA_SECRET_FILE": str(secret_file)},
    )

    if proc.returncode:
        raise WfaError(f"wfa exited with code {proc.returncode}; see {task.logs}")

    # Per-scene attempts live in this task output.
    scene_dir = task.output / "scenes" / token.strip()
    result_path = scene_dir / "result.json"
    status_path = scene_dir / "status.json"
    report_path = scene_dir / "report.md"

    # Resolve the real token dir (wfa may suffix-match the token)
    if not result_path.is_file():
        scenes_root = task.output / "scenes"
        if scenes_root.is_dir():
            matches = [d for d in scenes_root.iterdir()
                       if d.is_dir() and (d.name == token.strip() or d.name.endswith("-" + token.strip()))]
            if len(matches) == 1:
                scene_dir = matches[0]
                result_path = scene_dir / "result.json"
                status_path = scene_dir / "status.json"
                report_path = scene_dir / "report.md"

    if not result_path.is_file():
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        raise WfaError(
            f"wfa scene analysis produced no result.json for token {token!r} "
            f"(exit={proc.returncode}). {tail}"
        )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    status = {}
    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            status = {}
    if status.get("state") == "failed":
        raise WfaError(
            f"wfa scene analysis failed for {token!r}: "
            f"{status.get('error_type')} {status.get('message')}"
        )

    return WfaSceneResult(
        token=result.get("token", token.strip()),
        run_dir=str(task.root),
        status=result.get("status", status.get("state", "unknown")),
        failure_reason=result.get("failure_reason", ""),
        reason_category=result.get("reason_category", ""),
        observations=result.get("observations", []),
        candidate_assessment=result.get("candidate_assessment", []),
        uncertainty=result.get("uncertainty", ""),
        official_metrics=result.get("official_metrics", {}),
        report_path=str(report_path) if report_path.is_file() else "",
        result_path=str(result_path),
        raw=result,
    )

