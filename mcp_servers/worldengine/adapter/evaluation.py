"""Batch failure filtering with explicit scene/trajectory identity matching.

The collision metric remains a dynamic-object axis-aligned approximation.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any
from mcp_servers.process import run_logged
from mcp_servers.runtime import runtime_task, current_run
from .config import SIMENGINE_PYTHON, PROJECT_ROOT
from .schemas import EvaluationResult


@runtime_task("worldengine", "evaluation")
def filter_failure_cases(input_data: dict[str, Any]) -> EvaluationResult:
    source = input_data.get("scenario_pkl") or input_data.get("scenario_path")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("failure_case_filter requires scenario_pkl")
    scenario = Path(source).expanduser().resolve()
    if not scenario.is_file():
        raise FileNotFoundError(f"scenario_pkl does not exist: {scenario}")
    python = Path(input_data.get("python") or os.getenv("SIMENGINE_PYTHON") or SIMENGINE_PYTHON).expanduser()
    if not python.is_absolute():
        python = PROJECT_ROOT/python
    if not python.is_file():
        raise FileNotFoundError(f"SimEngine Python does not exist: {python}")
    command = [str(python), str(Path(__file__).with_name("evaluate_batch.py")), str(scenario)]
    if input_data.get("result_dir"):
        command += ["--result-dir", str(Path(input_data["result_dir"]).expanduser().resolve())]
    if input_data.get("scene_token"):
        command += ["--scene-token", input_data["scene_token"]]
    task = current_run("worldengine")
    process = run_logged(command, cwd=PROJECT_ROOT, log_path=task.logs/"evaluation.log",
                         json_stdout=True, timeout=input_data.get("timeout", 120))
    if process.returncode:
        raise RuntimeError(process.stderr.strip() or "Batch decision evaluation failed")
    result = json.loads(process.stdout)
    return {"run_dir": str(task.root), **result}
