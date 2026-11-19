"""Failure analyzer: measure, verify road context, and supplement environment.

The map and archive produce deterministic interaction facts. One BEV route image is
reviewed for map context only; one violation-frame camera overview supplies
weather and environmental text. The whole analysis runs in a dedicated worker
process so timeout remains a hard budget for data reading and every model
request.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from mcp_servers.process import run_logged
from mcp_servers.worldengine_failure_analysis.adapter.service import WfaError, failure_facts

from . import evidence
from .language import ego_cause
from .schemas import Output
from .visual_check import VisualCheckError, check as visual_check, check_bev

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TIMEOUT_S = 300.0
TOOL = "failure_analyzer"


def _token(input_data: dict[str, Any]) -> str:
    token = input_data.get("token")
    if not token:
        raise ValueError(
            "failure_analyzer requires a scene 'token' from a WorldEngine closed-loop run. "
            "Run policy_runner -> failure_case_filter first to obtain tokens.")
    return str(token)


def _simulation(input_data: dict[str, Any]) -> str:
    simulation = input_data.get("simulation")
    if not simulation:
        raise ValueError("failure_analyzer requires the simulation record that produced the metrics")
    return str(simulation)


def worker_command():
    """The child that owns the whole analysis."""
    return [sys.executable, "-m", "tools.tool01_failure_analyzer.worker"]


def _failure(kind, run_dir, stage, message, **extra):
    return dict(status="error", tool=TOOL,
                error=dict(type=kind, stage=stage, message=message,
                           evidence_dir=str(run_dir), **extra))


def _timeout_of(error) -> bool:
    return bool(getattr(error, "timeout", False)
                or getattr(error, "details", {}).get("timeout", False))


def analyze(input_data: dict[str, Any], run_dir) -> dict[str, Any]:
    """Measure, verify and describe one scene; every artefact stays in ``run_dir``."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        token = _token(input_data)
        simulation = _simulation(input_data)
    except ValueError as error:
        evidence.write_manifest(run_dir, status="failed", stage="input", error=str(error))
        return _failure("input_error", run_dir, "input", str(error))

    evidence.write_manifest(run_dir, stage="data")
    try:
        facts = failure_facts(token, simulation)
    except (WfaError, ValueError) as error:
        evidence.write_manifest(run_dir, status="failed", stage="data", error=str(error))
        return _failure("timeout" if _timeout_of(error) else "failure_facts_error",
                        run_dir, "data", str(error))

    evidence.write_manifest(run_dir, stage="bev", resolved_token=facts["token"],
                            simulation=facts.get("simulation_record"))
    try:
        bev = check_bev(facts, run_dir=run_dir)
    except (VisualCheckError, OSError, ValueError) as error:
        stage = getattr(error, "stage", "bev")
        evidence.write_manifest(run_dir, status="failed", stage=stage, error=str(error))
        return _failure("bev_check_error", run_dir, stage, str(error))

    measured_context = facts["attack"]["map_context"]
    corrections = []
    if bev["map_context"] != measured_context:
        corrections.append(dict(field="map_context", measured=measured_context,
                                final=bev["map_context"]))
        facts["attack"]["map_context"] = bev["map_context"]
        facts["failure"]["ego_cause"] = ego_cause(
            facts["failure"]["interaction"], facts["attack"], facts["attack"]["ego_action"],
            facts["attack"]["ego_lane_keeping"], facts["failure"]["outcome"])

    evidence.write_manifest(run_dir, stage="images", resolved_token=facts["token"],
                            simulation=facts.get("simulation_record"))
    try:
        visual = visual_check(facts["archive"], facts["token"].split("-")[-1], facts, run_dir=run_dir)
    except (VisualCheckError, OSError, ValueError) as error:
        stage = getattr(error, "stage", "images")
        evidence.write_manifest(run_dir, stage="environment_optional", optional_error=str(error))
        visual = {"sample_frames": [], "corrections": [],
                  "notes": [f"可选环境补充失败（{stage}），关键测量数据保持不变：{error}"],
                  "weather": "unknown", "extra": ""}
    evidence.write_manifest(
        run_dir,
        stage="verification",
        final=facts["attack"],
        corrections=corrections,
    )

    evidence.write_manifest(run_dir, status="success", stage="done", finished_at=evidence.stamp())
    return {
        "status": "success",
        "tool": TOOL,
        "result": {
            "token": facts["token"],
            "violation_step": facts["violation_step"],
            "simulation": facts["simulation_record"],
            "map_location": facts["map_location"],
            "evidence_dir": str(run_dir),
            "failure": facts["failure"],
            "attack": facts["attack"],
            "environment": {"time": facts["environment"]["time"],
                            "weather": visual["weather"],
                            "extra": visual["extra"]},
            "verification": {
                "sample_frames": visual["sample_frames"],
                "corrections": corrections + list(visual["corrections"]),
                "notes": list(bev["notes"]) + list(visual["notes"]),
            },
        },
    }


def run(input_data: dict[str, Any], *, runs_root=None) -> dict[str, Any]:
    """Run one analysis inside a hard budget and a fresh evidence directory."""
    started = time.monotonic()
    try:
        token = _token(input_data)
        simulation = _simulation(input_data)
    except ValueError as error:
        return dict(status="error", tool=TOOL,
                    error=dict(type="input_error", stage="input", message=str(error),
                               evidence_dir=""))

    timeout = float(input_data.get("timeout") or DEFAULT_TIMEOUT_S)
    run_dir = evidence.new_run_dir(simulation, token, root=runs_root)
    evidence.write_manifest(run_dir, tool=TOOL, status="running", stage="start",
                            simulation=simulation, token=token, run_id=run_dir.name,
                            timeout_s=timeout, started_at=evidence.stamp())
    (run_dir / "request.json").write_text(json.dumps(input_data, ensure_ascii=False, indent=2))

    remaining = timeout - (time.monotonic() - started)
    if remaining <= 0:
        evidence.write_manifest(run_dir, status="failed", stage="start", error="budget exhausted")
        return _failure("timeout", run_dir, "start", f"analysis budget of {timeout:g}s was exhausted")

    payload = json.dumps(dict(input_data=input_data, run_dir=str(run_dir)), ensure_ascii=False)
    try:
        completed = run_logged(worker_command(), cwd=PROJECT_ROOT, log_path=run_dir / "worker.log",
                               timeout=remaining, input=payload, json_stdout=True)
    except subprocess.TimeoutExpired:
        stage = evidence.read_manifest(run_dir).get("stage") or "start"
        evidence.write_manifest(run_dir, status="failed", stage=stage, error="timeout",
                                finished_at=evidence.stamp())
        return _failure("timeout", run_dir, stage,
                        f"analysis exceeded its {timeout:g}s budget during {stage}")
    except OSError as error:
        evidence.write_manifest(run_dir, status="failed", stage="start", error="worker could not start",
                                worker_stderr=type(error).__name__, finished_at=evidence.stamp())
        return _failure("analysis_error", run_dir, "start",
                        f"analysis worker could not start ({type(error).__name__})")

    stderr = (completed.stderr or "").strip()[-2000:]
    if completed.returncode:
        stage = evidence.read_manifest(run_dir).get("stage") or "worker"
        evidence.write_manifest(run_dir, status="failed", stage=stage, error="worker failed",
                                worker_stderr=stderr, finished_at=evidence.stamp())
        return _failure("analysis_error", run_dir, stage,
                        f"analysis worker exited with code {completed.returncode}", worker_stderr=stderr)

    try:
        result = json.loads(completed.stdout)
    except ValueError:
        evidence.write_manifest(run_dir, status="failed", stage="worker", error="invalid worker output",
                                worker_stderr=stderr, finished_at=evidence.stamp())
        return _failure("analysis_error", run_dir, "worker",
                        "analysis worker produced no parseable JSON result", worker_stderr=stderr)
    if not isinstance(result, dict) or result.get("status") not in {"success", "error"}:
        evidence.write_manifest(run_dir, status="failed", stage="worker", error="invalid worker shape",
                                worker_stderr=stderr, finished_at=evidence.stamp())
        return _failure("analysis_error", run_dir, "worker",
                        "analysis worker returned an unexpected result shape", worker_stderr=stderr)

    if result.get("tool") != TOOL:
        evidence.write_manifest(run_dir, status="failed", stage="worker",
                                error="worker reported another tool", finished_at=evidence.stamp())
        return _failure("analysis_error", run_dir, "worker",
                        "analysis worker reported a different tool identity")

    if result["status"] == "success":
        try:
            validated = Output.model_validate(result.get("result"))
        except (ValidationError, ValueError, TypeError):
            evidence.write_manifest(run_dir, status="failed", stage="worker",
                                    error="result does not match the output contract",
                                    finished_at=evidence.stamp())
            return _failure("analysis_error", run_dir, "worker",
                            "analysis worker result does not match the output contract")
        payload = validated.model_dump()
        payload["evidence_dir"] = str(run_dir)
        result["result"] = payload
        evidence.write_manifest(run_dir, status="success", stage="done", finished_at=evidence.stamp())
        return result

    error = result.get("error")
    if (not isinstance(error, dict) or not isinstance(error.get("type"), str) or not error["type"]
            or not isinstance(error.get("message"), str)):
        evidence.write_manifest(run_dir, status="failed", stage="worker",
                                error="malformed error payload", finished_at=evidence.stamp())
        return _failure("analysis_error", run_dir, "worker",
                        "analysis worker returned a malformed error payload")
    error.setdefault("evidence_dir", str(run_dir))
    evidence.write_manifest(run_dir, status="failed", stage=error.get("stage") or "worker",
                            error=error["type"], finished_at=evidence.stamp())
    return result
