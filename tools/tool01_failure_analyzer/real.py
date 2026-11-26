"""Failure analyzer: deterministic kinematics/BEV, BEV VLM script, optional 8-cam env."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from mcp_servers.process import run_logged
from mcp_servers.worldengine_failure_analysis.adapter.service import WfaError, failure_facts

from . import evidence
from .schemas import Output
from .visual_check import VisualCheckError, check as visual_check, check_bev

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TIMEOUT_S = 300.0
TOOL = "failure_analyzer"


def _token(input_data: dict[str, Any]) -> str:
    token = input_data.get("token")
    if not token:
        raise ValueError(
            "failure_analyzer requires a scene 'token' from a WorldEngine closed-loop run."
        )
    return str(token)


def _simulation(input_data: dict[str, Any]) -> str:
    simulation = input_data.get("simulation")
    if not simulation:
        raise ValueError("failure_analyzer requires the simulation record that produced the metrics")
    return str(simulation)


def worker_command():
    return [sys.executable, "-m", "tools.tool01_failure_analyzer.worker"]


def _failure(kind, run_dir, stage, message, **extra):
    return {
        "status": "error",
        "tool": TOOL,
        "error": {"type": kind, "stage": stage, "message": message, "evidence_dir": str(run_dir), **extra},
    }


def _timeout_of(error) -> bool:
    return bool(getattr(error, "timeout", False) or getattr(error, "details", {}).get("timeout", False))


def _target_class(vehicle: dict[str, Any] | None) -> str:
    if not vehicle:
        return "none"
    label = str(vehicle.get("label") or "").lower()
    size = vehicle.get("size_m") or [4.5, 2.0, 1.6]
    if label in ("bus", "truck"):
        return label
    if size[0] >= 6.5:
        return "bus"
    return "car"


def analyze(input_data: dict[str, Any], run_dir) -> dict[str, Any]:
    """Extract facts/BEV, classify via BEV VLM, and supplement environment."""
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
        return _failure("timeout" if _timeout_of(error) else "failure_facts_error", run_dir, "data", str(error))

    traj_target = run_dir / "trajectories.json"
    traj_source = Path(str(facts.get("trajectories_path") or ""))
    if not traj_source.is_file():
        evidence.write_manifest(run_dir, status="failed", stage="data", error="missing trajectories file")
        return _failure("failure_facts_error", run_dir, "data", "facts do not contain a readable trajectories_path")
    try:
        shutil.copyfile(traj_source, traj_target)
    finally:
        if traj_source.exists():
            traj_source.unlink()

    evidence.write_manifest(run_dir, stage="bev", resolved_token=facts["token"],
                            simulation=facts.get("simulation_record"))
    try:
        bev = check_bev(facts, run_dir=run_dir)
    except (VisualCheckError, OSError, ValueError) as error:
        stage = getattr(error, "stage", "bev")
        evidence.write_manifest(run_dir, status="failed", stage=stage, error=str(error))
        return _failure("bev_check_error", run_dir, stage, str(error))

    target_v = bev["selected_vehicle"]
    other_kin = None
    if target_v is not None:
        vk = target_v["kinematics"]
        other_kin = {
            "size_m": target_v["size_m"],
            "is_stationary": vk["is_stationary"],
            "speed_start_mps": vk["speed_start_mps"],
            "speed_violation_mps": vk["speed_violation_mps"],
            "displacement_m": vk["displacement_m"],
            "yaw_change_deg": vk["yaw_change_deg"],
            "tail_yaw_deg": vk["tail_yaw_deg"],
            "relative_heading_at_violation_deg": target_v["relative_heading_at_violation_deg"],
            "violation_distance_m": target_v["violation_distance_m"],
        }

    evidence.write_manifest(run_dir, stage="images", resolved_token=facts["token"],
                            simulation=facts.get("simulation_record"))
    try:
        visual = visual_check(facts["archive"], facts["token"].split("-")[-1], facts, run_dir=run_dir)
    except (VisualCheckError, OSError, ValueError) as error:
        stage = getattr(error, "stage", "images")
        evidence.write_manifest(run_dir, stage="environment_optional", optional_error=str(error))
        visual = {"sample_frames": [], "notes": [f"可选环境补充失败（{stage}）：{error}"],
                  "weather": "unknown", "extra": ""}

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
            "bev_image": bev["bev_image"],
            "failure": {
                "outcome": facts["outcome"],
                "failure_driver": bev["failure_driver"],
                "target_track": target_v["track"] if target_v else None,
                "target_class": _target_class(target_v),
                "map_context": bev["map_context"],
                "ego_behavior": bev["ego_behavior"],
                "other_behavior": bev["other_behavior"],
                "summary": bev["summary"],
            },
            "kinematics": {
                "ego": facts["ego_kinematics"],
                "other": other_kin,
                "trajectory_ref": str(traj_target),
            },
            "environment": {
                "time": facts["environment"]["time"],
                "weather": visual["weather"],
                "extra": visual["extra"],
            },
            "verification": {
                "sample_frames": visual["sample_frames"],
                "notes": list(visual["notes"]),
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
        return {"status": "error", "tool": TOOL,
                "error": {"type": "input_error", "stage": "input", "message": str(error), "evidence_dir": ""}}

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

    payload = json.dumps({"input_data": input_data, "run_dir": str(run_dir)}, ensure_ascii=False)
    try:
        completed = run_logged(worker_command(), cwd=PROJECT_ROOT, log_path=run_dir / "worker.log",
                               timeout=remaining, input=payload, json_stdout=True)
    except subprocess.TimeoutExpired:
        stage = evidence.read_manifest(run_dir).get("stage") or "start"
        evidence.write_manifest(run_dir, status="failed", stage=stage, error="timeout", finished_at=evidence.stamp())
        return _failure("timeout", run_dir, stage, f"analysis exceeded its {timeout:g}s budget during {stage}")
    except OSError as error:
        evidence.write_manifest(run_dir, status="failed", stage="start", error="worker could not start",
                                worker_stderr=type(error).__name__, finished_at=evidence.stamp())
        return _failure("analysis_error", run_dir, "start", f"analysis worker could not start ({type(error).__name__})")

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
    if not isinstance(result, dict) or result.get("status") not in {"success", "error"} or result.get("tool") != TOOL:
        evidence.write_manifest(run_dir, status="failed", stage="worker", error="invalid worker shape",
                                worker_stderr=stderr, finished_at=evidence.stamp())
        return _failure("analysis_error", run_dir, "worker", "analysis worker returned an unexpected result shape", worker_stderr=stderr)

    if result["status"] == "success":
        try:
            validated = Output.model_validate(result.get("result"))
        except (ValidationError, ValueError, TypeError) as error:
            evidence.write_manifest(run_dir, status="failed", stage="worker",
                                    error=f"result does not match output contract: {error}", finished_at=evidence.stamp())
            return _failure("analysis_error", run_dir, "worker", f"analysis worker result does not match output contract: {error}")
        dumped = validated.model_dump()
        dumped["evidence_dir"] = str(run_dir)
        result["result"] = dumped
        evidence.write_manifest(run_dir, status="success", stage="done", finished_at=evidence.stamp())
        return result

    error = result.get("error")
    if not isinstance(error, dict) or not isinstance(error.get("type"), str) or not error["type"] or not isinstance(error.get("message"), str):
        evidence.write_manifest(run_dir, status="failed", stage="worker", error="malformed error payload", finished_at=evidence.stamp())
        return _failure("analysis_error", run_dir, "worker", "analysis worker returned a malformed error payload")
    error.setdefault("evidence_dir", str(run_dir))
    evidence.write_manifest(run_dir, status="failed", stage=error.get("stage") or "worker",
                            error=error["type"], finished_at=evidence.stamp())
    return result
