"""Launch a WorldEngine closed-loop test with validated arguments."""

from __future__ import annotations

import os
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from .schemas import SimulationResult
from .config import WorldEngineConfig, SIMENGINE_PYTHON, ALGENGINE_PYTHON, AGENT_PYTHON
from mcp_servers.process import run_logged
from mcp_servers.runtime import runtime_task, current_run


@runtime_task("worldengine", "simulation")
def run_simulation(config_path: str, checkpoint_path: str, model_name: str, data_type: str,
                   react_type: str = "NR", asset_name: str | None = None,
                   added_name: str = "mcp", num_scenarios: int | None = None,
                   scenario_data_folder_path: str | None = None,
                   asset_folder_path: str | None = None,
                   scenario_pkl_file_name: str | None = None,
                   options: dict[str, Any] | None = None,
                   config: WorldEngineConfig | None = None) -> SimulationResult:
    options = options or {}
    runtime = config or WorldEngineConfig()
    script = runtime.testing_script()
    if not script.is_file():
        raise FileNotFoundError(f"WorldEngine testing script not found: {script}")
    cfg = Path(config_path).expanduser().resolve()
    ckpt = Path(checkpoint_path).expanduser().resolve()
    if not cfg.is_file():
        raise FileNotFoundError(f"WorldEngine config not found: {cfg}")
    if not ckpt.is_file():
        raise FileNotFoundError(f"WorldEngine checkpoint not found: {ckpt}")
    if react_type not in {"R", "NR"}:
        raise ValueError("react_type must be 'R' or 'NR'")
    env = current_run("worldengine").child_env()
    env["WORLDENGINE_ROOT"] = str(runtime.project_root)
    env.setdefault("SIMENGINE_PYTHON", str(SIMENGINE_PYTHON))
    env.setdefault("ALGENGINE_PYTHON", str(ALGENGINE_PYTHON))
    env.setdefault("WORLDENGINE_LAUNCH_PYTHON", str(AGENT_PYTHON))
    # WorldEngine shell scripts call "conda run"; ensure a conda installation
    # is on PATH even when the parent process was not a login shell.
    if not shutil.which("conda", path=env.get("PATH")):
        for conda_bin in (Path.home() / "miniconda3" / "bin", Path("/opt/conda/bin")):
            if (conda_bin / "conda").is_file():
                env["PATH"] = str(conda_bin) + os.pathsep + env.get("PATH", "")
                break
    # The WorldEngine shell scripts prepend to PYTHONPATH with
    # `PYTHONPATH=$SIMENGINE_ROOT:$ALGENGINE_ROOT:$PYTHONPATH`; when the parent
    # PYTHONPATH is empty this leaves a trailing colon, which makes Python put
    # the current working directory (WORLDENGINE_ROOT) on sys.path first and
    # shadows the real packages. Seed a neutral entry so the prepended project
    # roots resolve first.
    if not env.get("PYTHONPATH"):
        env["PYTHONPATH"] = str(runtime.project_root)
    if num_scenarios is not None:
        if num_scenarios < 1:
            raise ValueError("num_scenarios must be positive")
        env["NUM_SCENARIOS"] = str(num_scenarios)
    env["SCENARIO_ASSET_NAME"] = asset_name or data_type
    if scenario_data_folder_path:
        scenario_folder = Path(scenario_data_folder_path).expanduser()
        if not scenario_folder.is_absolute():
            scenario_folder = runtime.project_root / scenario_folder
        scenario_folder = scenario_folder.resolve()
        if not scenario_folder.is_dir():
            raise FileNotFoundError(f"WorldEngine scenario folder not found: {scenario_folder}")
        env["SCENARIO_DATA_FOLDER_PATH"] = str(scenario_folder)
    if asset_folder_path:
        env["SCENARIO_ASSET_FOLDER_PATH"] = str(Path(asset_folder_path).expanduser().resolve())
    if scenario_pkl_file_name:
        env["SCENARIO_PKL_FILE_NAME"] = str(scenario_pkl_file_name)
    option_env = {
        "with_dense_reward_manager": "WITH_DENSE_REWARD_MANAGER",
        "num_history": "NUM_HISTORY",
        "planner_interval_frames": "PLANNER_INTERVAL_FRAMES",
        "planner_history_stride": "PLANNER_HISTORY_STRIDE",
        "num_future": "NUM_FUTURE",
        "max_frame_per_scene": "MAX_FRAME_PER_SCENE",
        "sampling_poses": "SAMPLING_POSES",
        "ego_policy": "EGO_POLICY",
        "ego_controller": "EGO_CONTROLLER",
        "ego_client": "EGO_CLIENT",
        "ego_navigation": "EGO_NAVIGATION",
        "agent_policy": "AGENT_POLICY",
        "agent_navigation": "AGENT_NAVIGATION",
        "agent_client": "AGENT_CLIENT",
        "multi_agent_vadv2": "MULTI_AGENT_VADV2",
    }
    for key in ("planner_interval_frames", "planner_history_stride", "multi_agent_vadv2"):
        if options.get(key) is not None:
            raise ValueError(f"{key} is not supported by the native runner")
    for key, env_key in option_env.items():
        if key in options and options[key] is not None:
            value = options[key]
            env[env_key] = str(value).lower() if isinstance(value, bool) else str(value)
    # Serialize the source using the simulator's NumPy version. Inputs remain untouched.
    source_folder = Path(env.get("SCENARIO_DATA_FOLDER_PATH") or
        runtime.project_root / "data/sim_engine/scenarios/original" / data_type)
    source_scene = source_folder / (scenario_pkl_file_name or "all_scenarios.pkl")
    stage_dir = current_run("worldengine").input / "compatible_scenarios"
    staged_scene = stage_dir / "all_scenarios.pkl"
    prepare_command = [env["SIMENGINE_PYTHON"],
        str(Path(__file__).with_name("prepare_scenarios.py")),
        str(source_scene), str(staged_scene)]
    if num_scenarios is not None:
        prepare_command.append(str(num_scenarios))
    preparation = run_logged(prepare_command, cwd=runtime.project_root, env=env,
        log_path=current_run("worldengine").logs / "scene_preparation.log",
        timeout=int(options.get("timeout", runtime.timeout)))
    if preparation.returncode:
        raise RuntimeError("Scene preparation failed: " + preparation.stdout[-2000:])
    env["SCENARIO_DATA_FOLDER_PATH"] = str(stage_dir)
    env["SCENARIO_PKL_FILE_NAME"] = "all_scenarios.pkl"
    run_name = added_name or f"mcp-{datetime.now():%Y%m%d-%H%M%S}"
    command = ["bash", str(script), str(cfg), str(ckpt), model_name, data_type, react_type, asset_name or data_type, run_name]
    timeout = int(options.get("timeout", runtime.timeout))
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    task = current_run("worldengine")
    result_dir = task.output / "simulation"
    env["WORLDENGINE_TEST_PATH"] = str(result_dir)
    external_result_dir = result_dir
    log_path = task.logs / "simulation.log"
    completed = run_logged(command, cwd=runtime.project_root, env=env,
                           log_path=log_path, timeout=timeout)
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout).strip() or "WorldEngine simulation failed")
    report_files = sorted(result_dir.rglob("runner_report_*.json"))
    reports = []
    for report in report_files:
        try:
            reports.append(json.loads(report.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    if not reports:
        raise RuntimeError(
            f"WorldEngine produced no runner report; inspect {log_path}"
        )
    return {
        "run_dir": str(task.root),
        "command": command,
        "result_dir": str(result_dir),
        "external_result_dir": str(external_result_dir),
        "log_path": str(log_path),
        "reports": reports,
        "stdout_tail": completed.stdout[-4000:],
    }


@runtime_task("worldengine", "pdm_closed_loop")
def run_pdm(arguments):
    from mcp_servers.scene_editing.core.adversarial.pdm_tool import run_pdm_challenge
    task = current_run("worldengine")
    result = run_pdm_challenge(arguments, task.output/'pdm', task.logs/'pdm.log')
    summary = result['summary']
    baseline, attack = summary['results'].get('baseline'), summary['results']['attack']
    return dict(result_dir=str(task.root), reports=[summary], policy_result=str(task.output/'pdm/summary.json'),
        rollout_dir=None, reward_dir=None, candidate_rewards_present=False,
        pdm_challenge=dict(execution_mode='pdm_closed_loop', official_native_score=False,
            pdm_defense_succeeded=result['pdm_defense_succeeded'],
            traffic_light_policy=summary.get('traffic_light_policy'),
            takeover_alignment=summary.get('takeover_alignment'),
            background_geometry_policy=summary.get('background_geometry_policy'),
            probe_reused=summary.get('probe_reused',False),
            reuse_wall_seconds=summary.get('reuse_wall_seconds'),
            baseline_diagnostic_ran=baseline is not None, baseline_is_acceptance_gate=False,
            baseline=None if baseline is None else {key: baseline[key] for key in ('contact_samples', 'road_violation_samples', 'min_gap_m')},
            attack={key: attack[key] for key in ('contact_samples', 'road_violation_samples', 'min_gap_m',
                'min_target_gap_m', 'first_contact')}))


# Canonical entry points consumed by AdAgent tools.
from .client import run_policy
from .evaluation import filter_failure_cases
from .training import train_policy


# Native trajectory replay bypasses policy inference.
from .replay import run_replay
