"""Launch a WorldEngine closed-loop test with validated arguments."""

from __future__ import annotations

import os
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import WorldEngineConfig


def _prepare_dggt_scene(
    dggt_output_dir: str, data_type: str, runtime: WorldEngineConfig,
    instance_id: int, translation_scale: float,
) -> tuple[str, str, dict[str, Any]]:
    """Convert one edited DGGT instance into the WorldEngine custom-data layout."""
    from mcp_servers.worldengine.adapter.scene_conversion import prepare

    agent_root = Path(__file__).resolve().parents[3]
    source = Path(dggt_output_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"DGGT output directory not found: {source}")
    staged = agent_root / "mcp_servers" / "worldengine" / "input" / data_type
    if staged.exists():
        shutil.rmtree(staged)
    staging_result = prepare(source, staged, instance_id)

    builder = runtime.project_root / "scripts" / "build_highway_new_dynamic_input.py"
    if not builder.is_file():
        raise FileNotFoundError(f"WorldEngine scene builder not found: {builder}")
    output_root = runtime.project_root / "data" / "sim_engine" / "custom" / data_type
    python_candidates = (
        agent_root / "environments" / "algengine" / ".conda" / "bin" / "python",
        agent_root / "environments" / "dggt" / ".conda" / "bin" / "python",
    )
    builder_python = next((path for path in python_candidates if path.is_file()), None)
    if builder_python is None:
        raise FileNotFoundError("No compatible WorldEngine scene-builder Python environment found")
    scene_id = f"{data_type}-000"
    completed = subprocess.run(
        [str(builder_python), str(builder), "--source", str(staged),
         "--output-root", str(output_root), "--scene-id", scene_id,
         "--translation-scale", str(translation_scale)],
        cwd=runtime.project_root, capture_output=True, text=True, check=False,
    )
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout).strip() or "WorldEngine scene conversion failed")
    return (
        str(output_root / "scenarios"),
        str(output_root / "assets"),
        {"staging": staging_result, "output_root": str(output_root),
         "translation_scale": translation_scale, "builder_stdout": completed.stdout[-2000:]},
    )


def run_simulation(config_path: str, checkpoint_path: str, model_name: str, data_type: str,
                   react_type: str = "NR", asset_name: str | None = None,
                   added_name: str = "mcp", num_scenarios: int | None = None,
                   scenario_data_folder_path: str | None = None,
                   asset_folder_path: str | None = None,
                   scenario_pkl_file_name: str | None = None,
                   options: dict[str, Any] | None = None,
                   config: WorldEngineConfig | None = None) -> dict[str, Any]:
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
    conversion = None
    if options.get("dggt_output_dir"):
        scenario_data_folder_path, asset_folder_path, conversion = _prepare_dggt_scene(
            str(options["dggt_output_dir"]), data_type, runtime,
            int(options.get("instance_id", 0)),
            float(options.get("translation_scale", 10000.0)),
        )
        scenario_pkl_file_name = "all_scenarios.pkl"
    env = os.environ.copy()
    env["WORLDENGINE_ROOT"] = str(runtime.project_root)
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
        if not scenario_folder.is_relative_to(runtime.project_root):
            raise ValueError("WorldEngine scenario folder must be below WORLDENGINE_ROOT")
        env["SCENARIO_DATA_FOLDER_PATH"] = str(scenario_folder)
    if asset_folder_path:
        env["SCENARIO_ASSET_FOLDER_PATH"] = str(Path(asset_folder_path).expanduser().resolve())
    if scenario_pkl_file_name:
        env["SCENARIO_PKL_FILE_NAME"] = str(scenario_pkl_file_name)
    option_env = {
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
    for key, env_key in option_env.items():
        if key in options and options[key] is not None:
            value = options[key]
            env[env_key] = str(value).lower() if isinstance(value, bool) else str(value)
    run_name = added_name or f"mcp-{datetime.now():%Y%m%d-%H%M%S}"
    command = ["bash", str(script), str(cfg), str(ckpt), model_name, data_type, react_type, asset_name or data_type, run_name]
    timeout = int(options.get("timeout", runtime.timeout))
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    external_result_dir = runtime.project_root / "experiments" / "closed_loop_exps" / model_name / f"{data_type}_{react_type}"
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    local_run_dir = runtime.output_dir / run_id / str(model_name)
    local_result_dir = local_run_dir / f"{data_type}_{react_type}"
    log_path = runtime.logs_dir / f"{run_id}.log"
    runtime.output_dir.mkdir(parents=True, exist_ok=True)
    runtime.logs_dir.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(command, cwd=runtime.project_root, env=env, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:
        log_path.write_text(f"$ {' '.join(command)}\n{exc}\n", encoding="utf-8")
        raise
    log_path.write_text(
        f"$ {' '.join(command)}\n\n[stdout]\n{completed.stdout}\n\n[stderr]\n{completed.stderr}\n",
        encoding="utf-8",
    )
    if external_result_dir.is_dir():
        shutil.copytree(external_result_dir, local_result_dir, dirs_exist_ok=True)
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout).strip() or "WorldEngine simulation failed")
    result_dir = local_result_dir if local_result_dir.is_dir() else external_result_dir
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
        "command": command,
        "result_dir": str(result_dir),
        "external_result_dir": str(external_result_dir),
        "log_path": str(log_path),
        "reports": reports,
        "stdout_tail": completed.stdout[-4000:],
        "scene_conversion": conversion,
    }
