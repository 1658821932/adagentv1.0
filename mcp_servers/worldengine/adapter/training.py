"""Run ordinary supervised WorldEngine fine-tuning on converted scene data."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


from .config import WorldEngineConfig, ALGENGINE_PYTHON
from .schemas import TrainingResult
from mcp_servers.runtime import runtime_task, current_run

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_WORLDENGINE_ROOT = WorldEngineConfig().project_root
DEFAULT_PYTHON = ALGENGINE_PYTHON
LOSS_RE = re.compile(r"loss\.imi:\s*([0-9]+(?:\.[0-9]+)?)")


def _path(value: Any, name: str, *, required: bool = True) -> Path | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"policy_trainer requires a non-empty '{name}'")
    result = Path(value).expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(f"policy_trainer {name} does not exist: {result}")
    return result


@runtime_task("worldengine", "training")
def train_policy(input_data: dict[str, Any]) -> TrainingResult:
    mode = input_data.get("training_mode", "ilft")
    if mode != "ilft":
        raise ValueError("policy_trainer currently supports training_mode='ilft' only")

    training_data = input_data.get("training_data")
    if isinstance(training_data, str):
        training_data = {"ann_file": training_data}
    if not isinstance(training_data, dict):
        training_data = {}
    ann_file = _path(training_data.get("ann_file") or input_data.get("ann_file"), "training_data.ann_file")
    token_file = _path(
        training_data.get("token_file")
        or training_data.get("nav_filter_path")
        or input_data.get("token_file"),
        "training_data.token_file",
    )
    checkpoint = _path(input_data.get("base_checkpoint") or input_data.get("checkpoint_path"), "base_checkpoint")

    worldengine_root = Path(
        input_data.get("worldengine_root") or DEFAULT_WORLDENGINE_ROOT
    ).expanduser().resolve()
    config = _path(
        input_data.get("config_path")
        or str(worldengine_root / "experiments/posttrain/waymo_090_ilft_adagent_smoke/e2e_vadv2_50pct_waymo_ilft.py"),
        "config_path",
    )
    python = Path(
        input_data.get("python")
        or os.environ.get("WORLDENGINE_TRAIN_PYTHON")
        or DEFAULT_PYTHON
    ).expanduser().resolve()
    if not python.is_file():
        raise FileNotFoundError(f"policy_trainer Python executable does not exist: {python}")

    epochs = int(input_data.get("epochs", input_data.get("total_epochs", 1)))
    if epochs < 1:
        raise ValueError("policy_trainer epochs must be positive")
    task = current_run("worldengine")
    work_dir = task.output / "training"
    work_dir.mkdir(parents=True, exist_ok=False)
    log_path = task.logs / "training.log"
    export_parent = input_data.get("work_dir")

    command = [
        str(python),
        str(worldengine_root / "projects/AlgEngine/scripts/train.py"),
        str(config),
        "--work-dir",
        str(work_dir),
        "--no-validate",
        "--cfg-options",
        f"load_from={checkpoint}",
        f"total_epochs={epochs}",
        f"runner.max_epochs={epochs}",
        f"data.train.ann_file={ann_file}",
        f"data.train.nav_filter_path={token_file}",
        f"data.train.finetune_yaml=[\"{token_file}\"]",
    ]
    env = os.environ.copy()
    env["WORLDENGINE_ROOT"] = str(worldengine_root)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(worldengine_root / "projects/AlgEngine"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    timeout = int(input_data.get("timeout", os.environ.get("POLICY_TRAINER_TIMEOUT", "86400")))
    if timeout <= 0:
        raise ValueError("policy_trainer timeout must be positive")

    try:
        completed = subprocess.run(
            command,
            cwd=worldengine_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        log_path.write_text(
            f"$ {' '.join(command)}\n\nstdout:\n{exc.stdout or ''}\n\nstderr:\n{exc.stderr or ''}\n",
            encoding="utf-8",
        )
        raise RuntimeError(f"policy_trainer timed out after {timeout}s; log: {log_path}") from exc

    log_path.write_text(
        f"$ {' '.join(command)}\n\nstdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout)[-4000:]
        raise RuntimeError(f"policy_trainer failed; log: {log_path}\n{tail}")

    checkpoints = sorted(work_dir.glob("epoch_*.pth"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise RuntimeError(f"training finished without a checkpoint: {work_dir}")
    losses = [float(value) for value in LOSS_RE.findall(completed.stdout + "\n" + completed.stderr)]
    result = {
        "run_dir": str(task.root),
        "training_mode": mode,
        "checkpoint": str(checkpoints[-1]),
        "work_dir": str(work_dir),
        "log_path": str(log_path),
        "epochs": epochs,
    }
    if losses:
        result["final_loss"] = losses[-1]
    if export_parent:
        import shutil
        export_dir = Path(export_parent).expanduser().resolve() / task.root.name
        shutil.copytree(work_dir, export_dir)
        result["export_dir"] = str(export_dir)
    return result
