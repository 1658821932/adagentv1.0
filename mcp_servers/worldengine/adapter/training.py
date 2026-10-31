"""Mode-specific WorldEngine fine-tuning with isolated runtime data."""
from __future__ import annotations
import json
import os
import shutil
from pathlib import Path
from typing import Any
from .config import WorldEngineConfig, ALGENGINE_PYTHON
from .schemas import TrainingResult
from mcp_servers.process import run_logged
from mcp_servers.runtime import runtime_task, current_run

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORLDENGINE_ROOT = WorldEngineConfig().project_root
DEFAULT_PYTHON = ALGENGINE_PYTHON

def _path(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"policy_trainer requires a non-empty '{name}'")
    p = Path(value).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"policy_trainer {name} does not exist: {p}")
    return p

@runtime_task("worldengine", "training")
def train_policy(input_data: dict[str, Any]) -> TrainingResult:
    # Direct adapter calls receive the same validation as Agent tool calls.
    from tools.tool15_policy_trainer.schemas import Input
    args = Input.model_validate(input_data).model_dump()
    mode, data = args["training_mode"], args["training_data"]
    root = Path(args["worldengine_root"] or DEFAULT_WORLDENGINE_ROOT).expanduser().resolve()
    checkpoint = _path(args["base_checkpoint"], "base_checkpoint")
    _path(data["token_file"], "training_data.token_file")
    default = (root / "experiments/posttrain/e2e_vadv2_50pct_waymo_ilft.py" if mode == "ilft"
               else root / "projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct_rlft_rare_rollout.py")
    config = _path(args["config_path"] or str(default), "config_path")
    python = _path(args["python"] or os.getenv("WORLDENGINE_TRAIN_PYTHON") or str(DEFAULT_PYTHON), "python")
    _path(str(root / "projects/AlgEngine/scripts/train.py"), "train_entry")
    if mode == "ilft":
        _path(data["ann_file"], "training_data.ann_file")
    else:
        _path(data["vocab_path"], "training_data.vocab_path")
        if not Path(data["rollout_dir"]).expanduser().is_dir():
            raise FileNotFoundError("training_data.rollout_dir must be a WE openscene_format directory")

    task = current_run("worldengine")
    work = task.output / "training"
    work.mkdir(parents=True, exist_ok=False)
    request = {**args, "config_path": str(config), "base_checkpoint": str(checkpoint),
               "worldengine_root": str(root), "work_dir": str(work)}
    request_path = task.input / "training_request.json"
    request_path.write_text(json.dumps(request, indent=2))
    helper = Path(__file__).with_name("prepare_training.py")
    env = task.child_env()
    env["WORLDENGINE_ROOT"] = str(root)
    env["PYTHONPATH"] = os.pathsep.join([str(root / "projects/AlgEngine"),
        str(helper.parent), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    preparation_log = task.logs / "preparation.log"
    prepared = run_logged([str(python), str(helper), str(request_path)],
        cwd=root, env=env, log_path=preparation_log, timeout=args["timeout"])
    if prepared.returncode:
        raise RuntimeError(f"Training data preparation failed; log: {preparation_log}\n{prepared.stdout[-3000:]}")
    preparation = json.loads((work / "data_preparation.json").read_text())
    env["WORLDENGINE_ROOT"] = preparation["runtime_root"]
    command = [str(python), str(root / "projects/AlgEngine/scripts/train.py"),
        str(work / "training_config.py"), "--work-dir", str(work),
        "--no-validate", "--seed", str(args["seed"])]
    log_path = task.logs / "training.log"
    completed = run_logged(command, cwd=root, env=env, log_path=log_path, timeout=args["timeout"])
    if completed.returncode:
        raise RuntimeError(f"policy_trainer failed; log: {log_path}\n{completed.stdout[-3000:]}")
    checkpoints = sorted(work.glob("epoch_*.pth"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise RuntimeError(f"Training finished without a checkpoint: {work}")
    audit = json.loads((work / "training_audit.json").read_text())
    if not audit["finite_losses"] or not audit["changed_parameter_count"]:
        raise RuntimeError("Training must have finite losses and actual parameter updates")
    result = {"run_dir": str(task.root), "training_mode": mode,
        "checkpoint": str(checkpoints[-1]), "work_dir": str(work),
        "log_path": str(log_path), "epochs": args["epochs"],
        "training_config": str(work / "training_config.py"),
        "data_preparation": preparation, "training_audit": audit}
    if "loss" in audit["losses"]:
        result["final_loss"] = audit["losses"]["loss"]
    if args["work_dir"]:
        export = Path(args["work_dir"]).expanduser().resolve() / task.root.name
        shutil.copytree(work, export)
        result["export_dir"] = str(export)
    return result
