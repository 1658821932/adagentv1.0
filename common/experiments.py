"""Unified per-iteration / per-experiment workspace layout under outputs/experiments/.

Each iteration or ablation lives in a single self-contained directory:
  outputs/experiments/<exp_id>/
    meta.json              # parent, kind, hyperparams, metrics
    checkpoint.pth         # model checkpoint (file or symlink)
    eval_288/              # 288-scene Reactive closed-loop eval archive (for next-round Tool 01)
    base_sim_cache/        # per-model unedited base-scene sim trajectories (for Tool 04/05)
    used_base_scenes.json  # Tool 03 batch exclusion pool for this round
    generated_scenes/      # Tool 01-06 generated scenes for this round
    merged_rollout/        # merged RLFT rollout dataset for Tool 08
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = PROJECT_ROOT / "outputs/experiments"
EXPERIMENT_ENV_VAR = "ADAGENT_EXPERIMENT"


def experiments_root(root: Path | None = None) -> Path:
    return Path(root) if root is not None else EXPERIMENTS_ROOT


def experiment_dir(exp_id: str, *, root: Path | None = None) -> Path:
    name = exp_id.strip().strip("/")
    if not name or "/" in name or name in (".", "..", "active"):
        raise ValueError(f"Invalid experiment id: {exp_id!r}")
    return experiments_root(root) / name


def active_experiment_dir(*, root: Path | None = None) -> Path | None:
    base = experiments_root(root)
    env_val = os.environ.get(EXPERIMENT_ENV_VAR, "").strip()
    if env_val:
        cand = Path(env_val).expanduser()
        if not cand.is_absolute():
            cand = base / env_val
        return cand if cand.is_dir() else None
    active_link = base / "active"
    if active_link.exists() and active_link.is_dir():
        return active_link.resolve()
    return None


def read_experiment_meta(exp_path: Path) -> dict[str, Any]:
    meta_file = exp_path / "meta.json"
    if not meta_file.is_file():
        return {"exp_id": exp_path.name}
    try:
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("exp_id", exp_path.name)
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"exp_id": exp_path.name}


def write_experiment_meta(exp_path: Path, meta: dict[str, Any]) -> Path:
    exp_path.mkdir(parents=True, exist_ok=True)
    meta_file = exp_path / "meta.json"
    tmp = meta_file.with_name("meta.json.tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, meta_file)
    return meta_file


def init_experiment(
    exp_id: str,
    *,
    parent: str | None = None,
    kind: str = "iteration",
    notes: str = "",
    metrics: dict[str, Any] | None = None,
    root: Path | None = None,
) -> Path:
    exp_path = experiment_dir(exp_id, root=root)
    exp_path.mkdir(parents=True, exist_ok=True)
    if kind == "iteration":
        (exp_path / "base_sim_cache").mkdir(exist_ok=True)
    meta: dict[str, Any] = {
        "exp_id": exp_path.name,
        "kind": kind,
        "parent": parent,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes": notes,
    }
    if metrics:
        meta["metrics"] = metrics
    write_experiment_meta(exp_path, meta)
    return exp_path


def set_active_experiment(exp_id: str, *, root: Path | None = None) -> Path:
    base = experiments_root(root)
    target = experiment_dir(exp_id, root=root)
    if not target.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {target}")
    link = base / "active"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target.name)
    return target


def resolve_experiment_paths(exp_id: str | None = None, *, root: Path | None = None) -> dict[str, Any]:
    if exp_id:
        exp_path = experiment_dir(exp_id, root=root)
        if not exp_path.is_dir():
            raise FileNotFoundError(f"Experiment directory does not exist: {exp_path}")
    else:
        exp_path = active_experiment_dir(root=root)
        if exp_path is None:
            return {}
    meta = read_experiment_meta(exp_path)
    ckpt = exp_path / "checkpoint.pth"
    eval_dir = exp_path / "eval_288"
    sim_cache = exp_path / "base_sim_cache"
    used_pool = exp_path / "used_base_scenes.json"
    return {
        "exp_id": exp_path.name,
        "exp_dir": str(exp_path),
        "parent": meta.get("parent"),
        "kind": meta.get("kind"),
        "checkpoint": str(ckpt.resolve()) if ckpt.exists() else None,
        "eval_288_dir": str(eval_dir.resolve()) if eval_dir.exists() else None,
        "base_sim_cache": str(sim_cache.resolve()) if sim_cache.exists() else None,
        "used_base_scenes": str(used_pool),
        "meta": meta,
    }


def list_experiments(*, root: Path | None = None) -> list[dict[str, Any]]:
    base = experiments_root(root)
    if not base.is_dir():
        return []
    active = active_experiment_dir(root=root)
    rows: list[dict[str, Any]] = []
    for p in sorted(base.iterdir()):
        if p.name == "active" or not p.is_dir():
            continue
        meta = read_experiment_meta(p)
        rows.append({
            "exp_id": p.name,
            "active": active is not None and p.resolve() == active.resolve(),
            "kind": meta.get("kind", "unknown"),
            "parent": meta.get("parent"),
            "has_checkpoint": (p / "checkpoint.pth").exists(),
            "has_eval_288": (p / "eval_288").exists(),
            "has_base_sim_cache": (p / "base_sim_cache").exists(),
            "metrics": meta.get("metrics"),
            "notes": meta.get("notes", ""),
        })
    return rows
