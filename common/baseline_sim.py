"""Resolve and align unedited base-scene policy closed-loop simulation trajectories.

Used by Tool 04 (edit_planner) and Tool 05 (attack_vehicle_editor) so BEV rendering,
geometric feasibility checks, and attack-vehicle planning reference the current
policy's closed-loop simulation trajectory rather than human log-replay.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_BASE = PROJECT_ROOT / "data/navtrain_ep_per1/vad_baseline_cache"
DEFAULT_CHECKPOINT_STEM = "e2e_vadv2_50pct_ep8"
CHECKPOINT_ENV_VAR = "ADAGENT_POLICY_CHECKPOINT"
CACHE_DIR_ENV_VAR = "ADAGENT_VAD_BASELINE_CACHE_DIR"


def _checkpoint_stem(checkpoint: str | None = None) -> str:
    raw = checkpoint or os.environ.get(CHECKPOINT_ENV_VAR) or ""
    if raw.strip():
        return Path(raw.strip()).stem
    return DEFAULT_CHECKPOINT_STEM


def _candidate_cache_roots(checkpoint: str | None = None, cache_base: Path | None = None) -> list[Path]:
    roots: list[Path] = []
    env_dir = os.environ.get(CACHE_DIR_ENV_VAR, "").strip()
    if env_dir:
        roots.append(Path(env_dir).expanduser())
    if cache_base is None:
        from common.experiments import active_experiment_dir
        exp_dir = active_experiment_dir()
        if exp_dir is not None:
            exp_cache = exp_dir / "base_sim_cache"
            if exp_cache.is_dir():
                roots.append(exp_cache)
    base = Path(cache_base) if cache_base is not None else DEFAULT_CACHE_BASE
    stem = _checkpoint_stem(checkpoint)
    roots.append(base / stem)
    if stem != DEFAULT_CHECKPOINT_STEM:
        roots.append(base / DEFAULT_CHECKPOINT_STEM)
    return roots


def resolve_baseline_frames_path(
    scene_token: str | None,
    *,
    checkpoint: str | None = None,
    cache_base: Path | None = None,
) -> Path | None:
    """Return cached baseline_frames.json for scene_token if present and non-empty."""
    if not isinstance(scene_token, str) or not scene_token.strip():
        return None
    short = scene_token.strip().split("-")[-1]
    for root in _candidate_cache_roots(checkpoint=checkpoint, cache_base=cache_base):
        cand = root / short / "baseline_frames.json"
        if cand.is_file() and cand.stat().st_size > 100:
            return cand
    return None


def _ego_path_in_origin(frames: list[dict[str, Any]], origin_ego2global: np.ndarray, t0_us: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import numpy as np
    timestamps = np.array([f["timestamp"] for f in frames], dtype=float)
    times = (timestamps - t0_us) / 1e6
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("Missing or non-increasing timestamps")
    ego_world = np.array([f["ego2global"] for f in frames], dtype=float)
    origin_yaw = float(np.arctan2(origin_ego2global[1, 0], origin_ego2global[0, 0]))
    global_yaw = np.unwrap(np.arctan2(ego_world[:, 1, 0], ego_world[:, 0, 0]))
    # Align initial unwrap branch to origin_yaw
    yaw_offset = np.round((global_yaw[0] - origin_yaw) / (2.0 * np.pi)) * (2.0 * np.pi)
    global_yaw = global_yaw - yaw_offset
    origin_rot = np.array([[np.cos(origin_yaw), -np.sin(origin_yaw)],
                           [np.sin(origin_yaw), np.cos(origin_yaw)]])
    ego_xy = (ego_world[:, :2, 3] - origin_ego2global[:2, 3]) @ origin_rot
    ego_yaw = global_yaw - origin_yaw
    return times, ego_xy, ego_yaw


def _candidate_tokens_for_scene(scene_dir: Path | None, recorded_frames: list[dict[str, Any]]) -> list[str]:
    candidates: list[str] = []
    if scene_dir is not None:
        scene_path = Path(scene_dir).resolve()
        candidates.append(scene_path.name)
        if scene_path.parent != scene_path:
            candidates.append(scene_path.parent.name)
    if recorded_frames:
        f0 = recorded_frames[0]
        for key in ("scene_token", "log_name", "scene_name"):
            val = f0.get(key)
            if isinstance(val, str) and val.strip():
                candidates.append(val.strip())
    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        tok = item.split("-")[-1]
        if tok and tok not in seen:
            seen.add(tok)
            deduped.append(tok)
    return deduped


def reference_ego_path_from_scene(
    scene_dir: Path | str | None,
    recorded_frames: list[dict[str, Any]],
    *,
    baseline_frames_path: Path | str | None = None,
    checkpoint: str | None = None,
    cache_base: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return (times, ego_xy, ego_yaw, meta) aligned to recorded_frames timestamps.

    Prefers the cached policy closed-loop simulation trajectory (baseline_frames.json)
    in frame-0 local coordinates; smoothly extrapolates constant velocity past the
    simulation horizon if recorded_frames is longer. Falls back to recorded_frames
    only when no baseline simulation cache exists.
    """
    import numpy as np
    origin = np.array(recorded_frames[0]["ego2global"], dtype=float)
    t0_us = float(recorded_frames[0]["timestamp"])
    rec_times, rec_xy, rec_yaw = _ego_path_in_origin(recorded_frames, origin, t0_us)

    bp: Path | None = Path(baseline_frames_path) if baseline_frames_path else None
    if bp is None or not bp.is_file():
        bp = None
        for tok in _candidate_tokens_for_scene(Path(scene_dir) if scene_dir else None, recorded_frames):
            bp = resolve_baseline_frames_path(tok, checkpoint=checkpoint, cache_base=cache_base)
            if bp is not None:
                break

    if bp is None:
        return rec_times, rec_xy, rec_yaw, {
            "source": "recorded_replay",
            "baseline_frames_path": None,
            "sim_frames": 0,
            "recorded_frames": len(recorded_frames),
        }

    sim_frames = json.loads(bp.read_text(encoding="utf-8"))
    if not isinstance(sim_frames, list) or len(sim_frames) < 2:
        return rec_times, rec_xy, rec_yaw, {
            "source": "recorded_replay",
            "baseline_frames_path": str(bp),
            "sim_frames": 0,
            "recorded_frames": len(recorded_frames),
        }

    sim_times, sim_xy, sim_yaw = _ego_path_in_origin(sim_frames, origin, t0_us)
    frame0_err = float(np.linalg.norm(sim_xy[0]))
    if frame0_err > 0.5:
        return rec_times, rec_xy, rec_yaw, {
            "source": "recorded_replay",
            "baseline_frames_path": str(bp),
            "reason": f"frame0_mismatch_{frame0_err:.2f}m",
            "sim_frames": len(sim_frames),
            "recorded_frames": len(recorded_frames),
        }

    ego_xy = np.empty_like(rec_xy)
    ego_yaw = np.empty_like(rec_yaw)
    t_end = float(sim_times[-1])
    dt_last = float(sim_times[-1] - sim_times[-2])
    vel_last = (sim_xy[-1] - sim_xy[-2]) / dt_last if dt_last > 1e-6 else np.zeros(2)
    yaw_last = float(sim_yaw[-1])

    for i, t in enumerate(rec_times):
        if t <= t_end + 1e-6:
            ego_xy[i, 0] = float(np.interp(t, sim_times, sim_xy[:, 0]))
            ego_xy[i, 1] = float(np.interp(t, sim_times, sim_xy[:, 1]))
            ego_yaw[i] = float(np.interp(t, sim_times, sim_yaw))
        else:
            dt = float(t - t_end)
            ego_xy[i] = sim_xy[-1] + vel_last * dt
            ego_yaw[i] = yaw_last

    return rec_times, ego_xy, ego_yaw, {
        "source": "policy_simulation",
        "baseline_frames_path": str(bp),
        "sim_frames": len(sim_frames),
        "recorded_frames": len(recorded_frames),
    }
