"""Diagnostic closed-loop ego behavior and PDM sub-score reporting (read-only on WE artifacts)."""
from __future__ import annotations

import csv
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

from common.pdm_scores import score_file as resolve_score_file, score_files as resolve_score_files
from mcp_servers.worldengine.adapter.evaluation_horizon import DEFAULT_NATIVE_HISTORY

PDM_COLUMNS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "score",
)
STOP_SPEED_THRESHOLD_MPS = 0.2


def _aliases(value: Any) -> set[str]:
    if value is None:
        return set()
    text = str(value).strip()
    if not text:
        return set()
    if text.endswith(".pkl"):
        text = text[:-4]
    out = {text}
    if "-" in text:
        tail = text.rsplit("-", 1)[-1]
        if len(tail) == 16:
            out.add(tail.lower())
    return out


def read_pdm_subscores(result_dir: str | Path) -> dict[str, Any] | None:
    """Read official PDM sub-scores verbatim from all_scenes_pdm_averages_*.csv."""
    if not resolve_score_files(result_dir):
        return None
    path = resolve_score_file(result_dir)
    scenes: dict[str, dict[str, Any]] = {}
    overall: dict[str, Any] | None = None
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            token = (row.get("token") or "").strip()
            if not token:
                continue
            entry: dict[str, Any] = {}
            for col in PDM_COLUMNS:
                raw = row.get(col)
                if raw is not None and raw != "":
                    entry[col] = round(float(raw), 5)
            fvs = (row.get("first_violation_step") or "").strip()
            entry["first_violation_step"] = None if not fvs else float(fvs)
            if token == "overall_average":
                overall = entry
            else:
                scenes[token] = entry
    if not scenes and overall is None:
        return None
    return {
        "scores_file": str(path),
        "overall_average": overall,
        "scenes": scenes,
    }


def _load_scenarios(input_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    folder = input_data.get("scenario_data_folder_path")
    pkl_name = input_data.get("scenario_pkl_file_name")
    if not folder or not pkl_name:
        return {}
    scenario_path = Path(folder).expanduser() / pkl_name
    if not scenario_path.is_file():
        return {}
    from mcp_servers.worldengine.adapter.prepare_scenarios import SceneDataUnpickler

    with scenario_path.open("rb") as stream:
        payload = SceneDataUnpickler(stream).load()
    if isinstance(payload, dict) and "object_track" in payload:
        items = [payload]
    elif isinstance(payload, dict):
        items = [v for v in payload.values() if isinstance(v, dict)]
    elif isinstance(payload, list):
        items = [v for v in payload if isinstance(v, dict)]
    else:
        return {}

    index: dict[str, dict[str, Any]] = {}
    for scene in items:
        for key in (scene.get("id"), scene.get("scene_id"), scene.get("scenario_id"), scene.get("scene_token"), scene.get("token")):
            for alias in _aliases(key):
                index.setdefault(alias, scene)
    return index


def _kinematics(xy: Any, yaw: Any, times_s: Any) -> dict[str, Any]:
    import numpy as np

    yaw_u = np.unwrap(yaw)
    if len(xy) < 2:
        return {
            "absolute_progress_m": 0.0,
            "net_displacement_m": 0.0,
            "initial_speed_mps": 0.0,
            "final_speed_mps": 0.0,
            "min_speed_mps": 0.0,
            "max_speed_mps": 0.0,
            "yaw_change_deg": 0.0,
            "signed_yaw_change_deg": 0.0,
            "speeds": np.zeros(0, float),
            "dt": np.zeros(0, float),
        }
    dt = np.diff(times_s)
    diffs = np.diff(xy, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    safe_dt = np.where(dt > 1e-6, dt, 0.5)
    speeds = dists / safe_dt
    return {
        "absolute_progress_m": round(float(dists.sum()), 3),
        "net_displacement_m": round(float(np.linalg.norm(xy[-1] - xy[0])), 3),
        "initial_speed_mps": round(float(speeds[0]), 3),
        "final_speed_mps": round(float(speeds[-1]), 3),
        "min_speed_mps": round(float(speeds.min()), 3),
        "max_speed_mps": round(float(speeds.max()), 3),
        "yaw_change_deg": round(float(np.degrees(abs(yaw_u[-1] - yaw_u[0]))), 2),
        "signed_yaw_change_deg": round(float(np.degrees(yaw_u[-1] - yaw_u[0])), 2),
        "speeds": speeds,
        "dt": safe_dt,
    }


def _stop_detection(speeds: Any, dt: Any, eval_frames: list[dict[str, Any]], time0: float, us_scale: float) -> tuple[bool, int | None, float | None, int]:
    import numpy as np

    if len(speeds) == 0:
        return False, None, None, 0
    stopped_mask = speeds <= STOP_SPEED_THRESHOLD_MPS
    stopped_count = int(np.sum(stopped_mask))
    if not stopped_mask[-1]:
        return False, None, None, stopped_count
    first_tail = len(stopped_mask) - 1
    while first_tail > 0 and stopped_mask[first_tail - 1]:
        first_tail -= 1
    trailing_intervals = len(stopped_mask) - first_tail
    trailing_duration = float(np.sum(dt[first_tail:]))
    detected = trailing_duration >= 0.99 or trailing_intervals >= 2
    if not detected:
        return False, None, None, stopped_count
    frame_idx = int(eval_frames[first_tail]["frame_idx"])
    time_s = round((float(eval_frames[first_tail]["timestamp"]) - time0) / us_scale, 3)
    return True, frame_idx, time_s, stopped_count


def _reference_summary(scene: dict[str, Any] | None, start_step: int, end_step: int, default_dt: float = 0.5) -> dict[str, Any]:
    if not scene:
        return {}
    import numpy as np

    summary: dict[str, Any] = {}
    native = (scene.get("metadata") or {}).get("native_export") or {}
    rec_xy_raw = native.get("recorded_ego_rear_axle_global")
    rec_yaw_raw = native.get("recorded_ego_heading_rad")
    if rec_xy_raw is not None and rec_yaw_raw is not None:
        rec_xy = np.asarray(rec_xy_raw, float)[start_step:end_step + 1]
        rec_yaw = np.asarray(rec_yaw_raw, float)[start_step:end_step + 1]
        rec_t_raw = native.get("recorded_ego_time_s")
        rec_t = np.asarray(rec_t_raw, float)[start_step:end_step + 1] if rec_t_raw is not None else np.arange(len(rec_xy), dtype=float) * default_dt
        if len(rec_xy) >= 2:
            rk = _kinematics(rec_xy, rec_yaw, rec_t)
            summary["recorded_absolute_progress_m"] = rk["absolute_progress_m"]
            summary["recorded_net_displacement_m"] = rk["net_displacement_m"]
            summary["recorded_yaw_change_deg"] = rk["yaw_change_deg"]
            summary["recorded_final_speed_mps"] = rk["final_speed_mps"]
    ego_state = ((scene.get("object_track") or {}).get("ego") or {}).get("state") or {}
    exp_pos_raw = ego_state.get("position")
    exp_yaw_raw = ego_state.get("heading")
    if exp_pos_raw is not None and exp_yaw_raw is not None:
        exp_xy = np.asarray(exp_pos_raw, float)[start_step:end_step + 1, :2]
        exp_yaw = np.asarray(exp_yaw_raw, float)[start_step:end_step + 1]
        rec_t_raw = native.get("recorded_ego_time_s")
        exp_t = np.asarray(rec_t_raw, float)[start_step:end_step + 1] if rec_t_raw is not None else np.arange(len(exp_xy), dtype=float) * default_dt
        if len(exp_xy) >= 2:
            ek = _kinematics(exp_xy, exp_yaw, exp_t)
            summary["expert_absolute_progress_m"] = ek["absolute_progress_m"]
            summary["expert_yaw_change_deg"] = ek["yaw_change_deg"]
            summary["expert_final_speed_mps"] = ek["final_speed_mps"]
            exp_stopped = bool(len(ek["speeds"]) >= 2 and np.all(ek["speeds"][-2:] <= STOP_SPEED_THRESHOLD_MPS))
            summary["expert_terminal_stopped"] = exp_stopped
    return summary


def _summarize_policy_behavior_in_process(
    result_dir: str | Path,
    input_data: dict[str, Any],
    *,
    pdm_subscores: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    import numpy as np

    rollout = Path(result_dir) / "WE_output/openscene_format"
    meta_dir = rollout / "meta_datas"
    if not meta_dir.is_dir():
        return None
    sources = [meta_dir / "combined.pkl"] if (meta_dir / "combined.pkl").is_file() else sorted(meta_dir.glob("*.pkl"))
    if not sources:
        return None

    rows: list[dict[str, Any]] = []
    class _CompatUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            if module.startswith("numpy._core"):
                try:
                    return super().find_class(module, name)
                except ModuleNotFoundError:
                    return super().find_class("numpy.core" + module[len("numpy._core"):], name)
            return super().find_class(module, name)

    for path in sources:
        with path.open("rb") as stream:
            chunk = _CompatUnpickler(stream).load()
        if isinstance(chunk, list):
            rows.extend(item for item in chunk if isinstance(item, dict))
    if not rows:
        return None

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        scene_key = str(row.get("log_name") or row.get("scene_token") or row.get("token") or "scene_0")
        grouped.setdefault(scene_key, []).append(row)

    scenario_index = _load_scenarios(input_data)
    num_history = int(input_data.get("num_history") or DEFAULT_NATIVE_HISTORY)
    start_step = max(0, num_history - 1)
    pdm_scenes = (pdm_subscores or {}).get("scenes") or {}
    summaries: list[dict[str, Any]] = []

    for scene_key in sorted(grouped):
        frames = sorted(grouped[scene_key], key=lambda item: int(item.get("frame_idx", 0)))
        if len(frames) <= start_step + 1:
            continue
        max_end = len(frames) - 1
        requested_end = input_data.get("max_frame_per_scene")
        end_step = min(max_end, int(requested_end) - 1) if isinstance(requested_end, int) and requested_end > start_step + 1 else max_end
        eval_frames = frames[start_step:end_step + 1]
        if len(eval_frames) < 2:
            continue

        poses = np.asarray([f["ego2global"] for f in eval_frames], float)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            continue
        xy = poses[:, :2, 3]
        yaw = np.arctan2(poses[:, 1, 0], poses[:, 0, 0])
        raw_times = np.asarray([float(f["timestamp"]) for f in eval_frames], float)
        us_scale = 1_000_000.0 if len(raw_times) > 1 and float(np.median(np.diff(raw_times))) > 1000.0 else 1.0
        time0 = float(frames[0]["timestamp"])
        times_s = (raw_times - time0) / us_scale
        default_dt = float(np.median(np.diff(times_s))) if len(times_s) > 1 else 0.5

        kin = _kinematics(xy, yaw, times_s)
        stopped_detected, stopped_from_step, stopped_from_time_s, stopped_intervals = _stop_detection(
            kin["speeds"], kin["dt"], eval_frames, time0, us_scale
        )

        matched_scene = None
        for candidate_id in (scene_key, eval_frames[0].get("scene_token"), eval_frames[0].get("log_name"), eval_frames[0].get("token")):
            for alias in _aliases(candidate_id):
                if alias in scenario_index:
                    matched_scene = scenario_index[alias]
                    break
            if matched_scene is not None:
                break

        ref = _reference_summary(matched_scene, start_step, end_step, default_dt=default_dt)
        rec_prog = ref.get("recorded_absolute_progress_m")
        exp_prog = ref.get("expert_absolute_progress_m")
        rec_yaw_deg = ref.get("recorded_yaw_change_deg")

        progress_vs_recorded = round(kin["absolute_progress_m"] / rec_prog, 4) if rec_prog and rec_prog > 1e-6 else None
        progress_vs_expert = round(kin["absolute_progress_m"] / exp_prog, 4) if exp_prog and exp_prog > 1e-6 else None
        expert_vs_recorded = round(exp_prog / rec_prog, 4) if exp_prog is not None and rec_prog and rec_prog > 1e-6 else None

        turn_ratio: float | None = None
        if rec_yaw_deg is not None and rec_yaw_deg > 1e-6:
            turn_ratio = round(kin["yaw_change_deg"] / rec_yaw_deg, 4)
        # Describe actual motion only. Distance/yaw percentages and a terminal
        # wait do not establish whether a scene-specific maneuver was completed.
        behavior_class = "stopped_at_window_end" if stopped_detected else "moving_at_window_end"

        matched_pdm = None
        for alias in _aliases(scene_key) | _aliases(eval_frames[0].get("scene_token")):
            if alias in pdm_scenes:
                matched_pdm = pdm_scenes[alias]
                break
        if matched_pdm is None:
            for pdm_key, pdm_val in pdm_scenes.items():
                if _aliases(pdm_key) & (_aliases(scene_key) | _aliases(eval_frames[0].get("scene_token"))):
                    matched_pdm = pdm_val
                    break

        summaries.append({
            "scene_id": scene_key,
            "takeover_step": start_step,
            "end_step": end_step,
            "takeover_time_s": round(float(times_s[0]), 3),
            "end_time_s": round(float(times_s[-1]), 3),
            "behavior_classification": behavior_class,
            "absolute_progress_m": kin["absolute_progress_m"],
            "net_displacement_m": kin["net_displacement_m"],
            "initial_speed_mps": kin["initial_speed_mps"],
            "final_speed_mps": kin["final_speed_mps"],
            "min_speed_mps": kin["min_speed_mps"],
            "max_speed_mps": kin["max_speed_mps"],
            "stopped_detected": stopped_detected,
            "stopped_from_step": stopped_from_step,
            "stopped_from_time_s": stopped_from_time_s,
            "stopped_interval_count": stopped_intervals,
            "yaw_change_deg": kin["yaw_change_deg"],
            "signed_yaw_change_deg": kin["signed_yaw_change_deg"],
            "turn_completed": None,  # Deprecated compatibility field, no automatic verdict.
            "maneuver_completion_status": "not_assessed",
            "yaw_change_ratio_vs_recorded": turn_ratio,
            "turn_completion_ratio_vs_recorded": turn_ratio,
            "progress_ratio_vs_recorded": progress_vs_recorded,
            "progress_ratio_vs_expert": progress_vs_expert,
            "expert_progress_ratio_vs_recorded": expert_vs_recorded,
            **ref,
            "pdm_subscores": matched_pdm,
        })

    if not summaries:
        return None

    output: dict[str, Any] = {
        "status": "completed",
        "scene_count": len(summaries),
        "scenes": summaries,
    }
    if len(summaries) == 1:
        primary = summaries[0]
        for key in (
            "behavior_classification",
            "absolute_progress_m",
            "net_displacement_m",
            "initial_speed_mps",
            "final_speed_mps",
            "min_speed_mps",
            "stopped_detected",
            "stopped_from_step",
            "stopped_from_time_s",
            "yaw_change_deg",
            "signed_yaw_change_deg",
            "turn_completed",
            "maneuver_completion_status",
            "yaw_change_ratio_vs_recorded",
            "turn_completion_ratio_vs_recorded",
            "recorded_absolute_progress_m",
            "recorded_yaw_change_deg",
            "progress_ratio_vs_recorded",
            "expert_absolute_progress_m",
            "expert_yaw_change_deg",
            "expert_terminal_stopped",
            "progress_ratio_vs_expert",
            "expert_progress_ratio_vs_recorded",
        ):
            output[key] = primary.get(key)
    return output


def _summarize_via_subprocess(
    result_dir: str | Path,
    input_data: dict[str, Any],
    *,
    pdm_subscores: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    from mcp_servers.worldengine.adapter.config import ALGENGINE_PYTHON, PROJECT_ROOT, SIMENGINE_PYTHON, runtime_env

    candidates = [
        SIMENGINE_PYTHON,
        ALGENGINE_PYTHON,
        PROJECT_ROOT / "environments/scene_editing/bin/python",
    ]
    helper = next((path for path in candidates if Path(path).is_file()), None)
    if helper is None:
        raise RuntimeError("No NumPy-capable Python environment available for summarize_policy_behavior")
    env = runtime_env(os.environ)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{PROJECT_ROOT}{os.pathsep}{existing_pp}" if existing_pp else str(PROJECT_ROOT)
    payload = json.dumps(
        {
            "result_dir": str(result_dir),
            "input_data": input_data,
            "pdm_subscores": pdm_subscores,
        },
        ensure_ascii=False,
    )
    completed = subprocess.run(
        [str(helper), str(Path(__file__).resolve())],
        input=payload,
        text=True,
        capture_output=True,
        cwd=PROJECT_ROOT,
        env=env,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "policy_behavior helper failed")
    text = completed.stdout.strip()
    return json.loads(text) if text else None


def summarize_policy_behavior(
    result_dir: str | Path,
    input_data: dict[str, Any],
    *,
    pdm_subscores: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Summarize closed-loop policy motion, stop behavior, and progress vs recorded/expert."""
    try:
        import numpy as _np  # noqa: F401
    except ModuleNotFoundError:
        return _summarize_via_subprocess(result_dir, input_data, pdm_subscores=pdm_subscores)
    return _summarize_policy_behavior_in_process(result_dir, input_data, pdm_subscores=pdm_subscores)


def _main() -> int:
    request = json.loads(sys.stdin.read() or "{}")
    summary = _summarize_policy_behavior_in_process(
        request["result_dir"],
        request.get("input_data") or {},
        pdm_subscores=request.get("pdm_subscores"),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
