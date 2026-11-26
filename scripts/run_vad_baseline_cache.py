#!/usr/bin/env python3
"""Batch-generate empty-scene native VADv2 closed-loop baseline cache & unified BEVs.

Runs native WorldEngine / Tool 07 (num_history=4, max_frame_per_scene=16,
with_dense_reward_manager=False, visualize_video=False, visualize_BEV=False)
for all valid scenes under data/navtrain_ep_per1/samples/ and writes:
  <cache_root>/<scene_token>/baseline_frames.json
  <cache_root>/<scene_token>/metrics.json
  <cache_root>/<scene_token>/baseline_bev.png
plus <cache_root>/manifest.jsonl and <cache_root>/errors.jsonl.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SAMPLES_DIR = REPO_ROOT / "data/navtrain_ep_per1/samples"
DEFAULT_CACHE_BASE = REPO_ROOT / "data/navtrain_ep_per1/vad_baseline_cache"
SIMENGINE_PYTHON = REPO_ROOT / "environments/simengine/bin/python"
WORLDENGINE_RUNTIME_DIR = REPO_ROOT / "mcp_servers/worldengine/runtime"


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.write(line)
            fh.flush()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _is_scene_complete(scene_out_dir: Path) -> bool:
    for fname in ("baseline_frames.json", "metrics.json", "baseline_bev.png"):
        p = scene_out_dir / fname
        if not p.is_file() or p.stat().st_size == 0:
            return False
    return True


def _discover_scenes(samples_dir: Path) -> list[str]:
    tokens: list[str] = []
    for entry in sorted(samples_dir.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "metadata/frames.json").is_file() and (entry / "scenario/all_scenarios.pkl").is_file():
            tokens.append(entry.name)
    return tokens


def _detect_gpus() -> list[str]:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        ids = [x.strip() for x in cvd.split(",") if x.strip()]
        if ids:
            return ids
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        ids = [line.strip() for line in out.splitlines() if line.strip()]
        if ids:
            return ids
    except Exception:
        pass
    return ["0"]


def _extract_scene_pdm_subscores(pdm_subscores: dict[str, Any] | None, token: str) -> dict[str, Any] | None:
    if not isinstance(pdm_subscores, dict):
        return None
    scenes = pdm_subscores.get("scenes") or {}
    if isinstance(scenes, dict):
        for k, v in scenes.items():
            if k == token or k.endswith(token):
                return dict(v)
        if len(scenes) == 1:
            return dict(next(iter(scenes.values())))
    overall = pdm_subscores.get("overall_average")
    if isinstance(overall, dict):
        return dict(overall)
    return None


def _build_baseline_frames_and_bev(
    *,
    sample_dir: Path,
    rollout_dir: Path,
    out_frames_path: Path,
    out_bev_path: Path,
    num_history: int,
    max_frames: int,
    margin_m: float,
) -> dict[str, Any]:
    """Run in simengine python: merge closed-loop ego2global into baseline_frames.json and render BEV."""
    helper_code = r"""
import json, math, os, pickle, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

cfg = json.loads(sys.stdin.read())
sample_dir = Path(cfg["sample_dir"])
rollout_dir = Path(cfg["rollout_dir"])
out_frames_path = Path(cfg["out_frames_path"])
out_bev_path = Path(cfg["out_bev_path"])
num_history = int(cfg["num_history"])
max_frames = int(cfg["max_frames"])
margin_m = float(cfg["margin_m"])
repo_root = Path(cfg["repo_root"])

class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            try:
                return super().find_class(module, name)
            except ModuleNotFoundError:
                return super().find_class("numpy.core" + module[len("numpy._core"):], name)
        return super().find_class(module, name)

def mat_to_quat_wxyz(R):
    # Shepperd's method for 3x3 proper rotation matrix -> [w, x, y, z]
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    q /= np.linalg.norm(q)
    return q.tolist()

def reproject_anns(anns, T_old, T_new):
    if not isinstance(anns, dict):
        return anns
    boxes = anns.get("gt_boxes")
    if not boxes:
        return anns
    arr = np.asarray(boxes, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 7:
        return anns
    T_rel = np.linalg.inv(T_new) @ T_old
    R_rel = T_rel[:3, :3]
    t_rel = T_rel[:3, 3]
    yaw_old = math.atan2(float(T_old[1, 0]), float(T_old[0, 0]))
    yaw_new = math.atan2(float(T_new[1, 0]), float(T_new[0, 0]))
    dyaw = yaw_old - yaw_new
    new_arr = arr.copy()
    new_arr[:, :3] = (arr[:, :3] @ R_rel.T) + t_rel
    new_yaw = arr[:, 6] + dyaw
    new_arr[:, 6] = np.arctan2(np.sin(new_yaw), np.cos(new_yaw))
    out = dict(anns)
    out["gt_boxes"] = new_arr.tolist()
    vel = anns.get("velocity_3d")
    if vel:
        varr = np.asarray(vel, dtype=float)
        if varr.ndim == 2 and varr.shape[1] == 3:
            out["velocity_3d"] = (varr @ R_rel.T).tolist()
    return out

meta_dir = rollout_dir / "meta_datas"
sources = [meta_dir / "combined.pkl"] if (meta_dir / "combined.pkl").is_file() else sorted(meta_dir.glob("*.pkl"))
if not sources:
    raise RuntimeError(f"No rollout meta_datas pkl found under {meta_dir}")

rows = []
for p in sources:
    with p.open("rb") as fh:
        chunk = _CompatUnpickler(fh).load()
    if isinstance(chunk, list):
        rows.extend(item for item in chunk if isinstance(item, dict))
if not rows:
    raise RuntimeError(f"Rollout meta_datas empty under {meta_dir}")

rows.sort(key=lambda r: int(r.get("frame_idx", 0)))
rollout_by_idx = {int(r.get("frame_idx", i)): r for i, r in enumerate(rows)}

orig_frames = json.loads((sample_dir / "metadata/frames.json").read_text(encoding="utf-8"))
if len(orig_frames) < max_frames:
    raise RuntimeError(f"Original frames has {len(orig_frames)} < {max_frames}")

baseline_frames = []
replaced_count = 0
for idx in range(max_frames):
    frame = json.loads(json.dumps(orig_frames[idx]))
    if idx >= num_history:
        rframe = rollout_by_idx.get(idx)
        if rframe is None:
            raise RuntimeError(f"Missing rollout frame_idx={idx}; available={sorted(rollout_by_idx.keys())}")
        T_new = np.asarray(rframe["ego2global"], dtype=float)
        if T_new.shape != (4, 4) or not np.isfinite(T_new).all():
            raise RuntimeError(f"Invalid rollout ego2global at frame_idx={idx}")
        # Orthonormalize rotation via SVD to guarantee strict proper rotation
        U, _, Vt = np.linalg.svd(T_new[:3, :3])
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        T_new[:3, :3] = R
        T_new[3, :] = [0.0, 0.0, 0.0, 1.0]

        T_old = np.asarray(frame["ego2global"], dtype=float)
        frame["ego2global"] = T_new.tolist()
        if "ego2global_translation" in frame:
            frame["ego2global_translation"] = T_new[:3, 3].tolist()
        if "ego2global_rotation" in frame:
            frame["ego2global_rotation"] = mat_to_quat_wxyz(R)
        if "lidar2ego" in frame and frame["lidar2ego"] is not None:
            l2e = np.asarray(frame["lidar2ego"], dtype=float)
            if l2e.shape == (4, 4):
                frame["lidar2global"] = (T_new @ l2e).tolist()
        if "anns" in frame:
            frame["anns"] = reproject_anns(frame["anns"], T_old, T_new)
        replaced_count += 1
    baseline_frames.append(frame)

out_frames_path.parent.mkdir(parents=True, exist_ok=True)
tmp_frames = out_frames_path.with_suffix(f".json.tmp.{os.getpid()}")
tmp_frames.write_text(json.dumps(baseline_frames, ensure_ascii=False), encoding="utf-8")
os.replace(tmp_frames, out_frames_path)

# Stage sample_root symlinks so structure_facts.py finds asset_config.yaml, road_height_map, map_features.json
stage_dir = Path(tempfile.mkdtemp(prefix=f"vbc_bev_{sample_dir.name}_"))
try:
    (stage_dir / "metadata").mkdir(parents=True, exist_ok=True)
    staged_frames = stage_dir / "metadata/frames.json"
    shutil.copy2(out_frames_path, staged_frames)
    for rel in ("asset_config.yaml", "source_asset", "asset", "maps", "metadata/map_features.json"):
        src = sample_dir / rel
        if src.exists() or src.is_symlink():
            dst = stage_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(src.resolve(), dst)

    out_bev_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_bev = out_bev_path.with_suffix(f".tmp.{os.getpid()}.png")
    tmp_traj = stage_dir / "traj.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools/tool01_failure_analyzer/structure_facts.py"),
            "--frames-json", str(staged_frames),
            "--out-bev", str(tmp_bev),
            "--out-traj", str(tmp_traj),
            "--max-frames", str(max_frames),
            "--margin-m", str(margin_m),
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not tmp_bev.is_file() or tmp_bev.stat().st_size == 0:
        raise RuntimeError(f"structure_facts failed (rc={proc.returncode}): stdout={proc.stdout[-500:]} stderr={proc.stderr[-500:]}")
    os.replace(tmp_bev, out_bev_path)
    sf_payload = json.loads(proc.stdout.strip().splitlines()[-1])
    print(json.dumps({
        "status": "ok",
        "replaced_frames": replaced_count,
        "rollout_frame_count": len(rows),
        "bev_bytes": out_bev_path.stat().st_size,
        "ego_kinematics": sf_payload.get("ego_kinematics"),
    }, ensure_ascii=False))
finally:
    shutil.rmtree(stage_dir, ignore_errors=True)
"""
    payload = {
        "sample_dir": str(sample_dir.resolve()),
        "rollout_dir": str(rollout_dir.resolve()),
        "out_frames_path": str(out_frames_path.resolve()),
        "out_bev_path": str(out_bev_path.resolve()),
        "num_history": num_history,
        "max_frames": max_frames,
        "margin_m": margin_m,
        "repo_root": str(REPO_ROOT),
    }
    proc = subprocess.run(
        [str(SIMENGINE_PYTHON), "-c", helper_code],
        input=json.dumps(payload, ensure_ascii=False),
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Post-process failed (rc={proc.returncode}): {proc.stderr[-1200:] or proc.stdout[-1200:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _cleanup_worldengine_run(result_dir: str | None) -> None:
    if not result_dir:
        return
    p = Path(result_dir).resolve()
    rt = WORLDENGINE_RUNTIME_DIR.resolve()
    # Walk up to find the direct child run_* folder under mcp_servers/worldengine/runtime
    target = None
    cur = p
    while cur != cur.parent:
        if cur.parent == rt and cur.name.startswith("run_"):
            target = cur
            break
        cur = cur.parent
    if target is not None and target.is_dir():
        shutil.rmtree(target, ignore_errors=True)
    elif p.is_dir() and rt in p.parents:
        shutil.rmtree(p, ignore_errors=True)


def _process_one_scene(
    *,
    token: str,
    samples_dir: Path,
    cache_root: Path,
    checkpoint_path: Path,
    num_history: int,
    max_frames: int,
    margin_m: float,
    keep_temp: bool,
) -> dict[str, Any]:
    try:
        from tools.tool07_policy_runner.real import run as run_tool07
    except ModuleNotFoundError:
        from tools.tool06_policy_runner.real import run as run_tool07

    sample_dir = samples_dir / token
    scene_out_dir = cache_root / token
    scene_out_dir.mkdir(parents=True, exist_ok=True)

    asset_link = sample_dir / "runtime_asset"
    if not asset_link.exists():
        raise FileNotFoundError(f"Missing runtime_asset for {token}: {asset_link}")
    asset_folder_path = str(asset_link.resolve().parent)
    scenario_folder_path = str((sample_dir / "scenario").resolve())

    inp = {
        "execution_mode": "native",
        "checkpoint_path": str(checkpoint_path),
        "scenario_data_folder_path": scenario_folder_path,
        "scenario_pkl_file_name": "all_scenarios.pkl",
        "asset_folder_path": asset_folder_path,
        "num_history": num_history,
        "max_frame_per_scene": max_frames,
        "with_dense_reward_manager": False,
        "visualize_video": False,
        "visualize_BEV": False,
        "added_name": f"vbc_{token[:8]}",
    }

    t0 = time.time()
    result_dir: str | None = None
    WORLDENGINE_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    tmp_run_dir = Path(tempfile.mkdtemp(prefix=f"run_vbc_{token}_", dir=str(WORLDENGINE_RUNTIME_DIR)))
    prev_service = os.environ.get("ADAGENT_ACTIVE_SERVICE")
    prev_run = os.environ.get("ADAGENT_ACTIVE_RUN")
    os.environ["ADAGENT_ACTIVE_SERVICE"] = "worldengine"
    os.environ["ADAGENT_ACTIVE_RUN"] = str(tmp_run_dir)
    try:
        from mcp_servers.worldengine.adapter import planner_prewarm
        from mcp_servers.worldengine.adapter.config import WorldEngineConfig, DEFAULT_POLICY_CONFIG
        warm = planner_prewarm.locate({
            "worldengine_root": str(WorldEngineConfig().project_root),
            "config_path": str(DEFAULT_POLICY_CONFIG),
            "checkpoint_path": str(checkpoint_path),
        })
        if warm and warm.get("socket_path"):
            os.environ["ADAGENT_VAD_PREWARM_SOCKET"] = warm["socket_path"]
        res = run_tool07(inp)
        elapsed_s = round(time.time() - t0, 2)
        if res.get("status") != "success":
            raise RuntimeError(f"tool07 returned status={res.get('status')}: {res.get('error')}")
        r = res["result"]
        result_dir = r.get("result_dir")
        rollout_dir = r.get("rollout_dir")
        if not rollout_dir or not Path(rollout_dir).is_dir():
            raise RuntimeError(f"Missing rollout_dir in tool07 result: {rollout_dir}")

        out_frames_path = scene_out_dir / "baseline_frames.json"
        out_bev_path = scene_out_dir / "baseline_bev.png"
        out_metrics_path = scene_out_dir / "metrics.json"

        post = _build_baseline_frames_and_bev(
            sample_dir=sample_dir,
            rollout_dir=Path(rollout_dir),
            out_frames_path=out_frames_path,
            out_bev_path=out_bev_path,
            num_history=num_history,
            max_frames=max_frames,
            margin_m=margin_m,
        )

        scene_pdm = _extract_scene_pdm_subscores(r.get("pdm_subscores"), token)
        metrics = {
            "scene_token": token,
            "checkpoint_path": str(checkpoint_path),
            "status": "success",
            "elapsed_s": elapsed_s,
            "num_history": num_history,
            "max_frame_per_scene": max_frames,
            "pdm_subscores": scene_pdm,
            "policy_behavior_summary": r.get("policy_behavior_summary"),
            "postprocess": post,
        }
        _atomic_write_json(out_metrics_path, metrics)

        summary = {
            "scene_token": token,
            "status": "success",
            "elapsed_s": elapsed_s,
            "score": None if not scene_pdm else scene_pdm.get("score"),
            "no_at_fault_collisions": None if not scene_pdm else scene_pdm.get("no_at_fault_collisions"),
            "drivable_area_compliance": None if not scene_pdm else scene_pdm.get("drivable_area_compliance"),
            "ego_progress": None if not scene_pdm else scene_pdm.get("ego_progress"),
            "time_to_collision_within_bound": None if not scene_pdm else scene_pdm.get("time_to_collision_within_bound"),
            "comfort": None if not scene_pdm else scene_pdm.get("comfort"),
            "first_violation_step": None if not scene_pdm else scene_pdm.get("first_violation_step"),
            "bev_bytes": post.get("bev_bytes"),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        _append_jsonl(cache_root / "manifest.jsonl", summary)
        return summary
    finally:
        if prev_service is None:
            os.environ.pop("ADAGENT_ACTIVE_SERVICE", None)
        else:
            os.environ["ADAGENT_ACTIVE_SERVICE"] = prev_service
        if prev_run is None:
            os.environ.pop("ADAGENT_ACTIVE_RUN", None)
        else:
            os.environ["ADAGENT_ACTIVE_RUN"] = prev_run
        if not keep_temp:
            _cleanup_worldengine_run(result_dir)
            shutil.rmtree(tmp_run_dir, ignore_errors=True)


def _run_worker(args: argparse.Namespace, tokens: list[str], cache_root: Path, checkpoint_path: Path) -> None:
    manifest_path = cache_root / "manifest.jsonl"
    errors_path = cache_root / "errors.jsonl"
    total = len(tokens)
    done = 0
    skipped = 0
    failed = 0
    t_start = time.time()
    gpu_tag = os.environ.get("CUDA_VISIBLE_DEVICES", "auto")

    try:
        from mcp_servers.worldengine.adapter import planner_prewarm
        from mcp_servers.worldengine.adapter.config import WorldEngineConfig, DEFAULT_POLICY_CONFIG
        policy_info = {
            "worldengine_root": str(WorldEngineConfig().project_root),
            "config_path": str(DEFAULT_POLICY_CONFIG),
            "checkpoint_path": str(checkpoint_path),
        }
        warm = planner_prewarm.start(policy_info, cache_root / f"prewarm_gpu{gpu_tag}.json")
        if warm and warm.get("socket_path"):
            os.environ["ADAGENT_VAD_PREWARM_SOCKET"] = warm["socket_path"]
    except Exception:
        pass

    print(
        f"[worker gpu={gpu_tag} shard={args.shard_idx}/{args.num_shards}] "
        f"assigned={total} cache_root={cache_root}",
        flush=True,
    )

    for idx, token in enumerate(tokens, start=1):
        scene_out_dir = cache_root / token
        if not args.force and _is_scene_complete(scene_out_dir):
            skipped += 1
            continue
        try:
            summary = _process_one_scene(
                token=token,
                samples_dir=Path(args.samples_dir),
                cache_root=cache_root,
                checkpoint_path=checkpoint_path,
                num_history=args.num_history,
                max_frames=args.max_frames,
                margin_m=args.margin_m,
                keep_temp=args.keep_temp,
            )
            done += 1
            avg = (time.time() - t_start) / max(1, done + failed)
            rem = total - idx
            eta_min = (rem * avg) / 60.0
            print(
                f"[gpu={gpu_tag}] [{idx}/{total}] OK {token} | "
                f"t={summary['elapsed_s']}s | pdm={summary['score']} "
                f"nc={summary['no_at_fault_collisions']} dac={summary['drivable_area_compliance']} | "
                f"done={done} skip={skipped} fail={failed} | ETA={eta_min:.1f}m",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            err_rec = {
                "scene_token": token,
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:1000],
                "traceback": traceback.format_exc()[-2000:],
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            _append_jsonl(errors_path, err_rec)
            print(
                f"[gpu={gpu_tag}] [{idx}/{total}] FAIL {token}: {type(exc).__name__}: {str(exc)[:300]}",
                flush=True,
            )

    print(
        f"[worker gpu={gpu_tag} shard={args.shard_idx}/{args.num_shards}] FINISHED: "
        f"done={done} skipped={skipped} failed={failed} elapsed={(time.time()-t_start)/60.0:.1f}m",
        flush=True,
    )


def main() -> None:
    from mcp_servers.worldengine.adapter.config import DEFAULT_POLICY_CHECKPOINT

    parser = argparse.ArgumentParser(description="Batch-generate VADv2 native closed-loop baseline cache.")
    parser.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES_DIR)
    parser.add_argument("--cache-base", type=Path, default=DEFAULT_CACHE_BASE)
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_POLICY_CHECKPOINT))
    parser.add_argument("--num-history", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--margin-m", type=float, default=35.0)
    parser.add_argument("--tokens", type=str, default="", help="Comma-separated scene tokens to run (for smoke test).")
    parser.add_argument("--limit", type=int, default=0, help="Limit total scenes (0 = all).")
    parser.add_argument("--force", action="store_true", help="Re-run even if scene outputs already exist.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep WorldEngine runtime directories.")
    parser.add_argument("--gpus", type=str, default="", help="Comma-separated GPU indices; empty = auto-detect.")
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--worker", action="store_true", help="Internal flag: run as single shard worker.")
    args = parser.parse_args()

    ckpt = args.checkpoint.expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    cache_root = args.cache_base / ckpt.stem
    cache_root.mkdir(parents=True, exist_ok=True)

    if args.tokens:
        tokens = [t.strip() for t in args.tokens.split(",") if t.strip()]
    else:
        tokens = _discover_scenes(args.samples_dir)
    if args.limit and args.limit > 0:
        tokens = tokens[: args.limit]

    if args.worker or args.num_shards > 1:
        shard_tokens = [t for i, t in enumerate(tokens) if i % args.num_shards == args.shard_idx]
        _run_worker(args, shard_tokens, cache_root, ckpt)
        return

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()] if args.gpus else _detect_gpus()
    if len(gpus) <= 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus[0] if gpus else "0"
        _run_worker(args, tokens, cache_root, ckpt)
        return

    # Multi-GPU launcher: spawn one worker subprocess per GPU
    procs = []
    for shard_idx, gpu_id in enumerate(gpus):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--samples-dir", str(args.samples_dir),
            "--cache-base", str(args.cache_base),
            "--checkpoint", str(ckpt),
            "--num-history", str(args.num_history),
            "--max-frames", str(args.max_frames),
            "--margin-m", str(args.margin_m),
            "--shard-idx", str(shard_idx),
            "--num-shards", str(len(gpus)),
        ]
        if args.tokens:
            cmd += ["--tokens", args.tokens]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.force:
            cmd.append("--force")
        if args.keep_temp:
            cmd.append("--keep-temp")
        procs.append(subprocess.Popen(cmd, env=env, cwd=str(REPO_ROOT)))

    rc = 0
    for p in procs:
        r = p.wait()
        if r != 0:
            rc = r
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
