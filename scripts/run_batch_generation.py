#!/usr/bin/env python3
"""Two-round, 3-bucket batch synthetic scenario generator.

Strategy:
  - Configurable failure threshold: --pdm-threshold (default 0.5, via ADAGENT_PDM_FAILURE_THRESHOLD)
  - Round 1: run every source failure token once (Tool 05 hard-capped at --tool05-timeout, default 900s = 15 min)
      * Class 1 (class_1_success): Tool 06 PDM < threshold & first_violation_step >= 6 -> kept & base scene recorded in exclusion pool
      * Class 2 (class_2_retry_speed): Tool 06 ran, but PDM >= threshold (or violation too early < 6) -> retry queue (priority 1)
      * Class 3 (class_3_no_result): No Tool 06 score (Tool 05 timed out > 15m / unable_to_complete / error) -> retry queue (priority 2)
  - Round 2: process Class 2 FIRST, then Class 3, sharing the same used_base_scenes.json exclusion pool.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.scene_pool import (
    EXCLUDE_ENV_VAR,
    PDM_THRESHOLD_ENV_VAR,
    classify_run_result,
    get_pdm_failure_threshold,
    normalize_scene_token,
    record_used_scene_token,
)


def load_source_tokens(tokens_file: str | None, tokens_cli: list[str] | None) -> list[str]:
    tokens: list[str] = []
    if tokens_cli:
        tokens.extend(tokens_cli)
    if tokens_file:
        p = Path(tokens_file)
        raw = p.read_text(encoding="utf-8").strip()
        if p.suffix.lower() == ".json":
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                for item in parsed:
                    tokens.append(item["token"] if isinstance(item, dict) else str(item))
            elif isinstance(parsed, dict):
                for key in ("tokens", "scene_tokens", "failures"):
                    if isinstance(parsed.get(key), list):
                        tokens.extend(str(x) for x in parsed[key])
                        break
        elif p.suffix.lower() == ".csv":
            with p.open(encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    tok = row.get("token") or row.get("scene_token")
                    if tok:
                        tokens.append(tok)
        else:
            for line in raw.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    tokens.append(line.split()[0])
    seen: set[str] = set()
    ordered: list[str] = []
    for tok in tokens:
        norm = normalize_scene_token(tok)
        if norm and norm not in seen:
            seen.add(norm)
            ordered.append(norm)
    return ordered


def _save_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_name(f"{state_path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, state_path)


def _load_state(state_path: Path, *, simulation: str, pdm_threshold: float, tool05_timeout_s: int, pool_file: Path) -> dict[str, Any]:
    if state_path.is_file():
        data = json.loads(state_path.read_text(encoding="utf-8"))
        data["pdm_threshold"] = pdm_threshold
        data["tool05_timeout_s"] = tool05_timeout_s
        data["exclusion_pool_file"] = str(pool_file)
        return data
    return {
        "simulation": simulation,
        "pdm_threshold": pdm_threshold,
        "tool05_timeout_s": tool05_timeout_s,
        "exclusion_pool_file": str(pool_file),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "items": {},
    }


def round2_queue(state: dict[str, Any], tokens: list[str]) -> list[tuple[str, str]]:
    """Return Round 2 execution order: ALL Class 2 items first, then ALL Class 3 items."""
    class2: list[tuple[str, str]] = []
    class3: list[tuple[str, str]] = []
    items = state.get("items") or {}
    for tok in tokens:
        rec = items.get(tok) or {}
        if rec.get("round2"):
            continue
        r1 = rec.get("round1") or {}
        bucket = r1.get("bucket")
        if bucket == "class_2_retry_speed":
            class2.append((tok, bucket))
        elif bucket == "class_3_no_result":
            class3.append((tok, bucket))
    return class2 + class3


def _latest_new_run_dir(logs_root: Path, before_dirs: set[str]) -> Path | None:
    if not logs_root.is_dir():
        return None
    candidates = [p for p in logs_root.iterdir() if p.is_dir() and p.name not in before_dirs]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _build_task_prompt(simulation: str, token: str, round_idx: int, prev_bucket: str | None, pdm_threshold: float) -> str:
    base = (
        f"请分析仿真批次 {simulation} 中的失败场景 {token}，"
        f"从本地数据集检索合适底场景并构造同类失败合成场景，"
        f"通过 policy_runner (tool06) 进行闭环评估（目标 PDM < {pdm_threshold} 且 first_violation_step >= 6）。"
    )
    if round_idx == 2 and prev_bucket == "class_2_retry_speed":
        base += " 这是第二轮二类补救任务：上一轮已测出高分未失败或过早违规，请优先调整自车历史初速或更换更合适的底场景与攻击策略。"
    elif round_idx == 2 and prev_bucket == "class_3_no_result":
        base += " 这是第二轮三类补救任务：上一轮未能产出有效闭环仿真结果（超时或无解），请换用其他候选方案或新底场景重新构造。"
    return base


def _run_single_scene(
    *,
    python_bin: str,
    simulation: str,
    token: str,
    round_idx: int,
    prev_bucket: str | None,
    pdm_threshold: float,
    tool05_timeout_s: int,
    run_timeout_s: int,
    pool_file: Path,
    logs_root: Path,
) -> dict[str, Any]:
    before = {p.name for p in logs_root.iterdir() if p.is_dir()} if logs_root.is_dir() else set()
    env = os.environ.copy()
    env[PDM_THRESHOLD_ENV_VAR] = str(pdm_threshold)
    env["ADAGENT_TOOL05_TIMEOUT_S"] = str(tool05_timeout_s)
    env[EXCLUDE_ENV_VAR] = str(pool_file)

    prompt = _build_task_prompt(simulation, token, round_idx, prev_bucket, pdm_threshold)
    cmd = [python_bin, "-m", "agent.main", prompt]
    t0 = time.monotonic()
    timed_out = False
    returncode = None
    try:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, timeout=run_timeout_s)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
    elapsed_s = round(time.monotonic() - t0, 1)

    run_dir = _latest_new_run_dir(logs_root, before)
    if run_dir is not None:
        verdict = classify_run_result(run_dir, pdm_threshold=pdm_threshold)
    else:
        verdict = {
            "bucket": "class_3_no_result",
            "pdm_score": None,
            "first_violation_step": None,
            "base_scene_token": None,
            "tool05_status": "timeout" if timed_out else "error",
            "threshold": pdm_threshold,
            "reason": "run_timeout_no_dir" if timed_out else "no_run_dir",
        }
    if timed_out and verdict["bucket"] != "class_1_success":
        verdict["bucket"] = "class_3_no_result"
        verdict["reason"] = f"outer_run_timeout>{run_timeout_s}s ({verdict['reason']})"

    # Ensure actually-used base scene is recorded if Class 1 succeeded
    if verdict["bucket"] == "class_1_success" and verdict.get("base_scene_token"):
        record_used_scene_token(
            pool_file,
            verdict["base_scene_token"],
            source_token=token,
            metadata={"round": round_idx, "pdm_score": verdict["pdm_score"], "run_dir": str(run_dir) if run_dir else None},
        )

    return {
        **verdict,
        "round": round_idx,
        "elapsed_s": elapsed_s,
        "returncode": returncode,
        "timed_out": timed_out,
        "run_dir": str(run_dir) if run_dir else None,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }


def summarize_state(state: dict[str, Any]) -> dict[str, int]:
    counts = {"class_1_success": 0, "class_2_retry_speed": 0, "class_3_no_result": 0, "pending": 0}
    for rec in (state.get("items") or {}).values():
        latest = rec.get("round2") or rec.get("round1")
        if not latest:
            counts["pending"] += 1
        else:
            b = latest.get("bucket", "class_3_no_result")
            counts[b] = counts.get(b, 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Two-round 3-bucket batch synthetic scenario generator.")
    parser.add_argument("--simulation", required=True, help="WorldEngine simulation archive path for Tool 01")
    parser.add_argument("--tokens-file", help="Path to txt/csv/json file listing source failure tokens")
    parser.add_argument("--tokens", nargs="*", help="Explicit source failure tokens")
    parser.add_argument("--experiment", required=True, help="Experiment ID under outputs/experiments/<exp_id>")
    parser.add_argument("--pdm-threshold", type=float, default=None, help="PDM failure threshold (default: env ADAGENT_PDM_FAILURE_THRESHOLD or 0.5)")
    parser.add_argument("--tool05-timeout", type=int, default=900, help="Hard timeout in seconds for Tool 05 (default: 900 = 15 min)")
    parser.add_argument("--run-timeout", type=int, default=1500, help="Outer per-scene timeout in seconds (default: 1500 = 25 min)")
    parser.add_argument("--round", choices=("1", "2", "all"), default="all", help="Which round(s) to execute")
    parser.add_argument("--python", default=str(PROJECT_ROOT / "environments/agent/.venv/bin/python"), help="Agent Python interpreter")
    parser.add_argument("--dry-run", action="store_true", help="Print planned Round 1 / Round 2 queues without executing")
    args = parser.parse_args()

    pdm_threshold = get_pdm_failure_threshold(args.pdm_threshold)
    tokens = load_source_tokens(args.tokens_file, args.tokens)
    if not tokens:
        print("Error: no source tokens provided via --tokens-file or --tokens", file=sys.stderr)
        return 2

    exp_dir = PROJECT_ROOT / "outputs" / "experiments" / args.experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    pool_file = exp_dir / "used_base_scenes.json"
    state_path = exp_dir / "batch_state.json"
    logs_root = PROJECT_ROOT / "logs"

    state = _load_state(
        state_path,
        simulation=args.simulation,
        pdm_threshold=pdm_threshold,
        tool05_timeout_s=args.tool05_timeout,
        pool_file=pool_file,
    )
    for tok in tokens:
        state["items"].setdefault(tok, {})
    _save_state(state_path, state)

    if args.round in ("1", "all"):
        r1_pending = [tok for tok in tokens if not state["items"][tok].get("round1")]
        print(f"[Round 1] pending={len(r1_pending)}/{len(tokens)}, pdm_threshold={pdm_threshold}, tool05_timeout={args.tool05_timeout}s")
        for idx, tok in enumerate(r1_pending, 1):
            if args.dry_run:
                print(f"  [DRY-RUN R1 {idx}/{len(r1_pending)}] token={tok}")
                continue
            print(f"[R1 {idx}/{len(r1_pending)}] running {tok} ...", flush=True)
            res = _run_single_scene(
                python_bin=args.python,
                simulation=args.simulation,
                token=tok,
                round_idx=1,
                prev_bucket=None,
                pdm_threshold=pdm_threshold,
                tool05_timeout_s=args.tool05_timeout,
                run_timeout_s=args.run_timeout,
                pool_file=pool_file,
                logs_root=logs_root,
            )
            state["items"][tok]["round1"] = res
            _save_state(state_path, state)
            print(f"  -> {res['bucket']} ({res['reason']}, {res['elapsed_s']}s)", flush=True)

    if args.round in ("2", "all"):
        r2_list = round2_queue(state, tokens)
        n_c2 = sum(1 for _, b in r2_list if b == "class_2_retry_speed")
        n_c3 = sum(1 for _, b in r2_list if b == "class_3_no_result")
        print(f"[Round 2] total={len(r2_list)} (Class 2 first: {n_c2}, then Class 3: {n_c3}), pdm_threshold={pdm_threshold}")
        for idx, (tok, prev_bucket) in enumerate(r2_list, 1):
            if args.dry_run:
                print(f"  [DRY-RUN R2 {idx}/{len(r2_list)}] token={tok} (from {prev_bucket})")
                continue
            print(f"[R2 {idx}/{len(r2_list)}] running {tok} (from {prev_bucket}) ...", flush=True)
            res = _run_single_scene(
                python_bin=args.python,
                simulation=args.simulation,
                token=tok,
                round_idx=2,
                prev_bucket=prev_bucket,
                pdm_threshold=pdm_threshold,
                tool05_timeout_s=args.tool05_timeout,
                run_timeout_s=args.run_timeout,
                pool_file=pool_file,
                logs_root=logs_root,
            )
            state["items"][tok]["round2"] = res
            _save_state(state_path, state)
            print(f"  -> {res['bucket']} ({res['reason']}, {res['elapsed_s']}s)", flush=True)

    print("Summary:", json.dumps(summarize_state(state), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
