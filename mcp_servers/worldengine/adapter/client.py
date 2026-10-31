"""Agent adapter for the WorldEngine closed-loop runner."""
from __future__ import annotations

from mcp_servers.runtime import runtime_task, current_run
import json
import subprocess
from mcp_servers.process import run_logged
from pathlib import Path


from .config import PROJECT_ROOT, AGENT_PYTHON, AGENT_FRAME_COUNT


@runtime_task("worldengine", "policy")
def run_policy(input_data: dict) -> dict:
    frame_count = input_data.get("max_frame_per_scene", AGENT_FRAME_COUNT)
    if type(frame_count) is not int or frame_count < 5:
        raise ValueError("Agent policy_runner requires max_frame_per_scene>=5")
    required = ("config_path", "checkpoint_path", "model_name", "data_type")
    missing = [key for key in required if not input_data.get(key)]
    if missing:
        raise ValueError("policy_runner requires: " + ", ".join(missing))
    if not AGENT_PYTHON.is_file():
        raise RuntimeError("Agent environment is missing: environments/agent/.venv/bin/python")

    proc = run_logged(
        [str(AGENT_PYTHON), "-m", "mcp_servers.worldengine.adapter.runner"],
        input=json.dumps({**input_data, "max_frame_per_scene": frame_count}, ensure_ascii=False),
        cwd=PROJECT_ROOT,
        env=current_run("worldengine").child_env(),
        log_path=current_run("worldengine").logs / "policy.log",
        json_stdout=True,
        timeout=int(input_data.get("timeout", 86400)),
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(proc.stderr.strip() or "policy_runner returned invalid JSON") from exc
    if proc.returncode != 0 or payload.get("status") != "success":
        error = payload.get("error") or {}
        raise RuntimeError(error.get("message") or proc.stderr.strip() or "WorldEngine policy run failed")
    result = dict(payload["result"])
    reports = result.get("reports") or []
    report_entries = [entry for report in reports for entry in (report if isinstance(report, list) else [report])]
    if not report_entries or not all(isinstance(entry, dict) and entry.get("succeeded") is True for entry in report_entries):
        raise RuntimeError("WorldEngine runner report contains a failed scenario")
    result["policy_result"] = "completed"
    rollout = Path(result["result_dir"]) / "WE_output/openscene_format"
    if rollout.is_dir():
        result["rollout_dir"] = str(rollout)
    if input_data.get("with_dense_reward_manager", False):
        # The native planner owns this CSV; export its actual selections, never synthesize them.
        import shutil
        plan_csv = Path(result["result_dir"]) / "plan_traj/plan_idx.csv"
        reward_dir = rollout / "pdms_pkl"
        if not list(reward_dir.glob("*_scores.pkl")):
            raise RuntimeError("RL rollout completed without candidate reward files")
        if not plan_csv.is_file():
            raise RuntimeError("RL rollout is missing the actual plan_idx.csv")
        target = reward_dir / "plan_idx.csv"
        if target.resolve() != plan_csv.resolve():
            shutil.copy2(plan_csv, target)
        result["reward_dir"] = str(reward_dir)
        result["candidate_rewards_present"] = True
    return result
