"""Agent adapter for the WorldEngine closed-loop runner."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_PYTHON = PROJECT_ROOT / "environments" / "agent" / ".venv" / "bin" / "python"
AGENT_FRAME_COUNT = 40


def run(input_data: dict) -> dict:
    frame_count = input_data.get("max_frame_per_scene", AGENT_FRAME_COUNT)
    if frame_count not in (40, 60):
        raise ValueError("Agent policy_runner requires max_frame_per_scene=40 or 60")
    required = ("config_path", "checkpoint_path", "model_name", "data_type")
    missing = [key for key in required if not input_data.get(key)]
    if missing:
        raise ValueError("policy_runner requires: " + ", ".join(missing))
    if not AGENT_PYTHON.is_file():
        raise RuntimeError("Agent environment is missing: environments/agent/.venv/bin/python")

    proc = subprocess.run(
        [str(AGENT_PYTHON), "-m", "mcp_servers.worldengine.runner"],
        input=json.dumps({**input_data, "max_frame_per_scene": frame_count}, ensure_ascii=False),
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
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
    if not report_entries or not all(entry.get("succeeded") is True for entry in report_entries if isinstance(entry, dict)):
        raise RuntimeError("WorldEngine runner report contains a failed scenario")
    result["policy_result"] = "completed"
    return {"status": "success", "tool": "policy_runner", "result": result}
