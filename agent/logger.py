"""
Run logging for the Agent.

Each Agent run gets its own directory under the configured log root:

<logs_root>/run_<timestamp>/
    run.json
    agent.jsonl
    final_state.json
    tools/
        step_<n>_<tool>/
            input.json
            output.json
            execution.log

Agent-level decisions are kept separate from tool-level inputs, outputs and
execution details.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOGS_ROOT = PROJECT_ROOT / "logs"


class RunLogger:
    """Write structured logs for one Agent run."""

    def __init__(
        self,
        task: str,
        max_steps: int,
        llm_model: str,
        llm_base_url: str,
        tool_modes: Dict[str, str],
        logs_root: Optional[Path] = None,
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.logs_root = Path(logs_root) if logs_root is not None else DEFAULT_LOGS_ROOT
        self.run_id = f"run_{timestamp}"
        self.run_dir = self.logs_root / self.run_id
        self.tools_dir = self.run_dir / "tools"
        self.agent_log_path = self.run_dir / "agent.jsonl"
        self.final_state_path = self.run_dir / "final_state.json"
        self.run_metadata_path = self.run_dir / "run.json"
        self._tool_start_times: Dict[int, float] = {}

        self.tools_dir.mkdir(parents=True, exist_ok=False)

        self.run_metadata = {
            "run_id": self.run_id,
            "task": task,
            "start_time": self._now(),
            "end_time": None,
            "status": "running",
            "max_steps": max_steps,
            "llm": {
                "model": llm_model,
                "base_url": llm_base_url,
            },
            "tool_modes": dict(tool_modes),
            "final_step": None,
        }
        self._write_json(self.run_metadata_path, self.run_metadata)

    @staticmethod
    def _now() -> str:
        """Return the current local timestamp."""

        return datetime.now().astimezone().isoformat(timespec="seconds")

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        """Write UTF-8 JSON."""

        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    def _tool_dir(self, step: int, tool_name: str) -> Path:
        """Return the directory for one concrete tool invocation."""

        return self.tools_dir / f"step_{step:02d}_{tool_name}"

    def log_decision(self, step: int, decision: Dict[str, Any]) -> None:
        """Append one Agent decision to agent.jsonl."""

        event = {
            "timestamp": self._now(),
            "step": step,
            "action": decision.get("action"),
            "tool_name": decision.get("tool_name"),
            "reason": decision.get("reason"),
        }

        with self.agent_log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")

    def start_tool_call(
        self,
        step: int,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> None:
        """Create the tool-call directory and record its input."""

        tool_dir = self._tool_dir(step, tool_name)
        tool_dir.mkdir(parents=True, exist_ok=False)

        self._write_json(tool_dir / "input.json", arguments)

        self._tool_start_times[step] = time.perf_counter()
        with (tool_dir / "execution.log").open("a", encoding="utf-8") as file:
            file.write(f"[{self._now()}] tool started\n")

    def finish_tool_call(
        self,
        step: int,
        tool_name: str,
        result: Dict[str, Any],
    ) -> None:
        """Record tool output and mark the invocation as finished."""

        tool_dir = self._tool_dir(step, tool_name)
        self._write_json(tool_dir / "output.json", result)

        started = self._tool_start_times.pop(step, None)
        duration = None if started is None else time.perf_counter() - started
        status = result.get("status", "unknown")

        with (tool_dir / "execution.log").open("a", encoding="utf-8") as file:
            if duration is None:
                file.write(f"[{self._now()}] tool finished status={status}\n")
            else:
                file.write(
                    f"[{self._now()}] tool finished status={status} "
                    f"duration_seconds={duration:.3f}\n"
                )

    def log_tool_error(
        self,
        step: int,
        tool_name: str,
        error: Exception,
    ) -> None:
        """Record an exception raised while executing a tool."""

        tool_dir = self._tool_dir(step, tool_name)
        with (tool_dir / "execution.log").open("a", encoding="utf-8") as file:
            file.write(
                f"[{self._now()}] tool failed "
                f"error={type(error).__name__}: {error}\n"
            )

    def finalize(self, state: Dict[str, Any]) -> None:
        """Save final Agent state and update run-level metadata."""

        self._write_json(self.final_state_path, state)

        self.run_metadata["end_time"] = self._now()
        self.run_metadata["status"] = state.get("status")
        self.run_metadata["final_step"] = state.get("current_step")
        self._write_json(self.run_metadata_path, self.run_metadata)
