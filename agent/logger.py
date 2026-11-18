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
import fcntl
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from mcp_servers.runtime import write_json
from common.run_report import snapshot, summarize
from common.llm_audit import sum_usage
from agent.jobs import process_identity
from agent.continuation import RUN_ID


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
        lineage: Optional[Dict[str, Any]] = None,
        llm_reasoning_effort: Optional[str] = None,
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
        self._lease = (self.run_dir/'run.lock').open('w')
        fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        conversation = (lineage or {}).get('conversation_id') or self.run_id
        if not isinstance(conversation,str) or not RUN_ID.fullmatch(conversation):
            self._lease.close()
            raise ValueError('Invalid conversation ID')
        lease_root = self.logs_root/'.conversation_locks'
        lease_root.mkdir(parents=True,exist_ok=True)
        self._conversation_lease = (lease_root/(conversation+'.lock')).open('a')
        try:
            fcntl.flock(self._conversation_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._conversation_lease.close()
            self._lease.close()
            raise ValueError('该对话仍在执行，不能同时续跑同一对话') from exc

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
                "reasoning_effort": llm_reasoning_effort,
            },
            "tool_modes": dict(tool_modes),
            "final_step": None,
            "owner_pid": os.getpid(),
            "owner_process_start": process_identity(os.getpid()),
            "conversation_id": self.run_id,
            "execution_scope": self.run_id,
            **(lineage or {}),
        }
        self._write_json(self.run_metadata_path, self.run_metadata)
        snapshot(self.run_dir, PROJECT_ROOT)
        summarize(self.run_dir)

    @staticmethod
    def _now() -> str:
        """Return the current local timestamp."""

        return datetime.now().astimezone().isoformat(timespec="seconds")

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        """Write UTF-8 JSON."""

        write_json(path, data)

    def _tool_dir(self, step: int, tool_name: str) -> Path:
        """Return the directory for one concrete tool invocation."""

        return self.tools_dir / f"step_{step:02d}_{tool_name}"

    def log_decision(self, step: int, decision: Dict[str, Any], request_meta=None) -> None:
        """Append one Agent decision to agent.jsonl."""
        # Keep the frontend's per-decision numbers inclusive of every correction.
        requests = []
        for path in (self.run_dir/'api_requests').glob('*/result.json'):
            record = json.loads(path.read_text())
            if record.get('stage') == 'main_agent' and record.get('context', {}).get('step') == step:
                requests.append(record)
        if requests:
            request_meta = dict(request_meta or {})
            request_meta['metrics'] = dict(request_meta.get('metrics') or {},
                elapsed_seconds=round(sum(r.get('elapsed_seconds', 0) for r in requests), 6),
                request_count=len(requests), correction_requests=max(0, len(requests)-1))
            usage = sum_usage(record.get('usage') for record in requests)
            request_meta['usage'] = usage or None
        event = {
            "timestamp": self._now(),
            "step": step,
            "action": decision.get("action"),
            "tool_name": decision.get("tool_name"),
            "reason": decision.get("reason"),
            "llm_request": request_meta or {},
        }

        with self.agent_log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")
        summarize(self.run_dir)

    # ---- live thinking stream (polled by the console while a run is in flight) ----

    def open_thinking(self, step: int) -> None:
        """Start a fresh thinking stream for one decision."""

        self.thinking_path = self.run_dir / "thinking.jsonl"
        self._thinking_step = step
        self._thinking_started = time.monotonic()
        self._thinking_file = self.thinking_path.open("w", encoding="utf-8")

    def thinking_delta(self, kind: str, text: str) -> None:
        """Append one streamed delta so the console can render it live."""

        handle = getattr(self, "_thinking_file", None)
        if handle is None or not text:
            return
        event = {"step": self._thinking_step, "kind": kind, "text": text,
                 "elapsed": round(time.monotonic() - self._thinking_started, 2)}
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        handle.flush()

    def close_thinking(self) -> None:
        handle = getattr(self, "_thinking_file", None)
        if handle is not None:
            handle.close()
            self._thinking_file = None

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
        self._write_json(tool_dir / "call.json", {
            "agent_run_id": self.run_id, "tool_call_id": f"{self.run_id}:step_{step:02d}:{tool_name}",
            "tool": tool_name, "step": step})

        self._tool_start_times[step] = time.perf_counter()
        self._write_json(tool_dir / 'timing.json', dict(status='running', started_at=self._now()))
        with (tool_dir / "execution.log").open("a", encoding="utf-8") as file:
            file.write(f"[{self._now()}] tool started\n")

    def finish_tool_call(
        self,
        step: int,
        tool_name: str,
        result: Dict[str, Any],
    ) -> str:
        """Record full output and return its saved result reference path."""

        tool_dir = self._tool_dir(step, tool_name)
        self._write_json(tool_dir / "output.json", result)

        started = self._tool_start_times.pop(step, None)
        duration = None if started is None else time.perf_counter() - started
        status = result.get("status", "unknown")
        self._write_json(tool_dir / 'timing.json', dict(status=status,
                         elapsed_seconds=duration, finished_at=self._now(),
                         job_id=(result.get('job') or {}).get('job_id'),
                         job_elapsed_seconds=(result.get('job') or {}).get('elapsed_seconds')))

        with (tool_dir / "execution.log").open("a", encoding="utf-8") as file:
            if duration is None:
                file.write(f"[{self._now()}] tool finished status={status}\n")
            else:
                file.write(
                    f"[{self._now()}] tool finished status={status} "
                    f"duration_seconds={duration:.3f}\n"
                )

        summarize(self.run_dir)
        return str((tool_dir / "output.json").resolve())

    def log_tool_error(
        self,
        step: int,
        tool_name: str,
        error: Exception,
    ) -> None:
        """Record an exception raised while executing a tool."""

        tool_dir = self._tool_dir(step, tool_name)
        started = self._tool_start_times.pop(step, None)
        self._write_json(tool_dir/'timing.json', dict(status='error',
                         elapsed_seconds=None if started is None else time.perf_counter()-started,
                         error_type=type(error).__name__, finished_at=self._now()))
        with (tool_dir / "execution.log").open("a", encoding="utf-8") as file:
            file.write(
                f"[{self._now()}] tool failed "
                f"error={type(error).__name__}: {error}\n"
            )

    def save_state(self, state):
        """A checkpoint for inspection/recovery after each completed tool."""
        self._write_json(self.run_dir / "state.json", state)

    def finalize(self, state: Dict[str, Any]) -> None:
        """Save final Agent state and update run-level metadata."""

        self._write_json(self.final_state_path, state)

        self.run_metadata["end_time"] = self._now()
        self.run_metadata["status"] = state.get("status")
        self.run_metadata["final_step"] = state.get("current_step")
        self._write_json(self.run_metadata_path, self.run_metadata)
        # Include the elapsed time of a tool interrupted while waiting for a job.
        for step, started in list(self._tool_start_times.items()):
            for directory in self.tools_dir.glob(f'step_{step:02d}_*'):
                self._write_json(directory/'timing.json', dict(status='interrupted',
                                 elapsed_seconds=time.perf_counter()-started, finished_at=self._now()))
            self._tool_start_times.pop(step, None)
        summarize(self.run_dir)
        self._lease.close()
        self._conversation_lease.close()
