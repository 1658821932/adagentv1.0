"""
Tool execution layer.

The Agent selects a Tool; this executor selects that Tool's configured
implementation mode (mock or real), enforces an optional per-call timeout,
and validates the standard result envelope.
"""

import queue
import threading
from typing import Any, Dict, Optional

from agent.schemas import validate_tool_result
from tools.registry import DEFAULT_TOOL_MODES, call_tool

# Default per-tool wall-clock timeout in seconds. Long enough for DGGT
# reconstruction and WorldEngine closed-loop runs, short enough that a
# stalled endpoint does not hang the agent indefinitely.
DEFAULT_EXECUTION_TIMEOUT = 720


class ToolExecutor:
    """Execute tools using per-Tool mock/real modes with an optional timeout."""

    def __init__(
        self,
        tool_modes: Optional[Dict[str, str]] = None,
        execution_timeout: Optional[int] = DEFAULT_EXECUTION_TIMEOUT,
    ):
        self.tool_modes = dict(tool_modes or DEFAULT_TOOL_MODES)
        self.execution_timeout = execution_timeout

    @staticmethod
    def _execution_error(tool_name: str, error: Exception) -> Dict[str, Any]:
        """Convert a Tool exception into the standard Tool error envelope."""

        return {
            "status": "error",
            "tool": tool_name,
            "error": {
                "type": "execution_error",
                "message": str(error),
            },
        }

    @staticmethod
    def _invalid_result_error(
        tool_name: str,
        validation_errors: list,
    ) -> Dict[str, Any]:
        """Convert an invalid Tool return value into a standard error result."""

        return {
            "status": "error",
            "tool": tool_name,
            "error": {
                "type": "invalid_tool_result",
                "message": "Tool returned an invalid result envelope.",
                "validation_errors": validation_errors,
            },
        }

    @staticmethod
    def _timeout_error(tool_name: str, timeout: int) -> Dict[str, Any]:
        """Convert a tool timeout into the standard Tool error envelope."""

        return {
            "status": "error",
            "tool": tool_name,
            "error": {
                "type": "tool_timeout",
                "message": (
                    f"Tool {tool_name!r} did not finish within {timeout} seconds. "
                    "You may retry this tool with the same or adjusted arguments, "
                    "or adjust the plan."
                ),
            },
        }

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a Tool with its configured mode and validate the result."""

        try:
            try:
                mode = self.tool_modes[tool_name]
            except KeyError as exc:
                raise ValueError(
                    f"No mock/real mode configured for tool {tool_name!r}"
                ) from exc

            if self.execution_timeout:
                # Run the tool in a daemon thread so a stuck tool neither
                # blocks the agent past the timeout nor prevents interpreter
                # exit. Non-daemon threads (e.g. ThreadPoolExecutor workers)
                # would keep the process alive until the stuck call returns.
                result_queue: "queue.Queue" = queue.Queue(maxsize=1)

                def _run() -> None:
                    try:
                        result_queue.put(
                            (True, call_tool(tool_name, arguments, mode=mode))
                        )
                    except Exception as exc:  # noqa: BLE001
                        result_queue.put((False, exc))

                worker = threading.Thread(target=_run, daemon=True)
                worker.start()
                try:
                    ok, payload = result_queue.get(timeout=self.execution_timeout)
                except queue.Empty:
                    return self._timeout_error(tool_name, self.execution_timeout)
                if not ok:
                    raise payload
                result = payload
            else:
                result = call_tool(
                    tool_name,
                    arguments,
                    mode=mode,
                )
        except Exception as exc:
            return self._execution_error(
                tool_name,
                exc,
            )

        valid, errors = validate_tool_result(
            result,
            tool_name,
        )
        if not valid:
            return self._invalid_result_error(
                tool_name,
                errors,
            )

        return result
