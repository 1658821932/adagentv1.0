"""
Tool execution layer.

The Agent selects a Tool; this executor selects that Tool's configured
implementation mode (mock or real) and validates the standard result envelope.
"""

from typing import Any, Dict, Optional

from agent.schemas import validate_tool_result
from tools.registry import DEFAULT_TOOL_MODES, call_tool


class ToolExecutor:
    """Execute tools using per-Tool mock/real modes."""

    def __init__(self, tool_modes: Optional[Dict[str, str]] = None):
        self.tool_modes = dict(tool_modes or DEFAULT_TOOL_MODES)

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

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a Tool with its configured mode and validate the result."""

        try:
            try:
                mode = self.tool_modes[tool_name]
            except KeyError as exc:
                raise ValueError(
                    f"No mock/real mode configured for tool {tool_name!r}"
                ) from exc

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
