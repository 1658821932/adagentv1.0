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

    def __init__(self, tool_modes: Optional[Dict[str, str]] = None, reconstruction_backend: str = 'ufo'):
        self.tool_modes = dict(tool_modes or DEFAULT_TOOL_MODES)
        if reconstruction_backend not in {'ufo', 'dggt'}:
            raise ValueError('reconstruction_backend must be ufo or dggt')
        self.reconstruction_backend = reconstruction_backend
        self.reconstruction = None

    def reset(self):
        self.reconstruction = None

    def prepare_arguments(self, tool_name, arguments):
        arguments = dict(arguments)
        if tool_name == 'scene_validator':
            arguments['backend'] = self.reconstruction_backend
        if tool_name in {'reconstruction', 'scene_editor', 'edit_config_generator', 'instance_discovery', 'edit_plan_builder'}:
            arguments['backend'] = ('dggt_yolo' if tool_name == 'instance_discovery'
                                    and self.reconstruction_backend == 'dggt' else self.reconstruction_backend)
            if self.reconstruction_backend == 'ufo':
                if tool_name in {'instance_discovery', 'scene_editor'}:
                    if self.reconstruction is None:
                        raise ValueError('Run reconstruction successfully before UFO discovery/editing')
                    arguments['ufo_reconstruction'] = self.reconstruction
                if tool_name == 'scene_editor':
                    operation = arguments.get('operation', 'lane_change')
                    if operation not in {'lane_change', 'metric_edit', 'calibrate_metric'}:
                        raise ValueError('UFO Agent supports lane_change, calibrate_metric and metric_edit')
                    arguments['operation'] = operation
                    if operation == 'metric_edit':
                        import json
                        from pathlib import Path
                        from mcp_servers.ufo.metric_config import validate
                        config = validate(json.loads(Path(arguments['edit_config_path']).expanduser().read_text()), agent=True)
                        if config.get('target_selection_path'):
                            supplied = arguments.get('target_selection_path')
                            if supplied and Path(supplied).resolve() != Path(config['target_selection_path']).resolve():
                                raise ValueError('Editor target differs from metric config')
                            arguments['target_selection_path'] = config['target_selection_path']
                    elif operation == 'lane_change' and (not arguments.get('edit_config_path') or not arguments.get('target_selection_path')):
                        raise ValueError('UFO Agent editing requires edit_config_path and target_selection_path')
        return arguments

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

            if mode == 'real':
                if tool_name == 'reconstruction':
                    self.reconstruction = None
                arguments = self.prepare_arguments(tool_name, arguments)
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

        if tool_name == 'reconstruction' and result.get('status') == 'success':
            self.reconstruction = result['result']

        return result
