"""
Project runtime configuration.

Non-sensitive Agent runtime settings live in configs/agent.json.
Secrets remain in the local .env file; non-sensitive LLM settings use JSON.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from agent.tool_descriptions import get_tool_descriptions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "agent.json"
VALID_TOOL_MODES = {"mock", "real"}


@dataclass(frozen=True)
class AgentConfig:
    """Validated runtime configuration for the Agent."""

    max_steps: int
    decision_correction_attempts: int
    tool_modes: Dict[str, str]
    logs_root: Path
    reconstruction_backend: str = "ufo"

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AgentConfig":
        """Load and validate runtime configuration from JSON."""

        config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

        try:
            with config_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Agent config file not found: {config_path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Agent config is not valid JSON: {config_path}"
            ) from exc

        try:
            max_steps = data["agent"]["max_steps"]
            correction_attempts = data["orchestrator"][
                "decision_correction_attempts"
            ]
            tool_modes = data["tools"]["modes"]
            logs_root_value = data["logging"]["root_dir"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "Agent config is missing required fields"
            ) from exc

        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
            raise ValueError("agent.max_steps must be a positive integer")

        if (
            not isinstance(correction_attempts, int)
            or isinstance(correction_attempts, bool)
            or correction_attempts < 0
        ):
            raise ValueError(
                "orchestrator.decision_correction_attempts must be "
                "a non-negative integer"
            )

        if not isinstance(tool_modes, dict):
            raise ValueError("tools.modes must be a JSON object")

        known_tools = {tool["name"] for tool in get_tool_descriptions()}
        configured_tools = set(tool_modes)
        missing_tools = sorted(known_tools - configured_tools)
        unknown_tools = sorted(configured_tools - known_tools)

        if missing_tools:
            raise ValueError(
                "tools.modes is missing tools: " + ", ".join(missing_tools)
            )
        if unknown_tools:
            raise ValueError(
                "tools.modes contains unknown tools: " + ", ".join(unknown_tools)
            )

        for tool_name, mode in tool_modes.items():
            if mode not in VALID_TOOL_MODES:
                raise ValueError(
                    f"tools.modes.{tool_name} must be one of: mock, real"
                )

        if not isinstance(logs_root_value, str) or not logs_root_value.strip():
            raise ValueError("logging.root_dir must be a non-empty string")

        logs_root = Path(logs_root_value)
        if not logs_root.is_absolute():
            logs_root = PROJECT_ROOT / logs_root

        backend = os.environ.get('RECONSTRUCTION_BACKEND') or data['agent'].get('reconstruction_backend', 'ufo')
        if backend not in {'ufo', 'dggt'}:
            raise ValueError('reconstruction_backend must be ufo or dggt')
        return cls(
            max_steps=max_steps,
            decision_correction_attempts=correction_attempts,
            tool_modes=dict(tool_modes),
            logs_root=logs_root,
            reconstruction_backend=backend,
        )
