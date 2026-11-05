"""Registry and dispatcher for local Tool implementations."""

from __future__ import annotations

from typing import Callable

from tools.tool01_failure_analyzer.mock import run as mock_failure_analyzer
from tools.tool01_failure_analyzer.real import run as real_failure_analyzer
from tools.tool02_scene_instance_generator.mock import run as mock_scene_instance_generator
from tools.tool02_scene_instance_generator.real import run as real_scene_instance_generator
from tools.tool04_video_retriever.mock import run as mock_video_retriever
from tools.tool04_video_retriever.real import run as real_video_retriever
from tools.tool05_attack_vehicle_editor.mock import run as mock_attack_vehicle_editor
from tools.tool05_attack_vehicle_editor.real import run as real_attack_vehicle_editor
from tools.tool06_ego_trajectory_editor.mock import run as mock_ego_trajectory_editor
from tools.tool06_ego_trajectory_editor.real import run as real_ego_trajectory_editor
from tools.tool10_scene_validator.mock import run as mock_scene_validator
from tools.tool10_scene_validator.real import run as real_scene_validator
from tools.tool13_policy_runner.mock import run as mock_policy_runner
from tools.tool13_policy_runner.real import run as real_policy_runner
from tools.tool14_failure_case_filter.mock import run as mock_failure_case_filter
from tools.tool14_failure_case_filter.real import run as real_failure_case_filter
from tools.tool15_policy_trainer.mock import run as mock_policy_trainer
from tools.tool15_policy_trainer.real import run as real_policy_trainer


ToolCallable = Callable[[dict], dict]

MOCK_TOOL_REGISTRY: dict[str, ToolCallable] = {
    "attack_vehicle_editor": mock_attack_vehicle_editor,
    "ego_trajectory_editor": mock_ego_trajectory_editor,

    "failure_analyzer": mock_failure_analyzer,
    "scene_instance_generator": mock_scene_instance_generator,
    "video_retriever": mock_video_retriever,
    "scene_validator": mock_scene_validator,
    "policy_runner": mock_policy_runner,
    "failure_case_filter": mock_failure_case_filter,
    "policy_trainer": mock_policy_trainer,
}

REAL_TOOL_REGISTRY: dict[str, ToolCallable] = {
    "attack_vehicle_editor": real_attack_vehicle_editor,
    "ego_trajectory_editor": real_ego_trajectory_editor,

    "failure_analyzer": real_failure_analyzer,
    "scene_instance_generator": real_scene_instance_generator,
    "video_retriever": real_video_retriever,
    "scene_validator": real_scene_validator,
    "policy_runner": real_policy_runner,
    "policy_trainer": real_policy_trainer,
    "failure_case_filter": real_failure_case_filter,
}


from tools.contracts import validated
_BACKEND_MODULES = {
    "attack_vehicle_editor": "tools.tool05_attack_vehicle_editor",
    "ego_trajectory_editor": "tools.tool06_ego_trajectory_editor",

    "failure_analyzer": "tools.tool01_failure_analyzer",
    "video_retriever": "tools.tool04_video_retriever",
    "scene_validator": "tools.tool10_scene_validator",
    "policy_runner": "tools.tool13_policy_runner",
    "failure_case_filter": "tools.tool14_failure_case_filter",
    "policy_trainer": "tools.tool15_policy_trainer",
}
for _name, _module in _BACKEND_MODULES.items():
    MOCK_TOOL_REGISTRY[_name] = validated(_name, _module, MOCK_TOOL_REGISTRY[_name])
    REAL_TOOL_REGISTRY[_name] = validated(_name, _module, REAL_TOOL_REGISTRY[_name])


DEFAULT_TOOL_MODES = {
    tool_name: ("real" if tool_name in REAL_TOOL_REGISTRY else "mock")
    for tool_name in MOCK_TOOL_REGISTRY
}


def call_tool(tool_name: str, input_data: dict, mode: str | None = None) -> dict:
    """Call one Tool using its mock or real implementation.

    When mode is omitted, the current project default is used: Tools with
    an implemented real backend default to real and all others default to
    mock. The Agent itself always passes the mode from configs/agent.json.
    """

    if tool_name not in MOCK_TOOL_REGISTRY:
        available = ", ".join(MOCK_TOOL_REGISTRY)
        raise ValueError(
            f"Unknown tool {tool_name!r}. Available tools: {available}."
        )

    selected_mode = mode or DEFAULT_TOOL_MODES[tool_name]
    if selected_mode not in {"mock", "real"}:
        raise ValueError(
            f"Unsupported mode {selected_mode!r} for {tool_name!r}; "
            "expected 'mock' or 'real'."
        )

    if selected_mode == "mock":
        return MOCK_TOOL_REGISTRY[tool_name](input_data)

    try:
        real_tool = REAL_TOOL_REGISTRY[tool_name]
    except KeyError as exc:
        raise RuntimeError(
            f"Tool {tool_name!r} has no real implementation yet. "
            "Set its mode to 'mock' in configs/agent.json."
        ) from exc

    return real_tool(input_data)
