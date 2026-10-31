"""Registry and dispatcher for local Tool implementations."""

from __future__ import annotations

from typing import Callable

from tools.tool01_failure_analyzer.mock import run as mock_failure_analyzer
from tools.tool01_failure_analyzer.real import run as real_failure_analyzer
from tools.tool02_scene_requirement_generator.mock import run as mock_scene_requirement_generator
from tools.tool02_scene_requirement_generator.real import run as real_scene_requirement_generator
from tools.tool03_scene_conceiver.mock import run as mock_scene_conceiver
from tools.tool03_scene_conceiver.real import run as real_scene_conceiver
from tools.tool04_video_retriever.mock import run as mock_video_retriever
from tools.tool04_video_retriever.real import run as real_video_retriever
from tools.tool05_instance_discovery.mock import run as mock_instance_discovery
from tools.tool05_instance_discovery.real import run as real_instance_discovery
from tools.tool06_reconstruction.mock import run as mock_reconstruction
from tools.tool06_reconstruction.real import run as real_reconstruction
from tools.tool07_edit_plan_builder.mock import run as mock_edit_plan_builder
from tools.tool07_edit_plan_builder.real import run as real_edit_plan_builder
from tools.tool08_edit_config_generator.mock import run as mock_edit_config_generator
from tools.tool08_edit_config_generator.real import run as real_edit_config_generator
from tools.tool09_scene_editor.mock import run as mock_scene_editor
from tools.tool09_scene_editor.real import run as real_scene_editor
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
    "failure_analyzer": mock_failure_analyzer,
    "scene_requirement_generator": mock_scene_requirement_generator,
    "scene_conceiver": mock_scene_conceiver,
    "video_retriever": mock_video_retriever,
    "instance_discovery": mock_instance_discovery,
    "reconstruction": mock_reconstruction,
    "edit_plan_builder": mock_edit_plan_builder,
    "edit_config_generator": mock_edit_config_generator,
    "scene_editor": mock_scene_editor,
    "scene_validator": mock_scene_validator,
    "policy_runner": mock_policy_runner,
    "failure_case_filter": mock_failure_case_filter,
    "policy_trainer": mock_policy_trainer,
}

REAL_TOOL_REGISTRY: dict[str, ToolCallable] = {
    "failure_analyzer": real_failure_analyzer,
    "scene_requirement_generator": real_scene_requirement_generator,
    "scene_conceiver": real_scene_conceiver,
    "video_retriever": real_video_retriever,
    "reconstruction": real_reconstruction,
    "instance_discovery": real_instance_discovery,
    "edit_plan_builder": real_edit_plan_builder,
    "edit_config_generator": real_edit_config_generator,
    "scene_editor": real_scene_editor,
    "scene_validator": real_scene_validator,
    "policy_runner": real_policy_runner,
    "policy_trainer": real_policy_trainer,
    "failure_case_filter": real_failure_case_filter,
}


from tools.contracts import validated
_BACKEND_MODULES = {
    "failure_analyzer": "tools.tool01_failure_analyzer",
    "video_retriever": "tools.tool04_video_retriever",
    "instance_discovery": "tools.tool05_instance_discovery",
    "reconstruction": "tools.tool06_reconstruction",
    "edit_config_generator": "tools.tool08_edit_config_generator",
    "scene_editor": "tools.tool09_scene_editor",
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
