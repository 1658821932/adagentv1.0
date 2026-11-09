"""Canonical ten-tool registry; numbering matches the public workflow order."""
from __future__ import annotations
from typing import Callable
from tools.contracts import validated

from tools.tool01_failure_analyzer.mock import run as mock_failure_analyzer
from tools.tool01_failure_analyzer.real import run as real_failure_analyzer
from tools.tool02_scene_instance_generator.mock import run as mock_scene_instance_generator
from tools.tool02_scene_instance_generator.real import run as real_scene_instance_generator
from tools.tool03_video_retriever.mock import run as mock_video_retriever
from tools.tool03_video_retriever.real import run as real_video_retriever
from tools.tool04_edit_planner.mock import run as mock_edit_planner
from tools.tool04_edit_planner.real import run as real_edit_planner
from tools.tool05_attack_vehicle_editor.mock import run as mock_attack_vehicle_editor
from tools.tool05_attack_vehicle_editor.real import run as real_attack_vehicle_editor
from tools.tool06_ego_trajectory_editor.mock import run as mock_ego_trajectory_editor
from tools.tool06_ego_trajectory_editor.real import run as real_ego_trajectory_editor
from tools.tool08_policy_runner.mock import run as mock_policy_runner
from tools.tool08_policy_runner.real import run as real_policy_runner
from tools.tool09_failure_case_filter.mock import run as mock_failure_case_filter
from tools.tool09_failure_case_filter.real import run as real_failure_case_filter
from tools.tool10_policy_trainer.mock import run as mock_policy_trainer
from tools.tool10_policy_trainer.real import run as real_policy_trainer

ToolCallable = Callable[[dict], dict]

MOCK_TOOL_REGISTRY: dict[str, ToolCallable] = {
    "failure_analyzer": mock_failure_analyzer,
    "scene_instance_generator": mock_scene_instance_generator,
    "video_retriever": mock_video_retriever,
    "edit_planner": mock_edit_planner,
    "attack_vehicle_editor": mock_attack_vehicle_editor,
    "ego_trajectory_editor": mock_ego_trajectory_editor,
    "policy_runner": mock_policy_runner,
    "failure_case_filter": mock_failure_case_filter,
    "policy_trainer": mock_policy_trainer,
}

REAL_TOOL_REGISTRY: dict[str, ToolCallable] = {
    "failure_analyzer": real_failure_analyzer,
    "scene_instance_generator": real_scene_instance_generator,
    "video_retriever": real_video_retriever,
    "edit_planner": real_edit_planner,
    "attack_vehicle_editor": real_attack_vehicle_editor,
    "ego_trajectory_editor": real_ego_trajectory_editor,
    "policy_runner": real_policy_runner,
    "failure_case_filter": real_failure_case_filter,
    "policy_trainer": real_policy_trainer,
}

_BACKEND_MODULES = {
    "failure_analyzer": "tools.tool01_failure_analyzer",
    "video_retriever": "tools.tool03_video_retriever",
    "attack_vehicle_editor": "tools.tool05_attack_vehicle_editor",
    "ego_trajectory_editor": "tools.tool06_ego_trajectory_editor",
    "policy_runner": "tools.tool08_policy_runner",
    "failure_case_filter": "tools.tool09_failure_case_filter",
    "policy_trainer": "tools.tool10_policy_trainer",
}
for _name, _module in _BACKEND_MODULES.items():
    MOCK_TOOL_REGISTRY[_name] = validated(_name, _module, MOCK_TOOL_REGISTRY[_name])
    REAL_TOOL_REGISTRY[_name] = validated(_name, _module, REAL_TOOL_REGISTRY[_name])

DEFAULT_TOOL_MODES = {name: 'real' for name in MOCK_TOOL_REGISTRY}


def call_tool(tool_name: str, input_data: dict, mode: str | None = None) -> dict:
    if tool_name not in MOCK_TOOL_REGISTRY:
        raise ValueError(f"Unknown tool {tool_name!r}. Available tools: {', '.join(MOCK_TOOL_REGISTRY)}.")
    selected_mode = mode or DEFAULT_TOOL_MODES[tool_name]
    if selected_mode not in {'mock', 'real'}:
        raise ValueError(f"Unsupported mode {selected_mode!r} for {tool_name!r}; expected 'mock' or 'real'.")
    return (MOCK_TOOL_REGISTRY if selected_mode == 'mock' else REAL_TOOL_REGISTRY)[tool_name](input_data)
