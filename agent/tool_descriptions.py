"""
Tool descriptions for the LLM Orchestrator.

Aggregator only: each tool's detailed description now lives in that tool's
own description.py (tools/<tool>/description.py). This module imports and
combines them into the single TOOLS list consumed by the orchestrator.
"""

from tools.tool01_failure_analyzer.description import DESCRIPTION as TOOL01
from tools.tool02_scene_instance_generator.description import DESCRIPTION as TOOL02
from tools.tool04_video_retriever.description import DESCRIPTION as TOOL04
from tools.tool05_attack_vehicle_editor.description import DESCRIPTION as TOOL05
from tools.tool06_ego_trajectory_editor.description import DESCRIPTION as TOOL06
from tools.tool10_scene_validator.description import DESCRIPTION as TOOL10
from tools.tool13_policy_runner.description import DESCRIPTION as TOOL13
from tools.tool14_failure_case_filter.description import DESCRIPTION as TOOL14
from tools.tool15_policy_trainer.description import DESCRIPTION as TOOL15

TOOLS = [
    TOOL01,
    TOOL02,
    TOOL04,
    TOOL05,
    TOOL06,
    TOOL10,
    TOOL13,
    TOOL14,
    TOOL15,
]


def get_tool_descriptions(*, compact=False):
    if not compact:
        return TOOLS
    return [_compact_description(tool) for tool in TOOLS]


def _compact_description(tool):
    """Keep executable input constraints; omit duplicated/output JSON schemas."""
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items() if key != "title"}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value
    return {
        "name": tool["name"],
        "description": tool["description"],
        "when_to_use": tool.get("when_to_use", ""),
        "handoff": dict(tool.get("handoff", {})),
        "rules": list(tool.get("rules", [])),
        "input_schema": clean(tool["input_schema"]),
        "outputs": list(tool.get("outputs", {})),
    }
