"""
Tool descriptions for the LLM Orchestrator.

Aggregator only: each tool's detailed description now lives in that tool's
own description.py (tools/<tool>/description.py). This module imports and
combines them into the single TOOLS list consumed by the orchestrator.
"""

from tools.tool01_failure_analyzer.description import DESCRIPTION as TOOL01
from tools.tool02_scene_requirement_generator.description import DESCRIPTION as TOOL02
from tools.tool03_scene_conceiver.description import DESCRIPTION as TOOL03
from tools.tool04_video_retriever.description import DESCRIPTION as TOOL04
from tools.tool05_instance_discovery.description import DESCRIPTION as TOOL05
from tools.tool06_reconstruction.description import DESCRIPTION as TOOL06
from tools.tool07_edit_plan_builder.description import DESCRIPTION as TOOL07
from tools.tool08_edit_config_generator.description import DESCRIPTION as TOOL08
from tools.tool09_scene_editor.description import DESCRIPTION as TOOL09
from tools.tool10_scene_validator.description import DESCRIPTION as TOOL10
from tools.tool13_policy_runner.description import DESCRIPTION as TOOL13
from tools.tool14_failure_case_filter.description import DESCRIPTION as TOOL14
from tools.tool15_policy_trainer.description import DESCRIPTION as TOOL15

TOOLS = [
    TOOL01,
    TOOL02,
    TOOL03,
    TOOL04,
    TOOL05,
    TOOL06,
    TOOL07,
    TOOL08,
    TOOL09,
    TOOL10,
    TOOL13,
    TOOL14,
    TOOL15,
]


def get_tool_descriptions():
    return TOOLS
