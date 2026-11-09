"""Public tool catalog, ordered tool01 through tool10."""
from tools.tool01_failure_analyzer.description import DESCRIPTION as TOOL01
from tools.tool02_scene_instance_generator.description import DESCRIPTION as TOOL02
from tools.tool03_video_retriever.description import DESCRIPTION as TOOL03
from tools.tool04_edit_planner.description import DESCRIPTION as TOOL04
from tools.tool05_attack_vehicle_editor.description import DESCRIPTION as TOOL05
from tools.tool06_ego_trajectory_editor.description import DESCRIPTION as TOOL06
from tools.tool08_policy_runner.description import DESCRIPTION as TOOL08
from tools.tool09_failure_case_filter.description import DESCRIPTION as TOOL09
from tools.tool10_policy_trainer.description import DESCRIPTION as TOOL10
TOOLS = [TOOL01, TOOL02, TOOL03, TOOL04, TOOL05, TOOL06, TOOL08, TOOL09, TOOL10]



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
