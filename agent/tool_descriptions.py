"""Public tool catalog, ordered tool01 through tool08."""
from tools.tool01_failure_analyzer.description import DESCRIPTION as TOOL01
from tools.tool02_scene_instance_generator.description import DESCRIPTION as TOOL02
from tools.tool03_video_retriever.description import DESCRIPTION as TOOL03
from tools.tool04_edit_planner.description import DESCRIPTION as TOOL04
from tools.tool05_attack_vehicle_editor.description import DESCRIPTION as TOOL05
from tools.tool06_policy_runner.description import DESCRIPTION as TOOL06
from tools.tool07_failure_case_filter.description import DESCRIPTION as TOOL07
from tools.tool08_policy_trainer.description import DESCRIPTION as TOOL08
TOOLS = [TOOL01, TOOL02, TOOL03, TOOL04, TOOL05, TOOL06, TOOL07, TOOL08]

# Handoff fields are passed by reference: the LLM sends the upstream step's
# result_ref path string, never an inlined copy of a previous tool result.
# A task-supplied bound scene is also a supported direct attack input.
REFERENCE_FIELDS = {
    "edit_planner": {"retrieval_result": "Absolute result_ref path of the successful video_retriever history step."},
    "attack_vehicle_editor": {"scene_ref": "Absolute result_ref path of the successful video_retriever history step."},
    "policy_runner": {"attack_result": "Absolute result_ref path of the successful attack_vehicle_editor history step."},
}



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


def get_function_tools():
    """Advertise executable schemas once, in the provider's native tools field."""
    from copy import deepcopy
    import json
    functions = []
    for tool in get_tool_descriptions(compact=True):
        arguments = deepcopy(tool["input_schema"])
        for field, ref_description in REFERENCE_FIELDS.get(tool["name"], {}).items():
            if field in arguments.get("properties", {}):
                if tool["name"] == "attack_vehicle_editor" and field == "scene_ref":
                    arguments["properties"][field]["description"] = (
                        ref_description + " Prefer the exact result_ref string for an existing retrieval result; "
                        "do not reconstruct it from a summary. A complete bound scene_ref object supplied "
                        "directly by the task or a ready plan is also supported; no new retrieval is required."
                    )
                    continue
                arguments["properties"][field] = {
                    "type": "string", "minLength": 1,
                    "description": ref_description + " Reference passing: send the path string verbatim; do not inline or retype the JSON.",
                }
        for field, prop in list(arguments.get("properties", {}).items()):
            if isinstance(prop, dict) and prop.get("description", "").startswith("Mock-only"):
                del arguments["properties"][field]
                if "required" in arguments and field in arguments["required"]:
                    arguments["required"].remove(field)
        definitions = arguments.pop("$defs", {})
        parameters = {
            "type":"object",
            "properties":{
                "reason":{"type":"string","minLength":1,"description":"Why this call advances the task."},
                "arguments":arguments,
                "execution":{"type":"object","properties":{"retry":{"type":"boolean"},
                    "background":{"type":"boolean","description":"For policy_runner (tool06) only: submit and return the persistent job without waiting."},
                    "job_id":{"type":"string","minLength":1,"description":"Reconnect to the existing same-tool job; keep original arguments and use background=false to wait."}},
                             "additionalProperties":False,
                             "description":"Set retry=true only for a deliberate retry/new execution. Background jobs must be collected before finish."},
            },
            "required":["reason","arguments"],"additionalProperties":False,
        }
        if definitions:
            parameters["$defs"] = definitions
        description = {k:v for k,v in tool.items() if k not in ("name","input_schema")}
        functions.append({"type":"function","function":{
            "name":tool["name"],"description":json.dumps(description,ensure_ascii=False),
            "parameters":parameters}})
    for name, purpose in [("finish","Finish only when the requested outputs are complete."),
                          ("stop","Stop and explain a blocker; the task remains incomplete.")]:
        functions.append({"type":"function","function":{"name":name,"description":purpose,
            "parameters":{"type":"object","properties":{"reason":{"type":"string","minLength":1}},
                          "required":["reason"],"additionalProperties":False}}})
    return functions
