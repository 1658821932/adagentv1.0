"""
Tool description for the failure_case_filter tool (tool14_failure_case_filter).

This is the single source of truth shown to the LLM orchestrator when it
decides whether and how to call this tool. Keep orchestrator-facing rules
about this tool here, not in the global system prompt.
"""

DESCRIPTION = {
    "name": 'failure_case_filter',
    "description": 'Select failure scenes exclusively using saved official closed-loop PDM total scores.',
    "when_to_use": 'Use after policy_runner. Official PDM CSV and matching execution metadata are required. score == 0 selects failures. Missing scores raise an error.',
    'inputs': {
        'scenario_pkl': 'WorldEngine custom scenario pickle containing ego and dynamic-object tracks.',
        'result_dir': 'Required policy_runner result directory containing official PDM scores. Scene identities must match executed metadata; missing or ambiguous matches fail explicitly.',
        'scene_token': 'Optional full scene ID or unambiguous 16-character suffix to evaluate only one scene.',
    },
    'outputs': {
        'is_failure_case': 'Whether any selected scene has official PDM total score == 0.',
        'failure_cases': 'Selected zero-score scenes with scene_id, scenario_pkl and pdm_score.',
    },
}

from .schemas import Input, Output
DESCRIPTION["input_schema"] = Input.model_json_schema()
DESCRIPTION["output_schema"] = Output.model_json_schema()
DESCRIPTION["inputs"] = {key: DESCRIPTION["inputs"].get(key, value.get("description", value.get("title", key))) for key,value in DESCRIPTION["input_schema"]["properties"].items()}

DESCRIPTION["outputs"] = {key: DESCRIPTION["outputs"].get(key, value.get("description", value.get("title", key))) for key,value in DESCRIPTION["output_schema"]["properties"].items()}
