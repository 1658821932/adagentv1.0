"""
Tool description for the failure_analyzer tool (tool01_failure_analyzer).

This is the single source of truth shown to the LLM orchestrator when it
decides whether and how to call this tool. Keep orchestrator-facing rules
about this tool here, not in the global system prompt.
"""

DESCRIPTION = {
    "name": 'failure_analyzer',
    "description": 'Analyze one WorldEngine scene token using official metrics, image evidence and numeric verification.',
    "when_to_use": 'Use for evidence-based analysis of an existing WorldEngine scene with archived evaluation data. Supply token and, when needed, config_path.',
    'inputs': {
        'input_path': 'JSON file containing failure_cases with image paths and failure metadata; alternatively pass failure_cases directly as a list in the arguments.',
    },
    'outputs': {
        'failure_analysis': 'One dominant cross-case failure_type, failure_reason, key_scene_factors and risk_factors, with supporting and uncovered case IDs. Use only case IDs actually present in the input; never invent supporting evidence.',
    },
}

from .schemas import Input, Output
DESCRIPTION["input_schema"] = Input.model_json_schema()
DESCRIPTION["output_schema"] = Output.model_json_schema()
DESCRIPTION["inputs"] = {key: DESCRIPTION["inputs"].get(key, value.get("description", value.get("title", key))) for key,value in DESCRIPTION["input_schema"]["properties"].items()}

DESCRIPTION["outputs"] = {key: DESCRIPTION["outputs"].get(key, value.get("description", value.get("title", key))) for key,value in DESCRIPTION["output_schema"]["properties"].items()}
