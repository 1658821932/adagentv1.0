"""
Tool description for the failure_case_filter tool (tool14_failure_case_filter).

This is the single source of truth shown to the LLM orchestrator when it
decides whether and how to call this tool. Keep orchestrator-facing rules
about this tool here, not in the global system prompt.
"""

DESCRIPTION = {
    "name": 'failure_case_filter',
    "description": 'Filter new failure cases from simulation results and store useful training data.',
    "when_to_use": 'Optional dynamic-object geometric diagnosis after execution. Not an RL reward calculator or road-compliance evaluator. For RL use policy_runner dense rewards and policy_trainer sample_filter.',
    'inputs': {
        'scenario_pkl': 'WorldEngine custom scenario pickle containing ego and dynamic-object tracks.',
        'result_dir': 'Optional policy_runner result directory containing the actual WorldEngine ego trajectory.',
    },
    'outputs': {
        'is_failure_case': 'Whether a dynamic-vehicle collision failure was found.',
        'collision': 'Dynamic-vehicle collision result.',
        'failure_cases': 'Decision failures found by geometric collision evaluation.',
    },
}

from .schemas import Input, Output
DESCRIPTION["input_schema"] = Input.model_json_schema()
DESCRIPTION["output_schema"] = Output.model_json_schema()
DESCRIPTION["inputs"] = {key: DESCRIPTION["inputs"].get(key, value.get("description", value.get("title", key))) for key,value in DESCRIPTION["input_schema"]["properties"].items()}

DESCRIPTION["outputs"] = {key: DESCRIPTION["outputs"].get(key, value.get("description", value.get("title", key))) for key,value in DESCRIPTION["output_schema"]["properties"].items()}
