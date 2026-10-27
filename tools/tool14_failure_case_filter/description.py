"""
Tool description for the failure_case_filter tool (tool14_failure_case_filter).

This is the single source of truth shown to the LLM orchestrator when it
decides whether and how to call this tool. Keep orchestrator-facing rules
about this tool here, not in the global system prompt.
"""

DESCRIPTION = {
    "name": 'failure_case_filter',
    "description": 'Filter new failure cases from simulation results and store useful training data.',
    "when_to_use": 'Use after policy execution.',
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
