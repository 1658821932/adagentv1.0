"""
Tool description for the failure_analyzer tool (tool01_failure_analyzer).

This is the single source of truth shown to the LLM orchestrator when it
decides whether and how to call this tool. Keep orchestrator-facing rules
about this tool here, not in the global system prompt.
"""

DESCRIPTION = {
    "name": 'failure_analyzer',
    "description": 'Analyze autonomous driving failure cases and summarize failure type, failure reason, key scene factors and risk factors. Internally this tool may call an LLM/VLM for cross-case summarization.',
    "when_to_use": 'Use when the agent receives failure cases and needs to understand why the driving model failed. This is the usual first step of the generation loop.',
    'inputs': {
        'input_path': 'JSON file containing failure_cases with image paths and failure metadata; alternatively pass failure_cases directly as a list in the arguments.',
    },
    'outputs': {
        'failure_analysis': 'One dominant cross-case failure_type, failure_reason, key_scene_factors and risk_factors, with supporting and uncovered case IDs. Use only case IDs actually present in the input; never invent supporting evidence.',
    },
}
