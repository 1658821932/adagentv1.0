"""
Tool description for the recovery_decider tool (tool11_recovery_decider).

This is the single source of truth shown to the LLM orchestrator when it
decides whether and how to call this tool. Keep orchestrator-facing rules
about this tool here, not in the global system prompt.
"""

DESCRIPTION = {
    "name": 'recovery_decider',
    "description": 'Decide recovery actions when validation or execution fails.',
    "when_to_use": 'Use when a previous step fails and rollback or replanning is needed.',
    'inputs': {
        'evaluation': 'Failure information and execution history.',
    },
    'outputs': {
        'recovery_action': 'Retry, rollback or replan decision.',
    },
}
