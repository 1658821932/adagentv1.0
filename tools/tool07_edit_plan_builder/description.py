"""
Tool description for the edit_plan_builder tool (tool07_edit_plan_builder).

This is the single source of truth shown to the LLM orchestrator when it
decides whether and how to call this tool. Keep orchestrator-facing rules
about this tool here, not in the global system prompt.
"""

DESCRIPTION = {
    "name": 'edit_plan_builder',
    "description": 'Use the LLM to turn scene requirements and instances into a precise natural-language editing plan.',
    "when_to_use": 'Use after reconstruction and instance discovery when an edit must be specified.',
    'inputs': {
        'edit_request': 'High-level natural-language edit request; for front-car cut-in target ego-lane entry near frame 12 and lane holding through frame 39, with zero world-height translation. Trajectory offsets must compensate the original instance per-frame motion.',
        'scene_info': 'Scene and instance context.',
        'instances': 'Discovered instances from instance_discovery.',
    },
    'outputs': {
        'edit_plan': 'Detailed natural-language editing plan and operations.',
    },
}
