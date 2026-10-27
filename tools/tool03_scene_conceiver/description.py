"""
Tool description for the scene_conceiver tool (tool03_scene_conceiver).

This is the single source of truth shown to the LLM orchestrator when it
decides whether and how to call this tool. Keep orchestrator-facing rules
about this tool here, not in the global system prompt.
"""

DESCRIPTION = {
    "name": 'scene_conceiver',
    "description": 'Design a high-level scenario generation plan, including required base video and simple editing ideas.',
    "when_to_use": 'Use after obtaining scenario requirements and before retrieving data.',
    'inputs': {
        'scene_requirement': 'Target scenario requirements from scene_requirement_generator.',
    },
    'outputs': {
        'scene_plan': 'Base video requirements, edit ideas and expected effects.',
    },
}
