"""
Tool description for the scene_requirement_generator tool (tool02_scene_requirement_generator).

This is the single source of truth shown to the LLM orchestrator when it
decides whether and how to call this tool. Keep orchestrator-facing rules
about this tool here, not in the global system prompt.
"""

DESCRIPTION = {
    "name": 'scene_requirement_generator',
    "description": 'Generate targeted scenario requirements from failure analysis.',
    "when_to_use": 'Use after failure analysis to define what new scenario should be generated.',
    'inputs': {
        'failure_analysis': 'Failure analysis result produced by failure_analyzer.',
    },
    'outputs': {
        'scene_requirement': 'Scenario conditions, actors, risks and target effects. These are design targets for generation, not observed facts.',
    },
}
