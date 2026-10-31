"""
Tool description for the scene_validator tool (tool10_scene_validator).

This is the single source of truth shown to the LLM orchestrator when it
decides whether and how to call this tool. Keep orchestrator-facing rules
about this tool here, not in the global system prompt.
"""

DESCRIPTION = {
    "name": 'scene_validator',
    "description": 'Evaluate whether an edited scene satisfies requirements and remains physically and logically reasonable.',
    "when_to_use": 'Use after editing before simulation. Run visual validation on at least 8 uniformly spaced frames and treat bad grounding, off-road placement, or fragmented vehicle geometry as edit failure even if artifact files exist.',
    'inputs': {
        'rendered_video': 'Rendered video from the previous edit step.',
        'scene_requirement': 'Original requirement generated before editing.',
        'frame_count': 'Optional uniformly sampled frame count for Qwen-VL; defaults to 8.',
        'frame_paths': 'Optional locally extracted keyframe paths.',
        'edited_scene': 'Optional scene metadata with scene_config and scene_ply.',
        'visual_evaluation': 'Optional result from a local visual evaluator.',
    },
    'outputs': {
        'validation': 'Requirement and visual validation; without a local evaluator it fails explicitly.',
    },
}

from .schemas import Input, Output
DESCRIPTION["input_schema"] = Input.model_json_schema()
DESCRIPTION["output_schema"] = Output.model_json_schema()
DESCRIPTION["inputs"] = {key: DESCRIPTION["inputs"].get(key, value.get("description", value.get("title", key))) for key,value in DESCRIPTION["input_schema"]["properties"].items()}

DESCRIPTION["outputs"] = {key: DESCRIPTION["outputs"].get(key, value.get("description", value.get("title", key))) for key,value in DESCRIPTION["output_schema"]["properties"].items()}
