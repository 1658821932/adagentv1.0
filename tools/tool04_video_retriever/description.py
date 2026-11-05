"""
Tool description for the video_retriever tool (tool04_video_retriever).

This is the single source of truth shown to the LLM orchestrator when it
decides whether and how to call this tool. Keep orchestrator-facing rules
about this tool here, not in the global system prompt.
"""

DESCRIPTION = {
    "name": 'video_retriever',
    "description": 'Retrieve a dataset camera video and its bound multi-camera scene from the existing local Sentry index. Unbound videos raise an error.',
    "when_to_use": 'Use when a suitable indexed real driving scene is needed as the starting point for reconstruction or editing. This tool does not search the internet.',
    'inputs': {
        'requirement': 'One natural-language description of the required base driving video.',
        'library': 'Optional: default or waymo. Use waymo to search the separate Waymo front-camera index; default searches the existing local index.',
    },
    'outputs': {
        'video_path': 'Absolute local path of the matched video clip.',
        'match_score': 'Local retrieval similarity score.',
        'mode': 'Local retrieval mode.',
        'source_file': 'Original indexed camera video; video_path is the exported search clip.',
        'source_segment': 'Matched index segment in seconds relative to source_file; not necessarily the exported clip bounds.',
        'scene_ref': 'Required in real mode: complete scene identity, matched camera, all camera image availability, synchronized frame metadata, annotations and Gaussian asset paths. manifest_path stores ordered images/timestamps/calibrations. Real mode raises an error when scene binding is unavailable; mock may return null. Use this scene reference for dataset editing; camera videos and metadata describe the full source scene.',
    },
}

from .schemas import Input, Output
DESCRIPTION["input_schema"] = Input.model_json_schema()
DESCRIPTION["output_schema"] = Output.model_json_schema()
DESCRIPTION["inputs"] = {key: DESCRIPTION["inputs"].get(key, value.get("description", value.get("title", key))) for key,value in DESCRIPTION["input_schema"]["properties"].items()}

DESCRIPTION["outputs"] = {key: DESCRIPTION["outputs"].get(key, value.get("description", value.get("title", key))) for key,value in DESCRIPTION["output_schema"]["properties"].items()}
