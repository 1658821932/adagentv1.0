"""
Tool description for the video_retriever tool (tool04_video_retriever).

This is the single source of truth shown to the LLM orchestrator when it
decides whether and how to call this tool. Keep orchestrator-facing rules
about this tool here, not in the global system prompt.
"""

DESCRIPTION = {
    "name": 'video_retriever',
    "description": 'Retrieve a real base driving video from the existing local Sentry Video Search index.',
    "when_to_use": 'Use when a suitable indexed real driving scene is needed as the starting point for reconstruction or editing. This tool does not search the internet.',
    'inputs': {
        'requirement': 'One natural-language description of the required base driving video.',
        'library': 'Optional: default or waymo. Use waymo to search the separate Waymo front-camera index; default searches the existing local index.',
    },
    'outputs': {
        'video_path': 'Absolute local path of the matched video clip.',
        'match_score': 'Local retrieval similarity score.',
        'mode': 'local_hit.',
    },
}
