"""
Tool description for the video_retriever tool (tool04_video_retriever).

This is the single source of truth shown to the LLM orchestrator when it
decides whether and how to call this tool. Keep orchestrator-facing rules
about this tool here, not in the global system prompt.
"""

DESCRIPTION = {
    "name": 'video_retriever',
    "description": 'Retrieve a real base driving video from a local index. The nuplan library uses Kimi scene descriptions, local text recall and Kimi visual reranking through the configured gateway.',
    "when_to_use": 'Use when a suitable indexed real driving scene is needed as the starting point for reconstruction or editing. This tool does not search the internet.',
    'inputs': {
        'requirement': 'One natural-language description of the required base driving video.',
        'library': 'Optional: default, waymo, or nuplan. Defaults to the configured nuplan library. Use nuplan for original navtrain videos with calibrated annotations for ADSceneGraph reconstruction; explicit default/waymo retain their existing libraries.',
    },
    'outputs': {
        'video_path': 'Absolute local path of the matched video clip.',
        'match_score': 'Local retrieval similarity score.',
        'mode': 'local_hit.',
        'scene_manifest_path': 'For nuplan hits, the frame-aligned scene manifest for this exact exported video. Pass it with video_path to reconstruction using backend=adscenegraph.',
        'scene_id': 'Original official nuPlan navtrain scene identity when library=nuplan.',
    },
}
