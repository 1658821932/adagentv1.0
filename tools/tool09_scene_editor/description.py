"""
Tool description for the scene_editor tool (tool09_scene_editor).

This is the single source of truth shown to the LLM orchestrator when it
decides whether and how to call this tool. Keep orchestrator-facing rules
about this tool here, not in the global system prompt.
"""

DESCRIPTION = {
    "name": 'scene_editor',
    "description": 'Execute the DGGT Mode 2 edit_config against the source driving video.',
    "when_to_use": 'Use after edit_config_generator produces a JSON config.',
    'inputs': {
        'video_path': 'Source driving video.',
        'edit_config_path': 'Optional Smart Edit JSON; instance trajectory values add to each frame original instance position, not absolute target positions.',
        'camera_params_path': 'DGGT camera_params.json required for Smart Edit instance trajectory coordinates.',
        'meters_to_dggt': 'Explicit metric-to-DGGT scale. Historical instance-edit experiments suggest ~0.01 as a starting estimate (0.001 for smaller shifts); verify on this scene. Do not copy WorldEngine translation_scale=10000: that is a different conversion. DGGT config offsets are pre-normalization world units; PLY-derived corrections must multiply ply_world_transform.scale.',
        'coordinate_system': 'smart_edit or dggt_world.',
        'ego_trajectory_offsets_path': 'Optional JSON of [forward,left,up] offsets from each predicted ego pose in DGGT native units. These alter rendering camera poses only; they cannot force the WorldEngine closed-loop policy to brake.',
        'export_dynamic_frame_ply': 'Optional; defaults to true for edited scenes so WorldEngine receives per-frame dynamic PLY metadata.',
        'num_frames': 'Exactly 40; render only the first 40 video frames.',
    },
    'outputs': {
        'edited_scene': 'DGGT reconstruction result containing edited render, exact scene_config/edit_config_path, and scene artifacts.',
    },
}

from .schemas import Input, Output
DESCRIPTION["input_schema"] = Input.model_json_schema()
DESCRIPTION["output_schema"] = Output.model_json_schema()
DESCRIPTION["inputs"] = {key: DESCRIPTION["inputs"].get(key, value.get("description", value.get("title", key))) for key,value in DESCRIPTION["input_schema"]["properties"].items()}

DESCRIPTION["outputs"] = {key: DESCRIPTION["outputs"].get(key, value.get("description", value.get("title", key))) for key,value in DESCRIPTION["output_schema"]["properties"].items()}
