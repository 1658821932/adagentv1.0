"""
Tool description for the edit_config_generator tool (tool08_edit_config_generator).

This is the single source of truth shown to the LLM orchestrator when it
decides whether and how to call this tool. Keep orchestrator-facing rules
about this tool here, not in the global system prompt.
"""

DESCRIPTION = {
    "name": 'edit_config_generator',
    "description": 'Use Smart Edit MCP to convert an edit plan into a DGGT-compatible JSON configuration.',
    "when_to_use": 'Use after edit_plan_builder and before scene_editor. Set output_kind=ego_trajectory_offsets for natural-language ego render trajectory offsets; otherwise use rigid_config for instance edits.',
    'inputs': {
        'edit_plan': 'Detailed editing plan. Existing instance trajectory waypoints are per-frame offsets added to original positions; compensate the source vehicle existing displacement. For cut-in, hold ego-lane position from frame 12 through 39 and use zero height offset. Do not request a new object unless asset_path and asset_scale are supplied.',
        'instance_info_path': 'DGGT per-frame instance_info.json path for rigid instance edits.',
        'output_path': 'Required .json configuration path.',
        'output_kind': 'rigid_config for an existing traffic instance, or ego_trajectory_offsets for ego render-camera motion (not closed-loop policy braking).',
        'frame_count': 'For ego offsets, supply 40; Smart Edit must return one vector or exactly 40 vectors.',
        'scene_info': 'Optional scene context.',
        'asset_path': 'Required exact vehicle PLY path when adding or replacing an asset.',
        'asset_scale': 'Required explicit positive asset scale for add or replace; no default is assumed.',
    },
    'outputs': {
        'saved_path': 'Saved Smart Edit configuration path.',
        'tool_calls': 'Smart Edit MCP tools called by the client.',
    },
}

from .schemas import Input, Output
DESCRIPTION["input_schema"] = Input.model_json_schema()
DESCRIPTION["output_schema"] = Output.model_json_schema()
DESCRIPTION["inputs"] = {key: DESCRIPTION["inputs"].get(key, value.get("description", value.get("title", key))) for key,value in DESCRIPTION["input_schema"]["properties"].items()}

DESCRIPTION["outputs"] = {key: DESCRIPTION["outputs"].get(key, value.get("description", value.get("title", key))) for key,value in DESCRIPTION["output_schema"]["properties"].items()}
