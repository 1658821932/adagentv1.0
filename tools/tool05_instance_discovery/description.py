"""
Tool description for the instance_discovery tool (tool05_instance_discovery).

This is the single source of truth shown to the LLM orchestrator when it
decides whether and how to call this tool. Keep orchestrator-facing rules
about this tool here, not in the global system prompt.
"""

DESCRIPTION = {
    "name": 'instance_discovery',
    "description": 'Discover important traffic participants from RGB videos using the DGGT YOLO backend.',
    "when_to_use": 'Use after reconstruction to detect instances in its preprocessed input frames.',
    'inputs': {
        'backend': 'Must be dggt_yolo.',
        'image_dir': 'Use reconstruction.result.input_dir, the parent of the input scene folder; do not use reconstruction.result.output_dir or append the scene name.',
        'output_path': 'DGGT instance_info.json output path.',
        'scene_name': 'Use reconstruction.result.input_scene_name (normally 000); the output scene 001 is not the input scene name.',
        'sequence_length': 'Exactly 40, matching reconstruction num_frames; never 64.',
    },
    'outputs': {
        'instances': 'Object IDs, categories, boxes and masks. Note: YOLO instance_info provides 2D boxes only, not a calibrated 3D vehicle track or road height; do not claim exact metres-to-ego placement or road contact from it.',
        'instance_info_path': 'DGGT-compatible per-frame instance_info.json for dggt_yolo. The returned state contains only sampled bbox frames; pass this path to Smart Edit tools for complete per-frame data instead of copying every detection frame into arguments.',
    },
}

from .schemas import Input, Output
DESCRIPTION["input_schema"] = Input.model_json_schema()
DESCRIPTION["output_schema"] = Output.model_json_schema()
DESCRIPTION["inputs"] = {key: DESCRIPTION["inputs"].get(key, value.get("description", value.get("title", key))) for key,value in DESCRIPTION["input_schema"]["properties"].items()}

DESCRIPTION["outputs"] = {key: DESCRIPTION["outputs"].get(key, value.get("description", value.get("title", key))) for key,value in DESCRIPTION["output_schema"]["properties"].items()}
