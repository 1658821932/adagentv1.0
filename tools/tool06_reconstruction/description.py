"""
Tool description for the reconstruction tool (tool06).

This is the single source of truth shown to the LLM orchestrator when it
decides whether and how to call this tool. Keep orchestrator-facing rules
about DGGT reconstruction here, not in the global system prompt.
"""

DESCRIPTION = {
    "name": "reconstruction",
    "description": (
        "Reconstruct a single-view driving video into a DGGT Mode 2 Gaussian "
        "scene. This is a deterministic 3D-reconstruction backend: it uses no "
        "LLM internally, so every constraint must be passed as an argument."
    ),
    "when_to_use": (
        "Use after video_retriever when the task requires 3D scene "
        "reconstruction. Agent DGGT reconstruction processes only the first "
        "40 frames of the source video."
    ),
    "inputs": {
        "video_path": "Absolute local path of the single-view driving video.",
        "num_frames": (
            "Exactly 40; Agent reconstructs only the first 40 frames. Never "
            "request 64 frames."
        ),
        "export_render_state": (
            "Optional boolean, default false: export full native DGGT temporal "
            "Gaussians for faithful WorldEngine rendering; currently supports "
            "unedited reconstruction only, not scene_editor outputs."
        ),
    },
    "outputs": {
        "input_dir": (
            "Parent directory of preprocessed input frames. Pass this "
            "verbatim to instance_discovery.image_dir."
        ),
        "input_scene_name": (
            "Input scene name for instance_discovery.scene_name, normally 000; "
            "the output scene 001 is NOT the input scene name."
        ),
        "scene_ply": "Combined reconstructed Gaussian scene PLY.",
        "render_state": (
            "Native temporal Gaussian tensor file when export_render_state=true; "
            "must be converted to a DGGTScene WorldEngine asset, not treated as "
            "a PLY."
        ),
        "camera_params": "Camera parameters aligned with the exported scene PLY.",
        "rendered_video": "Rendered reconstruction video.",
        "output_dir": "DGGT reconstruction output directory.",
    },
}

from .schemas import Input, Output
DESCRIPTION["input_schema"] = Input.model_json_schema()
DESCRIPTION["output_schema"] = Output.model_json_schema()
DESCRIPTION["inputs"] = {key: DESCRIPTION["inputs"].get(key, value.get("description", value.get("title", key))) for key,value in DESCRIPTION["input_schema"]["properties"].items()}

DESCRIPTION["outputs"] = {key: DESCRIPTION["outputs"].get(key, value.get("description", value.get("title", key))) for key,value in DESCRIPTION["output_schema"]["properties"].items()}
