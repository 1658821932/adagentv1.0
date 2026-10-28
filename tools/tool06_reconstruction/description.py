"""
Tool description for the reconstruction tool (tool06).

This is the single source of truth shown to the LLM orchestrator when it
decides whether and how to call this tool. Keep orchestrator-facing rules
about DGGT reconstruction here, not in the global system prompt.
"""

DESCRIPTION = {
    "name": "reconstruction",
    "description": (
        "Reconstruct a driving scene with ADSceneGraph when nuPlan annotations "
        "are available, or DGGT Mode 2 for video-only input. These backends "
        "perform reconstruction without an internal LLM; pass constraints as arguments."
    ),
    "when_to_use": (
        "Use after video_retriever for scene reconstruction. For library=nuplan, "
        "pass its exact video_path and scene_manifest_path with backend=adscenegraph; "
        "all annotated frames are used. DGGT video-only reconstruction uses the first 40 frames."
    ),
    "inputs": {
        "backend": "Optional dggt or adscenegraph. For nuplan retrieval use adscenegraph and pass the matching scene_manifest_path.",
        "scene_manifest_path": "For ADSceneGraph: frame-aligned manifest returned by video_retriever library=nuplan, including original images, camera calibration, poses, tracks and LiDAR.",
        "iterations": "For ADSceneGraph: optional positive reconstruction optimization steps, default 3000. No Nexus inference is required.",
        "cameras": "For ADSceneGraph: optional camera ID list, default [0]; manifest may supply all 8 cameras.",
        "video_path": "Absolute local path of the single-view driving video.",
        "num_frames": (
            "DGGT only: exactly 40. ADSceneGraph uses all annotated clip frames; omit this parameter or match the manifest's frame_count."
        ),
        "export_render_state": (
            "Optional boolean, default false: export full native DGGT temporal "
            "Gaussians for faithful WorldEngine rendering; currently supports "
            "unedited reconstruction only, not scene_editor outputs."
        ),
    },
    "outputs": {
        "checkpoint_path": "For ADSceneGraph: native reconstructed scene checkpoint; this is not a DGGT PLY and cannot be passed to DGGT-specific editing tools.",
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
