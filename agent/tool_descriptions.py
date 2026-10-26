"""
Tool descriptions for the LLM Orchestrator.

This file defines the available tools, their purposes, inputs and outputs.
The descriptions will later be provided to the LLM so it can decide which
mock/MCP tool should be called.
"""

TOOLS = [
    {
        "name": "failure_analyzer",
        "description": "Analyze autonomous driving failure cases and summarize failure type, failure reason, key scene factors and risk factors.",
        "when_to_use": "Use when the agent receives failure cases and needs to understand why the driving model failed.",
        "inputs": {"input_path": "JSON file containing failure_cases with image paths and failure metadata; alternatively pass failure_cases directly."},
        "outputs": {"failure_analysis": "One dominant cross-case failure_type, failure_reason, key_scene_factors and risk_factors, with supporting and uncovered case IDs."},
    },
    {
        "name": "scene_requirement_generator",
        "description": "Generate targeted scenario requirements from failure analysis.",
        "when_to_use": "Use after failure analysis to define what new scenario should be generated.",
        "inputs": {"failure_analysis": "Failure analysis result."},
        "outputs": {"scene_requirement": "Scenario conditions, actors, risks and target effects."},
    },
    {
        "name": "scene_conceiver",
        "description": "Design a high-level scenario generation plan, including required base video and simple editing ideas.",
        "when_to_use": "Use after obtaining scenario requirements and before retrieving data.",
        "inputs": {"scene_requirement": "Target scenario requirements."},
        "outputs": {"scene_plan": "Base video requirements, edit ideas and expected effects."},
    },
    {
        "name": "video_retriever",
        "description": "Retrieve a real base driving video from the existing local Sentry Video Search index.",
        "when_to_use": "Use when a suitable indexed real driving scene is needed as the starting point for reconstruction or editing. This Tool does not search the internet.",
        "inputs": {"requirement": "One natural-language description of the required base driving video.", "library": "Optional: default or waymo. Use waymo to search the separate Waymo front-camera index; default searches the existing index."},
        "outputs": {"video_path": "Absolute local path of the matched video clip.", "match_score": "Local retrieval similarity score.", "mode": "local_hit."},
    },
    {
        "name": "instance_discovery",
        "description": "Discover important traffic participants from RGB videos.",
        "when_to_use": "Use dggt_yolo after reconstruction to detect instances in its preprocessed input frames.",
        "inputs": {"backend": "Must be dggt_yolo.", "image_dir": "Use reconstruction.result.input_dir, the parent of the input scene folder; do not use reconstruction.result.output_dir or append the scene name.", "output_path": "DGGT instance_info.json output path.", "scene_name": "Use reconstruction.result.input_scene_name (normally 000); output scene 001 is not the input scene name.", "sequence_length": "Exactly 40, matching reconstruction num_frames."},
        "outputs": {"instances": "Object IDs, categories, boxes and masks.", "instance_info_path": "DGGT-compatible per-frame instance_info.json for dggt_yolo."},
    },
    {
        "name": "reconstruction",
        "description": "Reconstruct a single-view driving video into a DGGT Mode 2 Gaussian scene.",
        "when_to_use": "Use after video_retriever when the task requires 3D scene reconstruction.",
        "inputs": {"video_path": "Absolute local path of the single-view driving video.", "num_frames": "Exactly 40; Agent reconstructs only the first 40 frames.", "export_render_state": "Optional boolean, default false: export full native DGGT temporal Gaussians for faithful WorldEngine rendering; currently supports unedited reconstruction only, not scene_editor outputs."},
        "outputs": {
            "input_dir": "Parent directory of preprocessed input frames for instance_discovery.image_dir.",
            "input_scene_name": "Input scene name for instance_discovery.scene_name, normally 000; distinct from output scene 001.",
            "scene_ply": "Combined reconstructed Gaussian scene PLY.",
            "render_state": "Native temporal Gaussian tensor file when export_render_state=true; must be converted to a DGGTScene WorldEngine asset, not treated as a PLY.",
            "camera_params": "Camera parameters aligned with the exported scene PLY.",
            "rendered_video": "Rendered reconstruction video.",
            "output_dir": "DGGT reconstruction output directory.",
        },
    },
    {
        "name": "edit_plan_builder",
        "description": "Use the LLM to turn scene requirements and instances into a precise natural-language editing plan.",
        "when_to_use": "Use after reconstruction and instance discovery when an edit must be specified.",
        "inputs": {"edit_request": "High-level natural-language edit request; for front-car cut-in target ego-lane entry near frame 12 and lane holding through frame 39, with zero world-height translation. Trajectory offsets must compensate the original instance's per-frame motion.", "scene_info": "Scene and instance context.", "instances": "Discovered instances."},
        "outputs": {"edit_plan": "Detailed natural-language editing plan and operations."},
    },
    {
        "name": "edit_config_generator",
        "description": "Use Smart Edit MCP to convert an edit plan into a DGGT-compatible JSON configuration.",
        "when_to_use": "Use after edit_plan_builder and before scene_editor. Set output_kind=ego_trajectory_offsets for natural-language ego render trajectory offsets; otherwise use rigid_config for instance edits.",
        "inputs": {"edit_plan": "Detailed editing plan. Existing instance trajectory waypoints are per-frame offsets added to original positions; compensate the source vehicle's existing displacement. For cut-in, hold ego-lane position from frame 12 through 39 and use zero height offset. Do not request a new object unless asset_path and asset_scale are supplied.", "instance_info_path": "DGGT per-frame instance_info.json path for rigid instance edits.", "output_path": "Required .json configuration path.", "output_kind": "rigid_config for an existing traffic instance, or ego_trajectory_offsets for ego render-camera motion (not closed-loop policy braking).", "frame_count": "For ego offsets, supply 40; Smart Edit must return one vector or exactly 40 vectors.", "scene_info": "Optional scene context.", "asset_path": "Required exact vehicle PLY path when adding or replacing an asset.", "asset_scale": "Required explicit positive asset scale for add or replace; no default is assumed."},
        "outputs": {"saved_path": "Saved Smart Edit configuration path.", "tool_calls": "Smart Edit MCP tools called by the client."},
    },
    {
        "name": "scene_editor",
        "description": "Execute the DGGT Mode 2 edit_config against the source driving video.",
        "when_to_use": "Use after edit_config_generator produces a JSON config.",
        "inputs": {"video_path": "Source driving video.", "edit_config_path": "Optional Smart Edit JSON; instance trajectory values add to each frame's original instance position, not absolute target positions.", "camera_params_path": "DGGT camera_params.json required for Smart Edit instance trajectory coordinates.", "meters_to_dggt": "Explicit metric-to-DGGT scale. Historical instance-edit experiments suggest ~0.01 as a starting estimate (0.001 for smaller shifts); verify on this scene. Do not copy WorldEngine translation_scale=10000. DGGT config offsets are pre-normalization world units; PLY-derived corrections must multiply ply_world_transform.scale.", "coordinate_system": "smart_edit or dggt_world.", "ego_trajectory_offsets_path": "Optional JSON of [forward,left,up] offsets from each predicted ego pose in DGGT native units.", "export_dynamic_frame_ply": "Optional; defaults to true for edited scenes so WorldEngine receives per-frame dynamic PLY metadata.", "num_frames": "Exactly 40; render only the first 40 video frames."},
        "outputs": {"edited_scene": "DGGT reconstruction result containing edited render, exact scene_config/edit_config_path, and scene artifacts."},
    },
    {
        "name": "scene_validator",
        "description": "Evaluate whether an edited scene satisfies requirements and remains physically and logically reasonable.",
        "when_to_use": "Use after editing before simulation.",
        "inputs": {"rendered_video": "Rendered video from the previous edit step.", "scene_requirement": "Original requirement generated before editing.", "frame_count": "Optional uniformly sampled frame count for Qwen-VL; defaults to 8.", "frame_paths": "Optional locally extracted keyframe paths.", "edited_scene": "Optional scene metadata with scene_config and scene_ply.", "visual_evaluation": "Optional result from a local visual evaluator."},
        "outputs": {"validation": "Requirement and visual validation; without a local evaluator it fails explicitly."},
    },
    {
        "name": "recovery_decider",
        "description": "Decide recovery actions when validation or execution fails.",
        "when_to_use": "Use when a previous step fails and rollback or replanning is needed.",
        "inputs": {"evaluation": "Failure information and execution history."},
        "outputs": {"recovery_action": "Retry, rollback or replan decision."},
    },
    {
        "name": "policy_runner",
        "description": "Load a WorldEngine scenario and run the autonomous driving policy in one closed-loop execution for imitation-learning data generation.",
        "when_to_use": "Use after scene validation succeeds to load the scenario, start SimEngine and run the policy.",
        "inputs": {"config_path": "Absolute existing AlgEngine config. Standard: <WORLDENGINE_ROOT>/projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct.py.", "checkpoint_path": "Absolute existing policy checkpoint. Standard: <WORLDENGINE_ROOT>/data/alg_engine/ckpts/e2e_vadv2_50pct_ep8.pth.", "model_name": "Experiment model name.", "data_type": "WorldEngine scenario data type.", "react_type": "R or NR.", "asset_name": "Optional renderer asset name.", "added_name": "Optional output suffix.", "num_scenarios": "Optional positive scenario count.", "scenario_data_folder_path": "Optional scenario directory; supports custom/DGGT-converted scenarios.", "asset_folder_path": "Optional scenario asset directory.", "scenario_pkl_file_name": "Optional scenario pickle filename.", "dggt_output_dir": "Optional DGGT edited output directory; policy_runner automatically stages the selected instance and converts it to WorldEngine data.", "instance_id": "DGGT instance to stage when dggt_output_dir is provided; defaults to 0.", "translation_scale": "WorldEngine metres per DGGT reconstruction unit; defaults to 10000 for normalized DGGT outputs.", "num_history": "Optional history frame count.", "num_future": "Optional future frame count.", "max_frame_per_scene": "Exactly 40; closed-loop simulation uses only the first 40 frames.", "sampling_poses": "Optional native rollout timing override in 0.5-second units."},
        "outputs": {"results": "WorldEngine closed-loop result, output directory and runner reports."},
    },
    {
        "name": "failure_case_filter",
        "description": "Filter new failure cases from simulation results and store useful training data.",
        "when_to_use": "Use after policy execution.",
        "inputs": {"scenario_pkl": "WorldEngine custom scenario pickle containing ego and dynamic-object tracks.", "result_dir": "Optional policy_runner result directory containing the actual WorldEngine ego trajectory."},
        "outputs": {"is_failure_case": "Whether a dynamic-vehicle collision failure was found.", "collision": "Dynamic-vehicle collision result.", "failure_cases": "Decision failures found by geometric collision evaluation."},
    },
    {
        "name": "policy_trainer",
        "description": "Fine-tune or retrain the autonomous driving decision model using generated data.",
        "when_to_use": "Use after collecting valid training samples.",
        "inputs": {"training_mode": "Currently ilft only.", "training_data": "Object containing ann_file (worldengine_infos.pkl) and token_file (worldengine_tokens.yaml).", "base_checkpoint": "Existing WorldEngine policy checkpoint.", "config_path": "AlgEngine training config.", "epochs": "Positive training epoch count."},
        "outputs": {"checkpoint": "Real checkpoint generated by WorldEngine training.", "log_path": "Training log path.", "final_loss": "Final imitation loss when present."},
    },
]


def get_tool_descriptions():
    return TOOLS
