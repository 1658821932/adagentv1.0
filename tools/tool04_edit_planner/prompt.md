# Task
Review the retrieved scene against scene_requirement and edit_strategy. Decide if this base is worth editing, and give a coarse attack-vehicle plan if needed. Keep target_effect unchanged for tool06.

# Visual evidence
You receive up to 12 CAM_F0 keyframes as three chronological 2x2 contact sheets. Read camera/frame/time labels; blank cells are missing, not observations.
If you need another view, return {"action":"request_views","cameras":["CAM_L0"],"purpose":"what to check"}. Up to 2 cameras per round, 2 extra rounds.
When you have enough evidence, return the final decision object.

# Decision
- ready: base scene is suitable. If edit_strategy is null and the scene already matches the failure mechanism, edit_plan can be null. If editing is needed, provide edit_plan with operation (edit_existing or add_vehicle), target_track (for edit_existing), maneuver (for add_vehicle), and a coarse instruction.
- unsuitable: layout or actor mismatch. Provide retrieval_suggestion, no edit_plan.
- needs_information: missing resources (e.g. no usable vehicle asset). Name what is needed.

# Rules
- edit_strategy from tool02 is a preliminary suggestion; you may adjust it based on what you actually see in the video.
- Any edit_plan you provide must help construct a scene similar to the failure mechanism.
- Prefer editing ONE other vehicle. Operations: edit_existing (choose from eligible_vehicle_tracks) or add_vehicle (stationary or left_cut_in).
- Give a coarse scene-specific instruction, not numeric trajectories. tool05 does the detailed search.
- Do NOT estimate asset dimensions from images; use the supplied vehicle asset data.
