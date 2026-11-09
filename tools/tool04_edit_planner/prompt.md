# Task
Review the actual retrieved scene against scene_requirement and the selected edit_strategy. Quickly decide whether this base is worth editing, and give a coarse attack-vehicle plan. Keep the global target_effect for the later ego editor.

# Evidence
Use attached source keyframes, camera/time labels, the annotated actor catalog and resource report. Review the whole bound scene; source_segment is a retrieval match, not a synchronized crop. Boxes use the stated source coordinate system. Judge visible layout and editing opportunity; leave precise trajectory feasibility to tool05.

# Visual review
Initially you receive up to 12 uniformly sampled CAM_F0 frames as three chronological 2x2 contact sheets, four frames per sheet. Read camera/frame/time labels; missing source frames are explicitly marked and blank cells are not observations.
If the target/edit strategy needs a side, rear or other view, return {"action":"request_views","cameras":["an actual available camera"],"purpose":"what needs checking"}. You may request up to two cameras per round and two additional rounds, as shown in visual_review_options. The host returns same-time contact sheets, then you continue reviewing. Do not assume unseen cameras were checked. When enough information is present, return the normal final suitability/reasons/edit_plan/retrieval_suggestion object. Use needs_information when required evidence remains unavailable.

# Decision
- ready: a suitable base and required resources are present. For edit_existing choose an ID from eligible_vehicle_tracks. State its behavior change, event order and preserved context.
- add_vehicle: use stationary or left_cut_in with target_track=null. The host attaches the supplied asset and dimensions. Without a usable vehicle asset, use needs_information; a coarse proposed plan may still be included.
- unsuitable: explain the layout/actor mismatch and provide a concrete retrieval_suggestion, with no edit plan.
- needs_information: name missing prerequisites. Missing downloads differ from an unsuitable scene.

# Scope
Prefer editing one other vehicle. Supported operations are edit_existing or add_vehicle(stationary/left_cut_in). Give a scene-specific coarse instruction, not control knots or a numeric trajectory. tool05 performs detailed attack search and tool06 solves the ego trajectory. The host constructs attack_input and preserves source identity, asset paths, dimensions and target_effect.
