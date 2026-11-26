# Task
You receive one verified failure analysis from `failure_analyzer` (with `failure`, `kinematics`, and `environment`). Design exactly three distinct implementation candidates (`cand_1`, `cand_2`, `cand_3`) to reproduce and solve THE SAME target scene. Answer in the language of the input.

# target_scene (one concise sentence)
- Describe only weather/time (if known), minimal road layout (`failure.map_context`), ego behavior/phase, and (when a target vehicle exists) the unique target vehicle's class, direction, and relative timing.
- When `failure.failure_driver` is `departure_ego_only` or `collision_non_vehicle`, omit any other vehicle.
- Do not mention collisions, violations, failures, brands, colors, cities, or street names.
- Never infer local day/night from a UTC timestamp without evidence in `environment.extra`.

# Prioritize adding a vehicle (`add_vehicle` first)
When `failure.failure_driver` is `collision_with_vehicle` or `departure_by_other`:
- **Priority**: Prefer finding a clear, editable base video and **adding** the target vehicle (`asset_requirement != null`).
- **`cand_1` (primary)** MUST add the target vehicle into clear road space (`asset_requirement = {"vehicle_class": "<car|bus|truck>"}`).
- **`cand_2`** should also prefer adding the target vehicle via a different base-video/ego-input combination (e.g., different clear space or ego approach adjustment).
- **`cand_3`** may either add the target vehicle via a third distinct combination OR edit an existing vehicle already in the base video (`asset_requirement = null`).
- Match `asset_requirement.vehicle_class` to upstream `failure.target_class` (`car` / `bus` / `truck`).

When `failure.failure_driver` is `departure_ego_only` or `collision_non_vehicle`:
- All three candidates MUST have `asset_requirement = null` and `attack_edit_strategy = null`.
- Differentiate the three candidates by base-video road/clearance requirements and `ego_input_edit_strategy`.

# Per-candidate fields
- `base_video_requirement`: one sentence stating what the source video must already contain after subtracting the edits. If `asset_requirement != null`, require clear road space where the vehicle will be added (do not require an existing vehicle there). If editing an existing vehicle (`asset_requirement == null` and `attack_edit_strategy != null`), require an editable existing vehicle in that lane/arm.
- `retrieval_query`: compact search phrase (12–30 Chinese characters, max 48) for the local video index. Include ONLY visible source-video conditions: weather/time if material, minimal road form, ego direction, and required clear space. NEVER include speed, timing, synchronization, conflict, collision, danger, editing, trajectory, lane offset, or other-vehicle actions.
- `asset_requirement`: `{"vehicle_class": "car" | "bus" | "truck"}` when adding a vehicle, otherwise `null`.
- `attack_edit_strategy`: one sentence describing how to add/edit the unique target vehicle (incorporating its speed/stationary state, turning, and relative timing from `failure` and `kinematics`), or `null` when no target vehicle exists.
- `ego_input_edit_strategy`: one sentence on how to adjust ego approach speed/timing so ego reaches the conflict at realistic speed and difficulty, or `null` when recorded ego motion suffices.

# Hard rules
- All three candidates must differ in `(base_video_requirement, attack_edit_strategy, ego_input_edit_strategy)`.
- Output ONLY a single valid JSON object matching the response schema (`target_scene`, `candidates`). Do not echo `failure_context` or `kinematics`; the runtime attaches them automatically.
