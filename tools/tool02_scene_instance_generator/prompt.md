# Task
You receive one verified failure analysis from failure_analyzer. Design exactly three ways to build and solve THE SAME target scene. Answer in the language of the input.

# target_scene: one fixed sentence
Cover only weather and time of day, road layout, the ego direction and lane position, and where the other vehicle comes from, which way it drives and its lane position.
- When `attack.other_origin` is `none`, omit the other vehicle from target_scene and do not add one to any base-video requirement.
- Take weather and time from environment. If the weather is unknown, say unknown or leave it out.
- Never turn a UTC timestamp into a local day or night claim without evidence.
- Do not mention collisions, danger, violations, failures or failure causes.
- Example: "多云白天，丁字路口，自车右转并保持在本车道，右侧支路一辆车左转并保持在本车道".

# attack: copy the measurement
Repeat the input attack unchanged: map_context, ego_action, ego_lane_keeping, other_origin, other_action, other_lane_keeping. All three candidates share it; never invent another road, direction or lane offset. When other_origin is none there is no target vehicle: other_action and other_lane_keeping are none, attack_edit_strategy must be null, and ego_input_edit_strategy still follows what building the dangerous scene needs.

Actions describe direction only: straight, turn_left, turn_right. A lane change or lane departure is represented by lane_keeping=left_offset/right_offset with action=straight. Do not infer speed or braking as an action.

# Three implementation paths
For each candidate
- attack_edit_strategy: how to edit the other vehicle so the scene becomes dangerous, or null when the source already provides it.
- ego_input_edit_strategy: how to adjust the ego approach speed and timing so the ego reaches the conflict with the edited attack vehicle at the same time. Preserve a realistic driving speed and meaningful difficulty; do not stop early, crawl, or make the conflict trivial. Null only when the recorded ego input already produces that synchronized conflict.
These two edits together build the dangerous scene: the other vehicle's timing and the ego timing must meet at the conflict.
- ego_correct_trajectory_strategy: how the ego drives to handle that danger safely. It must not remove the task by stopping unconditionally or avoiding the drive.

# base_video_requirement: what the source video must already contain
Subtract whatever the two edits will create, then write one sentence with the remaining conditions: road layout, weather, and the vehicle or clear space the edits rely on. Prefer a clear editable area where a vehicle can be added. If a vehicle is edited, the video must contain an editable vehicle; if a vehicle is added, the video must not be required to have it.

# retrieval_query: compact query for video search
Write a separate, compressed query for the local video index, not a copy of target_scene.
- Keep only visible source-video conditions: weather/time if material, road form, ego direction, and the required road arm or space being clear.
- Prefer constructing the conflict by adding a vehicle into clear space. Describe the clear space rather than the future attack vehicle's action.
- Target 12-30 Chinese characters and never exceed 48.
- Omit other-vehicle direction, lane offset, speed, timing, synchronization, conflict, editing and trajectory.
- Example for a right-arm failure: `白天丁字路口，自车右转，右侧支路清晰`.
- Example for a straight-road failure: `白天城市直路，自车直行，同车道前方清晰`.
- When `attack.other_origin` is `none`, omit any other-vehicle or insertion-space clause.

# Rules
- The three candidates share target_scene and attack and differ only in implementation path. Do not design three different roads, and do not fake variety by rewording the correct-behaviour sentence.
- Vehicles are generic: no brand, colour, city, street or landmark. Keep the background sparse, with no unrelated participants.
- One short sentence per field; no parameters, timestamps or trajectory points.
- `retrieval_query` is a short phrase, not a sentence, and the schema enforces its size and banned downstream-editing terms.
- No editing needed means null; keep ego_correct_trajectory_strategy for post-conflict safe behavior, not for compensating for a mistimed ego_input_edit_strategy.
- The ego input edit must preserve a meaningful approach speed and the intended task difficulty.
