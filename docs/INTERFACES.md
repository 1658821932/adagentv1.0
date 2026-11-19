# 工具接口参考

> 本文件由 `scripts/generate_interface_docs.py` 自动生成，请勿手工编辑。
> 修改工具契约（`tools/toolXX_*/schemas.py` 或 `description.py`）后重新生成：
> 
>     environments/video_retriever/.venv/bin/python scripts/generate_interface_docs.py

每个工具的用途和调用时机见 [当前工具说明](TOOLS.md)；
调用方式、mock/real 边界和错误类型见 [工具接口与运行管理](EXECUTION_CONTRACTS.md)。
本文档只列字段级契约。

## tool01_failure_analyzer: `failure_analyzer`

Analyze one WorldEngine scene from official metrics, images and numeric evidence.

**何时使用**：Diagnose one closed-loop scene from its archived official metrics and recorded images.

**上下游衔接**
- input: State BOTH the simulation record and the scene token: token alone does not say which closed-loop run produced the official metrics.
- output: Pass result.failure_analysis to scene_instance_generator.failure_analysis; result.simulation records which run produced it.

**工具规则**
- This tool analyzes one scene; it does not summarize a free-form list of failure cases.
- A scene can be simulated more than once. simulation selects the closed-loop run ("<experiment>/<batch>", a unique batch name, or an archive directory); token selects the scene inside it. Metrics and diagnosis differ between runs. Map user wording: 仿真号/simulation ID -> simulation, 场景号/scene ID -> token.
- The token is a WorldEngine closed-loop scene identity obtained from policy_runner -> failure_case_filter or supplied by the user. Short and full identifiers both work. Do not pass a retrieval scene_ref token.
- The token must belong to the chosen simulation archive; otherwise the call fails with the archive scene count.
- One scene takes tens of seconds to a few minutes and every call runs fresh, with no cached reuse. Do not duplicate the call while it is running.
- If the analysis reports missing archived data or an unusable archive, fix the input or choose another evidenced token; never substitute a guessed failure type.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `token` | string | ✅ | — | WorldEngine scene token to analyze (short or full identifier). |
| `simulation` | string \| null | — | null | Which closed-loop simulation record produced the official metrics: '<experiment>/<batch>', a unique batch name, or an absolute archive directory containing scene_index.csv. A scene may be simulated more than once with different policies, so this selects the run while token selects the scene. |
| `timeout` | integer >0 | — | `300` | Analysis process timeout in seconds. |

### 输出

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `failure_analysis` | object | ✅ | — | Compact single-scene diagnosis: failure_type, failure_reason, key_scene_factors (up to 5), official_metrics and uncertainty. |
| `token` | string | — | `""` | Full WorldEngine scene token that was analyzed. |
| `simulation` | string \| null | — | null | Closed-loop simulation record that produced the official metrics. |
| `evidence_dir` | string | — | `""` | Directory holding the evidence used for the analysis (images and numeric tables), for optional inspection. |


## tool02_scene_instance_generator: `scene_instance_generator`

Generate one target scene, one global ego goal and three source-video/edit-strategy candidates.

**何时使用**：Construct a new scenario from saved failure analysis.

**上下游衔接**
- input: failure_analysis: saved category or successful failure_analyzer result.
- output: Choose candidates[i]; pass base_video_requirement verbatim to video_retriever.requirement and retain its paired edit_strategy. target_effect applies to the whole target scene.

**工具规则**
- Candidate requirements describe source content to retrieve, not the final edited hazard.
- Planning supports adding, deleting or modifying vehicles and their trajectories, plus ego trajectory edits. Prefer changing at most one other vehicle; execution still follows the chosen editor contract.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `failure_analysis` | object | ✅ | — | Saved failure category or tool01 result with nonempty failure_reason or wrong_behavior. Preserve supplied case IDs. |
| `variation_hints` | object | — | — | Optional qualitative variation directions, e.g. traffic_density: sparse, target_actor: crossing bus. No distance/speed/timing values are required. |
| `existing_instances` | array<string> | — | — | Previous requirement summaries or successful tool02 outputs. Exact normalized descriptions are rejected; semantic diversity remains a model suggestion. |

### 输出

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `scene_requirement` | string | ✅ | — | Target scene = scene context + ego's WRONG behavior (the failure to reproduce). One short sentence. |
| `target_effect` | string | ✅ | — | Ego's CORRECT behavior that would have avoided the failure. One short sentence. |
| `candidates` | array<Candidate> | ✅ | — | Three variants of the same failure mechanism in slightly different road contexts. |

**`Candidate` 对象字段：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `base_video_requirement` | string | ✅ | — | Source-video scene description for retrieval. Describe ONLY observable road layout and other vehicles present. Do NOT include ego behavior or editing instructions. |
| `edit_strategy` | string \| null | — | null | Editing to apply to OTHER vehicles only (add/modify/delete a vehicle, change its trajectory). Set to null if no other-vehicle editing is needed. Do NOT edit ego's trajectory here. |


## tool03_video_retriever: `video_retriever`

Retrieve a local camera video and its bound annotated multi-camera scene.

**何时使用**：Retrieve a source scene for editing or downstream use.

**上下游衔接**
- input: Use the selected scene_instance_generator candidate.base_video_requirement as requirement.
- output: Pass the successful history entry.result_ref, envelope or result to edit_planner.retrieval_result. source_file is the indexed original; video_path is the exported search clip.

**工具规则**
- For scenario editing, pass the complete retrieval result or result_ref to edit_planner.retrieval_result before expensive attack search.
- No internet search. Real mode requires a valid scene binding.
- scene_ref describes the full source scene. source_segment is the matched index range, not a synchronized multi-camera crop.
- Check returned asset/image availability before requesting an operation that needs those files.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `requirement` | string | ✅ | — | Natural-language source-video search description. |
| `library` | string | — | `"default"` | Local index selection; only the default library is available. |
| `timeout` | integer >0 | — | `1800` | Retrieval process timeout in seconds. |
| `mock_video_path` | string \| null | — | null | Mock-only video path; it does not provide a real indexed scene. |

### 输出

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `video_path` | string | ✅ | — | Exported search clip path. |
| `match_score` | number | ✅ | — | Retrieval similarity score, not scenario-validation confidence. |
| `mode` | string | ✅ | — | Retrieval execution mode. |
| `source_file` | string \| null | — | null | Original indexed camera video. |
| `source_segment` | object \| null | — | null | Index-match start/end in source-video seconds; not exported clip bounds. |
| `scene_ref` | SceneReference \| null | — | null | Full source-scene identity, camera/image availability and annotation/Gaussian paths. Required by real retrieval; mock may return null. manifest_path locates ordered images, timestamps and calibrations. |

**`SceneReference` 对象字段：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `scene_id` | string | ✅ | — |  |
| `scene_token` | string | ✅ | — |  |
| `sample_root` | string | ✅ | — |  |
| `dataset` | string | ✅ | — |  |
| `matched_camera` | string | ✅ | — |  |
| `frame_count` | integer | ✅ | — |  |
| `fps` | number \| null | — | null |  |
| `manifest_path` | string | ✅ | — |  |
| `frames_path` | string | ✅ | — |  |
| `cameras_path` | string | ✅ | — |  |
| `scenario_pkl` | string \| null | — | null |  |
| `asset_config_path` | string \| null | — | null |  |
| `asset_path` | string \| null | — | null |  |
| `runtime_asset_path` | string \| null | — | null |  |
| `maps_path` | string \| null | — | null |  |
| `cameras` | object | ✅ | — |  |
| `images_complete` | boolean | ✅ | — |  |


## tool04_edit_planner: `edit_planner`

Review a retrieved base scene and produce 1..3 unranked edit alternatives plus an auto-selection attack-editor handoff.

**何时使用**：Between tool03 retrieval and tool05 attack editing.

**上下游衔接**
- input: retrieval_result accepts the successful tool03 envelope, result or history result_ref; also supply tool02 scene_requirement, target_effect and the selected edit_strategy. Optional vehicle supplies a real added-car asset and dimensions.
- output: ready: edit_options are suggestions for the attack SubAgent; pass result.attack_input (operation=auto, plan_options, optional asset) unchanged as attack_vehicle_editor arguments. Keep target_effect for tool06. unsuitable: retrieve again using retrieval_suggestion. needs_information: obtain the named resources, then review again.

**工具规则**
- Initial observation includes 12 CAM_F0 frames in three chronological 2x2 sheets and, when source transforms are available, an original-scene BEV with the recorded ego route. Internally the VLM may query a catalog vehicle or ego trajectory at most three times (24 sampled observations each), or request other cameras for two rounds of up to two cameras.
- This quick review does not run trajectory optimization, PDM, policy simulation or photorealistic scene rendering. Original future motion does not veto planning. Recorded proximity is advisory; detailed motion checks belong to tool05.
- Existing targets must be valid vehicles observed throughout the scene. Added-car assets and dimensions come from supplied input, not model estimates.
- edit_existing starts from the recorded t=0 pose by default; tool05 may create a bounded initial-position/yaw-adjusted candidate version with full rechecking. add_vehicle may choose a supported new initial pose and stationary behavior. Presence of a real actor alone does not require choosing it over an added vehicle.
- Non-ready outputs have attack_input=null; a coarse plan alone is not an executable attack request.
- Canonical added-vehicle asset: /inspire/hdd3/project/intelligent-driving-agent/public/workspace/ghc/adagent/data/assets/vehicle_01.ply with size_m [4.432, 2.033, 1.637] (length, width, height; see data/assets/manifest.json). When the caller supplies it, add_vehicle is available. Describe a proposed mapped turn or other movement in the option instruction.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `retrieval_result` | RetrievalContext | ✅ | — | Tool03 successful envelope, result object or absolute result_ref. Standalone scene_ref inputs are not accepted. |
| `scene_requirement` | string | ✅ | — | Target scene from tool02. |
| `target_effect` | string | ✅ | — | Unchanged global correct ego behavior from tool02. |
| `edit_strategy` | string \| null | — | null | Selected tool02 candidate editing idea; null means no other-vehicle editing needed. |
| `ego_input_edit_strategy` | string \| null | — | null | Optional intended change to ego history through 1.5s. Used only to plan stage 05A. |
| `vehicle` | VehicleAsset \| null | — | null | Optional real added-vehicle PLY and length/width/height. Required for executable add_vehicle arguments; do not infer from images. |

**`RetrievalContext` 对象字段：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `scene_ref` | SceneReference | ✅ | — |  |
| `video_path` | string \| null | — | null |  |
| `source_file` | string \| null | — | null |  |
| `source_segment` | object \| null | — | null |  |
| `mode` | string \| null | — | null |  |

**`VehicleAsset` 对象字段：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `asset_path` | string | ✅ | — |  |
| `size_m` | array<number >0> | ✅ | — | Length, width, height in metres; independently specified from the PLY asset. |

### 输出

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `suitability` | enum["ready", "unsuitable", "needs_information"] | ✅ | — | Suitability |
| `reasons` | array<string> | ✅ | — | Reasons |
| `edit_plan` | EditPlan \| null | — | null | edit_plan |
| `edit_options` | array<AttackPlanOption> | — | — | 1..3 unranked coarse alternatives when editing is needed. Consider existing and added vehicles together; do not force a count. Legacy edit_plan is accepted only when edit_options is empty. |
| `ego_history_plan` | EgoHistoryPlan \| null | — | null | One shared 05A decision before all other-vehicle options. Choose keep, an ego speed target, or a reachable ego route endpoint at 1.5s. |
| `retrieval_suggestion` | string \| null | — | null | Retrieval Suggestion |
| `scene_id` | string | ✅ | — | Scene Id |
| `scene_token` | string | ✅ | — | Scene Token |
| `target_effect` | string | ✅ | — | Target Effect |
| `evidence` | object | ✅ | — | Evidence |
| `attack_input` | object \| null | — | null | Program-built tool05 arguments validated against AttackInput. Forward this object unchanged when ready; null otherwise. |

**`EditPlan` 对象字段：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `operation` | enum["edit_existing", "add_vehicle"] | ✅ | — |  |
| `target_track` | string \| null | — | null |  |
| `maneuver` | enum["stationary", "left_cut_in"] \| null | — | null |  |
| `instruction` | string | ✅ | — | Scene-specific coarse vehicle behavior and event order; detailed trajectories are searched by tool05. |

**`AttackPlanOption` 对象字段：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `option_id` | string | ✅ | — |  |
| `operation` | enum["edit_existing", "add_vehicle"] | ✅ | — |  |
| `target_track` | string \| null | — | null |  |
| `maneuver` | enum["stationary", "left_cut_in"] \| null | — | null | Optional legacy hint for an added vehicle. Describe other motion in instruction. |
| `instruction` | string | ✅ | — | Brief mechanism and qualitative relative placement/motion. Leave coordinates, speed, exact timing and control search to tool05; numeric evidence is advisory unless explicitly required by the user. |
| `rationale` | string | ✅ | — |  |
| `uncertainty` | string | ✅ | — |  |

**`EgoHistoryPlan` 对象字段：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `mode` | enum["keep", "speed", "route"] | ✅ | — |  |
| `target_speed_mps` | number >=0 \| null | — | null |  |
| `route` | EgoHistoryRoute \| null | — | null |  |


## tool05_attack_vehicle_editor: `attack_vehicle_editor`

Edit one attack vehicle and export the accepted frozen scene.

**何时使用**：Apply a vehicle-edit strategy to a retrieved scene.

**上下游衔接**
- input: For a ready tool04 result, use result.attack_input unchanged as the call arguments. New tool04 handoffs use operation=auto and plan_options: preserve them so the SubAgent can choose/switch actors inside this call. Existing direct fixed-operation inputs remain supported. VADv2 target-collision and drivable-area self-test is enabled by default. Preserve it for attack tasks; false is an explicit geometry-only diagnostic. When revising after policy_runner shows an early safe stop or unmet maneuver failure, supply compact policy_behavior_summary (stopped_detected, progress_ratio_vs_recorded, signed_yaw_change_deg and task-specific failure_goal_met) in revision_feedback.evidence; on retry with unchanged inputs, pass reuse_completed_from only when recovery feedback supplies a completed object-selection checkpoint. When selection never completed, follow recovery_action without a checkpoint.
- output: The internal attack SubAgent edits and returns a frozen attack scene plus trajectory. With self-test enabled, completion includes exact-candidate policy failure evidence. With explicit self-test=false, completion covers geometry/task checks only. When vadv2_self_test=true the envelope also carries vadv2_probe_result_path, policy_evaluation and policy_collision_report, which may be reused only for that exact attack revision and test configuration. Tool07 evaluates this frozen attack package independently of tool06; the expert future must not be injected into the VAD test. No legacy PDM.

**工具规则**
- Internal selection receives a time-labelled conflict map and can preview1..3 initial hypotheses per cheap batch (at most6 evaluated hypotheses cumulatively) before committing an actor/placement. Hypothetical slower/coasting/braking paths are diagnostics, not VAD predictions or qualified expert trajectories. Exact-version VAD feedback uses fixed initial-ego coordinates and overlays actual motion with the reference route.
- For an optional same-task expert-margin revision, pass the successful tool06 expert_feedback.revision_feedback with the original attack instruction/contract. The backend validates the report and restores the unchanged shared history; at most one such revision shares existing cumulative search/API/probe budgets. Preserve the previous validated pair until new attack and defense are both checked.
- New runs may edit at any scene time from 0s. No implicit 3-second edit or attack-event delay applies. Existing and added actors can create a bounded initial-position/optional-yaw-adjusted candidate version through internal tools; this is not an in-trajectory teleport and requires regenerated/rigidly transformed motion, all checks and exact-version VAD evidence. Obey explicit task windows and user-owned requirements.
- A success envelope may contain status=unable_to_complete: no frozen scene exists and PDM/ego editing cannot run on it. Return the measured reasons so the caller can revise its upstream plan (target, scene or requirement) and retry. Only status=completed provides a frozen scene.
- Stage A executes the validated tool04 ego history plan locally through 1.5s with zero additional model calls. Stage B selects and edits the other vehicle on the frozen scene using the measured takeover state. Direct calls can set prepare_ego_history=false to keep the source history or provide ego_history_target_speed_mps to adjust it along the recorded route. Report shared_history and actual sampled takeover speed. Expert editing preserves that prefix; tool07 uses the same frozen package and navigation.
- For an explicit short ego path, supply ego_history_route with end_xy_m at 1.5s in the initial ego frame (rear axle), plus optional end_yaw_rad, end_speed_mps, via_xy_m, via_time_s and connector_id. Start position, heading and motion come from the recorded scene. This uses the same bicycle integrator as other-actor editing.
- For auto, supply 1..3 coarse plan_options and a real vehicle asset up front when available; options are suggestions, not hard acceptance gates. The SubAgent may switch object/operation twice inside the same cumulative API, constructor and action budgets. Explicit edit_existing/add_vehicle requests remain binding.
- For edit_existing, supply an evidenced target_track or let the backend select it from annotations. For add_vehicle, provide the asset, dimensions and supported maneuver required by input_schema.
- By default the SubAgent first requires sampled reference target-body contact inside 1.5–7.5s in both dense and native-replay geometry; near-miss alone is not eligible. It then uses native VADv2 collision and drivable-area feedback to revise or switch alternatives. Only contact-qualified candidates are tested; unchanged versions reuse results and all branches share vadv2_probe_budget (default 3). Probes omit video, official PDM and training export while preserving model/sensors/control; waiting time is outside the edit budget but inside the whole-tool timeout. Either sampled target collision or confirmed drivable-area violation by VADv2 after takeover qualifies. Missing map coverage is inconclusive.
- On editing_backend_not_configured, return the missing-backend evidence instead of inventing an attack; the caller decides the remedy.
- No legacy PDM planner self-check is available. Report VAD target collision and drivable-area compliance separately from editing checks. Budget exhaustion without a confirmed policy failure ends the search. Tool06 establishes expert feasibility.
- A frozen scene without simulation_input is not an executable WorldEngine package.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `scene_ref` | SceneReference \| string | ✅ | — | Full source-scene reference or saved retrieval-result JSON path. |
| `instruction` | string | ✅ | — | Qualitative vehicle-edit request for the selected source scene. |
| `operation` | enum["auto", "edit_existing", "add_vehicle"] | ✅ | — | auto lets the SubAgent select/switch existing or added actors using plan_options. Explicit operations remain binding caller requirements. |
| `plan_options` | array<AttackPlanOption> | — | — | Unranked coarse suggestions, not hard task contracts. The SubAgent chooses one or adapts it within the same scene/task/budget. |
| `target_track` | string \| null | — | null | Optional existing-vehicle track ID from annotations. If absent, the backend must select an evidenced track from scene annotations before editing; never guess a role ID or frame number. |
| `maneuver` | enum["stationary", "left_cut_in"] \| null | — | null | Required for add_vehicle; the documented new-vehicle constructors have different internal signatures. |
| `vehicle` | VehicleAsset \| null | — | null | vehicle |
| `prepare_ego_history` | boolean | — | `true` | Stage A automatically prepares and freezes feasible forward acceleration during 0–1.5s before the attack Agent selects/edits a vehicle. No additional Agent/API call. Explicit false preserves the source history. |
| `ego_history_target_speed_mps` | number >=0 \| null | — | null | Optional stage-A target speed at 1.5s. Omit for a bounded programmatic acceleration preference. Explicit targets are checked, never silently weakened. |
| `ego_history_route` | EgoHistoryRoute \| null | — | null | Optional short pose edit for stage A. Provide a reachable endpoint at 1.5s, with optional heading, speed, one intermediate point and connector guidance. |
| `constraints` | AttackConstraints | — | — | constraints |
| `timeout` | integer >0 | — | `3600` | Whole tool execution timeout, including selection/export. This is not the trajectory-stage budget. |
| `stage_timeout_s` | integer >0 | — | `420` | Trajectory editing budget excluding time waiting for native VADv2 probes. Whole-tool timeout and shared probe/API/constructor budgets still apply; retry does not reset them. |
| `vadv2_self_test` | boolean | — | `true` | Default native VADv2 feedback loop: test target collision and drivable-area compliance during 1.5–7.5s. Either confirmed failure qualifies. Probes skip video, official PDM scoring and training export, preserving native sensors/model/control. Explicit false permits geometry-only editing and makes no policy-failure claim. |
| `vadv2_probe_budget` | integer >=1 | — | `3` | Cumulative native policy tests across candidate edits and object reselection. Geometric/constructor checks happen before these expensive tests. |
| `task_contract` | object \| null | — | null | Optional user-supplied or explicitly approved attack_task.v2 hard contract with provenance. Do not convert the Agent's own coarse-plan preferences into this field. |
| `acceptance_policy` | object \| null | — | null | Optional named research acceptance profile and explicit overrides. |
| `revision_feedback` | object \| null | — | null | Compact review of the previous attack: stage, outcome, guidance and measured evidence including policy_behavior_summary (stopped_detected, progress_ratio_vs_recorded, signed_yaw_change_deg and task-specific failure_goal_met) and expert motion when relevant. Preserve the original instruction/contract; do not attach archived conversations or reasoning. |
| `reuse_completed_from` | string \| null | — | null | Prior attack service run. Reuse verified completed object_selection and verified trajectory candidates only when scene, instruction, assets and constraints match; recompute cleanly if incomplete or stale. |

**`SceneReference` 对象字段：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `scene_id` | string | ✅ | — |  |
| `scene_token` | string | ✅ | — |  |
| `sample_root` | string | ✅ | — |  |
| `dataset` | string | ✅ | — |  |
| `matched_camera` | string | ✅ | — |  |
| `frame_count` | integer | ✅ | — |  |
| `fps` | number \| null | — | null |  |
| `manifest_path` | string | ✅ | — |  |
| `frames_path` | string | ✅ | — |  |
| `cameras_path` | string | ✅ | — |  |
| `scenario_pkl` | string \| null | — | null |  |
| `asset_config_path` | string \| null | — | null |  |
| `asset_path` | string \| null | — | null |  |
| `runtime_asset_path` | string \| null | — | null |  |
| `maps_path` | string \| null | — | null |  |
| `cameras` | object | ✅ | — |  |
| `images_complete` | boolean | ✅ | — |  |

**`AttackPlanOption` 对象字段：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `option_id` | string | ✅ | — |  |
| `operation` | enum["edit_existing", "add_vehicle"] | ✅ | — |  |
| `target_track` | string \| null | — | null |  |
| `maneuver` | enum["stationary", "left_cut_in"] \| null | — | null | Optional legacy hint for an added vehicle. Describe other motion in instruction. |
| `instruction` | string | ✅ | — | Brief mechanism and qualitative relative placement/motion. Leave coordinates, speed, exact timing and control search to tool05; numeric evidence is advisory unless explicitly required by the user. |
| `rationale` | string | ✅ | — |  |
| `uncertainty` | string | ✅ | — |  |

**`VehicleAsset` 对象字段：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `asset_path` | string | ✅ | — |  |
| `size_m` | array<number >0> | ✅ | — | Length, width, height in metres; independently specified from the PLY asset. |

**`EgoHistoryRoute` 对象字段：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `end_xy_m` | array<number> | ✅ | — | Ego rear-axle [x,y] at 1.5s in scene-local metres. The recorded initial state is fixed. |
| `end_yaw_rad` | number \| null | — | null | Optional end heading in scene-local radians. |
| `end_speed_mps` | number >=0 \| null | — | null | Optional continuous rear-axle speed at 1.5s. |
| `via_xy_m` | array<number> \| null | — | null | Optional intermediate rear-axle point. |
| `via_time_s` | number >0 \| null | — | null | Optional time for the intermediate point. Omit to allow any time inside the prefix. |
| `connector_id` | string \| null | — | null | Optional scene-map connector used as route guidance. |

**`AttackConstraints` 对象字段：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `no_ego_contact_until_s` | number >=0 | — | `0.0` | Additional no-contact delay. Zero adds no extra delay beyond the configured attack event window (default 1.5–7.5s); it does not change WE history or the earliest edit time. |
| `target_window_s` | array<number> \| null | — | null |  |

### 输出

（无字段）


## tool06_ego_trajectory_editor: `ego_trajectory_editor`

Generate a correct ego expert trajectory for the frozen attack scene.

**何时使用**：Create the expert reference after attack editing.

**上下游衔接**
- input: attack_result: exact successful attack result_ref, result or envelope. Tool05 normally includes an exact-version native VAD collision self-test and freezes the shared 0–1.5s history. Preserve that history and takeover state. target_effect: the one global scene_instance_generator.target_effect.
- output: Return expert trajectory, expert_motion_summary, checks, optional video and simulation_input to Main Agent. Keep this expert trajectory separate from the frozen tool05 package used for VAD evaluation. Tool07 tests the tool05 package independently and can run in parallel with this editor; the expert future must not replace the policy test input. Current backend returns ilft_training_data=null: no converted ILFT annotations are exported; never pass the expert CSV as ann_file.

**工具规则**
- Selecting an initial candidate automatically supplies its trajectory diagram in the next model request; manual inspection remains available for other candidates. A diagram is not recorded as delivered until that request completes. Completed experts return measured clearance and motion headroom as expert_feedback for Main to decide whether one same-task attack revision is worthwhile; no new minimum-gap gate is imposed.
- Preserve the complete shared ego history through 1.5s, including the takeover pose. Edit only after 1.5s and pass motion/road/safety checks. Both VAD and the expert use the same tool05 history; do not repair that prefix by moving its poses.
- FOCI is available by default but is not run during initial preparation. The internal editor can request FOCI only for a concrete improvement need; all new candidates share its bounded generation/edit budget and unchanged acceptance checks.
- A success envelope with status=defense_not_found means completed bounded defense search: return compact feedback to attack editing on the original instruction and hard contract. An execution error does not justify lowering attack difficulty.
- Distinguish the final task goal from intermediate actions: stopping to wait and then proceeding is a driving goal, not a terminal-stop task. require_route_progress is an optional intent hint, inferred when null; it does not add a progress percentage or automatic maneuver-completion gate. Review the actual target_effect against scene evidence and measured progress/safety; safe waiting alone is not proof of failure.
- Preserve the frozen attack scene and its target vehicle.
- An expert trajectory is a reference, not a policy closed-loop success. Its CSV is not ann_file training data; use the explicit converted handoff blocks.
- On editing_backend_not_configured, return the missing-backend evidence; the caller decides the remedy.
- For an explicit retry after preparation_timeout or job_interrupted, pass the returned run_dir as reuse_completed_from. Completed understanding, constructor and initial FOCI checkpoints are reused only after content validation; missing or stale stages are recomputed. Never reuse a different task or attack revision.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `attack_result` | AttackOutput \| string | ✅ | — | Full attack result, successful envelope or saved attack-result JSON path. |
| `target_effect` | string | ✅ | — | One correct ego behavior for the target scene. |
| `allow_safe_stop` | boolean | — | `true` | Allow Safe Stop |
| `require_route_progress` | boolean \| null | — | null | Motion-goal hint for Agent review; null infers from target_effect, true means continue/proceed and false means a deliberate-stop goal. Does not impose any progress percentage, turn-angle threshold or automatic completion verdict. Temporary waiting can be part of a continued-driving task; allow_safe_stop controls explicit stop permission. |
| `use_foci` | boolean | — | `true` | Use Foci |
| `max_edit_candidates` | integer >=0 \| null | — | null | Optional new-edit resource budget. Zero permits selecting, inspecting and submitting existing candidates only; acceptance checks stay unchanged. |
| `reuse_completed_from` | string \| null | — | null | Prior ego service run. Reuse completed understanding, constructor and initial FOCI stages only when content-bound checkpoints match task/scene/policy/code; recompute missing or stale stages and start a fresh editor request. |
| `export_only` | boolean | — | `false` | Recheck and export the already completed expert in reuse_completed_from for the identical attack and target_effect. Does not call an Agent, constructor, FOCI or renderer. Requires reuse_completed_from. |
| `timeout` | integer >0 | — | `3600` | Timeout |

**`AttackOutput` 对象字段：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `schema_version` | string | — | `"editing_stage.v1"` |  |
| `stage` | string | — | `"attack_editing"` |  |
| `status` | string | — | `"completed"` |  |
| `mode` | enum["real", "mock"] | ✅ | — |  |
| `run_dir` | string | ✅ | — |  |
| `source_scene_ref` | SceneReference | ✅ | — |  |
| `scene_id` | string | ✅ | — |  |
| `scene_token` | string | ✅ | — |  |
| `operation` | enum["edit_existing", "add_vehicle"] \| null | — | null | Attack operation used to create the frozen scene. |
| `target_track` | string | ✅ | — |  |
| `selected_candidate_id` | string | ✅ | — |  |
| `plan_choice` | object \| null | — | null | Actual object/operation choice, associated suggestion and reason. Does not assert the suggested plan was followed exactly. |
| `shared_history` | object \| null | — | null | Frozen common 0–1.5s history, speed request/evidence and manifest path. Expert and VAD branches must preserve it; target speeds are distinguished from delivered sampled takeover speed. |
| `frozen_scene_path` | string | ✅ | — |  |
| `frames_path` | string | ✅ | — |  |
| `trajectory_path` | string | ✅ | — |  |
| `provenance_path` | string | ✅ | — |  |
| `submission_path` | string | ✅ | — |  |
| `checks` | StageChecks | ✅ | — |  |
| `annotation_changes` | object | ✅ | — |  |
| `simulation_input` | SimulationInput \| null | — | null | Prepared exact edited native scene/asset paths for tool07; null if export-report explains unavailable assets/conversion. A frozen metadata directory alone is not executable. |
| `native_export` | object \| null | — | null | Native export status, reason and report_path. Inspect this when simulation_input is null; never guess an executable folder. |
| `rendered_videos` | object | — | — | Camera name to edited render path; empty until rendered. Original recordings must not be relabelled as edited renders. |
| `pdm_probe_result_path` | string \| null | — | null | Completed self-check bound to the submitted candidate; tool07 verifies it against the frozen scene before reuse. |
| `vadv2_probe_result_path` | string \| null | — | null | Exact submitted candidate native VADv2 self-test evidence; reuse rather than repeating evaluation. |
| `policy_evaluation` | object \| null | — | null | VADv2 target collision and drivable-area checks, combined result and failure reasons. |
| `policy_collision_report` | object \| null | — | null | Measured native contact with the selected attack track after VAD takeover, independently of PDM total or road compliance. |

### 输出

（无字段）


## tool07_policy_runner: `policy_runner`

Run a policy in WorldEngine closed loop for evaluation or post-training data collection.

**何时使用**：Test a frozen attack against the driving policy, evaluate prepared scenes, or collect post-training rollouts.

**上下游衔接**
- input: Pass the frozen scene/asset paths from tool05 attack simulation_input and its target_track as collision_target_track. Tool06 produces a separate expert response and is not needed to launch this evaluation. Native model config/checkpoint/model_name/data_type are optional: omit them to use the fixed configured adapter policy. Supply overrides only for an explicitly requested different policy.
- output: Read official pdm_subscores, motion measurements in policy_behavior_summary and collision_report. Progress and yaw ratios impose no acceptance threshold or automatic maneuver verdict; turn_completed is deprecated and null. Evaluate target_effect against the scene and actual motion; a safe wait can be appropriate. For an explicitly requested attack test, actual target contact after takeover is attack success; PDM zero alone is not collision evidence. result_dir feeds failure_case_filter, and eligible dense-reward exports feed RLFT.

**工具规则**
- PDM mode simulates the edited scene only by default. challenge_baseline_scene_path is the original data source; challenge_run_baseline_diagnostic=true optionally runs the original scene for diagnosis, never as an acceptance gate.
- For the normal full pipeline, evaluate tool05 simulation_input with collision_target_track and its shared 0–1.5s history and frozen navigation. Run independently of tool06; never inject the expert future into this VAD test. Full scoring/video collection is separate from the light internal attack probe.
- Omit config_path, checkpoint_path, model_name and data_type to use the adapter defaults. Actual model files are checked by the backend before execution; never invent paths or ask the user for overrides merely because these optional fields are absent.
- Normal native scoring and RL collection use the same 1.5-7.5s window at 2Hz: num_history=4, max_frame_per_scene=16, num_future=12 or omitted, sampling_poses omitted. Dense collection needs 24 real source frames including its 8-frame lookahead. Only change this window when the user explicitly requests it; never extend scoring to the whole source scene as a fallback after export failure.
- with_dense_reward_manager is on by default, so every native run also exports an RLFT package; set it false only for a pure score-only diagnosis. Native collection exports validated annotations, camera bindings, actual plan indices and candidate rewards. no_eligible_scenes means no official PDM-zero training scene; skip training.
- target_collision_detected=null or incomplete coverage is unknown; interpolated possible contact and pre-takeover contact do not establish VAD failure.
- Match history/future frame counts to the native scenario. Do not pad missing observations.
- In pdm_closed_loop mode, pass the accepted attack pdm_probe_result_path as challenge_probe_result_path when available. The backend verifies exact scene/history/target/runtime/map semantics before reusing its real simulation; a mismatch is an error, never a successful test.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `execution_mode` | enum["native", "pdm_closed_loop"] | — | `"native"` | Native full simulation or the independent PDM closed-loop challenge. |
| `collision_target_track` | string \| null | — | null | Native mode: exact attack track from tool05. Return actual post-takeover body-contact evidence independently of PDM score, fault attribution and road compliance. Use tool05 simulation_input with original ego history for attack testing. |
| `config_path` | string \| null | — | null | Optional existing AlgEngine configuration override. Omit to use the policy_config configured in the WorldEngine adapter. |
| `checkpoint_path` | string \| null | — | null | Optional existing policy checkpoint override. Omit to use the adapter policy_checkpoint, shared with post-training. |
| `model_name` | string \| null | — | null | Optional experiment label override; the adapter supplies its fixed default when omitted. |
| `data_type` | string \| null | — | null | Optional native scenario data type override; the adapter supplies its fixed default when omitted. |
| `challenge_run_baseline_diagnostic` | boolean | — | `false` | Optionally simulate original scene for diagnosis only; no original-scene safety gate. |
| `challenge_baseline_scene_path` | string \| null | — | null | Read-only original scene for timestamps, map and source integrity. It is not simulated by default. |
| `challenge_attack_scene_path` | string \| null | — | null | Frozen attack scene used for the PDM challenge. |
| `challenge_target_track` | string \| null | — | null | Attack target track present in the frozen scene. |
| `challenge_target_is_new` | boolean | — | `false` | Whether the attack introduced a new target track. |
| `challenge_ignore_traffic_lights` | boolean | — | `false` | Explicit signal-agnostic PDM experiment; original light annotations remain unchanged, signal compliance is not evaluated. |
| `challenge_history_s` | number >0 | — | `3.0` | Recorded history before PDM takeover. |
| `challenge_probe_result_path` | string \| null | — | null | Optional accepted attack pdm_probe_result_path; reuse only after exact scene/history/target/runtime/map verification. |
| `react_type` | enum["R", "NR"] | — | `"NR"` | Reactive R or nonreactive NR background traffic. |
| `asset_name` | string \| null | — | null | Optional native renderer asset group. |
| `added_name` | string | — | `"mcp"` | Run-name suffix. |
| `num_scenarios` | integer >0 \| null | — | null | Optional maximum number of scenarios to execute. |
| `scenario_data_folder_path` | string \| null | — | null | Prepared native scenario directory; may come from editor simulation_input. |
| `asset_folder_path` | string \| null | — | null | Prepared native Gaussian/renderer asset directory. |
| `scenario_pkl_file_name` | string \| null | — | null | Scenario pickle name within scenario_data_folder_path. |
| `num_history` | integer >0 | — | `4` | Input history frame count; normal native runs use 4 for takeover at 1.5s at 2Hz. |
| `num_future` | integer >0 \| null | — | null | Future frame count; derived from the total budget when omitted. |
| `max_frame_per_scene` | integer >=5 | — | `16` | Normal native score-only and RL collection both use 16 frames: score 1.5-7.5s at 2Hz with num_history=4. Dense collection needs at least 24 real source frames for its 8-frame lookahead. Change only for an explicitly requested other window; num_history + num_future must equal this budget. |
| `sampling_poses` | integer >0 \| null | — | null | Derived automatically from scene frames minus history for full-rollout PDM. Omit this field; a legacy explicit value must equal the derived interval count. Does not change VAD prediction horizon. |
| `with_dense_reward_manager` | boolean | — | `true` | Collect native 8192-candidate PDM rewards for subsequent RLFT. On by default so every native run yields a training package; the last 8 source frames are reserved as the reward lookahead and are not driven. Turn off only for a pure score-only diagnosis. |
| `ego_policy` | string \| null | — | null | Optional native ego policy override. |
| `ego_controller` | string \| null | — | null | Optional native ego controller override. |
| `ego_client` | string \| null | — | null | Optional native planner-client override. |
| `ego_navigation` | string \| null | — | null | Optional native ego navigation override. |
| `agent_policy` | string \| null | — | null | Optional native background-agent policy override. |
| `agent_navigation` | string \| null | — | null | Optional native background-agent navigation override. |
| `agent_client` | string \| null | — | null | Optional native background-agent client override. |
| `timeout` | integer >0 | — | `86400` | Simulation process timeout in seconds. |

### 输出

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `startup_timing_report` | string \| null | — | null | Per-process cold-start timing report: imports, configuration, model/checkpoint/CUDA setup, Gaussian asset loading, history wait and first plan. Nested and parallel durations overlap. |
| `evaluation_scope` | object \| null | — | null | Verified full-rollout PDM window and actual state/reference/velocity/obstacle counts per scene. Extended-horizon score, not the standard fixed 4-second NAVSIM benchmark. |
| `collision_report` | object \| null | — | null | Actual closed-loop contact with the requested attack track. target_collision_detected=true after policy takeover is the attack criterion; false requires complete coverage; null is unknown. Interpolated possible contact and pre-takeover contact do not establish policy failure. |
| `pdm_subscores` | object \| null | — | null | Official PDM sub-scores read verbatim from all_scenes_pdm_averages_*.csv without altering PDM computation. |
| `policy_behavior_summary` | object \| null | — | null | Closed-loop ego kinematics over the evaluation horizon: absolute progress, speed profile, stop detection, yaw change/turn completion, and comparison against recorded and expert trajectories. |
| `result_dir` | string | ✅ | — | Closed-loop output directory containing official evaluation artifacts. |
| `reports` | array | ✅ | — | Native execution reports. |
| `policy_result` | string | ✅ | — | Policy-run result identifier or path. |
| `rollout_dir` | string \| null | — | null | WorldEngine openscene_format directory for rollout-based training. |
| `reward_dir` | string \| null | — | null | Dense candidate-reward directory including actual plan_idx.csv when collected. |
| `candidate_rewards_present` | boolean | — | `false` | Whether candidate rewards were exported by this run. |
| `rlft_training_data` | RLTrainingData \| null | — | null | Ready RL annotation/reward package for tool09.training_data; null when dense rewards were not requested or no PDM-zero scene is eligible. |
| `rlft_data_report` | object \| null | — | null | Annotation export status, counts and manifest_path; no_eligible_scenes is not a failed simulation. |
| `pdm_challenge` | object \| null | — | null | Independent baseline/attack PDM outcomes; not an official native PDM score. |

**`RLTrainingData` 对象字段：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `rollout_dir` | string | ✅ | — |  |
| `token_file` | string | ✅ | — |  |
| `vocab_path` | string | ✅ | — |  |
| `manifest_path` | string | ✅ | — |  |


## tool08_failure_case_filter: `failure_case_filter`

Select zero-score failure scenes from official closed-loop PDM results.

**何时使用**：Filter a completed run's official PDM results.

**上下游衔接**
- input: Use policy_runner.result_dir and the scenario_pkl corresponding to that execution; scene_token optionally selects one scene.
- output: failure_cases contains selected scene identities and PDM scores; pass their tokens to failure_analyzer when analysis is needed.

**工具规则**
- Only official PDM total score == 0 selects a failure. Missing scores or ambiguous scene matches are errors; do not substitute geometry checks.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `scenario_pkl` | string | ✅ | — | Scenario pickle whose identities correspond to the evaluated run. |
| `result_dir` | string | ✅ | — | policy_runner output containing exactly one official PDM CSV and the matching execution metadata. |
| `scene_token` | string \| null | — | null | Optional full scene identity or unambiguous 16-character suffix. |
| `timeout` | integer >0 | — | `120` | Score-reading process timeout in seconds. |
| `python` | string \| null | — | null | Optional interpreter for reading native scenario metadata. |

### 输出

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `is_failure_case` | boolean | ✅ | — | Whether any selected scene has official PDM total score zero. |
| `failure_cases` | array<object> | ✅ | — | Zero-PDM scene identities, scenario paths and scores. |
| `pdm_scores_file` | string | ✅ | — | Official PDM CSV actually read. |
| `scene_count` | integer >0 | — | `1` | Number of matched evaluated scenes. |
| `scene_results` | array<object> | — | — | Per-scene official score and failure selection. |


## tool09_policy_trainer: `policy_trainer`

Post-train a driving policy using imitation learning (ilft) or reinforcement learning (rlft).

**何时使用**：Train when the requested mode's converted data is ready.

**上下游衔接**
- input: ILFT: training_data.ann_file and token_file. RLFT: pass a ready policy_runner rlft_training_data bundle unchanged, including manifest_path; the trainer verifies annotation and reward provenance.
- output: Use the returned checkpoint in a subsequent policy_runner call to measure post-training performance.

**工具规则**
- RLFT requires rollout images/metadata, native 8192-candidate PDM rewards and plan_idx.csv. It keeps zero-PDM failure scenes and all their valid frames.
- Videos, frozen metadata or an expert CSV alone are not training_data. Do not silently switch RLFT to ILFT.
- A saved checkpoint and training_audit establish execution; improvement requires a separate evaluation.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `training_mode` | enum["ilft", "rlft"] | — | `"ilft"` | Post-training objective; required data depends on this mode. |
| `training_data` | TrainingData | ✅ | — | Converted ILFT annotations or complete RLFT rollout data, as specified by TrainingData. |
| `base_checkpoint` | string \| null | — | null | Policy checkpoint to update; the adapter resolves the same fixed artifact policy_runner uses when omitted. |
| `config_path` | string \| null | — | null | Optional mode-appropriate native model/training configuration. |
| `epochs` | integer >0 | — | `1` | Number of training epochs. |
| `worldengine_root` | string \| null | — | null | Optional WorldEngine source root. |
| `python` | string \| null | — | null | Optional AlgEngine training interpreter. |
| `work_dir` | string \| null | — | null | Optional export parent; execution uses an isolated runtime task. |
| `timeout` | integer >0 | — | `86400` | Training process timeout in seconds. |
| `seed` | integer >=0 | — | `0` | Random seed for this training run. |

**`TrainingData` 对象字段：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `manifest_path` | string \| null | — | null | RL annotation package manifest returned by tool07; verifies frame/image/plan/reward lineage before training. |
| `ann_file` | string \| null | — | null | ILFT converted annotation file; expert trajectory CSV alone is not this file. |
| `token_file` | string | ✅ | — | Token-selection file matching the training scenes. |
| `rollout_dir` | string \| null | — | null | RLFT openscene_format directory with meta_datas/, sensor_blobs/, pdms_pkl/, plan_idx.csv and official all_scenes_pdm_averages_*.csv. |
| `vocab_path` | string \| null | — | null | RLFT (8192,40,3) trajectory vocabulary used to generate the supplied candidate rewards. |

### 输出

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `checkpoint` | string | ✅ | — | Checkpoint produced by training. |
| `training_mode` | enum["ilft", "rlft"] | ✅ | — | Actual executed mode. |
| `log_path` | string | ✅ | — | Training log path. |
| `training_audit` | object \| null | — | null | Recorded losses, finite-value checks and parameter-update audit. |
| `data_preparation` | object \| null | — | null | Mode-specific training input preparation summary. |

