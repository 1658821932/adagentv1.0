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

**何时使用**：Diagnose a scene with archived evidence; reuse saved analysis when sufficient.

**上下游衔接**
- input: Use the existing scene token and analysis configuration.
- output: Pass result.failure_analysis to scene_instance_generator.failure_analysis; report_path/result_path locate the full evidence.

**工具规则**
- This tool analyzes one scene; it does not summarize a free-form list of failure cases.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `token` | string | ✅ | — | WorldEngine scene token to analyze; scene_token is also accepted.（别名：`scene_token`） |
| `config_path` | string \| null | — | null | Optional WFA configuration for archived evaluation data and scene metadata. |
| `run_name` | string \| null | — | null | Optional analysis run label. |
| `resume` | boolean | — | `true` | Reuse an existing completed analysis when available. |
| `timeout` | integer >0 | — | `1800` | Analysis process timeout in seconds. |

### 输出

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `failure_analysis` | object | ✅ | — | Single-scene failure_type, failure_reason and scene factors; forensics retains observations, candidate assessments and full evidence/report paths. |


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
| `variation_hints` | object | — | — | Optional qualitative variation directions, e.g. traffic_density: sparse, target_actor: crossing bus. Legacy parameter_assignment is accepted. No distance/speed/timing values are required.（别名：`parameter_assignment`） |
| `existing_instances` | array<string> | — | — | Previous requirement summaries or successful tool02 outputs. Exact normalized descriptions are rejected; semantic diversity remains a model suggestion. |

### 输出

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `scene_requirement` | string | ✅ | — | ONE target scene with context, participant interaction and event progression. All three candidates construct this same target. |
| `target_effect` | string | ✅ | — | What ego should do in the target scene. |
| `candidates` | array<Candidate> | ✅ | — | Exactly three alternative base-video retrieval descriptions paired with corresponding editing strategies. |

**`Candidate` 对象字段：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `base_video_requirement` | string | ✅ | — | Natural-language description of the BASE video to retrieve. Pass this string verbatim to video_retriever.requirement. Describe visible road context and actors needed for THIS strategy; an empty road/intersection is valid when adding a vehicle. Do not require the final hazard to already exist. |
| `edit_strategy` | string | ✅ | — | Qualitative strategy for this base video: add, delete or modify a vehicle and/or its trajectory; optionally adjust ego trajectory. Prefer changing at most one non-ego vehicle. No fixed object IDs or numeric trajectories are needed. |


## tool04_video_retriever: `video_retriever`

Retrieve a local camera video and its bound annotated multi-camera scene.

**何时使用**：Retrieve a source scene for editing or downstream use.

**上下游衔接**
- input: Use the selected scene_instance_generator candidate.base_video_requirement as requirement.
- output: Pass the successful history entry.result_ref or result.scene_ref to attack_vehicle_editor.scene_ref. source_file is the indexed original; video_path is the exported search clip.

**工具规则**
- No internet search. Real mode requires a valid scene binding.
- scene_ref describes the full source scene. source_segment is the matched index range, not a synchronized multi-camera crop.
- Check returned asset/image availability before requesting an operation that needs those files.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `requirement` | string | ✅ | — | Natural-language source-video search description. |
| `library` | enum["default", "waymo"] | — | `"default"` | Local index selection: default library or the separate Waymo library. |
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


## tool05_attack_vehicle_editor: `attack_vehicle_editor`

Edit one attack vehicle and export the accepted frozen scene.

**何时使用**：Apply a vehicle-edit strategy to a retrieved scene.

**上下游衔接**
- input: scene_ref: retrieval result_ref or unchanged scene_ref. instruction: selected tool02 candidate.edit_strategy.
- output: Pass this call result_ref to ego_trajectory_editor.attack_result. Non-null simulation_input can supply native scene/asset paths to policy_runner.

**工具规则**
- For edit_existing, supply an evidenced target_track or let the backend select it from annotations. For add_vehicle, provide the asset, dimensions and supported maneuver required by input_schema.
- On editing_backend_not_configured, stop: the stage has no configured algorithm backend.
- A frozen scene without simulation_input is not an executable WorldEngine package.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `scene_ref` | SceneReference \| string | ✅ | — | Full source-scene reference or saved retrieval-result JSON path. |
| `instruction` | string | ✅ | — | Qualitative vehicle-edit request for the selected source scene. |
| `operation` | enum["edit_existing", "add_vehicle"] | ✅ | — | Operation |
| `target_track` | string \| null | — | null | Optional existing-vehicle track ID from annotations. If absent, the backend must select an evidenced track from scene annotations before editing; never guess a role ID or frame number. |
| `maneuver` | enum["stationary", "left_cut_in"] \| null | — | null | Required for add_vehicle; the documented new-vehicle constructors have different internal signatures. |
| `vehicle` | VehicleAsset \| null | — | null | vehicle |
| `constraints` | AttackConstraints | — | — | constraints |
| `timeout` | integer >0 | — | `3600` | Timeout |

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

**`VehicleAsset` 对象字段：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `asset_path` | string | ✅ | — |  |
| `size_m` | array<number >0> | ✅ | — | Length, width, height in metres; independently specified from the PLY asset. |

**`AttackConstraints` 对象字段：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `no_ego_contact_until_s` | number >=0 | — | `3.0` |  |
| `target_window_s` | array<number> \| null | — | null |  |

### 输出

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `schema_version` | string | — | `"editing_stage.v1"` | Schema Version |
| `stage` | string | — | `"attack_editing"` | Stage |
| `status` | string | — | `"completed"` | Status |
| `mode` | enum["real", "mock"] | ✅ | — | Mode |
| `run_dir` | string | ✅ | — | Run Dir |
| `source_scene_ref` | SceneReference | ✅ | — | source_scene_ref |
| `scene_id` | string | ✅ | — | Scene Id |
| `scene_token` | string | ✅ | — | Scene Token |
| `target_track` | string | ✅ | — | Target Track |
| `selected_candidate_id` | string | ✅ | — | Selected Candidate Id |
| `frozen_scene_path` | string | ✅ | — | Frozen Scene Path |
| `frames_path` | string | ✅ | — | Frames Path |
| `trajectory_path` | string | ✅ | — | Trajectory Path |
| `provenance_path` | string | ✅ | — | Provenance Path |
| `submission_path` | string | ✅ | — | Submission Path |
| `checks` | StageChecks | ✅ | — | checks |
| `annotation_changes` | object | ✅ | — | Annotation Changes |
| `simulation_input` | SimulationInput \| null | — | null | Prepared native scene/asset paths for tool13; null until converted. A frozen metadata directory alone is not executable. |
| `rendered_videos` | object | — | — | Camera name to edited render path; empty until rendered. Original recordings must not be relabelled as edited renders. |

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

**`StageChecks` 对象字段：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `reference_screen` | string | ✅ | — |  |
| `checks` | object | ✅ | — | Only actually evaluated checks may be passed; skipped checks are not_checked. |
| `report_path` | string | ✅ | — |  |
| `physical_avoidance` | string | — | `"undetermined"` |  |

**`SimulationInput` 对象字段：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `scenario_data_folder_path` | string | ✅ | — |  |
| `asset_folder_path` | string | ✅ | — |  |
| `scenario_pkl_file_name` | string | — | `"all_scenarios.pkl"` |  |


## tool06_ego_trajectory_editor: `ego_trajectory_editor`

Generate a correct ego expert trajectory for the frozen attack scene.

**何时使用**：Create the expert reference after attack editing.

**上下游衔接**
- input: attack_result: successful attack editor result_ref, result or envelope. target_effect: the one global scene_instance_generator.target_effect.
- output: Use rendered_videos for scene_validator; non-null simulation_input for policy_runner; non-null ilft_training_data for policy_trainer ILFT.

**工具规则**
- Preserve the frozen attack scene and its target vehicle.
- An expert trajectory is a reference, not a policy closed-loop success. Its CSV is not ann_file training data; use the explicit converted handoff blocks.
- On editing_backend_not_configured, stop.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `attack_result` | AttackOutput \| string | ✅ | — | Full attack result, successful envelope or saved attack-result JSON path. |
| `target_effect` | string | ✅ | — | One correct ego behavior for the target scene. |
| `allow_safe_stop` | boolean | — | `true` | Allow Safe Stop |
| `use_foci` | boolean | — | `true` | Use Foci |
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
| `target_track` | string | ✅ | — |  |
| `selected_candidate_id` | string | ✅ | — |  |
| `frozen_scene_path` | string | ✅ | — |  |
| `frames_path` | string | ✅ | — |  |
| `trajectory_path` | string | ✅ | — |  |
| `provenance_path` | string | ✅ | — |  |
| `submission_path` | string | ✅ | — |  |
| `checks` | StageChecks | ✅ | — |  |
| `annotation_changes` | object | ✅ | — |  |
| `simulation_input` | SimulationInput \| null | — | null | Prepared native scene/asset paths for tool13; null until converted. A frozen metadata directory alone is not executable. |
| `rendered_videos` | object | — | — | Camera name to edited render path; empty until rendered. Original recordings must not be relabelled as edited renders. |

### 输出

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `schema_version` | string | — | `"editing_stage.v1"` | Schema Version |
| `stage` | string | — | `"ego_editing"` | Stage |
| `status` | string | — | `"completed"` | Status |
| `mode` | enum["real", "mock"] | ✅ | — | Mode |
| `run_dir` | string | ✅ | — | Run Dir |
| `scene_id` | string | ✅ | — | Scene Id |
| `scene_token` | string | ✅ | — | Scene Token |
| `attack_run_dir` | string | ✅ | — | Attack Run Dir |
| `frozen_scene_path` | string | ✅ | — | Frozen Scene Path |
| `target_track` | string | ✅ | — | Target Track |
| `selected_candidate_id` | string | ✅ | — | Selected Candidate Id |
| `selected_source` | enum["constructor", "foci", "agent_edit", "repair"] | ✅ | — | Selected Source |
| `expert_trajectory_path` | string | ✅ | — | Expert Trajectory Path |
| `trajectory_metadata_path` | string | ✅ | — | Trajectory Metadata Path |
| `conflict_brief_path` | string | ✅ | — | Conflict Brief Path |
| `submission_path` | string | ✅ | — | Submission Path |
| `checks` | StageChecks | ✅ | — | checks |
| `annotation_changes` | object | ✅ | — | Annotation Changes |
| `simulation_input` | SimulationInput \| null | — | null | simulation_input |
| `ilft_training_data` | ILTrainingData \| null | — | null | Converted expert labels for tool15 ilft. CSV alone is not ann_file; null until export is implemented. |
| `rendered_videos` | object | — | — | Rendered Videos |

**`StageChecks` 对象字段：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `reference_screen` | string | ✅ | — |  |
| `checks` | object | ✅ | — | Only actually evaluated checks may be passed; skipped checks are not_checked. |
| `report_path` | string | ✅ | — |  |
| `physical_avoidance` | string | — | `"undetermined"` |  |

**`SimulationInput` 对象字段：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `scenario_data_folder_path` | string | ✅ | — |  |
| `asset_folder_path` | string | ✅ | — |  |
| `scenario_pkl_file_name` | string | — | `"all_scenarios.pkl"` |  |

**`ILTrainingData` 对象字段：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `ann_file` | string | ✅ | — |  |
| `token_file` | string | ✅ | — |  |


## tool10_scene_validator: `scene_validator`

Check edited-render requirement consistency and visible geometry, motion and traffic plausibility.

**何时使用**：Check a new or revised edited render.

**上下游衔接**
- input: Use an actual path from editing result.rendered_videos and the original scene_requirement; original recordings are not edited renders.
- output: Continue only when result.passed=true. Otherwise use issues and frame evidence to revise the affected retrieval/editing stage.

**工具规则**
- The configured VLM evaluates sampled frames unless an explicit local evaluation is supplied. File existence alone does not validate the scene.
- Visual plausibility does not establish policy behavior, exact metric distances or dynamics guarantees.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `rendered_video` | string | ✅ | — | Actual edited-render video to inspect. |
| `scene_requirement` | string \| object | ✅ | — | Original target-scene requirement. |
| `frame_count` | integer >=8 | — | `8` | Uniformly sampled frames for VLM review; at least eight. |
| `frame_paths` | array<string> \| null | — | null | Optional already extracted keyframes of this same edited render. |
| `edited_scene` | object \| null | — | null | Optional editing-result metadata; does not replace the rendered video. |
| `visual_evaluation` | VisualEvaluation \| null | — | null | Optional result from an actual external visual evaluator; otherwise use the configured VLM. |
| `local_evaluator` | string \| array<string> \| null | — | null | Optional local evaluator command; receives scene/video context and returns JSON. |
| `scene_config` | string \| null | — | null | Optional scene configuration file to check for existence and valid JSON. |
| `scene_ply` | string \| null | — | null | Optional scene PLY artifact to check for existence. |
| `ignored_reconstruction_artifacts` | array<string> | — | — | Known source-render defects excluded from edit-quality issues unless they hide the target. |
| `mock_force_fail` | boolean | — | `false` | Mock-only forced validation failure. |

**`VisualEvaluation` 对象字段：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `requirement_consistency` | boolean | ✅ | — |  |
| `visual_valid` | boolean | ✅ | — |  |
| `geometry_valid` | boolean | ✅ | — |  |
| `physics_valid` | boolean | ✅ | — |  |
| `traffic_logic_valid` | boolean | ✅ | — |  |
| `confidence` | number >=0 | — | `0.0` |  |
| `issues` | array<string> | — | — |  |
| `evidence` | array | — | — |  |
| `ignored_reconstruction_artifacts` | array<string> | — | — |  |
| `model` | string \| null | — | null |  |
| `sampled_frames` | integer >=0 | — | `0` |  |

### 输出

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `passed` | boolean | ✅ | — | True only when all five checks pass and issues is empty. |
| `requirement_consistency` | boolean | ✅ | — | Rendered scene matches the requested scenario. |
| `geometry_valid` | boolean | ✅ | — | Visible vehicle placement, scale and road contact are plausible. |
| `physics_valid` | boolean | ✅ | — | Visible motion is plausible; not a dynamics guarantee. |
| `traffic_logic_valid` | boolean | ✅ | — | Visible participant interaction is plausible. |
| `visual_valid` | boolean | ✅ | — | Target is recognizable and visually continuous without edit-induced damage. |
| `issues` | array<string> | ✅ | — | Actionable failed checks and their evidence. |


## tool13_policy_runner: `policy_runner`

Run a policy in WorldEngine closed loop for evaluation or post-training data collection.

**何时使用**：Evaluate prepared scenes or collect post-training rollouts.

**上下游衔接**
- input: Use existing model config/checkpoint paths and native scene/asset paths. Merge an editor simulation_input only when it is non-null.
- output: result_dir feeds failure_case_filter. rollout_dir and native dense rewards feed policy_trainer RLFT.

**工具规则**
- Validate new edited renders before policy evaluation; prepared native dataset scenes can run directly.
- For RLFT data collection enable with_dense_reward_manager; ordinary evaluation does not imply candidate rewards are present.
- Match history/future frame counts to the native scenario. Do not pad missing observations.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `config_path` | string | ✅ | — | Existing AlgEngine model configuration path. |
| `checkpoint_path` | string | ✅ | — | Existing driving-policy checkpoint path. |
| `model_name` | string | ✅ | — | Experiment label for this policy. |
| `data_type` | string | ✅ | — | Native WorldEngine scenario data type. |
| `react_type` | enum["R", "NR"] | — | `"NR"` | Reactive R or nonreactive NR background traffic. |
| `asset_name` | string \| null | — | null | Optional native renderer asset group. |
| `added_name` | string | — | `"mcp"` | Run-name suffix. |
| `num_scenarios` | integer >0 \| null | — | null | Optional maximum number of scenarios to execute. |
| `scenario_data_folder_path` | string \| null | — | null | Prepared native scenario directory; may come from editor simulation_input. |
| `asset_folder_path` | string \| null | — | null | Prepared native Gaussian/renderer asset directory. |
| `scenario_pkl_file_name` | string \| null | — | null | Scenario pickle name within scenario_data_folder_path. |
| `num_history` | integer >0 | — | `4` | Input history frame count. |
| `num_future` | integer >0 \| null | — | null | Future frame count; derived from the total budget when omitted. |
| `max_frame_per_scene` | integer >=5 | — | `40` | Total frame budget; num_history + num_future must equal this value. |
| `sampling_poses` | integer >0 | — | `8` | Native rollout timing override in 0.5-second units. |
| `with_dense_reward_manager` | boolean | — | `false` | Collect native 8192-candidate PDM rewards for subsequent RLFT. |
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
| `result_dir` | string | ✅ | — | Closed-loop output directory containing official evaluation artifacts. |
| `reports` | array | ✅ | — | Native execution reports. |
| `policy_result` | string | ✅ | — | Policy-run result identifier or path. |
| `rollout_dir` | string \| null | — | null | WorldEngine openscene_format directory for rollout-based training. |
| `reward_dir` | string \| null | — | null | Dense candidate-reward directory including actual plan_idx.csv when collected. |
| `candidate_rewards_present` | boolean | — | `false` | Whether candidate rewards were exported by this run. |


## tool14_failure_case_filter: `failure_case_filter`

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
| `scenario_pkl` | string | ✅ | — | Scenario pickle whose identities correspond to the evaluated run.（别名：`scenario_path`） |
| `result_dir` | string | ✅ | — | policy_runner output containing official PDM CSV and execution metadata. |
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


## tool15_policy_trainer: `policy_trainer`

Post-train a driving policy using imitation learning (ilft) or reinforcement learning (rlft).

**何时使用**：Train when the requested mode's converted data is ready.

**上下游衔接**
- input: ILFT: training_data.ann_file and token_file. RLFT: policy_runner rollout_dir, matching token_file and the vocabulary used for its candidate rewards.
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
| `base_checkpoint` | string | ✅ | — | Existing policy checkpoint to update.（别名：`checkpoint_path`） |
| `config_path` | string \| null | — | null | Optional mode-appropriate native model/training configuration. |
| `epochs` | integer >0 | — | `1` | Number of training epochs.（别名：`total_epochs`） |
| `worldengine_root` | string \| null | — | null | Optional WorldEngine source root. |
| `python` | string \| null | — | null | Optional AlgEngine training interpreter. |
| `work_dir` | string \| null | — | null | Optional export parent; execution uses an isolated runtime task. |
| `timeout` | integer >0 | — | `86400` | Training process timeout in seconds. |
| `seed` | integer >=0 | — | `0` | Random seed for this training run. |

**`TrainingData` 对象字段：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `ann_file` | string \| null | — | null | ILFT converted annotation file; expert trajectory CSV alone is not this file. |
| `token_file` | string | ✅ | — | Token-selection file matching the training scenes.（别名：`nav_filter_path`） |
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

