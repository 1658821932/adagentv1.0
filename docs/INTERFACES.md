# 工具接口参考

> 本文件由 `scripts/generate_interface_docs.py` 自动生成，请勿手工编辑。
> 修改工具契约（`tools/toolXX_*/schemas.py` 或 `description.py`）后重新生成：
> 
>     environments/video_retriever/.venv/bin/python scripts/generate_interface_docs.py

每个工具的用途和调用时机见 [当前工具说明](TOOLS.md)；
调用方式、mock/real 边界和错误类型见 [工具接口与运行管理](EXECUTION_CONTRACTS.md)。
本文档只列字段级契约。

## tool01_failure_analyzer: `failure_analyzer`

Analyze one WorldEngine scene token using official metrics, image evidence and numeric verification.

**何时使用**：Use for evidence-based analysis of an existing WorldEngine scene with archived evaluation data. Supply token and, when needed, config_path.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `token` | string | ✅ | — | Token（别名：`scene_token`） |
| `config_path` | string \| null | — | null | Config Path |
| `run_name` | string \| null | — | null | Run Name |
| `resume` | boolean | — | `true` | Resume |
| `timeout` | integer >0 | — | `1800` | Timeout |

### 输出

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `failure_analysis` | object | ✅ | — | One dominant cross-case failure_type, failure_reason, key_scene_factors and risk_factors, with supporting and uncovered case IDs. Use only case IDs actually present in the input; never invent supporting evidence. |


## tool02_scene_instance_generator: `scene_instance_generator`

Expand a failure category into one target scene, ego's expected behavior, and three alternative base-video/edit-strategy pairs. Includes the former tool03 conception responsibility.

**何时使用**：Use saved failure_analysis. Select one candidate and pass candidates[i].base_video_requirement unchanged as video_retriever.requirement. Retain that candidate's edit_strategy for downstream editing. Empty base scenes are valid for adding a vehicle. Prefer editing one other vehicle and optionally ego trajectory.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `failure_analysis` | object | ✅ | — | Saved failure evidence or a successful tool01 envelope. Accepts failure_reason or a saved group with wrong_behavior and id. Keep original case IDs. |
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

Retrieve a dataset camera video and its bound multi-camera scene from the existing local Sentry index. Unbound videos raise an error.

**何时使用**：Use when a suitable indexed real driving scene is needed with annotations and Gaussian assets for downstream use. This tool does not search the internet.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `requirement` | string | ✅ | — | One natural-language description of the required base driving video. |
| `library` | enum["default", "waymo"] | — | `"default"` | Optional: default or waymo. Use waymo to search the separate Waymo front-camera index; default searches the existing local index. |
| `timeout` | integer >0 | — | `1800` | Timeout |
| `mock_video_path` | string \| null | — | null | Mock Video Path |

### 输出

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `video_path` | string | ✅ | — | Absolute local path of the matched video clip. |
| `match_score` | number | ✅ | — | Local retrieval similarity score. |
| `mode` | string | ✅ | — | Local retrieval mode. |
| `source_file` | string \| null | — | null | Original indexed camera video; video_path is the exported search clip. |
| `source_segment` | object \| null | — | null | Matched index segment in seconds relative to source_file; not necessarily the exported clip bounds. |
| `scene_ref` | SceneReference \| null | — | null | Required in real mode: complete scene identity, matched camera, all camera image availability, synchronized frame metadata, annotations and Gaussian asset paths. manifest_path stores ordered images/timestamps/calibrations. Real mode raises an error when scene binding is unavailable; mock may return null. Use this scene reference for dataset editing; camera videos and metadata describe the full source scene. |

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

Edit one attack vehicle using existing scene annotations and Gaussian assets; the stage owns candidate generation, checks, submission and frozen-scene export.

**何时使用**：Use after tool04. Pass scene_ref unchanged and one selected tool02 candidate.edit_strategy as instruction. edit_existing uses an evidenced track ID when supplied; otherwise its backend selects the target from annotations before editing; add_vehicle requires an explicit asset, size and stationary/left_cut_in maneuver. Never guess target IDs or asset paths. A separate operation is needed for unsupported manoeuvres. Backend configuration is required; on editing_backend_not_configured stop rather than retrying or claiming a new scene. Output is an accepted frozen scene, not merely a proposed plan. Any simulation_input may be passed to tool13 together with its required model arguments; null means scene conversion remains unavailable.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `scene_ref` | SceneReference | ✅ | — | Pass tool04.result.scene_ref, retaining its dataset identity, metadata and asset paths. |
| `instruction` | string | ✅ | — | Selected tool02 candidate.edit_strategy or an explicit user edit request. |
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

Generate and select a correct ego reference trajectory for the frozen attack scene; internally run conflict understanding, constructors, optional FOCI, inspection and repair.

**何时使用**：Use after attack_vehicle_editor. Pass its successful result/envelope under attack_result and the one global tool02.target_effect. Preserve the frozen scene and attack vehicle. Output is an expert reference CSV, not evidence that VADv2 drove safely. A null ilft_training_data means expert-label export is still unavailable: do not pass CSV as tool15.ann_file. RLFT still requires tool13 dense candidate rewards. Only rendered_videos contains edited renders; original scene cameras are not edited videos. Backend configuration is required; stop on editing_backend_not_configured.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `attack_result` | AttackOutput | ✅ | — | Pass the preceding attack_vehicle_editor.result or its successful full envelope. Frozen scene and target identity are retained. |
| `target_effect` | string | ✅ | — | Pass tool02.result.target_effect, the one global correct ego behaviour. |
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

Evaluate whether an edited scene satisfies requirements and remains physically and logically reasonable.

**何时使用**：Use after editing before simulation. Run visual validation on at least 8 uniformly spaced frames and treat bad grounding, off-road placement, or fragmented vehicle geometry as edit failure even if artifact files exist.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `rendered_video` | string | ✅ | — | Rendered video from the previous edit step. |
| `scene_requirement` | string \| object | ✅ | — | Original requirement generated before editing. |
| `frame_count` | integer >=8 | — | `8` | Optional uniformly sampled frame count for the visual model; defaults to 8. |
| `frame_paths` | array<string> \| null | — | null | Optional locally extracted keyframe paths. |
| `edited_scene` | object \| null | — | null | Optional scene metadata with scene_config and scene_ply. |
| `visual_evaluation` | VisualEvaluation \| null | — | null | Optional result from a local visual evaluator. |
| `local_evaluator` | string \| array<string> \| null | — | null | Local Evaluator |
| `scene_config` | string \| null | — | null | Scene Config |
| `scene_ply` | string \| null | — | null | Scene Ply |
| `ignored_reconstruction_artifacts` | array<string> | — | — | Ignored Reconstruction Artifacts |
| `mock_force_fail` | boolean | — | `false` | Mock Force Fail |

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
| `passed` | boolean | ✅ | — | Passed |
| `requirement_consistency` | boolean | ✅ | — | Requirement Consistency |
| `geometry_valid` | boolean | ✅ | — | Geometry Valid |
| `physics_valid` | boolean | ✅ | — | Physics Valid |
| `traffic_logic_valid` | boolean | ✅ | — | Traffic Logic Valid |
| `visual_valid` | boolean | ✅ | — | Visual Valid |
| `issues` | array<string> | ✅ | — | Issues |


## tool13_policy_runner: `policy_runner`

Run a driving policy in WorldEngine closed loop for evaluation or IL/RL data collection.

**何时使用**：Use after scene validation succeeds to load the scenario, start SimEngine and run the policy.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `config_path` | string | ✅ | — | Absolute existing AlgEngine config. Standard: <WORLDENGINE_ROOT>/projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct.py. |
| `checkpoint_path` | string | ✅ | — | Absolute existing policy checkpoint. Standard: <WORLDENGINE_ROOT>/data/alg_engine/ckpts/e2e_vadv2_50pct_ep8.pth. |
| `model_name` | string | ✅ | — | Experiment model name. |
| `data_type` | string | ✅ | — | WorldEngine scenario data type. |
| `react_type` | enum["R", "NR"] | — | `"NR"` | R or NR. |
| `asset_name` | string \| null | — | null | Optional renderer asset name. |
| `added_name` | string | — | `"mcp"` | Optional output suffix. |
| `num_scenarios` | integer >0 \| null | — | null | Optional positive scenario count. |
| `scenario_data_folder_path` | string \| null | — | null | Optional scenario directory; supports prepared dataset scenarios. |
| `asset_folder_path` | string \| null | — | null | Optional scenario asset directory. |
| `scenario_pkl_file_name` | string \| null | — | null | Optional scenario pickle filename. |
| `num_history` | integer >0 | — | `4` | Optional history frame count. |
| `num_future` | integer >0 \| null | — | null | Optional future frame count. |
| `max_frame_per_scene` | integer >=5 | — | `40` | Positive frame budget (at least 5), default 40. num_history + num_future must equal this budget; future is derived when omitted. Never pad observations. |
| `sampling_poses` | integer >0 | — | `8` | Optional native rollout timing override in 0.5-second units. |
| `with_dense_reward_manager` | boolean | — | `false` | Set true to collect native 8192-candidate rewards for RL; false preserves ordinary IL/evaluation collection. |
| `ego_policy` | string \| null | — | null | Ego Policy |
| `ego_controller` | string \| null | — | null | Ego Controller |
| `ego_client` | string \| null | — | null | Ego Client |
| `ego_navigation` | string \| null | — | null | Ego Navigation |
| `agent_policy` | string \| null | — | null | Agent Policy |
| `agent_navigation` | string \| null | — | null | Agent Navigation |
| `agent_client` | string \| null | — | null | Agent Client |
| `timeout` | integer >0 | — | `86400` | Timeout |

### 输出

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `result_dir` | string | ✅ | — | WorldEngine closed-loop output directory. |
| `reports` | array | ✅ | — | Native runner reports. |
| `policy_result` | string | ✅ | — | Policy Result |
| `rollout_dir` | string \| null | — | null | WE openscene_format directory; pass as training_data.rollout_dir to RL training. |
| `reward_dir` | string \| null | — | null | Candidate rewards and actual plan_idx.csv when dense rewards were requested. |
| `candidate_rewards_present` | boolean | — | `false` | Candidate Rewards Present |


## tool14_failure_case_filter: `failure_case_filter`

Select failure scenes exclusively using saved official closed-loop PDM total scores.

**何时使用**：Use after policy_runner. Official PDM CSV and matching execution metadata are required. score == 0 selects failures. Missing scores raise an error.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `scenario_pkl` | string | ✅ | — | WorldEngine custom scenario pickle containing ego and dynamic-object tracks.（别名：`scenario_path`） |
| `result_dir` | string | ✅ | — | Required policy_runner result directory containing official PDM scores. Scene identities must match executed metadata; missing or ambiguous matches fail explicitly. |
| `scene_token` | string \| null | — | null | Optional full scene ID or unambiguous 16-character suffix to evaluate only one scene. |
| `timeout` | integer >0 | — | `120` | Timeout |
| `python` | string \| null | — | null | Python |

### 输出

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `is_failure_case` | boolean | ✅ | — | Whether any selected scene has official PDM total score == 0. |
| `failure_cases` | array<object> | ✅ | — | Selected zero-score scenes with scene_id, scenario_pkl and pdm_score. |
| `pdm_scores_file` | string | ✅ | — | Pdm Scores File |
| `scene_count` | integer >0 | — | `1` | Scene Count |
| `scene_results` | array<object> | — | — | Scene Results |


## tool15_policy_trainer: `policy_trainer`

Fine-tune the driving policy with imitation learning (ilft) or reinforcement learning (rlft).

**何时使用**：Use ilft for trajectory annotations. Use rlft only after WorldEngine exports complete rollout images, metadata and 8192-candidate PDM rewards. Scene packages and videos alone are not RL training data. Evaluate the new checkpoint independently before accepting it.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `training_mode` | enum["ilft", "rlft"] | — | `"ilft"` | ilft (default) or rlft. Missing RL inputs cause an error; never silently fall back to IL. |
| `training_data` | TrainingData | ✅ | — | ilft: {ann_file, token_file}. rlft: {rollout_dir, token_file, vocab_path}. rollout_dir is the WE openscene_format directory with meta_datas/, sensor_blobs/, pdms_pkl/ including plan_idx.csv, plus all_scenes_pdm_averages_*.csv from the closed-loop run. Only scenes whose closed-loop PDM total score is 0 are kept as failure cases; all their valid frames are used for training. vocab_path is the (8192,40,3) vocabulary used to compute those rewards. |
| `base_checkpoint` | string | ✅ | — | Existing policy checkpoint. Record the architecture/config when comparing checkpoints.（别名：`checkpoint_path`） |
| `config_path` | string \| null | — | null | Optional mode-appropriate AlgEngine config; defaults to existing IL config or native RL rollout config. |
| `epochs` | integer >0 | — | `1` | Positive epoch count, default 1.（别名：`total_epochs`） |
| `worldengine_root` | string \| null | — | null | Optional WorldEngine code root. |
| `python` | string \| null | — | null | Optional AlgEngine interpreter. |
| `work_dir` | string \| null | — | null | Optional export parent; execution always uses an isolated runtime directory. |
| `timeout` | integer >0 | — | `86400` | Positive per-process timeout in seconds. |
| `seed` | integer >=0 | — | `0` | Nonnegative random seed, default 0. |

**`TrainingData` 对象字段：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `ann_file` | string \| null | — | null |  |
| `token_file` | string | ✅ | — | （别名：`nav_filter_path`） |
| `rollout_dir` | string \| null | — | null |  |
| `vocab_path` | string \| null | — | null |  |

### 输出

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `checkpoint` | string | ✅ | — | Checkpoint saved by real training. |
| `training_mode` | enum["ilft", "rlft"] | ✅ | — | The actual mode used. |
| `log_path` | string | ✅ | — | Training log. |
| `training_audit` | object \| null | — | null | Mode-specific losses, finite-value check and actual parameter updates. |
| `data_preparation` | object \| null | — | null | Training inputs and rollout preparation summary. |

