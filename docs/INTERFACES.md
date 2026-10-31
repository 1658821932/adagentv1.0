# 工具接口参考

> 本文件由 `scripts/generate_interface_docs.py` 自动生成，请勿手工编辑。
> 修改工具契约（`tools/toolXX_*/schemas.py` 或 `description.py`）后重新生成：
> 
>     environments/video_retriever/.venv/bin/python scripts/generate_interface_docs.py

每个工具的用途和调用时机见 [tools 目录与 13 个 Tool 说明](TOOLS.md)；
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


## tool02_scene_requirement_generator: `scene_requirement_generator`

Use an LLM to generate targeted scenario requirements from failure evidence.

**何时使用**：Use after failure analysis, or when supplied analysis needs to become generation requirements. Proposed targets are not observed facts.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `failure_analysis` | object | ✅ | — | Non-empty failure analysis; accepts tool01 result or its successful envelope. |

### 输出

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `scene_requirement` | string | ✅ | — | Target scenario to generate; a design goal, not an observed result. |
| `failure_type` | string | — | `""` | Failure category supported by the source analysis. |
| `failure_reason` | string | — | `""` | Source-supported failure reason, preserving uncertainty. |
| `required_conditions` | array<string> | ✅ | — | Desired road, weather and traffic conditions. |
| `required_actors` | array<string> | ✅ | — | Required participant roles. |
| `risk_factors` | array<string> | ✅ | — | Risk factors supported by the analysis. |
| `target_effect` | string | ✅ | — | Driving-policy behavior the generated scenario should test. |
| `hazard_summary` | string | — | `""` | Concise hazard description. |
| `source_analysis` | object | ✅ | — | Original analysis attached by code, never reconstructed by the LLM. |


## tool03_scene_conceiver: `scene_conceiver`

Use an LLM to plan a scenario, source-video requirements and feasible editing ideas.

**何时使用**：Use after requirements are available and before video retrieval. Pass retrieval_query or base_video_requirement to video_retriever.requirement.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `scene_requirement` | object \| string | ✅ | — | Requirements from tool02, a successful envelope, or a direct natural-language requirement. |

### 输出

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `scene_plan` | string | ✅ | — | High-level scenario generation plan. |
| `base_video_requirement` | string | ✅ | — | Description of source video that should be retrieved. |
| `required_conditions` | array<string> | — | — | Requested road, weather and traffic conditions. |
| `required_actors` | array<string> | — | — | Participant roles required in the source video. |
| `edit_ideas` | array<EditIdea> | ✅ | — | Proposed edits, with roles and parameters. |
| `expected_effect` | string | ✅ | — | Intended scenario effect; actual policy response requires evaluation. |
| `retrieval_query` | string | ✅ | — | Natural-language query for video_retriever.requirement. |
| `source_requirement` | object \| string | ✅ | — | Original requirement attached by code. |


## tool04_video_retriever: `video_retriever`

Retrieve a real base driving video from the existing local Sentry Video Search index.

**何时使用**：Use when a suitable indexed real driving scene is needed as the starting point for reconstruction or editing. This tool does not search the internet.

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
| `mode` | string | ✅ | — | local_hit. |


## tool05_instance_discovery: `instance_discovery`

Discover important traffic participants from RGB videos using the DGGT YOLO backend.

**何时使用**：Use after reconstruction to detect instances in its preprocessed input frames.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `image_dir` | string | ✅ | — | Use reconstruction.result.input_dir, the parent of the input scene folder; do not use reconstruction.result.output_dir or append the scene name. |
| `output_path` | string | ✅ | — | DGGT instance_info.json output path. |
| `backend` | string | — | `"dggt_yolo"` | Must be dggt_yolo. |
| `scene_name` | string | — | `"000"` | Use reconstruction.result.input_scene_name (normally 000); the output scene 001 is not the input scene name. |
| `sequence_length` | integer | — | `40` | Exactly 40, matching reconstruction num_frames; never 64. |
| `start_idx` | integer >=0 | — | `0` | Start Idx |
| `weights` | string | — | `"yolo11m.pt"` | Weights |
| `conf` | number >=0 | — | `0.25` | Conf |
| `timeout` | integer >0 | — | `3600` | Timeout |
| `vis_dir` | string \| null | — | null | Vis Dir |

### 输出

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `instances` | array<object> | ✅ | — | Object IDs, categories, boxes and masks. Note: YOLO instance_info provides 2D boxes only, not a calibrated 3D vehicle track or road height; do not claim exact metres-to-ego placement or road contact from it. |
| `instance_info_path` | string | ✅ | — | DGGT-compatible per-frame instance_info.json for dggt_yolo. The returned state contains only sampled bbox frames; pass this path to Smart Edit tools for complete per-frame data instead of copying every detection frame into arguments. |


## tool06_reconstruction: `reconstruction`

Reconstruct a single-view driving video into a DGGT Mode 2 Gaussian scene. This is a deterministic 3D-reconstruction backend: it uses no LLM internally, so every constraint must be passed as an argument.

**何时使用**：Use after video_retriever when the task requires 3D scene reconstruction. Agent DGGT reconstruction processes only the first 40 frames of the source video.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `video_path` | string | ✅ | — | Absolute local path of the single-view driving video. |
| `num_frames` | integer | — | `40` | Exactly 40; Agent reconstructs only the first 40 frames. Never request 64 frames. |
| `export_render_state` | boolean | — | `false` | Optional boolean, default false: export full native DGGT temporal Gaussians for faithful WorldEngine rendering; currently supports unedited reconstruction only, not scene_editor outputs. |
| `timeout` | integer >0 | — | `86400` | Timeout |

### 输出

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `scene_ply` | string | ✅ | — | Combined reconstructed Gaussian scene PLY. |
| `input_dir` | string | ✅ | — | Parent directory of preprocessed input frames. Pass this verbatim to instance_discovery.image_dir. |
| `input_scene_name` | string | ✅ | — | Input scene name for instance_discovery.scene_name, normally 000; the output scene 001 is NOT the input scene name. |
| `output_dir` | string | ✅ | — | DGGT reconstruction output directory. |
| `rendered_video` | string | ✅ | — | Rendered reconstruction video. |


## tool07_edit_plan_builder: `edit_plan_builder`

Use an LLM to turn an edit request and discovered scene context into a grounded natural-language editing plan.

**何时使用**：Use after reconstruction and instance discovery when an edit must be specified. Supply actual instances so IDs can be checked. Existing trajectory edits are relative to original per-frame motion; cut-in timing comes from the current task.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `edit_request` | string \| array<object \| string> | ✅ | — | Required natural-language edit request or non-empty list of editing ideas, supplied under edit_request. |
| `scene_info` | object \| string \| null | — | null | Scene context, including frame count, coordinates and constraints. |
| `instances` | array<object \| integer \| string> \| object | — | — | Actual discovered instances or the successful instance_discovery result/envelope. IDs must come from this context. |

### 输出

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `edit_plan` | string | ✅ | — | Precise natural-language plan for SmartEdit; not its executable JSON config. |
| `target_instances` | array<integer \| string> | — | — | Only instance IDs actually present in the input context. |
| `operations` | array<Operation> | ✅ | — | Concrete proposed operations. |
| `constraints` | array<string> | — | — | Geometry, motion and evidence constraints. |


## tool08_edit_config_generator: `edit_config_generator`

Use Smart Edit MCP to convert an edit plan into a DGGT-compatible JSON configuration.

**何时使用**：Use after edit_plan_builder and before scene_editor. Set output_kind=ego_trajectory_offsets for natural-language ego render trajectory offsets; otherwise use rigid_config for instance edits.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `edit_plan` | string \| object | ✅ | — | Detailed editing plan. Existing instance trajectory waypoints are per-frame offsets added to original positions; compensate the source vehicle existing displacement. For cut-in, hold ego-lane position from frame 12 through 39 and use zero height offset. Do not request a new object unless asset_path and asset_scale are supplied. |
| `output_path` | string | ✅ | — | Required .json configuration path. |
| `output_kind` | enum["rigid_config", "ego_trajectory_offsets"] | — | `"rigid_config"` | rigid_config for an existing traffic instance, or ego_trajectory_offsets for ego render-camera motion (not closed-loop policy braking). |
| `instance_info_path` | string \| null | — | null | DGGT per-frame instance_info.json path for rigid instance edits. |
| `frame_count` | integer | — | `40` | For ego offsets, supply 40; Smart Edit must return one vector or exactly 40 vectors. |
| `scene_info` | object \| string \| null | — | null | Optional scene context. |
| `asset_path` | string \| null | — | null | Required exact vehicle PLY path when adding or replacing an asset. |
| `asset_scale` | number >0 \| null | — | null | Required explicit positive asset scale for add or replace; no default is assumed. |
| `timeout` | integer >0 | — | `1800` | Timeout |

### 输出

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `saved_path` | string | ✅ | — | Saved Smart Edit configuration path. |
| `tool_calls` | array<string> | ✅ | — | Smart Edit MCP tools called by the client. |


## tool09_scene_editor: `scene_editor`

Execute the DGGT Mode 2 edit_config against the source driving video.

**何时使用**：Use after edit_config_generator produces a JSON config.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `video_path` | string | ✅ | — | Source driving video. |
| `edit_config_path` | string \| null | — | null | Optional Smart Edit JSON; instance trajectory values add to each frame original instance position, not absolute target positions. |
| `ego_trajectory_offsets_path` | string \| null | — | null | Optional JSON of [forward,left,up] offsets from each predicted ego pose in DGGT native units. These alter rendering camera poses only; they cannot force the WorldEngine closed-loop policy to brake. |
| `camera_params_path` | string \| null | — | null | DGGT camera_params.json required for Smart Edit instance trajectory coordinates. |
| `meters_to_dggt` | number >0 \| null | — | null | Explicit metric-to-DGGT scale. Historical instance-edit experiments suggest ~0.01 as a starting estimate (0.001 for smaller shifts); verify on this scene. Do not copy WorldEngine translation_scale=10000: that is a different conversion. DGGT config offsets are pre-normalization world units; PLY-derived corrections must multiply ply_world_transform.scale. |
| `coordinate_system` | enum["smart_edit", "dggt_world"] | — | `"smart_edit"` | smart_edit or dggt_world. |
| `export_dynamic_frame_ply` | boolean | — | `true` | Optional; defaults to true for edited scenes so WorldEngine receives per-frame dynamic PLY metadata. |
| `num_frames` | integer | — | `40` | Exactly 40; render only the first 40 video frames. |
| `timeout` | integer >0 | — | `86400` | Timeout |

### 输出

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `scene_ply` | string | ✅ | — | Scene Ply |
| `rendered_video` | string | ✅ | — | Rendered Video |
| `output_dir` | string | ✅ | — | Output Dir |


## tool10_scene_validator: `scene_validator`

Evaluate whether an edited scene satisfies requirements and remains physically and logically reasonable.

**何时使用**：Use after editing before simulation. Run visual validation on at least 8 uniformly spaced frames and treat bad grounding, off-road placement, or fragmented vehicle geometry as edit failure even if artifact files exist.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `rendered_video` | string | ✅ | — | Rendered video from the previous edit step. |
| `scene_requirement` | string \| object | ✅ | — | Original requirement generated before editing. |
| `frame_count` | integer >=8 | — | `8` | Optional uniformly sampled frame count for Qwen-VL; defaults to 8. |
| `frame_paths` | array<string> \| null | — | null | Optional locally extracted keyframe paths. |
| `edited_scene` | object \| null | — | null | Optional scene metadata with scene_config and scene_ply. |
| `visual_evaluation` | VisualEvaluation \| null | — | null | Optional result from a local visual evaluator. |
| `local_evaluator` | string \| array<string> \| null | — | null | Local Evaluator |
| `scene_config` | string \| null | — | null | Scene Config |
| `scene_ply` | string \| null | — | null | Scene Ply |
| `ignored_reconstruction_artifacts` | array<string> | — | — | Ignored Reconstruction Artifacts |
| `mock_force_fail` | boolean | — | `false` | Mock Force Fail |

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
| `scenario_data_folder_path` | string \| null | — | null | Optional scenario directory; supports custom/DGGT-converted scenarios. |
| `asset_folder_path` | string \| null | — | null | Optional scenario asset directory. |
| `scenario_pkl_file_name` | string \| null | — | null | Optional scenario pickle filename. |
| `dggt_output_dir` | string \| null | — | null | Optional DGGT edited output directory; policy_runner automatically stages the selected instance and converts it to WorldEngine data. |
| `instance_id` | integer >=0 | — | `0` | DGGT instance to stage when dggt_output_dir is provided; defaults to 0. |
| `translation_scale` | number >0 | — | `10000.0` | WorldEngine metres per DGGT reconstruction unit; defaults to 10000 for normalized DGGT outputs. |
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

Filter new failure cases from simulation results and store useful training data.

**何时使用**：Optional dynamic-object geometric diagnosis after execution. Not an RL reward calculator or road-compliance evaluator. For RL, policy_trainer keeps failure scenes by closed-loop PDM total score.

### 输入

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `scenario_pkl` | string | ✅ | — | WorldEngine custom scenario pickle containing ego and dynamic-object tracks.（别名：`scenario_path`） |
| `result_dir` | string \| null | — | null | Optional policy_runner result directory. Scene identities must match executed metadata; missing or ambiguous matches fail explicitly. |
| `scene_token` | string \| null | — | null | Optional full scene ID or unambiguous 16-character suffix to evaluate only one scene. |
| `timeout` | integer >0 | — | `120` | Timeout |
| `python` | string \| null | — | null | Python |

### 输出

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `is_failure_case` | boolean | ✅ | — | Whether a dynamic-vehicle collision failure was found. |
| `failure_cases` | array<object> | ✅ | — | Decision failures found by geometric collision evaluation. |
| `collision` | boolean | ✅ | — | Dynamic-vehicle collision result. |
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

### 输出

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `checkpoint` | string | ✅ | — | Checkpoint saved by real training. |
| `training_mode` | enum["ilft", "rlft"] | ✅ | — | The actual mode used. |
| `log_path` | string | ✅ | — | Training log. |
| `training_audit` | object \| null | — | null | Mode-specific losses, finite-value check and actual parameter updates. |
| `data_preparation` | object \| null | — | null | Training inputs and rollout preparation summary. |

