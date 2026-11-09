# 新编辑接口：editing_stage.v1

最新业务规则已实现，见[攻防循环最新规则](EDITING_INTERFACES.md)。tool05结果为 `completed | unable_to_complete`，tool06为 `completed | defense_not_found`；只有completed攻击可进入防御。默认仅测试编辑场景PDM，原始对照可显式开启且不作为验收门槛。

本接口参考《全流程智能编辑》中的三个阶段：攻击车编辑、自车冲突理解、自车轨迹编辑。
主 Agent 对外只有 attack_vehicle_editor、ego_trajectory_editor 两个工具；后者内部负责理解、构造器、可选 FOCI 与编辑修复。
接口、适配桥、校验、mock 和真实后端均已实现；实际后端及 PDM 挑战见[攻防闭环后端接入说明](EDITING_INTERFACES.md)。接口接入与真实场景质量验收分别记录。

## 上游调用

```python
from tools.registry import call_tool

# retrieval 是 tool03 的成功结果；requirements 是 tool02 的成功结果。
attack = call_tool("attack_vehicle_editor", {
    "scene_ref": retrieval["result"]["scene_ref"],
    "instruction": requirements["result"]["candidates"][0]["edit_strategy"],
    "operation": "edit_existing"
}, mode="real")

if attack["status"] == "success" and attack["result"]["status"] == "completed":
    accepted = attack["result"]
    pdm = call_tool("policy_runner", {
        "execution_mode": "pdm_closed_loop",
        "challenge_baseline_scene_path": accepted["source_scene_ref"]["sample_root"],
        "challenge_attack_scene_path": accepted["frozen_scene_path"],
        "challenge_target_track": accepted["target_track"],
        "challenge_target_is_new": accepted["operation"] == "add_vehicle",
        "challenge_history_s": 3.0
    }, mode="real")
    if pdm["status"] == "success" and not pdm["result"]["pdm_challenge"]["pdm_defense_succeeded"]:
        ego = call_tool("ego_trajectory_editor", {
            "attack_result": attack,
            "target_effect": requirements["result"]["target_effect"],
            "allow_safe_stop": True,
            "use_foci": True
        }, mode="real")
    # PDM 成功防御：在预算内将紧凑结果传入下一次 tool05.revision_feedback。
    # 执行错误：检查错误，不把它算成 PDM 失败。
```

已有车 target_track 可显式指定真实标注 ID；未提供时，新后端桥必须在任务准备阶段从标注选择目标并记入最终输出，不新增实例检测工具。
角色文字、帧序号不能当作对象 ID。不同模块使用别名时，桥接后端需要保存实际 track token 的映射和来源。

新增车输入使用 operation=add_vehicle、maneuver=stationary 或 left_cut_in，并提供 vehicle.asset_path 及 size_m=[长,宽,高]。
新增车辆的 ID 由后端分配。未提供资产、尺寸或模式时，在调用后端前报参数错误。
当前不声明支持删除车辆、右侧切入或任意新增行为。

tool05.scene_ref 也可传主 Agent 历史中成功检索条目的 result_ref 绝对路径；tool06.attack_result 可传成功攻击编辑条目的 result_ref。读取后仍校验完整对象；原来的对象输入继续支持。文件必须是相应工具 status=success 的标准输出 envelope，不能把服务内部任意 JSON 当作结果引用。

## 攻击输出

成功 envelope 的 result 按status区分。`completed`使用AttackOutput，主要字段：

| 字段 | 含义 |
|---|---|
| schema_version / stage / status | editing_stage.v1 / attack_editing / completed |
| mode / run_dir | real 或 mock；本次服务任务目录 |
| source_scene_ref | 原始 tool04 场景引用，完整保留 |
| scene_id / scene_token | 原始场景身份 |
| target_track / selected_candidate_id | 最终目标车和已接受候选 |
| frozen_scene_path / frames_path | 新冻结场景及编辑后 frames.json |
| trajectory_path | 已提交攻击车中心轨迹 CSV |
| provenance_path / submission_path | 冻结来源和已接受提交记录 |
| checks | reference_screen、逐项检查状态、报告文件与检查范围 |
| annotation_changes | 新增或更新哪些标注字段 |
| simulation_input | 已转换原生仿真数据的位置；未完成转换时为 null |
| rendered_videos | 相机名到编辑后渲染视频的映射；未渲染时为空 |

必须先接受候选并冻结场景，才能返回 completed。生成候选、看图、主动结束均不等于完成。
原始标注不覆盖，攻击编辑保持自车原始位姿及完整时间轴。自车阶段引用这份冻结版本。

攻击内部构造评估累计预算耗尽且没有候选同时通过轨迹检查和固定任务要求时，tool05直接返回unable_to_complete并结束；stop_reason为constructor_budget_exhausted_without_qualified_candidate，证据包含实际预算、候选数与失败检查。单次构造失败但额度未耗尽仍可继续；已有合格候选待PDM自检时继续验收。第14项规则只适用于攻击，防御仍允许Agent编辑/修复。预算耗尽的构造、已用完次数的对象重选工具按轮隐藏。

## 自车输出

EgoOutput 保留相同的场景身份、target_track、frozen_scene_path，并通过 attack_run_dir 绑定攻击版本。

| 字段 | 含义 |
|---|---|
| selected_candidate_id / selected_source | 最终轨迹及 constructor/foci/agent_edit/repair 来源 |
| expert_trajectory_path | 正确自车参考轨迹 CSV |
| trajectory_metadata_path | 轨迹坐标、参考点和单位说明 |
| conflict_brief_path | 内部冲突理解和车道建议 |
| submission_path / checks | 已接受提交及检查报告 |
| annotation_changes | 默认导出自车轨迹，不改写攻击冻结标注 |
| simulation_input | 后端真正导出的原生场景及资产信息，默认 null |
| ilft_training_data | 后端真正导出的 ann_file、token_file，默认 null |
| rendered_videos | 可交给 tool10 的已编辑渲染视频，默认空 |

CSV 约定包含 time_s、ego_x_m、ego_y_m、ego_yaw_rad，每个原始帧一行。
time_s 来自冻结场景实际微秒时间戳，允许 1 ms 浮点误差，不按固定帧率外推。
坐标原点为首帧自车后轴，x 向前、y 向左，yaw 为该局部坐标下朝向；专家位置参考点为自车后轴。
trajectory_metadata_path 的 JSON 必须包含：

```json
{
  "coordinate_frame": "initial_ego_rear_axle",
  "position_reference": "ego_rear_axle",
  "length_unit": "m",
  "angle_unit": "rad",
  "time_unit": "s"
}
```

攻击轨迹的参考点是目标车框中心，不能混用自车后轴坐标。写入逐帧 LIDAR 框和速度时的转换由后端完成。
原始相机图片只代表原始记录，新车可见性必须从编辑渲染验证。当前仅保留参考筛查语义，physical_avoidance=undetermined，不宣称动力学跟踪或连续时间安全已验证。

## 接入后端

mcp_servers/scene_editing/adapter/settings.json 的 commands.attack_editing 与 commands.ego_editing 已配置为绝对 Python 路径及 JSON 后端脚本。后端未配置时 real 返回 editing_backend_not_configured，不会自动使用 mock。
命令以项目根目录为 cwd，运行环境可在 argv 中选择不同解释器，不使用 shell 拼接。统一 JSON 由后端桥转换，既有 CLI 不直接接受该协议。
AttackInput 的 task_contract、acceptance_policy 分别携带调用方合同和验收配置；revision_feedback 携带前一轮紧凑评测结果，进入攻击对象选择与编辑上下文。回攻仍使用原始场景和原任务，不叠加上轮新增车、不引入旧对话或新硬门槛。AttackOutput.operation 记录攻击操作类型，供下游渲染使用。

候选级自检通过 pdm_self_check=true 启用，pdm_probe_budget 控制一次攻击内共享的模拟次数。AttackOutput.pdm_probe_result_path 可传 tool08.challenge_probe_result_path；后端核对完整场景和运行语义后复用，pdm_challenge.probe_reused 明确标记。

完整有界工具验收入口为 scripts/run_adversarial_validation.py；PDM 成功后自动回攻，达到预算停止。主 Agent 对应规则已在 agent/prompt.md，示例中的成功攻击不应直接跳过 PDM 进入防御。

stdin 为一个对象：

```json
{
  "schema_version": "editing_stage.v1",
  "stage": "attack_editing",
  "arguments": {},
  "output_dir": "/absolute/runtime/run_id/output"
}
```

arguments 是校验后的完整阶段输入。后端 stdout 只输出符合 AttackOutput 或 EgoOutput 的单个 JSON；mode/run_dir 由适配层强制写入实际值，日志输出到 stderr。
后端所有新增产物使用 output_dir 下的绝对路径，并返回其真实文件。检查报告、提交结果和来源文件必须存在。
后端必须返回 status=completed；未提交、无可用候选或预算耗尽等状态转换成 editing_not_completed，原始响应保存在 logs/stage_response.json。
格式错误、产物缺失、场景身份或时间轴不一致分别报错。执行前后校验源场景时间戳、token 和自车位姿等语义一致性，禁止覆盖源标注；不使用原始数据内容哈希。正式视频可位于任务 output 或显式 public_pipeline_root 下，其他产物仍须在 output 内。

## 下游衔接

- tool10：选择 rendered_videos 中的真实视频，结合原场景需求检查；没有渲染输出就不能宣称视觉通过。
- tool08：simulation_input 非空时，可合并 config_path、checkpoint_path、model_name、data_type 等必填模型参数。冻结 metadata 目录本身不是原生场景 pkl。
- tool10 IL：ilft_training_data 非空时，可作为 training_data；还需 base_checkpoint、与新数据相符的 config_path。CSV 不能当作 ann_file。
- tool10 RL：仍先由 tool08 生成稠密候选轨迹奖励和 rollout，再交给训练工具。新专家 CSV 本身不是 RL 奖励。

参考轨迹回放与驾驶策略闭环评测是不同用途。专家轨迹通过检查不代表 VADv2 已经成功驾驶；本次不修改训练损失或 synthetic imitation mask。

## 文件位置

- tools/tool05_attack_vehicle_editor/：主 Agent 攻击编辑入口。
- tools/tool06_ego_trajectory_editor/：主 Agent 自车编辑入口。
- mcp_servers/scene_editing/adapter/schemas.py：版本化契约。
- mcp_servers/scene_editing/adapter/service.py：JSON 桥与文件、身份、时间、坐标检查。
- mcp_servers/scene_editing/adapter/settings.json：后端命令。
- mcp_servers/scene_editing/core/adversarial/：已接入攻击、防御、PDM检查和三视角渲染后端。

mock 返回显式 mode=mock 的接口样例，不生成真实数据。桥接测试使用测试桩，仅证明数据协议和适配控制流，不证明编辑质量。

候选自检还可启用 pdm_defense_feedback=true（需同时启用 pdm_self_check）：返回有界防御构造器与刹停诊断，明确是建议反馈；匹配的池可供正式防御复用并重新核算，不改变场景硬要求。

## 已完成阶段续接与资源预算

EgoInput.reuse_completed_from 可指向同一攻击版本的失败防御任务。后端核对攻击输出、目标、停车策略、FOCI开关、理解配置，以及完整场景/地图/标线/政策/候选预算/实现绑定后，才复制已完成理解和兼容构造器；模型编辑请求重新构建，不继承旧对话。来源单独归档，不伪装成新的模型理解。

EgoInput.max_edit_candidates 默认None，采用原有5条新增编辑预算；显式0只允许选择、看图和提交已有候选，原有验收门槛不变。视觉请求v12采用阶段工具子集+tool_choice=auto，本地继续检查暴露工具、必需/额外字段、枚举和看图提交状态。此调整没有消除所有上游连接故障。

probe_pdm.defense_feasibility.witnesses 同时返回native_max_acceleration_mps2和generated_curve_max_acceleration_mps2；max_acceleration_mps2取最大值。当前参考检查、连续曲线运动极值与物理执行仍分别标明。

AttackInput.pdm_ignore_traffic_lights与tool08.challenge_ignore_traffic_lights默认false。显式true用于忽略信号灯的研究运行：源标注不改，PDM不接收灯态，结果traffic_light_policy.mode=ignored、traffic_light_compliance=not_evaluated；缓存绑定该开关。不是交通灯状态转换。


## 十工具调整

原独立视觉验收工具及提示词已删除，没有迁入编辑或回放工具。新编号见 [TOOLS.md](TOOLS.md)。


### Agent v16先编辑后比较

新受控防御运行增加refinement_workflow_version=refine_then_compare.v1。refinement_workflow给出pending、checked_changed_candidate_ids、unchanged_attempt_candidate_ids、保留的初始候选和剩余资源。资源足够时需实际生成/检查有变化的编辑候选后才开放最终提交；直接编辑、修复或父候选FOCI均可，来源不限。失败编辑仍可完成尝试，最终可选原候选；仅追加构造、空调用和无变化不计。资源不足明确记录未尝试，硬验收不变。旧运行不自动开启新阶段。

## 历史速度入口：先表达目标，再由后端算参数（已由下方统一接口替代）

`adjust_trajectory_speed` 用于“这条轨迹整体可取，只想改速度或停车过程”。必填项只有父候选、开始时间、目标和修改理由。例如保留某候选的路线与前段，从10秒开始修复末端停车：

```json
{
  "parent_candidate_id": "foci_001",
  "start_time_s": 10.0,
  "goal": "stop",
  "reach_target_time_s": 11.3,
  "reason": "保留前段进度和路线，修复末端仍在移动；重新检查净距与制动强度"
}
```

这里的时间仅演示参数含义，不表示任意场景均可行。`goal=stop`自动使用末速0并停车保持；`goal=reduce_speed`需要正的`target_speed_mps`。后端继承实际位置/速度并计算制动强度。默认`braking_style=smooth_deceleration`平顺增加和释放制动；也可选`constant_deceleration`。显式目标时间不会被自动挪动；省略时优先采用舒适参考，并在硬限制内适应剩余路程和时间，反馈实际参数。

默认`path_source=parent`保留父候选的空间路线；已经停下的父路线不能凭空延长。明确希望沿原道路路线继续前进时，可选`recorded_baseline`，后端从实际接管状态构造连接，重新检查曲线加速度、道路与接触，不直接吸附到原路线。开始时间之前的父轨迹及前三秒历史保持不变。

`preview_only=true`返回可行性与参考检查，不创建候选、不消耗候选额度，仍消耗一次模型动作；预览不可提交。确认执行时使用相同参数并去掉预览标志。目标不可达会返回所需制动强度或时间界限，属于该请求的局部失败。低层位置、路径进度和偏移工具继续可用。

FOCI的`optimize_defense_candidate`可选`preserve_route=true`、`terminal_behavior=stop`、`improvement_goal=balanced|recover_progress|smoother_motion`。保持路线采用沿父路线调整进度；停车目标要求窗口覆盖末尾两段采样，并在最后一个采样区间保持不动。这是显式采样端点约束，不宣称FOCI具备连续动力学保证。原三参数请求仍可用，默认自由末态、平衡目标。

连续制动轨迹在任意接管时刻使用重建曲线的位置和导数；只有CSV依据时明确使用采样估计。停车意图检查认可重建曲线中明确的零速度保持段，不再要求停车之前的倒数第二帧也必须静止。道路、碰撞及运动硬检查保持原策略。曲线系数只保存在本地证据中，不发送到Agent上下文。

## 当前速度接口：edit_speed_profile（2026-11-09）

Agent侧直接轨迹编辑只保留`edit_speed_profile`和`edit_route`。旧`brake_along_path`、`adjust_trajectory_speed`及坐标/偏移入口不再同时发布；底层旧操作仍能重建历史实验。

速度工具有两种操作：

- `set_speed_targets`：提供`parent_candidate_id`、`start_time_s`、`targets`与`reason`。每个目标只有绝对`time_s`和有符号`speed_mps`。正值前进，0停车，负值倒车但车头不翻转；前进/倒车之间必须显式给出零速目标。两个零速目标之间保持静止，后接正速度可重新起步。最后目标速度保持到场景结束，不强制停车。
- `retime_braking`：提供父候选、`braking_start_shift_s`、`preserve=stop_position|stop_time`及理由。正值推迟制动、负值提前。后端从最早受影响时刻重建原制动，保留指定停车目标，另一个停车结果重新计算并报告误差。只有可重建的明确制动段支持此操作；FOCI采样轨迹不凭空识别制动按钮，应使用速度目标。

例如保留FOCI前段，只修末端停车：

```json
{
  "parent_candidate_id": "foci_001",
  "operation": "set_speed_targets",
  "start_time_s": 10.0,
  "targets": [{"time_s": 11.3, "speed_mps": 0}],
  "reason": "保留前段进度和空间路线，只修复末端仍在移动"
}
```

例如推迟已有制动、保留原停车位置：

```json
{
  "parent_candidate_id": "constructor_024",
  "operation": "retime_braking",
  "braking_start_shift_s": 0.1,
  "preserve": "stop_position",
  "reason": "推迟制动介入，保留停车位置，检查新制动强度是否合适"
}
```

上述时间只是参数示例，不保证任意输入可行。默认`path_source=parent`。新速度目标默认`transition_style=smooth`；`retime_braking`省略此参数时继承父候选制动方式，不再自动换成平顺过渡。可显式选择`recorded_baseline`作为空间参考或指定速度过渡方式。更长路线、后方路线必须来自已知参考，不能凭空外推。空间路线不是随时间运动的偏移锚点；倒车沿已知路线反向走，不把车头旋转180度。参考路线存在不等于后方观测完整，道路、其他车辆、接触与运动检查照常执行。

`max_acceleration_mps2`只能降低既有上限；切向加速度满足上限不保证曲率导致的平面加速度合格。后端计算连续位置/速度，严格保持接管前轨迹和前三秒历史；显式速度目标不悄悄挪动。派生时间网格合并与目标相差1ms以内的构造点，但目标时间本身不合并或移动。显式等待按独立阶段积分，零速段不积累位置漂移。

`preview_only=true`检查请求但不创建候选，不消耗候选额度，仍消耗模型动作。合法但不可达的目标返回`not_feasible`及原因；实际生成但检查失败的候选仍保存，供Agent修复或比较。反馈包含实际速度目标、接管状态、停车目标保留误差、检查结果及相对父候选变化。连续曲线系数和空间路线系数只存本地，不进入模型上下文。

FOCI v8新增`speed_targets`，与上述目标使用相同的有符号速度/绝对时间含义。它们明确是采样速度的**软搜索偏好**，不是精确速度约束或新验收门槛；返回`speed_target_feedback`报告实际误差，不能宣称仅传入目标就达成。需要精确速度状态时，再用统一速度工具修复。已有`preserve_route`、`terminal_behavior`、`improvement_goal`仍有效；旧v5/v6/v7的无新目标重放语义保留。


### 路线复用修正与手动试用（2026-11-09）

新速度操作归档`speed_schedule_version=route_reuse.v2`，生成`signed_speed_schedule.v2`。对于已有`speed_route`的父候选，直接复用相同空间曲线系数；对于没有保存空间路线的旧构造/FOCI候选，首次从位置几何重建，接管处继承实际运动切线，车头朝向只用于姿态和运动方向判定，不混作各采样点的几何切线。重建曲线在接管点两侧分别保持二阶连续、接管点精确匹配位置/速度；新生成候选将空间路线存入本地证据，后续速度编辑不再重新拟合。采样轨迹的首次重建仍是插值，不能称为恢复了未知的真实连续运动。

制动时机调整默认继承父制动方式，并归档可继续调整的明确制动段；显式设置`transition_style`仍可覆盖默认值。新速度目标默认平顺过渡不变。反馈中的`actual_motion`给出实际峰值发生时间、纵向/法向分量及原有硬上限；“时间不够”“路线太短”“换向前需停车”等失败增加可读摘要和有条件的重试建议，不静默更换路线。

旧操作没有新版本标记时，独立重建和恢复使用冻结的v1数值实现，历史实验结果不回写。空间路线与连续曲线系数继续只存本地，模型上下文只保留摘要。

## 检索后粗规划到攻击编辑

主 Agent 把 tool03 的完整成功结果或绝对 result_ref，连同 tool02 的 scene_requirement、target_effect 和选定 edit_strategy，传入 tool04.retrieval_result。仅接受 retrieval_result 字段；旧 scene_ref 参数不再兼容。

ready 时，tool04.result.attack_input 已包含原始场景、操作、对象及编辑指令，并通过 AttackInput 校验，直接作为 tool05 调用参数。新增车辆还需预先提供真实 PLY 与尺寸；缺资源返回 needs_information，场景不合适返回 unsuitable 与重新检索建议。

粗规划只选全时段有有效 vehicle 标注的已有对象，不执行轨迹搜索或 PDM；tool05 保留精细求解能力。target_effect 独立保留给 tool06。


---

# 攻防循环最新规则



## 1. 流程目标

执行上游给定的攻击指令，生成符合要求且通过检查的攻击场景，再完成自车防御编辑。默认只对编辑后的场景运行PDM，原始场景PDM对照按需用于排查，默认不执行，也不作为提交或进入防御的门槛。

原始场景仍用于确定编辑对象、保护标注、固定前三秒观测及绑定编辑版本；取消原始PDM模拟不会取消这些数据依据。

```text
上游攻击指令
  → 攻击对象选择与轨迹编辑
      ├─ 未找到符合指令的合格攻击
      │     → 拒绝提交，返回原因与证据，结束本次任务
      └─ 攻击需求、几何、运动、原生导出检查通过
            → 仅测试编辑后的场景PDM
                ├─ PDM安全通过
                │     → 回到攻击Agent，增强挑战并重新检查
                └─ PDM未安全通过
                      → 自车理解与自车编辑
                          ├─ 搜索结束但未找到合格防御
                          │     → 反馈原因，回到攻击Agent重新设计
                          └─ 合格防御提交
                                → 复核、渲染、视频验收
```

回攻保持上游原始指令、指定对象和硬要求，不能为保留一个结果而把要求改松。攻击尝试、工具调用、API和时间预算均有上限；耗尽后如实返回当前阶段与原因。

## 2. 攻击拒绝与防御失败

| 情况 | 业务含义 | 后续动作 |
|---|---|---|
| 攻击Agent未找到符合原指令的合格方案 | 本次攻击任务未完成，允许拒绝提交 | 返回理由与证据，结束任务；不启动防御 |
| PDM在编辑后场景安全通过 | 当前方案没有通过PDM挑战 | 在预算内回攻增强挑战 |
| 防御搜索完成但未提交合格方案 | 反馈攻击场景与现有防御工具之间的问题 | 在预算内回攻重新设计，给防御留出可行空间 |
| API、依赖、工具协议或执行错误 | 执行尚未完成，不能评价场景可行性 | 有界恢复或暂停修复，保留原错误 |
| 已提交防御之后渲染失败 | 编辑结果保留，交付仍缺有效视频 | 复用已接受轨迹，修复资产或渲染，不重跑攻击 |

防御失败后的调整依据实际检查结果，例如介入时间不足、道路空间不足或运动超限；不能仅因API故障便降低攻击难度。调整之后如果攻击Agent也找不到符合原指令的方案，它可以正式拒绝提交，结束任务。

## 3. 攻击拒绝结果

拒绝提交是正常业务结果，应与执行异常分开。当前接口使用有状态区分的结果模型，拒绝结果不携带伪造的冻结场景或已接受轨迹。示例：

```json
{
  "schema_version": "editing_stage.v1",
  "stage": "attack_editing",
  "status": "unable_to_complete",
  "mode": "real",
  "run_dir": "/runtime/run",
  "reason": "当前搜索未找到同时完成切入并回正的合格轨迹",
  "stop_reason": "finished_without_submission",
  "failed_requirements": ["heading_alignment"],
  "attempts": [{"action": "construct_attack_candidates", "evidence_path": "/runtime/run/output/attack/trajectory_editing/tool_calls/turn_01/result.json"}],
  "evidence_paths": ["/runtime/run/output/attack/trajectory_editing/output.json"],
  "scope": "current_tools_and_budget",
  "global_infeasibility_proven": false,
  "detail": {}
}
```

理由需要对应实际尝试和检查。没有结果不等于理论上无解；不能虚构未运行的搜索或碰撞证据。调度器因预算停止时，应保留停止来源，不能写成模型已明确拒绝。接口调用可以正常结束并携带业务拒绝，不应统一包装成editing_not_completed异常。

## 4. 已同步的实现

- PDM runner与pdm_tool：默认只执行编辑后场景；保留可选原始对照诊断；删除原始对照必须安全的默认前置条件。
- 工具协议与缓存：同步输入字段、结果范围和模式绑定。原始场景路径可作为数据来源，但不代表必须执行原始PDM模拟；旧双组运行缓存不能冒充新模式。
- 攻击阶段：支持正式业务拒绝及原因、失败要求、实际搜索和证据引用。
- 外层循环：攻击拒绝结束任务；防御业务失败回攻；PDM安全通过回攻增强；执行错误分类恢复或暂停。
- 提示词与工具说明：同步上述去向，修改时告知用户；不新增案例专用窗口、净距目标或行为要求。
- 验证：业务状态、回攻、可选对照与渲染均有回归检查；案例2、3已在独立新目录重新启动。案例3原失败帧18、23已完成三视角重渲染与目视检查。

## 5. 案例2、3的剩余问题

| 案例 | 当前证据 | 后续工作 |
|---|---|---|
| 2：左侧新增车切入 | 上轮构造得到接近/接触后斜停的回退候选，未完成同向回正；尚无本轮合格新攻击提交 | 新规则落地后继续通用构造与编辑；无法满足原指令时正式拒绝提交 |
| 3：已有前车减速 | 攻击提交、编辑场景PDM失守、防御提交和独立复核通过 | 保留轨迹，修复车辆高斯位姿对齐与完整时窗渲染 |

案例2原始PDM对照的64个接触样本保留为历史诊断；按本规则，它不再是该案例恢复搜索的前置阻塞。但切入后的回正、固定历史、道路和运动要求继续有效。

案例3的高斯回放在9秒与标注相差约0.58m，超过原有0.5m对齐质量限制；目标原生位姿只覆盖到10.50秒，场景持续至11.50秒。当前18帧预览仅覆盖0–8.5秒，没有完整事件视觉验收。可研究标注绝对位姿驱动并验证读回，或更换全窗口对齐合格资产；不调大现有限制来让视频过关。

案例3防御较保守：约7.44秒停车，最小目标净距29.58m，进度约原记录45.1%。加速度峰值3.28m/s²超过2.4舒适性参考，但通过原有研究运动检查；这些是优化空间，未造成本轮渲染阻塞。

已修复的接管时间、背景回放、攻击接缝及重复工具调用问题和旧实验结果见[PDM基准与攻击接缝修复记录](EDITING_INTERFACES.md)。


## 6. 本轮接口与渲染变更

- tool05/tool06 继续返回标准 `status=success` envelope。其 result 通过状态区分：攻击 `completed | unable_to_complete`，自车 `completed | defense_not_found`。业务未完成没有可消费的合格轨迹；tool06仍只接受 completed 攻击。
- `challenge_baseline_scene_path` 保留旧字段名，但只读取原始数据以核对时间、ego、地图和导航。新增 `challenge_run_baseline_diagnostic=false`；默认无 baseline 仿真目录，结果中的 baseline 为 null。
- PDM缓存版本升级到v4，并绑定是否执行对照；旧双组缓存不会冒充新模式。
- 车辆高斯改用 `absolute_annotation_pose.v1`：canonical Gaussian 按候选绝对XY/朝向与道路高度摆放；原生位置误差仍保存为诊断。编辑后位置回读上限0.0002m、朝向上限0.00002rad。未调高旧0.5m原生回放限制。
- 时间覆盖外只复用同一刚体形状，将外观/姿态采样钳在已有资产时间内；这是显式渲染假设，不是新增真实观测。未编辑对象超过自身资产生命周期时省略并逐帧记录，禁止调用上游错误的时间外推。
- 镜头位置遇到道路高度小孔洞时，使用已有的包围测量局部平面插值，半径0.75m、最大残差0.05m保持不变。没有周围测量的区域仍拒绝渲染。


## 7. 2026-11-08本轮结果

案例3 `case3_policy_fix_20261108_run02`：新攻击candidate_024在第二次PDM自检中使PDM失守（首次8.20秒、12个接触采样），自车constructor_019提交、独立复核、24帧三视角渲染及历史视觉检查（已删除）验收通过。耗时1184.074秒，15次API请求。防御加速度4.69m/s²超过2.4舒适性参考，仍作为优化项记录。

案例2 `cases23_policy_fix_20261108_run01/case2`：五次构造累计1000次评估未找到同时满足回正与道路表面检查的候选；第二分支下一请求在300.006秒、4197流块后超时。没有进入PDM/防御。未生成完整结束动作，故状态为执行暂停，不伪造攻击拒绝。

原案例3通过轨迹的渲染恢复另存 `case3_render_recovery_20261108_run01`，也已完成24帧与历史视觉检查（已删除）验收；它与本轮新搜索分别归档。

---

# 轨迹编辑工具与原生导出

当前05（攻击车）和06（专家自车）使用同一套高层编辑入口：`edit_speed_profile`、`edit_route`。Agent先指定目标，工具生成并检查候选，Agent再比较；检查通过只代表候选可用，不代表最优。构造器、FOCI、检查/查看、候选混合和提交保留。旧坐标/偏移调用及其历史重放入口已删除。

## 调整速度

`edit_speed_profile`有两种操作：

- `set_speed_targets`：指定`start_time_s`和若干`{time_s,speed_mps}`。正值前进、零停车、负值倒车；反向前要显式经过零速。连续两个零速目标表示等待。最后目标速度保持到场景末尾。
- `retime_braking`：对已有明确停车制动段，指定`braking_start_shift_s`（正值推迟）和`preserve=stop_position|stop_time`。省略`transition_style`继承父候选的制动方式。普通FOCI采样轨迹没有可凭空移动的制动段，应使用速度目标。

`path_source`默认`parent`，不会自动延长停车候选的路线。明确选择`recorded_baseline`才表示沿已知原始路线继续。预览使用`preview_only=true`，不创建候选。

## 调整空间路线

`edit_route`表达平滑侧移、沿偏移路线行驶，以及可选的返回，不要求Agent计算绝对坐标或转角。例如：

```json
{
  "parent_candidate_id": "edit_001",
  "start_time_s": 5.0,
  "left_offset_m": -0.3,
  "transition_duration_s": 2.0,
  "hold_duration_s": 0.5,
  "return_duration_s": 2.0,
  "end_behavior": "return_to_route",
  "speed_policy": "preserve_speed",
  "preview_only": true,
  "reason": "缓慢向右避让，然后回到原路线"
}
```

- 正偏移向左、负偏移向右，方向以父路线的前进切线为准。05位置是攻击车车体中心，06位置是自车后轴；Agent不需要自行转换。
- `return_to_route`在侧移、保持和返回三个阶段后消除偏移；指回到空间路线，不保证追上父候选同一时刻的位置。
- `keep_offset`表示完成侧移后沿偏移路线继续至场景结束，不填写保持/返回时长。
- `preserve_speed`默认保留父候选的总车速，侧移会花费一部分前进距离；弯道上代价随偏移方向不同。
- `preserve_progress`保留父候选沿参考路线的进度，总车速可能增加。实际变化通过`route_edit_summary`返回；硬检查照常执行。
- 工具当前支持前进父路线，不支持从静止状态横向平移，也不支持对正在倒车的路线直接做侧移。先用速度工具准备可用的运动，再编辑空间路线。
- 过快的侧移、已知路线不够长、道路/车体/运动检查不通过，都会返回具体失败。合法但不可达的请求返回`not_feasible`；真正生成但检查失败的候选仍保存，供后续修复。

两类工具都继承接管位置与速度，前三秒及接管前的交付轨迹保持不变。空间编辑后的候选仍可接着调速。

速度目标表达沿路线行驶的有符号总车速；05车体中心在转弯时的朝前分量可能略小，两者分别返回，不能把它们混为一个测量值。检查反馈的`target_evidence`给出实际总车速及`actual_body_forward_speed_mps`。

## 连续路线和峰值

新速度编辑归档版本为`state_continuity.v4`，新空间编辑为`state_continuity.v2`。速度v1/v2/v3及空间v1实现分别保留，恢复旧实验继续使用原算法；旧攻击操作的版本缺省值在恢复入口显式补全。已保存的空间路线精确复用。对于只有采样依据或部分连续曲线的父候选，统一建立带1厘米位置误差上限的运动表示，报告实际误差；原有明确连续后段和停车保持仍保留。不能在容差内初始化时要求显式连接，不静默换路。

05按车体中心继承位置、速度、车体朝向和朝向变化率，并以0.6倍车长的现有参考轴距换算后轴关系；06按后轴继承状态。缺失的连续运动是数值重建，不是恢复了真实车辆状态。观测期和接管前交付轨迹不改，检查网格不减采样。朝向连接保留已有转向状态，采样残余侧向速度在一个原生采样间隔内平滑释放；该重建假设另行归档，运动检查照常执行。

输出时间节点保留明确请求的时刻，并考虑经过空间曲线节点的时刻；父曲线节点与请求时刻仅有浮点误差时合并重复值，避免构造极短多项式段。保沿路进度的空间编辑按父轨迹在相同空间路线上的实际进度构造，不混用上一采样区间的平均速度。

平面加速度同时包含纵向加减速和路线弯曲产生的法向分量，纵向制动力减小不保证合成峰值更低。反馈包含峰值时间、区间、纵向和法向分量。连续曲线的极值与道路/接触采样检查范围分别报告，不以粗采样降低峰值，也未放宽验收政策。

05/06的新版本共用编辑意图验收：速度目标误差、明确停车位置/时间的保留误差、停车保持、侧移目标和返回误差都参与资格检查。05预览、候选和独立复算采用相同检查。主要`max_acceleration_mps2`取采样与生成连续曲线的较大值，并单列两种口径；不再仅在附属报告中显示连续峰值。局部不可达返回`error_code`、`feasibility_limits`及按当前策略给出的建议，例如最短变速时长、路线缺少长度、最早可编辑时刻。不会自动更改Agent指定目标或验收阈值。攻击阶段另明确说明现有运动政策拒绝倒车，通用生成器的有符号速度能力不覆盖该政策。

## 05/06 → 07

`native_scene_export.py`将冻结攻击帧及可选专家CSV导出为WE原生`all_scenarios.pkl`和真实资产绑定。05返回攻击版本；06返回保留同一攻击、更新自车的专家版本。`simulation_input`可直接交给07，使用输出中的确切`scene_id`。

导出保留真实相机标定和高斯检查点，明确后轴/车体中心转换、原始时间戳与统一时间网格误差。完整车道中心线和拓扑通过共同官方多边形验证平移关系；无法建立坐标依据时拒绝导出。`annotation_changes.native_export_report_path`指向来源hash和导出检查报告。

支持已有车编辑和新增车资产合并。新增车的独立PLY由`native_vehicle_asset.py`按指定尺寸缩放点与高斯协方差，加入新的原生检查点，并绑定冻结轨迹的逐帧贴地位姿；源检查点不回写。资产、标定或地图绑定不足时仍返回`simulation_input=null`及原因。`exported`只代表原生包已准备，不能代替07回放与视频解码成功。

原生回放子进程对静态刚体资产启用编辑姿态覆盖，使重建时静止的车辆也能按编辑轨迹显示；源WorldEngine文件和检查点不回写，08策略评测不安装该适配。共享动态库目录由WorldEngine组件`paths.shared_library_dir`或`WORLDENGINE_LIBRARY_DIR`配置。

离线自测报告、图、攻防两版原生包位于`outputs/case3/20261109_editing_tools_selftest/`。GPU在普通工具沙箱中被隔离，获准的沙箱外执行可访问设备，详见自测目录gpu-access-diagnosis.json。

2026-11-09重启后已续接原生验证：`outputs/case3/20261109_attack_tool_hands_on/`下的`replay_attack01/`、`replay_existing01/`和`replay_added01/`分别完成已有车攻击版本、专家版本、新增移动车辆测试包的WE真实回放，三相机均24帧，tool07返回`native_replay=passed`和`video_decode=passed`。此前修正的序列化、坐标和逐相机渲染路径得到真实执行验证。新增车测试包未经过正式攻击提交，不能据此称攻击/PDM/专家全链通过；最新版统一工具的模型自主使用验证仍单独记录。随后57次攻击工具直接试用的成功、失败和限制见该目录README.md。

当前Agent说明只介绍已发布的速度/路线编辑入口及候选混合，不再指导调用隐藏的旧坐标工具。新运行记录`prompt_contract_version=goal_edit.v2`，普通运行器v18、受控视觉运行器v21；旧运行不隐式按新提示词恢复。

第一轮Agent易用性整合保留05/06职责、数值算法和验收政策，新增共用`agent_tool_view.py`：速度/路线编辑结果增加`decision_summary`，用`outcome`、候选身份、失败项和`next_step`概括本次结果，原始检查字段不删除。`recommended_previews`在路线不足等明确场景给出带适用条件的完整重试参数；`create_candidate_call`把通过的预览原样转为正式创建请求。它们都只是建议，不自动修改目标，也不代表候选已满足查看、PDM或提交门槛。

两个工具说明各加入一个经过自身schema校验的最小预览示例。05候选列表和06候选摘要共用`candidate_card`身份、参考检查、任务检查和关键指标字段；06的时序、行为和曲线细节仍按原视图保留。新提示词要求先读决策摘要，再按需查看原始证据。

最新接管与反馈修复、原57次请求对照和真实Agent试用见`outputs/case3/20261109_attack_continuity_fix/README.md`。同样请求的新编辑通过调用从8增至20，历史前缀及旧版本重放通过；真实Agent本轮只做了一次预览并回退原候选，不将直接工具成功当作模型自主生成成功。