# 新编辑接口：editing_stage.v1

本接口参考《全流程智能编辑》中的三个阶段：攻击车编辑、自车冲突理解、自车轨迹编辑。
主 Agent 对外只有 attack_vehicle_editor、ego_trajectory_editor 两个工具；后者内部负责理解、构造器、可选 FOCI 与编辑修复。
当前已实现接口、适配桥、校验、mock 和测试，未复制或连接实际轨迹编辑、渲染和导出算法。

## 上游调用

```python
from tools.registry import call_tool

# retrieval 是 tool04 的成功结果；requirements 是 tool02 的成功结果。
attack = call_tool("attack_vehicle_editor", {
    "scene_ref": retrieval["result"]["scene_ref"],
    "instruction": requirements["result"]["candidates"][0]["edit_strategy"],
    "operation": "edit_existing"
}, mode="real")

if attack["status"] == "success":
    ego = call_tool("ego_trajectory_editor", {
        "attack_result": attack,
        "target_effect": requirements["result"]["target_effect"],
        "allow_safe_stop": True,
        "use_foci": True
    }, mode="real")
```

已有车 target_track 可显式指定真实标注 ID；未提供时，新后端桥必须在任务准备阶段从标注选择目标并记入最终输出，不新增实例检测工具。
角色文字、帧序号不能当作对象 ID。不同模块使用别名时，桥接后端需要保存实际 track token 的映射和来源。

新增车输入使用 operation=add_vehicle、maneuver=stationary 或 left_cut_in，并提供 vehicle.asset_path 及 size_m=[长,宽,高]。
新增车辆的 ID 由后端分配。未提供资产、尺寸或模式时，在调用后端前报参数错误。
当前不声明支持删除车辆、右侧切入或任意新增行为。

tool05.scene_ref 也可传主 Agent 历史中成功检索条目的 result_ref 绝对路径；tool06.attack_result 可传成功攻击编辑条目的 result_ref。读取后仍校验完整对象；原来的对象输入继续支持。文件必须是相应工具 status=success 的标准输出 envelope，不能把服务内部任意 JSON 当作结果引用。

## 攻击输出

成功 envelope 的 result 为 AttackOutput，主要字段：

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

在 mcp_servers/scene_editing/adapter/settings.json 填入 commands.attack_editing 与 commands.ego_editing 的 argv 数组，例如绝对 Python 路径及 JSON 桥脚本路径。
默认数组为空，real 返回 editing_backend_not_configured，任务 manifest 记录 error；不会自动使用 mock。
命令以项目根目录为 cwd，运行环境可在 argv 中选择不同解释器，不使用 shell 拼接。
现有 run_instruction_attack.py、run_ego_conflict_brief.py 等 CLI 还需由后端桥转换，不能直接假定接受统一 JSON。

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
格式错误、产物缺失、场景身份或时间轴不一致分别报错。执行前后校验输入 frames.json 哈希，禁止覆盖源标注。

## 下游衔接

- tool10：选择 rendered_videos 中的真实视频，结合原场景需求检查；没有渲染输出就不能宣称视觉通过。
- tool13：simulation_input 非空时，可合并 config_path、checkpoint_path、model_name、data_type 等必填模型参数。冻结 metadata 目录本身不是原生场景 pkl。
- tool15 IL：ilft_training_data 非空时，可作为 training_data；还需 base_checkpoint、与新数据相符的 config_path。CSV 不能当作 ann_file。
- tool15 RL：仍先由 tool13 生成稠密候选轨迹奖励和 rollout，再交给训练工具。新专家 CSV 本身不是 RL 奖励。

参考轨迹回放与驾驶策略闭环评测是不同用途。专家轨迹通过检查不代表 VADv2 已经成功驾驶；本次不修改训练损失或 synthetic imitation mask。

## 文件位置

- tools/tool05_attack_vehicle_editor/：主 Agent 攻击编辑入口。
- tools/tool06_ego_trajectory_editor/：主 Agent 自车编辑入口。
- mcp_servers/scene_editing/adapter/schemas.py：版本化契约。
- mcp_servers/scene_editing/adapter/service.py：JSON 桥与文件、身份、时间、坐标检查。
- mcp_servers/scene_editing/adapter/settings.json：后端命令。
- mcp_servers/scene_editing/core/：预留独立编辑项目位置，尚未包含算法。

mock 返回显式 mode=mock 的接口样例，不生成真实数据。桥接测试使用测试桩，仅证明数据协议和适配控制流，不证明编辑质量。
