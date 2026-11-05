# 当前工具

当前主 Agent 注册 9 个工具，编号沿用现有目录，不重新编号。新的两个编辑工具目前仅提供阶段接口，真实后端尚未配置。

| 工具 | 输入概要 | 输出概要 | 执行位置 |
|---|---|---|---|
| tool01 failure_analyzer | 场景 token、取证配置 | 失败分析及证据 | WFA adapter/core |
| tool02 scene_instance_generator | 已有失败分析，可选变化方向 | scene_requirement、target_effect、三个候选 | LLMTool 公共运行器 |
| tool04 video_retriever | requirement、library | video_path、source_file、scene_ref | 检索 adapter/core |
| tool05 attack_vehicle_editor | scene_ref、instruction、operation | 冻结场景、目标 ID、攻击轨迹 | 新编辑 adapter，后端待配置 |
| tool06 ego_trajectory_editor | attack_result、target_effect | 专家轨迹、冲突理解、提交和检查 | 新编辑 adapter，后端待配置 |
| tool10 scene_validator | 渲染视频、场景需求 | passed、issues、evidence 等 | 工具内视觉评估 |
| tool13 policy_runner | 模型配置、权重、原生场景和资产位置 | 仿真结果、可选 rollout 与奖励 | WorldEngine adapter |
| tool14 failure_case_filter | 场景 pkl、官方结果目录 | 0 分失败场景、逐场景得分 | WorldEngine adapter |
| tool15 policy_trainer | ilft 标注或 rlft rollout、token 清单、基础权重 | 新 checkpoint、训练审计 | WorldEngine adapter |

字段级约束见自动生成的 [INTERFACES.md](INTERFACES.md)。

## 关键衔接

- tool02.candidates[i].base_video_requirement 原文可作为 tool04.requirement。
- tool04 real 强制绑定场景；scene_ref 引用全部相机、同步标注和高斯资产。逐帧列表保存在 manifest_path。
- tool10 可以验证外部提供的视频，但自身不会创建或修复编辑场景；视觉通过不能代替正式仿真评分。
- tool13 的输入是已准备的原生场景和资产，不再接受已退役编辑后端的输出目录。
- tool14 只读取官方 PDM 总分，不做纯几何碰撞检测。
- tool15 的 RL 当前只使用闭环 PDM 总分为 0 的 rollout；普通日志基线需另行适配。

工具执行成功仅表示本步骤完成；产物质量、场景可用性和后训练收益需要独立检查。
