# tools 目录与 13 个 Tool 说明

Tool 是主 Agent 可以选择并调用的一个功能入口，例如“检索视频”或“重建场景”。每个入口接收参数，执行功能，再返回统一格式的结果。一个后端可以支持多个 Tool；一个 Tool 内部也可以调用多个后端步骤。

本项目当前注册 13 个 Agent Tool，编号沿用已有目录，因此不连续。SmartEdit 服务内部的 15 个 MCP 子工具属于另一层，见 [MCP 组件说明](MCP_SERVERS.md)。

## 每个 Tool 是干什么的

下表按照当前 real 实现和 Schema 说明。输入、输出仅列常用字段，完整参数见各目录中的源码。

| 编号与工具 | 主要做什么、什么时候用 | 主要输入 | 主要输出 | 调用的功能 |
|---|---|---|---|---|
| [tool01：failure_analyzer](../tools/tool01_failure_analyzer/real.py)，失败分析 | 分析一个已有驾驶场景为什么失败，为后续设计针对性场景提供证据。 | 场景 token、WFA 评测配置路径。 | failure_analysis、证据详情、报告路径。 | WFA adapter 启动失败取证，结合官方结果、图像和数值分析。 |
| [tool02：scene_requirement_generator](../tools/tool02_scene_requirement_generator/real.py)，场景需求生成 | 将失败分析转换为“需要生成什么场景”，明确条件、参与者和目标效果。 | failure_analysis。 | scene_requirement、required_conditions、required_actors、target_effect。 | 公共 LLM 运行器，根据本工具 prompt.md 生成需求。 |
| [tool03：scene_conceiver](../tools/tool03_scene_conceiver/real.py)，场景构思 | 将需求展开成场景方案，决定找什么基础视频、可以做哪些编辑。 | scene_requirement，可为需求对象或文字。 | scene_plan、base_video_requirement、retrieval_query、edit_ideas。 | 公共 LLM 运行器，生成方案和检索需求。 |
| [tool04：video_retriever](../tools/tool04_video_retriever/real.py)，视频检索 | 在已有本地索引中寻找适合重建和编辑的驾驶视频。 | requirement；可选 library。 | 视频路径、匹配分数和候选记录。 | Sentry adapter 调用召回、重排和片段导出。 |
| [tool05：instance_discovery](../tools/tool05_instance_discovery/real.py)，实例发现 | 从重建预处理后的图像中识别并跟踪车辆，确定后续编辑哪个实例。 | image_dir、scene_name、output_path；40 帧。 | 实例 ID、逐帧检测框、instance_info_path。 | DGGT adapter 调用 YOLO/跟踪脚本。 |
| [tool06：reconstruction](../tools/tool06_reconstruction/real.py)，三维重建 | 将单目前视驾驶视频重建为可用于后续编辑的三维高斯场景。 | video_path；num_frames=40。 | 点云、相机参数、渲染视频、预处理图像目录和输出目录。 | DGGT adapter 组织预处理及 mode 2 推理。 |
| [tool07：edit_plan_builder](../tools/tool07_edit_plan_builder/real.py)，编辑计划生成 | 根据编辑要求和实际实例信息，明确修改对象、操作和约束。 | edit_request、instances；可选 scene_info。 | edit_plan、target_instances、operations、constraints。 | 公共 LLM 运行器；输入提供实例时校验引用的实例 ID。 |
| [tool08：edit_config_generator](../tools/tool08_edit_config_generator/real.py)，编辑配置生成 | 将编辑计划转换为 DGGT 可执行的 JSON 配置。 | edit_plan、output_path；按操作需要提供实例清单、资产和比例。 | saved_path、内部 MCP 工具调用记录。 | SmartEdit adapter 启动内部模型和 MCP 服务，组装并保存配置。 |
| [tool09：scene_editor](../tools/tool09_scene_editor/real.py)，场景编辑执行 | 实际应用编辑配置，生成修改后的场景和渲染视频。 | video_path、edit_config_path 或 ego_trajectory_offsets_path；轨迹编辑可能需要相机参数和坐标比例。 | 编辑后视频、场景产物、配置路径和坐标转换信息。 | DGGT adapter 进行必要的坐标转换并执行编辑渲染。 |
| [tool10：scene_validator](../tools/tool10_scene_validator/real.py)，场景验证 | 检查编辑结果是否满足原始需求，发现悬空、破碎、位置异常等画面问题。 | rendered_video、原始 scene_requirement；可提供已有视觉评估结果或本地评估器。 | passed、各检查项、evidence、issues。 | 检查文件与配置，调用视觉模型或指定的本地评估器。 |
| [tool13：policy_runner](../tools/tool13_policy_runner/real.py)，驾驶策略运行 | 将场景交给驾驶策略做闭环仿真，观察策略实际行为。 | 模型配置、checkpoint、场景类型和运行参数；也可提供 DGGT 编辑输出。 | 运行目录、场景报告、仿真产物等。 | WorldEngine adapter，按需转换场景后启动仿真。 |
| [tool14：failure_case_filter](../tools/tool14_failure_case_filter/real.py)，失败案例筛选 | 根据场景和实际运行轨迹判断是否出现动态对象碰撞，返回失败案例。 | scenario_pkl；可选 policy_runner 的 result_dir。 | is_failure_case、collision、碰撞事件和 failure_cases。 | WorldEngine adapter 中的几何筛选实现。 |
| [tool15：policy_trainer](../tools/tool15_policy_trainer/real.py)，IL/RL训练 | 按training_mode选择模仿学习或强化学习后训练。 | IL：ann_file、token_file；RL：rollout_dir、token_file、vocab_path、sample_filter；另需基础权重，可指定配置和epochs。 | checkpoint、日志、实际配置、数据准备统计和参数更新检查。 | WorldEngine adapter启动AlgEngine；支持ilft（默认）和rlft，RL奖励缺失时明确失败。 |

## 容易混淆的几组工具

**tool02 与 tool03**：02 明确需要什么条件和目标；03 将需求展开为具体场景构思、基础视频要求和编辑思路。

**tool07、tool08 与 tool09**：07 生成编辑计划，08 将计划变成配置文件，09 执行配置并渲染。生成配置文件并不表示场景已经编辑完成。

**tool10 与 tool14**：10 检查生成画面和需求是否一致；14 检查仿真轨迹中的动态对象碰撞。视觉验证不能代替驾驶策略闭环评测，当前几何筛选也不等同于官方道路合规和碰撞责任指标。

**tool13 与 tool15**：13 运行策略并收集结果；15 更新模型权重。13 的结果不会自动成为可训练的标注文件，进入 15 前需要准备对应训练数据。

## tools 根目录的文件

| 文件 | 主要做什么 |
|---|---|
| [registry.py](../tools/registry.py) | 保存工具名称与函数的对应关系，按 mock/real 模式分发调用。 |
| [llm_runtime.py](../tools/llm_runtime.py) | 为 tool02/03/07 提供公共模型调用、输入输出校验、有限纠错和结果包装。 |
| `scripts/` | 存放开发验证脚本，例如纯文本工具 smoke 和主 Agent 恢复流程验证。 |
| `__init__.py` | Python 包入口。 |

## 每个工具目录里的文件

| 文件 | 主要做什么 |
|---|---|
| `description.py` | 主 Agent 看到的工具说明，包括用途、调用时机和参数。由 agent/tool_descriptions.py 汇总。 |
| `real.py` | 实际执行入口。后端工具调用对应 adapter 并包装结果；tool02/03/07 绑定公共 LLM 运行器；tool10 保留自己的评估实现。 |
| `mock.py` 或 `tool.py` | 提供模拟结果，用于验证接口和编排流程。各目录情况不同，真正使用哪个文件以 registry.py 的注册为准。 |
| `schemas.py` | tool02/03/07 的输入输出结构与业务校验。 |
| `prompt.md` | tool02/03/07 内部模型的指令。 |
| `config.json` | tool02/03/07 的模型角色、提示词版本、输出长度和纠错设置。 |
| `__init__.py` | 将目录组织成 Python 包。 |

并非每个工具都有上述全部文件。对接后端的进程启动和格式转换已集中到各组件 adapter；当前工具入口不再承担这些完整实现。

## 调用与结果怎么看

主调用链是：

```text
主 Agent 决策
  → agent/executor.py
  → tools/registry.py：选择 mock 或 real
  → 对应工具的 run(input_data)
  → 返回结果，写入状态和日志
```

成功返回 `status="success"`、`tool` 和 `result`；错误由工具或执行器转换成标准错误结果。部分工具还返回模型用量、耗时等 meta。

`status="success"` 只说明该调用完成，不代表业务目标已经满足。例如 tool10 必须看 `result.passed`，tool14 则看 `result.is_failure_case`。下一步重试、修改方案或结束，由主 Agent 继续决策。

典型顺序为：

```text
失败分析 → 需求生成 → 场景构思 → 视频检索
  → 重建 → 实例发现 → 编辑计划 → 编辑配置 → 编辑执行 → 场景验证
  → 按用户目标执行仿真、失败筛选或模仿学习训练
```

用户已提供视频、需求或已有产物时，可以从对应步骤开始，不要求每次调用全部工具。

## 运行时说明的待同步项

核对文档时发现，部分 description.py 仍有旧表述：tool01 仍描述自由失败列表输入；tool10 的输出说明仍称必须使用本地评估器；tool13 的帧数说明只写 40，但实际实现也允许 60；tool14 的用途描述提到存储训练数据，但实际入口主要返回筛选结果。

本文按当前 real 实现说明。这些提供给主 Agent 的旧描述需要在后续代码整理中同步修正，当前文档更新不代表它们已经修复。

后端文件与子工具清单见 [MCP 组件说明](MCP_SERVERS.md)，纯文本工具开发方式见 [LLM 工具规范](LLM_TOOLS.md)。
