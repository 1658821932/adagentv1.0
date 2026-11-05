# adagent 项目结构与分工

## 当前范围

当前为 9 工具、4 后端组件。旧重建、实例发现及编辑链路已移除，新的攻击车与自车编辑已注册阶段接口，实际后端尚未配置。
不把接口占位或 mock 产物当作已经可用的真实编辑能力。

## 主流程

```text
用户任务
→ 主 LLM 根据工具说明和执行历史选择动作
→ ToolExecutor 调用工具并校验结果
→ 记录状态、产物、错误
→ 主 LLM 决定继续、修正、完成或停止
```

目前可完成失败分析、需求生成和场景检索；已有外部场景可以按接口进行验证、仿真和训练。
请求新场景编辑时，Agent 应说明缺失能力并停止，保留此前成功的分析与检索产物。

## 顶层目录

| 目录/文件 | 职责 |
|---|---|
| agent/ | LLM 决策循环、执行器、状态和日志 |
| tools/ | 9 个工具的说明、Schema、real 与 mock 入口 |
| mcp_servers/ | 四个后端的独立核心与 AdAgent 适配 |
| configs/agent.json | 最大步数、工具模式、LLM 角色 |
| configs/api.json | Kimi 与检索 API 的非敏感配置 |
| api_config.py | 统一解析 API profile 与密钥变量 |
| environments/ | 主 Agent 和各服务环境，不进入 Git |
| data/ | 场景、标注、高斯资产和地图，不进入 Git |
| logs/ | 主 Agent 决策、工具记录和运行状态，不进入 Git |
| tests/ | 编排、工具契约、服务和数据衔接测试 |
| scripts/ | 自动生成工具接口文档 |
| docs/ | 项目结构、接口和运行说明 |

## agent/ 文件职责

| 文件 | 职责 |
|---|---|
| main.py | 创建一次运行，循环决策、执行和写日志 |
| orchestrator.py | 主提示词、工具选择、决策 JSON 校验与格式纠正 |
| executor.py | 根据配置选择 real/mock，将异常转换成标准错误 |
| state.py | 任务、阶段、调用历史和运行状态 |
| schemas.py | 通用决策和工具结果 envelope 校验 |
| tool_descriptions.py | 汇总各工具的 description.py |
| llm_client.py | 解析统一模型配置并发送请求 |
| config.py | 校验运行配置和工具模式 |
| logger.py | 每步日志、调用追踪和状态快照 |

主 LLM 每次看到原始任务、所有可用工具说明和执行历史；工具内部 prompt 不作为全部历史提示词重复注入。
参数由主 LLM 从已有结果中组装，Schema 验证类型与必填字段。状态快照支持检查，不等于自动续跑。
恢复决策由主 Agent 作出；当前没有独立长期记忆库，也没有针对全部工具的自动质量检查器。

## tools/ 当前工具

| 编号 | 工具 | 主要负责人关注点 |
|---|---|---|
| 01 | failure_analyzer | WFA 证据、场景身份与输出分析 |
| 02 | scene_instance_generator | 失败分析到目标需求、正确行为与候选方案 |
| 04 | video_retriever | 检索、重排序、强制多相机场景绑定 |
| 05 | attack_vehicle_editor | 攻击阶段接口、冻结场景与来源 |
| 06 | ego_trajectory_editor | 自车理解、专家轨迹与下游导出接口 |
| 10 | scene_validator | 渲染视频抽帧与视觉质量反馈 |
| 13 | policy_runner | 原生场景和资产的仿真启动与产物 |
| 14 | failure_case_filter | 官方 PDM 文件读取、身份匹配与 0 分筛选 |
| 15 | policy_trainer | IL/RL 数据准备、执行和参数更新审计 |

每个工具保留 description.py、schemas.py、real.py、mock.py；tool02 另有内部提示词和 LLM 配置。
可执行契约与字段说明见 INTERFACES.md，不在多个文档重复维护参数默认值。

## mcp_servers/ 分层

| 服务 | core 职责 | adapter 职责 |
|---|---|---|
| sentry_video_search | 视频索引、召回、重排序、原生 CLI | 解释器隔离、公共入口、多相机场景绑定 |
| worldengine_failure_analysis | 取证、分析、原生 CLI 和配置模板 | Agent 调用、路径和运行任务管理 |
| scene_editing | 独立编辑后端预留位置 | 阶段 JSON 桥、身份与轨迹校验 |
| worldengine | SimEngine、AlgEngine、原生模型与配置 | 仿真启动、PDM 筛选、训练准备与审计 |

调用方向为 Tool → adapter/service.py → core。core 不导入主 Agent；任务目录与过程追踪在适配层管理。
各服务 runtime/<run_id>/ 包含 manifest.json、input/、logs/、output/。

## 新编辑接入边界

后续从 tool04.scene_ref 读取原始标注和已有高斯资产，分别对接攻击车编辑与自车专家轨迹编辑。
冻结场景、专家轨迹和原场景应保留来源及坐标/时间说明。需要转换成原生场景和训练数据后，才交给仿真与后训练。
上述新工具仅完成接口接入，后端未配置时停止，不使用旧编辑转换链路作为隐式替代。

## 配置与协作

.env 只放密钥；API 模型与端点在 configs/api.json；各后端解释器及路径在 adapter/settings.json。
环境是非 Git 运行资源，旧环境仍可能被下载任务或视觉抽帧复用，不因源码清理而删除。

修改工具 Schema 后运行 scripts/generate_interface_docs.py；改注册表时同步工具说明、配置、测试。
通用测试执行 python -m pytest；真实工具测试另需有效数据、外部 API 或 GPU。训练成功不代表指标提升。
