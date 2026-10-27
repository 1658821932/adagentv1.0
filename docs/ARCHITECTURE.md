# adagent 项目结构文档

> 面向协作者的项目导览。逐目录说明每个模块的职责、关键文件和相互关系。
> 阅读顺序建议：先读本文档建立整体认识，再按需深入对应目录的 README。

## 项目做什么

adagent 是一个**面向自动驾驶数据生成与评测的 Agent 系统**。给定一个失败模式（例如"直行时接近同车道低速前车未及时减速"），它能自动完成一整条链路：

分析失败原因 → 生成场景需求 → 检索合适的基础视频 → 三维重建 → 编辑场景（改前车轨迹、加 NPC 等）→ 渲染验证 → 闭环仿真评测 → 筛选失败案例 → 监督微调驾驶策略。

核心思想是：用 LLM 做编排和决策，把视频检索、三维重建（DGGT）、场景编辑（SmartEdit）、闭环仿真（WorldEngine）、策略训练（AlgEngine）这些异构能力统一成一套可调用的"工具"，由 Agent 按需串联。

## 整体架构

系统分三层，职责严格分离：

    ┌─────────────────────────────────────────────┐
    │  agent/        编排层   LLM 决定下一步调哪个工具    │
    ├─────────────────────────────────────────────┤
    │  tools/        工具层   14 个标准 Tool，各有        │
    │                         mock / real 两种实现      │
    ├─────────────────────────────────────────────┤
    │  mcp_servers/  能力层   视频检索/三维重建/编辑/      │
    │                         仿真/训练等外部组件         │
    └─────────────────────────────────────────────┘

运转循环（在 agent/main.py）：

    任务 → LLM 决策(选工具+参数) → 执行工具 → 结果写入状态 → 再决策 → … → finish

LLM 只做"选哪个工具、传什么参数"，不直接操作文件或模型。每个工具返回统一格式的结果信封，由 agent/schemas.py 校验。

## 顶层目录一览

| 目录 / 文件 | 职责 | 你需要改它吗 |
|---|---|---|
| agent/ | Agent 编排核心（LLM 决策、工具执行、状态、日志） | 改编排逻辑时 |
| tools/ | 14 个标准工具，每个有 mock/real 两实现 | 新增/修改工具时 |
| mcp_servers/ | 外部能力组件（检索、重建、编辑、仿真、训练） | 接入新组件时 |
| configs/agent.json | 运行配置：工具 mock/real 模式、LLM、步数上限 | 改运行行为时 |
| environments/ | 各组件独立的 Python 环境（不随源码分发） | 仅本地配置 |
| tests/ | pytest 测试 | 改代码必须同步 |
| _dev_scripts_archive/ | 一次性研发脚本归档 | 一般不动 |
| .env / .env.example | 密钥（**只放密钥**，配置在 configs） | 仅本地 |
| logs/ | 运行日志（每次 run 一个目录） | 不随源码分发 |

## agent/ — Agent 编排核心

这是系统的大脑。每个文件职责单一：

| 文件 | 职责 | 关键点 |
|---|---|---|
| main.py | 执行入口和主循环 | Agent.run(task)：循环 {LLM 决策→执行工具→更新状态}，直到 finish 或达到 max_steps |
| orchestrator.py | LLM 决策器 | 构造 system prompt（含所有工具描述和使用约束），让 LLM 输出 JSON 决策（call_tool/finish），带纠错重试 |
| executor.py | 工具执行器 | 按 configs/agent.json 里该工具的 mock/real 模式调用对应实现，并校验结果信封 |
| state.py | 运行时状态 | AgentState 记录任务、历史（决策+工具结果）、当前阶段、生命周期；to_orchestrator_dict() 会压缩逐帧检测结果以免撑爆 prompt |
| schemas.py | 数据校验 | 校验 LLM 决策格式和工具结果信封是否合法 |
| llm_client.py | LLM 客户端 | 封装 OpenAI 兼容 API；密钥从 .env 读，base_url/model 从 configs/agent.json 的 llm 块读（环境变量优先） |
| config.py | 配置加载 | 加载并强校验 configs/agent.json（缺字段/未知工具/非法模式都会报错） |
| tool_descriptions.py | 工具清单 | 定义每个工具的 name/description/when_to_use/inputs/outputs，供 LLM 决策时参考 |
| logger.py | 结构化日志 | 每次 run 建一个目录，记录决策、工具调用、错误、最终状态 |

### 编排层的关键设计

**1. LLM 只决策，不执行。** orchestrator.py 的 system prompt 里写死了大量领域约束（比如"instance_discovery 的输入要用重建的 input_dir""轨迹偏移是相对原始逐帧位置的，不是绝对位置""前 40 帧"等）。这些规则保证 LLM 不会乱传参数。**如果你改工具行为，很可能要同步改这里的 prompt。**

**2. 状态是 LLM 的唯一信息来源。** LLM 每步看到的 current_state 来自 state.to_orchestrator_dict()。大体积数据（如逐帧检测框）会被压缩成采样帧，完整数据写进日志文件，prompt 里只放路径。这是控制 context 长度的关键手段。

**3. 决策有校验和纠错。** LLM 输出先过 schemas.validate_decision，非法则按 decision_correction_attempts 次数把错误信息反馈给 LLM 让它重试。

## tools/ — 工具层（14 个 Tool）

每个 tool 是一个独立目录，标准结构是：

    toolNN_xxx/
      tool.py      # mock 实现（占位，返回接口兼容的假结果）
      mock.py      # 部分工具的 mock 在这里（历史原因，两种命名都有）
      real.py      # real 实现（真正调用 mcp_servers 里的能力）
      __init__.py

tools/registry.py 是总调度：维护 MOCK_TOOL_REGISTRY 和 REAL_TOOL_REGISTRY 两张表，call_tool(name, args, mode) 按模式分发。**新增工具要在这里登记。**

### 14 个工具按链路顺序

| # | 名称 | 干什么 | 属于阶段 |
|---|---|---|---|
| 01 | failure_analyzer | 分析失败案例，归纳跨案例的失败类型、原因、关键因素和风险 | 失败分析 |
| 02 | scene_requirement_generator | 由失败分析生成"要造什么场景"的需求 | 需求生成 |
| 03 | scene_conceiver | 设计高层场景方案：需要什么样的基础视频、怎么编辑 | 场景构思 |
| 04 | video_retriever | 按自然语言需求检索本地视频库，重排并导出片段 | 数据检索 |
| 05 | instance_discovery | 对视频做车辆检测跟踪，生成逐帧实例框和实例编号 | 实例发现 |
| 06 | reconstruction | 用 DGGT 把单目前视视频重建成动态三维高斯场景 | 三维重建 |
| 07 | edit_plan_builder | 把自然语言编辑意图转成结构化编辑计划 | 编辑规划 |
| 08 | edit_config_generator | 由编辑计划生成具体编辑配置（轨迹偏移、资产等） | 编辑配置 |
| 09 | scene_editor | 执行编辑，调 DGGT 渲染编辑后的场景 | 场景编辑 |
| 10 | scene_validator | 检查输出文件和渲染画面是否符合需求（VLM 视觉验证） | 结果验证 |
| 11 | recovery_decider | 验证不通过时决定怎么重试/调整 | 失败恢复 |
| 13 | policy_runner | 用 WorldEngine 跑闭环策略仿真 | 闭环仿真 |
| 14 | failure_case_filter | 从仿真结果里筛出失败案例 | 失败筛选 |
| 15 | policy_trainer | 用 AlgEngine 对筛选数据做监督微调（ILFT，**不含 RL**） | 策略训练 |

> 注意：**没有 tool12**（编号跳过）。tool11_recovery_decider/ 里有个 tool12_simulation_loader 子目录，是历史遗留的内部辅助，不是独立工具。

### mock / real 双模式

每个工具都有两种实现，由 configs/agent.json 的 tools.modes 控制：

- **mock**：返回接口兼容的占位结果，不真正跑模型/渲染/仿真。用于快速验证 Agent 编排逻辑、调 prompt、跑测试。
- **real**：真正调用 mcp_servers 里的组件。需要对应环境、权重、数据就绪。

**给协作者的建议：改编排逻辑时先把相关工具设成 mock，跑通了再切 real。**

## mcp_servers/ — 外部能力服务

这里是被工具层调用的"重资产"组件。命名带 "mcp_servers"，但**只有 SmartEdit 真正走 MCP 协议**，其余主要通过 JSON 子进程或项目脚本适配。

| 目录 | 对应工具 | 职责 | 来源 |
|---|---|---|---|
| worldengine_failure_analysis/ | tool01 | 基于 WorldEngine 官方评测结果，用 VLM 看图查数做失败归因 | 自研（wfa） |
| sentry_video_search/ | tool04 | 视频建库、语义召回、重排、片段导出 | 自研（含 vendor） |
| dggt/ | tool06/09 | 动态高斯重建 + 实例编辑渲染 | fork 自 xiaomi-research/dggt，含定制 |
| smart_edit/ | tool07/08 | 自然语言编辑要求 → 编辑配置（MCP 服务） | 自研 |
| worldengine/ | tool13 | 闭环策略仿真、评测 | OpenDriveLab/WorldEngine |

### 各组件要点

**worldengine_failure_analysis（wfa）**：输入场景 token + 评测配置，VLM 看多相机图、查自车/周车数值，输出失败现象、候选原因、证据和不确定性。配置用 configs/complete.json（参考 .example），VLM 密钥放 .secrets/vlm.json（权限 0600，不入库）。

**sentry_video_search**：召回/重排用 DashScope qwen3-vl-embedding，需要真实 DASHSCOPE_API_KEY（中转站不支持该端点）。视频索引库体积大，不随源码分发。

**dggt**：core/ 里是 fork 的上游代码（含 inference.py、dggt/ 包、pretrained/），外层 runner.py/service.py/config.py 是适配 Agent 的封装。Agent 用 mode 2 处理前 40 帧。

**smart_edit**：core/ 是完整子项目（有自己的 pyproject.toml、clients/、tools/、tests/）。通过 MCP 协议交互。

**worldengine**：core/ 是上游 WorldEngine（含 projects/、experiments/、data/）。外层 runner.py 等是适配层。

## configs/ — 运行配置

只有一个文件 configs/agent.json，控制所有**非密钥**运行配置：

    {
      "agent": { "max_steps": 20 },
      "llm": { "base_url": "...", "model": "..." },
      "orchestrator": { "decision_correction_attempts": 1 },
      "tools": { "modes": { "failure_analyzer": "real" } },
      "logging": { "root_dir": "logs" }
    }

**密钥和配置分离原则**：.env 只放密钥（OPENAI_API_KEY、DASHSCOPE_API_KEY、OSS_*），其余配置都在 configs/agent.json 或各组件自己的配置里。

## environments/ — Python 环境

各组件依赖冲突严重，所以**每个组件一个独立环境**：

| 环境 | 用途 |
|---|---|
| agent/ | Agent 主环境（最轻，只需 python-dotenv + openai） |
| video_retriever/ | 视频检索（也用来跑 pytest） |
| dggt/ | DGGT 重建主环境 |
| dggt_segformer/ | SegFormer 预处理 |
| smart-edit/ | SmartEdit |
| simengine/ | WorldEngine 仿真 |
| algengine/ | AlgEngine 训练 |

这些环境**不随源码分发**，需要协作者按 README 自行创建（或用软链接指向已有环境）。

## tests/ — 测试

pytest，47 项。覆盖配置加载、决策校验、工具结果信封、坐标转换、场景验证等。跑法：

    environments/video_retriever/.venv/bin/python -m pytest tests/

## 其他目录

- _dev_scripts_archive/：研发过程中的一次性脚本（对齐地图、渲染诊断、数据准备等），已归档，**不是产品代码**，一般不用管。
- logs/：每次 run 的日志，不随源码分发。
- tools/scripts/：当前为空，预留放工具辅助脚本。

## 数据怎么流转

一条典型生成链路的数据流：

    失败案例/需求
      └─tool01→ failure_analysis (失败类型/原因/风险)
      └─tool02→ scene_requirement (要造什么场景)
      └─tool03→ scene_plan (用什么基础视频+怎么编辑)
      └─tool04→ 检索到的基础视频
      └─tool06→ 三维高斯场景 (点云+相机参数)
      └─tool05→ 逐帧实例框 (instance_info)
      └─tool07→ edit_plan (编辑意图)
      └─tool08→ edit_config (具体轨迹偏移/资产)
      └─tool09→ 编辑后的场景+渲染视频
      └─tool10→ 验证结论 (通过/不通过)
           ├─通过→ 可选 tool13 仿真 → tool14 筛失败 → tool15 微调
           └─不通过→ tool11 决定怎么重试

工具之间不直接通信，全部通过 **AgentState 的历史记录**传递：每个工具的结果写进 state，下一个工具的参数由 LLM 根据 state 里的路径和结果来构造。

## 协作者上手路径

1. **先跑 mock 全链路**：把 configs/agent.json 里所有工具设 mock，配好 .env 的 LLM 密钥，跑 python -m agent.main "..."，看 Agent 怎么一步步决策。
2. **读懂编排层**：重点读 agent/orchestrator.py 的 system prompt 和 agent/main.py 的主循环。
3. **深入你负责的模块**：按上表找到对应 tool 和 mcp_server，读各自的 README。
4. **改代码前跑测试**：确认 pytest tests/ 全绿，改完再跑一次。
5. **切 real 验证**：把你负责的工具切成 real，配好对应环境和权重，单独验证后再跑整链路。

---

*本文档对应 adagent 开源版本（Apache 2.0）。组件级细节以各子目录 README 为准。*

