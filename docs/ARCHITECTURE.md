# adagent 项目结构文档

> 面向协作者的项目导览。逐目录说明每个模块的职责、关键文件和相互关系。
> 阅读顺序建议：先读本文档建立整体认识，再按需深入对应目录的 README。

服务提供的工具、Agent 衔接文件及适配层整理建议见 [MCP 组件与工具接入说明](MCP_SERVERS.md)。

## 项目做什么

adagent 是一个**面向自动驾驶数据生成与评测的 Agent 系统**。给定一个失败模式（例如"直行时接近同车道低速前车未及时减速"），它能自动完成一整条链路：

分析失败原因 → 生成场景需求 → 检索合适的基础视频 → 三维重建 → 编辑场景（改前车轨迹、加 NPC 等）→ 渲染验证 → 闭环仿真评测 → 筛选失败案例 → 监督微调驾驶策略。

核心思想是：用 LLM 做编排和决策，把视频检索、三维重建（DGGT）、场景编辑（SmartEdit）、闭环仿真（WorldEngine）、策略训练（AlgEngine）这些异构能力统一成一套可调用的"工具"，由 Agent 按需串联。

## 整体架构

系统分三层，职责严格分离：

    ┌─────────────────────────────────────────────┐
    │  agent/        编排层   LLM 决定下一步调哪个工具    │
    ├─────────────────────────────────────────────┤
    │  tools/        工具层   13 个标准 Tool，各有        │
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
| tools/ | 13 个标准工具，每个有 mock/real 两实现 | 新增/修改工具时 |
| mcp_servers/ | 外部能力组件（检索、重建、编辑、仿真、训练） | 接入新组件时 |
| configs/agent.json | 运行配置：工具 mock/real 模式、LLM、步数上限 | 改运行行为时 |
| environments/ | 各组件独立的 Python 环境（不随源码分发） | 仅本地配置 |
| tests/ | pytest 测试 | 改代码必须同步 |
| _dev_scripts_archive/ | 一次性研发脚本归档 | 一般不动 |
| .env / .env.example | 密钥（**只放密钥**，配置在 configs） | 仅本地 |
| logs/ | 运行日志（每次 run 一个目录） | 不随源码分发 |

## agent/ — Agent 编排核心

agent 逐文件的职责与调用关系，以及 configs、logs 的用途，见 [Agent 核心与运行目录说明](AGENT_FILES.md)。

这是系统的大脑。每个文件职责单一：

| 文件 | 职责 | 关键点 |
|---|---|---|
| main.py | 执行入口和主循环 | Agent.run(task)：循环 {LLM 决策→执行工具→更新状态}，直到 finish 或达到 max_steps |
| orchestrator.py | LLM 决策器 | 全局 system prompt 加 available_tools 与 current_state，让 LLM 输出 JSON 决策（call_tool/finish/stop），带纠错重试 |
| executor.py | 工具执行器 | 按 configs/agent.json 里该工具的 mock/real 模式调用对应实现，并校验结果信封 |
| state.py | 运行时状态 | AgentState 记录任务、历史（决策+工具结果）、当前阶段、生命周期；to_orchestrator_dict() 会压缩逐帧检测结果以免撑爆 prompt |
| schemas.py | 数据校验 | 校验 LLM 决策格式和工具结果信封是否合法 |
| llm_client.py | LLM 客户端 | 封装 OpenAI 兼容 API；密钥从 .env 读，base_url/model 从 configs/agent.json 的 llm 块读（环境变量优先） |
| config.py | 配置加载 | 加载并强校验 configs/agent.json（缺字段/未知工具/非法模式都会报错） |
| tool_descriptions.py | 工具清单 | 只汇总各工具目录的 description.py；三个纯文本 LLM 工具的参数说明由各自 schemas.py 自动生成 |
| logger.py | 结构化日志 | 每次 run 建一个目录，记录决策、工具调用、错误、最终状态 |

### 编排层的关键设计

**1. 编排器选择工具，执行器负责调用。** 全局 system prompt 只保留身份、JSON 决策格式和证据等全局规则。工具用途与领域约束位于各工具目录的 description.py，通过 available_tools 提供给编排器。tool02/03/07 的内部提示词独立放在 prompt.md，输入输出由 schemas.py 和公共运行器校验；主 Agent 不需要读取其完整内部提示词。修改参数时应改 Schema，修改内部推理规则时应改该工具 prompt.md。

**2. 状态是 LLM 的唯一信息来源。** LLM 每步看到的 current_state 来自 state.to_orchestrator_dict()。大体积数据（如逐帧检测框）会被压缩成采样帧，完整数据写进日志文件，prompt 里只放路径。这是控制 context 长度的关键手段。

**3. 决策有校验和纠错。** LLM 输出先过 schemas.validate_decision，非法则按 decision_correction_attempts 次数把错误信息反馈给 LLM 让它重试。

### 主 Agent 直接负责恢复

恢复规则写在 agent/orchestrator.py 的 system prompt 内，不再请求独立恢复模型。
主 Agent 读取工具状态、原始需求、issues/evidence 和成功产物，直接输出下一次 call_tool：
瞬时错误最多原参数重试一次；参数错误先修正参数；场景错误修改受影响的计划或配置，
执行后续渲染并重新验证，复用未受影响的产物。

status=success 只说明工具执行完成；生成场景必须取得当前渲染对应的显式 passed=true。
只完成场景生成的任务，在验证通过后 finish，不擅自运行仿真或训练。
缺凭据、环境或权重且没有已知补救手段，或恢复选项耗尽时，输出 stop 并说明未完成目标。
stop 由主循环记录为 error，而非成功完成；全局最大步数仍限制整个执行过程。
这些重试与修复选择由提示词指导，当前没有新增独立恢复策略引擎。

验证命令（从项目根目录执行）：

    environments/agent/.venv/bin/python -m tools.scripts.smoke_agent_recovery --mode mock
    environments/agent/.venv/bin/python -m tools.scripts.smoke_agent_recovery --mode real
    environments/agent/.venv/bin/python -m tools.scripts.smoke_agent_recovery --mode real --suite integration

第一条使用脚本化模型，后两条调用配置的真实主模型。decisions 测试五种合成状态：
视觉失败、验证通过、首次超时、重复超时和缺权重。integration 使用真实编排模型与 mock 工具，
首次视觉验证被注入失败，要求修正并重渲染、重验后才能通过测试。
这些测试验证控制流，不证明真实 DGGT 编辑质量或模仿学习收益。

## tools/ — 工具层（13 个 Tool）

每个 Tool 的用途、输入输出和调用功能见 [tools 目录与工具说明](TOOLS.md)。

每个 tool 是一个独立目录。所有工具的编排说明位于 description.py。
三个纯文本 LLM 工具（tool02/03/07）统一采用：

    toolNN_xxx/
      description.py  # 主 Agent 看到的用途；参数说明从 Schema 生成
      schemas.py      # 输入、模型回复、最终输出与业务约束
      prompt.md       # 工具内部 LLM 的任务指令
      config.json     # 模型角色、提示词版本、超时、输出长度、校验重试
      real.py         # 绑定契约，调用公共 LLM 运行器
      mock.py         # 通过相同契约的确定性模拟实现
      __init__.py

这三个工具不再提供 tool.py 旧入口，调用统一使用注册表或明确的 real/mock 模块。公共实现是 tools/llm_runtime.py。完整约定与验证命令见 [LLM 工具开发规范](LLM_TOOLS.md)。
其他后端工具的 real.py 负责调用对应组件 adapter 并包装标准返回格式；环境选择、进程启动和后端格式转换集中在 mcp_servers/<组件>/adapter/。mock 仍由各工具自己的 tool.py/mock.py 提供。

tools/registry.py 是总调度：维护 MOCK_TOOL_REGISTRY 和 REAL_TOOL_REGISTRY 两张表，call_tool(name, args, mode) 按模式分发。**新增工具要在这里登记。**

### 13 个工具按链路顺序

| # | 名称 | 干什么 | 属于阶段 |
|---|---|---|---|
| 01 | failure_analyzer | 以 WorldEngine 场景 token 调用 WFA，保留指标、图像证据、归因和不确定性 | 失败分析 |
| 02 | scene_requirement_generator | 由失败分析生成"要造什么场景"的需求 | 需求生成 |
| 03 | scene_conceiver | 设计高层场景方案：需要什么样的基础视频、怎么编辑 | 场景构思 |
| 04 | video_retriever | 按自然语言需求检索本地视频库，重排并导出片段 | 数据检索 |
| 05 | instance_discovery | 对视频做车辆检测跟踪，生成逐帧实例框和实例编号 | 实例发现 |
| 06 | reconstruction | 用 DGGT 把单目前视视频重建成动态三维高斯场景 | 三维重建 |
| 07 | edit_plan_builder | 把自然语言编辑意图转成结构化编辑计划 | 编辑规划 |
| 08 | edit_config_generator | 由编辑计划生成具体编辑配置（轨迹偏移、资产等） | 编辑配置 |
| 09 | scene_editor | 执行编辑，调 DGGT 渲染编辑后的场景 | 场景编辑 |
| 10 | scene_validator | 检查输出文件和渲染画面是否符合需求（VLM 视觉验证） | 结果验证 |
| 13 | policy_runner | 用 WorldEngine 跑闭环策略仿真 | 闭环仿真 |
| 14 | failure_case_filter | 从仿真结果里筛出失败案例 | 失败筛选 |
| 15 | policy_trainer | 用 AlgEngine 对筛选数据做监督微调（ILFT，**不含 RL**） | 策略训练 |

> 编号保留现有目录命名，当前为 tool01–tool10、tool13–tool15。独立 tool11 已移除，其恢复决策规则位于 agent/orchestrator.py 的主提示词中；tool12 不作为独立工具提供。

### mock / real 双模式

每个工具都有两种实现，由 configs/agent.json 的 tools.modes 控制：

- **mock**：返回接口兼容的占位结果，不真正跑模型/渲染/仿真。用于快速验证 Agent 编排逻辑、调 prompt、跑测试。
- **real**：纯文本 LLM 工具调用配置的模型端点；服务适配工具调用对应组件，需要相应环境、权重、数据就绪。

**给协作者的建议：改编排逻辑时先把相关工具设成 mock，跑通了再切 real。**

## mcp_servers/ — 后端组件与适配层

Agent 工具通过各组件的 adapter/ 调用后端。SmartEdit 使用 MCP stdio，其余主要通过 Python 接口或 JSON 子进程交互。

| 组件 | Agent 工具 | adapter 中的主要文件 | 核心能力 |
|---|---|---|---|
| worldengine_failure_analysis/ | tool01 | service.py | 按场景 token 调用 WFA 失败取证。 |
| sentry_video_search/ | tool04 | client.py、runner.py | 调用本地检索；core/ 保存独立检索实现，adapter/ 对接 Agent。 |
| dggt/ | tool05/06/09 | instances.py、client.py、editing.py、coordinates.py、runner.py、service.py、config.py | 实例发现、mode 2 重建和编辑渲染。 |
| smart_edit/ | tool08 | service.py、client.py、config.py、prompts/ | 驱动内部 15 个 MCP 子工具生成配置；tool07 只生成编辑计划，不直接调用此服务。 |
| worldengine/ | tool13/14/15 | client.py、runner.py、service.py、scene_conversion.py、evaluation.py、training.py、config.py | 仿真、动态对象碰撞筛选及 ILFT 监督微调。 |

典型调用链：tools/toolXX/real.py → mcp_servers/组件/adapter/service.py → core/；adapter 内部可进一步调用客户端、runner 或其他模块。

五个项目的业务代码全部位于各自 core/；Agent 封装全部位于 adapter/。单次运行文件位于 runtime/<run_id>/，权重、环境和长期数据沿用已配置的路径。

SmartEdit 内部配置生成指令已独立放在 smart_edit/adapter/prompts/。WFA 原生配置模板在 core/configs/，Agent 接入配置在 adapter/config.py，本地凭据仍在 .secrets/。视频检索索引不随源码分发。

每个文件的职责、服务子工具清单与当前验证进度见 [MCP 组件与工具接入说明](MCP_SERVERS.md)。

## configs/ — 运行配置

目前只有 configs/agent.json，控制 **Agent 层的非敏感运行配置**；各后端自己的配置仍放在对应组件中：

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
           └─不通过→ 主 Agent 直接修改计划/配置，重渲染后重验

工具之间不直接通信，全部通过 **AgentState 的历史记录**传递：每个工具的结果写进 state，下一个工具的参数由 LLM 根据 state 里的路径和结果来构造。

## 协作者上手路径

1. **先跑 mock 全链路**：把 configs/agent.json 里所有工具设 mock，配好 .env 的 LLM 密钥，跑 python -m agent.main "..."，看 Agent 怎么一步步决策。
2. **读懂编排层**：重点读 agent/orchestrator.py 的 system prompt 和 agent/main.py 的主循环。
3. **深入你负责的模块**：按上表找到对应 tool 和 mcp_server，读各自的 README。
4. **改代码前跑测试**：确认 pytest tests/ 全绿，改完再跑一次。
5. **切 real 验证**：把你负责的工具切成 real，配好对应环境和权重，单独验证后再跑整链路。

---

*本文档对应 adagent 开源版本（Apache 2.0）。组件级细节以各子目录 README 为准。*

