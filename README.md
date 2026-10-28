# adagent

面向自动驾驶的场景生成与评测 Agent。通过自然语言组织失败分析、视频检索、三维重建和场景编辑，将已有驾驶案例转换为可用于测试与数据扩充的新场景，并按任务需要调用闭环仿真与监督微调。

例如，面对“直行接近同车道低速前车时未及时减速”这一失败模式，adagent 可以分析已有案例、形成测试需求、检索合适的视频，再编辑前车的运动，构造需要减速跟停的测试场景。自车是否正确响应，需要在策略仿真中另行验证。

## 主要功能

### 失败分析与场景需求

结合 WorldEngine 官方指标、多相机图像、BEV 和运动记录，描述失败现象、比较候选原因，并保留证据与不确定性。分析结果用于生成场景需求，明确自车正在做什么、环境中有哪些关键参与者、发生了什么错误，以及期望测试哪种响应。

也可以直接从用户给出的场景需求或视频开始，无需每次都执行失败分析。

### 视频检索与实例发现

根据自然语言需求检索本地视频库，对候选片段进行重排并导出合适的视频。支持默认库和独立的 Waymo 前视角库；互联网检索通过独立 CLI 显式调用。

对视频中的车辆进行检测和跟踪，生成逐帧实例框与实例编号，供后续编辑选择目标。实例信息必须与被编辑的视频对应。

### 三维重建与场景编辑

使用 DGGT 从单目前视视频重建动态三维高斯场景，导出静态、动态及天空点云、相机参数和渲染视频。Agent 当前使用 mode 2，处理视频前40帧并沿用源视频帧率。

通过自然语言生成编辑计划与配置，支持车辆轨迹调整、旋转、缩放、删除、添加和替换，再由 DGGT 渲染编辑结果。已有实例的轨迹编辑需要匹配的坐标转换；添加或替换需要指定资产和比例。

### 场景验证与任务恢复

tool10 检查输出文件、编辑配置及渲染画面，返回验证报告。主 Agent 根据报告或执行错误直接决定重试、修改计划或结束；不再调用独立的恢复决策工具。视觉检查用于发现场景问题，不能代替官方碰撞责任和道路合规评分。

### 策略仿真与监督微调

通过 WorldEngine 运行闭环策略仿真，并按配置保存仿真产物与指标。自定义高斯场景需先完成格式转换；地图与指标所需数据必须单独准备。

支持对已转换的训练数据调用 AlgEngine 执行监督微调，保存训练日志和 checkpoint。当前 Agent 训练工具支持 ILFT，不提供 RL 训练。动态对象几何筛选工具不覆盖静态高斯和道路边界，也不等同于官方 NC/DAC。

## MCP 服务与适配组件

adagent 将以下能力统一接入工具调用流程。其中 SmartEdit 通过 MCP 服务交互，其他组件主要通过 JSON 子进程或项目脚本适配；并非所有组件都直接提供独立的 MCP 协议服务。

| 服务 / 组件 | 职责 | 主要输入 | 主要输出 |
|---|---|---|---|
| [失败分析 WFA](mcp_servers/worldengine_failure_analysis/README.md) | 复用官方结果，驱动 VLM 查看图像、查询数值并归因 | 场景 token、评测数据配置 | 现象、候选原因、证据、最终原因与不确定性 |
| [Sentry 视频检索](mcp_servers/sentry_video_search/README.md) | 视频建库、语义召回、候选重排与片段导出 | 场景描述、视频库 | 选中视频、候选分数与检索记录 |
| [DGGT](mcp_servers/dggt/core/README.md) | 动态高斯重建及实例编辑渲染 | 视频、可选编辑配置 | 高斯点云、相机参数、渲染与对比视频 |
| [SmartEdit](mcp_servers/SMART_EDIT_README.md) | 将自然语言编辑要求转换为服务工具调用并保存配置 | 编辑计划、实例信息或指定资产 | 编辑 JSON 或相机轨迹偏移配置 |
| [WorldEngine / AlgEngine](https://github.com/OpenDriveLab/WorldEngine) | 闭环策略仿真、评测及监督微调 | 场景、模型配置与权重、训练标注 | 仿真结果、配置启用的指标、训练日志与 checkpoint |

需求生成、场景构思、编辑计划和视觉验证由工具层提供，下一步选择及失败恢复由主 Agent 决策。当前共 13 个工具。各阶段按任务需要调用，不要求每次完整运行全部流程。

后端衔接代码集中在各组件的 adapter/，Agent 工具保留统一调用入口。组件与工具的对应关系、适配文件职责见 [MCP 组件说明](docs/MCP_SERVERS.md)。

## 典型工作流

```text
失败案例或场景需求
  → 场景构思与视频检索
  → 三维重建与实例发现
  → 编辑计划与配置生成
  → 场景编辑、渲染与验证
  → 按需执行策略仿真、失败筛选或监督微调
```

生成任务的主要产物是编辑配置、三维场景、相机参数和视频；失败分析保存独立的证据报告，仿真与训练保存各自的日志和结果。单工具测试成功不代表整条工作流已完成验证。

## 快速开始

使用 Python 3.10 或以上版本，从项目根目录创建 Agent 主环境：

```bash
python3 -m venv environments/agent/.venv
environments/agent/.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

在 .env 只填写 OPENAI_API_KEY（兼容 LLM_API_KEY）；模型、端点等非敏感配置填写 configs/agent.json 的 llm 块，兼容已有环境变量覆盖。然后按需准备各组件的运行环境、权重、数据及服务密钥，设置 `SMART_EDIT_ROOT`、`WORLDENGINE_ROOT` 等路径；主环境的安装不包含全部 real 工具依赖。

```bash
environments/agent/.venv/bin/python -m agent.main \
  "使用 /absolute/path/to/driving.mp4，设计并生成一个同车道前车减速的测试场景"
```

`configs/agent.json` 控制每个工具的 `mock` / `real` 模式和 Agent 最大执行步数。mock 返回接口兼容的占位结果，不生成真实视频或点云；Agent 编排仍需要模型接口。real 模式需要相应组件的依赖与输入数据就绪。

## 纯文本 LLM 工具

需求生成、场景构思、编辑计划这 3 个工具采用统一的 Schema、独立提示词、非敏感配置和执行器；
mock/real 共用输入输出校验。目录规范、模型角色和逐工具测试命令见 [LLM 工具开发规范](docs/LLM_TOOLS.md)。

## 密钥与模型分工

- Agent 主模型：`OPENAI_API_KEY`/`OPENAI_BASE_URL`/`OPENAI_MODEL`（或旧 `LLM_*`），例 `qwen3.5-plus`。
- tool10 视觉验证：默认 `kimi-k3`，base 取 `LLM_BASE_URL`，密钥回退 `LLM_API_KEY`；可用 `SCENE_VALIDATOR_MODEL` 覆盖。
- tool04 视频检索：召回/重排用 DashScope `qwen3-vl-embedding`，需真实 `DASHSCOPE_API_KEY`（中转站不支持该端点）。
- OSS（可选）：`OSS_*` 供重排产物上传；不入库。

Agent 日志保存在 `logs/`，组件产物写入各自的 output、outputs 或 runs 目录。模型接口可能接收场景文本、图片或视频，具体范围见组件说明。密钥、模型权重、运行环境、检索索引和大体积数据不随源码分发。

## 开发文档

- [项目架构与目录分工](docs/ARCHITECTURE.md)
- [Agent 文件、configs 和 logs 说明](docs/AGENT_FILES.md)
- [tools 目录与 13 个工具说明](docs/TOOLS.md)
- [后端组件、工具映射与适配层说明](docs/MCP_SERVERS.md)
- [纯文本 LLM 工具规范](docs/LLM_TOOLS.md)

## 许可

DGGT、WorldEngine 及其他第三方依赖遵循各自的原始许可与声明。Agent 所需的 DGGT 编辑扩展不属于官方原版能力，官方 README 用于说明上游项目

## nuPlan 检索与 ADSceneGraph 接入

新增有标注nuPlan库与工具6原生重建入口；原DGGT调用结构保留。配置、数据准备和使用命令见 [接入说明](docs/NUPLAN_INTEGRATION.md)，分支/密钥/产物管理见 [协作规范](docs/GITHUB_COLLABORATION.md)。数据和密钥仅存本地。
