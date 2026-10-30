# MCP 组件、Agent 工具与适配文件说明

本文按当前已归拢的代码说明：每个后端支持哪些 Agent 工具、哪些文件负责衔接，以及修改功能时应该看哪里。

## 三层代码怎么分工

```text
agent/                         根据任务和历史结果选择工具
  → tools/toolXX_*/real.py      接收 Agent 参数，调用适配接口，返回工具标准格式
  → mcp_servers/组件/adapter/service.py  统一接入入口
  → adapter 内部模块           选择环境、调用后端、转换数据、整理结果
  → 组件/core/                 实际完成检索、重建、编辑、取证、仿真或训练
```

Agent 工具的说明仍位于各工具的 `description.py`；适配层不负责替主 Agent 选择任务步骤。核心算法与模型实现也没有因为这次归拢而搬入 adapter。

当前共有 13 个 Agent 工具。其中 SmartEdit 后端另外注册了 15 个 MCP 子工具，供 tool08 的内部客户端按需调用，两者分开统计。

各 Tool 的中文用途、主要输入输出和 tools 目录文件职责见 [13 个 Tool 说明](TOOLS.md)。

## 每个组件支持哪些 Agent 工具

| 组件目录 | 提供的能力 | Agent 工具 | 实际调用方式 |
|---|---|---|---|
| `worldengine_failure_analysis/` | 读取官方评测结果，结合图像和数值分析失败原因。 | tool01 `failure_analyzer` | Python 调用适配接口，内部启动 WFA 命令行。 |
| `sentry_video_search/` | 视频建库、召回、重排、片段导出。当前 Agent 工具使用本地库。 | tool04 `video_retriever` | 适配客户端启动检索环境，通过 JSON 调用 runner。 |
| `dggt/` | 实例发现、三维重建、应用编辑配置并渲染。 | tool05 `instance_discovery`；tool06 `reconstruction`；tool09 `scene_editor` | 实例发现由 adapter 启动核心脚本；重建和编辑共用 JSON runner 与推理 service。 |
| `smart_edit/` | 将编辑计划转成可保存的配置。 | tool08 `edit_config_generator` | adapter 客户端连接模型与 SmartEdit MCP 服务，调用内部子工具。 |
| `worldengine/` | 闭环仿真、动态对象碰撞筛选、模仿学习监督微调。 | tool13 `policy_runner`；tool14 `failure_case_filter`；tool15 `policy_trainer` | 仿真经 JSON runner/service；筛选、训练分别调用 adapter 中的评估和训练入口。 |

`mcp_servers/` 是后端组件的统一存放目录。本项目当前只有 SmartEdit 链路使用 MCP stdio 协议；其余链路通过 Python 接口或 JSON 子进程调用。归拢 adapter 没有改变通信协议。

以下 4 个工具在工具层实现，不通过上述组件适配接口：

| 工具 | 实现方式 |
|---|---|
| tool02 `scene_requirement_generator` | 通过 `tools/llm_runtime.py` 调用模型生成需求。 |
| tool03 `scene_conceiver` | 通过同一运行器生成场景构思。 |
| tool07 `edit_plan_builder` | 通过同一运行器生成编辑计划，不直接调用 SmartEdit 服务。 |
| tool10 `scene_validator` | 在自己的 real 实现中检查文件，调用视觉模型或指定的本地评估器。 |

## WFA：失败分析

```text
tools/tool01_failure_analyzer/real.py
  → worldengine_failure_analysis/adapter/service.py：analyze_scene()
  → worldengine_failure_analysis/core/run.py
  → scripts/ 中的官方结果读取、证据整理、VLM 和数值查询
```

| 文件 | 主要做什么 |
|---|---|
| [tool01/real.py](../tools/tool01_failure_analyzer/real.py) | 接收场景 token，将 WFA 结果映射成 Agent 下游使用的 failure_analysis，并保留取证详情。 |
| [adapter/service.py](../mcp_servers/worldengine_failure_analysis/adapter/service.py) | 选择解释器，检查配置和凭据，启动取证，读取结果及报告。 |
| [run.py](../mcp_servers/worldengine_failure_analysis/core/run.py) | WFA 自身的命令行工作流入口。 |
| `scripts/` | WFA 本身的证据分析实现。 |

输入需要 WorldEngine 场景 token 和对应评测数据。原生配置模板在 `core/configs/`，接入配置在 `adapter/config.py`；凭据通过 WFA_SECRET_FILE 或组件根目录的本地 .secrets/vlm.json 提供。

## Sentry：视频检索

```text
tools/tool04_video_retriever/real.py
  → sentry_video_search/adapter/service.py：retrieve_video()
  → adapter/client.py：启动检索环境
  → adapter/runner.py
  → adapter/service.py：search_local()
  → core/service.py：search_local()
  → 视频召回、重排和片段导出
```

| 文件 | 主要做什么 |
|---|---|
| [tool04/real.py](../tools/tool04_video_retriever/real.py) | 调用检索适配接口，包装 Agent 工具结果。 |
| [adapter/client.py](../mcp_servers/sentry_video_search/adapter/client.py) | 检查需求与视频库选择，启动检索环境并处理 JSON 返回。 |
| [adapter/runner.py](../mcp_servers/sentry_video_search/adapter/runner.py) | 接收进程请求，将需求交给本地检索接口。 |
| [config.py](../mcp_servers/sentry_video_search/core/config.py) | 定义索引、视频库、输出位置等组件配置。 |
| [service.py](../mcp_servers/sentry_video_search/core/service.py) | 检索组件的业务入口，组织建库、检索和导出。 |

检索自身的 config.py、service.py、pipeline/、vendor/ 和 reranker.py 全部位于 core/，独立 CLI 不依赖 AdAgent。

service 还提供 `index_local()`、`search_internet()`、`retrieve_video()` 等接口；当前 tool04 只使用 `search_local()`，不会自动转为互联网检索。adapter 中同名的 `retrieve_video()` 是 Agent 本地检索调用包装，应按模块路径区分。

## DGGT：实例发现、重建和编辑

```text
tool05/real.py → adapter/service.py → adapter/instances.py → core/tools/generate_yolo_instance_info.py

tool06/real.py → adapter/service.py → adapter/client.py
                                      → adapter/runner.py
tool09/real.py → adapter/service.py → adapter/editing.py     → adapter/service.py
                                        → core 的抽帧、预处理、mode 2 推理与渲染
```

| 文件 | 主要做什么 |
|---|---|
| [tool05/real.py](../tools/tool05_instance_discovery/real.py)、[tool06/real.py](../tools/tool06_reconstruction/real.py)、[tool09/real.py](../tools/tool09_scene_editor/real.py) | 保留各自 Agent 工具入口，调用对应适配函数并包装结果。 |
| [adapter/instances.py](../mcp_servers/dggt/adapter/instances.py) | 调用 YOLO/跟踪脚本，读取实例 ID 和逐帧检测框。 |
| [adapter/client.py](../mcp_servers/dggt/adapter/client.py) | 处理重建请求，通过 JSON runner 启动后端。 |
| [adapter/editing.py](../mcp_servers/dggt/adapter/editing.py) | 检查编辑输入，按需转换轨迹坐标，再调用同一个 DGGT runner。 |
| [adapter/coordinates.py](../mcp_servers/dggt/adapter/coordinates.py) | 将 SmartEdit 局部轨迹偏移转为 DGGT 所需的逐帧世界坐标偏移。 |
| [adapter/runner.py](../mcp_servers/dggt/adapter/runner.py) | 接收重建或编辑的 JSON 请求。 |
| [adapter/service.py](../mcp_servers/dggt/adapter/service.py) | 组织视频预处理、推理、编辑渲染及产物收集。 |
| [adapter/config.py](../mcp_servers/dggt/adapter/config.py) | 管理核心源码、解释器、权重及运行目录。 |

当前 Agent 使用 mode 2、40 帧。SmartEdit 负责生成编辑配置，DGGT 负责执行配置并渲染。

核心源码仍在 `dggt/core/`，包含项目需要的编辑定制；不能将其描述为完全未经修改的官方原版。新任务的输入、输出和日志统一位于组件 runtime/<run_id>/ 的 input/、output/、logs/。旧运行目录已清理。

## SmartEdit：编辑配置生成

```text
tools/tool08_edit_config_generator/real.py
  → smart_edit/adapter/service.py：generate_edit_config()
  → adapter/client.py：连接模型，驱动 MCP 调用
  → core/smart_edit/server.py
  → core/smart_edit/tools/definitions.py 和 handlers.py
  → 保存配置，adapter/service.py 校验并返回
```

| 文件 | 主要做什么 |
|---|---|
| [tool08/real.py](../tools/tool08_edit_config_generator/real.py) | 调用配置生成适配接口，返回标准工具结果。 |
| [adapter/service.py](../mcp_servers/smart_edit/adapter/service.py) | 传递编辑计划、实例、资产和输出路径，检查生成的配置是否可供 DGGT 使用。 |
| [adapter/client.py](../mcp_servers/smart_edit/adapter/client.py) | 获取 MCP 子工具清单，让内部模型执行配置操作，并确认实际调用保存工具。 |
| [adapter/config.py](../mcp_servers/smart_edit/adapter/config.py) | 解析核心目录、解释器，以及项目模型配置和密钥来源。 |
| [prompts/rigid_config.md](../mcp_servers/smart_edit/adapter/prompts/rigid_config.md) | 车辆等刚体配置生成的内部模型指令。 |
| [prompts/ego_trajectory_offsets.md](../mcp_servers/smart_edit/adapter/prompts/ego_trajectory_offsets.md) | 自车渲染轨迹偏移配置的内部模型指令。 |
| [core/smart_edit/server.py](../mcp_servers/smart_edit/core/smart_edit/server.py) | 提供 MCP 工具和资源接口，属于 SmartEdit 服务本身。 |
| [definitions.py](../mcp_servers/smart_edit/core/smart_edit/tools/definitions.py)、[handlers.py](../mcp_servers/smart_edit/core/smart_edit/tools/handlers.py) | 声明子工具参数并执行配置操作，仍保留在核心实现中。 |

当前 MCP 服务实际注册 15 个子工具：

| 子工具 | 作用 |
|---|---|
| `list_instances` | 读取实例清单，确认实例 ID。 |
| `list_assets` | 查询可用三维资产。 |
| `start_config` | 开始新配置，关联实例清单。 |
| `edit_rigid_trajectory` | 编辑已有实例的轨迹。 |
| `rotate_rigid_instance` | 旋转已有实例。 |
| `scale_rigid_instance` | 缩放已有实例。 |
| `add_rigid_asset` | 添加车辆资产。 |
| `remove_rigid_instance` | 标记删除已有实例。 |
| `replace_rigid_instance` | 用指定资产替换已有实例。 |
| `remove_added_rigid_asset` | 撤销草稿中新增的资产。 |
| `set_ego_trajectory_offsets` | 设置渲染相机的相对轨迹偏移。 |
| `preview_config` | 查看草稿。 |
| `save_config` | 保存刚体编辑配置。 |
| `save_ego_trajectory_offsets` | 保存自车渲染轨迹偏移。 |
| `reset_config` | 清空草稿。 |

计数依据是服务实际注册清单，源码中未注册的工具定义不计入。配置保存成功后还需调用 DGGT 执行和渲染；相机轨迹偏移也不等于闭环策略已经改变制动或规划决策。

## WorldEngine：仿真、筛选和模仿学习

```text
tool13/real.py → adapter/service.py → adapter/client.py → adapter/runner.py
                                    → adapter/service.py → 仿真脚本

tool14/real.py → adapter/service.py → adapter/evaluation.py → 动态对象碰撞筛选

tool15/real.py → adapter/service.py → adapter/training.py → AlgEngine 监督微调脚本
```

| 文件 | 主要做什么 |
|---|---|
| [tool13/real.py](../tools/tool13_policy_runner/real.py)、[tool14/real.py](../tools/tool14_failure_case_filter/real.py)、[tool15/real.py](../tools/tool15_policy_trainer/real.py) | 保留 Agent 工具入口，调用仿真、筛选或训练的适配接口。 |
| [adapter/client.py](../mcp_servers/worldengine/adapter/client.py) | 检查仿真输入，启动 JSON runner，并检查场景运行报告。 |
| [adapter/runner.py](../mcp_servers/worldengine/adapter/runner.py) | 接收仿真 JSON 请求。 |
| [adapter/service.py](../mcp_servers/worldengine/adapter/service.py) | 准备环境和命令，按需转换 DGGT 场景，运行仿真并收集结果。 |
| [adapter/scene_conversion.py](../mcp_servers/worldengine/adapter/scene_conversion.py) | 为 DGGT 编辑产物整理场景转换输入；由仿真 service 使用。 |
| [adapter/evaluation.py](../mcp_servers/worldengine/adapter/evaluation.py) | 使用场景数据与实际运行轨迹筛选动态对象碰撞。 |
| [adapter/training.py](../mcp_servers/worldengine/adapter/training.py) | 组装 AlgEngine 监督微调命令，保存训练日志并查找生成的 checkpoint。 |
| [adapter/config.py](../mcp_servers/worldengine/adapter/config.py) | 定义核心目录、仿真脚本、超时、输出及日志位置。 |

核心仍在 `worldengine/core/`，也可以由 `WORLDENGINE_ROOT` 指定其他安装位置。场景准备代码已归入 WorldEngine adapter，后端不再反向导入 tools 根目录的旧脚本。

当前训练入口仅支持 `training_mode="ilft"`，不提供 RL 训练。tool14 的筛选是项目自带的动态对象几何评估，不覆盖静态高斯、道路边界，也不等同于官方碰撞责任或道路合规指标。

## 这次归拢后的边界

- 工具名称、mock/real 选择和标准返回格式保持不变；各工具的详细说明仍在自身目录。
- 视频检索、实例发现、重建、配置生成、编辑、仿真、失败筛选和训练的后端调用集中到对应 adapter。
- tool01 的 WFA 结果映射保留在工具层，因为它定义了 Agent 下游使用的失败分析契约。
- SmartEdit 内部的两类模型指令独立保存在 adapter/prompts；主 Agent 的全局提示词仍在 `agent/orchestrator.py`。
- Sentry 和 WFA 的业务实现已整体迁入 core/；原生项目不反向导入 AdAgent，独立入口已验证。
- 旧的外层 runner/service 模块、根目录 SmartEdit 客户端，以及 tools 根目录的场景准备入口已迁移，不保留旧路径兼容文件。直接导入这些模块的外部脚本需要改用新路径。
- 本次没有移动或复制环境、权重和数据，也没有修改第三方核心算法。

修改 Agent 看到的参数和说明，先看 tools；修改如何启动后端或如何转换结果，先看 adapter；修改实际算法，再看组件核心。

## 当前验证进度

本轮结构迁移验证：主框架、五个适配层、WFA 原生测试及结构边界测试共 120 项通过；SmartEdit 原生环境另有 17 项通过，包含实际 MCP stdio 交互。
视频检索通过 Agent 入口完成一次真实召回、重排序和片段导出，匹配分数 0.854686。WFA 和检索均通过“提取 core 到临时目录后运行独立 CLI”的测试。
DGGT 检查了环境、权重路径和调用接口；WorldEngine 检查了启动脚本和接口。本轮未重新执行 GPU 重建、仿真或训练。

在已安装 requirements-test.txt 依赖的环境中运行：

```bash
python -m pytest -q
```

pytest.ini 限定主框架和服务集成测试范围，避免误收集上游算法工程的大型测试集。SmartEdit 原生测试使用其自身环境：

```bash
PYTHONPATH=mcp_servers/smart_edit/core environments/smart-edit/.venv/bin/python -m unittest discover -s mcp_servers/smart_edit/core/tests -v
```

相关文档：[Agent 核心与运行目录说明](AGENT_FILES.md)、[项目架构](ARCHITECTURE.md)、[纯文本 LLM 工具规范](LLM_TOOLS.md)。
