# adagent

基于 LLM 的自动驾驶场景需求生成、检索、验证与后训练框架。主 Agent 根据用户任务和工具返回结果选择下一步，工具通过统一输入输出契约调用后端。

## 当前状态

当前启用 9 个工具，按 01–09 编号，涵盖失败分析、需求生成、检索、快速粗规划、攻防编辑、策略评测、失败筛选和后训练。
攻击车与自车编辑后端已接入；独立视觉验收工具已直接删除，没有迁入其他工具。
独立原生回放后端保留，但没有注册为主 Agent 工具。编辑结果的 `simulation_input` / `ilft_training_data` 若为 null，须读取导出报告，不能用原始场景代替编辑结果。

## 运行环境

从项目根目录创建 Agent 环境（Python 3.10+，需安装对应版本的 venv 支持）：

```bash
python3 -m venv environments/agent/.venv
environments/agent/.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

`.env` 只填写密钥；`configs/api.json` 管理模型、端点、超时和思考强度。主 Agent 使用 `active_profile` 指定的 API；各工具默认跟随，也可通过 `tool_profiles` 和 `tool_efforts` 分别指定模型和思考强度。阿里接口另用于视频检索嵌入与重排序，检索重排序还需要配置 OSS 凭证。

可在 `.env` 同时保存多把 key，再用命令统一选择模型与 API。完整步骤、配置示例和兼容性说明见 [API 配置与统一切换](docs/API_CONFIGURATION.md)。

```bash
python scripts/api_profile.py list
python scripts/api_profile.py show
```

```bash
environments/agent/.venv/bin/python -m agent.main \
  "根据给定失败类型构思三个底场景方案，并检索一个已绑定标注与高斯资产的场景"
```

`configs/agent.json` 控制最大步数及各工具 mock/real 模式。Agent 编排需要模型接口；mock 工具不生成真实产物。
各 real 后端需独立准备环境、数据、权重和索引，安装主环境不包含 GPU 服务依赖。

## 工具与后端

| 编号 | 工具 | 职责 |
|---|---|---|
| 01 | failure_analyzer | 分析已有失败场景与证据 |
| 02 | scene_instance_generator | 生成目标场景、全局自车目标与三个候选 |
| 03 | video_retriever | 检索并强制绑定多相机场景、标注和资产 |
| 04 | edit_planner | 快速判断可编辑性，给出针对底场景的粗计划 |
| 05 | attack_vehicle_editor | 规划、搜索、编辑并冻结攻击车场景 |
| 06 | ego_trajectory_editor | 规划、编辑并复核正确的自车专家轨迹 |
| 07 | policy_runner | 驾驶策略原生闭环仿真与正式评分；保留旧 PDM 挑战模式 |
| 08 | failure_case_filter | 读取官方评分并筛选失败案例 |
| 09 | policy_trainer | 执行 ILFT/RLFT 并输出模型与更新审计 |

主 Agent 按任务选择调用顺序。05 输出通过几何与任务检查的冻结攻击场景，06 在该冻结场景上编辑自车专家轨迹。05 默认开启 `vadv2_self_test`，原生策略自检默认最多运行 3 次，跨候选和对象重选共享额度。显式传入 `vadv2_self_test=false` 可仅做几何和任务检查。旧 PDM 自检由独立验证脚本和 `policy_runner` 执行。
05 不要求生成中间攻击视频。06 输出专家轨迹、检查，以及实际生成的视频；关闭渲染时视频字段为空。是否导出原生仿真包须检查 `simulation_input` 和导出报告；当前专家 ILFT 标注出口尚未接通，`ilft_training_data` 为 null。
07 是独立策略评测工具，与05可选自测复用原生后端。05未自测时不能声称 VADv2 已碰撞；已有证据只可用于完全匹配的场景与轨迹版本。碰撞攻击依据模型接管后的实际目标车体接触；PDM零分、道路违规、插值推测或接管前接触不能代替成功证据。
07 原生评测默认使用4帧历史、16帧场景预算，在2 Hz场景中覆盖1.5–7.5秒；`full_rollout_pdm` 表示覆盖本次配置的 rollout，不保证覆盖源场景全长。显式调整帧预算时评分范围随之变化，须读取返回的 `evaluation_scope`。VADv2单次预测与密集奖励窗口独立，不能把该评分直接等同于标准4秒NAVSIM分数。
场景帧数须与历史/未来帧预算一致；原生编辑场景由适配层启用与回放共用的坐标修正，并将实际帧数传入 AlgEngine 监视器。

四个后端组件保存在 `mcp_servers/`，均采用 `core/`、`adapter/`、`runtime/` 分层：

- `sentry_video_search`：检索、入库与重排序，详见[多相机场景绑定](mcp_servers/sentry_video_search/README.md)。
- `worldengine_failure_analysis`：失败取证、图像及数值分析。
- `scene_editing`：攻击车与自车编辑阶段接口，详见[编辑接口](docs/EDITING_INTERFACES.md)。
- `worldengine`：仿真、官方评分读取及训练。

这些组件通过 Python 或 JSON 子进程调用；目录名称不表示每个组件都提供 MCP 协议服务。

## 多相机场景绑定

tool03 real 要求索引源视频带有效 `.scene.json`。结果包含 `scene_ref`：场景身份、相机清单、同步图像元数据、标注与高斯资产路径。未绑定直接报错；未下载文件明确标记缺失。
tool04 根据实际场景做可编辑性检查及粗规划，tool05/06 再执行详细轨迹搜索和编辑。

## 验证与测试

```bash
python3 -m venv environments/test/.venv
environments/test/.venv/bin/python -m pip install -r requirements-test.txt
environments/test/.venv/bin/python -m pytest -q
environments/agent/.venv/bin/python -m tools.scripts.smoke_llm_tools --mode mock
```

测试依赖见 `requirements-test.txt`。接口测试不能代替真实场景质量、闭环表现或后训练收益验证。
Agent 记录保存在 `logs/`；服务产物保存在 `mcp_servers/<service>/runtime/<run_id>/`。
可读结果可按案例整理到 `outputs/<case>/<run>/`；后续 tool06 参考渲染默认写入 `outputs/rendered/`。`outputs/` 已被 Git 忽略，历史实验记录保持原位置。
环境、密钥、数据、权重、索引与运行产物不随源码提交。现有后台数据准备任务依赖的环境暂时保留。

## 文档

- [API 配置与统一切换](docs/API_CONFIGURATION.md)
- [项目结构与分工](docs/ARCHITECTURE.md)
- [Agent 核心与运行记录](docs/ARCHITECTURE.md)
- [对话续跑后端与前端接入](docs/CONVERSATION_CONTINUATION.md)
- [全流程日志、上下文与计时](docs/RUN_LOGGING.md)
- [工具说明](docs/TOOLS.md)与[自动生成接口](docs/INTERFACES.md)
- [后端组件](docs/ARCHITECTURE.md)与[执行契约](docs/ARCHITECTURE.md)
- [提示词分层](docs/PROMPTS.md)与[LLM 工具规范](docs/PROMPTS.md)
- [IL/RL 训练](docs/TOOLS.md)

## 许可

自有代码使用 [Apache 2.0](LICENSE)，声明见 [NOTICE](NOTICE)。内置第三方代码保留各自许可证与声明，见对应 core 目录。

### 配置位置

- `configs/api.json`：统一模型配置入口；`active_profile` 选择主 Agent 及跟随它的工具所用的连接配置，`tool_profiles` / `tool_efforts` 可逐工具覆盖。每组包含地址、模型、密钥变量名称、请求 timeout 和 max_tokens；另含独立的检索 embedding/rerank 参数。
- 根目录 `.env`：仅存 API/OSS 密钥，不放模型或运行参数。
- `configs/agent.json`：Agent 步数、工具模式、纠错次数、日志目录。
- `tools/tool*/config.json`：提示词版本、输出格式纠错次数，不覆盖模型参数。
- `tools/tool04_edit_planner/observation.json`：抽帧、拼图和补充视角预算。
- `mcp_servers/*/adapter/settings.json`：各服务解释器、路径与运行参数。
- 工具/仿真/渲染的执行时限属于任务预算，与 api.json 的单次模型请求超时分开管理。

编辑后端的 `mcp_servers/scene_editing/core/adversarial/backend_settings.json` 分别保存编辑预算、可选 VADv2 自测、旧 PDM 后端和渲染参数。各路径相对于项目根目录解析，绝对路径仍可用于本机部署；解释器软链接不会被解析为系统 Python。公共默认配置要求另行准备相应环境、模型、地图和系统库。`render.gsplat_extension` 可指定预编译扩展，留空时使用渲染环境自身的扩展加载方式。修改本机部署路径和开关后，应保留为本地设置，不随公共代码改动一起提交。

主 Agent、tool02/04、失败分析及内置编辑/检索的推理请求均读取当前公共 API 配置。代码中的 `get_profile("kimi")` 保留为公共推理配置别名，解析到 `active_profile`；支持逐工具选择的入口还会读取对应覆盖项。推理 API 与检索 embedding/rerank API 独立选择，Qwen 也可作为推理模型。

### 公共工具任务执行

所有经主 Agent 执行的工具统一登记在 `logs/jobs/<job_id>/`，该目录不进入 Git。
任务在独立 worker 进程执行；默认 720 秒是单次等待时限，不是后台任务的终止时限。
等待结束返回 `status=running` 和 `job.job_id`，主 Agent 自动继续查询原任务，
不消耗新的 LLM 决策步骤。工具自己的仿真/训练进程超时仍然有效。

同一 Agent run、工具、模式和相同输入只启动一份；成功结果直接复用。
失败任务不会自动重新执行，明确重试时在原工具参数旁增加
`"_execution":{"retry":true}`。也可用 `"_execution":{"job_id":"..."}`
查询原任务；此字段由公共执行层剥离，不传给工具。
输入中的绝对路径资源元信息改变后，不复用旧的完成结果；这不是完整内容哈希缓存。
不同 run 不共享完成结果，刻意重跑实验仍会执行。

Python 调试可用 `ToolExecutor.query(job_id)` 查询，或设置相同 `scope` 恢复该 run。
任务目录包含 manifest、原始请求、worker.log 和最终 result.json。
worker 意外退出但已登记子进程组还在时标记 orphaned，阻止再次启动；
需要先检查并处理残留进程，不会自动杀死其他实验。直接调用底层 adapter 不经过此任务层。

### 主 Agent 调用协议

主 Agent 使用当前所选模型的原生 Function Calling + 流式接收，每轮一个调用。
不保留旧的文本 JSON 决策协议。完整接收并校验参数后才交给公共任务层执行；
finish/stop 作为控制函数，工具编号不变。每轮耗时、首 chunk 延迟、
chunk 数量和 token 用量记录在 Agent 的 agent.jsonl 中。

主 Agent、tool02/04 共用客户端及内置编辑 Agent 均显式设置 SDK max_retries=0；SDK 不自动重发请求。输出纠错、业务内显式恢复及任务重试是独立机制。

### 上下文精简

`agent/context.py` 为所有工具生成主 Agent 决策摘要，保留目标、来源版本、
检查/失败反馈和下游必需路径；完整工具结果继续通过 result_ref 传递。
不增加额外 LLM 调用。原始状态、工具结果和日志不被裁剪。
