# adagent

基于 LLM 的自动驾驶场景需求生成、检索、验证与后训练框架。主 Agent 根据用户任务和工具返回结果选择下一步，工具通过统一输入输出契约调用后端。

## 当前状态

当前注册 9 个工具，涵盖失败分析、需求生成、多相机场景检索、攻击车/自车编辑接口、场景验证、仿真与训练。
旧的实例发现、视频重建、编辑计划、配置生成和编辑执行工具，以及相应旧编辑服务已移除。
新的攻击车编辑、自车专家轨迹编辑接口已注册，实际后端尚未配置。现阶段不能由 Agent 自动生成编辑场景；可以检索已有场景，也可以验证或仿真用户提供的合规输入。

## 运行环境

从项目根目录创建 Agent 环境（Python 3.10+）：

```bash
python3 -m venv environments/agent/.venv
environments/agent/.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

`.env` 只填写密钥；`configs/api.json` 管理模型、端点、超时。Kimi 用于主 Agent、tool02、视觉验证及分析；阿里接口用于视频检索嵌入与重排序。检索重排序还需要配置 OSS 凭证。

```bash
environments/agent/.venv/bin/python -m agent.main \
  "根据给定失败类型构思三个底场景方案，并检索一个已绑定标注与高斯资产的场景"
```

`configs/agent.json` 控制最大步数及各工具 mock/real 模式。Agent 编排需要模型接口；mock 工具不生成真实产物。
各 real 后端需独立准备环境、数据、权重和索引，安装主环境不包含 GPU 服务依赖。

## 工具与后端

| 编号 | 工具 | 能力 |
|---|---|---|
| 01 | failure_analyzer | 使用 WFA 分析已有场景与官方评测证据 |
| 02 | scene_instance_generator | 从失败分析构思目标场景、正确自车行为及三个候选方案 |
| 04 | video_retriever | 检索任一路已注册相机，强制绑定完整多相机场景 |
| 05 | attack_vehicle_editor | 标注驱动攻击车编辑接口，后端待配置 |
| 06 | ego_trajectory_editor | 冲突理解与自车专家轨迹编辑接口，后端待配置 |
| 10 | scene_validator | 对外部提供的渲染视频进行需求与视觉检查 |
| 13 | policy_runner | 使用准备好的场景和资产运行 WorldEngine |
| 14 | failure_case_filter | 读取官方 PDM 总分，筛选 0 分场景 |
| 15 | policy_trainer | 使用标注执行 ILFT，或使用 rollout 和候选奖励执行 RLFT |

四个后端组件保存在 `mcp_servers/`，均采用 `core/`、`adapter/`、`runtime/` 分层：

- `sentry_video_search`：检索、入库与重排序，详见[多相机场景绑定](mcp_servers/sentry_video_search/README.md)。
- `worldengine_failure_analysis`：失败取证、图像及数值分析。
- `scene_editing`：攻击车与自车编辑阶段接口，详见[编辑接口](docs/EDITING_INTERFACES.md)。
- `worldengine`：仿真、官方评分读取及训练。

这些组件通过 Python 或 JSON 子进程调用；目录名称不表示每个组件都提供 MCP 协议服务。

## 多相机场景绑定

tool04 real 要求索引源视频带有效 `.scene.json`。结果包含 `scene_ref`：场景身份、相机清单、同步图像元数据、标注与高斯资产路径。未绑定直接报错；未下载文件明确标记缺失。
新编辑模块将从这里读取现有数据集资产，详细接入仍待实现。

## 验证与测试

```bash
environments/video_retriever/.venv/bin/python -m pytest -q
environments/agent/.venv/bin/python -m tools.scripts.smoke_llm_tools --mode mock
```

测试依赖见 `requirements-test.txt`。接口测试不能代替真实场景质量、闭环表现或后训练收益验证。
Agent 记录保存在 `logs/`；服务产物保存在 `mcp_servers/<service>/runtime/<run_id>/`。
环境、密钥、数据、权重、索引与运行产物不随源码提交。现有后台数据准备任务依赖的环境暂时保留。

## 文档

- [项目结构与分工](docs/ARCHITECTURE.md)
- [Agent 核心与运行记录](docs/AGENT_FILES.md)
- [工具说明](docs/TOOLS.md)与[自动生成接口](docs/INTERFACES.md)
- [后端组件](docs/MCP_SERVERS.md)与[执行契约](docs/EXECUTION_CONTRACTS.md)
- [LLM 工具规范](docs/LLM_TOOLS.md)
- [IL/RL 训练](docs/policy_training.md)

## 许可

自有代码及内置第三方代码遵循各自许可证与声明，第三方项目许可见对应 core 目录。
