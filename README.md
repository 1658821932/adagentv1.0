# adagent

基于 LLM 的自动驾驶场景需求生成、检索、验证与后训练框架。主 Agent 根据用户任务和工具返回结果选择下一步，工具通过统一输入输出契约调用后端。

## 当前状态

当前启用 9 个工具，预留 01–10 编号（07 场景回放暂不注册），涵盖失败分析、需求生成、检索、快速粗规划、攻防编辑、原生回放、策略评测、失败筛选和后训练。
攻击车与自车编辑后端已接入；独立视觉验收工具已直接删除，没有迁入其他工具。
新增原生回放入口读取已准备的 WE 场景包；当前编辑结果的 `simulation_input` / `ilft_training_data` 若为 null，表示原生导出尚未完成，不能用原始场景代替编辑结果。

## 运行环境

从项目根目录创建 Agent 环境（Python 3.10+，需安装对应版本的 venv 支持）：

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

| 编号 | 工具 | 职责 |
|---|---|---|
| 01 | failure_analyzer | 分析已有失败场景与证据 |
| 02 | scene_instance_generator | 生成目标场景、全局自车目标与三个候选 |
| 03 | video_retriever | 检索并强制绑定多相机场景、标注和资产 |
| 04 | edit_planner | 快速判断可编辑性，给出针对底场景的粗计划 |
| 05 | attack_vehicle_editor | 规划、搜索、编辑并冻结攻击车场景 |
| 06 | ego_trajectory_editor | 规划、编辑并复核正确的自车专家轨迹 |
| 07 | scene_replayer | 用 WE 原生轨迹回放输出相机视频 |
| 08 | policy_runner | 执行 PDM 挑战或驾驶策略闭环仿真 |
| 09 | failure_case_filter | 读取官方评分并筛选失败案例 |
| 10 | policy_trainer | 执行 ILFT/RLFT 并输出模型与更新审计 |

四个后端组件保存在 `mcp_servers/`，均采用 `core/`、`adapter/`、`runtime/` 分层：

- `sentry_video_search`：检索、入库与重排序，详见[多相机场景绑定](mcp_servers/sentry_video_search/README.md)。
- `worldengine_failure_analysis`：失败取证、图像及数值分析。
- `scene_editing`：攻击车与自车编辑阶段接口，详见[编辑接口](docs/EDITING_INTERFACES.md)。
- `worldengine`：仿真、官方评分读取及训练。

这些组件通过 Python 或 JSON 子进程调用；目录名称不表示每个组件都提供 MCP 协议服务。

## 多相机场景绑定

tool04 real 要求索引源视频带有效 `.scene.json`。结果包含 `scene_ref`：场景身份、相机清单、同步图像元数据、标注与高斯资产路径。未绑定直接报错；未下载文件明确标记缺失。
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

- [项目结构与分工](docs/ARCHITECTURE.md)
- [Agent 核心与运行记录](docs/ARCHITECTURE.md)
- [工具说明](docs/TOOLS.md)与[自动生成接口](docs/INTERFACES.md)
- [后端组件](docs/ARCHITECTURE.md)与[执行契约](docs/ARCHITECTURE.md)
- [提示词分层](docs/PROMPTS.md)与[LLM 工具规范](docs/PROMPTS.md)
- [IL/RL 训练](docs/TOOLS.md)

## 许可

自有代码使用 [Apache 2.0](LICENSE)，声明见 [NOTICE](NOTICE)。内置第三方代码保留各自许可证与声明，见对应 core 目录。
