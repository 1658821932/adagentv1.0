# 工具接口与运行管理

## 工具调用

正式入口是 tools.registry.call_tool，Agent 的 ToolExecutor 也使用这个入口：

```python
from tools.registry import call_tool
result = call_tool("video_retriever", {
    "requirement": "白天高速公路，第一视角",
    "library": "default",
}, mode="real")
```

每个 Tool 都有 real.py、mock.py、schemas.py 和 description.py。调用统一使用 registry（旧 tool.py 兼容入口已删除）。
三个纯文本 LLM 工具由 LLMTool 校验，其余后端工具在 registry 的统一边界校验。

- schemas.py 的 Input/Output 是可执行契约；description.py 提供用途和字段解释，并从 Schema 生成参数结构。
- 参数缺失、类型错误、未知字段在启动后端前返回 input_validation_error。
- 后端声称成功但缺少必需结果字段时，返回 output_validation_error。
- mock 与 real 共用契约。mock 不创建真实模型、视频或训练产物，不能将 mock 路径作为 real 文件使用。

## 服务配置

五个服务各自的 adapter/settings.json 保存非敏感接入配置，环境变量可覆盖。
解释器路径支持项目相对路径和绝对路径；默认仍使用 environments 下已有环境或软链接。

| 服务 | 解释器覆盖变量 |
|---|---|
| DGGT | AGENT_PYTHON、DGGT_PYTHON、DGGT_SEGFORMER_PYTHON |
| SmartEdit | SMART_EDIT_PYTHON |
| 视频检索 | VIDEO_RETRIEVER_PYTHON |
| WorldEngine | AGENT_PYTHON、SIMENGINE_PYTHON、ALGENGINE_PYTHON、WORLDENGINE_TRAIN_PYTHON |
| WFA | WFA_PYTHON |

SmartEdit 的模型默认值由自己的 adapter/settings.json 管理，不再读取 configs/agent.json。
纯文本 LLM 工具（tool02/03/07）的配置在各自目录 config.json，由 tools/llm_runtime.py 统一加载。
视觉验证使用 tools/tool10_scene_validator/config.json，由工具自己的 config.py 解析，默认仍为 Kimi。
密钥继续从本地环境/.env 读取，不写入上述 JSON。
基础重建缺陷只允许通过某次验证的 ignored_reconstruction_artifacts 显式传入，不再对所有场景应用固定排除项。

## WorldEngine 仿真

adapter/run_testing.sh 是轻量入口，实际命令构建与进程监督位于 adapter/launch.py。

- 显式传入最终 data_file_path，避免绝对场景目录再次拼接 WORLDENGINE_ROOT。
- num_history + num_future 必须等于 max_frame_per_scene；省略 num_future 时自动推导。
- 默认 40 帧，也支持原生短场景预算，最少 5 帧；源场景不足时明确失败，不补帧。
- 同一帧数同时传入仿真时域与规划监视器，替代规划端固定 12 帧。
- num_scenarios、sampling_poses、可选稠密奖励及参与者策略显式传入后端。
- planner_interval_frames、planner_history_stride、multi_agent_vadv2 当前没有原生实现，明确拒绝，不静默接受。
- output/simulation/effective_config.json 保存最终路径、参数与启动命令。
- 一方异常时结束同一任务的另一方；正常结束允许规划器先响应完成标记，随后等待仿真释放资源。
- 保留单次运行的帧、轨迹和元数据，便于检查和后训练。

训练的 IL 默认配置位置为 core/experiments/posttrain/e2e_vadv2_50pct_waymo_ilft.py。
训练模式及数据准备以当前 tool15 schemas 和训练模块为准；本次接口整理保留同时进行的 IL/RL 模块改动。

## 长任务

mcp_servers/process.py 提供统一的 run_logged：

- stdout/stderr 直接写日志文件，运行中可以 tail，不必等待结束。
- JSON 子进程的 stdout 单独保存在 *.stdout.json，进度/错误写在对应 .log。
- 超时或取消时向该任务进程组终止，再清理未退出的子孙进程。
- manifest 区分 success、error、timed_out、cancelled；强制 SIGKILL 或断电仍可能留下 running。
- 大型日志只读取末尾供错误摘要使用，不在内存中累计整份日志。

目前仍是同步工具调用，没有新增后台队列或自动重跑昂贵任务。

## 追踪和状态快照

Agent 为每次调用生成 tool_call_id。服务 manifest 和子进程环境携带：
agent_run_id、tool_call_id、parent_run_id。

Agent 日志的 tools/step_*/call.json 保存同一关联信息。
每次工具返回后写 state.json，供查看进度和人工恢复定位；最终状态仍在 final_state.json。
state.json 不表示已实现自动续跑：恢复训练仍应显式选择已有 checkpoint 和任务参数。
