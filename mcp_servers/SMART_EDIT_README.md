# SmartEdit 客户端与服务

SmartEdit 核心代码位于 `smart_edit/core/`；与 Agent 的衔接代码集中在 `smart_edit/adapter/`。

tool08 `edit_config_generator` 将编辑计划交给适配层，适配客户端通过 MCP 子工具组装并保存 JSON。实际的高斯编辑和渲染由 tool09 `scene_editor` 调用 DGGT 完成。

## 文件职责

| 文件 | 作用 |
|---|---|
| [adapter/service.py](smart_edit/adapter/service.py) | 接收编辑请求，启动客户端并校验配置。 |
| [adapter/client.py](smart_edit/adapter/client.py) | 连接模型与 MCP 服务，驱动子工具调用。 |
| [adapter/config.py](smart_edit/adapter/config.py) | 读取核心目录、运行环境与模型配置。 |
| [adapter/prompts/rigid_config.md](smart_edit/adapter/prompts/rigid_config.md) | 刚体编辑配置生成提示词。 |
| [adapter/prompts/ego_trajectory_offsets.md](smart_edit/adapter/prompts/ego_trajectory_offsets.md) | 自车渲染轨迹偏移生成提示词。 |
| [core/smart_edit/server.py](smart_edit/core/smart_edit/server.py) | SmartEdit MCP 服务入口。 |

## 配置

- 核心目录默认使用项目内 `mcp_servers/smart_edit/core`；可用 `SMART_EDIT_ROOT` 指定其他安装目录。
- 解释器为项目内 `environments/smart-edit/.venv/bin/python`，需要安装 SmartEdit、`mcp`、`openai`、`python-dotenv` 及服务依赖。
- 模型与端点读取根目录 `configs/agent.json` 的 llm 设置，支持 editor 角色及现有环境变量覆盖；密钥读取本地 `.env` 中的 `OPENAI_API_KEY` 或 `LLM_API_KEY`。客户端也会加载核心目录的 `.env.local`，已经存在的环境变量优先。
- 客户端以包模块 `mcp_servers.smart_edit.adapter.client` 启动，服务仍在 SmartEdit 核心目录内运行。

## 输入与输出

Agent 工具输入包括 `edit_plan`、JSON `output_path`；修改已有车辆需同一段视频的 `instance_info_path`。add/replace 需明确提供 `asset_path` 和 `asset_scale`。返回结果包含实际保存的配置路径和内部工具调用记录。

配置支持轨迹、旋转、缩放、删除、添加和替换等操作。实例检测框不是米制三维坐标；将轨迹配置交给 scene_editor 时，需要对应的相机参数和明确的 `meters_to_dggt`。

可选 `output_kind=ego_trajectory_offsets` 生成相机轨迹偏移，只改变渲染视角，不代表改变闭环驾驶策略。

当前服务注册的 15 个 MCP 子工具及完整调用关系见 [MCP 组件与工具接入说明](../docs/MCP_SERVERS.md)。
