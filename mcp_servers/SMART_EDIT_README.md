# SmartEdit 客户端

`smart_edit_client.py` 将自然语言编辑要求转换为外部 SmartEdit MCP 服务的工具调用，并要求服务保存 JSON。它负责生成配置；真正的高斯编辑和渲染由 `scene_editor` 调用 DGGT 完成。

## 配置

- `SMART_EDIT_ROOT`：外部 SmartEdit 项目目录，本项目未内置其完整源码。
- 解释器固定为项目内 `environments/smart-edit/.venv/bin/python`；该环境需安装 SmartEdit、`mcp`、`openai`、`python-dotenv` 及服务自身依赖，并能在外部项目目录执行 `python -m smart_edit`。
- 配置 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`。客户端还会读取外部项目的 `.env.local`，已存在的环境变量优先。`OPENAI_TEMPERATURE` 可省略。

## 输入与输出

Agent 工具名为 `edit_config_generator`。输入包括 `edit_plan`、JSON `output_path`；修改已有车辆需同一段视频的 `instance_info_path`。add/replace 需指定 `asset_path` 和 `asset_scale`，不能凭空估计比例。输出包含已保存配置路径及校验信息。

配置支持轨迹、旋转、缩放、删除、添加、替换等操作。实例框不是米制三维坐标；SmartEdit 配置交给 `scene_editor` 时须按其接口提供坐标系，轨迹转换需匹配的相机参数及明确的 `meters_to_dggt`。

可选 `output_kind=ego_trajectory_offsets` 生成相机轨迹偏移。它只改变渲染视角，不改变闭环策略的制动或规划决策。
