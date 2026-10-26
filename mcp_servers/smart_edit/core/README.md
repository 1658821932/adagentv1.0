# Smart Edit

DGGT Mode 2 场景编辑配置生成器。常用方式是通过 `smart-edit-chat` 连接一个
OpenAI-compatible 模型服务,由终端客户端自动启动本地 MCP server 并调用工具。

## 设计

服务器默认只做**确定性**的工作:扫描资产目录、维护当前配置草稿、校验参数、
组装 `Nodes.RigidNodes` 编辑配置并输出 JSON。对已有实例编辑需提供
`instance_info_path`；插入和替换需明确提供资产 `scale`，没有预设缩放系数。

**不**调用任何外部 LLM。自然语言解析交给客户端 LLM(Claude/Cursor 等),通过结构化工具调用驱动本服务器。

当前 MCP 接口仅暴露 DGGT 刚体编辑：平移、旋转、缩放、删除、插入、替换。
插入采用第一帧 `anchor_pixel`；不提供 `ref_id` 或旧的背景/SMPL 编辑工具。

自车渲染轨迹使用独立输出链路。客户端自然语言请求中指定
`output_kind=ego_trajectory_offsets`，模型调用 `set_ego_trajectory_offsets`，再调用
`save_ego_trajectory_offsets`。输出格式为：

```json
{
  "coordinate_system": "camera_local_forward_left_up",
  "offsets": [[0, 0.02, 0]]
}
```

`offsets` 使用 DGGT 原生单位；单个向量作用于所有帧，多个向量按帧作用。Smart Edit 不推断米制比例。

## 安装

```bash
pip install -e .
# 或仅装运行依赖
pip install -r requirements.txt
```

依赖: `mcp>=1.0.0,<2`、`openai>=1.0.0`、`python-dotenv>=1.0.0`、`pyyaml>=6.0`,Python 3.10+。
当前服务器使用 MCP 1.x 的接口,请勿升级到 MCP 2.x。

## 启动模型服务

终端客户端只连接已经启动好的 OpenAI-compatible 模型服务,不会启动模型服务。
如果使用本地 Qwen3 + SGLang,先启动兼容 OpenAI Chat Completions API 的模型服务:

```bash
python -m sglang.launch_server \
  --model-path models/Qwen3-8B/ \
  --host 127.0.0.1 \
  --port 30000 \
  --max-total-tokens 32768 \
  --tool-call-parser qwen \
  --reasoning-parser qwen3
```

其中 `--tool-call-parser qwen` 会把模型输出解析成 OpenAI 标准
`tool_calls`;否则模型可能只是在文本里打印 `<tool_call>...</tool_call>`,
客户端不会真的调用 MCP 工具。`--reasoning-parser qwen3` 用于解析 Qwen3 的
thinking 内容。

也可以连接提供商的 API，下面的客户端示例支持通过环境变量切换。

## 终端交互客户端

本地模型服务启动后,在项目根目录安装当前包:

```bash
python3 -m pip install -e .
```

安装后可以运行 console script:

```bash
smart-edit-chat \
  --base-url http://127.0.0.1:30000/v1 \
  --api-key EMPTY \
  --model models/Qwen3-8B/
```

如果没有安装当前包,或者 `smart-edit-chat` 不在 PATH 里,也可以直接运行脚本:

```bash
python3 clients/terminal_chat.py \
  --base-url http://127.0.0.1:30000/v1 \
  --api-key EMPTY \
  --model models/Qwen3-8B/
```

OpenRouter 示例:

```bash
OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
OPENAI_API_KEY="$OPENROUTER_API_KEY" \
OPENAI_MODEL=qwen/qwen3-8b \
smart-edit-chat
```

也可以把连接配置放在项目根目录的 `.env.local` 中,客户端启动时会自动读取。
命令行参数优先于环境变量,已有环境变量优先于 `.env.local`。例如百炼配置:

```dotenv
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=deepseek-v4-flash-0731
OPENAI_API_KEY=替换为你的百炼API密钥
```

配置好后直接运行 `smart-edit-chat`。`.env.local` 已被 Git 忽略。

客户端会启动本地 `smart-edit` MCP server,读取工具列表,把 MCP 工具转换成
OpenAI `tool_calls`,再将工具结果回填给模型。终端内置命令:

| 命令 | 说明 |
|---|---|
| `/tools` | 显示当前 MCP 工具 |
| `/assets` | 调用 `list_assets` 查看资产 |
| `/trace` | 开关工具调用过程显示 |
| `/reset` | 清空对话历史 |
| `/exit` / `/quit` | 退出 |

常用参数:

| 参数/环境变量 | 默认值 | 说明 |
|---|---|---|
| `--base-url` / `OPENAI_BASE_URL` | `http://localhost:30000/v1` | OpenAI-compatible API 地址 |
| `--api-key` / `OPENAI_API_KEY` | `EMPTY` | API key;本地 sglang 通常任意非空即可 |
| `--model` / `OPENAI_MODEL` | `qwen3` | API 暴露的模型名 |
| `--temperature` / `OPENAI_TEMPERATURE` | `0.2` | 采样温度 |
| `--max-tokens` / `OPENAI_MAX_TOKENS` | `2048` | 单次回复最大 token 数 |
| `--show-tool-process` / `SHOW_TOOL_PROCESS` | 关闭 | 显示每轮模型请求、工具参数和工具返回摘要 |
| `--assets-dir` / `ASSETS_DIR` | `assets` | 资产目录 |
| `--smpl-assets-dir` / `SMPL_ASSETS_DIR` | `assets/pedestrian` | SMPL 资产目录 |
| `--output-dir` / `OUTPUT_DIR` | `outputs` | 配置输出目录 |

示例输入:

```text
你> 加一个交通锥,在车前 8 米、右侧 1.5 米,持续全程,保存成 cone.yaml
```

## 工具

### `list_assets`
列出可用资产,按类别分组。可选参数 `category` 筛选单个类别。

### `list_smpl_assets`
列出可用 SMPL 资产,按类别分组。可选参数 `category` 筛选单个类别。

### 草稿式配置工具

MCP 服务端维护一份当前配置草稿。模型不需要一次性提交完整 YAML,而是通过小工具
逐步修改草稿,最后预览或保存。最终输出始终是标准结构:

```yaml
debug: false
weather_type: ""
is_legend: false
Nodes:
  Background:
    add:
      add_obj:
      - assets/cone/traffic_cone.ply
      offset:
      - [5, -3, 0]
      time_window:
      - [10, 50, 5]
  RigidNodes:
    trajectory_edit:
      instance_id: [0]
      waypoints:
      - - [0, -3, 0]
        - [6, -1.5, 0]
        - [12, 0, 0]
      method: [cubic]
```

| 工具 | 说明 |
|---|---|
| `start_config` | 清空当前草稿并创建新配置,可记录 DGGT 实例清单路径 |
| `set_globals` | 设置 `debug`、`weather_type`、`is_legend` 或 `legend` |
| `set_randomize` | 设置保存时的随机批量生成参数 |
| `add_background_asset` | 追加一个 Background 资产 |
| `edit_rigid_trajectory` | 追加一个原场景刚体轨迹编辑 |
| `rotate_rigid_instance` | 旋转一个原场景刚体实例 |
| `scale_rigid_instance` | 等比例缩放一个原场景刚体实例 |
| `add_rigid_asset` | 设置唯一的 DGGT 像素锚点刚体插入 |
| `remove_rigid_instance` | 按实例 ID 生成原场景刚体的删除配置 |
| `replace_rigid_instance` | 用车辆 3DGS PLY 和显式 scale 替换一个原场景刚体实例 |
| `remove_added_rigid_asset` | 按草稿序号撤销一个新增刚体资产 |
| `add_smpl_person` | 追加一个 SMPL 行人 |
| `preview_config` | 查看当前草稿 |
| `save_config` | 保存当前草稿到文件 |
| `reset_config` | 重置当前草稿 |

资产分类约束:

| 类型 | 工具 | 说明 |
|---|---|---|
| 普通/其它静态资产 | `add_background_asset` | 如交通锥、路障、标志牌等,写入 `Nodes.Background` |
| 车辆资产 | `add_rigid_asset` / `edit_rigid_trajectory` | 新增车辆资产或编辑已有车辆/刚体实例,写入 `Nodes.RigidNodes` |
| 行人/人体资产 | `add_smpl_person` | 使用 SMPL 模型和运动数据,写入 `Nodes.SMPLNodes` |

示例流程:

```text
start_config(debug=false, weather_type="", is_legend=false)
list_assets(category="cone")
add_background_asset(asset="assets/cone/traffic_cone.ply", position=[5, -3, 0], start_frame=10, end_frame=50, interval=5)
edit_rigid_trajectory(instance_id=0, waypoints=[[0, -3, 0], [6, -1.5, 0], [12, 0, 0]], method="cubic")
save_config(output_path="scene_config.yaml")
```

### 刚体编辑

编辑 DGGT 已有实例时,使用 `start_config(instance_info_path="/path/to/instance_info.json")`
将实例清单路径写入配置顶层;后续编辑工具只需传入 `instance_id`。
实例清单直接记录每个实例的逐帧模型输入图 bbox，例如：

```json
{
  "instances": [{
    "instance_id": 0,
    "frames": [
      {"frame_index": 7, "bbox": [75, 195, 175, 265]},
      {"frame_index": 8, "bbox": [78, 194, 178, 264]}
    ]
  }]
}
```

编辑配置只引用这份清单。DGGT 在编辑运行中用 bbox 与当前 Dynamic Map、非天空区域取交集，不需要提前生成 mask 或 PLY。

完整操作说明、中文指令、工具参数和 YAML 示例见 [刚体与静态资产操作指南](RIGID_OPERATIONS_GUIDE.md)，
其中第 4.4 节介绍交通锥（雪糕桶）、路障、标志牌的新增方法。

当前聊天工具使用 DGGT Mode2 的唯一插入格式新增车辆：资产路径、第一帧像素锚点
`anchor_pixel=[x,y]` 和统一缩放倍数 `scale`。该像素必须使用 DGGT 处理后的第一帧图像坐标。
插入配置不使用 `position`、`offset`、`ref_id`、`waypoints` 或 `method`。

示例：

```text
add_rigid_asset(asset="assets/car/taxi.ply", anchor_pixel=[259, 250], scale=0.02)
save_config(output_path="add_car.json")
```

生成：

```json
{
  "Nodes": {
    "RigidNodes": {
      "add": {
        "add_obj": ["assets/car/taxi.ply"],
        "anchor_pixel": [[259, 250]],
        "scale": [0.02]
      }
    }
  }
}
```

当前 DGGT 只执行一个插入对象；再次调用 `add_rigid_asset` 会覆盖草稿中前一次插入。
请保存为 JSON，因为 DGGT 当前的编辑入口使用 `json.load()`。

删除原场景车辆时,提供真实的实例 ID。例如“删除车辆 0 和 3,保存为 remove.yaml”:

```yaml
Nodes:
  RigidNodes:
    remove:
      instance_id: [0, 3]
```

`remove_rigid_instance(instance_id=0)` 会标记删除,同时撤销草稿中该实例的轨迹编辑;
重复调用不会产生重复 ID。此后不能在同一草稿里继续修改该实例的轨迹。
这只是编辑配置,需要保存并交给场景编辑器执行才能改变场景。

若要撤销刚新增的车辆,先用 `preview_config` 查看 `RigidNodes.add.add_obj`,
再调用 `remove_added_rigid_asset(index=0)` 撤销该插入。`anchor_pixel` 和 `scale` 会同步删除，磁盘上的 `.ply` 资产文件保留。

旋转使用世界固定 XYZ 轴外旋角度 `[rx, ry, rz]`,单位为度;缩放为绕实例中心的等比例倍数。
替换资产必须来自车辆类别。配置生成不移动点云或执行渲染,需要交给 DGGT 执行。

### 随机批量生成

默认保持单份配置行为。需要批量扰动时,先设置随机化参数,再保存:

```text
set_randomize(
  enabled=true,
  count=100,
  seed=42,
  offset_jitter=[1.0, 0.5, 0.0],
  waypoint_jitter=[1.0, 0.5, 0.0],
  time_window_jitter=[5, 5, 2]
)
save_config(output_path="edit_config_{index}.yaml")
```

随机扰动规则:

| 参数 | 默认值 | 说明 |
|---|---|---|
| `count` | `10` | 生成数量,范围 `1-1000` |
| `seed` | 无 | 指定后可复现同一批结果 |
| `offset_jitter` | `[1, 1, 0]` | 扰动所有 `offset` 的 `[x, y, z]`,单位米 |
| `waypoint_jitter` | `[1, 1, 0]` | 扰动所有 `waypoints` 的 `[x, y, z]`,单位米 |
| `time_window_jitter` | `[5, 5, 2]` | 扰动 `[start, end, interval]`;`end=-1` 保持不变 |

`save_config` 也支持传入一次性的 `randomize` 覆盖当前设置。开启随机化后,
如果 `output_path` 包含 `{index}`,会按 `0, 1, 2, ...` 展开保存多个配置文件;
否则保存为一个包含 `configs` 和 `randomize` 的批量文件。

## 资源

- `assets://catalog` —— 当前资产清单(JSON 格式)
- `smpl-assets://catalog` —— 当前 SMPL 模型与运动资产清单(JSON 格式)

## 输出格式

YAML 对纯数字内嵌列表(如 `[x, y, z]`)使用 flow 行内样式,字符串列表(如 `add_obj`)使用 block 样式:

```yaml
is_legend: false
weather_type: ''
debug: false
Nodes:
  Background:
    add:
      add_obj:
      - assets/cone/traffic_cone.ply
      offset:
      - [5, -1.8, -1.2]
      time_window:
      - [10, 50, 5]
```

## 资产组织

资产按类别分目录,每个子目录放 `.ply` 文件。新增类别只需新建子目录,服务器下次启动时自动识别。详见 [assets/README.md](assets/README.md)。

## 项目结构

```
smart_edit/                 主包
├── __init__.py
├── __main__.py             python -m smart_edit 入口
├── assets.py               资产扫描
├── builder.py              配置组装与校验
├── yaml_format.py          自定义 YAML dumper
├── io.py                   JSON/YAML 写入
└── server.py               MCP 服务器(工具/资源/main)

tests/test_server.py        MCP 资源接口集成测试(无需模型服务)
outputs/                    save_config 默认输出目录
assets/                     3D 资产
```

## MCP Server 直连

一般不需要手动启动 MCP server;`smart-edit-chat` 会自动启动。需要接入
Claude Desktop、Cursor 或 MCP Inspector 时,可以直接运行:

```bash
python -m smart_edit
# 安装为可执行命令后
smart-edit-mcp
```

服务器通过 stdio 与 MCP 客户端通信。

Claude Desktop 可参考 [claude_desktop_config.example.json](claude_desktop_config.example.json),
核心配置类似:

```json
{
  "mcpServers": {
    "smart-edit": {
      "command": "python",
      "args": ["-m", "smart_edit"],
      "cwd": "/Users/xy/Projects/smart-edit",
      "env": {
        "ASSETS_DIR": "/Users/xy/Projects/smart-edit/assets",
        "SMPL_ASSETS_DIR": "/Users/xy/Projects/smart-edit/assets/pedestrian",
        "OUTPUT_DIR": "/Users/xy/Projects/smart-edit/outputs"
      }
    }
  }
}
```

## 测试 / 调试

```bash
python -m unittest discover -s tests -v                         # MCP 资源接口集成测试
npx @modelcontextprotocol/inspector python -m smart_edit         # 交互式调试
```

配置好 `.env.local` 后,可以显式运行真实模型的刚体多轮验收:

```bash
python tools/test_rigid_api.py --rounds 3
# 单独重测新增车辆的撤销场景
python tools/test_rigid_api.py --rounds 1 --scenario insertion
```

每轮覆盖实例编辑、DGGT 像素锚点插入、撤销插入和缺少实例 ID 的追问。该脚本调用当前配置的模型 API,
把逐轮结果、工具调用记录和 YAML 保存到 `outputs/rigid-multiround-*/`。
