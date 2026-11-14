# API 配置与统一切换

主 Agent、攻击编辑 Agent（05）、自车编辑 Agent（06），以及失败分析、需求生成、粗规划和检索中的推理请求，共用 `configs/api.json`。主 Agent 使用 `active_profile`；工具默认跟随，也可通过 `tool_profiles` 固定模型、通过 `tool_efforts` 固定思考强度。

逐工具设置为 `active_profile` 时跟随公共选择；已固定模型的工具不会随 `use NAME` 切换。实际调用还应检查运行日志中的模型和思考参数，见 [模型调用配置](MODEL_CALL_CONFIGURATION.md)。

检索用的 embedding/rerank API 和 OSS 凭证独立管理，不参与推理 API 切换。

[各模型与主Agent、05、06的统一思考配置](MODEL_CALL_CONFIGURATION.md)记录了已实施的请求策略、最高档位、验证状态及实验边界。

## 当前已保存的配置

以下是 2026-11-11 保存的配置。`kimi`、`qwen_flash`、`ds_flash`、`mimo_pro`、`glm_flash` 使用 `https://apicz.boyuerichdata.com/v1`；`ds_official_flash` 使用 DeepSeek 官方地址 `https://api.deepseek.com`。后续如有修改，以 `python scripts/api_profile.py list` 的输出为准。

| 配置名 | 接口中的准确模型名称 | 密钥变量 | 本次验证 |
|---|---|---|---|
| `kimi` | `kimi-k3` | `OPENAI_API_KEY` | 原有配置，可用 `use kimi` 切回 |
| `qwen_flash` | `qwen3.8-flash` | `API_KEY_1` | 四项基础检查通过；案例3经工具协议及图像兼容修复后恢复完成，专家独立检查与视频验收通过 |
| `ds_flash` | `deepseek-v4.1-flash-1017` | `API_KEY_2` | 同上四项基础检查通过 |
| `mimo_pro` | `mimo-v2.5-pro` | `API_KEY_3` | 可读取模型列表；调用返回 HTTP 503“无可用渠道”，待服务端恢复或开通 |
| `glm_flash` | `z-ai/glm-5.3-flash` | `API_KEY_4` | 同上四项基础检查通过 |
| `ds_official_flash` | `deepseek-flash` | `API_KEY_5` | 文字/图片/原生调用通过；案例3兼容修复后05完成，06主动结束未提交，暂无专家视频 |

基础检查只使用简短文本、无副作用的工具定义和合成纯色图片，不代表已用所有新模型跑通完整攻防案例。配置成功与接口当前可用是两个状态；MiMo 的配置保留，但本次未验证可调用。添加配置本身不改变 `active_profile`，当前选择以 `show` 为准。2026-11-11的Qwen案例3恢复结果在 `outputs/case3/20261111_qwen_recovery02/`，包含专家视频、`README.md` 与 `final-audit.json`；原失败轮及400诊断分别保留在相邻 `20261111_qwen_flash02/`、`20261111_qwen_recovery01/`。

例如切换到 DeepSeek 或 GLM：

```bash
python scripts/api_profile.py use ds_flash
python scripts/api_profile.py use ds_official_flash
python scripts/api_profile.py use glm_flash
```

官方DS配置与原第三方DS配置分别保存，可用 `use ds_official_flash` 和 `use ds_flash` 切换。官方模型名及版本映射参见 [DeepSeek官方说明](https://api-docs.deepseek.com/quick_start/pricing/)。完整调用仍需实际检查文字、图片、原生工具调用以及项目提示词组合。

## 配置放在哪里

| 文件 | 内容 |
|---|---|
| [根目录 .env 示例](../.env.example) | 各组 API key 和 OSS 凭证；实际密钥填在本地 `.env` |
| [configs/api.json](../configs/api.json) | 每组配置的模型、地址、密钥变量名称、请求预算，以及当前 `active_profile` |
| [scripts/api_profile.py](../scripts/api_profile.py) | 新增、查看和切换配置的命令行入口 |

`.env` 可以同时保存多把 key，名称自定，例如：

```dotenv
API_KEY_1=第一把实际密钥
API_KEY_2=第二把实际密钥
```

模型名称和 API 地址写入 `configs/api.json`，不放进 `.env`。已有 shell 环境变量优先于 `.env`；修改 `.env` 后若仍读到旧 key，应检查 shell 中是否已设置同名变量。

## 新增并切换

在项目根目录执行。以下 `python` 可替换为 `environments/agent/.venv/bin/python`。地址和模型名都是占位示例，需要替换为服务商提供的真实值。

先在 `.env` 填好上述密钥，再新增配置：

```bash
python scripts/api_profile.py add api1 \
  --base-url https://your-provider.example/v1 \
  --model your-model \
  --key-env API_KEY_1

python scripts/api_profile.py add api2 \
  --base-url https://other-provider.example/v1 \
  --model other-model \
  --key-env API_KEY_2
```

`--key-env` 后面填变量名称，不能填实际 key。配置名 `api1`、`api2` 可自行更换。新增配置不会自动选用，也不会覆盖已有同名配置。

```bash
python scripts/api_profile.py list       # 查看已保存的推理配置
python scripts/api_profile.py use api1   # 主 Agent 和未固定模型的工具使用 api1
python scripts/api_profile.py show       # 查看当前配置
python scripts/api_profile.py use api2   # 切到 api2
python scripts/api_profile.py use kimi   # 切回原有配置
```

切换完成后，再启动主 Agent：

```bash
environments/agent/.venv/bin/python -m agent.main "你的任务"
```

**请在新任务开始前切换。已运行的主 Agent、后台工具任务及子进程需要结束或等待完成后重新启动，不能依靠修改配置在运行中统一切换。**

## 命令的实际行为

| 命令 | 行为 |
|---|---|
| `add NAME` | 保存模型、地址和密钥变量名称；默认请求超时 240 秒、最大输出 8192 tokens |
| `list` | 列出配置、当前选中状态和对应密钥是否已设置，不显示 key |
| `show` | 查看当前统一配置，不显示 key |
| `use NAME` | 检查配置和密钥变量，更新 `active_profile`；缺少密钥时拒绝切换，保留逐工具覆盖项 |

这些命令都不会调用外部 API。`key_configured: true` 只表示本地变量非空，不代表 key 有效、账户有余额或模型可用；`use` 成功也不代表完成了真实接口测试。

新增配置时可指定请求预算：

```bash
python scripts/api_profile.py add api3 \
  --base-url https://another-provider.example/v1 \
  --model another-model \
  --key-env API_KEY_3 \
  --timeout 300 \
  --max-tokens 8192
```

另有可选参数 `--reasoning-effort low|medium|high|xhigh|max`。是否支持该参数取决于模型和调用入口，不能仅凭保存配置判断兼容。

这里的 `timeout` 是单次模型请求预算。Agent 步数、工具任务时限、仿真和渲染预算仍由各自配置管理。

## 配置结构与维护

脚本将配置写入 `configs/api.json`。以下只展示相关结构，编辑时应保留原有配置和 `retrieval` 部分：

```json
{
  "active_profile": "api1",
  "api1": {
    "base_url": "https://your-provider.example/v1",
    "model": "your-model",
    "api_key_env": "API_KEY_1",
    "timeout": 240,
    "max_tokens": 8192
  }
}
```

修改已有配置的模型、地址或预算时，编辑对应配置项；更新 key 时，编辑 `.env` 中对应变量。完成后用 `show` 核对配置，并在新任务中使用。

保留原有 `kimi` 配置即可用 `use kimi` 切回。旧配置没有 `active_profile` 时默认读取 `kimi`。代码中的 `get_profile("kimi")` 是兼容已有入口的公共推理配置别名，实际解析到 `active_profile`，不代表始终使用 Kimi 模型。

## 统一主/子Agent思考配置

五个试验配置已经统一：第三方DS保留max，官方DS按最新决定使用high，Kimi和GLM使用high，Qwen3.8 Flash按最新决定从xhigh降为medium。主Agent、05、06共享所选profile的策略；GLM使用max_tokens=131072、timeout=2400（连续生成超时与65536长度截断分别诊断后提高），其余四个max_tokens=65536、timeout=600。工具传输auto并保留本地原生调用校验。MiMo保留原配置，待渠道和视觉能力核实。

新字段为request_dialect、thinking_mode、reasoning_effort、tool_choice_policy。新增配置时可用`--request-dialect thinking|qwen|kimi --thinking-mode enabled --reasoning-effort 档位 --tool-choice-policy auto`；字段必须成套符合对应模型，max/xhigh不再被后台统一枚举拒绝。新profile不能一边启用thinking一边声明required_tool_thinking=disabled。详见[统一思考配置](MODEL_CALL_CONFIGURATION.md)。

## 模型兼容性与密钥保护范围

配置切换支持保存不同的地址、模型和密钥绑定。完整攻防流程还要求服务兼容 Chat Completions、图像输入和原生工具调用；目前部分入口仍带有 Kimi 专用请求参数，新模型需要单独验证和适配。配置保存成功不能代替接口兼容性验证。

以下required/带图auto/关闭thinking描述的是旧的未统一策略兼容路径；当前五个统一思考配置优先按上文使用auto，保留本地原生校验。

06需要每轮返回一个原生工具调用，默认发送 `tool_choice=required`。若阶段只允许某个函数，先缩小可见函数集；候选ID和看图资格等限制继续保留。普通文字或正文JSON代码块都不会被当作工具执行。没有原生调用时，会在现有模型轮数及无效调用预算内给出纠正反馈；达到上限时返回协议执行错误，不能归为“没有合格防御”。每次纠正同样消耗预算，协议记录保存在编辑目录的 `protocol-errors/`，中断恢复时一并核验。

当前Qwen接口在实际场景中拒绝“图片＋required”，返回HTTP 400；同一请求去掉图片或将工具选择改为auto可被接受。为保留图片，`qwen_flash` 已显式设置 `image_tool_choice: "auto"`：仅带图片的请求采用auto传输，纯文本仍用required，本地仍必须收到原生调用才执行。其他配置默认不启用这个兼容选项。新增配置时可用 `--image-tool-choice auto` 显式指定；是否适用应通过该服务的组合测试确认，不能把小文本工具调用测试当成图片工具兼容性证明。

DeepSeek官方接口在thinking模式下不支持required或指定函数的工具选择。早期曾用required_tool_thinking=disabled让强制工具轮次关闭思考；当前ds_official_flash已改为统一enabled＋high＋auto，不再因强制工具而关闭思考。旧兼容选项保留给未采用新策略的配置，新启思考配置与该关闭选项冲突时会拒绝。依据：[官方Chat Completions说明](https://api-docs.deepseek.com/api/create-chat-completion/)。

主Agent客户端导入时不读取 `.env`，实例初始化通过公共配置加载凭据。测试使用临时配置和虚构凭据；测试失败输出也应先脱敏再显示，避免断言详情回显实际密钥。这不代表其他所有调用链的日志脱敏已经完成。

API 地址使用 HTTPS，不能内嵌账号、密码、查询参数或片段。密钥仅通过环境加载；切换脚本不接受实际 key 参数，不将 key 写入公共配置，也不在输出中显示 key。本地 `.env` 已被 Git 忽略，仍应保持仅本人可读的权限。

真实请求会把 key 发送给配置的 API 服务端进行认证，请核对服务地址。项目其他调用链的错误日志脱敏尚有待完善；切换脚本的保护不等于全项目日志已经完成密钥防泄露修复。

最新两组追加试验已经完成：Qwen medium每轮API364.695/385.808/600.002秒，平均450.168秒，2/3交付；官方DS high每轮144.612/315.901/399.397秒，平均286.637秒，3/3交付。报告逐轮显示编辑尝试次数、三次API合计与包含失败的算术平均：[三轮汇总](../outputs/case3/20261112_defense_medium_official01/README.md)。旧max/xhigh历史快照未改写。
