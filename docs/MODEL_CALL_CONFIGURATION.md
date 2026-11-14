# 模型调用配置：统一主Agent与SubAgent思考

更新：2026-11-11。用户已确认实施：主Agent、05攻击编辑、06场景理解与防御编辑使用同一思考配置。新一轮案例3记录在`outputs/case3/20261111_thinking_profiles01/`，旧实验保留。

当前逐工具模型和思考强度可通过 `configs/api.json` 的 `tool_profiles`、`tool_efforts` 覆盖，未固定的工具才跟随公共选择。下表保留的是 2026-11-11 的实验配置记录，不表示当前机器正在使用这些参数；实际配置和调用以配置文件及对应运行日志为准。

## 2026-11-11 实验配置

| 配置 | 请求model | 思考字段 | 强度 | 工具传输 | 输出上限/请求超时 |
|---|---|---|---|---|---|
| `ds_official_flash` | `deepseek-flash` | `thinking.type=enabled` | `reasoning_effort=high` | auto＋本地原生校验 | 65536 / 600秒 |
| `ds_flash` | `deepseek-v4.1-flash-1017` | `thinking.type=enabled` | `reasoning_effort=max` | 同上 | 同上 |
| `kimi` | `kimi-k3` | 不发送旧thinking开关，保留原生思考 | `reasoning_effort=high` | 同上 | 同上 |
| `glm_flash` | `z-ai/glm-5.3-flash` | `thinking.type=enabled` | `reasoning_effort=high` | 同上 | 131072 / 2400秒 |
| `qwen_flash` | `qwen3.8-flash` | `enable_thinking=true` | `reasoning_effort=medium` | 同上 | 同上 |
| `mimo_pro` | `mimo-v2.5-pro` | 尚未纳入统一思考试验 | 未验证 | 渠道503，视觉能力待核实 | 保留原8192 / 240秒 |

只有DS官方配置使用`https://api.deepseek.com`；其余使用现有聚合地址`https://apicz.boyuerichdata.com/v1`。密钥变量分别为API_KEY_5、API_KEY_2、OPENAI_API_KEY、API_KEY_4、API_KEY_1、API_KEY_3；不保存实际密钥值。`active_profile`是公共选择入口，可通过逐工具配置覆盖。

65536是输出上限，并非每次必然生成的长度，也不等于上下文长度或各模型的独立思考预算。不同服务对思考和回答的计数规则可能不同，比较时保留用量与截断状态。

## Qwen最高档已核实

Qwen3.8 Flash原生支持`low`、`medium`、`xhigh`，最高是`xhigh`；兼容值`high`、`max`映射为`xhigh`。原最高档试验保存原生值xhigh；用户最新要求降一档，当前保存medium。通过OpenAI SDK的extra_body发送enable_thinking和reasoning_effort，不同时设置thinking_budget。[官方参数说明](https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions)

这补全了前一版审阅表中待核实的Qwen档位。默认/档位对应的思考预算由服务决定，不能把当前max_tokens=65536解释为Qwen实际思考长度。

## 各阶段如何保持一致

| 入口 | 思考设置来源 | 工具与会话处理 |
|---|---|---|
| 主Agent原生决策 | 选中的统一profile | auto；每轮只执行一个合法函数；完整收到流式结束后才允许执行 |
| 主客户端普通chat入口 | 同一profile | 按模型翻译思考字段；无工具请求不添加工具选择 |
| 05车辆选择 | 同一profile经适配器注入 | auto；保留对象要求、候选资格和参数检查 |
| 05轨迹构造/编辑 | 同上 | 原本指定某函数的阶段先缩小可见工具集；本地仍限制该阶段能执行的函数 |
| 06场景理解 | 同上 | 独立请求，无原生工具时不额外添加tool_choice |
| 06比较/编辑/提交 | 同上 | auto；保留图片、候选ID白名单、看图资格、预算和原生调用纠正 |

主Agent不再一刀切关闭thinking，也不再漏发思考强度。后台调用方的旧默认值经过统一策略覆盖。Kimi不发送不适用的旧thinking字段；Qwen使用其原生字段；DS/GLM使用thinking对象和顶层reasoning_effort。

不同模型使用不同字段表达相同的项目意图，不能把同名强度解释成相同算力。聚合渠道接受请求也不能单独证明思考强度按原厂含义生效；本轮验证记录发送字段、响应及用量，不解读或展示模型推理正文。

## 主模型连接重试

主Agent的`chat`和原生工具决策请求遇到`APIConnectionError`/`APITimeoutError`时，最多自动重试3次，退避为1、2、4秒。SDK仍保持`max_retries=0`，不与应用重试叠加；HTTP参数、鉴权及其他业务错误立即失败。流式请求只有尚未收到任何chunk时才允许重试，已有部分输出则直接失败，避免重复生成或重复工具调用。每次尝试独立归档，`attempt_index=1..4`并计入运行报告的重试次数。该传输重试不替代`decision_correction_attempts`的格式纠正，也不会重新执行已经成功的工具。

## 工具传输约束

官方DS的思考模式不支持required或指定函数的tool_choice。因此新配置采用auto，避免为强制工具而关闭思考。[官方约束](https://api-docs.deepseek.com/api/create-chat-completion/)

项目约束依然由本地执行：

- 必须得到合法的原生函数调用，正文JSON或代码块不会被执行。
- 指定阶段缩小可见工具集，并检查函数与候选资格；auto不增加可执行工具权限。
- 主Agent和06保留有界纠正；05未得到可解码原生调用时按执行异常处理，不把它记作轨迹无解。
- 不完整流式响应或截断结果不能触发工具执行；禁止并行执行多个模型工具动作。
- 主Agent需要连续纠正时保留服务返回的必要协议字段，不把思考字段拼进任务文本或长期状态。05/06仍以重建结构化状态为主，未改成无限追加历史。

旧的required_tool_thinking=disabled只作为未采用新统一策略的兼容选项保留。新启思考profile不能同时声明该关闭选项，冲突会在请求前拒绝。

## 新配置字段

| 字段 | 含义/允许值 |
|---|---|
| `request_dialect` | `thinking`（DS/GLM）、`kimi`、`qwen`；其他 managed dialect 会在请求前拒绝 |
| `thinking_mode` | enabled/disabled；Kimi K3只允许enabled |
| `reasoning_effort` | profile 原生值：thinking/kimi 为 low/high/max；qwen 为 low/medium/xhigh |
| `tool_choice_policy` | auto/preserve；新试验配置均为auto |
| `max_tokens` | GLM为131072，其余四个为65536 |
| `timeout` | GLM为2400秒，其余四个为600秒；与Agent轮数、工具总时限和仿真/渲染时间分开 |

配置校验按dialect拒绝不支持的档位，不将Qwen的max误当成其原生档位，也不默默降档。CLI新增对应字段，支持max/xhigh；切换时会清除后台遗留策略，防止上一个profile影响下一个。

当前公共覆盖值和逐工具覆盖值使用 low/medium/high：thinking/kimi 映射为 low/high/max，qwen 映射为 low/medium/xhigh。前端面板使用这三档项目强度；不把尚未实现的第四档或其他 dialect 当作可用能力。`run.json.llm.reasoning_effort`保存请求实际生效值；旧运行能否从日志恢复须核对当前前端入口，不能用当前配置倒推。

示例仅用于说明结构，现有配置已保存：

```json
{
  "request_dialect": "thinking",
  "thinking_mode": "enabled",
  "reasoning_effort": "max",
  "tool_choice_policy": "auto",
  "max_tokens": 65536,
  "timeout": 600
}
```

temperature/top_p没有统一硬塞数值，遵循各模型的模式约定。stream与原生调用收集保留；SDK自动重试不承担业务重试决策。

## 后续计时与单环节测试

按用户最新要求，后续主指标只记录Agent API请求elapsed_seconds的累计值（含网络等待与模型生成），不将构造器、FOCI执行、画图、候选检查、策略模拟或视频渲染耗时加入。历史整轮、工具总时长和Agent循环墙钟仍保留原义，不能直接当作API耗时。

Qwen xhigh与GLM high的新测试位于`outputs/case3/20261111_defense_api01/`：固定同一已验证攻击、场景理解、初始构造器和FOCI候选池，只运行真实防御SubAgent选择/编辑/提交和独立数值检查。不传入DS最终编辑答案，不重新运行05、VAD/PDM、准备阶段、视频或Main。每模型最多10轮，累计API预算600秒；不会自动扩大预算或重跑。该600秒是本次基准的累计限额，不改写profile的长期容量设置。

该单环节测试已完成：Qwen xhigh累计API **348.791秒/7调用**，从FOCI初选经两次速度编辑提交edit_002，进度62.7%、目标最小净距3.673m；GLM high累计API **450.987秒/6调用**，生成一个通过检查的新编辑后最终提交原候选constructor_020，进度59.7%、净距6.601m。两者独立检查通过、API错误0、源输入/代码指纹不变；未生成视频。详见[本轮报告](../outputs/case3/20261111_defense_api01/README.md)。

Kimi high也已在同一公共攻击/候选池完成一次仅防御补测：API累计553.298秒/9调用，API错误0；因3次无效参数调用耗尽调用纠错预算而未提交（两次FOCI多传reason，一次JSON损坏）。第8轮有可行速度编辑预览，但未正式生成候选。原输入/代码指纹一致，不视为网络故障或防御无解。[Kimi补测记录](../outputs/case3/20261111_kimi_defense_api01/README.md)。

用户要求Kimi再独立重跑一次，未修改输入/代码。本次Kimi high **184.517秒/4调用**成功：从constructor_020经FOCI生成两条新候选，提交edit_002，进度59.7%、目标净距8.019m，独立检查通过，API/无效调用均0。前一轮失败保留，两轮累计737.815秒/13调用；结果差异说明不能以一次成功推断稳定性。[第二次Kimi防御报告](../outputs/case3/20261111_kimi_defense_api02/README.md)。

用户确认下一批为每次累计API600秒，两路DS各测max/high，加Qwen xhigh、GLM high、Kimi high，共7组各3次、21次，并发2路。新批次`outputs/case3/20261111_defense_repeat01/`使用同一公共防御材料，保持产品代码与工具规则不变，只测防御loop；所有尝试分开记录，错误先暂停派发排查，不自动补跑。按组快照覆盖DS思考档位，不改全局配置。结果在该批README/comparison.json刷新，不用旧成功轮充数。

## 验证与实验边界

初轮145项相关离线回归通过；工作区提示词发生并行变化后，又复核当前13组相关测试，136项全部通过。覆盖主/子请求一致、档位校验、阶段工具与候选限制、切换清理、会话纠正和既有攻防流程。五个目标配置均通过真实主客户端的文字工具、图片工具、连续工具会话预检；请求使用实际目标档位和65536上限，未用低档预检替代。

这些检查证明字段和组合可调用，不证明某档一定优于另一档。案例3各轮由真实主Agent从相同原始任务开始，05内做WE/VADv2自测，成功后06编辑、独立检查和视频；业务失败、主动结束、API/协议错误均保留，不为得到成功结果重选候选或执行正文命令。

GLM首轮max请求约4.6秒收到响应头，持续生成到599.985秒，被原600秒总预算截断（29416个已接收片段）。这与无首包连接超时不同。因此先把GLM请求预算增到1800秒，单独运行`glm_flash_02`。该轮约591秒又明确耗尽65536输出上限，未形成完整原生调用。随后核对[官方128K输出上限](https://docs.z.ai/guides/vlm/glm-5.3-flash)，并通过现有渠道的131072上限文字/图片控制调用，最终设131072/2400秒，另开`glm_flash_03`。三个正式轮及主控曾发起但被取消的重复调用都保留，上述历史轮思考强度始终max。Kimi首轮另有600秒无响应头及Main连接断开，轻量文字/图片控制均成功，按服务/传输异常保留并安排一次防御恢复。

用户随后因耗时过长要求重新规划，旧完整队列已停止：GLM第三轮完成攻击，防御理解与初始构造/FOCI也已完成，但首次防御选择请求被用户中断，不能记为防御失败；Qwen正式轮和Kimi防御恢复未启动。按用户最新决定，GLM从max降至下一档high，Main/05/06共享该设置。为隔离变量，131072输出容量与2400秒请求时限暂保留，它们是上限，不强制用满；旧max实验快照不改写。新high配置的真实轻量文字/图片原生调用均通过（约8.7/4.7秒），不等于已完成新案例3测试。

测试期间工作区Main提示词和05/06工具说明发生并行修改；第一轮Main重连跨过变化。原始任务/输入相同，但不能将本批当作严格的同版本模型排名。批次`workspace-change-evidence.json`保留指纹与说明。

MiMo仍有两项前置问题：聚合渠道503，以及原厂集成资料把V2.5 Pro列为文本生成、V2.5列为图像多模态。未核实前不默认将Pro用于05/06图像任务，也不自动换型号冒充同一模型结果。[官方型号说明](https://github.com/XiaomiMiMo/awesome-mimo-agent/blob/main/docs/cursor.md)

原厂依据：[DS思考参数](https://api-docs.deepseek.com/guides/thinking_mode/)、[Kimi K3](https://huggingface.co/moonshotai/Kimi-K3#6-model-usage)、[GLM思考参数](https://docs.z.ai/guides/capabilities/thinking)、[Qwen参数](https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions)。

实现位置：[共享策略](../mcp_servers/scene_editing/core/adversarial/llm_request_policy.py)、[主客户端](../agent/llm_client.py)、[后台注入](../mcp_servers/scene_editing/adapter/service.py)、[后台读取](../mcp_servers/scene_editing/core/adversarial/llm_profile.py)、[后台兼容入口](../mcp_servers/scene_editing/core/adversarial/kimi_request_parameters.py)、[06工具传输](../mcp_servers/scene_editing/core/adversarial/ego_tool_transport.py)。操作命令见[API配置说明](API_CONFIGURATION.md)。

## 重复评测暂停状态

用户已要求暂停，全部本批进程停止。有效5组15次中已结束7次：GLM high 2/2提交（293.128/331.260秒API），Kimi high 1/1（290.443秒），第三方DS high 1/1（281.731秒），Qwen xhigh 0/2（各600秒预算退出），第三方DS max 0/1（600秒预算退出）。官方DS两档因余额不足按用户要求排除。另两轮用户中断不计失败，剩余8次等待明确恢复。详[暂停交接](../outputs/case3/20261111_defense_repeat01/SESSION_HANDOFF.md)与resume-plan.json，不能直接重启旧队列。

## 重复评测最终结果（2026-11-12）

用户恢复后补齐8轮，和原7轮合并，每组3次共15条有效尝试，全部结束。累计API预算每次600秒、并发2，其他阶段不计时。Kimi high 3/3（API中位244.731秒），第三方DS high 3/3（281.731秒），GLM high 2/3（全轮中位331.260秒），Qwen xhigh与第三方DS max均0/3预算内提交（各轮约600秒）。成功8条均独立验收通过，7条预算退出；官方DS两档因余额不足排除。Qwen/GLM末轮有合格新编辑但后续决策被截止，未提交，不视为防御无解。详[最终报告](../outputs/case3/20261112_defense_repeat_resume01/RESULTS.md)。暂停期间外围版本变化与接口等待均单独记录，不做严格因果或泛化排名。

## 最新追加测试：Qwen medium、官方DS high

用户在官方DS充值后要求只测high，Qwen从xhigh降一档medium。两个profile已持久更新，轻量文字/图片预检通过。本批两组各3次、并发2、每轮API累计600秒，固定公共防御输入/代码，不跑max或攻击/模拟/视频。报告按用户要求逐轮API总耗时与编辑尝试次数、三次合计和算术平均（含失败）呈现，另列合格新候选数；历史15轮不覆盖。[活动报告](../outputs/case3/20261112_defense_medium_official01/README.md)。

## 追加试验最终结果与新计时展示

Qwen medium与充值后的官方DS high各3次均结束。Qwen API364.695/385.808/600.002秒，编辑尝试2/2/1次，平均450.168秒（含预算退出），2/3交付。官方DS API144.612/315.901/399.397秒，编辑尝试1/2/5次，平均286.637秒，3/3交付。本次未运行任何max配置，旧15轮仅作为历史对照。

按用户最新要求，报告每格为每轮API总耗时/编辑尝试次数，另列三次API合计与算术平均（非中位数，失败与预算退出都纳入）。编辑尝试含预览/失败/无效参数，不含选轨/看图/提交，一次FOCI生成多候选仍记一次调用；合格新候选数另列。详[完整21条三轮对照](../outputs/case3/20261112_defense_medium_official01/README.md)。
