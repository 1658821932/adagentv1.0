# 提示词组织

主提示词位于 agent/prompt.md，只负责工具调度、结果读取、完成条件及错误恢复。
九个已启用工具的 description.py 使用 make_description，由 schemas.py 生成输入输出说明，handoff 说明传递关系。

| 调用 | 文件 | 职责 |
|---|---|---|
| tool02 场景构思 | tools/tool02_scene_instance_generator/prompt.md | 一个目标、一个全局自车行为和三个候选 |
| tool04 快速粗规划 | tools/tool04_edit_planner/prompt.md | 基于检索结果、全时段车辆标注及源关键帧判断可编辑性，程序生成 tool05 attack_input |
| 检索计划 | mcp_servers/sentry_video_search/core/prompts/search_plan.md | 检索需求解析 |
| 检索预筛 | mcp_servers/sentry_video_search/core/prompts/preview_review.md | 源视频候选判断 |
| 第一视角判断 | mcp_servers/sentry_video_search/core/prompts/first_person.md | 检索辅助判断 |
| WFA 取证 | mcp_servers/worldengine_failure_analysis/core/prompts/forensic_analysis.md | 图像与数值取证 |
| WFA 图片观察 | mcp_servers/worldengine_failure_analysis/core/prompts/image_review.md | 图片证据观察 |
| WFA 分类 | mcp_servers/worldengine_failure_analysis/core/prompts/scene_groups.md | 已有案例归组 |

攻防后端内部提示词位于 scene_editing/core/adversarial 的对应 Agent 模块，由后端独立管理。
原独立视觉验收工具的源码和提示词均已删除，不在 tool07 或 tool06 中继续调用。

tool02 与 tool04 共用 tools/llm_runtime.py：system 为任务提示词与 Response Schema；user 为任务上下文。
tool04 的 user 内容可包含实际源关键帧；确定性的资源缺失检查在请求模型前执行。
API 统一使用根 configs/api.json 的 Kimi 配置；阿里凭证仅用于检索嵌入和重排序。
修改任务提示词时更新各自 config.json 的 prompt_version。不要重复注入工具内部提示词到主 Agent。

接口文档更新：environments/test/.venv/bin/python scripts/generate_interface_docs.py。

## tool04 视觉观察

初始从完整绑定场景均匀选择12个时刻，固定看CAM_F0前视角，按时间顺序每4帧拼成一张2×2图，通常共3张。短场景只使用真实存在的帧；缺图单元标记缺失，不复制其他帧补齐。
VLM可在工具内部返回request_views，指定实际相机名称和观察目的。每轮最多请求2路相机，最多补看2轮；补看沿用相同时间点，不新增主Agent工具或对外输入字段。最终仍返回原有粗规划结果。
设置位于tools/tool04_edit_planner/observation.json；拼图和实际补看记录保存在忽略上传的logs/tool04_observations/下。evidence.visual_observation记录图片、是否成功送达、补看请求及model_calls；meta.attempts仍是外层格式校验尝试数。


---

# LLM 工具开发规范

当前统一覆盖 tool02 场景与候选方案生成，以及 tool04 检索后可编辑性判断与粗规划。
它通过公共运行器执行一次受约束的文本转换；WFA 和 GPU 后端保留各自实现；旧独立视觉验收已删除。

## 文件职责

每个工具目录具有：
- description.py：主 Agent 看到的用途、调用时机、handoff 衔接和 rules 专属规则；字段说明从 Schema 生成。
- schemas.py：Input、Response、Output，以及必要的输入规范化和业务校验。
- prompt.md：工具内部模型的任务指令；不包含 API 密钥。
- config.json：模型角色、提示词版本、超时、输出长度和校验重试上限。
- real.py：声明 LLMTool，绑定上述契约及可选的上下文附加/业务校验函数。
- mock.py：确定性结果，通过与 real 相同的输入、输出和业务校验。
- __init__.py：包说明。

提示词分层及各文件入口见 [PROMPTS.md](PROMPTS.md)。

公共实现位于 tools/llm_runtime.py。工具字段只在 schemas.py 定义，description.py 生成
inputs、outputs、input_schema 和 output_schema；不再手写第二份参数清单。

## 调用与返回

统一函数签名为 run(input_data: dict) -> dict。调用时使用注册表，或明确导入 real.run / mock.run；不提供 tool.py 旧入口（与其他工具一致，已删除）。
推荐调用注册表，以便遵循明确的 mock/real 模式：
```python
from tools.registry import call_tool

result = call_tool(
    "scene_instance_generator",
    {"failure_analysis": {"failure_reason": "Delayed response to a slowing lead vehicle."}},
    mode="real",
)
```
成功返回 status、tool、result、meta。业务字段仍放在 result，不增加通用 data 层，
tool02 输出 scene_requirement、target_effect 和 candidates。
meta 包含 mode、model、prompt_version、prompt_sha256、attempts、elapsed_seconds；
真实 SDK 返回 usage 时附 last_call_usage（仅最后一次模型调用的用量，不是全部尝试总量）。

错误返回 status="error"、tool、error 和 meta。error.type 区分：
- input_validation_error：在调用模型前拦截缺字段、错误类型和未知参数。
- configuration_error：配置、提示词、角色或模型客户端初始化失败。
- llm_request_error：模型端点请求失败；不会转成 JSON 纠错请求。
- output_validation_error：模型回复未通过结构或业务校验，且校验纠错次数已用尽。

最大校验纠错次数由 validation_retries 控制，默认 1，最多 2。
SDK 自身的传输重试策略独立于此计数。
为兼容现有第三方 OpenAI-compatible 端点，当前通过提示词提供 JSON Schema 并在本地强制校验；
未假定该端点支持原生 strict Structured Outputs。
不完整 JSON、非对象 JSON 和类型错误不会作为成功结果放行。

## 标准输入与业务约束

- tool02：接受非空 failure_analysis、tool01.result 或成功的 tool01 完整 envelope；
  可选 variation_hints、existing_instances；输出仅 scene_requirement、target_effect、candidates（三个候选，各含 base_video_requirement、edit_strategy）。
  原始分析及调用输入保存在运行记录，路径见 meta.record_dir。
- 原 tool03 已合并到 tool02，不再独立调用；将 candidates[i].base_video_requirement 直接传给 tool03.requirement。



mock 同样要求合法输入。它只验证接口与执行路径，不证明场景、视觉或驾驶策略真实有效。

恢复规则位于 agent/prompt.md 中；不再提供独立 tool11 或兼容入口。流程与测试见 [架构文档](ARCHITECTURE.md)。

## 配置与密钥

根 .env 只存密钥，沿用已有 OPENAI_API_KEY / LLM_API_KEY。
默认模型与端点统一来自 configs/api.json 的 kimi profile；configs/agent.json 的 llm.api_profile 引用它，兼容原环境变量覆盖规则。
llm.roles 当前定义 planner 角色；空对象表示沿用默认模型。
角色只允许 model、timeout、max_tokens，不接受密钥或端点字段。
各工具 config.json 选择 model_role，指定 timeout/max_tokens，并管理 prompt_version 和 validation_retries。

角色中显式指定的 model 覆盖默认模型；工具 config.json 显式设置的 timeout/max_tokens 优先于角色和默认值，省略时继承角色或全局值。
修改角色不改变主 Agent 的模型。
修改 prompt.md 时应更新 prompt_version；运行时同时记录文件 SHA-256，便于追溯。
_llm_client 是仅供 Python 测试注入的内部入口，不进入公开参数 Schema，也不会发送给外部模型。

## 安装与验证

从项目根目录执行：
```bash
environments/agent/.venv/bin/python -m pip install -r requirements.txt

# 离线工具契约 smoke：不调用外部模型
environments/agent/.venv/bin/python -m tools.scripts.smoke_llm_tools --mode mock

# 真实 LLM smoke：发送合成文本，使用项目已配置的端点和密钥
environments/agent/.venv/bin/python -m tools.scripts.smoke_llm_tools --mode real

# 独立测试环境
python3 -m venv .venv-test
.venv-test/bin/python -m pip install -r requirements-test.txt
.venv-test/bin/python -m pytest tests/ -q
```
smoke 通过公开注册表调用 tool02；其候选检索描述直接传给 tool03（mock），real smoke 只检验文本生成和检索输入契约。
报告默认写入被 Git 忽略的 logs/llm_tools_smoke/。真实 smoke 不启动 GPU 仿真，也不证明编辑或后训练收益。

## 新增同类工具

1. 在 schemas.py 定义 Input、Response；只有代码附加上下文时才另定义 Output。
2. 编写内部 prompt.md 和非敏感 config.json。
3. 在 real.py 声明 LLMTool；仅在必要时增加业务校验或上下文附加函数。
4. 提供符合相同契约的 mock.py；description.py 使用 make_description。
5. 注册 real/mock 入口，补充非法输入、输出纠错和关键业务约束的测试。

## tool04 视觉观察

初始从完整绑定场景均匀选择12个时刻，固定看CAM_F0前视角，按时间顺序每4帧拼成一张2×2图，通常共3张。短场景只使用真实存在的帧；缺图单元标记缺失，不复制其他帧补齐。
VLM可在工具内部返回request_views，指定实际相机名称和观察目的。每轮最多请求2路相机，最多补看2轮；补看沿用相同时间点，不新增主Agent工具或对外输入字段。最终仍返回原有粗规划结果。
设置位于tools/tool04_edit_planner/observation.json；拼图和实际补看记录保存在忽略上传的logs/tool04_observations/下。evidence.visual_observation记录图片、是否成功送达、补看请求及model_calls；meta.attempts仍是外层格式校验尝试数。