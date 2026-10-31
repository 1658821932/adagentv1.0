# 纯文本 LLM 工具开发规范

当前统一覆盖三个工具：tool02 需求生成、tool03 场景构思、tool07 编辑计划。
它们通过公共运行器执行一次受约束的文本转换；WFA、SmartEdit、视觉验证和 GPU 后端保留各自实现。

## 文件职责

每个工具目录具有：
- description.py：主 Agent 看到的用途和调用时机；参数说明从 Schema 生成。
- schemas.py：Input、Response、Output，以及必要的输入规范化和业务校验。
- prompt.md：工具内部模型的任务指令；不包含 API 密钥。
- config.json：模型角色、提示词版本、超时、输出长度和校验重试上限。
- real.py：声明 LLMTool，绑定上述契约及可选的上下文附加/业务校验函数。
- mock.py：确定性结果，通过与 real 相同的输入、输出和业务校验。
- __init__.py：包说明。

公共实现位于 tools/llm_runtime.py。工具字段只在 schemas.py 定义，description.py 生成
inputs、outputs、input_schema 和 output_schema；不再手写第二份参数清单。

## 调用与返回

统一函数签名为 run(input_data: dict) -> dict。调用时使用注册表，或明确导入 real.run / mock.run；不提供 tool.py 旧入口（与其他工具一致，已删除）。
推荐调用注册表，以便遵循明确的 mock/real 模式：
```python
from tools.registry import call_tool

result = call_tool(
    "scene_requirement_generator",
    {"failure_analysis": {"failure_reason": "Delayed response to a slowing lead vehicle."}},
    mode="real",
)
```
成功返回 status、tool、result、meta。业务字段仍放在 result，不增加通用 data 层，
保留原链路的 scene_requirement、scene_plan、edit_plan 和 action 等字段。
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
  原始 source_analysis 由代码附加，模型不得重写。
- tool03：接受需求对象、成功的上游 envelope 或直接的自然语言需求；
  source_requirement 由代码附加；编辑建议统一使用 edit_ideas，不再返回 simple_edits。
- tool07：必须通过 edit_request 提交编辑请求，不接受 edit_ideas 参数别名。支持实际实例列表及发现工具的成功返回。
  target_instances 和 operations.target 必须引用输入提供的实例 ID；无 ID 时只能使用空列表/null，
  角色放在 parameters.target_role。帧编号不能当实例 ID。


mock 同样要求合法输入。它只验证接口与执行路径，不证明场景、视觉或驾驶策略真实有效。

恢复规则直接位于 agent/orchestrator.py 的主提示词中；不再提供独立 tool11 或兼容入口。流程与测试见 [架构文档](ARCHITECTURE.md)。

## 配置与密钥

根 .env 只存密钥，沿用已有 OPENAI_API_KEY / LLM_API_KEY。
默认模型与端点来自 configs/agent.json 的 llm 块，兼容原环境变量覆盖规则。
llm.roles 定义 planner、editor 等角色；空对象表示沿用默认模型。
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
smoke 通过公开注册表调用三个工具，其中 tool02 输出直接传给 tool03。恢复决策由主 Agent 提示词负责，使用 tools.scripts.smoke_agent_recovery 单独测试。报告默认写入被 Git 忽略的 logs/llm_tools_smoke/。
真实 smoke 不启动 DGGT/WorldEngine，不读取视频，不证明完整场景生成链路或模仿学习收益。

## 新增同类工具

1. 在 schemas.py 定义 Input、Response；只有代码附加上下文时才另定义 Output。
2. 编写内部 prompt.md 和非敏感 config.json。
3. 在 real.py 声明 LLMTool；仅在必要时增加业务校验或上下文附加函数。
4. 提供符合相同契约的 mock.py；description.py 使用 make_description。
5. 注册 real/mock 入口，补充非法输入、输出纠错和关键业务约束的测试。
