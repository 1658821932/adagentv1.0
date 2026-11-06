# 提示词组织

## 三层职责

| 层 | 内容 | 不应重复维护 |
|---|---|---|
| 主 Agent | 任务调度、结果读取、完成条件、纠错 | 每个工具的参数和具体算法 |
| 工具说明 | 用途、调用时机、上下游映射、专属规则 | 手写第二份参数/输出 Schema |
| 内部任务 | 该次 LLM 调用的判断标准与业务步骤 | 主 Agent 的全流程调度 |

主 Agent 提示词在 agent/prompt.md，由 agent/orchestrator.py 加载。
9 个 tools/toolXX_*/description.py 统一使用 make_description，结构为：
name、description、when_to_use、handoff、rules，以及从 schemas.py 生成的字段说明和 Schema。

主 Agent 接收精简目录：保留输入 Schema、handoff、rules 和输出字段名，不重复发送完整输出 Schema 或输出说明表；输出如何传递只在 handoff 中说明。
字段类型、必填项、默认值和字段语义维护在 schemas.py。输出结果保留在日志，通过 result_ref 传递完整对象。

## 内部任务入口

| 调用 | 文件 | 主要职责 |
|---|---|---|
| 场景构思 tool02 | tools/tool02_scene_instance_generator/prompt.md | 一个目标、一个全局自车行为、三个底视频与编辑策略候选 |
| 视觉验证 tool10 | tools/tool10_scene_validator/prompt.md | 原始需求与编辑渲染的一致性、可见质量和逐帧证据 |
| 检索计划 | mcp_servers/sentry_video_search/core/prompts/search_plan.md | 解析需求及生成平台检索词 |
| 下载预筛 | mcp_servers/sentry_video_search/core/prompts/preview_review.md | 宽松判断候选是否值得下载；要求汽车视角 |
| 第一视角分类 | mcp_servers/sentry_video_search/core/prompts/first_person.md | 通用第一视角与画面清晰度；骑行/行人第一视角仍属于此分类 |
| WFA 行为取证 | mcp_servers/worldengine_failure_analysis/core/prompts/forensic_analysis.md | 图像观察、候选、数值核验与最终原因 |
| WFA 图片观察 | mcp_servers/worldengine_failure_analysis/core/prompts/image_review.md | 单组图片观察，不给最终原因 |
| WFA 案例归组 | mcp_servers/worldengine_failure_analysis/core/prompts/scene_groups.md | 既有分析按动作和环境归组，不重新归因 |

两个 native core 的提示词随各自代码保存，脱离 AdAgent 时仍可读取。攻击车/自车编辑的算法后端尚未接入，本仓库只提供它们的调用说明和接口，不新增假定的内部算法提示词。

## 请求如何组装

- 主 Agent：system 放主提示词；user 放 task、available_tools、current_state。
- tool02：system 放业务提示词和运行器追加的 Response JSON Schema；user 放规范化输入。JSON 格式规则由运行器提供，业务提示词不再手写输出示例。
- tool10：任务模板填入已知缺陷、原始需求和从 VisualEvaluation 生成的响应 Schema，再附关键帧。model、sampled_frames 由运行器记录，不要求模型填写。
- 检索：system 放任务规则；user 放需求/候选元数据或关键帧。检索计划、下载预筛与第一视角分类保留各自职责。
- WFA：行为取证与图片观察使用各自 system 提示词，user 只传本轮证据、问题和动作协议。公共规则不再同时塞进 system 和 user。

参数或输出不合法时，纠错消息仅说明当前错误并要求修正，不重发完整业务规则。
更改 tool02 提示词时更新 config.json 的 prompt_version，运行记录同时保存提示词哈希。

## 修改与验证

修改工具用途或衔接，编辑对应 description.py；修改字段定义，编辑 schemas.py；修改任务判断，编辑对应 Markdown。
不要把同一规则同时写进主提示词、工具说明和内部任务提示词。
在根目录执行：

```bash
environments/test/.venv/bin/python scripts/generate_interface_docs.py
environments/test/.venv/bin/python -m pytest -q
environments/test/.venv/bin/python -m tools.scripts.smoke_llm_tools --mode mock
```

测试检查契约、请求组装和流程衔接；真实生成与视觉质量需另行评估。
