# 主 Agent 对话续跑

## 当前实现范围

后端已支持从已有运行恢复完整决策状态，新的追问会与原任务一起传给主 Agent。每轮使用独立日志目录，原工具输出文件仍以原始 `result_ref` 传递，主 Agent 按现有压缩逻辑接收决策摘要，不复制图片或模型私有推理到任务历史。

2026-11-14 重新核对：后端 CLI 支持续跑；前端独立模块仍保留，但共享 `app.js` / `server.mjs` 曾被覆盖，当前未完整接入续跑、活动记录和图片预览。部署前须验证实际入口，不能仅凭模块存在或旧测试记录宣称浏览器续跑可用。下面的前端使用说明记录 2026-11-12 完整接入版本的行为，不代表当前共享页面已恢复全部功能。

## 前端完整接入版本的使用记录

1. 强制刷新页面（Ctrl+Shift+R），在“历史运行”选择要接续的旧任务。
2. 在对话框发送“错误已修复，请继承已有结果继续”，或明确限制本轮范围，如“只重试失败分析”。
3. 主 Agent 恢复原任务、已有工具结果和追问历史；本轮会建立独立日志并关联父运行。提交追问后仍在当前页面追加新消息和工具卡片，不再清空之前的执行记录。历史菜单按 `conversation_id` 合并同一任务的多轮运行；点击后沿父运行链恢复各轮消息、工具卡片和左侧调用流程，并定位到最新一轮。对话标题栏显示该轮实际主模型；失败总结提供可展开详情，续跑仍在对话框中输入并发送。
4. 要开始不同任务时点击“新对话”，再输入需求，不会带上原运行。

执行期间输入框、历史切换及“新对话”按钮暂时禁用；可以停止或等本轮结束后再追问。停止会先终止主 Agent，再停止该执行范围内所有 worker 及登记的后端子进程组，并写入运行和工具的中断状态；不会只隐藏前端执行状态。发送失败会保留输入和父运行，服务端未支持续跑时拒绝降级成新任务。示例仍是演示，不会发起真实工具。

主对话滚动时，左侧工具会标记当前浏览位置并保持在可视区域；该高亮独立于工具执行状态。历史面板点击后立即展开，优先显示缓存并后台刷新，连续点击复用同一请求。对话标题栏显示主模型及实际思考强度：新运行读取运行快照，旧运行回退到真实请求日志，未记录时不按当前配置猜测。

工具图片和视频统一放在步骤卡片的“产物预览”中。检索视频、原始场景关键帧、攻击/专家轨迹检查图和实际渲染回放分别标注来源；几何检查图不会标成相机渲染。视频点击后才加载，折叠图片展开后才请求，加载失败可重试。浏览器不支持的视频由服务端按需转为 H.264，转码使用临时文件原子发布缓存，避免中断后的残缺文件被重复播放。

## 命令行

```bash
environments/agent/.venv/bin/python -m agent.main \
  --resume-from run_20261112_060801_663582 -- \
  '攻击编辑的程序错误已经修复，请从05重试，复用之前完成的阶段。'
```

这会调用真实模型并继续执行用户要求的阶段。去掉 `--resume-from` 则是新任务。`--` 后的内容始终按用户消息解析，不会被识别为命令行参数。

Python 调用：

```python
Agent(config).run('程序已修复，请继续', resume_from='run_20261112_060801_663582')
```

## 保存与恢复

- `state.task` 保留原始任务；`current_request` 是本轮追问；`user_messages` 按顺序保存历史追问。新指令可以调整后续范围，例如“先只重试攻击编辑”。旧的 finish/stop 不代表本轮应该直接结束。
- 优先读取父运行 `final_state.json`，其次 `state.json`。如果进程中断在检查点写入之前，恢复器从父运行 `tools/step_*/input.json` 和 `output.json` 补齐已记录的工具调用及结果，兼容旧日志。
- 新 `run.json` 保存 `parent_run_id`、`conversation_id`、`original_task`、`execution_scope`、继承历史条目数及进程身份；`initial_state.json` 保存本轮起点。旧日志内容不覆盖。
- 新步骤编号接在已有步骤之后；每次追问按当前配置提供新的主循环步数预算。模型配置也按当前配置加载，配置快照独立保存。
- 同一对话继承原工具任务 scope：相同已完成调用仍可复用；失败任务需要模型显式使用 `execution.retry=true` 才会新执行，不会因续跑自动清空缓存或重跑前置阶段。
- 如果上轮在收集工具结果前中断，先从持久化 JobStore 找到已有 job，接回其真实结果。已完成的结果直接读取；运行中的继续等待。没有可确认的 job 时拒绝自动重复有副作用的调用，并报告不确定性，避免重复训练或仿真。
- 独立运行锁和对话锁阻止同一活动对话同时续跑。锁在主进程退出后自动释放；服务器重启不要求清理锁文件。旧运行没有锁时仍可恢复，其活动状态需由前端已知进程记录核对。
- 当前不支持在主 Agent 正忙时注入追问。先等本轮结束，或停止后再续跑；后台工具若仍运行，会接回它而不是另开一份。

## 前端接线与维护

独立辅助模块已提供：

- `frontend/conversation.js`：浏览器可用的 `conversationSubmission(text, viewingRun)`，以及建议实际接线使用的 `submitConversation(request, text, viewingRun)`。选中历史或当前对话时附带 `resume_from`，新对话为 null 时不附带；发送续跑前检查服务是否支持，旧入口不支持时拒绝提交，避免悄悄新开无上下文任务。
- `frontend/run-continuation.mjs`：Node 使用的 `prepareRunRequest(body, logsRoot)`、`agentArguments(task, resumeFrom)`、`runIsAlive(metadata)`。前两者校验父运行、拒绝越界路径并构造正确 CLI 参数；第三者结合 PID 与进程启动时间判断服务重启后的运行是否仍活着。
- `frontend/run-coordinator.mjs`：独立的 `createRunCoordinator({root, logsRoot, python, activeRuns})`，通过 `submit(body)` 完成 POST 校验、单运行与启动互斥、CLI 启动及进程注册。传入共享服务的 `activeRuns` Map，继续兼容现有停止/控制台逻辑。`conversationDetails(runDir)` 为 GET 详情补充继承信息和 `continuation_supported=true` 标志。

当前接线结构（维护时须保持两端一起生效）：

1. 浏览器原 POST 调用改为 `submitConversation(api, text, viewingRun)`；`api` 应继续使用能保留 VSCode 代理路径前缀的现有客户端。点击“新对话”继续把 `viewingRun` 设为 null。
2. Node 初始化 `const coordinator = createRunCoordinator({root: ROOT, logsRoot: LOGS, python: PYTHON, activeRuns})`；POST 解析 JSON 后调用 `await coordinator.submit(body)`，返回结果。异常响应状态用 `error.status || 500`。启动互斥、父运行校验、正确 CLI 参数及活动注册由协调器完成，旧 `launchRun` 不应再与之重复启动。
3. 保持现有 `cwd`、Python 路径、密钥环境和配置选择功能；协调器使用参数数组启动 CLI，不拼 shell 字符串。它会读取持久化进程身份防止服务重启后重复提交。
4. GET 详情返回值末尾合并 `...await conversationDetails(runDir)`。此修改要和 POST 路由一起生效，再发布 `continuation_supported` 标志；避免只开启能力标志却仍忽略 POST 的父运行。从当前页面发起续跑时使用 `continueRunView` 保留已有消息与工具卡片并追加本轮内容；从历史菜单打开对话时使用 `loadConversationChain` 和 `beginConversationView` 恢复完整父运行链。主工具步骤的 `activities` 从 `operations/` 与 `tool_events/` 提取公开动作、状态和耗时，用于 05/06 气泡内的子 Agent 时间线，不回传模型私有推理或完整候选数据。
5. GET 运行列表与提交前活动检查可结合 `runIsAlive`，防止前端服务重启后把仍在执行的主 Agent 当成可重新启动的旧任务。
6. 提交失败时保留用户输入和原 `viewingRun`；不要退回无上下文的新任务。新旧运行保持各自的日志和耗时，避免把旧 API 消耗再算入本轮。

续跑是否调用工具、先修哪一步由主 Agent 根据历史和最新消息决定；框架不把整个流程硬编码重跑。恢复原先已提交的 pending job 属于取回真实结果，不重新发起同一操作。

## 离线验证

- `tests/test_continuation.py`：继承历史目标与引用、跨多轮追问、旧检查点补全、实际持久化 mock job 接回、父日志不变、活动会话互斥、路径边界、缺失 job 不重复执行，以及 CLI 解析。
- `frontend/run-continuation.test.mjs`：追问携带父运行、新对话清空、CLI 字面文本、父运行校验、目录边界、服务重启后的进程识别。
- `frontend/continuation-http.test.mjs`：分别测试独立临时入口和实际生产 `server.mjs`，以临时目录/本机随机端口启动真实 Python CLI 与 harness；只替换模型和工具为离线替身。验证首轮 → 追问继承、不重复分析 → 新对话重置、父日志不变、重复提交 409 和错误输入 400；配置/脚本路由与能力标志同时验证。
- `frontend/session-ui.test.mjs`：执行实际页面发送和新对话处理函数，验证父运行传递、失败保留输入、重复点击保护、新对话清空和演示不发真实任务。此为 DOM 桩测试，不是完整浏览器点击测试。

这些测试使用离线模型替身和 mock 工具，不发送真实模型请求。

```bash
node --test frontend/api-client.test.mjs frontend/config-panel.test.mjs \
  frontend/run-continuation.test.mjs frontend/session-ui.test.mjs frontend/continuation-http.test.mjs
```

上线验证：67项Python回归、19项Node测试通过；实际5173 `/api/config`、历史、静态脚本及旧运行的 `continuation_supported=true` 均验证。未用真实模型跑攻防。进程信息以 `frontend/.cache/frontend-server.json` 为准，仅监听127.0.0.1。
