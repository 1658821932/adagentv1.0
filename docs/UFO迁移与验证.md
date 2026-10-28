# UFO 重建与米制编辑迁移

## 来源与兼容方式

- 功能源：服务器 `adagentv1.0_ufo_metric_update` 的 `710ccd02b860b1cca128bdd4452ee7a02eab3512`。
- 合入基线：目标仓库 `1658821932/adagentv1.0` 的 main `07617b3718099e286d59673a52f25e1e81e76941`。
- 两边没有共同 Git 祖先；在 main 上新建功能分支，迁移 UFO 与必要的 SAM 目标选择/绑定代码，避免覆盖 main 的服务适配层重构。
- 保留 main 的 typed LLMTool、有限纠错、模型角色配置和主 Agent 恢复逻辑；没有恢复已删除的 recovery_decider，也没有替换 DGGT、Sentry 或 WorldEngine 服务适配实现。
- UFO 后端实现放在 `mcp_servers/ufo/adapter.py`，tool05/06/08/09 只分发后端。UFO 计划单独扩展主分支结构化规划契约，DGGT 原规划契约保持可用。
- Agent 默认 UFO；配置 `agent.reconstruction_backend`、环境变量 `RECONSTRUCTION_BACKEND` 或 CLI `--reconstruction-backend` 选择。环境变量优先于 JSON，CLI 设置本次进程的环境变量。直接调用现有 DGGT 工具且未传 backend 时保留原 DGGT 行为。

## 本次 CPU 验证

使用 Python 3.13 的独立虚拟环境，安装 `requirements-ufo-test.txt`。测试目录为 `tests`，不包含第三方依赖项目的 GPU 测试。

```bash
python3 -m venv /tmp/adagent-test
/tmp/adagent-test/bin/python -m pip install -r requirements-ufo-test.txt
# 原主分支部分适配测试要求项目中有解释器路径，但通过 mock 隔离了实际模型推理。
mkdir -p environments/agent environments/video_retriever
ln -s /tmp/adagent-test environments/agent/.venv
ln -s /tmp/adagent-test environments/video_retriever/.venv
TMPDIR=/tmp /tmp/adagent-test/bin/python -m pytest tests -q
```

仅在路径不存在时创建这些环境链接，不覆盖已有运行环境。macOS 可用 `TMPDIR=/private/tmp`，以避免 `/var` 与 `/private/var` 的路径别名影响旧测试中的字符串断言。

覆盖 UFO/DGGT 后端固定与路由、失败重建后旧场景失效、40 帧限制、恢复阶段、前端估计米制单位与刚体导出验证、独立深度拒绝报告、已有目标刚体平移、资产尺寸/轴向/接地点、插入及自车平移参照、道路代理、实际位移审计、视频比较，以及 main 现有 LLMTool 与恢复回归测试。另覆盖 UFO 结构化计划到 config 的字段传递与资产/目标不可替换检查。

## GPU 证据边界

历史 GPU 结果记录见 [2026-10-27 GPU 审核](UFO_GPU审核记录_20261027.md)。它验证的是源版本 `710ccd0` 的工具执行路径，不是本次迁移分支的 GPU 重跑，也不是自然语言端到端理解准确率评测。

- 记录中新插入车辆右移约 `2.9999998785` 个估计米制单位，逐点数值误差小于 `4.8e-7 m`。
- 原车右移仍有重建残影，不能标为视觉通过。
- 前端单位验证不认证真实绝对距离；独立深度拟合失败仍为 rejected。
- `planar_proxy` 必须显式启用并承认其改变静态道路几何。
- 不提供闭环策略、碰撞安全或自动车道保持保证。

服务器未跟踪的 `assets/1.ply`、模型权重、已有环境、视频及输出未纳入代码提交。新增资产应自行提供明确的单位/设计尺寸和清单。
