# AdAgent

面向端到端自动驾驶的闭环递归自进化（RSI）智能体框架：基于 WorldEngine 3DGS 闭环仿真，自动完成**失败诊断 → 场景检索与规划 → 3DGS 高难场景合成 → 闭环评测与奖励采集 → 带 Reverse KL 锚定的 RLFT 后训练**。

---

## 核心工具（Tool 01 – Tool 08）

主 Agent 按任务目标编排以下 8 个工具，各工具通过统一 Schema 契约与 `result_ref` 衔接：

| 编号 | 工具名 | 核心职责 |
| :--- | :--- | :--- |
| **01** | `failure_analyzer` | 分析 WorldEngine 闭环失败场景：基于带车道中心线的高清 BEV 与物理测量提取最小道路形态、相对车道位姿与自车系相对坐标，并结合违规前 8 相机图补充天气与环境。 |
| **02** | `scene_instance_generator` | 将失败分析转化为统一目标场景描述与 3 套差异化实现候选（优先新增车辆），生成精简底场景检索词。 |
| **03** | `video_retriever` | 基于多模态向量检索与重排序，从本地 3DGS 数据集检索并绑定完整的多相机场景与标注资产；支持跨场景已用底场景硬去重池（`exclude_scene_tokens`）。 |
| **04** | `edit_planner` | 双 BEV 对照规划器：对比原失败 BEV 与底场景原始 BEV（含车道中心线、3DGS 路面高程与可放置车道），判定可编辑性与速度差（按需触发自适应速度对齐），输出确定性编辑方案（支持双车与 `ego_only`）。 |
| **05** | `attack_vehicle_editor` | 3DGS 场景编辑器：支持 `add_vehicle`、`edit_existing`、`ego_only` 三种模式；结合路面高程、运动学与地图约束求解攻击车轨迹，支持自适应速度对齐（自车前 1.5s 历史调速并接收 `ego_reference_frames_path` 真实闭环轨迹参考），导出冻结场景包。 |
| **06** | `policy_runner` | WorldEngine 原生闭环策略评测与 RL 数据采集工具（默认 1.5–7.5s / 16 帧窗口）：计算官方 PDM 指标与 8192 候选词表稠密奖励，自动导出 16 帧自车真实闭环轨迹（`ego_sim_frames.json`），支持单场景与常驻批量模式（`batch_scenario_pkls`）。 |
| **07** | `failure_case_filter` | 读取 `policy_runner` 产出的官方 PDM 评分结果，按低分失效、非早死与词表可解性等规则筛选有效失败样本，用于闭环验收与下一轮失败场景池挖掘。 |
| **08** | `policy_trainer` | 驾驶策略后训练工具：支持 ILFT 与带 Reverse KL 参考锚定的 RLFT，输出新权重与训练审计报告。 |

---

## 快速开始

### 1. 基础环境与密钥

```bash
python3 -m venv environments/agent/.venv
environments/agent/.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env   # 填入各模型 API Key 与 DashScope/OSS 凭证
```

模型与思考强度在 `configs/api.json` 统一配置，可用命令查看：

```bash
environments/agent/.venv/bin/python scripts/api_profile.py show
```

### 2. 运行方式

```bash
# 命令行调用主 Agent
environments/agent/.venv/bin/python -m agent.main "<任务描述>"

# 启动 Web 前端控制台（默认 127.0.0.1:5173）
node frontend/server.mjs
```

---

## 许可

自有代码采用 [Apache 2.0](LICENSE) 许可（声明见 [NOTICE](NOTICE)），内置第三方组件遵循其各自许可证。
