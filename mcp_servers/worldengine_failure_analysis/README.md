# 失败分析

基于 WorldEngine 官方闭环评测数据，通过外部 VLM 输出现象、候选行为原因、数值核验及最终原因。不推断模型内部根因，也不重新计算或改写官方评分。

## 流程与结果

官方指标确定失败 → VLM查看12张多相机总览并按需补看相机/BEV → 提出现象和候选 → 查询自车、周车、边界及规划数值 → 比较候选并输出原因和不确定性。

图像和数值由程序提供，原因由 VLM 判断。官方首次违规时间不能代替每类事件的独立时间；EP=0不能证明静止；周车速度由同一对象的世界坐标历史计算；几何接触候选不能冒充官方责任判定。缺失证据保留不确定，不填0或强行定因。

## 配置与运行

独立分析环境需要 `requirements.txt` 中的依赖；通过 Agent 调用时可用 `WFA_PYTHON` 指定解释器，否则依次查找项目的 simengine、dggt、agent 环境。

复制 `configs/complete.json` 为自己的配置，设置 `worldengine_root`、`result_root`、可选数值缓存和 `vlm_allowed_tokens`。示例中的本地数据路径须改为本机路径。配置 `.secrets/vlm.json`：

```json
{"endpoint":"https://your-provider/v1/chat/completions","model":"your-vlm-model","api_key":"your-key"}
```

密钥文件权限设为0600，不入库。分析会向该端点发送场景图片和相关证据。以下命令在本组件目录执行，使用已装依赖的 Python：

```bash
python run.py scene --token <scene-token> --config /path/to/config.json --run-name trial
python run.py report --run-name trial
```

续跑加 `--resume`；新实验使用新 run-name。Agent 工具 `failure_analyzer` 接受 `token`（也兼容 `scene_token`）、可选 `config_path`、`run_name` 和 `resume`，返回供需求生成使用的 `failure_analysis` 以及完整取证结果路径。

## 产物与汇总

`runs/<run-name>/scenes/<token>/` 保存 `result.json`、`report.md`、候选、图片、数值证据及模型调用记录；运行根目录保存汇总。失败或未结束的调用不计为完整分析。

`scripts/summarize_edit_scenarios.py` 对已完成案例分组，输出“自车动作→环境及他车→错误行为→应有响应”，保留成员及依据。应有响应是后续场景测试目标，不能据此直接修改自车轨迹。已有分组仅改描述时使用 `--groups-from` 保持成员不变。

原始指标、视频、密钥及运行结果不随源码分发。历史分析报告留在原研究项目，不作为当前使用说明。
