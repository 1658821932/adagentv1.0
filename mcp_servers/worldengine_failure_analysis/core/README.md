# WorldEngine 失败分析

读取 WorldEngine 官方评测文件、场景图片与数值证据，使用 VLM 分析失败原因。官方评分来自评测产物，本工具不重新计算评分。

## 独立运行

需要 Python 及 requirements.txt 中的依赖，不需要安装 AdAgent。

```bash
python -m pip install -r requirements.txt
cp configs/complete.json.example /path/to/analysis.json
cp api.example.json /path/to/api.json
export OPENAI_API_KEY="<your-key>"
python run.py scene --token <scene-token> --config /path/to/analysis.json   --api-config /path/to/api.json --run-dir /path/to/output --run-name trial
```

在 analysis.json 填写 worldengine_root、result_root 等数据路径；api.json 设置模型、端点和 api_key_env，密钥只通过相应环境变量提供。
也可用 WFA_API_CONFIG 指向 API 配置；未指定时读取本目录 api.example.json。API 示例不是密钥文件，应按实际服务修改。
场景图片及相关证据会发送到所配置的端点。

加 --resume 续跑；使用 --run-dir 明确选择输出目录，未指定时在本组件上一级 runtime/ 分配任务。
场景产物位于输出目录 scenes/<token>/：result.json、report.md、图像、候选、数值证据及模型调用记录。

## 分组与接入

scripts/summarize_edit_scenarios.py 对已完成案例归组，支持 --api-config 指定相同 API 配置，--run-dir 指定分析输出；--groups-from 可在不改变成员时更新描述。
scripts/watch_batch.py 查看已有批次，具体参数见 --help。

AdAgent 的 adapter 负责解释器、任务目录和主项目配置注入，tool01 返回 failure_analysis 与完整证据路径。独立 core 不导入根 api_config 或 agent/tools。
原因分析保留证据范围；候选几何接触不等于官方责任判定，专家行为建议不等于已经验证的驾驶轨迹。
数据、密钥和运行产物不随源码提交。

## 提示词

静态任务说明集中在本 core 的 prompts/；原生代码按文件位置读取，不依赖主 Agent。任务数据和返回协议仍由调用代码组装。
