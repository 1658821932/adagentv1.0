# 后端组件

当前有 4 个服务组件，共注册 9 个主 Agent 工具；scene_editing 目前仅完成接口。
目录名称 mcp_servers 是组件存放位置，当前接入主要通过 Python 或 JSON 子进程完成。

| 组件 | 主工具 | 主要适配文件 |
|---|---|---|
| sentry_video_search | tool04 | service.py、client.py、runner.py、scene_binding.py、config.py |
| worldengine_failure_analysis | tool01 | service.py、config.py 及原生取证入口 |
| scene_editing | tool05/06 | service.py、schemas.py、config.py；后端命令待配置 |
| worldengine | tool13/14/15 | service.py、client.py、launch.py、evaluation.py、evaluate_batch.py、training.py、prepare_training.py |

## 通用边界

core/ 保存上游或独立项目代码，adapter/ 保存 AdAgent 封装，runtime/ 保存单次任务输入、日志与产物。
工具层保持统一 envelope 和 Schema 校验，主 Agent 负责下一步选择。

## 视频检索

Tool → adapter/service.py → client.py/runner.py → core/service.py。
服务内 search_local 自动对真实命中来源执行强制场景绑定，缺少 .scene.json 或身份不一致时报错。
scene_binding.py 返回紧凑 scene_ref，并把完整的相机图像、时间戳和参数写入任务 output/scene_binding.json。
注册新相机视频和索引方式见服务 README；普通无场景绑定的视频不能作为 tool04 real 成功结果。

## WorldEngine

- 仿真：adapter/client.py 通过 runner 调用 service.run_simulation；launch.py 管理原生仿真进程。
- 失败筛选：evaluation.py 调用官方 PDM 文件匹配和筛选，不使用旧几何判定。
- 训练：training.py 与 prepare_training.py 准备标注或奖励数据、生成配置并启动训练，training_audit.py 验证损失和参数更新。

仿真需已转换的原生场景及资产目录；旧编辑输出的临时场景转换已移除。
专家轨迹 CSV 到仿真/训练格式的新接入尚未实现。

## 失败分析

WFA 的 core/run.py 提供独立 CLI；adapter 负责传入场景 token、配置和独立任务目录。
API 统一通过根配置解析，服务配置不重复保存凭证。

## 扩展编辑模块

新攻击车编辑与自车编辑已提供阶段接口；未配置后端时返回错误，不提供空壳 real 成功结果。
新编辑接口已使用 core/adapter/runtime 分层，从 tool04.scene_ref 进入，输出冻结场景、专家轨迹及检查结果。
