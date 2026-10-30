# 服务结构与运行目录

五个项目统一使用：

```text
<service>/
├── core/                 # 独立项目实现、原生入口、依赖和文档
├── adapter/
│   ├── service.py        # AdAgent 统一入口
│   ├── config.py         # 接入路径、解释器等配置
│   ├── schemas.py        # 输入/输出涉及的返回类型
│   ├── runner.py         # 按需提供 JSON 子进程入口
│   └── tests/            # 适配层集成测试
├── runtime/              # 不进入 Git
└── README.md             # Agent 接入说明
```

调用方向固定为 Agent Tool → adapter/service.py → core。
core 不导入 mcp_servers、agent 或 Agent 工具；独立项目保留自身入口与原生配置。
配置、runner 和内部模块按实际功能提供，不创建空占位。
所有服务的适配层接入配置由各自 adapter/config.py 管理；原生算法参数位于 core。

| 项目 | core 中的主要内容 |
|---|---|
| dggt | 模型、重建、检测、编辑与预处理源码 |
| smart_edit | 原生 MCP server、配置组装工具和资产 |
| sentry_video_search | pipeline、vendor、检索、重排序和 CLI |
| worldengine | 仿真、训练、评测与原生配置 |
| worldengine_failure_analysis | 取证脚本、数值核验、原生 CLI、模板和自身测试 |

WFA 原生入口现在是 worldengine_failure_analysis/core/run.py；检索独立入口是 sentry_video_search/core/cli.py。
检索的原生 API 和库配置分别在 core/service.py、core/config.py；Agent 入口统一在 adapter/service.py。
旧运行目录已清理，资源链接保留原位置。密钥和环境没有搬入 core。

---

所有服务的新任务使用同一布局：

```text
mcp_servers/<service>/runtime/<run_id>/
├── manifest.json       # 服务、操作、运行状态与时间
├── input/
│   └── request.json    # 调用参数记录，敏感字段脱敏
├── logs/               # 执行日志、检索阶段记录
└── output/
    └── result.json     # 返回结果；算法产物放在同目录或子目录
```

- 每次调用独立生成 run_id；服务内嵌套调用共享任务，子进程显式继承目录。
- 返回的 run_dir 指向任务根目录，文件路径仍通过原有结果字段提供。
- 成功或异常退出会更新 manifest；进程被强制终止时可能保留 running 状态。
- 默认不复制输入视频、权重与环境；request.json 记录引用路径。自动生成的暂存数据放在 input。
- 可以用 ADAGENT_RUNTIME_ROOT 指定运行根目录，此时目录为 <root>/<service>/<run_id>。
- runtime 被 Git 忽略。旧 input/output/outputs/logs/runs 及确认无用的实验产物已清理，新任务只写入 runtime。

## 各服务

| 服务 | 单次运行文件 |
|---|---|
| dggt | 解帧、编辑坐标转换配置、实例检测、重建和编辑产物 |
| smart_edit | 生成的配置、LLM/MCP 执行日志 |
| sentry_video_search | 检索阶段记录、预览、下载暂存、召回视频片段 |
| worldengine | 场景转换、仿真结果、评测记录、训练 checkpoint |
| worldengine_failure_analysis | 场景证据、分析尝试、报告和汇总 |

长期数据不是单次任务：视频库与索引继续使用检索服务的 data、data_waymo_front 和 runtime/.sentrysearch；WorldEngine 数据、模型权重和虚拟环境继续使用已配置的位置。

## 兼容与重试

实例检测的 output_path、SmartEdit 的 output_path 继续作为显式导出位置；返回路径指向本次任务内的正式产物。实例可视化在任务 output/visualizations 下；显式 vis_dir 也会导出一份。

训练的 work_dir 如显式传入，作为导出父目录，训练完成后复制到 <work_dir>/<run_id>/；实际训练始终在本次任务 output/training 内进行，避免读取旧 checkpoint。

WFA 的 run_name 是标签，不再决定共享结果目录。Agent 每次调用建立独立任务。CLI 如需查看或续跑已有分析，使用：
```bash
python mcp_servers/worldengine_failure_analysis/core/run.py report --config <config.json> --run-dir <任务目录/output>
python mcp_servers/worldengine_failure_analysis/core/run.py scene --config <config.json> --token <token> --run-dir <任务目录/output> --resume --attempts 2
```

统一任务管理的范围是 AdAgent 的服务入口；直接运行 core 原生 CLI 时遵循其参数和原生任务管理。WFA 原生 CLI 可用 --run-dir 指定已有任务输出目录。配置对象中的旧 input_dir/output_dir/logs_dir 字段保留兼容，Agent 服务调用使用统一任务目录覆盖它们。

WorldEngine 默认使用 adapter/run_testing.sh 启动，算法源码仍从 WORLDENGINE_ROOT 读取；上游 core 的启动脚本保持原样。显式自定义启动脚本必须支持 WORLDENGINE_TEST_PATH 指定的独立输出路径。

这些改动只规范文件布局，不改变训练算法；policy_trainer 当前仍为 IL。
