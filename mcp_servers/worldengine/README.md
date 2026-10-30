# WorldEngine

统一分层：`core/` 保存项目本身，`adapter/` 保存 AdAgent 封装，`runtime/` 保存单次运行文件。

| 目录 | 内容 |
|---|---|
| core/ | 独立算法、原生入口和说明；不依赖 AdAgent |
| adapter/ | service.py 对外入口、config.py 接入配置、schemas.py 返回约定 |
| adapter/tests/ | 服务接口、路径隔离和导出兼容性测试 |
| runtime/<run_id>/ | manifest.json、input/、logs/、output/，不进入 Git |

## Agent 接口

从 `mcp_servers.worldengine.adapter.service` 导入：run_policy(input_data)、filter_failure_cases(input_data)、train_policy(input_data)。

环境：`environments/simengine/.conda、environments/algengine/.conda`。WORLDENGINE_ROOT；启动脚本为 adapter/run_testing.sh；训练当前仍为 IL。

输入输出沿用原工具格式；结果中的 run_dir 指向任务根目录，具体产物通过原有路径字段返回。显式导出路径继续支持，旧运行目录已清理。

## 独立使用与验证

项目自身的安装和使用说明见 [core/README.md](core/README.md)。在 AdAgent 根目录，使用已安装测试依赖的 Python 执行：

```bash
python -m pytest mcp_servers/worldengine/adapter/tests -q
```

权重、环境、密钥、长期数据与运行文件不随源码提交。服务目录之外的依赖通过配置或软链接接入。
