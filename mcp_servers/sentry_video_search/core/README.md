# 独立项目入口

安装 requirements.txt 后，在本目录运行 `python cli.py --help`，无需 AdAgent。
默认库和索引位于上一级的 data/、data_waymo_front/ 和 runtime/.sentrysearch；现有链接保持不变。
可通过 SENTRY_LIBRARY_DIR、SENTRY_INDEX_HOME、SENTRY_WAYMO_ROOT 指定资源路径，SENTRY_RUNTIME_ROOT 指定独立 CLI 的运行目录。
Agent 接入使用 adapter/service.py，将同一个任务目录传给原生检索流程。

---

# 视频检索

内置 Sentry 视频检索代码。Agent 的 `video_retriever` 仅查询本地索引：召回候选、视频重排、导出片段，返回选中视频路径和检索记录。互联网检索需显式使用 CLI，不会由 Agent 本地检索自动触发。

## 环境与配置

在项目根目录执行：

```bash
python3 -m venv environments/video_retriever/.venv
environments/video_retriever/.venv/bin/python -m pip install -r mcp_servers/sentry_video_search/core/requirements.txt
```

CLI 和 runner 读取根目录 `.env`。召回和重排使用阿里接口，配置 `DASHSCOPE_API_KEY`；当前视频重排模型为 `qwen3-vl-rerank`，还需 `OSS_ACCESS_KEY_ID`、`OSS_ACCESS_KEY_SECRET`、`OSS_BUCKET`、`OSS_ENDPOINT`。最多4个候选压缩片段临时上传 OSS，接口通过签名 URL 访问，结束后尝试清理临时对象；本地检索仍涉及外部模型调用。

默认素材位于组件 `data/`，索引位于 `runtime/.sentrysearch/`。Waymo 库使用 `data_waymo_front/runtime/.sentrysearch/`，与默认库分开。代码已内置，现有索引和素材可能是指向旧项目的软链接；迁移时需另行复制或重建，并确保索引记录的源视频路径有效。

## 使用

以下命令均从项目根目录执行：

```bash
environments/video_retriever/.venv/bin/python -m mcp_servers.sentry_video_search.core.cli index /path/to/videos
environments/video_retriever/.venv/bin/python -m mcp_servers.sentry_video_search.core.cli search-local "直行接近低速前车"
environments/video_retriever/.venv/bin/python -m mcp_servers.sentry_video_search.core.cli search-internet "白天城市驾驶" --platform bilibili
```

本地建库和检索可加 `--library waymo`。Agent 输入为 `requirement` 和可选 `library`（default/waymo）。未命中时返回失败，不自动改用互联网。

导出视频在组件 `output/run_*/`，日志在 `logs/`；重排详情记录在运行产物的 `09_rerank.json`。默认建库会导入外部素材，Waymo 建库保留原视频位置；索引、素材、密钥和产物不入库。
