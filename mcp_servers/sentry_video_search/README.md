# 视频检索

统一分层：`core/` 保存项目本身，`adapter/` 保存 AdAgent 封装，`runtime/` 保存单次运行文件。

| 目录 | 内容 |
|---|---|
| core/ | 独立算法、原生入口和说明；不依赖 AdAgent |
| adapter/ | service.py 对外入口、config.py 接入配置、schemas.py 返回约定 |
| adapter/tests/ | 服务接口、路径隔离和导出兼容性测试 |
| runtime/<run_id>/ | manifest.json、input/、logs/、output/，不进入 Git |

## Agent 接口

从 `mcp_servers.sentry_video_search.adapter.service` 导入：retrieve_video(input_data)；服务环境内部使用 search_local、index_local、search_internet。

环境：`environments/video_retriever/.venv`。adapter/config.py 负责 Agent 解释器；core/config.py 负责库和检索参数。

输入输出沿用原工具格式；结果中的 run_dir 指向任务根目录，具体产物通过原有路径字段返回。显式导出路径继续支持，旧运行目录已清理。

## 独立使用与验证

项目自身的安装和使用说明见 [core/README.md](core/README.md)。在 AdAgent 根目录，使用已安装测试依赖的 Python 执行：

```bash
python -m pytest mcp_servers/sentry_video_search/adapter/tests -q
```

权重、环境、密钥、长期数据与运行文件不随源码提交。服务目录之外的依赖通过配置或软链接接入。

## 多相机场景绑定（tool04 real 强制）

检索仍命中一条相机视频，但返回 `source_file`（索引原视频）和 `scene_ref`（完整场景）。
绑定使用源视频旁的 `<stem>.scene.json`，不根据裁剪视频文件名猜测场景。
没有绑定、token 不一致或相机不属于该场景时直接报错，普通无标注视频不再作为 tool04 real 的成功结果。

为任一路相机视频注册场景后，按原有方式建索引：

```python
from mcp_servers.sentry_video_search.adapter.scene_binding import register_scene_video
register_scene_video("/data/CAM_L0.mp4", "/data/dataset/samples/anchor", "CAM_L0")
```

样本目录需含 `manifest.json`（anchor_token/scene_token）、`metadata/frames.json`、`metadata/cameras.json`。
帧元数据使用 timestamp（微秒）及 cams[相机名].data_path；相对图片路径以 original_images/ 为根。
视频应覆盖该样本的原始完整时间轴；相机视频的生成和同步由数据准备流程负责，注册不会重新编码或校验视频画面。
入库复制视频时同时保留绑定文件，重排、裁剪后仍按原始命中视频绑定。

`scene_ref` 包含场景 token、命中相机、全部相机的图片可用数量、标注路径、场景 pkl、资产配置、高斯资产和地图路径。
`manifest_path` 指向本次 runtime/output/scene_binding.json，保存按时间排序的各路原始图片、时间戳及相机参数，避免把逐帧列表放进 Agent 上下文。
相机视频存在显式绑定时返回 video_path；没有生成视频时为 null，仍可从 manifest 读取原始图像序列。
文件完整性按磁盘实况计算，不使用可能过时的下载状态字段。尚未下载的图片和资产明确标记缺失。

这些路径对应完整底场景；`source_segment` 是检索命中的索引片段范围（原视频秒数），不等于最终导出片段边界，也不表示八路已经统一裁剪。
mock 保留占位输出，不执行真实场景绑定。

## 批量相机图像建库

在项目根目录运行：

```bash
environments/video_retriever/.venv/bin/python scripts/index_scene_cameras.py
```

默认读取 data/navtrain_ep_per1/samples 下全部场景，为每路相机生成一个原分辨率 H.264/yuv420p MP4。
帧顺序和时间间隔来自 metadata/frames.json，视频时间戳精度为毫秒，原始微秒时间戳保存在 .scene.json。
生成后完整解码检查帧数；视频写入检索服务 data/navtrain_ep_per1_multicam/<token>/<camera>.mp4，并注册场景绑定。
每个完整视频作为一条向量写入默认索引，不复制到其他视频目录，不依赖高斯资产下载是否完成。

采用现有 retrieval API profile，默认四个工作线程、每分钟最多45次初始请求。
源图片签名和已有索引元数据用于断点跳过；同一数据集通过文件锁避免重复启动。
可先使用 --limit-scenes 1 验证一个场景，--encode-only 只合成视频；再次执行原命令继续未完成项。
进度、逐视频记录及错误位于 data/navtrain_ep_per1/video_index/ 的 progress.json、events.jsonl、errors.json。
这些视频、索引与运行记录均不进入 Git。

训练集批量入库默认只处理前视 `CAM_F0`（525 个场景对应 525 个 MP4）；仍保留完整多相机场景绑定。运行：`environments/video_retriever/.venv/bin/python scripts/index_scene_cameras.py --workers 4 --rpm 25`。
