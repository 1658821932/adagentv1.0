# UFO 离线重建与变道工具测试

启动脚本为 `scripts/test_ufo_lane_change.sh`，直接调用 Agent 的 `reconstruction` 和 `scene_editor` 工具，不启动 Agent 或 LLM 服务。默认视频为 `/inspire/hdd3/project/intelligent-driving-agent/public/workspace/ghc/adagentv1.0/mydata/高速第一视角-去水印版.mp4`，默认解释器为该项目 `environments/ufoex/bin/python`。

## 启动

在测试代码目录执行，场景名使用一个未占用的名字：

```bash
bash scripts/test_ufo_lane_change.sh --name highway_lane_test_001 --gpu 0
```

默认截取视频前 40 帧，完成单视频相机、MoGe/Omega 前端、SAM 跟踪、语义分割、UFO 100k 推理、time-v2 资产打包和 WorldEngine 重建渲染，然后让一个车辆 track 平滑向右变道并渲染。每个推理子进程使用 ufoex；参考脚本通过独立调度执行，不修改参考目录里的代码或解释器。

模型从服务器现有本地目录加载；设置 Hugging Face/Transformers 离线标志，同时对 Python 子进程禁止连接非本机地址。若缺少本地权重或导入依赖则报错，不安装包、不下载模型。

视频读取、编码和前后对照拼接均由 PyAV 完成，无需系统安装 `ffmpeg` 或 `ffprobe` 命令。

只做 CPU 环境与本地权重预检：

```bash
bash scripts/test_ufo_lane_change.sh --name environment_check --check-only
```

默认预检只检查本地文件、解释器位置和包是否已安装，不导入 Torch、Torchvision 等大库。正常运行由重建进程检查一次，不再额外启动进程完整导入依赖。大库在实际处理阶段按需加载；阶段开始、结束和秒数会同时显示在终端和总日志中。

如要排查某个库导入耗时，可单独运行完整导入诊断；每个模块会打印耗时，超过 60 秒会输出当前 Python 调用栈：

```bash
bash scripts/test_ufo_lane_change.sh --name import_diagnostic --check-imports
```

离线加载仍需读取共享盘上的 Python 文件和动态库；它不表示模型已驻留在内存或显存。整个测试还包含相机、SAM、语义分割、背景导出及 WorldEngine 渲染，耗时不能直接等同于 UFO 网络的一次前馈时间。

## 选车与变道参数

默认仅为通用工具测试，选择连续可见帧数最多的 car/truck/bus track，同等帧数时选靠近图像中心者，要求至少 80% 帧可见。它不执行自然语言目标识别。目标框在 `target_preview.jpg` 中标出，所有候选写入 `targets.json`。

如希望先看重建和候选，再指定车辆：

```bash
bash scripts/test_ufo_lane_change.sh --name highway_lane_test_001 --gpu 0 --targets-only
# 查看输出的 targets.json，使用其中真实的 track_id：
bash scripts/test_ufo_lane_change.sh --name highway_lane_test_001 --gpu 0 \
  --skip-reconstruction --track-id ACTUAL_TRACK_ID --direction right
```

`--direction left` 改为左变道。默认侧移量为目标车高斯横向 10%～90% 跨度的 1.8 倍；可用 `--lateral-distance-native POSITIVE_NUMBER` 显式设定。单位为重建原生单位，报告不宣称米制精度。

轨迹从片段时间 20% 处开始，到 70% 处完成，使用平滑插值；之后补偿目标原有横向运动，使其保持目标相机相对横向位置。前向运动保留，世界 Z 高度不变，其他车辆高斯不变。该测试不包含车道线约束或碰撞规划，实际画面需运行后检查。

## 输出与日志

默认结果位于原项目的 `outputs/ufo_tool_tests/SCENE/`：

- `tool_reconstruction.json`：重建工具结果；`tool_scene_editor.json`：编辑工具结果。
- `output/worldengine_time_v2/reconstruction.mp4`：原重建；`comparison.mp4`：输入与重建对比。
- `worldengine_asset/assets/SCENE/`：原始 time-v2 场景资产。
- `lane_change_right/target_preview.jpg`、`target_selection.json`、`targets.json`：目标预览与选择记录。
- `lane_change_right/edited.mp4`：编辑结果；`before_after.mp4`：左原重建、右变道后的对照。
- `lane_change_right/assets/SCENE_lane_right/EDIT_AUDIT.json`：每帧实际侧移、目标高斯数量、其他车辆与高度不变校验。
- `offline_reconstruction.log`：总重建日志；`logs/offline/`：各阶段日志；`lane_change_right/editor.log`：编辑与渲染日志。

同视频、同名字、同代码可恢复已完成的重建阶段。只做重建可加 `--reconstruction-only`。如果已完成编辑，换名字重测以保留已有资产；若编辑阶段中断且已经生成 `ASSET.json`，相同参数重启会重用已生成的编辑资产并重新渲染。

如果只是测试适配层修复，输入、环境、checkpoint 及外部模型/参考代码都未改变，可使用相同名字加 `--resume`：

```bash
bash scripts/test_ufo_lane_change.sh --name highway_lane_001 --gpu 0 --resume
```

程序会核对已有阶段完成标记和所需产物，打印 `RESUME decode/cameras/sam/adapter/semantics` 并从失败阶段继续。旧运行合同归档到 `contract_history/`，已生成文件不删除；若输入、权重或外部模型实现发生变化，会拒绝复用。`--skip-reconstruction` 适用于重建已完整成功的情况；重建中途失败应使用 `--resume`。

单前视模型加载会显式传入 `num_cams=num_max_cameras=1`，与原单相机 checkpoint 的 affine token 维度一致；该适配只补足构造参数，不扩展、丢弃或重写 checkpoint 参数。UFO 推理和神经背景导出使用同一适配。

如果 `tool_reconstruction.json` 已生成，而编辑视频编码失败，可以用相同名字和参数加 `--skip-reconstruction` 重试。存在完整 `ASSET.json` 与匹配的 `EDIT_AUDIT.json` 时，会复用编辑资产后重新渲染。编码优先使用资产中的原始有理数帧率，逐帧时间戳保持不变。

脚本支持 `UFO_ENV_ROOT`、`UFO_ENTRY_ROOT`、`UFO_WORLDENGINE_ROOT` 环境变量和 `--output-root`、`--video` 参数。远端采用独立测试代码目录，避免覆盖原项目的未提交改动。
