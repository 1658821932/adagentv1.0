# UFO 重建与米制编辑接入记录

新增米制控制分支 `codex/ufo-metric-asset-editing` 请使用 [UFO米制编辑与资产插入.md](UFO米制编辑与资产插入.md) 的 `calibrate_metric` / `metric_edit` 新接口。下文是早期实验记录，不是该分支默认入口。

当前使用 `environments/ufoex` 的离线工具测试，请按 [UFO 离线重建与变道测试](UFO离线重建与变道测试.md) 运行。该入口直接调用 reconstruction 与 scene_editor，默认使用指定高速视频进行原生坐标变道；下文保留参考入口和米制编辑接入记录。

## 第一阶段：只验证重建

服务器已经有 `/root/miniconda3/envs/dggt_data/bin/python`、`ufo_video_entry_20261012/run_video.sh`，且 WorldEngine 位于 `yx/ufo-temporal-baseline` 分支。`.env` 记录 `RECONSTRUCTION_BACKEND=ufo`、`UFO_ENTRY_ROOT`、`UFO_OUTPUT_ROOT` 等路径；它不是 Python 虚拟环境，也不负责安装 UFO。先在服务器终端激活已有环境，再从本项目工作树运行适配器：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dggt_data
cd /absolute/path/to/adagentv1.0_edit_update
python -m mcp_servers.ufo.service \
  --video /absolute/path/to/front_video.mp4 \
  --scene-name ufo_reconstruction_test_001 --gpu 0
```

适配器要求视频至少 40 帧（原始 UFO 脚本允许至少 20 帧，但本 Agent 当前固定使用 40 帧），会选前 40 帧，调用 `run_video.sh --baseline worldengine-time-v2`，并核对重建视频、资产、相机、模型权重与帧数。先独立查看 `workspaces/ufo_reconstruction_test_001/output/worldengine_time_v2/reconstruction.mp4` 和 `comparison.mp4`，再按参考示例加载资产首帧。此阶段不调用 `scene_editor` 或米制拟合。

`worldengine-time-v2` 的原生产物是 `assets/SCENE/background/SCENE.ckpt`、`assets/SCENE/dynamic/*.pt`、`cameras.npz` 与 `ASSET.json`，**不是** DGGT 的 `scene.ply` 和逐帧动态 PLY。需要让现有依赖 PLY 的下游继续工作时，须另做格式转换和渲染一致性验收；第一阶段直接使用 UFO 的 WorldEngine 资产检验重建，不把 `.pt` 改名成 `.ply`。

## 目标

以服务器上的 `ufo_video_entry_20261012/run_video.sh` 取代旧 DGGT 重建，随后让车辆轨迹编辑以米为单位，并报告编辑前后实际三维位移。

## 已核实的接口

- 正式入口：`bash run_video.sh --baseline worldengine-time-v2 --video VIDEO --name UNIQUE_NAME --gpu GPU`。
- 已完成结果：`workspaces/NAME/output/worldengine_time_v2/reconstruction.mp4`、`DONE.json`；WorldEngine 资产位于 `WorldEngine/data/ufo_time_v2/assets/NAME/`。
- 模型权重为 `e2_front352_posefree_moge_full/.../ckpt_099999.pth`，MoGe 伪深度训练标签单位为米。
- 参考入口的 `runtime_paths.py` 实际从 `exufo_eval_meeting_20261008/source/ufo` 导入模型；已只读比对过它与指定的 `e2_front352_20260930/source/ufo` 的全部 Python 文件，内容相同。适配器每次重建前会重新校验字节一致，并验证实际 `BUILD.json` 使用指定的绝对 checkpoint 路径；出现差异即停止。
- 参考入口的 `ASSET.json` 明确标注：资产使用原生重建单位，**尚无外部米制标尺校准**。动态车辆是按观察时间回放的高斯集合，尚无经验证的单车 canonical 轨迹编辑。

因此，MoGe 米制训练监督不能直接当作导出场景的米制证明。重建阶段返回 `metric_calibrated=false`；编辑阶段会重新运行 MoGe-2，在同帧静态像素上拟合米/原生单位，验证后才执行偏移。

## 已实现的编辑路径

1. `reconstruction` 默认调用 UFO `worldengine-time-v2`，将长视频截成前 40 帧，保留原视频引用，并检查 `READY.json`、`DONE.json`、资产及相机文件。
2. `instance_discovery` 可直接读取重建输入视频；目标选择仍使用现有视觉绑定。Smart Edit 生成单车 `trajectory_edit`，其中偏移按米解释。
3. `scene_editor(backend="ufo")` 运行 `mcp_servers.ufo.metric_edit`：用 MoGe-2 对同帧的 UFO 静态深度拟合尺度，验证各帧比例和残差；把被选目标的二维检测框匹配到 SAM track ID。
4. `actor_edit.py` 复现参考 `assemble_scene.py` 的动态高斯筛选顺序，精确找出目标 track 在逐帧 `actors` 张量中的索引；只移动这些高斯的 `means`。输出独立资产，源资产不修改。
5. 逐帧把实际三维位移投影回 `[前, 左, 上]` 米制，要求最大数值误差小于 0.01 m；随后通过原生 WorldEngine 渲染，返回视频和 `EDIT_AUDIT.json`。

## 服务器运行

在服务器上的本项目工作树执行。`run_video.sh` 需要 GPU；已有通过 `worldengine-time-v2` 生成的 40 帧工作区可以直接用于后面的编辑命令。每次换一个未使用的 `NAME` 和 `EDIT_NAME`，避免覆盖资产。

```bash
ENTRY=/inspire/hdd3/project/intelligent-driving-agent/public/workspace/yx/ufo_video_entry_20261012
WORLDENGINE=/inspire/hdd3/project/intelligent-driving-agent/public/workspace/yx/WorldEngine
MOGE=/inspire/hdd3/project/intelligent-driving-agent/public/workspace/yx/moge_depth_full_20260922/source
MOGE_CKPT=/inspire/hdd3/project/intelligent-driving-agent/public/workspace/yx/adagent_posefree/checkpoints/moge-2-vitl/model.pt
NAME=your_unique_scene
EDIT_NAME=your_unique_metric_edit
GPU=0

bash "$ENTRY/run_video.sh" --baseline worldengine-time-v2 \
  --video /absolute/path/to/40_frame_video.mp4 --name "$NAME" --gpu "$GPU"
```

选车与 Smart Edit 配置生成后，使用 `target_selector` 的 `selection_path` 和包含单车 `trajectory_edit` 的 JSON。`scene_editor` 传 `backend=ufo`、完整重建结果 `ufo_reconstruction`、`target_selection_path`、`edit_config_path` 和 `gpu` 即可。独立调用底层编辑器时，先将 Smart Edit 的每帧偏移写成如下 JSON，40 个偏移或一个全程恒定偏移均可：

```json
{"units":"metres","coordinate_system":"camera_local_forward_left_up","offsets":[[0,-3,0]]}
```

右移 3 米对应左轴 `-3`。然后在项目根目录运行：

```bash
"$MOGE/python_offline.sh" -m mcp_servers.ufo.metric_edit \
  --workspace "$ENTRY/workspaces/$NAME" \
  --asset-root "$WORLDENGINE/data/ufo_time_v2" \
  --output-asset-root "$ENTRY/workspaces/$NAME/metric_edits" \
  --new-name "$EDIT_NAME" --target-selection /absolute/path/to/selection.json \
  --offsets-json /absolute/path/to/offsets.json \
  --moge-source "$MOGE/vendor_moge" --moge-checkpoint "$MOGE_CKPT" \
  --worldengine-root "$WORLDENGINE" --gpu "$GPU"
```

命令结束后检查输出 JSON 的 `rendered_video`、`asset_manifest` 和 `frames`；`EDIT_AUDIT.json` 中有每帧请求与执行位移、目标高斯数量和数值误差。若尺度拟合、跨帧稳定性或目标绑定不通过，程序会报错并停止，不应把该结果称为米制编辑成功。

这些是代码路径和数值验收门槛；**服务器当前没有 GPU，尚未完成端到端实测**。MoGe 估计的绝对尺度本身仍存在模型误差，数值误差小于 0.01 m 只验证执行器准确应用了拟合比例，不等于独立的测量仪器对真实距离的认证。当前 `run_video.sh` 的资产不能静默套用旧 DGGT 的 `meters_to_dggt=0.01`。
