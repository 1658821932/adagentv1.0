# UFO 米制编辑与资产插入

2026-10-27 已完成真实 GPU 验证。**当前审核结果、估计米制单位验证、局部道路代理及复现命令请优先看 [GPU审核记录](UFO_GPU审核记录_20261027.md)**。下文默认 `depth_fit` 是独立几何一致性检查；当前视频该项仍未通过，不与前端估计米制坐标契约混为一谈。

## 版本与目录

分支 `codex/ufo-metric-asset-editing` 从 `679c041` 建立，保留多候选选择、UFO/DGGT 重建切换和原视觉变道。

远端工作目录：

```bash
cd /inspire/hdd3/project/intelligent-driving-agent/public/workspace/ghc/adagentv1.0_ufo_metric_update
```

本功能不重新重建。以下命令复用已完成的高速视频前 40 帧；不要从旧测试快照运行。

```bash
RECON=/inspire/hdd3/project/intelligent-driving-agent/public/workspace/ghc/adagentv1.0/outputs/ufo_tool_tests/highway_lane_001/tool_reconstruction.json
CALIB=/inspire/hdd3/project/intelligent-driving-agent/public/workspace/ghc/adagentv1.0/outputs/ufo_tool_tests/highway_lane_001/metric_scene_v1.json
```

`scripts/ufo_metric_edit.sh` 使用原有 `environments/ufoex` 和离线资源。通过 `UFO_GPU=0` 选择 GPU；不要另外安装或下载模型。新机器可用 `UFO_ENV_ROOT`、`UFO_MOGE_SOURCE`（直接包含 `moge/` 的源码目录）、`UFO_MOGE_CHECKPOINT`、`UFO_WORLDENGINE_ROOT` 指向已有资源。

## 1. 标定场景尺度和道路平面

标定使用原生 WorldEngine 的 `RGB+ED` **渲染期望 camera-z 深度**，与 UFO 训练的深度监督定义一致。早期实现将单个高斯中心 z 当成对应像素深度，这是错误的；高斯重叠、遮挡和透明度合成后，渲染深度与中心深度不等价。因此早期中心深度拟合失败不能直接证明整个场景几何失去米制尺度。

修正后的标定从已有资产渲染深度，并缓存到 `moge_metric_depths/<重建摘要>/rendered_native_v1/`；只执行高斯光栅化，不重新运行 UFO 前馈。已预测的 MoGe 深度继续复用。道路平面也从同一渲染深度反投影取得，自动标定不再用原始高斯中心拟合道路。

```bash
UFO_GPU=0 bash scripts/ufo_metric_edit.sh calibrate \
  --reconstruction-json "$RECON" --output "$CALIB"
```

首次用本机 MoGe-2 米制 checkpoint，使用 UFO 相机 FOV，并先校正水平/垂直焦距不一致导致的图片拉伸，再将深度映射回同一 RGB 像素网格，使用静态非天空区域拟合一个 `meters_per_native_unit`。采样覆盖每个实际选用的重建窗口；拟合要求每帧至少 100 个有效样本，合并至少 500 个，深度相对残差中位数不超过 15%、P90 不超过 25%，各采样帧尺度相对全局比例不超过 10%。这是拒绝明显失配的门槛，不是声称真实距离精确到这些数值。

随后用多帧静态道路语义点估计局部平面：15 cm 平面残差内至少 70% 点一致，并拒绝退化点集。不能找到一致道路或尺度时会停止，不自动改成画面偏移，也不逐帧使用不同缩放系数。此时需要修复重建对齐，或提供有依据的局部参考。

成功结果保存到 `$CALIB`，绑定原资产、相机、BUILD 和源文件/动态文件元信息。相同输入再次执行会复用标定；已预测的米制深度另存为带 RGB/权重/FOV 校验的缓存，修改编辑距离不重跑 MoGe。原始上游缓存没有相同契约时不会盲目当作可用的米制深度。

若出现 `metric calibration residual ... exceeds ...`，表示单一全局比例没有通过验证，不是可以忽略的依赖错误。程序会保留深度缓存，并在缓存目录下的 `rendered_depth_diagnostics/frame_XXXXXX.json` 保存全部样本、不同距离区间和语义类别的拟合统计；这些分组比例只用于定位问题，不能当作已验证的局部标尺。历史 `diagnostics/` 中基于高斯中心的失败报告不属于修正后的检验。相机/像素对应正确但渲染深度的不同距离比例仍明显变化时，需要进一步核对 MoGe 与重建的几何输出，不能提高阈值后继续宣称精确米制编辑。

### 有实测标尺时

如果已测得一个原生重建距离对应的真实长度，可提供 JSON：

```json
{
  "source": "measured_reference",
  "description": "说明实际测量了哪两个点、如何取得对应原生距离",
  "native_distance": 2.0,
  "distance_m": 5.0,
  "relative_uncertainty": 0.02
}
```

以上数值只是格式示例，必须替换为实测数据。使用 `--reference /path/to/reference.json` 可绕过 MoGe 推理；报告会明确记录没有执行跨帧深度尺度验证。原生距离和米制距离都必须为正，不能将猜测的车道宽度标为实测。

可选 `--road-plane /path/to/road_plane.json` 使用有依据的原生世界道路平面：`{"normal_world":[0,0,1],"offset_native":1.5,"description":"测量来源"}`。方程为 `normal_world·point_native + offset_native = 0`，法向必须为单位向量且朝上；示例参数也必须按当前场景填写。`--force` 显式重做标定；默认不覆盖不同输入的旧标定。

## 2. 先测试原车向右移动 3 m

```bash
UFO_GPU=0 bash scripts/ufo_metric_edit.sh edit \
  --reconstruction-json "$RECON" \
  --config configs/examples/ufo_metric_translate.json
```

示例明确指定历史测试车辆的 `track_id=10001`，在 0.2–1.8 秒叠加从零到 `[0,-3,0]` 米的平滑偏移。它用于直接工具测试，不替代 Agent 的语义选车。新视频或不同标定路径需修改配置。想先只导出资产和数值审计，可加 `--assets-only`，不运行 GPU 渲染。

坐标是固定的 `[前,左,上]` 道路坐标，右移 3 m 写为 `[0,-3,0]`。坐标原点为编辑开始时最近一帧的相机位置在道路平面上的投影，不是经过标定的自车后轴；坐标方向取该帧相机前向在道路平面上的投影。使用同一个固定基底，不随逐帧相机转向。记录的参考帧和实际时间可在审计中查看。

`trajectory_mode=offset_from_original` 表示在原轨迹上叠加位移，不保证移到某个车道后保持车道位置。禁止非零 up 偏移，以保持道路法向高度。五次平滑插值在 start/end 处速度与加速度连续归零；end 不能超出视频末帧时间。目标在终点不可见时拒绝声称已完成距离验证。

## 3. 准备真实新资产

复制 `configs/examples/ufo_gaussian_asset.template.json` 到自己的资产目录，填入真实信息。模板的 `null` 是必须填写的项，不能直接执行。

| 字段 | 约定 |
| --- | --- |
| asset_path | 实际本地资产路径；相对路径按资产清单所在目录解析 |
| format | `3dgs_ply`、`gaussian_npz` 或 `gaussian_pt` |
| meters_per_unit | 资产一个原生长度单位是多少米；已知单位时优先提供 |
| dimensions_m | 可选的实际 `[长,宽,高]`；单位未知时必须提供，用一个统一比例拟合，不会非均匀拉伸车辆 |
| asset_to_flu | 将资产坐标转成前、左、上的 3×3 正交旋转矩阵；已是 FLU 时填单位矩阵 |
| ground_contact_point | 资产原始坐标中的底部接地点，不能随意用包围盒中心 |
| quaternion_order | `wxyz` |
| color_space | `linear_rgb` 或 `srgb`，按实际资产填写 |

只给 `dimensions_m` 时，根据转换后高斯中心包围盒拟合统一比例；若与声明尺寸相差超过 10% 会拒绝。接地点到最低高斯中心面的误差不能超过 15 cm。尺寸和接地均基于高斯中心，不含无限延伸的高斯尾部，也不等于经过物理碰撞检测。

支持的 NPZ/PT 字段是 `means[N,3]`、`scales[N,3]`（正数实体尺度，不能是 log）、`quats[N,4]`（wxyz）、`opacities[N]`（0–1 概率，不能是 logit）、`colors[N,3]`。PT 为这些 CPU tensor 的字典。

PLY 需要标准 3DGS 的位置、`scale_0..2`、`rot_0..3`、`opacity`、`f_dc_0..2`。导入会正确激活 log scales 与 opacity logits，并转换 SH DC。**普通点云 PLY、OBJ/GLB 网格不直接支持**，需先转换/生成完整高斯资产。PLY 需显式 `appearance=dc_only`：当前导入丢弃高阶视角相关 SH，外观可能变化，报告会记录这一限制。

先做 CPU 检查：

```bash
bash scripts/ufo_metric_edit.sh inspect-asset \
  --asset-manifest /你的实际路径/car.asset.json
```

检查输出的尺寸、单位和接地点后，再进行插入。

## 4. 插入车辆，再向右移动 3 m

复制 `configs/examples/ufo_metric_insert_translate.json`，将 `asset_manifest_path` 改为上一步的真实清单路径，必要时修改 calibration_path、位置与时间。示例是在道路坐标 `[前20m,左3m,上0m]` 处插入车辆，再平移至左偏移减少 3m。位置是请求值，不代表检测到了对应车道或没有碰撞。

```bash
UFO_GPU=0 bash scripts/ufo_metric_edit.sh edit \
  --reconstruction-json "$RECON" --config /你的实际路径/insert_edit.json
```

同一资产的所有高斯使用相同的刚体变换，尺度与四元数一起转换；不使用上一版的逐高斯画面偏移。新车默认固定在道路世界位置，只有配置的 translate 才会移动；如需要同时前进，应在 offset_m 的 forward 分量明确给出距离，不会自动猜车速。yaw 固定为插入时指定角度，不模拟转向机构。

## 输出与复用

默认输出根目录：

```text
.../adagentv1.0/outputs/ufo_tool_tests/highway_lane_001/metric_edits_v1/
  assets/<场景名>_metric_<参数摘要>/
    ASSET.json
    EDIT_AUDIT.json
    cameras.npz
    background/...
    dynamic/000000.pt ... 000039.pt
  <场景名>_metric_<参数摘要>/
    edit_config.json
    result.json
    edited.mp4
    before_after.mp4
```

`before_after.mp4` 左原重建、右编辑后。可用 `--output-root` 指定独立位置。资产在 CPU 构建完成后原子发布；相同配置/标定/资产可复用，渲染失败重试不会重建场景，已完成编辑资产也可复用。修改距离或资产会产生新摘要目录。源资产和相机保持不变。默认 native 模式不修改背景；显式 road_geometry=planar_proxy 会修改新导出资产的静态道路几何，不覆盖原资产，也不修改原有动态车辆。

`EDIT_AUDIT.json` 保存逐帧请求/实际位移，逐点数值误差必须 ≤1 mm；这仅验证在所用标尺下执行了请求。真实距离精度仍受 MoGe/实测标尺、重建几何和道路估计误差影响。旧车辆仍是单视角可见表面，即使刚体平移正确，也可能拉散、缺面或暴露背景残影；本版本不生成补洞、真实投影阴影，也不进行车道或碰撞验收。

## Agent 工具接入

1. `reconstruction` 保持 UFO 默认、DGGT 可选。
2. `scene_editor(backend=ufo, operation=calibrate_metric, ufo_reconstruction=...)` 返回 calibration_path；或使用已验证标定。
3. 编辑原车时保持 instance_discovery → grounded target_selector，传 selected_instance_info_path；仅插入新资产时无需选一个原车。
4. `edit_plan_builder(operation=metric_edit)` 使用已提供资产与尺度，不猜单位/资产路径。
5. `edit_config_generator(backend=ufo, operation=metric_edit, calibration_path=..., operations=[...], output_path=...)` 生成确定性 JSON；原车必须具有选择绑定，Agent 不允许用手写 track_id 绕过。
6. `scene_editor(operation=metric_edit, edit_config_path=..., ufo_reconstruction=...)` 执行，再做 scene_validator。

完整 Agent 可从新分支运行 `bash start.sh --reconstruction-backend ufo '你的任务'`。在任务中提供真实 asset_manifest_path；没有新资产文件不能完成真实插入。独立脚本可先测试，无需启动整个 Agent。
