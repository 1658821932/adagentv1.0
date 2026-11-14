# 已知问题

## EDIT-003：攻击前缀锁被误写进观察历史字段，05入口反复冲突

- 复现：`logs/run_20261114_112554_497193` 的7次05调用均在搜索前报 `Conflicting task and acceptance setting: observation_end_s`，主Agent改计划/场景后仍复现，耗尽20步。06–09未执行，不能认定攻击或防御无解。
- 根因：backend把 `attack_prefix_lock_s=0` 写进顶层 `observation_end_s`，workflow又将其与acceptance中的实际历史比较。两个不同概念被当作同一个设置；错误包装只留下ValueError，主Agent看不到字段、值和处理建议。
- 修复：backend分别传递历史和攻击前缀；共享配置归一化保留真正的同字段冲突检查；对象选择、攻击编辑器采样与构造任务读取独立前缀。旧v1归档的明确历史/锁边界保留。当前观察历史1.5秒、攻击锁0秒和事件窗口2–7秒可同时成立；本次不修改危险窗口、物理/道路约束或07的1.5–7.5秒评分。
- 错误反馈：配置错误类型为 `editing_configuration_error`，附带 `failure_stage=configuration`、`retryable=false`、冲突字段/值、日志路径和恢复建议；经过backend JSON、service、05/06工具封装与主Agent上下文压缩完整保留。主提示词禁止通过改场景或原样重试绕过配置错误。其它执行错误也保留原始有界原因。
- 验证：86项相关离线回归通过，覆盖两种攻击入口、旧归档、同字段真冲突、错误协议/工具/上下文全链路、评分窗口和取消比例后的行为。更新了旧测试中的缺失依赖，以及将源frame_count误当07入参的夹具；没有放宽实际07参数。
- 真实输入预检：复用首个失败05保存的请求，实际backend→workflow通过配置、场景和资产检查并写出请求准备产物，在读取模型profile之前强制停止。0次API、0次仿真，未重跑完整攻击或改写源场景。记录：`outputs/attack_time_configuration_fix_20261114_122802/verification.json`。这证明原入口错误已消除，不代表攻击轨迹已经生成或后续验收必然成功。
- 后续：原01–04结果可保留；在同一execution_scope对失败作业显式retry后从05继续，不能直接复用旧失败缓存。前端是否支持同会话续跑应以当前实际入口接线为准。

## 2026-11-14 审查补充

见 [工作流审查与配置面板交接](REVIEW_20261114.md)。本轮统一05默认关闭自测与06证据/ILFT说明，保留1.5–7.5秒窗口。用户随后要求取消自动50%规则，已清理构造器/编辑器/复核的默认回退，修正中间等待与最终停车的目标提示；07不再自动判断动作完成，只返回测量。108项相关Python回归通过；未跑真实模型或仿真。配置面板及历史加载、状态显示已接入，39项相关Node测试通过、本机5173提供新版资源，未跑浏览器自动化。当前页面续跑接线缺口仍需单独恢复。

## WE-004：统一普通原生评测与RL采集的时间范围

- 当前决定：普通07评分和奖励采集都固定使用4帧历史、16帧rollout，在2Hz场景中评分1.5–7.5秒。主Agent提示词、04规划说明、07公开字段/工具说明、直接客户端及原生launcher默认值一致。只有用户明确要求不同范围时才覆盖。
- 撤回：移除未完成的全程奖励窗口适配及其native入口，保持WorldEngine原有8帧奖励前瞻与训练采样逻辑。保留WE-003的导出修复。
- 边界：这不保证7.5秒之后通过；全程评分仍可显式作为另一项诊断，但不得替代标准窗口结果或借用其它rollout奖励。导出失败后禁止Agent自行改窗口兜底。
- 验证：57项离线回归通过，覆盖公开默认参数、直接客户端、launcher的1.5–7.5秒一致性、显式其它窗口、RL导出及GL路径修复。没有启动模型API、GPU仿真或训练。

## WE-003：RL标注导出丢失失败筛选函数与直接脚本导入路径

- 复现：`run_20261112_114634_003f9a141468` 与 `run_20261112_115133_a0bdd7035cfe` 的原生仿真完成后，`rlft_annotations.py` 导入 `failed_scene_anchors` 失败。
- 根因：统一官方分数CSV读取时移除了RL导出仍使用的筛选函数；恢复函数后，实际AlgEngine直接执行脚本还暴露仓库根目录未加入模块搜索路径，无法导入 `common.pdm_scores`。
- 修复：保留共享CSV读取器，恢复原有零分场景与允许anchor交集筛选；直接启动RL导出脚本时显式加入仓库根目录。没有修改评分、奖励、仿真窗口或训练准入条件。
- 验证：19项筛选、标注/奖励/动作校验、直接脚本入口回归通过。复用第二次真实16帧rollout独立导出成功（8.129秒，16帧标注、13帧奖励）；验证文件 `outputs/rl_annotation_export_fix_20261112_120957/verification.json`，未重新仿真或训练。初次预检发现的common导入错误保留在 `outputs/rl_annotation_export_fix_20261112_120856/`。
- 历史结果：两次16帧采集评估1.5–7.5秒，官方PDM为1，导出正确返回 `no_eligible_scenes`；另一次24帧无奖励全程评分为0，首次违规第17帧。当前按WE-004统一采用短窗口；历史全程结果不能替代该窗口结果或与其奖励拼接。

## WE-002：07原生仿真未继承已配置的GL库路径

- 复现：`logs/run_20261112_113216_013351` 两次07调用，原生仿真在导入cv2时缺少 `libGL.so.1`。服务器已有GL库，GPU驱动正常；不是缺少策略权重，也不需要安装libgl1或更换CUDA。
- 根因：WorldEngine adapter/settings.json 已有 shared_library_dir，但policy客户端、simulation service和原生launcher没有将其加入子进程的 `LD_LIBRARY_PATH`。WFA与WE是不同启动入口，修复WFA不会自动修复07。
- 修复：config.runtime_env统一构造路径、保留调用方已有路径/trace/CUDA设备环境；policy runner、场景准备、双进程launcher、dense奖励注释导出及后训练都显式使用。effective_config.json记录实际库目录和搜索路径，便于后续核对。路径已注入则去重，不覆盖调用方其它环境。
- 回归：32项离线测试通过，覆盖直接launcher、仿真服务脚本、默认策略调用及dense分支；真实双环境预检通过，记录在 `outputs/worldengine_library_preflight_20261112_113759/verification.json`。SimEngine OpenCV4.10和专家包坐标适配器加载通过（83.9秒），AlgEngine OpenCV4.9、MMCV及策略配置加载通过（39.168秒），两侧Torch2.0.1+cu118均报告CUDA可用。策略侧首次冷导入120秒超时已保留，分阶段重试通过，不能误报为新的libGL故障。预检未启动完整仿真、模型API或训练。

## WE-001：07 输入校验强制模型参数，拦截已有的默认策略解析

- 复现：`logs/run_20261112_103911_730785` 在01–06已有结果后调用07，主Agent的原生工具调用在输入模型校验阶段被拒绝，没有启动实际仿真。
- 根因：prompt要求省略默认 `config_path/checkpoint_path/model_name/data_type`，适配层已有 `resolve_policy_inputs` 且默认文件真实存在，但公开 Input 的 after-validator 仍将四字段当作 native 必填。同时07描述仍写着必须在06前测攻击，与当前全流程顺序冲突。
- 修复：四字段作为可选覆盖交由适配器统一解析并核实文件；保留帧预算、PDM模式与显式缺失文件检查。工具字段说明和流程描述与“06专家包→07”一致；只有用户明确要求攻击对照时才测05包。
- 同路径修复：run_policy末尾函数内重复导入 WorldEngineConfig 会遮蔽顶部导入，在dense分支较早使用时触发局部变量未赋值；移除重复导入，并对dense导出调用路径增加回归。
- 验证：45项离线回归通过；原始被拒07调用回放可通过Orchestrator校验，实际专家包与默认VADv2配置/权重存在，16帧+dense设置保持。未运行真实模型、GPU仿真或训练。证据：`outputs/policy_defaults_fix_20261112_110850/verification.json`。
- 默认策略仍由 `mcp_servers/worldengine/adapter/settings.json` 与config.py维护，和后训练默认checkpoint一致，不要求用户重新提供已配置资源。

## WFA-002：异常 details 接口不一致，掩盖再次出现的 libGL 依赖错误

- 复现：`logs/run_20261112_084130_479533` 的两次失败分析。原始后端均报 `ImportError: libGL.so.1`，工具包装再访问不存在的 `WfaError.details`，最终只呈现 AttributeError。
- 实际状态：当时 service.py 的 WfaError 恢复为空异常类，且启动代码没有使用已存在的 runtime_env；工具包装仍依赖新版 details 接口。不能将此错误归因于归档或场景输入。
- 修复：异常构造器默认初始化 details，包装层兼容缺失/None/非字典旧字段；子进程通过 runtime_env 注入既有 GL 库路径，并保留后来新增的 WFA_LLM_PROFILE 工具模型选择。非零退出先读取场景状态再上报，依赖错误明确标记为 environment、不可原样重试，保留原始原因和日志路径。
- 回归：45项离线测试通过，新增了对实际启动参数同时传入库路径、模型 profile、trace 的断言，以及旧异常接口兼容测试。
- 真实归档验证：通过实际工具 → 原生 CLI → 12帧图像/数值预处理 → 工具结果回传，120张证据图，耗时43.033秒；仅模型诊断替换为明确标注的离线桩并禁止模型请求，不是新的AI诊断，也未跑完整攻防。记录：`outputs/wfa_regression_preflight_20261112_084710/verification.json`。

## EDIT-002：攻击对象选择的计时器与局部 time 变量冲突

- 状态：已修复（2026-11-12）。复现运行：`logs/run_20261112_060801_663582`。
- 根因：补子工具计时时新增 `time.perf_counter()`，但 `select_object` 中原有 `for time, ...` 使 `time` 成为局部变量，第一次处理模型响应前即触发 `UnboundLocalError`。三次攻击调用均未进入轨迹构造。
- 修复：攻击模块独立导入 `perf_counter`，与场景时间变量分离。新增执行对象选择首个响应的回归，并更新攻击子循环测试的计时依赖和有效调用 ID。
- 验证：33 项离线回归通过；用三次已保存的公开原生响应在真实场景回放，无新 API 调用。前两次 `edit_existing` 正常完成对象选择；第三次 `add_vehicle` 的原始放置命令被道路几何检查拒绝（初始车身越出道路允许范围），需模型根据反馈换位置。反复回放同一命令仅用于核验几何，不代表模型实际的后续修改。
- 回放记录：`outputs/attack_selection_replay_20261112_062743/verification.json`。验证仅覆盖对象选择，未恢复整轮攻防。
- 原全流程已结束；可复用本轮检索场景和粗规划，从 05 恢复。首轮已有车辆粗计划：`logs/run_20261112_060801_663582/tools/step_04_edit_planner/output.json`。

## WFA-001：失败分析缺少 OpenCV 动态库路径，错误被误报为归档问题

- 状态：已修复（2026-11-12）。复现运行：`logs/run_20261112_055756_432741`。
- 实际错误：WFA 已完成归档和场景定位，但图像预处理导入 OpenCV 时缺少 `libGL.so.1`，子进程退出码为 2；原适配层只回传退出码，主 Agent 因此反复切换归档别名。
- 修复：WFA 适配层用自己的 `adapter/settings.json` 的 `shared_library_dir` 显式构造 `LD_LIBRARY_PATH`；支持 `WFA_LIBRARY_DIR` 覆盖，并保留原环境及追踪字段。复用服务器已有库，不修改 CUDA 或安装系统依赖。
- 错误反馈：非零退出时先读取本轮场景状态，保留真实错误类型及诊断；依赖导入错误返回 `failure_stage=environment`、`retryable=false`、归档/场景校验标志和日志路径。这些字段保留在主 Agent 的压缩上下文中，提示词要求此时不再切换归档别名重试。
- 真实验证：相同归档与 `dd162898ecf351d4` 场景的官方指标读取、12 帧解码、120 张证据图和数值预处理通过，耗时 38.181 秒；未调用外部 API。记录：`outputs/wfa_preflight_20261112_060442/verification.json`。
- 验证范围：原始依赖故障与预处理链路已验证；不等同于整轮攻防、仿真和训练完成。

## EDIT-001：tool06 内部回放未按新增资产 size_m 缩放

- 记录日期：2026-11-10（Asia/Shanghai）。状态：已修复（2026-11-10）；已完成数值与真实 PLY/CUDA 小图回归。
- 当前编号：tool06 为自车轨迹编辑与内部回放；tool07 为策略仿真，tool09 为后训练。
- 根因：core/adversarial/render_added_vehicle_mtgs.py 的 added_vehicle_model 读取原 PLY 后直接构造 AddedVehicle。size_m 仅参与道路平整度检查，高斯中心/协方差没有缩放；模型中心高度也使用原始 PLY 高度。
- 调用方：render_ego_mtgs_three_view.py 在初始化和逐帧更新时调用该函数。
- 对照路径：native_vehicle_asset.py 的 resize_gaussians 已按目标尺寸缩放中心与协方差，merge 按目标高度计算贴地位置；tool07 使用此原生导出路径。
- 数值复现：现有 1.ply 请求 size_m=[6,2.5,2]，内部路径几何仍为 [4.4321833,2.0330679,1.6374142]；原生导出为 [6,2.5,2]。平地中心高度分别为 0.8187071m 与 1m。
- 验证范围：读取实际 PLY，对现有函数做 CPU 数值对照；没有执行尺寸对比 GPU 渲染。
- 影响：尺寸改变时，内部展示几何与标注/原生仿真几何不一致；不能据此认为朝向未生效，现有旋转路径已传递 yaw 并旋转中心和四元数。
- 建议修复：共用资产读取/缩放逻辑，统一缩放后贴地高度；增加两条路径几何和协方差的一致性验证。
- 效率改进：按资产绝对路径、文件版本和目标尺寸缓存 CPU 读取/缩放结果；不同姿态共享静态几何。GPU 模型仍由原逐帧路径构造，未扩展此次修复范围。


### 修复与验证

- 内部回放与原生导出共用 read_vehicle_ply / resize_gaussians，移除重复的内部 PLY 解码实现。
- 同步缩放中心与高斯协方差，并将 Gaussian 参数转换到与原生 checkpoint 一致的 float32；内部 opacity 保持渲染器所需的一维形状。
- 贴地中心使用请求的高度 / 2，记录原尺寸、目标尺寸、轴向缩放比例与实际缩放后几何尺寸。
- resize_gaussians 不再原位归一化输入四元数，避免污染共享缓存。
- 数值回归覆盖：非等比缩放/旋转协方差、平移/朝向/高度、非法尺寸、不同尺寸及文件版本的缓存隔离。
- 实际 1.ply / gsplat CUDA 验证：原尺寸、[6,2.5,2]、[3,1.4,1.2] 均通过。对应前景像素数 10583、17260、5205；放大/缩小在实际渲染中生效。
- GPU 旋转协方差验证通过，源 PLY SHA-256 前后相同。
- 产物：logs/asset_size_fix_20261110_063812/report.json 及 original.png、large.png、small.png。
- 范围：实际内部渲染模型与 CUDA 光栅化的小图测试；本次未重跑完整 tool06 场景视频。
