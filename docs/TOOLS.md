# 当前工具列表

预留 tool01–tool10 十个编号，当前主 Agent 注册九个工具；tool07 保留代码但不启用，注册名用于实际调用。

| 编号 | 注册名 | 职责 |
|---|---|---|
| 01 | failure_analyzer | 失败分析与证据归纳 |
| 02 | scene_instance_generator | 目标场景、自车目标和三个候选方案 |
| 03 | video_retriever | 底视频检索与多相机场景绑定 |
| 04 | edit_planner | 实际底场景的快速可编辑性判断与粗规划 |
| 05 | attack_vehicle_editor | 攻击车规划、轨迹编辑与冻结 |
| 06 | ego_trajectory_editor | 自车专家轨迹规划、编辑与复核 |
| 07（未启用） | scene_replayer | WE 原生给定轨迹回放与视频导出 |
| 08 | policy_runner | PDM 挑战或驾驶策略闭环仿真 |
| 09 | failure_case_filter | 基于官方 PDM 评分筛选失败案例 |
| 10 | policy_trainer | ILFT/RLFT 后训练 |

## 衔接

- tool02 的选定候选检索需求原文传给 tool03；保留全局 scene_requirement、target_effect 与配对 edit_strategy。
- tool04 通过 retrieval_result 读取 tool03 完整结果或保存的结果引用，不接受旧 scene_ref 参数。ready 才进入编辑；unsuitable 提供新检索方向，needs_information 返回缺失信息。它不运行轨迹优化或 PDM。
- tool05 使用原 scene_ref 和粗计划的 operation、target_track/maneuver、instruction。新增车辆还需实际资产和尺寸。tool05/06 内部继续负责精细轨迹搜索。
- tool06 使用冻结攻击结果及全局 target_effect，保持攻击场景不变。
- tool07 使用准备好的原生 simulation_input 和精确 scene_id，以 trajectory_policy/log_play_controller 回放，不启动 AlgEngine。编辑结果的 simulation_input 为 null 时需要原生导出后才能回放，不能拿未编辑原场景冒充。
- tool08 测试驾驶策略，tool07 执行给定轨迹。独立 PDM challenge 不等于官方 PDM 总分。
- tool09 读取官方 PDM 总分，零匹配也是有效筛选结果。tool10 的训练完成不等于驾驶指标提升。

## 视觉工具删除

原独立视觉验收工具已直接删除，没有迁入新工具或共享模块。tool07 仅检查原生回放执行、完整时间窗及视频解码，不提供 VLM 语义验收。历史运行产物保持原样。

## mock / real

每个工具具有 description.py、schemas.py、real.py、mock.py；LLM 工具另有 prompt.md 和 config.json。mock 不代表真实产物或验收通过。解释器、数据、权重和 API 配置是 real 模式的独立前提。

## tool04 视觉观察

初始从完整绑定场景均匀选择12个时刻，固定看CAM_F0前视角，按时间顺序每4帧拼成一张2×2图，通常共3张。短场景只使用真实存在的帧；缺图单元标记缺失，不复制其他帧补齐。
VLM可在工具内部返回request_views，指定实际相机名称和观察目的。每轮最多请求2路相机，最多补看2轮；补看沿用相同时间点，不新增主Agent工具或对外输入字段。最终仍返回原有粗规划结果。
设置位于tools/tool04_edit_planner/observation.json；拼图和实际补看记录保存在忽略上传的logs/tool04_observations/下。evidence.visual_observation记录图片、是否成功送达、补看请求及model_calls；meta.attempts仍是外层格式校验尝试数。


---

# IL / RL 训练工具
工具名称保持 policy_trainer，training_mode 可选 ilft（默认）或 rlft。

## IL：保留原有标注入口
```json
{
  "training_mode": "ilft",
  "training_data": {"ann_file": "/data/infos.pkl", "token_file": "/data/train_tokens.yaml"},
  "base_checkpoint": "/weights/base.pth",
  "epochs": 1
}
```

## RL：使用 WE 运行后导出的数据
```json
{
  "training_mode": "rlft",
  "training_data": {
    "rollout_dir": "/data/run/WE_output/openscene_format",
    "token_file": "/data/navtrain_tokens.yaml",
    "vocab_path": "/data/test_8192_kmeans.npy"
  },
  "base_checkpoint": "/weights/base.pth",
  "epochs": 1,
  "seed": 0
}
```

rollout_dir 必须包含 meta_datas、sensor_blobs、pdms_pkl（包括 plan_idx.csv 和各时刻8192条候选轨迹的真实奖励），以及闭环评测导出的 all_scenes_pdm_averages_*.csv。仅有官方场景pkl、高斯ckpt或视频不能训练RL。输入应来自本地可信的WE输出，不能直接加载任意来源的pickle。

默认RL配置为原生VADv2 RL rollout配置，并显式开启RL和奖励分支。可用config_path指定相同数据约定的配置。
失败案例按整段场景闭环PDM总分筛选：仅保留 all_scenes_pdm_averages_*.csv 中 score 为 0 的场景，使用这些场景的全部有效帧训练；各时刻8192条候选轨迹的奖励继续作为RL奖励。token_file 中没有0分场景时明确报错，不再提供 v1/v2 逐帧筛选。当前RL不混入原始训练集，后续可以单独扩展混合采样。

执行在服务runtime任务目录内完成。元数据、奖励缓存和生成配置保存在该任务中；原始仿真输出不修改。
返回新checkpoint、日志、数据准备统计、模式对应损失和参数更新检查。训练成功不等于性能提升，仍需相同结构和固定场景独立评测。
IL配置默认沿用既有Waymo适配配置，其他数据格式须显式传config_path。

本次扩展训练入口，不代表检索、编辑和RL评测的整个Agent闭环已全部接通。

## 上游工具的分工
- policy_runner：IL沿用原有数据生成；RL调用时设置 with_dense_reward_manager=true，返回rollout_dir和reward_dir，并导出实际plan_idx.csv。
- failure_case_filter：只读取官方 PDM 总分筛选失败案例，不提供几何碰撞诊断。
- policy_trainer：ilft读取标注；rlft按整段场景闭环PDM总分==0筛选失败场景后，用真实候选轨迹奖励训练。RL可以直接由policy_runner接policy_trainer；tool09 同样只读取官方 PDM 结果进行筛选。

仿真入口会使用仿真解释器在运行目录重存科学数据，以兼容NumPy 1/2的pickle模块路径，不修改源场景包。

## 本次验证
- 19项相关接口、执行器和仿真导出测试通过；未运行完整测试套件。
- IL：沿用已有Waymo转换标注，通过工具注册入口完成1次更新并保存权重，60个参数张量变化。
- RL：navtrain单场景经policy_runner生成12帧仿真数据和9份候选奖励，直接交给policy_trainer完成1次更新并保存权重，100个参数张量变化。
- 数据预检覆盖PDM 0分场景筛选、无0分场景报错、IL历史不足及RL奖励缺失。
- 这是工具级串联验证，不包含LLM自主决策、编辑场景接入或训练后性能评测；不代表批量训练已验证。

## 当前接口边界
只接受当前字段：tool01.token、tool02.variation_hints、tool09.scenario_pkl、
tool10.base_checkpoint、epochs 和结构化 training_data（token_file）。
旧字段别名及扁平训练参数已删除。历史 scene_instance 摘要不再作为 existing_instances 输入。
编辑只发布 edit_speed_profile 和 edit_route；旧轨迹算法与无版本历史编辑重放不再支持。
攻击任务合同必须为 attack_task.v2。恢复任务必须使用当前上下文版本。
API 地址与模型读取 JSON 配置，密钥读取 profile 指定的环境变量；不再读取 LLM_* 旧变量。
