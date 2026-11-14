# RL 训练标注

tool07 在原生 WE 闭环仿真中开启 with_dense_reward_manager 后，自动导出 RL 标注包。普通评测、PDM challenge 和专家视频回放不等同于 RL 数据采集。

## 当前时间范围约定

- 普通原生评分与RL采集统一使用 `num_history=4`、`max_frame_per_scene=16`、`num_future=12`（可省略），`sampling_poses` 自动推导。在2Hz场景中，策略接管及评分范围为 **1.5–7.5秒**。
- 奖励模块维持原有8帧未来窗口，采集时源场景至少需要24帧真实数据。源场景较长不会自动延长正式评分；不得补帧。
- 本次没有修改奖励模块、尾帧采样规则或训练算法。尚未完成的全程奖励适配已撤回。
- 标注导出失败时，不得自行改成全程无奖励评测作为本轮替代结果；不同rollout的分数与奖励不得拼接。只有用户明确要求其它时间范围时才覆盖帧预算，并明确报告实际范围。
- `full_rollout_pdm` 表示覆盖本次配置的完整rollout，不表示覆盖源场景全部帧。标准16帧的正分只代表1.5–7.5秒内通过评分。

## 包含内容

- annotations.pkl：WE 实际导出的逐帧标注，包括自车位姿、历史/未来轨迹及有效掩码、他车框/速度/身份、相机内外参与图像路径。
- manifest.json：场景/帧索引、实际选择的轨迹词表索引、候选奖励路径、源文件校验值及配置/检查点来源。
- tokens.json：符合现有筛选规则的场景；只选官方闭环 PDM 总分为零的场景。
- all_scene_tokens.json：本次全部场景。

图像、8192 候选奖励和轨迹词表保留在原路径，通过清单引用，不重复复制。标注包位于该次任务 output/rlft_annotations 下，不应上传 GitHub。

## 衔接 tool09

当 rlft_data_report.status 为 ready，将 tool07 返回的 rlft_training_data 原样作为 tool09 的 training_data，并指定 training_mode=rlft、base_checkpoint 和训练参数。四个路径字段是 rollout_dir、token_file、vocab_path、manifest_path。

训练准备阶段验证源文件并读取 annotations.pkl，然后生成隔离的数据目录、奖励缓存和训练配置。实际动作与奖励按官方规则对齐：plan_idx.csv 的 step - 1 等于标注 frame_idx。轨迹词表必须与采集奖励所用词表一致。

如果返回 no_eligible_scenes，标注仍已导出，但没有符合零分规则的训练场景，rlft_training_data 为 null，不应启动训练。

## 标注边界

这里保留的是仿真实际状态、轨迹和候选奖励，不会自动生成“正确专家轨迹”。编辑后的正确专家轨迹属于 ILFT 数据，不能用来冒充 RL 的实际动作或奖励。缺失标注、图片、动作或奖励时明确报错。

## 已验证范围

真实现有 rollout 的 12 帧、8 路相机、9 组奖励已完成导出校验；零分训练包到 tool09 数据准备使用受控测试数据验证。没有启动 GPU 训练，也没有用测试数据证明模型提升。
