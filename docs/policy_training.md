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
    "vocab_path": "/data/test_8192_kmeans.npy",
    "sample_filter": "v2"
  },
  "base_checkpoint": "/weights/base.pth",
  "epochs": 1,
  "seed": 0
}
```

rollout_dir 必须包含 meta_datas、sensor_blobs、pdms_pkl，后者包括 plan_idx.csv 和各时刻8192条候选轨迹的真实奖励。仅有官方场景pkl、高斯ckpt或视频不能训练RL。输入应来自本地可信的WE输出，不能直接加载任意来源的pickle。

默认RL配置为原生VADv2 RL rollout配置，并显式开启RL和奖励分支。可用config_path指定相同数据约定的配置。
v2只保留每个场景第一个可规划时刻，适合接口验证；v1按原生困难样本规则筛选，可能得到空数据。当前RL不混入原始训练集，后续可以单独扩展混合采样。

执行在服务runtime任务目录内完成。元数据、奖励缓存和生成配置保存在该任务中；原始仿真输出不修改。
返回新checkpoint、日志、数据准备统计、模式对应损失和参数更新检查。训练成功不等于性能提升，仍需相同结构和固定场景独立评测。
IL配置默认沿用既有Waymo适配配置，其他数据格式须显式传config_path。

本次扩展训练入口，不代表检索、编辑和RL评测的整个Agent闭环已全部接通。

## 上游工具的分工
- policy_runner：IL沿用原有数据生成；RL调用时设置 with_dense_reward_manager=true，返回rollout_dir和reward_dir，并导出实际plan_idx.csv。
- failure_case_filter：保留动态对象几何诊断，用于辅助分析，不作为RL奖励，也不能代替官方道路/碰撞评分。
- policy_trainer：ilft读取标注；rlft按真实候选奖励执行v1/v2样本筛选后训练。RL可以直接由policy_runner接policy_trainer，无需通过旧几何筛选器。

仿真入口会使用仿真解释器在运行目录重存科学数据，以兼容NumPy 1/2的pickle模块路径，不修改源场景包。

## 本次验证
- 19项相关接口、执行器和仿真导出测试通过；未运行完整测试套件。
- IL：沿用已有Waymo转换标注，通过工具注册入口完成1次更新并保存权重，60个参数张量变化。
- RL：navtrain单场景经policy_runner生成12帧仿真数据和9份候选奖励，直接交给policy_trainer完成1次更新并保存权重，100个参数张量变化。
- 数据预检覆盖有效v2样本、v1空筛选、IL历史不足及RL奖励缺失。
- 这是工具级串联验证，不包含LLM自主决策、编辑场景接入或训练后性能评测；不代表批量训练已验证。
