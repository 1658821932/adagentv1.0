// 真实 adagent 项目 9 工具的展示映射
export const toolNames = {
  failure_analyzer: '失败分析',
  scene_instance_generator: '场景需求生成',
  video_retriever: '视频检索',
  edit_planner: '攻防编辑规划',
  attack_vehicle_editor: '攻击车编辑',
  ego_trajectory_editor: '自车轨迹编辑',
  policy_runner: '策略评测',
  failure_case_filter: '失败案例筛选',
  policy_trainer: '后训练'
};
export const stageMap = {
  failure_analyzer: ['任务理解', 'failure_analysis'],
  scene_instance_generator: ['场景构造', 'requirement_generation'],
  video_retriever: ['场景构造', 'retrieval'],
  edit_planner: ['场景编辑', 'edit_planning'],
  attack_vehicle_editor: ['场景编辑', 'attack_editing'],
  ego_trajectory_editor: ['场景编辑', 'ego_editing'],
  policy_runner: ['闭环仿真', 'simulation'],
  failure_case_filter: ['策略训练', 'failure_filtering'],
  policy_trainer: ['策略训练', 'training']
};
export const observations = {
  failure_analyzer: '失败分析结果已返回，继续将失败原因转成场景需求。',
  scene_instance_generator: '场景需求与候选方案已生成，继续检索匹配的基础视频。',
  video_retriever: '已检索到候选视频，继续制定编辑方案。',
  edit_planner: '编辑方案已明确，继续执行场景编辑。',
  attack_vehicle_editor: '攻击车编辑已完成，检查产物并继续。',
  ego_trajectory_editor: '自车轨迹编辑已完成，检查产物并继续。',
  policy_runner: '闭环评测结果已返回，继续筛选失败案例。',
  failure_case_filter: '失败案例筛选完成，存在有效训练数据时进行后训练。',
  policy_trainer: '训练结果已返回，检查任务目标与输出，准备结束。'
};
