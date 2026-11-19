/** Current tool02/tool03 results. Rendering uses the shared detail-group layout. */
const text = value => typeof value === 'string' ? value.trim() : '';
const LABELS = {
  straight_road: '直路', curve: '弯道', t_junction: '丁字路口',
  cross_intersection: '十字路口', merge: '汇流',
  straight: '直行', turn_left: '左转', turn_right: '右转',
  left_offset: '左偏移', keep: '保持', right_offset: '右偏移', unknown: '未知',
  ahead_same_lane: '同车道前方', behind_same_lane: '同车道后方',
  left_adjacent: '左相邻车道', right_adjacent: '右相邻车道',
  oncoming: '对向车道', left_arm: '左侧支路', right_arm: '右侧支路',
  none: '无',
};
const label = value => LABELS[value] || text(value);
const edit = value => value === null ? '无需编辑' : text(value);
const score = value => Number.isFinite(value) ? value.toFixed(3) : '';

function groupsWithValues(groups) {
  return groups.map(group => ({ ...group, rows: group.rows.filter(([, value]) => value !== '') }))
    .filter(group => group.rows.length);
}

function errorGroups(error) {
  return error ? groupsWithValues([{ title: '执行失败', rows: [
    ['错误类型', text(error.type)], ['说明', text(error.message)],
  ] }]) : [];
}

export function scenarioSummary(result) {
  return text(result?.target_scene);
}

export function scenarioDetailGroups(result, error) {
  if (!result?.target_scene) return errorGroups(error);
  const attack = result.attack || {};
  const groups = [
    { title: '整体目标场景', rows: [['目标场景', text(result.target_scene)]] },
    { title: '共同交互机制', rows: [
      ['道路形态', label(attack.map_context)], ['自车动作', label(attack.ego_action)],
      ['自车车道保持', label(attack.ego_lane_keeping)],
      ['他车来向', label(attack.other_origin)], ['他车动作', label(attack.other_action)],
      ['他车车道保持', label(attack.other_lane_keeping)],
    ] },
  ];
  for (const candidate of result.candidates || []) {
    groups.push({ title: `候选 ${text(candidate.candidate_id)}`, rows: [
      ['检索条件', text(candidate.base_video_requirement)],
      ['检索词', text(candidate.retrieval_query)],
      ['攻击车编辑', edit(candidate.attack_edit_strategy)],
      ['自车输入编辑', edit(candidate.ego_input_edit_strategy)],
      ['自车正确轨迹', text(candidate.ego_correct_trajectory_strategy)],
    ] });
  }
  return groupsWithValues(groups);
}


const PLANNER_VERDICTS = { ready: '可编辑', unsuitable: '不适合编辑', needs_information: '信息不足' };
const PLANNER_CHECKS = { road: '道路', lanes: '车道', space: '空间' };
const PLANNER_STATES = { pass: '通过', fail: '不符', unknown: '未知' };
const scalar = value => text(value) || (Number.isFinite(value) ? String(value) : '');
const shortText = (value, limit = 110) => {
  const valueText = text(value);
  return valueText.length > limit ? valueText.slice(0, limit) + '…' : valueText;
};

export function editPlannerSummary(result) {
  if (!result) return '';
  const checks = result.checks || {};
  const checkText = Object.entries(PLANNER_CHECKS)
    .map(([key, label]) => checks[key] ? `${label}${PLANNER_STATES[checks[key]] || checks[key]}` : '')
    .filter(Boolean).join(' · ');
  const plan = result.adapted_plan || {};
  const action = shortText(plan.attack_edit_strategy || plan.ego_input_edit_strategy || plan.ego_correct_trajectory_strategy);
  return [PLANNER_VERDICTS[result.suitability] || text(result.suitability), checkText, action]
    .filter(Boolean).join(' — ');
}

export function editPlannerDetailGroups(result, error) {
  if (!result) return errorGroups(error);
  const checks = result.checks || {};
  const plan = result.adapted_plan || {};
  const facts = result.evidence?.map_facts || {};
  const target = facts.target_junction || {};
  const reasons = (result.reasons || []).map(text).filter(Boolean).join('；');
  const handoff = result.attack_input;
  const handoffLabel = handoff?.operation
    ? ({ auto: '自动选择车辆', edit_existing: '编辑现有车辆', add_vehicle: '新增车辆' }[handoff.operation] || handoff.operation)
    : '';
  const handoffText = handoffLabel
    ? `${handoffLabel}${Array.isArray(handoff.plan_options) && handoff.plan_options.length ? ` · ${handoff.plan_options.length} 个方案` : ''}`
    : '';
  return groupsWithValues([
    { title: '结构检查', rows: [
      ['结论', PLANNER_VERDICTS[result.suitability] || text(result.suitability)],
      ['道路', PLANNER_STATES[checks.road] || text(checks.road)],
      ['车道', PLANNER_STATES[checks.lanes] || text(checks.lanes)],
      ['空间', PLANNER_STATES[checks.space] || text(checks.space)],
      ['目标路口', scalar(facts.target_junction_fid ?? target.intersection_fid)],
      ['物理支路', scalar(target.physical_branches)],
      ['进入帧', scalar(facts.target_entry_frame_index)],
      ['进入距离', scalar(facts.target_entry_distance_m)],
    ] },
    { title: '适配方案', rows: [
      ['攻击车编辑', edit(plan.attack_edit_strategy)],
      ['工具⑤交接', handoffText],
      ['自车输入编辑', edit(plan.ego_input_edit_strategy)],
      ['自车正确轨迹', text(plan.ego_correct_trajectory_strategy)],
    ] },
    { title: '核对依据', rows: [
      ['来源', text(result.evidence?.verification_source)],
      ['原因', reasons], ['检索建议', text(result.retrieval_suggestion)],
      ['候选', text(result.candidate_id)],
    ] },
  ]);
}

export function retrievalSummary(result, input) {
  if (!result) return '';
  return [
    result.mode === 'mock' ? '模拟检索' : '',
    text(result.candidate_id) ? `候选 ${result.candidate_id}` : '',
    score(result.match_score) ? `相关分数 ${score(result.match_score)}` : '',
    text(result.scene_ref?.scene_token) ? `场景 ${result.scene_ref.scene_token}` : '',
    text(input?.requirement),
  ].filter(Boolean).join(' · ');
}

export function retrievalDetailGroups(result, input, error) {
  if (!result) return errorGroups(error);
  const scene = result.scene_ref;
  const segment = result.source_segment;
  const range = Number.isFinite(segment?.start_time) && Number.isFinite(segment?.end_time)
    ? `${segment.start_time}–${segment.end_time} 秒` : '';
  return groupsWithValues([
    { title: '检索结果', rows: [
      ['候选编号', text(result.candidate_id)], ['检索条件', text(input?.requirement)],
      ['相关分数', score(result.match_score)],
      ['执行模式', result.mode === 'mock' ? '模拟结果，无真实检索绑定' : text(result.mode)],
      ['索引命中区间', range],
    ] },
    { title: '源场景绑定', rows: scene ? [
      ['场景编号', text(scene.scene_id)], ['场景标识', text(scene.scene_token)],
      ['命中相机', text(scene.matched_camera)],
      ['场景帧数', Number.isInteger(scene.frame_count) ? String(scene.frame_count) : ''],
      ['原图完整性', scene.images_complete === true ? '完整' : scene.images_complete === false ? '不完整' : ''],
      ['使用范围', '完整源场景；索引命中区间不是多相机同步裁剪'],
    ] : [['绑定状态', '未绑定真实场景']] },
    { title: '视频来源', rows: [
      ['检索导出视频', text(result.video_path)], ['原始索引视频', text(result.source_file)],
    ] },
  ]);
}
