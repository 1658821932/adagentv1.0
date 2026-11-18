/** Tool 01 wording and detail rows shared by the run view and the frontend tests. */

const OUTCOME = { collision: '碰撞', road_departure: '出界', collision_and_departure: '碰撞且出界' };
const INTERACTION = {
  rear_end: '追尾', crossing: '交叉冲突', side_swipe: '侧向刮擦',
  lane_change: '变道冲突', single_vehicle: '单方事故',
};
const MAP = {
  straight_road: '直路', curve: '弯道', t_junction: '丁字路口',
  cross_intersection: '十字路口', merge: '汇流',
};
const EGO_ACTION = {
  straight: '直行', turn_left: '左转', turn_right: '右转',
  lane_change_left: '向左变道', lane_change_right: '向右变道',
};
const OTHER_ORIGIN = {
  ahead_same_lane: '同车道前方', behind_same_lane: '同车道后方',
  left_adjacent: '左相邻车道', right_adjacent: '右相邻车道',
  oncoming: '对向车道', left_arm: '左侧支路', right_arm: '右侧支路',
  none: '无目标车辆',
};
const OTHER_ACTION = {
  straight: '直行', turn_left: '左转', turn_right: '右转',
  cut_in_left: '从左侧切入', cut_in_right: '从右侧切入',
  brake: '急刹', stop: '停止', squeeze_left: '从左侧贴近', squeeze_right: '从右侧贴近',
};
const WEATHER = { clear: '晴', cloudy: '多云', rainy: '雨', foggy: '雾', snowy: '雪', unknown: '未知' };
const STAGE = {
  start: '启动', input: '输入检查', data: '读取数据', images: '准备图像',
  vlm: '模型核对', verification: '核对结论', done: '完成',
};

const named = (table, value) => (value && table[value]) || (typeof value === 'string' ? value : '');
const text = value => (typeof value === 'string' ? value.trim() : '');

export function stageLabel(stage) {
  return named(STAGE, stage);
}

/** One-sentence result: current format first, historical runs second. */
export function failureSummary(result) {
  if (!result || typeof result !== 'object') return '';
  const current = text(result.failure?.ego_cause);
  if (current) return current;
  return text(result.failure_analysis?.failure_reason);
}

function failureGroup(result) {
  const failure = result.failure;
  if (failure && typeof failure === 'object') {
    return {
      title: '失败判定',
      rows: [
        ['判定', named(OUTCOME, failure.outcome)],
        ['交互类型', named(INTERACTION, failure.interaction)],
        ['自车原因', text(failure.ego_cause)],
      ],
    };
  }
  const legacy = result.failure_analysis;
  if (!legacy || typeof legacy !== 'object') return null;
  const types = Array.isArray(legacy.failure_type) ? legacy.failure_type.join('、') : legacy.failure_type;
  return {
    title: '失败判定（旧格式）',
    rows: [['失败类型', text(types)], ['失败原因', text(legacy.failure_reason)]],
  };
}

function errorGroup(error) {
  if (!error || typeof error !== 'object' || !Object.keys(error).length) return null;
  const notes = Array.isArray(error.notes) ? error.notes.filter(note => text(note)).join('；') : '';
  return {
    title: '失败信息',
    rows: [
      ['错误类型', text(error.type)],
      ['失败阶段', stageLabel(error.stage)],
      ['说明', text(error.message)],
      ['核对备注', notes],
      ['证据目录', text(error.evidence_dir)],
    ],
  };
}

/** Sectioned detail rows for the run view; empty groups are dropped. */
export function failureDetailGroups(result, error) {
  const data = result && typeof result === 'object' ? result : {};
  const groups = [];
  const lead = errorGroup(error);
  if (lead) groups.push(lead);
  const failure = failureGroup(data);
  if (failure) groups.push(failure);

  const attack = data.attack;
  if (attack && typeof attack === 'object') {
    groups.push({
      title: '复刻结构',
      rows: [
        ['地图', named(MAP, attack.map_context)],
        ['自车动作', named(EGO_ACTION, attack.ego_action)],
        ['目标车辆来向', named(OTHER_ORIGIN, attack.other_origin)],
        ['目标车辆动作', named(OTHER_ACTION, attack.other_action)],
      ],
    });
  }

  const environment = data.environment;
  if (environment && typeof environment === 'object') {
    groups.push({
      title: '环境',
      rows: [
        ['时间', text(environment.time)],
        ['天气', named(WEATHER, environment.weather)],
        ['描述', text(environment.extra)],
      ],
    });
  }

  const verification = data.verification;
  if (verification && typeof verification === 'object') {
    const matched = verification.matched === true ? '画面与测量一致'
      : verification.matched === false ? '画面与测量不一致' : '';
    const notes = Array.isArray(verification.notes)
      ? verification.notes.filter(note => text(note)).join('；') : '';
    groups.push({ title: '视觉核对', rows: [['核对结果', matched], ['备注', notes]] });
  }

  const evidence = text(data.evidence_dir) || text(error?.evidence_dir);
  groups.push({
    title: '溯源',
    rows: [
      ['token', text(data.token)],
      ['simulation', text(data.simulation)],
      ['地图', text(data.map_location)],
      ['证据目录', evidence],
    ],
  });

  return groups
    .map(group => ({ ...group, rows: group.rows.filter(([, value]) => value) }))
    .filter(group => group.rows.length);
}