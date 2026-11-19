/** Tool 01 result wording and detail rows used by the run view. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { failureDetailGroups, failureSummary, stageLabel } from './tool01-result.mjs';

const CURRENT = {
  token: '2021.09.16.19.49.00_veh-42_00990_01609-4abb45d9ee1a5c5c',
  simulation: 'e2e_vadv2_50pct_fullR_bev/navtest_failures_R_complete',
  map_location: 'us-pa-pittsburgh-hazelwood',
  evidence_dir: '/project/logs/tool01/sim-a/scene/run-1',
  violation_step: 8,
  failure: { outcome: 'collision', interaction: 'crossing',
    ego_cause: '在丁字路口右转时，没有对从右侧支路左转过来的车辆让行' },
  attack: { map_context: 't_junction', ego_action: 'turn_right', ego_lane_keeping: 'keep',
    other_origin: 'right_arm', other_action: 'turn_left', other_lane_keeping: 'keep' },
  environment: { time: '2021-09-16T20:16:41+00:00', weather: 'cloudy', extra: '傍晚，路面干燥' },
  verification: { sample_frames: [6, 8, 10], corrections: [], notes: [] },
};

function rowsOf(groups, title) {
  const group = groups.find(entry => entry.title === title);
  assert.ok(group, `missing group ${title}`);
  return Object.fromEntries(group.rows);
}

test('summary prefers the current ego_cause', () => {
  assert.equal(failureSummary(CURRENT), CURRENT.failure.ego_cause);
});

test('summary still reads historical failure_analysis runs', () => {
  const legacy = { failure_analysis: { failure_type: ['foreground'], failure_reason: 'insufficient braking response' } };
  assert.equal(failureSummary(legacy), 'insufficient braking response');
});

test('summary ignores malformed results', () => {
  assert.equal(failureSummary(null), '');
  assert.equal(failureSummary({}), '');
  assert.equal(failureSummary({ failure: { ego_cause: 7 } }), '');
});

test('detail rows expose the measured fields with readable labels', () => {
  const groups = failureDetailGroups(CURRENT, null);
  assert.deepEqual(rowsOf(groups, '失败判定'), {
    判定: '碰撞', 首次违规帧: '8', 交互类型: '交叉冲突', 自车原因: CURRENT.failure.ego_cause,
  });
  assert.deepEqual(rowsOf(groups, '复刻结构'), {
    地图: '丁字路口', 自车动作: '右转', 自车车道保持: '保持',
    目标车辆来向: '右侧支路', 目标车辆动作: '左转', 目标车辆车道保持: '保持',
  });
  assert.deepEqual(rowsOf(groups, '环境'), {
    时间: '2021-09-16T20:16:41+00:00', 天气: '多云', 描述: '傍晚，路面干燥',
  });
  assert.deepEqual(rowsOf(groups, '视觉核对'), { 核对帧: '6、8、10', 修正项: '无' });
  assert.deepEqual(rowsOf(groups, '溯源'), {
    token: CURRENT.token, simulation: CURRENT.simulation,
    地图: 'us-pa-pittsburgh-hazelwood', 证据目录: CURRENT.evidence_dir,
  });
});

test('a nearby vehicle without an official collision is shown as a near miss', () => {
  const result = {
    ...CURRENT,
    violation_step: 6,
    failure: {
      outcome: 'road_departure', collision_confirmed: false, interaction: 'near_miss',
      ego_cause: '在丁字路口直行并保持在本车道内时驶出可行驶道路，首次违规时刻5米内最近车辆位于右相邻车道并直行',
    },
    attack: {
      ...CURRENT.attack,
      other_origin: 'right_adjacent', other_lane_keeping: 'unknown',
    },
  };
  assert.deepEqual(rowsOf(failureDetailGroups(result, null), '失败判定'), {
    判定: '出界',
    碰撞确认: '否',
    首次违规帧: '6',
    交互类型: '近距冲突',
    自车原因: result.failure.ego_cause,
  });
  assert.equal(rowsOf(failureDetailGroups(result, null), '复刻结构').目标车辆来向, '右相邻车道');
});

test('corrected verification keeps its count and notes', () => {
  const result = { ...CURRENT, verification: {
    sample_frames: [6, 8, 10],
    corrections: [{ field: 'ego_action', measured: 'turn_right', final: 'straight' }],
    notes: ['画面里自车在直行'],
  } };
  assert.deepEqual(rowsOf(failureDetailGroups(result, null), '视觉核对'), {
    核对帧: '6、8、10', 修正项: '1 项', 备注: '画面里自车在直行',
  });
});

test('failed runs show the error stage and their evidence directory', () => {
  const error = { type: 'timeout', stage: 'vlm', message: 'analysis exceeded its 30s budget',
    evidence_dir: '/project/logs/tool01/sim-a/scene/run-1' };
  const groups = failureDetailGroups({}, error);
  assert.equal(groups[0].title, '失败信息');
  assert.deepEqual(Object.fromEntries(groups[0].rows), {
    错误类型: 'timeout', 失败阶段: '模型核对',
    说明: 'analysis exceeded its 30s budget',
    证据目录: '/project/logs/tool01/sim-a/scene/run-1',
  });
  assert.equal(stageLabel('data'), '读取数据');
});

test('historical failure_analysis still renders as its own group', () => {
  const result = { failure_analysis: { failure_type: ['foreground', 'occlusion'], failure_reason: 'reason' } };
  const groups = failureDetailGroups(result, null);
  assert.equal(groups[0].title, '失败判定（旧格式）');
  assert.deepEqual(Object.fromEntries(groups[0].rows), {
    失败类型: 'foreground、occlusion', 失败原因: 'reason',
  });
});

test('empty payloads produce no detail block', () => {
  assert.deepEqual(failureDetailGroups(null, null), []);
  assert.deepEqual(failureDetailGroups({}, {}), []);
});
