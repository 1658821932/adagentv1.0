import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { scenarioSummary, scenarioDetailGroups, retrievalSummary, retrievalDetailGroups, editPlannerSummary, editPlannerDetailGroups } from './scenario-results.mjs';

const app = await readFile(new URL('./app.js', import.meta.url), 'utf8');
const demo = JSON.parse(await readFile(new URL('./demo/run.json', import.meta.url), 'utf8'));
const scenario = demo.steps.find(step => step.tool === 'scene_instance_generator').output.result;
const retrieval = {
  candidate_id: 'cand_2', mode: 'local_hit_reranked', match_score: .9,
  video_path: '/project/clip.mp4', source_file: '/project/source.mp4',
  source_segment: { start_time: 2, end_time: 5 },
  scene_ref: { scene_id: 'source', scene_token: 'token', matched_camera: 'CAM_R0',
    frame_count: 18, images_complete: true },
};

test('tool02 exposes the whole target, all mechanism fields and every candidate strategy', () => {
  assert.equal(scenarioSummary(scenario), scenario.target_scene);
  const groups = scenarioDetailGroups(scenario);
  assert.equal(groups[0].rows[0][1], scenario.target_scene);
  assert.equal(groups[1].rows.length, 6);
  assert.equal(groups.length, 5);
  scenario.candidates.forEach((candidate, index) => {
    assert.equal(groups[index + 2].title, `候选 ${candidate.candidate_id}`);
    assert.deepEqual(groups[index + 2].rows.map(([key]) => key),
      ['检索条件', '检索词', '攻击车编辑', '自车输入编辑', '自车正确轨迹']);
    assert.equal(groups[index + 2].rows[3][1], candidate.ego_input_edit_strategy ?? '无需编辑');
  });
});


test('tool02 displays the compact retrieval query separately from source-video conditions', () => {
  const result = structuredClone(scenario);
  result.candidates[0].retrieval_query = '白天丁字路口，自车右转，右侧支路清晰';
  const candidate = scenarioDetailGroups(result)[2];
  assert.deepEqual(candidate.rows.map(([key]) => key),
    ['检索条件', '检索词', '攻击车编辑', '自车输入编辑', '自车正确轨迹']);
  assert.equal(candidate.rows[1][1], result.candidates[0].retrieval_query);
});

test('tool02 does not read the removed output fields or candidate aliases', () => {
  const old = { scene_requirement: 'obsolete', target_effect: 'obsolete',
    candidates: [{ id: 'old', edit_strategy: 'obsolete' }] };
  assert.equal(scenarioSummary(old), '');
  assert.deepEqual(scenarioDetailGroups(old), []);
});

test('tool03 shows the selected candidate, finite relevance and bound source scope', () => {
  const input = { requirement: 'the exact selected retrieval sentence' };
  const summary = retrievalSummary(retrieval, input);
  assert.match(summary, /候选 cand_2/);
  assert.match(summary, /相关分数 0\.900/);
  assert.ok(summary.includes(input.requirement));
  const groups = retrievalDetailGroups(retrieval, input);
  assert.equal(groups[0].rows.find(([key]) => key === '检索条件')[1], input.requirement);
  assert.equal(groups[0].rows.find(([key]) => key === '索引命中区间')[1], '2–5 秒');
  assert.equal(groups[1].rows.find(([key]) => key === '命中相机')[1], 'CAM_R0');
  assert.match(groups[1].rows.find(([key]) => key === '使用范围')[1], /完整源场景/);
  assert.doesNotMatch(retrievalSummary({ ...retrieval, match_score: NaN }, input), /NaN/);
  assert.match(retrievalSummary({ ...retrieval, match_score: -.1 }, input), /-0\.100/);
});

test('mock and failures are explicit instead of claiming a real scene', () => {
  const mock = { candidate_id: 'cand_1', mode: 'mock', match_score: 1, scene_ref: null };
  assert.match(retrievalSummary(mock, {}), /模拟检索/);
  assert.equal(retrievalDetailGroups(mock, {})[1].rows[0][1], '未绑定真实场景');
  for (const groups of [scenarioDetailGroups(null, { type: 'bad_input', message: 'missing source' }),
    retrievalDetailGroups(null, {}, { type: 'bad_input', message: 'missing source' })]) {
    assert.equal(groups[0].title, '执行失败');
    assert.equal(groups[0].rows[1][1], 'missing source');
  }
});

function rendered(step) {
  const children = [];
  const body = { append(element) { children.push(element); } };
  const card = { querySelector(selector) { return selector === '.step-body' ? body : null; } };
  const context = vm.createContext({
    scenarioSummary, scenarioDetailGroups, retrievalSummary, retrievalDetailGroups, editPlannerSummary, editPlannerDetailGroups,
    escape: value => String(value ?? '').replace(/[&<>"']/g, c =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])),
    document: { createElement() { return { innerHTML: '', className: '', querySelectorAll() { return []; } }; } },
    stepStateHtml: () => '', stepTimes: () => '', evidenceNote: () => '',
  });
  const summary = app.slice(app.indexOf('function firstSentence('), app.indexOf('const primaryOutput ='));
  const fill = app.slice(app.indexOf('function fillToolResult('), app.indexOf('/* ---------- run 渲染与轮询'));
  vm.runInContext(summary + '\nconst primaryOutput = summaryFor;\n' + fill, context);
  context.fillToolResult(card, step);
  return children;
}

test('the actual app renderer displays current tool02/tool03 fields and escapes values', () => {
  const result = structuredClone(scenario);
  result.candidates[0].base_video_requirement = '<img src=x onerror=alert(1)>';
  const scenarioNodes = rendered({ tool: 'scene_instance_generator', output: { result } });
  const detail = scenarioNodes.find(node => node.className === 'detail-groups').innerHTML;
  assert.ok(detail.includes(scenario.target_scene));
  for (const candidate of scenario.candidates) assert.ok(detail.includes(`候选 ${candidate.candidate_id}`));
  assert.match(detail, /自车输入编辑/);
  assert.match(detail, /无需编辑/);
  assert.match(detail, /&lt;img src=x/);
  assert.doesNotMatch(detail, /<img src=x/);
  const retrievalNodes = rendered({ tool: 'video_retriever', input: { requirement: 'selected query' },
    output: { result: retrieval } });
  assert.match(retrievalNodes.find(node => node.className === 'summary-block').innerHTML, /cand_2/);
  assert.match(retrievalNodes.find(node => node.className === 'detail-groups').innerHTML, /selected query/);
});

test('the actual app renderer shows structured failure messages', () => {
  for (const tool of ['scene_instance_generator', 'video_retriever']) {
    const nodes = rendered({ tool, output: { status: 'error', error: { type: 'input_validation_error', message: 'missing field' } } });
    assert.match(nodes.find(node => node.className === 'detail-groups').innerHTML, /missing field/);
  }
});


test('tool04 shows a validated attack_input handoff for tool05', () => {
  const result = {
    suitability: 'ready',
    checks: { road: 'pass', lanes: 'pass', space: 'pass' },
    adapted_plan: { attack_edit_strategy: '编辑现有车辆', ego_input_edit_strategy: '调整速度',
      ego_correct_trajectory_strategy: '安全避让' },
    attack_input: { operation: 'auto', plan_options: [{}, {}] },
    evidence: {},
  };
  const plan = editPlannerDetailGroups(result).find(group => group.title === '适配方案');
  assert.equal(plan.rows.find(([key]) => key === '工具⑤交接')[1], '自动选择车辆 · 2 个方案');
});

test('tool04 summary and details use checks and adapted_plan, never removed edit-plan fields', () => {
  const result = {
    suitability: 'needs_information', candidate_id: 'cand_1',
    reasons: ['目标路口已固定，空间证据不足。'], retrieval_suggestion: '补充目标路口侧向画面。',
    checks: { road: 'pass', lanes: 'unknown', space: 'unknown' },
    adapted_plan: { attack_edit_strategy: '调整右侧支路车辆时机',
      ego_input_edit_strategy: null, ego_correct_trajectory_strategy: '自车减速让行后通过。' },
    evidence: { verification_source: 'visual_review',
      map_facts: { target_junction_fid: 45326, target_entry_frame_index: 0,
        target_entry_distance_m: 0, target_junction: { physical_branches: 4 } } },
    edit_plan: { operation: 'edit_existing', target_track: 'obsolete', instruction: 'obsolete' },
  };
  assert.equal(editPlannerSummary(result),
    '信息不足 — 道路通过 · 车道未知 · 空间未知 — 调整右侧支路车辆时机');
  const groups = editPlannerDetailGroups(result);
  assert.equal(groups[0].title, '结构检查');
  assert.equal(groups[0].rows.find(([key]) => key === '目标路口')[1], '45326');
  assert.equal(groups[0].rows.find(([key]) => key === '物理支路')[1], '4');
  assert.equal(groups[1].rows.find(([key]) => key === '自车输入编辑')[1], '无需编辑');
  assert.equal(groups[2].rows.find(([key]) => key === '来源')[1], 'visual_review');
  assert.equal(editPlannerSummary({ edit_plan: result.edit_plan }), '');
});
