import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';
import {
  summaryFor,
  renderStepStructured,
  renderStepMetrics,
  renderStepDeliverables,
  renderRunDeliverables,
} from './step-output.js';
import { previewMedia } from './run-media.mjs';

const frontendDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = dirname(frontendDir);
const demoRun = JSON.parse(await readFile(join(frontendDir, 'demo', 'run.json'), 'utf8'));

test('summaryFor provides structured conclusions for all 9 tools and avoids [object Object]', () => {
  const byTool = Object.fromEntries(demoRun.steps.map(s => [s.tool, s]));

  assert.match(summaryFor(byTool.failure_analyzer), /跟车未减速\/追尾.*自车在接近前方近乎静止车辆/);
  assert.match(summaryFor(byTool.scene_instance_generator), /3 个候选方案/);
  assert.match(summaryFor(byTool.video_retriever), /匹配度 0\.810.*场景 54c166c4ad5c5ad9.*片段 0s–11\.5s/);
  assert.match(summaryFor(byTool.edit_planner), /可编辑.*新增车辆.*静止阻挡.*不要写死碰撞/);

  const attackSummary = summaryFor(byTool.attack_vehicle_editor);
  assert.notEqual(attackSummary, 'completed');
  assert.match(attackSummary, /已完成攻击场景冻结.*synthetic_vehicle_001.*seed_000.*已导出仿真包/);

  const egoSummary = summaryFor(byTool.ego_trajectory_editor);
  assert.notEqual(egoSummary, 'completed');
  assert.match(egoSummary, /已完成自车专家轨迹.*foci_00.*来源 FOCI.*已导出仿真包/);

  const policySummary = summaryFor(byTool.policy_runner);
  assert.doesNotMatch(policySummary, /\[object Object\]/);
  assert.match(policySummary, /目标碰撞已发生（t=6\.5s）.*最小间距 0m.*评测窗口 1\.5s–7\.5s（16帧）.*已导出候选奖励/);

  assert.match(summaryFor(byTool.failure_case_filter), /命中零分失败案例.*共评测 1 个场景（失败 1 个）.*PDM 总分 0/);

  const trainerSummary = summaryFor(byTool.policy_trainer);
  assert.match(trainerSummary, /RLFT 微调完成.*迭代 1 轮.*Loss 0\.4126 → 0\.2841.*参数已更新/);
});

test('renderStepStructured renders clean single summary block without removed widgets', () => {
  const byTool = Object.fromEntries(demoRun.steps.map(s => [s.tool, s]));

  const s01 = renderStepStructured(byTool.failure_analyzer);
  assert.match(s01, /class="summary-block"/);
  assert.match(s01, /class="step-summary"/);
  assert.match(s01, /summary-list/);
  assert.match(s01, /自车速度从 0\.74 m\/s/);
  assert.doesNotMatch(s01, /step-badges|step-metrics-wrap|checks-matrix|timing-breakdown|deliverable-box/);

  const s02 = renderStepStructured(byTool.scene_instance_generator);
  assert.match(s02, /summary-list/);
  assert.match(s02, /候选 cand_01.*same_lane_stationary/);

  const s03 = renderStepStructured(byTool.video_retriever);
  assert.match(s03, /class="summary-block"/);
  assert.doesNotMatch(s03, /timing-breakdown|耗时分解/);

  const s04Blocked = renderStepStructured({
    tool: 'edit_planner',
    output: {
      result: {
        suitability: 'unsuitable',
        reasons: ['前方没有合适遮挡物'],
        retrieval_suggestion: '检索城市双车道有静止卡车的场景',
      },
    },
  });
  assert.match(s04Blocked, /不适合编辑/);
  assert.match(s04Blocked, /重检索建议：检索城市双车道有静止卡车的场景/);

  const s05 = renderStepStructured(byTool.attack_vehicle_editor);
  assert.match(s05, /已完成攻击场景冻结/);
  assert.doesNotMatch(s05, /checks-matrix|几何与安全检查/);

  const s06 = renderStepStructured(byTool.ego_trajectory_editor);
  assert.match(s06, /已完成自车专家轨迹/);
  assert.doesNotMatch(s06, /checks-matrix|几何与安全检查/);

  const s07 = renderStepStructured(byTool.policy_runner);
  assert.match(s07, /目标碰撞已发生（t=6\.5s）/);
  assert.doesNotMatch(s07, /diag-grid|step-metrics-wrap/);

  const s08 = renderStepStructured(byTool.failure_case_filter);
  assert.match(s08, /命中零分失败案例/);
  assert.doesNotMatch(s08, /structured-table/);

  const s09 = renderStepStructured(byTool.policy_trainer);
  assert.match(s09, /Loss 0\.4126 → 0\.2841/);
  assert.doesNotMatch(s09, /loss-trace|step-metrics-wrap/);

  const sErr = renderStepStructured({
    tool: 'policy_runner',
    output: { error: { type: 'RuntimeError', message: 'WorldEngine crashed' } },
  });
  assert.match(sErr, /class="structured-error"/);
  assert.match(sErr, /RuntimeError: WorldEngine crashed/);
});

test('edit planner displays all unranked options with requirement, goal and traceable identities', () => {
  const step = {
    tool: 'edit_planner',
    input: { scene_requirement: '城市道路前车减速。' },
    output: { result: {
      suitability: 'ready', target_effect: '自车及时减速避让。', edit_plan: null,
      edit_options: [
        { option_id: 'existing', operation: 'edit_existing', target_track: 'lead-123',
          instruction: '让前车从原位置缓慢制动。', rationale: '前车处于跟车路径上。', uncertainty: '检查初始转向状态。' },
        { option_id: 'added', operation: 'add_vehicle', maneuver: 'stationary',
          instruction: '在前方合适位置添加静止车辆。' },
      ],
    } },
  };
  assert.match(summaryFor(step), /可编辑.*城市道路前车减速.*目标：自车及时减速避让.*2 个候选方案（不分排名）/);
  const html = renderStepStructured(step);
  assert.match(html, /class="step-summary"/);
  assert.match(html, /<ul class="summary-list">/);
  assert.equal((html.match(/<li>/g) || []).length, 2);
  assert.match(html, /<b>候选方案 01<\/b>（existing）.*修改现有车辆.*目标车辆 lead-123.*让前车从原位置缓慢制动/);
  assert.match(html, /<b>候选方案 02<\/b>（added）.*新增车辆.*静止阻挡.*在前方合适位置添加静止车辆/);
  assert.match(html, /理由：前车处于跟车路径上。；待核实：检查初始转向状态。/);
  assert.ok(html.indexOf('（existing）') < html.indexOf('（added）'));
});

test('edit planner keeps pending options visible and escapes all option text', () => {
  const step = { tool: 'edit_planner', output: { result: {
    suitability: 'needs_information', reasons: ['缺少新增车辆素材。'],
    edit_options: [null, {
      option_id: '<img src=x>', operation: 'add_vehicle', maneuver: 'stationary',
      instruction: '<script>alert(1)</script>', rationale: '<b>理由</b>', uncertainty: '检查 A&B',
    }],
  } } };
  const html = renderStepStructured(step);
  assert.match(html, /信息不足.*缺少新增车辆素材.*1 个候选方案/);
  assert.match(html, /&lt;img src=x&gt;/);
  assert.match(html, /&lt;script&gt;alert\(1\)&lt;\/script&gt;/);
  assert.match(html, /&lt;b&gt;理由&lt;\/b&gt;/);
  assert.match(html, /检查 A&amp;B/);
  assert.doesNotMatch(html, /<img|<script|\[object Object\]/);
  assert.equal((html.match(/<li>/g) || []).length, 1);
  assert.doesNotMatch(renderStepStructured({tool: 'edit_planner', output: {result: {
    suitability: 'ready', edit_options: [], reasons: ['无需编辑他车。'],
  }}}), /summary-list|候选方案/);
});

test('removed metrics and deliverables renderers return empty strings', () => {
  const byTool = Object.fromEntries(demoRun.steps.map(s => [s.tool, s]));
  const fileUrl = p => '/api/file?path=' + encodeURIComponent(p);

  assert.equal(renderStepMetrics(byTool.failure_analyzer), '');
  assert.equal(renderStepMetrics(byTool.policy_trainer), '');
  assert.equal(renderStepDeliverables(byTool.policy_trainer, { fileUrl }), '');
  assert.equal(renderStepDeliverables(byTool.ego_trajectory_editor, { fileUrl }), '');
  assert.equal(renderRunDeliverables({ ...demoRun, report_path: '/tmp/run_x/REPORT.md' }, { fileUrl }), '');
  assert.equal(renderRunDeliverables({ status: 'error', error: 'boom' }), '');
});

test('previewMedia includes attack_vehicle_editor and extra ego_trajectory_editor rendered_videos', async () => {
  const root = await mkdtemp(join(tmpdir(), 'adagent-render-media-'));
  try {
    const attackVid = join(root, 'attack_cam.mp4');
    const egoManual = join(root, 'manual.mp4');
    const egoCam = join(root, 'cam_f0.mp4');
    await writeFile(attackVid, '');
    await writeFile(egoManual, '');
    await writeFile(egoCam, '');

    const options = { projectRoot: root, weRoot: '/we', closedLoopRoot: join(root, 'closed') };
    const attackMedia = previewMedia('attack_vehicle_editor', {
      result: { rendered_videos: { CAM_F0: attackVid } },
    }, options);
    assert.equal(attackMedia.length, 1);
    assert.match(attackMedia[0].title, /攻击场景渲染 · CAM_F0/);
    assert.equal(attackMedia[0].primary, true);

    const egoMedia = previewMedia('ego_trajectory_editor', {
      result: { rendered_videos: { manual_explained: egoManual, CAM_F0: egoCam } },
    }, options);
    assert.equal(egoMedia.length, 2);
    assert.equal(egoMedia[0].title, '专家轨迹渲染回放');
    assert.equal(egoMedia[0].primary, true);
    assert.equal(egoMedia[1].title, '专家轨迹渲染 · CAM_F0');
    assert.equal(egoMedia[1].primary, false);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('server.mjs getRunDetail extracts pth/pkl/log artifacts, report_path, and summary.json stages', { timeout: 15000 }, async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'adagent-server-deliverables-'));
  const runId = 'run_20261115_120000_000001';
  const runDir = join(root, 'logs', runId);
  const stepDir = join(runDir, 'tools', 'step_00_policy_trainer');
  await mkdir(stepDir, { recursive: true });
  await mkdir(join(root, 'configs'), { recursive: true });
  await writeFile(join(root, 'configs', 'api.json'), JSON.stringify({ active_profile: 'mock' }));
  await writeFile(join(root, 'configs', 'agent.json'), JSON.stringify({ agent: { max_steps: 5 }, tools: { modes: {} } }));

  const ckptPath = join(root, 'outputs', 'iter_1.pth');
  const logPath = join(root, 'outputs', 'train.log');
  await mkdir(join(root, 'outputs'), { recursive: true });
  await writeFile(ckptPath, '');
  await writeFile(logPath, '');

  await writeFile(join(runDir, 'run.json'), JSON.stringify({
    run_id: runId,
    task: 'test deliverables',
    status: 'finished',
    start_time: '2026-11-15T10:00:00Z',
    end_time: '2026-11-15T10:01:00Z',
  }));
  await writeFile(join(runDir, 'REPORT.md'), '# Report\n');
  await writeFile(join(runDir, 'summary.json'), JSON.stringify({
    api_calls: 3,
    api_seconds: 5.2,
    wall_seconds: 60.0,
    stages: [{ stage: 'main_agent', retry_requests: 2 }],
  }));
  await writeFile(join(stepDir, 'input.json'), JSON.stringify({ training_mode: 'rlft' }));
  await writeFile(join(stepDir, 'output.json'), JSON.stringify({
    status: 'success',
    tool: 'policy_trainer',
    result: {
      checkpoint: ckptPath,
      log_path: logPath,
      training_mode: 'rlft',
      training_audit: { final_loss: 0.28, iterations: 1 },
    },
  }));

  const server = spawn(process.execPath, [join(frontendDir, 'server.mjs')], {
    cwd: frontendDir,
    env: {
      ...process.env,
      PORT: '0',
      HOST: '127.0.0.1',
      ADAGENT_PROJECT_ROOT: root,
      ADAGENT_AGENT_PYTHON: join(projectRoot, 'environments/agent/.venv/bin/python'),
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let output = '';
  server.stdout.on('data', d => { output += d; });
  server.stderr.on('data', d => { output += d; });
  t.after(async () => {
    server.kill('SIGTERM');
    await new Promise(r => (server.exitCode !== null ? r() : server.once('exit', r)));
    await rm(root, { recursive: true, force: true });
  });

  for (let i = 0; i < 100 && !/http:\/\/127\.0\.0\.1:\d+/.test(output); i += 1) {
    await delay(20);
  }
  const base = output.match(/http:\/\/127\.0\.0\.1:\d+/)?.[0];
  assert.ok(base, output);

  const detail = await (await fetch(`${base}/api/runs/${runId}`)).json();
  assert.equal(detail.report_path, join(runDir, 'REPORT.md'));
  assert.equal(detail.summary.api_calls, 3);
  assert.equal(detail.summary.stages[0].retry_requests, 2);
  const paths = detail.steps[0].artifacts.map(a => a.path);
  assert.ok(paths.includes(ckptPath), 'checkpoint .pth must be collected');
  assert.ok(paths.includes(logPath), 'training .log must be collected');

  const html = await (await fetch(`${base}/`)).text();
  assert.match(html, /app\.js\?v=/);
});

test('app.js places always-open step-reason above step-main-output and omits removed widgets', async () => {
  const appSource = await readFile(join(frontendDir, 'app.js'), 'utf8');
  const reasonIdx = appSource.indexOf('class="step-reason"');
  const mainIdx = appSource.indexOf('class="step-main-output"');
  const drawersIdx = appSource.indexOf('class="step-drawers"');
  assert.ok(reasonIdx > 0 && mainIdx > reasonIdx && drawersIdx > mainIdx,
    'step-reason must be directly below step-head and above step-main-output');
  assert.doesNotMatch(appSource, /<details class="step-reason"|reasonEl\.open\s*=/);
  assert.match(appSource, /renderStepStructured\(s\)/);
  assert.doesNotMatch(appSource, /renderStepMetrics|renderStepDeliverables|renderRunDeliverables|renderTimings/);
});
