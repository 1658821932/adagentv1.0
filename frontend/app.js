import { toolNames, stageMap } from './agent-ui.js';
import { resetWorkbench, setWorkbenchState, updateCall } from './workbench.js';
import { createApiClient } from './api-client.js';
import { createConfigPanelController } from './config-panel-controller.mjs';
import { panelLoadingHtml, panelErrorHtml, readPanelData } from './panel-loading.mjs';
import { failureDetailGroups, failureSummary, stageLabel } from './tool01-result.mjs';
import { scenarioSummary, scenarioDetailGroups, retrievalSummary, retrievalDetailGroups, editPlannerSummary, editPlannerDetailGroups } from './scenario-results.mjs';
import { submitConversation } from './conversation.js';
window.addEventListener('error', (e) => {
  const box = document.createElement('div');
  box.className = 'js-error';
  box.textContent = '前端脚本错误：' + (e.message || String(e.error || ''));
  document.body.append(box);
});

const messages = document.querySelector('#messages');
const stopBtn = document.querySelector('#stop');
const replayBtn = document.querySelector('#replay-btn');
const prompt = document.querySelector('#prompt');
const send = document.querySelector('#send');
const status = document.querySelector('#status');
const historyMenu = document.querySelector('#history');
const historyBtn = document.querySelector('#history-btn');
const historyPanel = document.querySelector('#history-panel');
const historyCount = document.querySelector('#history-count');
const historyFeedback = document.querySelector('#history-feedback');
const historyItems = document.querySelector('#history-items');
const modelLabel = document.querySelector('#conversation-model');
const userMark = `<svg aria-hidden="true" focusable="false" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8.6" r="3.3"/><path d="M5.6 19.4c0-3.3 2.9-5.6 6.4-5.6s6.4 2.3 6.4 5.6"/></svg>`;
const agentMark = `<svg aria-hidden="true" focusable="false" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M19.5 8a8.4 8.4 0 0 0-14.7-1.2M4.5 16a8.4 8.4 0 0 0 14.7 1.2"/><path d="m16.3 7.2 3.5 1.1.7-3.6M7.7 16.8l-3.5-1.1-.7 3.6"/><path d="m8 10 1.5-3h5L16 10M7.5 10h9v5h-9zM9 15v1M15 15v1M10 12.5h4"/></svg>`;
const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const { request: api, serviceUrl } = createApiClient(import.meta.url);
const fileUrl = path => serviceUrl('/api/file?path=' + encodeURIComponent(path));

/* ---------- API ---------- */
/* ---------- 状态 ---------- */
let viewingRun = null;        // 当前查看的 run_id
let pollTimer = null;
let rendered = new Map();     // step -> {decisionEl, cardEl, state}
let streamEl = null, progressEl = null, phaseEl = null, activityEl = null, answerEl = null, subagentEl = null;
let replayToken = 0;         // 每次重放自增，用于中止旧任务
let replayPaused = false;
let replaySpeed = 4;         // 4x 默认，录屏节奏比较合适
let running = false;

/* ---------- 基础渲染（结构同原版） ---------- */
function scroll(){ messages.scrollTop = messages.scrollHeight; }
function message(role, html){
  const el = document.createElement('article');
  el.className = 'message ' + role;
  el.innerHTML = `<div class="avatar">${role === 'user' ? userMark : agentMark}</div><div class="message-body"><div class="author">${role === 'user' ? '你' : '智能体'}</div>${html}</div>`;
  messages.append(el); scroll(); return el;
}
function setRunning(value){
  running = value; send.disabled = value; prompt.disabled = value;
  stopBtn.classList.toggle('hidden', !value);
  status.textContent = value ? 'Agent 正在真实执行…' : '';
}

/* ---------- 步骤渲染 ---------- */
function stageOf(tool){ return (stageMap[tool] || ['执行中'])[0]; }
function nameOf(tool){ return toolNames[tool] || tool; }

function fmtTokens(n){
  if (typeof n !== 'number' || !n) return '0';
  return n >= 10000 ? (n / 1000).toFixed(1) + 'k' : String(n);
}

// 主 Agent 单次决策的 LLM 开销
function renderLlmLine(llm, inline){
  if (!llm || typeof llm.elapsed_seconds !== 'number') return '';
  const text = `思考 ${llm.elapsed_seconds.toFixed(2)}s`;
  return inline ? escape(text) : `<div class="decision-llm"><span>${escape(text)}</span></div>`;
}

// 用上一步的真实结局描述当前思路，而不是套模板
function stepOutcome(prev){
  if (!prev) return '';
  const name = nameOf(prev.tool);
  if (prev.status === 'running') return `上一步 ${name} 仍在执行中`;
  if (prev.status === 'error' || prev.status === 'failed') {
    const err = prev.output?.error;
    const raw = typeof err === 'string' ? err : (err?.message || err?.type || '');
    const why = String(raw).split(';')[0].slice(0, 60);
    return `上一步 ${name} 执行失败${why ? `（${why}）` : ''}`;
  }
  return `上一步 ${name} 已返回结果`;
}

function stepNumber(n){ return String(n + 1).padStart(2, '0'); }

// 一步一张卡：决策 + 工具执行合并到同一头部，减少界面元素
function renderStepCard(s, prevStep){
  const card = document.createElement('section');
  card.className = 'step-card';
  const reason = s.decision?.reason || '';
  card.innerHTML = `<div class="step-head">`
    + `<span class="step-state">${stepStateHtml(s)}</span>`
    + `<b class="step-name">${escape(nameOf(s.tool))}</b>`
    + `<code class="step-code" title="${escape(s.tool)}">${escape(s.tool)}</code>`
    + `<span class="step-times">${stepTimes(s)}</span></div>`
    + (reason ? `<details class="step-reason" open><summary>调用原因</summary><p>${escape(reason)}</p></details>` : '')
    + `<div class="step-body"></div>`;
  return card;
}

function stepTimes(s){
  const bits = [];
  if (s.llm && typeof s.llm.elapsed_seconds === 'number') bits.push('思考 ' + s.llm.elapsed_seconds.toFixed(1) + 's');
  if (typeof s.duration === 'number') {
    bits.push('执行 ' + fmtDuration(s.duration));
  } else if (s.status === 'running') {
    const base = typeof s.elapsed_seconds === 'number' ? s.elapsed_seconds : 0;
    bits.push(`执行 <span class="step-time-live" data-base="${base}" data-stamp="${Date.now()}">${fmtDuration(base)}</span>`);
  } else if (s.status === 'error') {
    bits.push('执行实际耗时未知');
  }
  return bits.join(' · ');
}

function stepStateHtml(s){
  if (s.status === 'running') return '<span class="thinking-spinner"></span>执行中';
  if (s.status === 'success' || s.status === 'done') return '✓';
  return '✕';
}


function firstSentence(text, limit = 90){
  if (typeof text !== 'string') return '';
  const cut = text.split(/[。；\n]/)[0].trim();
  return cut.length > limit ? cut.slice(0, limit) + '…' : cut;
}

// 每个工具产出的一句话结论
function summaryFor(s){
  const r = s.output?.result;
  if (!r || typeof r !== 'object') return '';

  if (s.tool === 'failure_analyzer') {
    return firstSentence(failureSummary(r)) || '';
  }
  if (s.tool === 'scene_instance_generator') {
    return firstSentence(scenarioSummary(r), 140);
  }
  if (s.tool === 'video_retriever') {
    return firstSentence(retrievalSummary(r, s.input), 220);
  }
  if (s.tool === 'edit_planner') {
    return firstSentence(editPlannerSummary(r), 220) || '';
  }
  if (s.tool === 'attack_vehicle_editor' || s.tool === 'ego_trajectory_editor') {
    const status = r.status === 'unable_to_complete' ? '未能完成' : r.status === 'defense_not_found' ? '未找到可行防御' : r.status || '';
    return [status, firstSentence(r.reason || r.stop_reason)].filter(Boolean).join(' — ');
  }
  if (s.tool === 'policy_runner') {
    const c = r.collision_report || {};
    const bits = [];
    if (typeof c.target_collision_detected === 'boolean') bits.push(c.target_collision_detected ? '目标碰撞已发生' : '未发生目标碰撞');
    if (r.evaluation_scope) bits.push(String(r.evaluation_scope).slice(0, 40));
    return bits.join(' · ');
  }
  if (s.tool === 'failure_case_filter') {
    return r.is_failure_case === true ? `失败案例 · ${r.scene_count || 1} 个场景` : '非失败案例';
  }
  for (const path of ['summary', 'message', 'requirement']) {
    const v = path.split('.').reduce((o, k) => (o && typeof o === 'object' ? o[k] : undefined), r);
    if (typeof v === 'string' && v) return firstSentence(v, 140);
  }
  return '';
}

const primaryOutput = summaryFor;

// 工具①证据摘要：本次 run 目录、终态与截图数量
function evidenceNote(summary){
  if (!summary || typeof summary !== 'object') return '';
  const bits = [];
  const manifest = summary.manifest;
  if (manifest) {
    if (manifest.run_id) bits.push(`证据 ${manifest.run_id}`);
    if (manifest.status) bits.push(`${manifest.status}${manifest.stage ? ' · ' + stageLabel(manifest.stage) : ''}`);
  }
  if (Number.isFinite(summary.images) && summary.images > 0) bits.push(`${summary.images} 张截图`);
  if (summary.error_type) {
    bits.push(`错误 ${summary.error_type}${summary.stage ? ' · ' + stageLabel(summary.stage) : ''}`);
  }
  if (!bits.length && summary.evidence_dir) bits.push(summary.evidence_dir);
  return bits.filter(Boolean).join(' · ');
}

// 官方评分字段的中文标签与展示优先级
const METRIC_LABELS = {
  score: '总分',
  first_violation_step: '首次违规步',
  no_at_fault_collisions: '碰撞',
  drivable_area_compliance: '道路合规',
  ego_progress: '行驶进度',
  time_to_collision_within_bound: 'TTC 安全',
  comfort: '舒适性',
  driving_direction_compliance: '方向合规',
  step: '评测步数',
};
const METRIC_ORDER = Object.keys(METRIC_LABELS);
function metricLabel(key){ return METRIC_LABELS[key] || key; }
function metricRank(key){
  const i = METRIC_ORDER.indexOf(key);
  return i === -1 ? METRIC_ORDER.length : i;
}

// 耗时分解：把 timings 的每个阶段渲染成带比例条的列表
const TIMING_LABELS = {
  store_init_seconds: '建立向量库连接（含 import chromadb）',
  embedding_seconds: '文本转向量（embedding API）',
  vector_query_seconds: '向量检索（chroma）',
  local_search_seconds: '召回小计',
  rerank_prep_seconds: '候选准备（剪辑 · 转码 · 上传）',
  rerank_api_seconds: '多模态重排序推理',
  rerank_seconds: '重排序小计',
  export_seconds: '导出结果视频',
  total_seconds: '全流程总计',
};
const TIMING_ORDER = ['store_init_seconds', 'embedding_seconds', 'vector_query_seconds', 'local_search_seconds',
  'rerank_prep_seconds', 'rerank_api_seconds', 'rerank_seconds', 'export_seconds', 'total_seconds'];
const TIMING_SUBTOTAL = new Set(['local_search_seconds', 'rerank_seconds', 'total_seconds']);

function renderTimings(t){
  if (!t || typeof t !== 'object') return '';
  const numeric = Object.values(t).filter(v => typeof v === 'number');
  if (!numeric.length) return '';
  const total = Number(t.total_seconds) || Math.max(...numeric);
  if (!(total > 0)) return '';
  const keys = TIMING_ORDER.filter(k => typeof t[k] === 'number' && t[k] >= 0);
  if (!keys.length) return '';
  const rows = keys.map(k => {
    const v = Number(t[k]);
    const pct = Math.max(Math.min(100, (v / total) * 100), v > 0 ? 2 : 0);
    const cls = 'timing-row'
      + (TIMING_SUBTOTAL.has(k) ? ' subtotal' : '')
      + (k === 'total_seconds' ? ' total' : '');
    return `<div class="${cls}"><span class="t-label">${escape(TIMING_LABELS[k] || k)}</span>`
      + `<span class="t-bar"><i style="width:${pct.toFixed(1)}%"></i></span>`
      + `<span class="t-val">${v.toFixed(2)}s</span></div>`;
  }).join('');
  return `<details class="timing-breakdown"><summary><span class="timing-title">耗时分解</span>`
    + `<b class="timing-total">${total.toFixed(2)}s</b></summary>`
    + `<div class="timing-body">${rows}</div></details>`;
}

function fillToolResult(card, s){
  const stateEl = card.querySelector('.step-state');
  if (stateEl) stateEl.innerHTML = stepStateHtml(s);
  const timesEl = card.querySelector('.step-times');
  if (timesEl) timesEl.innerHTML = stepTimes(s);

  const body = card.querySelector('.step-body') || card;
  const text = primaryOutput(s);
  if (text) {
    const wrap = document.createElement('div');
    wrap.className = 'summary-block';
    wrap.innerHTML = `<div class="raw-title">结果简述</div><p class="step-summary">${escape(text)}</p>`;
    body.append(wrap);
  }

  const groups = s.tool === 'failure_analyzer'
    ? failureDetailGroups(s.output?.result, s.output?.error)
    : s.tool === 'scene_instance_generator'
      ? scenarioDetailGroups(s.output?.result, s.output?.error)
      : s.tool === 'edit_planner'
        ? editPlannerDetailGroups(s.output?.result, s.output?.error)
        : s.tool === 'video_retriever'
          ? retrievalDetailGroups(s.output?.result, s.input, s.output?.error) : [];
  if (groups.length) {
    const box = document.createElement('div');
    box.className = 'detail-groups';
    box.innerHTML = groups.map(group => `<div class="detail-group">`
      + `<div class="raw-title">${escape(group.title)}</div>`
      + group.rows.map(([label, value]) =>
        `<div class="detail-row"><span>${escape(label)}</span><b>${escape(value)}</b></div>`).join('')
      + `</div>`).join('');
    body.append(box);
  }

  const raw = [];
  if (s.input != null) raw.push(`<div class="raw-block"><div class="raw-title">输入 JSON</div><pre>${escape(JSON.stringify(s.input, null, 2))}</pre></div>`);
  if (s.output != null) raw.push(`<div class="raw-block"><div class="raw-title">输出 JSON</div><pre>${escape(JSON.stringify(s.output, null, 2))}</pre></div>`);
  if (raw.length) {
    const d = document.createElement('div');
    d.className = 'json-block';
    d.innerHTML = raw.join('');
    body.append(d);
  }

  const evidenceItems = Array.isArray(s.evidence) ? s.evidence : [];
  const note = evidenceNote(s.evidence_summary);
  if (evidenceItems.length || note) {
    const box = document.createElement('div');
    box.className = 'evidence-videos';
    const images = evidenceItems.filter(item => item.kind === 'image');
    const videos = evidenceItems.filter(item => item.kind !== 'image');
    box.innerHTML = (note ? `<div class="evidence-summary">${escape(note)}</div>` : '')
      + (images.length
        ? `<div class="raw-title">本次分析证据</div><div class="evidence-images">`
          + images.map(item => `<figure class="evidence-image">`
            + `<img loading="lazy" alt="${escape(item.title)}" src="${escape(item.src)}">`
            + `<figcaption>${escape([item.title, item.note].filter(Boolean).join(' · '))}</figcaption>`
            + `</figure>`).join('')
          + `</div>`
        : '')
      + videos.map(v =>
        `<div class="raw-title">${escape(v.title)}</div>`
        + `<video controls playsinline preload="auto" width="1920" height="1080" src="${escape(v.src)}"></video>`
      ).join('');
    box.querySelectorAll('video').forEach(vid => {
      vid.addEventListener('play', e => {
        messages.querySelectorAll('video').forEach(o => { if (o !== e.target) o.pause(); });
      });
    });
    body.append(box);
  }
}

/* ---------- run 渲染与轮询 ---------- */
// 右上角：步数 + 总耗时
function setProgressMeta(d, running){
  const el = progressEl && progressEl.querySelector('.agent-count');
  if (!el) return;
  const bits = [];
  if (d.steps.length) bits.push(`${d.steps.length} 步`);
  if (typeof d.duration_seconds === 'number') {
    const s = d.duration_seconds;
    if (running) {
      const scale = arguments[2] || 1;
      bits.push(`已用 <span class="progress-live" data-base="${s}" data-stamp="${Date.now()}" data-scale="${scale}">${fmtDuration(s)}</span>`);
    } else {
      bits.push(`总耗时 ${fmtDuration(s)}`);
    }
  }
  el.innerHTML = bits.join(' · ');
}

function beginRunView(d){
  messages.replaceChildren();
  resetWorkbench();
  message('user', `<p>${escape(d.task || '(无任务描述)').replace(/\n/g, '<br>')}</p>`);
  rendered = new Map();
  mountAgentResponse();
}

function mountAgentResponse(continued = false){
  const note = continued ? '保留以上执行记录' : '准备决策';
  const response = message('assistant', `<div class="agent-progress"><span class="thinking-spinner"></span><b class="agent-phase">任务理解</b><span class="agent-activity">${note}</span><span class="agent-count"></span></div><div class="answer-slot"></div><div class="subagent-activity"></div><div class="tool-stream"></div>`);
  streamEl = response.querySelector('.tool-stream');
  progressEl = response.querySelector('.agent-progress');
  phaseEl = response.querySelector('.agent-phase');
  activityEl = response.querySelector('.agent-activity');
  answerEl = response.querySelector('.answer-slot');
  subagentEl = response.querySelector('.subagent-activity');
}

// 续聊沿用当前对话，只新增一份回答，不清空用户消息和此前的工具记录。
function continueRunView(d){
  rendered = new Map();
  mountAgentResponse(true);
}


/* ---------- 重放：按记录的时间节奏逐步揭示历史步骤 ---------- */

// 可暂停的等待（暂停时不计入耗时）
function replayWait(ms, token) {
  return new Promise((resolve) => {
    let remain = ms;
    let last = Date.now();
    const tick = () => {
      if (token !== replayToken) return resolve();
      const now = Date.now();
      if (!replayPaused) remain -= now - last;
      last = now;
      if (remain <= 0) return resolve();
      setTimeout(tick, 80);
    };
    setTimeout(tick, 80);
  });
}

function replayBarHtml() {
  return `<div class="replay-bar"><span class="replay-dot"></span><span>重放中</span>`
    + `<button type="button" data-act="pause">暂停</button>`
    + `<button type="button" data-act="speed">${replaySpeed}x</button>`
    + `<button type="button" data-act="stop">停止</button></div>`;
}

function mountReplayBar() {
  let bar = messages.querySelector('.replay-bar');
  if (bar) bar.remove();
  bar = document.createElement('div');
  bar.className = 'replay-wrap';
  bar.innerHTML = replayBarHtml();
  const head = messages.querySelector('.conversation-heading');
  if (head && head.parentElement) head.parentElement.insertBefore(bar, head.nextSibling);
  else messages.prepend(bar);
  bar.addEventListener('click', (e) => {
    const act = e.target?.dataset?.act;
    if (!act) return;
    if (act === 'pause') {
      replayPaused = !replayPaused;
      e.target.textContent = replayPaused ? '继续' : '暂停';
    } else if (act === 'speed') {
      replaySpeed = replaySpeed >= 16 ? 1 : replaySpeed * 2;
      e.target.textContent = replaySpeed + 'x';
    } else if (act === 'stop') {
      replayToken += 1;
      unmountReplayBar();
    }
  });
}

function unmountReplayBar() {
  messages.querySelector('.replay-wrap')?.remove();
}


// 重放时的进度：步数 + 从 0 开始累积的记录时间
function setReplayProgress(stepCount, baseSeconds) {
  const el = progressEl && progressEl.querySelector('.agent-count');
  if (!el) return;
  const bits = [];
  if (stepCount) bits.push(`${stepCount} 步`);
  bits.push(`已用 <span class="progress-live" data-base="${baseSeconds}" data-stamp="${Date.now()}" `
    + `data-scale="${replaySpeed}">${fmtDuration(baseSeconds)}</span>`);
  el.innerHTML = bits.join(' · ');
}

// 按记录的真实节奏揭示步骤；speed 为倍速（越大越快）
async function playTimeline(d, token, speed, finalDoc = d) {
  const steps = d.steps || [];
  const scale = 1 / speed;
  let prevStep = null;

  let clock = 0;   // 已重放的“记录时间”（秒），从 0 开始
  const keyOf = (s) => `s${s.step}_${s.tool}`;
  for (const s of steps) {
    if (token !== replayToken) return;
    // 决策阶段
    phaseEl.textContent = s.step === 0 ? '正在理解任务' : '正在整理结果';
    activityEl.textContent = '';
    setReplayProgress(s.step, clock);
    setWorkbenchState('正在思考下一步', `已调用 ${s.step} 个工具`, 'running');
    const think = (s.llm && s.llm.elapsed_seconds) || 1.5;
    await replayWait(Math.max(300, think * 1000 * scale), token);
    if (token !== replayToken) return;
    clock += think;

    // 执行阶段：左侧先挂上"执行中"，和真实运行一致
    phaseEl.textContent = `正在调用工具 · 第 ${s.step + 1} 个`;
    setReplayProgress(s.step, clock);
    setWorkbenchState(`正在调用工具 · 第 ${s.step + 1} 个`, `已调用 ${s.step} 个工具`, 'running');
    updateCall(keyOf(s), nameOf(s.tool), s.tool, 'running', null,
      s.llm && typeof s.llm.elapsed_seconds === 'number' ? `思考 ${s.llm.elapsed_seconds.toFixed(1)}s` : null);
    const dur = s.duration || 5;
    await replayWait(Math.max(400, dur * 1000 * scale), token);
    if (token !== replayToken) return;
    clock += dur;

    // 揭示这一步
    const card = renderStepCard(s, prevStep);
    streamEl.append(card);
    fillToolResult(card, s);
    updateCall(keyOf(s), nameOf(s.tool), s.tool, s.status === 'error' ? 'error' : 'done', () => card.scrollIntoView({ behavior: 'smooth', block: 'start' }));
    prevStep = s;
    scroll();
  }

  if (token !== replayToken) return;
  // 收尾
  phaseEl.textContent = statusLabel(finalDoc);
  activityEl.textContent = '';
  setWorkbenchState(statusLabel(finalDoc),
    [finalDoc.error, `共 ${steps.length} 步`,
     finalDoc.summary ? `思考 ${finalDoc.summary.llm_seconds}s · 工具 ${finalDoc.summary.tool_seconds}s` : null]
      .filter(Boolean).join(' · '),
    (finalDoc.status === 'success' || finalDoc.status === 'finished') ? 'done' : 'error');
  setProgressMeta(finalDoc, false);
  const finishDec = (finalDoc.extra_decisions || []).filter((x) => x.action === 'finish').pop() || null;
  const finishReason = finishDec && typeof finishDec.reason === 'string' ? finishDec.reason.trim() : '';
  if (answerEl) answerEl.innerHTML = answerCard(finalDoc, finishReason || '任务已完成。', finishDec);
}

async function replayRun(runId) {
  let d;
  try {
    d = await api(`/api/runs/${runId}`);
  } catch (e) {
    status.textContent = '读取运行失败：' + e.message;
    return;
  }
  replayToken += 1;
  const token = replayToken;
  replayPaused = false;
  viewingRun = runId;
  setRunning(false);
  beginRunView(d);
  mountReplayBar();
  await playTimeline(d, token, replaySpeed);
  unmountReplayBar();
}

// 示例：固定的演示时间线。进入示例模式后，由用户自己输入并发送触发。
const DEMO_SPEED = 12;
let demoArmed = false;
const demoBtn = document.querySelector('#demo-btn');

async function playDemo(text) {
  let d;
  try {
    d = await api('/api/demo');
  } catch (e) {
    message('assistant', `<p>示例数据不可用：${escape(e.message)}</p>`);
    return;
  }
  replayToken += 1;
  const token = replayToken;
  replayPaused = false;
  viewingRun = '';
  prompt.value = '';
  document.querySelector('#welcome')?.remove();
  setRunning(false);
  const timeline = { ...d, status: 'running', task: text };
  beginRunView(timeline);          // 和真实运行一样的头部与骨架
  replayBtn.classList.add('hidden');
  await playTimeline(timeline, token, DEMO_SPEED, d);
}

async function renderRun(runId){
  const d = await api(`/api/runs/${runId}`);
  if (viewingRun !== runId) return d;
  if (!streamEl) beginRunView(d);
  showRunModel(d);
  fillToolActivities(subagentEl, d.activities);

  setWorkbenchState(
    d.status === 'running' ? phaseNarrative(d) : statusLabel(d),
    d.status === 'running'
      ? `已调用 ${d.steps.length} 个工具`
      : [d.error, `共 ${d.steps.length} 步`,
         d.summary ? `思考 ${d.summary.llm_seconds}s · 工具 ${d.summary.tool_seconds}s` : null]
          .filter(Boolean).join(' · '),
    d.status === 'running' ? 'running' : (d.status === 'success' || d.status === 'finished') ? 'done' : 'error'
  );

  let prevStep = null;
  for (const s of d.steps) {
    try {
    const key = `s${s.step}_${s.tool}`;
    let rec = rendered.get(s.step);
    if (!rec) {
      const card = renderStepCard(s, prevStep);
      streamEl.append(card);
      rec = { card, filled: false };
      rendered.set(s.step, rec);
      updateCall(key, nameOf(s.tool), s.tool, s.status === 'running' ? 'running' : 'selected', () => card.scrollIntoView({ behavior: 'smooth', block: 'start' }));
      if (s.status !== 'running') updateCall(key, nameOf(s.tool), s.tool, s.status === 'error' ? 'error' : 'done');
      scroll();
    }
    if (s.status !== 'running' && !rec.filled) {
      rec.filled = true;
      fillToolResult(rec.card, s);
      updateCall(key, nameOf(s.tool), s.tool, s.status === 'error' ? 'error' : 'done');
      scroll();
    }
    prevStep = s;
    } catch (err) { console.error('step render failed', s.step, err); }
  }

  if (d.status === 'running') {
    const cur = d.steps.at(-1);
    phaseEl.textContent = phaseNarrative(d);
    activityEl.textContent = '';
    setProgressMeta(d, true);
  } else if (!progressEl.classList.contains('finished')) {
    progressEl.classList.add('finished');
    const spinner = progressEl.querySelector('.thinking-spinner');
    if (spinner) { spinner.className = 'progress-check'; spinner.textContent = '✓'; }
    phaseEl.textContent = statusLabel(d);
    activityEl.textContent = d.error || '';
    setProgressMeta(d, false);
    const finishDec = (d.extra_decisions || []).filter(x => x.action === 'finish').pop() || null;
    const finishReason = finishDec && typeof finishDec.reason === 'string' ? finishDec.reason.trim() : '';
    const doneMsg = finishReason
      || ((d.status === 'success' || d.status === 'finished')
        ? '任务已完成。'
        : `运行结束（${statusLabel(d)}）${d.error ? '：' + d.error : ''}`);
    if (answerEl) answerEl.innerHTML = answerCard(d, doneMsg, finishDec);
    setRunning(false);
  }
  return d;
}

// 运行中的计时逐秒跳动（步骤执行时间 + 顶部已用时间）
setInterval(() => {
  document.querySelectorAll('.step-time-live, .progress-live').forEach((el) => {
    const base = Number(el.dataset.base), stamp = Number(el.dataset.stamp);
    if (!Number.isFinite(base) || !Number.isFinite(stamp)) return;
    const scale = Number(el.dataset.scale) || 1;
    el.textContent = fmtDuration(base + ((Date.now() - stamp) / 1000) * scale);
  });
}, 1000);

// 耗时格式化
function fmtDuration(sec){
  if (typeof sec !== 'number' || !Number.isFinite(sec)) return '';
  return sec >= 60 ? `${Math.floor(sec / 60)}m${Math.round(sec % 60)}s` : `${Math.round(sec)}s`;
}

function errorPresentation(d, detail){
  const message = String(d?.error || detail || '');
  if (/Native execution requires/i.test(message)) {
    return {
      title: '07 策略评测未启动',
      explanation: 'WorldEngine 尚未执行：本机缺少原生评测所需的配置、检查点、模型或数据类型。',
      retryNote: '',
    };
  }
  if (/APIConnectionError|Connection error/i.test(message)) {
    const retry = (d?.summary?.stages || [])
      .reduce((total, stage) => total + (Number(stage.retry_requests) || 0), 0);
    return {
      title: '模型服务连接失败',
      explanation: firstSentence(message, 140),
      retryNote: retry > 0 ? `已自动重试 ${retry} 次` : '',
    };
  }
  return { title: statusLabel(d), explanation: firstSentence(message, 160), retryNote: '' };
}

// 收尾卡片：标题为“任务总结”，右上角为该次总结的思考耗时
function answerCard(d, text, dec){
  const ok = d.status === 'success' || d.status === 'finished';
  const mark = ok ? '✓' : '✕';
  const presentation = ok ? null : errorPresentation(d, text);
  const title = ok ? '任务总结' : presentation.title;
  const think = dec && dec.llm && typeof dec.llm.elapsed_seconds === 'number'
    ? `思考 ${dec.llm.elapsed_seconds.toFixed(1)}s` : '';
  return `<div class="step-card answer-card">`
    + `<div class="step-head"><span class="step-state">${mark}</span>`
    + `<b class="step-name">${escape(title)}</b>`
    + (think ? `<span class="step-times">${escape(think)}</span>` : '')
    + `</div><p class="answer-text">${escape(presentation ? presentation.explanation : text)}</p>`
    + (presentation?.retryNote ? `<p class="answer-retry">${escape(presentation.retryNote)}</p>` : '')
    + (!ok ? '<details class="answer-error"><summary>查看详细错误</summary><pre>'
      + escape(String(d.error || text || '')) + '</pre></details>' : '')
    + '</div>';
}

const ACTIVITY_LABELS = {
  'ego.foci': 'FOCI 优化',
  'ego.understanding': '理解轨迹',
  'ego.constructor': '构造候选',
  'ego.visual_agent': '视觉核对',
  'ego.edit': '编辑轨迹',
  'ego.submit': '提交候选',
  'attack.object_selection': '选择目标',
  'attack.inspect': '检查候选',
  'attack.submit': '提交攻击',
};

function activityRow(record){
  const tone = record.status === 'invalid' || record.status === 'error' || record.status === 'failed'
    ? 'failed' : record.status === 'fallback' ? 'warning' : record.status === 'preview' ? 'preview' : 'normal';
  const marker = tone === 'failed' ? '✕' : tone === 'warning' ? '!' : tone === 'preview' ? '○' : '·';
  const state = tone === 'failed' ? '参数无效' : tone === 'warning' ? '未完全达标'
    : tone === 'preview' ? '预览' : (record.status || '已返回');
  const elapsed = typeof record.elapsed_seconds === 'number' ? fmtDuration(record.elapsed_seconds) : '';
  return `<div class="activity-row ${tone}" data-kind="${escape(record.kind)}">`
    + `<span class="activity-marker">${marker}</span>`
    + `<span class="activity-name">${escape(ACTIVITY_LABELS[record.name] || record.name)}</span>`
    + `<span class="activity-state">${escape(state)}</span>`
    + (record.summary ? `<span class="activity-summary">${escape(record.summary)}</span>` : '')
    + (elapsed ? `<span class="activity-time">${escape(elapsed)}</span>` : '')
    + `</div>`;
}

function fillToolActivities(container, activities){
  if (!container || !Array.isArray(activities) || !activities.length) return;
  container.innerHTML = activities.map(activityRow).join('');
}

function showRunModel(d){
  if (!modelLabel) return;
  const model = d?.llm_model || '未记录';
  const effort = d?.llm_reasoning_effort || '未记录';
  modelLabel.textContent = `主 Agent · ${model} · 思考 ${effort}`;
  modelLabel.title = `本次运行实际使用：${model}；实际思考强度：${effort}`;
}

async function loadSelectedModel(){
  try {
    const cfg = await api('/api/config');
    const active = (cfg.profiles || []).find(profile => profile.active);
    if (active) showRunModel({ llm_model: active.model, llm_reasoning_effort: cfg.reasoning_effort || '跟随 profile' });
  } catch { /* 配置不可读时保留运行详情里的真实模型标签 */ }
}


// 顶栏只讲 agent 在做什么，不暴露具体工具
function phaseNarrative(d){
  const steps = d.steps || [];
  const last = steps.at(-1);
  if (!steps.length) return '正在理解任务';
  if (last.status === 'running') return `正在调用工具 · 第 ${steps.length} 个`;
  return '正在整理结果';
}

function statusLabel(d){
  return { success: '任务完成', finished: '任务完成', running: '执行中', error: '运行出错', interrupted: '已中断', incomplete: '中断' }[d.status] || d.status;
}

async function poll(){
  clearTimeout(pollTimer);
  if (!viewingRun) return;
  try {
    const d = await renderRun(viewingRun);
    if (d.status === 'running') pollTimer = setTimeout(poll, 1000);
  } catch (e) {
    status.textContent = '刷新失败：' + e.message;
    if (phaseEl) phaseEl.textContent = '读取失败';
    if (activityEl) activityEl.textContent = e.message;
    pollTimer = setTimeout(poll, 5000);
  }
}

/* ---------- 发起真实运行 ---------- */
async function run(text){
  if (running || !text.trim()) return;
  if (demoArmed) return playDemo(text.trim());
  const task = text.trim();
  const parentRun = viewingRun;
  document.querySelector('#welcome')?.remove();
  prompt.value = '';
  setRunning(true);
  replayBtn.classList.add('hidden');
  // 先把骨架画出来：服务器要等 Python 启动并打印日志才返回 run_id，
  // 不能让它决定首屏什么时候出现。
  if (parentRun) {
    message('user', `<p>${escape(task).replace(/\n/g, '<br>')}</p>`);
    continueRunView({ task, steps: [], status: 'running', parent_run_id: parentRun });
  } else {
    beginRunView({ task, steps: [], status: 'running' });
  }
  try {
    const { run_id } = await submitConversation(api, text, parentRun);
    viewingRun = run_id;
    closeHistory();
    loadHistory({ force:true });
    poll();
  } catch (e) {
    if (typeof phaseEl !== 'undefined' && phaseEl) phaseEl.textContent = '提交失败';
    if (typeof activityEl !== 'undefined' && activityEl) activityEl.textContent = '';
    if (typeof setWorkbenchState === 'function') setWorkbenchState('提交失败', String(e.message || e), 'error');
    message('assistant', `<p>提交失败：${escape(e.message)}</p>`);
    prompt.value = task;
    setRunning(false);
  }
}

/* ---------- 历史运行 ---------- */
function closeHistory(){
  historyPanel.classList.add('hidden');
  historyBtn.setAttribute('aria-expanded', 'false');
}

function relativeTime(iso){
  if (!iso) return '';
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return '';
  const mins = Math.round((Date.now() - t) / 60000);
  if (mins < 1) return '刚刚';
  if (mins < 60) return `${mins} 分钟前`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours} 小时前`;
  return `${Math.round(hours / 24)} 天前`;
}

function historyItemHtml(r){
  const state = (r.status === 'success' || r.status === 'finished') ? 'done'
    : r.status === 'running' ? 'running' : 'error';
  const steps = r.steps ?? 0;
  const secs = typeof r.duration_seconds === 'number' ? fmtDuration(r.duration_seconds) : '';
  const task = (r.task || '').replace(/\s+/g, ' ').trim();
  return `<button type="button" class="history-item" role="option" data-run="${escape(r.run_id)}" data-state="${state}">`
    + `<span class="history-dot"></span>`
    + `<span class="history-main"><b>${escape(task.slice(0, 46) || r.run_id)}</b>`
    + `<small>${escape(relativeTime(r.started_at))}${steps ? ` · ${steps} 步` : ''}${secs ? ` · ${secs}` : ''}</small></span>`
    + `<span class="history-status">${escape(statusLabel(r))}</span>`
    + `</button>`;
}

let historyConversations = null;
let historyMarkup = '';
let historyFetchedAt = 0;
let historyRequest = null;
async function loadHistory({ force = false } = {}){
  if (historyRequest) return historyRequest;
  if (!force && historyConversations !== null && Date.now() - historyFetchedAt < 10000) return;
  historyPanel.setAttribute('aria-busy', 'true');
  historyFeedback.innerHTML = panelLoadingHtml(historyConversations === null ? '正在读取历史运行…' : '正在更新历史运行…', historyConversations === null ? 3 : 0);
  historyFeedback.classList.remove('hidden');
  historyRequest = (async () => {
    try {
      const { runs } = await readPanelData(options => api('/api/runs', options));
      historyConversations = Array.isArray(runs) ? runs : [];
      historyCount.textContent = String(historyConversations.length);
      const html = historyConversations.length ? historyConversations.map(historyItemHtml).join('')
        : '<p class="history-empty">还没有运行记录</p>';
      if (html !== historyMarkup) historyItems.innerHTML = html;
      historyMarkup = html;
      historyFetchedAt = Date.now();
      historyFeedback.classList.add('hidden');
    } catch (error) {
      historyFeedback.innerHTML = panelErrorHtml(error.message, 'data-history-retry', historyConversations !== null);
    } finally {
      historyPanel.setAttribute('aria-busy', 'false');
      historyRequest = null;
    }
  })();
  return historyRequest;
}

async function loadConversationChain(id){
  const chain = [];
  const seen = new Set();
  let current = id;
  while (current && !seen.has(current)) {
    seen.add(current);
    const detail = await api(`/api/runs/${current}`);
    chain.unshift(detail);
    current = detail.parent_run_id || null;
  }
  return chain;
}

function archivedRunHtml(d){
  const finalDecision = (d.extra_decisions || []).filter(item => item.action === 'finish' || item.action === 'stop').pop();
  const text = finalDecision?.reason || d.error || (d.status === 'success' || d.status === 'finished'
    ? '任务完成。' : `运行结束（${statusLabel(d)}）`);
  return `<div class="archived-run"><b>${escape(statusLabel(d))}</b>`
    + `<p>${escape(firstSentence(text, 220))}</p>`
    + `<small>${d.steps?.length || 0} 步 · ${escape(d.run_id)}</small></div>`;
}

function beginConversationChain(chain){
  messages.replaceChildren();
  rendered = new Map();
  resetWorkbench();
  for (const previous of chain.slice(0, -1)) {
    message('user', `<p>${escape(previous.task || '(无任务描述)').replace(/\n/g, '<br>')}</p>`);
    message('assistant', archivedRunHtml(previous));
  }
  const latest = chain.at(-1);
  message('user', `<p>${escape(latest.task || '(无任务描述)').replace(/\n/g, '<br>')}</p>`);
  mountAgentResponse(chain.length > 1);
}

async function openRun(id){
  if (!id) return;
  closeHistory();
  viewingRun = id;
  setRunning(false);
  try {
    const chain = await loadConversationChain(id);
    const d = chain.at(-1);
    if (viewingRun !== id) return;
    showRunModel(d);
    beginConversationChain(chain);
    replayBtn.classList.remove('hidden');
    poll();
  } catch (e) {
    beginRunView({ task: '', steps: [], status: 'incomplete' });
    status.textContent = '读取运行失败：' + e.message;
  }
}

historyBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  const open = historyPanel.classList.contains('hidden');
  if (!open) return closeHistory();
  historyPanel.classList.remove('hidden');
  historyBtn.setAttribute('aria-expanded', 'true');
  void loadHistory();
});
historyPanel.addEventListener('click', (e) => {
  if (e.target.closest('[data-history-retry]')) { void loadHistory({ force:true }); return; }
  const item = e.target.closest('.history-item');
  if (!item) return;
  historyPanel.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
  item.classList.add('active');
  openRun(item.dataset.run);
});
document.addEventListener('click', (e) => {
  if (!historyMenu.contains(e.target)) closeHistory();
});
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeHistory(); });

/* ---------- 事件 ---------- */
document.querySelector('#composer').addEventListener('submit', e => { e.preventDefault(); run(prompt.value); });
prompt.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); run(prompt.value); } });
stopBtn.addEventListener('click', async () => {
  if (!viewingRun || stopBtn.disabled) return;
  stopBtn.disabled = true;
  status.textContent = '正在停止任务…';
  try {
    await api(`/api/runs/${viewingRun}/stop`, { method: 'POST' });
    status.textContent = '已请求停止';
  } catch (e) {
    status.textContent = '停止失败：' + e.message;
  } finally {
    stopBtn.disabled = false;
    if (viewingRun) await poll();
  }
});

document.querySelector('#example')?.addEventListener('click', () => run('检索一个城市直道前方有低速车辆的场景视频，完成后结束'));
document.querySelector('#replay-btn').addEventListener('click', () => {
  if (!viewingRun) { status.textContent = '先从「历史运行」里选一条记录'; return; }
  replayRun(viewingRun);
});

document.querySelector('#clear').addEventListener('click', () => {
  viewingRun = null; clearTimeout(pollTimer);
  replayToken += 1; unmountReplayBar(); replayBtn.classList.add('hidden');
  setRunning(false); messages.replaceChildren(); resetWorkbench();
  closeHistory();
  demoArmed = false; demoBtn?.classList.remove('hidden');
  message('assistant', '<p>新对话已准备好。输入任务将提交到服务器真实执行。</p>');
  prompt.value = ''; prompt.focus();
});

loadHistory();

/* ---------- 配置面板 ---------- */
const modal = document.querySelector('#config-modal');
const cfgMsg = document.querySelector('#cfg-msg');

const cfgSave = document.querySelector('#cfg-save');
const cfgLoading = document.querySelector('#cfg-loading');
const cfgFields = document.querySelector('#cfg-fields');
cfgSave.disabled = true;

function fillConfig(cfg){
  // 模型 profile：预选当前生效项
  const sel = document.querySelector('#cfg-profile');
  sel.innerHTML = cfg.profiles.map(p =>
    `<option value="${escape(p.name)}" ${p.active ? 'selected' : ''}>${escape(p.name)} · ${escape(p.model)}${p.active ? '（当前生效）' : ''}</option>`).join('');
  const showInfo = () => {
    const p = cfg.profiles.find(x => x.name === sel.value);
    document.querySelector('#cfg-profile-info').textContent = p ? `端点 ${p.base_url}` : '';
  };
  sel.onchange = showInfo; showInfo();
  const effort = document.querySelector('#cfg-effort');
  if (effort) effort.value = cfg.reasoning_effort || 'profile';
  // 工具内部模型：每个工具一行下拉，active_profile = 跟随主 agent
  const profiles = cfg.profiles || [];
  const chosen = cfg.tool_profiles || {};
  const components = cfg.tool_components || [
    { key: 'failure_analyzer', label: '失败分析（①）' },
    { key: 'edit_planner', label: '编辑规划（④）' },
    { key: 'attack_editor', label: '攻击车辆编辑（⑤）' },
    { key: 'ego_editor', label: '专家轨迹编辑（⑥）' },
  ];
  const box = document.querySelector('#cfg-tool-profiles');
  const efforts = cfg.tool_efforts || {};
  const effortOptions = (current) => [
    `<option value="active_profile" ${current === 'active_profile' ? 'selected' : ''}>跟随全局</option>`,
    `<option value="low" ${current === 'low' ? 'selected' : ''}>低</option>`,
    `<option value="medium" ${current === 'medium' ? 'selected' : ''}>中</option>`,
    `<option value="high" ${current === 'high' ? 'selected' : ''}>高</option>`,
  ].join('');
  box.innerHTML = components.map(({ key, label }) => {
    const current = chosen[key] || 'active_profile';
    const options = [`<option value="active_profile" ${current === 'active_profile' ? 'selected' : ''}>跟随主 agent</option>`]
      .concat(profiles.map(p =>
        `<option value="${escape(p.name)}" ${current === p.name ? 'selected' : ''}>${escape(p.name)} · ${escape(p.model)}</option>`))
      .join('');
    const effort = efforts[key] || 'active_profile';
    return `<div class="cfg-row cfg-tool-profile" data-component="${escape(key)}">`
      + `<label>${escape(label)}</label>`
      + `<select class="cfg-tool-model">${options}</select>`
      + `<select class="cfg-tool-effort" title="思考程度">${effortOptions(effort)}</select></div>`;
  }).join('');
  // 参数：预填当前值
  document.querySelector('#cfg-max-steps').value = cfg.max_steps;
  document.querySelector('#cfg-corrections').value = cfg.decision_correction_attempts;
  // 工具模式：预选当前模式
  const toolsBox = document.querySelector('#cfg-tools');
  toolsBox.innerHTML = '';
  for (const [tool, label] of Object.entries(toolNames)) {
    const mode = cfg.tool_modes[tool] || 'real';
    const row = document.createElement('div');
    row.className = 'cfg-tool';
    row.innerHTML = `<span class="tool-label">${escape(label)}<code>${escape(tool)}</code></span>
      <span class="cfg-toggle" data-tool="${escape(tool)}">
        <button type="button" data-mode="real" class="${mode === 'real' ? 'on-real' : ''}">real</button>
        <button type="button" data-mode="mock" class="${mode === 'mock' ? 'on-mock' : ''}">mock</button>
      </span>`;
    toolsBox.append(row);
  }
  toolsBox.querySelectorAll('.cfg-toggle button').forEach(btn => {
    btn.onclick = () => {
      const group = btn.parentElement;
      group.querySelectorAll('button').forEach(b => b.className = '');
      btn.className = btn.dataset.mode === 'real' ? 'on-real' : 'on-mock';
    };
  });
}

function clearConfigFields(){
  for (const id of ['cfg-profile','cfg-tool-profiles','cfg-tools']) document.querySelector('#'+id).innerHTML = '';
  for (const id of ['cfg-max-steps','cfg-corrections','cfg-effort']) document.querySelector('#'+id).value = '';
  document.querySelector('#cfg-profile-info').textContent = '';
}
function collectConfigFields(){
  const tool_modes = {};
  document.querySelectorAll('.cfg-toggle').forEach(t => {
    const on = t.querySelector('button.on-real, button.on-mock');
    tool_modes[t.dataset.tool] = on ? on.dataset.mode : 'real';
  });
  const tool_profiles = {};
  const tool_efforts = {};
  document.querySelectorAll('.cfg-tool-profile').forEach(row => {
    tool_profiles[row.dataset.component] = row.querySelector('.cfg-tool-model').value;
    tool_efforts[row.dataset.component] = row.querySelector('.cfg-tool-effort').value;
  });
  return {
    active_profile: document.querySelector('#cfg-profile').value,
    reasoning_effort: document.querySelector('#cfg-effort')?.value || 'profile',
    tool_profiles,
    tool_efforts,
    max_steps: Number(document.querySelector('#cfg-max-steps').value),
    decision_correction_attempts: Number(document.querySelector('#cfg-corrections').value),
    tool_modes,
  };
}
const configPanel = createConfigPanelController({
  modal, message:cfgMsg, saveButton:cfgSave, fill:fillConfig, clear:clearConfigFields,
  setEditable(enabled){
    document.querySelectorAll('#config-modal input, #config-modal select, #cfg-tools button')
      .forEach(element => { element.disabled = !enabled; });
  },
  read: options => api('/api/config', options),
  write: body => api('/api/config', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),
  onState(state, detail = {}){
    modal.dataset.state = state;
    modal.setAttribute('aria-busy', String(state === 'loading' || state === 'saving'));
    cfgFields.classList.toggle('hidden', state === 'error' || (state === 'loading' && !detail.cached));
    cfgLoading.classList.toggle('hidden', !['loading','error'].includes(state));
    if (state === 'loading') cfgLoading.innerHTML = panelLoadingHtml(detail.cached ? '正在更新配置…' : '正在读取运行配置…', detail.cached ? 0 : 3);
    if (state === 'error') cfgLoading.innerHTML = panelErrorHtml(detail.message, 'data-config-retry');
  },
});
document.querySelector('#config-btn').onclick = configPanel.open;
for (const event of ['pointerenter','focus']) {
  document.querySelector('#config-btn').addEventListener(event, () => { void configPanel.prefetch().catch(() => {}); });
}
document.querySelector('#config-close').onclick = configPanel.close;
document.querySelector('#cfg-cancel').onclick = configPanel.close;
modal.addEventListener('click', event => { if (event.target === modal) configPanel.close(); });
cfgLoading.addEventListener('click', event => { if (event.target.closest('[data-config-retry]')) void configPanel.open(); });
document.querySelector('#cfg-save').onclick = () => configPanel.save(collectConfigFields);

demoBtn?.addEventListener('click', () => {
  demoArmed = true;
  demoBtn.classList.add('hidden');
  prompt.focus();
});
