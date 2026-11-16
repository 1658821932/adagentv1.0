import { toolNames, stageMap } from './agent-ui.js';
import { resetWorkbench, setWorkbenchState, updateCall, setCurrentCall } from './workbench.js';
import { createToolScrollSpy } from './tool-scroll-spy.js';
import { createArtifactPreview, unloadArtifactPreview } from './media-preview.js';
import { submitConversation, groupConversationRuns } from './conversation.js';
import { createApiClient } from './api-client.js';
import { createConfigPanelController } from './config-panel-controller.mjs';
import { panelLoadingHtml, panelErrorHtml, readPanelData } from './panel-loading.mjs';
import {
  firstSentence,
  summaryFor,
  renderStepStructured,
} from './step-output.js';
window.addEventListener('error', (e) => {
  const box = document.createElement('div');
  box.className = 'js-error';
  box.textContent = '前端脚本错误：' + (e.message || String(e.error || ''));
  document.body.append(box);
});

const messages = document.querySelector('#messages');
const toolScrollSpy = createToolScrollSpy(messages, setCurrentCall);
const modelLabel = document.querySelector('#conversation-model');
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
let streamEl = null, progressEl = null, phaseEl = null, activityEl = null, answerEl = null;
let replayToken = 0;         // 每次重放自增，用于中止旧任务
let replayPaused = false;
let replaySpeed = 4;         // 4x 默认，录屏节奏比较合适
let running = false;

/* ---------- 基础渲染（结构同原版） ---------- */
let followLatest = true;
messages.addEventListener('scroll', () => {
  followLatest = messages.scrollHeight - messages.scrollTop - messages.clientHeight < 100;
}, { passive: true });
function scroll(){ if (followLatest) messages.scrollTop = messages.scrollHeight; }
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

function showRunModel(d){
  if (!modelLabel) return;
  const effort = d.llm_reasoning_effort || '未记录';
  modelLabel.textContent = d.llm_model ? `主 Agent · ${d.llm_model} · 思考 ${effort}` : '主 Agent';
  modelLabel.title = `实际思考强度：${effort}`;
}
async function loadSelectedModel(){
  try {
    const cfg = await api('/api/config');
    if (viewingRun) return;
    const profile = cfg.profiles?.find(p => p.active);
    showRunModel({ llm_model: profile?.model, llm_reasoning_effort: profile?.reasoning_effort });
  } catch { /* 配置面板仍可独立重试 */ }
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
    + `<div class="step-body">`
    + (reason ? `<div class="step-reason"><div class="raw-title">调用原因</div><p>${escape(reason)}</p></div>` : '')
    + `<div class="step-main-output"></div>`
    + `<div class="step-drawers"></div></div>`;
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
  } else if (s.status === 'interrupted' || s.output?.error?.type === 'job_interrupted') {
    bits.push('实际耗时未知');
  }
  return bits.join(' · ');
}

function stepStateHtml(s){
  if (s.status === 'running') return '<span class="thinking-spinner"></span>执行中';
  if (s.status === 'success' || s.status === 'done') return '✓';
  return '✕';
}

const primaryOutput = summaryFor;

const ACTIVITY_LABELS = {
  model_request:'模型 API 调用',
  'attack.object_selection':'选择攻击车辆', 'attack.inspect':'查看攻击轨迹',
  'attack.submit':'提交攻击轨迹', 'attack.materialize_added':'生成新增车辆',
  'attack.independent_check':'独立检查攻击结果', 'ego.understanding':'理解冻结场景',
  'ego.constructor':'生成防御候选', 'ego.foci':'FOCI 优化',
  'ego.visual_agent':'防御编辑 Agent', 'ego.select_constructor':'选择防御候选',
  'ego.edit':'编辑防御轨迹', 'ego.submit':'提交专家轨迹', 'ego.independent_check':'独立检查防御结果',
  select_vehicle:'选择车辆', construct_attack_candidates:'构造攻击候选',
  construct_trajectory:'构造轨迹', edit_speed_profile:'调整速度', edit_route:'调整行驶路线',
  repair_candidate:'修复候选', inspect_trajectory:'查看轨迹图', evaluate_candidate:'检查候选',
  submit_candidate:'提交候选', finish_search:'结束搜索', request_object_reselection:'重新选择车辆',
  probe_vadv2:'VADv2 自测', probe_pdm:'PDM 自测',
};
function activityRow(a){
  const labels = { invalid:'参数无效', error:'失败', failed:'失败', fallback:'未完全达标',
    no_feasible_goal:'未找到达标候选', running:'执行中', started:'执行中',
    preview:'预览', returned:'已返回', completed:'已完成', success:'已完成', constructed:'已生成',
    selected:'已选择', submitted:'已提交', interrupted:'已中断' };
  const failed = ['invalid','error','failed','interrupted'].includes(a.status);
  const warning = ['fallback','no_feasible_goal'].includes(a.status);
  const live = ['running','started'].includes(a.status) && !a.finished_at;
  const state = failed ? 'failed' : warning ? 'warning' : a.status === 'preview' ? 'preview' : live ? 'running' : 'done';
  const elapsed = typeof a.elapsed_seconds === 'number' ? fmtDuration(a.elapsed_seconds)
    : live && a.timestamp ? fmtDuration(Math.max(0,(Date.now()-Date.parse(a.timestamp))/1000)) : '';
  return `<li class="activity-row ${state}"><span>${failed ? '✕' : warning ? '△' : state === 'preview' ? '○' : live ? '◌' : '✓'}</span>`
    + `<div><b>${escape(ACTIVITY_LABELS[a.name] || a.name)}</b><small>${escape(labels[a.status] || a.status || '')}${elapsed ? ' · ' + escape(elapsed) : ''}</small>`
    + (a.summary ? `<p>${escape(a.summary)}</p>` : '') + `</div></li>`;
}
function fillToolActivities(card, s){
  if (!s.activities?.length) return;
  let box = card.querySelector('.subagent-activity');
  if (!box) {
    box = document.createElement('details');
    box.className = 'subagent-activity';
    box.open = s.status === 'running';
    box.innerHTML = '<summary></summary><ol></ol>';
    const host = card.querySelector('.step-drawers') || card.querySelector('.step-body');
    const jsonBlock = host.querySelector('.json-block');
    if (jsonBlock) host.insertBefore(box, jsonBlock);
    else host.append(box);
  }
  box.querySelector('summary').textContent = `子 Agent 活动 · ${s.activities.length} 条`;
  const html = s.activities.map(activityRow).join('');
  if (box._activityHtml !== html) { box.querySelector('ol').innerHTML = html; box._activityHtml = html; }
}

function fillToolResult(card, s){
  const stateEl = card.querySelector('.step-state');
  if (stateEl) stateEl.innerHTML = stepStateHtml(s);
  const timesEl = card.querySelector('.step-times');
  if (timesEl) timesEl.innerHTML = stepTimes(s);

  const mainHost = card.querySelector('.step-main-output') || card.querySelector('.step-body') || card;
  const drawerHost = card.querySelector('.step-drawers') || card.querySelector('.step-body') || card;
  mainHost.replaceChildren();

  const structuredHtml = renderStepStructured(s);
  if (structuredHtml) mainHost.insertAdjacentHTML('beforeend', structuredHtml);

  const rawMedia = (Array.isArray(s.media) && s.media.length)
    ? s.media
    : (s.evidence || []).map(v => ({ ...v, kind: 'video', primary: true }));
  const preview = createArtifactPreview(rawMedia.map(item => ({ ...item, src: serviceUrl(item.src) })));
  if (preview) mainHost.append(preview);

  const raw = [];
  if (s.input != null) raw.push(`<div class="raw-block"><div class="raw-title">输入 JSON</div><pre>${escape(JSON.stringify(s.input, null, 2))}</pre></div>`);
  if (s.output != null) raw.push(`<div class="raw-block"><div class="raw-title">输出 JSON</div><pre>${escape(JSON.stringify(s.output, null, 2))}</pre></div>`);
  if (raw.length && !drawerHost.querySelector('.json-block')) {
    const d = document.createElement('details');
    d.className = 'json-block';
    d.innerHTML = '<summary>查看输入与输出 JSON</summary>' + raw.join('');
    drawerHost.append(d);
  }

  fillToolActivities(card, s);
  const actEl = card.querySelector('.subagent-activity');
  if (actEl && s.status !== 'running') actEl.open = false;
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
  unloadArtifactPreview(messages);
  messages.replaceChildren();
  followLatest = true;
  rendered = new Map();
  toolScrollSpy.reset();
  resetWorkbench();
  message('user', `<p>${escape(d.task || '(无任务描述)').replace(/\n/g, '<br>')}</p>`);
  mountAgentResponse(d);
}
function mountAgentResponse(d, continued = false){
  rendered = new Map();
  showRunModel(d);
  const response = message('assistant', (continued ? '<p class="continuation-note">已接续历史对话，保留以上执行记录。</p>' : '')
    + `<div class="agent-progress"><span class="thinking-spinner"></span><b class="agent-phase">任务理解</b><span class="agent-activity">准备决策</span><span class="agent-count"></span></div><div class="answer-slot"></div><div class="tool-stream"></div>`);
  streamEl = response.querySelector('.tool-stream');
  progressEl = response.querySelector('.agent-progress');
  phaseEl = response.querySelector('.agent-phase');
  activityEl = response.querySelector('.agent-activity');
  answerEl = response.querySelector('.answer-slot');
}
function continueRunView(d){ mountAgentResponse(d, true); }


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
  const keyOf = (s) => `${d.run_id || viewingRun}_s${s.step}_${s.tool}`;
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
    toolScrollSpy.observe(card, keyOf(s));
    fillToolResult(card, s);
    updateCall(keyOf(s), nameOf(s.tool), s.tool, ['error','interrupted','failed'].includes(s.status) ? 'error' : 'done', () => card.scrollIntoView({ behavior: 'smooth', block: 'start' }));
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
  if (answerEl) {
    answerEl.innerHTML = answerCard(finalDoc, finishReason || '任务已完成。', finishDec);
  }
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
  return updateRunView(d);
}
function updateRunView(d){
  const active = d.active ?? (d.status === 'running');
  if (running !== active) setRunning(active);
  showRunModel(d);

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
    const key = `${d.run_id || viewingRun}_s${s.step}_${s.tool}`;
    let rec = rendered.get(s.step);
    if (!rec) {
      const card = renderStepCard(s, prevStep);
      streamEl.append(card);
      toolScrollSpy.observe(card, key);
      rec = { card, filled: false };
      rendered.set(s.step, rec);
      updateCall(key, nameOf(s.tool), s.tool, s.status === 'running' ? 'running' : 'selected', () => {
        followLatest = false; toolScrollSpy.select(key);
        card.scrollIntoView({ behavior:'smooth', block:'start' });
      });
      if (s.status !== 'running') updateCall(key, nameOf(s.tool), s.tool, ['error','interrupted','failed'].includes(s.status) ? 'error' : 'done');
      scroll();
    }
    fillToolActivities(rec.card, s);
    if (s.status === 'running') {
      rec.card.querySelector('.step-times').innerHTML = stepTimes(s);
    }
    if (s.status !== 'running' && !rec.filled) {
      rec.filled = true;
      fillToolResult(rec.card, s);
      updateCall(key, nameOf(s.tool), s.tool, ['error','interrupted','failed'].includes(s.status) ? 'error' : 'done');
      scroll();
    }
    prevStep = s;
    } catch (err) { console.error('step render failed', s.step, err); }
  }

  if (d.status === 'running') {
    const cur = d.steps.at(-1);
    phaseEl.textContent = phaseNarrative(d);
    const latest = cur?.activities?.at(-1);
    activityEl.textContent = latest ? `${ACTIVITY_LABELS[latest.name] || latest.name} · ${latest.summary || latest.status}` : '';
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
    if (answerEl) {
      answerEl.innerHTML = answerCard(d, doneMsg, finishDec);
    }
    setRunning(Boolean(d.active));
  }
  toolScrollSpy.schedule();
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
  if (sec > 0 && sec < .1) return `${Math.round(sec*1000)}ms`;
  return sec >= 60 ? `${Math.floor(sec / 60)}m${Math.round(sec % 60)}s` : `${Math.round(sec)}s`;
}

// 收尾卡片：标题为“任务总结”，右上角为该次总结的思考耗时
function errorPresentation(d, text){
  const detail = d.error || text || '';
  if (/Native execution requires|config_path.*checkpoint_path|execution_mode=native/.test(detail))
    return { title:'07 策略评测未启动', message:'模型配置校验未通过，WorldEngine 尚未执行。' };
  if (/APIConnectionError|Connection error|frontend_network_error/.test(detail)) {
    const retries = d.summary?.stages?.find(s => s.stage === 'main_agent')?.retry_requests;
    return { title:'模型服务连接失败', message:typeof retries === 'number' && retries > 0 ? `已自动重试 ${retries} 次，仍未连接成功。` : '请检查本轮详细错误与服务连接。' };
  }
  return { title:statusLabel(d), message:firstSentence(detail, 180) };
}
function answerCard(d, text, dec){
  const ok = d.status === 'success' || d.status === 'finished';
  const mark = ok ? '✓' : '✕';
  const error = ok ? null : errorPresentation(d, text);
  const title = ok ? '任务总结' : error.title;
  const think = dec && dec.llm && typeof dec.llm.elapsed_seconds === 'number'
    ? `思考 ${dec.llm.elapsed_seconds.toFixed(1)}s` : '';
  return `<div class="step-card answer-card">`
    + `<div class="step-head"><span class="step-state">${mark}</span>`
    + `<b class="step-name">${escape(title)}</b>`
    + (think ? `<span class="step-times">${escape(think)}</span>` : '')
    + `</div><p class="answer-text">${escape(ok ? text : error.message)}</p>`
    + (!ok && (d.error || text) ? `<details><summary>查看详细错误</summary><pre>${escape(d.error || text)}</pre></details>` : '')
    + '</div>';
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
  const selected = viewingRun;
  try {
    const d = await renderRun(selected);
    if (viewingRun !== selected) return;
    if (d.active || d.status === 'running') pollTimer = setTimeout(poll, 1000);
  } catch (e) {
    if (viewingRun !== selected) return;
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
  const parentRun = viewingRun;
  document.querySelector('#welcome')?.remove();
  prompt.value = '';
  followLatest = true;
  stopBtn.disabled = false;
  setRunning(true);
  replayBtn.classList.add('hidden');
  // 先把骨架画出来：服务器要等 Python 启动并打印日志才返回 run_id，
  // 不能让它决定首屏什么时候出现。
  if (parentRun) message('user', `<p>${escape(text.trim()).replace(/\n/g, '<br>')}</p>`);
  else beginRunView({ task: text.trim(), steps: [], status: 'running' });
  try {
    const { run_id } = await submitConversation(api, text, parentRun);
    viewingRun = run_id;
    rememberRun(run_id);
    const detail = await api(`/api/runs/${run_id}`);
    if (parentRun) continueRunView(detail);
    closeHistory();
    loadHistory({ force:true });
    poll();
  } catch (e) {
    prompt.value = text;
    phaseEl.textContent = '提交失败';
    activityEl.textContent = '';
    setWorkbenchState('提交失败', String(e.message || e), 'error');
    message('assistant', `<p>提交失败：${escape(e.message)}</p>`);
    setRunning(false);
  }
}

function rememberRun(id){
  try { if (id) localStorage.setItem('adagent.selectedRun', id); else localStorage.removeItem('adagent.selectedRun'); } catch { /* 浏览器可能禁用存储 */ }
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
      historyConversations = groupConversationRuns(Array.isArray(runs) ? runs : []);
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
  const chain = [], seen = new Set();
  while (id && !seen.has(id)) {
    seen.add(id);
    const detail = await api(`/api/runs/${id}`);
    chain.unshift(detail);
    id = detail.parent_run_id;
  }
  return chain;
}

async function openRun(id){
  if (!id) return;
  closeHistory();
  viewingRun = id;
  rememberRun(id);
  clearTimeout(pollTimer);
  replayToken += 1;
  unmountReplayBar();
  setRunning(false);
  try {
    const chain = await loadConversationChain(id);
    if (viewingRun !== id) return;
    chain.forEach((d, index) => {
      if (index === 0) beginRunView(d);
      else {
        message('user', `<p>${escape(d.user_messages?.at(-1)?.content || d.task || '').replace(/\n/g,'<br>')}</p>`);
        continueRunView(d);
      }
      updateRunView(d);
    });
    replayBtn.classList.toggle('hidden', chain.at(-1).status === 'running');
    poll();
  } catch (e) {
    if (viewingRun !== id) return;
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
    await poll();
  } catch (e) {
    status.textContent = '停止失败：' + e.message;
  } finally {
    stopBtn.disabled = false;
  }
});

document.querySelector('#example')?.addEventListener('click', () => run('检索一个城市直道前方有低速车辆的场景视频，完成后结束'));
document.querySelector('#replay-btn').addEventListener('click', () => {
  if (!viewingRun) { status.textContent = '先从「历史运行」里选一条记录'; return; }
  replayRun(viewingRun);
});

document.querySelector('#clear').addEventListener('click', () => {
  viewingRun = null; clearTimeout(pollTimer);
  rememberRun(null);
  replayToken += 1; unmountReplayBar(); replayBtn.classList.add('hidden');
  unloadArtifactPreview(messages); toolScrollSpy.reset();
  setRunning(false); messages.replaceChildren(); resetWorkbench();
  closeHistory();
  demoArmed = false; demoBtn?.classList.remove('hidden');
  message('assistant', '<p>新对话已准备好。输入任务将提交到服务器真实执行。</p>');
  prompt.value = ''; prompt.focus();
});

loadHistory();
try { const saved = localStorage.getItem('adagent.selectedRun'); if (saved) void openRun(saved); } catch { /* 无持久存储时仍可从历史打开 */ }
void loadSelectedModel();

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
