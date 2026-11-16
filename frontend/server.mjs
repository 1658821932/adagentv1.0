/**
 * adagent 真实接入服务器
 * - 静态前端
 * - /api/runs           GET  历史运行列表
 * - /api/runs           POST {task} 发起真实 Agent 运行
 * - /api/runs/:id       GET  运行详情（实时步骤：decision + 工具输入输出 + 产物）
 * - /api/runs/:id/stop  POST 停止运行
 * - /api/runs/:id/console GET 控制台输出
 * - /api/file?path=     GET  项目内文件代理（视频支持 Range）
 */
import http from 'node:http';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { stat, readdir, readFile, writeFile, mkdir, realpath, rename, rm } from 'node:fs/promises';
import { createReadStream, existsSync, readdirSync, statSync } from 'node:fs';
import { resolve, extname, sep, join, basename, dirname } from 'node:path';
import { resolveRunStatus } from './run-status.mjs';
import { createRunCoordinator, conversationDetails } from './run-coordinator.mjs';
import { stopRun } from './run-stop.mjs';
import { previewMedia } from './run-media.mjs';
import { readSubagentActivity } from './run-activity.mjs';
import { resolveFfmpeg } from './video-tools.mjs';
import { videoCacheIdentity } from './video-cache.mjs';

const ROOT = resolve(process.env.ADAGENT_PROJECT_ROOT || fileURLToPath(new URL('..', import.meta.url)));
const LOGS = join(ROOT, 'logs');
const PYTHON = process.env.ADAGENT_AGENT_PYTHON || join(ROOT, 'environments/agent/.venv/bin/python');
const FRONTEND_DIR = resolve(process.cwd());
const CONFIGS = join(ROOT, 'configs');
const PORT = Number(process.env.PORT || 5173);
const HOST = process.env.HOST || '127.0.0.1';

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.json': 'application/json; charset=utf-8',
  '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
  '.gif': 'image/gif', '.webp': 'image/webp', '.svg': 'image/svg+xml',
  '.mp4': 'video/mp4', '.webm': 'video/webm', '.pdf': 'application/pdf',
  '.txt': 'text/plain; charset=utf-8', '.log': 'text/plain; charset=utf-8',
  '.md': 'text/plain; charset=utf-8', '.yaml': 'text/plain; charset=utf-8',
  '.yml': 'text/plain; charset=utf-8', '.csv': 'text/csv; charset=utf-8',
};
const MEDIA_EXT = new Set(['.mp4', '.webm', '.jpg', '.jpeg', '.png', '.gif', '.webp']);
const ARTIFACT_FILE_EXT = new Set(['.json', '.pdf', '.csv', '.md', '.txt', '.pth', '.ckpt', '.pkl', '.ply', '.yaml', '.yml', '.npy', '.log']);

/** run_id -> { proc, task, startedAt, console: string[], exitCode } */
const activeRuns = new Map();
const coordinator = createRunCoordinator({ root:ROOT, logsRoot:LOGS, python:PYTHON, activeRuns });

// Follow only the project's declared dataset links, not an entire shared checkout.
async function resourceAllowed(path) {
  const candidate = resolve(path);
  if (candidate.startsWith(ROOT + sep)) return true;
  const actual = await realpath(candidate).catch(() => null);
  if (!actual) return false;
  const children = await readdir(join(ROOT,'data')).catch(() => []);
  const roots = await Promise.all(children.map(name => realpath(join(ROOT,'data',name)).catch(() => null)));
  return roots.some(root => root && (actual === root || actual.startsWith(root + sep)));
}

/* ---------------- 工具函数 ---------------- */

async function readJson(path) {
  try { return JSON.parse(await readFile(path, 'utf-8')); }
  catch { return null; }
}

async function readTextTail(path, maxBytes = 20000) {
  try {
    const s = await stat(path);
    const start = Math.max(0, s.size - maxBytes);
    const stream = createReadStream(path, { start });
    const chunks = [];
    for await (const c of stream) chunks.push(c);
    return Buffer.concat(chunks).toString('utf-8');
  } catch { return ''; }
}

/** 主产物字段：这些字段的媒体直接内嵌展示；其余（如 source_file 溯源引用）归入文件列表 */
/** 从 agent.jsonl 的一条决策里提取主 Agent 的 LLM 开销 */
function llmOf(decision) {
  const lm = decision?.llm_request?.metrics || {};
  const lu = decision?.llm_request?.usage || {};
  return {
    elapsed_seconds: typeof lm.elapsed_seconds === 'number' ? lm.elapsed_seconds : null,
    first_chunk_seconds: typeof lm.first_chunk_seconds === 'number' ? lm.first_chunk_seconds : null,
    chunks: typeof lm.chunks === 'number' ? lm.chunks : null,
    prompt_tokens: typeof lu.prompt_tokens === 'number' ? lu.prompt_tokens : null,
    completion_tokens: typeof lu.completion_tokens === 'number' ? lu.completion_tokens : null,
    cached_tokens: typeof lu.prompt_tokens_details?.cached_tokens === 'number' ? lu.prompt_tokens_details.cached_tokens : null,
    reasoning_tokens: typeof lu.completion_tokens_details?.reasoning_tokens === 'number' ? lu.completion_tokens_details.reasoning_tokens : null,
  };
}



// ---- 仿真场景八宫格：WE 归档里的 camera_view 是 FMP4，浏览器放不了，按需转码并缓存 ----
const WE_ROOT = '/inspire/hdd3/project/intelligent-driving-agent/public/workspace/ghc/WorldEngine';
const CLOSED_LOOP_ROOT = '/inspire/hdd3/project/intelligent-driving-agent/public/workspace/ghc/WorldEngine/experiments/closed_loop_exps';
const EIGHTCAM_CACHE = join(ROOT, 'frontend', '.cache', 'eightcam');
const eightcamJobs = new Map();

function eightcamSource(simulation, token) {
  if (!simulation || !token) return null;
  const record = String(simulation).trim();
  const short = String(token).trim().split('-').pop();
  if (!record || !short) return null;
  const dir = record.startsWith('/') ? record : join(CLOSED_LOOP_ROOT, record, 'visualizations', 'cameras');
  const base = record.startsWith('/') ? dir : join(CLOSED_LOOP_ROOT, record, 'visualizations', 'cameras');
  const candidates = [
    join(base, `camera_view_${short}.mp4`),
    join(CLOSED_LOOP_ROOT, record, `camera_view_${short}.mp4`),
  ];
  for (const c of candidates) if (existsSync(c)) return c;
  return null;
}

async function ensurePlayableEightcam(src) {
  await mkdir(EIGHTCAM_CACHE, { recursive: true });
  const key = await videoCacheIdentity(src);
  const out = join(EIGHTCAM_CACHE, key);
  if (existsSync(out) && (await stat(out)).size > 0) return out;
  if (eightcamJobs.has(out)) return eightcamJobs.get(out);
  const ffmpeg = resolveFfmpeg(ROOT);
  const tmp = join(EIGHTCAM_CACHE, `${key}.${process.pid}.${Date.now()}.tmp.mp4`);
  const job = new Promise((resolvePromise, rejectPromise) => {
    const proc = spawn(ffmpeg, [
      '-y', '-loglevel', 'error', '-i', src,
      '-vf', 'scale=1920:-2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', '-r', '2', '-movflags', '+faststart', tmp,
    ], { stdio: 'ignore' });
    proc.on('exit', async (code) => {
      try {
        if (code === 0 && existsSync(tmp)) {
          if (await videoCacheIdentity(src) !== key) throw new Error('视频已更新，请重新加载');
          await rename(tmp, out);
          resolvePromise(out);
        } else {
          await rm(tmp, { force: true }).catch(() => {});
          rejectPromise(new Error(`ffmpeg exit ${code}`));
        }
      } catch (err) {
        await rm(tmp, { force: true }).catch(() => {});
        rejectPromise(err);
      } finally {
        eightcamJobs.delete(out);
      }
    });
    proc.on('error', async (err) => {
      eightcamJobs.delete(out);
      await rm(tmp, { force: true }).catch(() => {});
      rejectPromise(err);
    });
  });
  eightcamJobs.set(out, job);
  return job;
}


/** 各工具的关键视频证据（统一返回可直接播放的 URL） */
function evidenceVideos(tool, output) {
  const r = output?.result || {};
  const list = [];
  if (tool === 'failure_analyzer') {
    const src = eightcamSource(r.simulation, r.token);
    if (src) list.push({ title: '关键证据', src: '/api/eightcam?path=' + encodeURIComponent(src) });
  }
  if (tool === 'policy_runner' && Array.isArray(r.videos)) {
    r.videos.slice(0, 2).forEach((p, i) => {
      if (existsSync(p)) list.push({ title: i === 0 ? '仿真回放' : `仿真回放 ${i + 1}`, src: '/api/wevideo?path=' + encodeURIComponent(p) });
    });
  }
  if (tool === 'ego_trajectory_editor' && r.rendered_videos) {
    const v = r.rendered_videos.manual_explained;
    if (typeof v === 'string' && existsSync(v)) {
      list.push({ title: '专家轨迹回放', src: '/api/wevideo?path=' + encodeURIComponent(v) });
    }
  }
  if (tool === 'video_retriever') {
    if (r.video_path && existsSync(r.video_path)) {
      // 无损导出可能是 4:4:4（浏览器解不了），优先用同 rank+score 的 compressed 版本
      let playable = r.video_path;
      const tag = /(rank\d+_score[\d.]+)_lossless\.mp4$/i.exec(r.video_path);
      if (tag) {
        try {
          const dir = dirname(r.video_path);
          const hit = readdirSync(dir).find((f) => f.includes(tag[1]) && f.endsWith('_compressed.mp4'));
          if (hit) playable = join(dir, hit);
        } catch { /* 目录不可读就用原文件 */ }
      }
      list.push({ title: '前视检索视频', src: '/api/file?path=' + encodeURIComponent(playable) });
    }
    const root = r.scene_ref?.sample_root;
    if (root) {
      const grid = join(root, 'preview', 'original_eightcam_2hz.mp4');
      if (existsSync(grid)) list.push({ title: '溯源八宫格', src: '/api/file?path=' + encodeURIComponent(grid) });
    }
  }
  return list;
}

/** Only expose progress metadata; provider-private reasoning is not a UI artifact. */
async function readThinking(runDir) {
  const file = join(runDir, 'thinking.jsonl');
  if (!existsSync(file)) return null;
  const text = await readTextTail(file, 200_000);
  let elapsed = 0, step = null;
  for (const line of text.split('\n')) {
    if (!line.trim()) continue;
    try {
      const ev = JSON.parse(line);
      if (typeof ev.step === 'number') step = ev.step;
      if (typeof ev.elapsed === 'number') elapsed = ev.elapsed;
    } catch { /* 忽略半行 */ }
  }
  return { step, elapsed };
}

const PRIMARY_MEDIA_KEYS = new Set([
  'video_path', 'rendered_video', 'output_video', 'result_video', 'render_video',
  'image_path', 'output_image', 'preview_path', 'result_image', 'figure_path',
]);

/** 递归收集产物文件路径与数值指标 */
function extractArtifactsAndMetrics(node, artifacts, metrics, keyPath = '') {
  if (node == null) return;
  if (typeof node === 'string') {
    if (node.startsWith(ROOT) || node.startsWith('/inspire/')) {
      const ext = extname(node).toLowerCase();
      if (MEDIA_EXT.has(ext) || ARTIFACT_FILE_EXT.has(ext)) {
        const leaf = keyPath.split('.').pop()?.replace(/\[\d+\]$/, '') || '';
        artifacts.push({
          path: node,
          kind: MEDIA_EXT.has(ext) ? (ext === '.mp4' || ext === '.webm' ? 'video' : 'image') : 'file',
          label: keyPath,
          primary: PRIMARY_MEDIA_KEYS.has(leaf) && !keyPath.includes('.'),
        });
      }
    }
    return;
  }
  if (typeof node === 'number' && Number.isFinite(node)) {
    const short = keyPath.split('.').pop();
    if (short && !/(^|_)(id|index|token)$/i.test(short)) metrics.push({ key: keyPath, value: node });
    return;
  }
  if (Array.isArray(node)) {
    node.slice(0, 20).forEach((v, i) => extractArtifactsAndMetrics(v, artifacts, metrics, `${keyPath}[${i}]`));
    return;
  }
  if (typeof node === 'object') {
    for (const [k, v] of Object.entries(node)) {
      extractArtifactsAndMetrics(v, artifacts, metrics, keyPath ? `${keyPath}.${k}` : k);
    }
  }
}

/** 解析一个 run 目录，返回详情 */
async function getRunDetail(runId) {
  if (!/^run_[\w-]+$/.test(runId)) return null;
  const runDir = join(LOGS, runId);
  if (!existsSync(runDir)) return null;

  const meta = await readJson(join(runDir, 'run.json')) || {};
  const finalState = await readJson(join(runDir, 'final_state.json'));
  const runSummary = await readJson(join(runDir, 'summary.json'));
  const runState = await resolveRunStatus(meta, finalState, activeRuns.get(runId));
  const [conversation, activities] = await Promise.all([conversationDetails(runDir), readSubagentActivity(runDir)]);

  // decision 流（运行中实时可读）
  const decisions = new Map();
  const jsonlPath = join(runDir, 'agent.jsonl');
  if (existsSync(jsonlPath)) {
    const text = await readTextTail(jsonlPath, 5_000_000);
    for (const line of text.split('\n')) {
      if (!line.trim()) continue;
      try {
        const ev = JSON.parse(line);
        const d = ev.decision || ev;
        const step = ev.step ?? d.step;
        if (typeof step === 'number') decisions.set(step, d);
      } catch { /* 忽略半行 */ }
    }
  }

  // 工具步骤目录
  const toolsDir = join(runDir, 'tools');
  const steps = [];
  if (existsSync(toolsDir)) {
    const entries = await readdir(toolsDir, { withFileTypes: true });
    for (const e of entries) {
      if (!e.isDirectory()) continue;
      const m = /^step_(\d+)_(.+)$/.exec(e.name);
      if (!m) continue;
      const stepNum = Number(m[1]);
      const tool = m[2];
      const dir = join(toolsDir, e.name);
      const input = await readJson(join(dir, 'input.json'));
      const output = await readJson(join(dir, 'output.json'));
      const logTail = await readTextTail(join(dir, 'execution.log'), 8000);
      const durationMatch = /duration_seconds=([0-9.]+)/.exec(logTail);
      const timing = await readJson(join(dir, 'timing.json'));
      const duration = typeof timing?.elapsed_seconds === 'number' ? timing.elapsed_seconds : durationMatch ? Number(durationMatch[1]) : null;
      // 运行中的工具：用开始时间戳算已执行秒数，前端据此走表
      const startedMatch = /\[([^\]]+)\] tool started/.exec(logTail);
      const startedAt = Date.parse(timing?.started_at || startedMatch?.[1]);
      const elapsedSeconds = Number.isFinite(startedAt) && duration == null
        ? Math.max(0, (Date.now() - startedAt) / 1000)
        : null;

      const artifacts = [];
      const metrics = [];
      if (output?.result) extractArtifactsAndMetrics(output.result, artifacts, metrics);
      // 视频检索: 附加数据集 preview 下的八宫格溯源视频(离线批量生成, 存在才展示)
      if (tool === 'video_retriever' && output?.status === 'success') {
        const sampleRoot = output?.result?.scene_ref?.sample_root;
        if (sampleRoot) {
          const grid = join(sampleRoot, 'preview', 'original_eightcam_2hz.mp4');
          if (existsSync(grid)) artifacts.push({ path: grid, kind: 'video', label: 'eightcam_grid', primary: true });
        }
      }

      const decision = decisions.get(stepNum) || null;

      steps.push({
        step: stepNum,
        tool,
        status: output ? (output.status || 'done') : (runState.active ? 'running' : 'interrupted'),
        decision,
        duration,
        elapsed_seconds: runState.active ? elapsedSeconds : null,
        evidence: evidenceVideos(tool, output),
        media: previewMedia(tool, output, { projectRoot:ROOT, weRoot:WE_ROOT, closedLoopRoot:CLOSED_LOOP_ROOT }),
        activities: activities.filter(item => item.step === stepNum).map(item => !runState.active && ['running','started'].includes(item.status) && !item.finished_at ? { ...item, status:'interrupted' } : item),
        // 主 Agent 思考开销（来自 agent.jsonl 的 llm_request）
        llm: llmOf(decision),
        input,
        output,
        artifacts,
        metrics: metrics.slice(0, 30),
        log_tail: logTail.slice(-4000),
      });
    }
  }
  steps.sort((a, b) => a.step - b.step);

  const thinking = await readThinking(runDir);

  // 运行级汇总：累计思考时间 vs 累计工具执行时间
  const ordered = [...decisions.entries()].sort((a, b) => a[0] - b[0]);
  let llmSeconds = 0, toolSeconds = 0, promptTokens = 0, completionTokens = 0, cachedTokens = 0;
  for (let i = 0; i < ordered.length; i += 1) {
    const dec = ordered[i][1];
    const elapsed = dec?.llm_request?.metrics?.elapsed_seconds;
    if (typeof elapsed === 'number') llmSeconds += elapsed;
    const usage = dec?.llm_request?.usage || {};
    if (typeof usage.prompt_tokens === 'number') promptTokens += usage.prompt_tokens;
    if (typeof usage.completion_tokens === 'number') completionTokens += usage.completion_tokens;
    if (typeof usage.prompt_tokens_details?.cached_tokens === 'number') cachedTokens += usage.prompt_tokens_details.cached_tokens;
    const t1 = Date.parse(dec?.timestamp);
    const next = ordered[i + 1]?.[1];
    const t2 = Date.parse(next?.timestamp);
    if (Number.isFinite(t1) && Number.isFinite(t2)) {
      const nextElapsed = next?.llm_request?.metrics?.elapsed_seconds || 0;
      toolSeconds += Math.max(0, (t2 - t1) / 1000 - nextElapsed);
    }
  }
  const summary = {
    decisions: ordered.length,
    llm_seconds: Math.round(llmSeconds * 10) / 10,
    tool_seconds: Math.round(toolSeconds * 10) / 10,
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    cached_tokens: cachedTokens,
    ...(runSummary ? {
      api_calls: runSummary.api_calls,
      api_seconds: runSummary.api_seconds,
      wall_seconds: runSummary.wall_seconds,
      stages: runSummary.stages,
    } : {}),
  };

  // 无工具目录的决策（通常是最后的 finish/stop），单独返回以便展示最终思考
  const stepNums = new Set(steps.map((s) => s.step));
  const extraDecisions = ordered
    .filter(([n]) => !stepNums.has(n))
    .map(([n, dec]) => ({
      step: n,
      action: dec?.action || null,
      tool_name: dec?.tool_name || null,
      reason: dec?.reason || null,
      timestamp: dec?.timestamp || null,
      llm: llmOf(dec),
    }));

  // 整体状态
  const reportFile = join(runDir, 'REPORT.md');
  return {
    run_id: runId,
    task: meta.task || finalState?.task || '',
    status:runState.status,
    error: finalState?.error || null,
    started_at: meta.start_time || meta.started_at || meta.created_at || null,
    duration_seconds:runState.duration_seconds,
    ...conversation,
    llm_model: meta.llm?.model || meta.llm_model || null,
    llm_reasoning_effort: meta.llm?.reasoning_effort || null,
    active:runState.active,
    report_path: existsSync(reportFile) ? reportFile : null,
    summary,
    thinking,
    extra_decisions: extraDecisions,
    steps,
  };
}

/** 历史运行列表 */
async function listRuns() {
  const entries = await readdir(LOGS, { withFileTypes: true });
  const directories = entries.filter(e => e.isDirectory() && /^run_\d{8}_\d{6}_\d+$/.test(e.name));
  const runs = [];
  // Bound parallel filesystem work, rather than awaiting every file serially.
  for (let offset=0; offset<directories.length; offset+=12) {
    const batch=await Promise.all(directories.slice(offset,offset+12).map(async e => {
    const runDir = join(LOGS, e.name);
    const [metadata, finalState, tools, st] = await Promise.all([
      readJson(join(runDir,'run.json')), readJson(join(runDir,'final_state.json')),
      readdir(join(runDir,'tools')).catch(()=>[]), stat(runDir).catch(()=>null),
    ]);
    const meta=metadata || {};
    const runState=await resolveRunStatus(meta,finalState,activeRuns.get(e.name));
    return {
      run_id: e.name,
      task: meta.task || finalState?.task || '',
      conversation_id:meta.conversation_id || e.name, original_task:meta.original_task || meta.task || '',
      parent_run_id:meta.parent_run_id || null,
      status:runState.status,
      error: finalState?.error || null,
      steps:tools.length,
      started_at:meta.start_time || meta.started_at || null,
      duration_seconds:runState.duration_seconds,
      mtime: st ? st.mtimeMs : 0,
    };
    }));
    runs.push(...batch);
  }
  runs.sort((a, b) => b.run_id.localeCompare(a.run_id));
  return runs;
}

/* ---------------- 配置读写 ---------------- */

// 工具内部自己调模型的组件；这些可以在配置面板里单独指定 profile
const TOOL_COMPONENTS = [
  ['failure_analyzer', '失败分析（①）'],
  ['edit_planner', '编辑规划（④）'],
  ['attack_editor', '攻击车辆编辑（⑤）'],
  ['ego_editor', '专家轨迹编辑（⑥）'],
];

async function getConfig() {
  const [apiJson, agentJson] = await Promise.all([
    readJson(join(CONFIGS, 'api.json')), readJson(join(CONFIGS, 'agent.json')),
  ]);
  if (!apiJson || !agentJson) throw new Error('配置文件缺失或格式错误，请检查 configs/api.json 和 configs/agent.json');
  const profiles = Object.entries(apiJson)
    .filter(([k, v]) => k !== 'active_profile' && v && typeof v === 'object' && v.model && v.base_url)
    .map(([name, v]) => ({ name, model: v.model, base_url: v.base_url, active: name === apiJson.active_profile }));
  const toolProfiles = apiJson.tool_profiles || {};
  return {
    active_profile: apiJson.active_profile || '',
    profiles,
    // 每个工具可以单独选模型；active_profile 表示跟随主 agent
    tool_profiles: Object.fromEntries(TOOL_COMPONENTS.map(([key]) => [key, toolProfiles[key] || 'active_profile'])),
    tool_efforts: Object.fromEntries(TOOL_COMPONENTS.map(([key]) => [key, (apiJson.tool_efforts || {})[key] || 'active_profile'])),
    tool_components: TOOL_COMPONENTS.map(([key, label]) => ({ key, label })),
    // profile = 各模型自己配置的思考程度；low/medium/high = 全局覆盖
    reasoning_effort: apiJson.reasoning_effort || 'profile',
    max_steps: agentJson.agent?.max_steps ?? 20,
    decision_correction_attempts: agentJson.orchestrator?.decision_correction_attempts ?? 1,
    tool_modes: agentJson.tools?.modes || {},
  };
}

async function saveConfig(body) {
  const apiJson = await readJson(join(CONFIGS, 'api.json')) || {};
  const agentJson = await readJson(join(CONFIGS, 'agent.json')) || {};
  const errors = [];
  if (body.active_profile != null) {
    if (apiJson[body.active_profile]?.model) apiJson.active_profile = body.active_profile;
    else errors.push('未知 profile: ' + body.active_profile);
  }
  if (body.max_steps != null) {
    const n = Number(body.max_steps);
    if (Number.isInteger(n) && n >= 1 && n <= 100) (agentJson.agent ||= {}).max_steps = n;
    else errors.push('max_steps 必须是 1-100 的整数');
  }
  if (body.decision_correction_attempts != null) {
    const n = Number(body.decision_correction_attempts);
    if (Number.isInteger(n) && n >= 0 && n <= 5) (agentJson.orchestrator ||= {}).decision_correction_attempts = n;
    else errors.push('决策修正次数必须是 0-5 的整数');
  }
  if (body.reasoning_effort != null) {
    const allowed = ['profile', 'low', 'medium', 'high'];
    if (allowed.includes(body.reasoning_effort)) apiJson.reasoning_effort = body.reasoning_effort;
    else errors.push('思考程度只能是 profile/low/medium/high');
  }
  if (body.tool_profiles && typeof body.tool_profiles === 'object') {
    const known = new Set([...(Object.keys(apiJson).filter(k => apiJson[k] && typeof apiJson[k] === 'object' && apiJson[k].model)), 'active_profile']);
    const target = (apiJson.tool_profiles ||= {});
    for (const { key } of TOOL_COMPONENTS.map(([key]) => ({ key }))) {
      const value = body.tool_profiles[key];
      if (value == null) continue;
      if (!known.has(value)) { errors.push(`工具模型未知: ${key}=${value}`); continue; }
      target[key] = value;
    }
  }
  if (body.tool_efforts && typeof body.tool_efforts === 'object') {
    const allowed = ['active_profile', 'low', 'medium', 'high'];
    const target = (apiJson.tool_efforts ||= {});
    for (const [key] of TOOL_COMPONENTS) {
      const value = body.tool_efforts[key];
      if (value == null) continue;
      if (!allowed.includes(value)) { errors.push(`工具思考程度非法: ${key}=${value}`); continue; }
      target[key] = value;
    }
  }
  if (body.tool_modes && typeof body.tool_modes === 'object') {
    const modes = ((agentJson.tools ||= {}).modes ||= {});
    for (const [k, v] of Object.entries(body.tool_modes)) {
      if (!['mock', 'real'].includes(v)) { errors.push(`工具模式只能是 mock/real: ${k}`); continue; }
      modes[k] = v;
    }
  }
  if (errors.length) return { error: errors.join('；') };
  await writeFile(join(CONFIGS, 'api.json'), JSON.stringify(apiJson, null, 2) + '\n');
  await writeFile(join(CONFIGS, 'agent.json'), JSON.stringify(agentJson, null, 2) + '\n');
  return { ok: true };
}

/* ---------------- HTTP ---------------- */

function sendJson(res, code, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store', 'X-Adagent-Service': 'frontend-api' });
  res.end(body);
}


// 前端构建号：由静态资源的最新 mtime 决定，改动后自动变化
function frontendBuildId() {
  let newest = 0;
  for (const f of ['app.js', 'step-output.js', 'styles.css', 'workbench.js', 'agent-ui.js', 'index.html', 'api-client.js', 'config-panel-controller.mjs', 'panel-loading.mjs', 'conversation.js', 'tool-scroll-spy.js', 'media-preview.js']) {
    try {
      const st = statSync(join(FRONTEND_DIR, f));
      newest = Math.max(newest, st.mtimeMs);
    } catch { /* 文件可能不存在 */ }
  }
  return Math.floor(newest).toString(36);
}

async function serveFile(res, filePath, req) {
  const s = await stat(filePath);
  if (!s.isFile()) { res.writeHead(404).end(); return; }
  const ext = extname(filePath).toLowerCase();
  const headers = { 'Content-Type': MIME[ext] || 'application/octet-stream', 'Accept-Ranges': 'bytes', 'Cache-Control': 'no-cache' };
  let start = 0, end = s.size - 1, code = 200;
  if (req.headers.range) {
    const m = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range);
    if (m && (m[1] || m[2])) {
      if (m[1]) { start = Number(m[1]); end = m[2] ? Math.min(Number(m[2]), end) : end; }
      else start = Math.max(0, s.size - Number(m[2]));
      if (start <= end && start < s.size) {
        code = 206;
        headers['Content-Range'] = `bytes ${start}-${end}/${s.size}`;
      }
    }
  }
  headers['Content-Length'] = code === 206 ? end - start + 1 : s.size;
  res.writeHead(code, headers);
  if (req.method === 'HEAD' || s.size === 0) { res.end(); return; }
  createReadStream(filePath, { start, end }).pipe(res);
}

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://x');
    const path = decodeURIComponent(url.pathname);

    /* ---------- API ---------- */
    if (path === '/api/runs' && req.method === 'GET') {
      return sendJson(res, 200, { runs: await listRuns() });
    }

    if (path === '/api/runs' && req.method === 'POST') {
      let body = '';
      for await (const c of req) body += c;
      let parsed;
      try { parsed = JSON.parse(body || '{}'); } catch { return sendJson(res, 400, { error:'JSON 解析失败' }); }
      try { return sendJson(res, 200, await coordinator.submit(parsed)); }
      catch (error) { return sendJson(res, error.status || 500, { error:error.message }); }
    }

    if (path === '/api/demo' && req.method === 'GET') {
      const file = join(FRONTEND_DIR, 'demo', 'run.json');
      if (!existsSync(file)) return sendJson(res, 404, { error: '示例数据未生成（缺少 demo/run.json）' });
      try {
        return sendJson(res, 200, await readJson(file));
      } catch (e) {
        return sendJson(res, 500, { error: '示例数据读取失败: ' + e.message });
      }
    }

    if (path === '/api/config' && req.method === 'GET') {
      return sendJson(res, 200, await getConfig());
    }
    if (path === '/api/config' && req.method === 'POST') {
      let body = '';
      for await (const c of req) body += c;
      let parsed;
      try { parsed = JSON.parse(body || '{}'); } catch { return sendJson(res, 400, { error: 'JSON 解析失败' }); }
      const result = await saveConfig(parsed);
      return sendJson(res, result.error ? 400 : 200, result);
    }

    const runMatch = /^\/api\/runs\/(run_[\w-]+)(\/stop|\/console)?$/.exec(path);
    if (runMatch) {
      const [, runId, sub] = runMatch;
      if (sub === '/stop' && req.method === 'POST') {
        try { return sendJson(res, 200, await stopRun({ root:ROOT, logsRoot:LOGS, python:PYTHON, runId })); }
        catch (error) { return sendJson(res, error.status || 500, { error:error.message }); }
      }
      if (sub === '/console' && req.method === 'GET') {
        const rec = activeRuns.get(runId);
        return sendJson(res, 200, { lines: rec ? rec.console.slice(-300) : [] });
      }
      if (!sub && req.method === 'GET') {
        const detail = await getRunDetail(runId);
        if (!detail) return sendJson(res, 404, { error: 'run 不存在' });
        return sendJson(res, 200, detail);
      }
    }

    // 项目内任意 mp4：按需转成浏览器可播的 H.264，带磁盘缓存
    if (path === '/api/video' && req.method === 'GET') {
      const src = url.searchParams.get('path') || '';
      if (!await resourceAllowed(src) || !src.endsWith('.mp4')) {
        return sendJson(res, 403, { error: '仅允许访问项目目录内的视频' });
      }
      if (!existsSync(src)) return sendJson(res, 404, { error: '视频不存在' });
      try {
        const playable = await ensurePlayableEightcam(src);
        return await serveFile(res, playable, req);
      } catch (e) {
        return sendJson(res, 500, { error: '转码失败: ' + e.message });
      }
    }
    if (path === '/api/wevideo' && req.method === 'GET') {
      const src = url.searchParams.get('path') || '';
      if (!src.startsWith(WE_ROOT + sep) || !src.endsWith('.mp4')) {
        return sendJson(res, 403, { error: '仅允许访问 WorldEngine 目录内的视频' });
      }
      if (!existsSync(src)) return sendJson(res, 404, { error: '视频不存在' });
      try {
        const playable = await ensurePlayableEightcam(src);
        return await serveFile(res, playable, req);
      } catch (e) {
        return sendJson(res, 500, { error: '转码失败: ' + e.message });
      }
    }
    if (path === '/api/eightcam' && req.method === 'GET') {
      const src = url.searchParams.get('path') || '';
      if (!src.startsWith(CLOSED_LOOP_ROOT + sep) || !src.endsWith('.mp4')) {
        return sendJson(res, 403, { error: '仅允许访问仿真归档内的视频' });
      }
      if (!existsSync(src)) return sendJson(res, 404, { error: '场景视频不存在' });
      try {
        const playable = await ensurePlayableEightcam(src);
        return await serveFile(res, playable, req);
      } catch (e) {
        return sendJson(res, 500, { error: '转码失败: ' + e.message });
      }
    }
    if (path === '/api/file' && req.method === 'GET') {
      const p = url.searchParams.get('path') || '';
      const resolvedPath = resolve(p);
      if (!await resourceAllowed(resolvedPath)) return sendJson(res, 403, { error: '仅允许访问项目及其绑定数据集内文件' });
      return await serveFile(res, resolvedPath, req);
    }

    /* ---------- 静态前端 ---------- */
    const rel = path === '/' ? '/index.html' : path;
    const filePath = resolve(FRONTEND_DIR, '.' + rel);
    if (!filePath.startsWith(FRONTEND_DIR + sep)) { res.writeHead(403).end(); return; }
    if (rel === '/index.html') {
      // 自动带上构建号，避免浏览器拿旧脚本
      const build = frontendBuildId();
      const html = (await readFile(filePath, 'utf8'))
        .replace(/(app\.js\?v=)[\w.-]+/g, `$1${build}`)
        .replace(/(styles\.css\?v=)[\w.-]+/g, `$1${build}`);
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store' });
      res.end(html);
      return;
    }
    return await serveFile(res, filePath, req);
  } catch (error) {
    if (req.url?.includes('/api/')) return sendJson(res, 500, { error:error.message || '接口读取失败' });
    res.writeHead(404).end('Not found');
  }
});

server.listen(PORT, HOST, () => console.log(`adagent console: http://${HOST}:${server.address().port}`));
