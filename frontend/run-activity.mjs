import { readdir, readFile, stat } from 'node:fs/promises';
import { basename, join } from 'node:path';

const VISIBLE_OPERATIONS = new Set([
  'attack.object_selection', 'attack.inspect', 'attack.submit', 'attack.materialize_added', 'attack.independent_check',
  'ego.understanding', 'ego.constructor', 'ego.foci', 'ego.visual_agent', 'ego.select_constructor',
  'ego.edit', 'ego.submit', 'ego.independent_check',
]);

async function readJson(path) {
  try { return JSON.parse(await readFile(path, 'utf8')); }
  catch { return null; }
}

function activitySummary(record, kind) {
  const result = record.result || {};
  if (Array.isArray(result.candidates)) {
    const ids = result.candidates.map(item => item?.candidate_id).filter(Boolean).slice(0, 4);
    return `生成 ${result.candidates.length} 个候选${ids.length ? '：' + ids.join('、') : ''}`;
  }
  if (result.candidate_id) return `${result.candidate_id} · ${result.status || '已返回'}`;
  if (result.target_track) return `目标 ${result.target_track} · ${result.status || '已返回'}`;
  if (result.status) return String(result.status);
  return kind === 'operation' ? (record.result_status || record.status || '') : '';
}

function runtimeReference(value, marker) {
  if (typeof value !== 'string') return null;
  const index = value.indexOf(marker);
  if (index < 0) return null;
  return { root: value.slice(0, index), path: value };
}

async function supplementToolExchanges(root, step, activities) {
  for (const relative of ['output/attack/trajectory_editing/tool_calls', 'output/defense_agent/tool-exchanges']) {
    const directory = join(root, relative);
    const entries = await readdir(directory, { withFileTypes: true }).catch(() => []);
    for (const entry of entries) {
      if (!entry.isDirectory() || !/^turn_\d+$/.test(entry.name)) continue;
      const folder = join(directory, entry.name);
      const archiveRef = join(folder);
      if (activities.some(item => item.archive_ref === archiveRef)) continue;
      const call = await readJson(join(folder, 'call.json'));
      const result = await readJson(join(folder, 'result.json'));
      const name = call?.function?.name;
      if (!name) continue;
      const info = await stat(join(folder, result ? 'result.json' : 'call.json')).catch(() => null);
      activities.push({
        id: `tool:runtime:${archiveRef}`,
        kind: 'tool', step, name,
        status: result?.status || 'running',
        timestamp: info ? info.mtime.toISOString() : null,
        finished_at: result ? info?.mtime.toISOString() || null : null,
        elapsed_seconds: null,
        summary: result ? activitySummary({ result }, 'tool') : '执行中',
        archive_ref: archiveRef,
      });
    }
  }
}

export async function readSubagentActivity(runDir) {
  const activities = [];
  const runtimes = new Map();
  for (const [folder, kind] of [['operations', 'operation'], ['tool_events', 'tool']]) {
    const entries = await readdir(join(runDir, folder), { withFileTypes: true }).catch(() => []);
    for (const entry of entries) {
      if (!entry.isFile() || !entry.name.endsWith('.json')) continue;
      const record = await readJson(join(runDir, folder, entry.name));
      const step = Number(record?.context?.step);
      if (!Number.isInteger(step)) continue;
      if (kind === 'operation' && !VISIBLE_OPERATIONS.has(record.name)) continue;
      if (kind === 'tool' && !record.tool_name) continue;
      activities.push({
        id: `${kind}:${record.event_id || basename(entry.name, '.json')}`,
        kind,
        step,
        name: kind === 'operation' ? record.name : record.tool_name,
        status: kind === 'operation' ? record.status : (record.result?.status || 'completed'),
        timestamp: record.started_at || record.timestamp || null,
        finished_at: record.finished_at || null,
        elapsed_seconds: typeof record.elapsed_seconds === 'number' ? record.elapsed_seconds : null,
        summary: activitySummary(record, kind),
        archive_ref: record.archive_ref || null,
      });
      const reference = runtimeReference(record.archive_ref, '/output/');
      if (reference && !runtimes.has(reference.root)) runtimes.set(reference.root, step);
    }
  }
  for (const [root, step] of runtimes) await supplementToolExchanges(root, step, activities);
  return activities.sort((a, b) => String(a.timestamp).localeCompare(String(b.timestamp)));
}
