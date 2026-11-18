/** Resolve persistent jobs and group submission/collection into one tool card. */
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { runIsAlive } from './run-continuation.mjs';

async function json(path) {
  try { return JSON.parse(await readFile(path, 'utf8')); } catch { return null; }
}
const pending = new Set(['queued','starting','running','orphaned']);

async function jobFor(id, tool, logsRoot, scope, now) {
  if (!/^[a-f0-9]{32}$/.test(id || '')) return null;
  const folder = join(logsRoot, 'jobs', id);
  const manifest = await json(join(folder,'manifest.json'));
  if (!manifest || manifest.tool !== tool || manifest.scope !== scope) return null;
  const result = await json(join(folder,'result.json'));
  let active = false;
  if (!result && pending.has(manifest.status)) {
    active = await runIsAlive({ owner_pid:manifest.pid, owner_process_start:manifest.process_start });
    if (!active) {
      const children = await json(join(folder,'children.json'));
      for (const child of Array.isArray(children) ? children : []) {
        if (await runIsAlive({ owner_pid:child.pid, owner_process_start:child.process_start })) { active = true; break; }
      }
    }
  }
  const started = typeof manifest.started_at === 'number' ? manifest.started_at : null;
  const finished = typeof manifest.finished_at === 'number' ? manifest.finished_at : null;
  const elapsed = started != null && (finished != null || active)
    ? Math.max(0, (finished ?? now / 1000) - started) : null;
  const status = result?.status || (active ? 'running' : 'interrupted');
  return {
    id, active, status, started_at:started == null ? null : new Date(started * 1000).toISOString(),
    finished_at:finished == null ? null : new Date(finished * 1000).toISOString(),
    elapsed_seconds:elapsed,
    output:result || (active ? {status:'running',tool,job:{job_id:id,state:manifest.status}}
      : {status:'error',tool,error:{type:'job_interrupted',message:'任务进程已退出，未返回结果；已保留日志和产物。'},job:{job_id:id,state:'interrupted'}}),
    manifest_path:join(folder,'manifest.json'),
  };
}

export async function resolveJobSteps({ steps, logsRoot, scope, now = Date.now() }) {
  const cache = new Map(), groups = new Map(), resolved = [];
  for (const step of steps) {
    const id = step.output?.job?.job_id || step.input?._execution?.job_id;
    const cacheKey = `${step.tool}:${id}`;
    if (id && !cache.has(cacheKey)) cache.set(cacheKey, jobFor(id,step.tool,logsRoot,scope,now));
    const job = id ? await cache.get(cacheKey) : null;
    if (!job) { resolved.push(step); continue; }
    const key = `${step.tool}:${id}`;
    const existing = groups.get(key);
    if (existing) {
      if (step.output?.status === job.status && ['success','error'].includes(job.status)) existing.output = step.output;
      existing.dispatch_steps.push(step.step);
      existing.dispatch_refs.push({step:step.step,decision:step.decision,input:step.input});
      const ids = new Set(existing.activities.map(item=>item.id));
      existing.activities.push(...(step.activities || []).filter(item=>!ids.has(item.id)));
      if (typeof step.llm?.elapsed_seconds === 'number')
        existing.llm = {...existing.llm,elapsed_seconds:(existing.llm?.elapsed_seconds || 0)+step.llm.elapsed_seconds};
      continue;
    }
    const value = {
      ...step, job_id:id, job, status:job.status,
      output:step.output?.status === job.status && ['success','error'].includes(job.status) ? step.output : job.output,
      duration:job.active ? null : job.elapsed_seconds,
      elapsed_seconds:job.active ? job.elapsed_seconds : null,
      activities:[...(step.activities || [])],
      dispatch_steps:[step.step],dispatch_refs:[{step:step.step,decision:step.decision,input:step.input}],
    };
    groups.set(key,value);resolved.push(value);
  }
  const active_jobs = [...groups.values()].filter(step=>step.job.active).map(step=>({
    job_id:step.job_id,tool:step.tool,step:step.step,elapsed_seconds:step.elapsed_seconds,
  }));
  return { steps:resolved,active_jobs,dispatch_count:steps.length };
}
