/** Stop a persisted run even after the frontend server has restarted. */
import { readFile, writeFile, rename, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { setTimeout as delay } from 'node:timers/promises';
import { runIsAlive } from './run-continuation.mjs';

const execute = promisify(execFile);
async function json(path) { try { return JSON.parse(await readFile(path, 'utf8')); } catch { return null; } }
async function save(path, value) {
  const temporary = `${path}.${process.pid}.tmp`;
  await writeFile(temporary, JSON.stringify(value, null, 2) + '\n', { mode:0o600 });
  await rename(temporary, path);
}

export async function stopRun({ root, logsRoot, python, runId }) {
  if (!/^run_\d{8}_\d{6}_\d+$/.test(runId)) throw Object.assign(new Error('运行编号无效'), { status:400 });
  const folder = join(logsRoot, runId), path = join(folder, 'run.json');
  const meta = await json(path);
  if (!meta) throw Object.assign(new Error('找不到该运行'), { status:404 });
  const scope = meta.execution_scope || runId;
  if (await runIsAlive(meta)) {
    const signal = async name => {
      if (!await runIsAlive(meta)) return;
      const stat = await readFile(`/proc/${meta.owner_pid}/stat`, 'utf8');
      const fields = stat.slice(stat.lastIndexOf(')') + 1).trim().split(/\s+/);
      try { process.kill(Number(fields[2]) === meta.owner_pid ? -meta.owner_pid : meta.owner_pid, name); }
      catch (error) { if (error.code !== 'ESRCH') throw error; }
    };
    await signal('SIGTERM');
    for (let i=0; i<40 && await runIsAlive(meta); i++) await delay(50);
    await signal('SIGKILL');
    for (let i=0; i<40 && await runIsAlive(meta); i++) await delay(50);
    if (await runIsAlive(meta)) throw new Error('主进程尚未退出，请稍后重试停止');
  }
  // Tool workers start their own sessions, so killing the main group is insufficient.
  const jobsRoot = join(logsRoot, 'jobs');
  const jobs = await readdir(jobsRoot).catch(() => []);
  const scoped = await Promise.all(jobs.filter(name => /^[a-f0-9]{32}$/.test(name))
    .map(name => json(join(jobsRoot,name,'manifest.json'))));
  const hadBackgroundJobs = scoped.some(job => job?.scope === scope && ['queued','starting','running','orphaned'].includes(job.status));
  if (jobs.some(name => /^[a-f0-9]{32}$/.test(name))) {
    await execute(python, ['-c', 'import sys; from agent.jobs import JobStore; JobStore(sys.argv[1]).stop_scope(sys.argv[2])', jobsRoot, scope],
      { cwd:root, timeout:30000, maxBuffer:1024*1024 });
  }
  const now = new Date().toISOString();
  const latest = await json(path) || meta;
  if (hadBackgroundJobs || !latest.end_time && !['finished','success','error','stopped'].includes(latest.status)) {
    await save(path, { ...latest, previous_status:latest.status, status:'interrupted', end_time:now, stop_reason:'user_requested', stopped_at:now });
    const statePath = join(folder, 'state.json'), state = await json(statePath);
    if (state && (hadBackgroundJobs || !['finished','success','error'].includes(state.status)))
      await save(statePath, { ...state, status:'interrupted' });
  }
  return { ok:true, status:'interrupted' };
}
