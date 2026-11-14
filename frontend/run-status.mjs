/** Resolve display state from persisted process identity, without rewriting logs. */
import { runIsAlive } from './run-continuation.mjs';

const TERMINAL = new Set(['finished', 'success', 'error', 'interrupted', 'stopped', 'incomplete']);

export async function resolveRunStatus(meta = {}, finalState = null, active = null) {
  const recorded = finalState?.status || meta.status;
  const localAlive = active?.exitCode == null && active?.proc?.exitCode === null && !active.proc.killed;
  const alive = !TERMINAL.has(recorded) && !meta.end_time
    && (await runIsAlive(meta) || localAlive);
  const status = TERMINAL.has(recorded) ? recorded : alive ? 'running'
    : (meta.start_time || meta.owner_pid || active || recorded === 'running') ? 'interrupted' : 'incomplete';
  const start = Date.parse(meta.start_time || meta.started_at || meta.created_at);
  const end = Date.parse(meta.end_time);
  const duration = Number.isFinite(start) && (Number.isFinite(end) || alive)
    ? Math.max(0, ((Number.isFinite(end) ? end : Date.now()) - start) / 1000) : null;
  return { status, active: Boolean(alive), duration_seconds: duration };
}
