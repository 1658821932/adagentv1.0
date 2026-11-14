/** Shared validation for conversation submissions; no API/model side effects. */
import { readFile, realpath } from 'node:fs/promises';
import { join, dirname } from 'node:path';

export async function runIsAlive(meta) {
  if (!Number.isInteger(meta?.owner_pid) || !meta?.owner_process_start || meta.end_time) return false;
  try {
    const value = await readFile(`/proc/${meta.owner_pid}/stat`, 'utf8');
    const fields = value.slice(value.lastIndexOf(')') + 1).trim().split(/\s+/);
    return fields[0] !== 'Z' && fields[19] === String(meta.owner_process_start);
  } catch { return false; }
}

export async function prepareRunRequest(body, logsRoot) {
  if (!body || typeof body.task !== 'string' || !body.task.trim()) {
    throw new Error('任务内容不能为空');
  }
  const resumeFrom = body.resume_from ?? null;
  if (resumeFrom !== null) {
    if (typeof resumeFrom !== 'string' || !/^run_\d{8}_\d{6}_\d+$/.test(resumeFrom)) {
      throw new Error('续跑编号无效，请从历史运行选择任务');
    }
    let folder, metadata;
    try {
      folder = await realpath(join(logsRoot, resumeFrom));
      if (dirname(folder) !== await realpath(logsRoot)) throw new Error('outside logs');
      metadata = JSON.parse(await readFile(join(folder, 'run.json'), 'utf8'));
      if (!metadata.task) throw new Error('missing task');
    } catch { throw new Error('找不到可续跑的历史任务'); }
    if (await runIsAlive(metadata)) throw new Error('该任务仍在执行，请等待完成或先停止');
  }
  return { task: body.task.trim(), resume_from: resumeFrom };
}

export function agentArguments(task, resumeFrom = null) {
  return ['-m', 'agent.main', ...(resumeFrom ? ['--resume-from', resumeFrom] : []), '--', task];
}
