/** Independent POST /api/runs implementation, ready for the shared server. */
import { spawn } from 'node:child_process';
import { readFile, readdir } from 'node:fs/promises';
import { basename, join } from 'node:path';
import { prepareRunRequest, agentArguments, runIsAlive } from './run-continuation.mjs';

async function json(path) {
  try { return JSON.parse(await readFile(path, 'utf8')); } catch { return null; }
}

function error(status, message) {
  return Object.assign(new Error(message), { status });
}

export async function conversationDetails(runDir) {
  const meta = await json(join(runDir, 'run.json')) || {};
  const state = await json(join(runDir, 'final_state.json')) || await json(join(runDir, 'state.json')) || {};
  return {
    continuation_supported: true,
    original_task: meta.original_task || state.task || meta.task || '',
    parent_run_id: meta.parent_run_id || null,
    conversation_id: meta.conversation_id || basename(runDir),
    user_messages: state.user_messages || [],
  };
}

export function createRunCoordinator({ root, logsRoot = join(root, 'logs'), python,
  activeRuns = new Map(), env = process.env, startupTimeoutMs = 60000 }) {
  const starting = new Set();
  let accepting = false;

  async function persistedActive() {
    for (const name of await readdir(logsRoot).catch(() => [])) {
      if (!/^run_\d{8}_\d{6}_\d+$/.test(name)) continue;
      if (await runIsAlive(await json(join(logsRoot, name, 'run.json')))) return name;
    }
    return null;
  }

  function launch(task, resumeFrom) {
    return new Promise((resolve, reject) => {
      const proc = spawn(python, agentArguments(task, resumeFrom), {
        cwd: root, detached: true, env: { ...env, PYTHONUNBUFFERED: '1' },
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      const record = { proc, task, startedAt: Date.now(), console: [], exitCode: null };
      starting.add(record);
      let output = '', settled = false;
      const timer = setTimeout(() => {
        if (!settled) {
          settled = true;
          reject(error(504, '等待运行编号超时；进程仍在后台，请查看历史，避免重复提交'));
        }
      }, startupTimeoutMs);
      const lines = data => {
        record.console.push(...data.toString('utf8').split('\n'));
        if (record.console.length > 3000) record.console.splice(0, record.console.length - 3000);
      };
      proc.stdout.on('data', data => {
        lines(data);
        output += data.toString('utf8');
        const match = /Logs:\s*\n\s*(\S+)/.exec(output);
        if (match && !record.runId) {
          record.runId = basename(match[1]);
          activeRuns.set(record.runId, record);
        }
        if (match && !settled) {
          settled = true;
          clearTimeout(timer);
          resolve({ run_id: record.runId, parent_run_id: resumeFrom });
        }
      });
      proc.stderr.on('data', lines);
      proc.on('error', failure => {
        starting.delete(record);
        record.exitCode = -1;
        clearTimeout(timer);
        if (!settled) { settled = true; reject(failure); }
      });
      proc.on('exit', code => {
        record.exitCode = code ?? -1;
        starting.delete(record);
        clearTimeout(timer);
        if (!settled) {
          settled = true;
          reject(error(500, '主 Agent 启动失败：' + record.console.slice(-15).join('\n')));
        }
      });
    });
  }

  return {
    activeRuns,
    async submit(body) {
      const busy = [...activeRuns.values(), ...starting].find(r => r.exitCode == null);
      if (accepting || busy) throw error(409, '已有任务执行中，请等待完成或先停止');
      accepting = true;
      try {
        let request;
        try { request = await prepareRunRequest(body, logsRoot); }
        catch (failure) { throw error(400, failure.message); }
        if (await persistedActive()) throw error(409, '已有主 Agent 在后台执行，请等待完成或先停止');
        return await launch(request.task, request.resume_from);
      } finally { accepting = false; }
    },
  };
}
