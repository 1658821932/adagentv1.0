/** A follow-up belongs to the selected run until the user starts a new chat. */
export function conversationSubmission(task, viewingRun) {
  return { task: task.trim(), ...(viewingRun ? { resume_from: viewingRun } : {}) };
}

export function groupConversationRuns(runs) {
  const groups = new Map();
  for (const run of runs) {
    const key = run.conversation_id || run.run_id;
    let group = groups.get(key);
    if (!group) {
      group = { ...run, task: run.original_task || run.task, conversation_rounds: 0, conversation_steps: 0 };
      groups.set(key, group);
    }
    group.conversation_rounds += 1;
    group.conversation_steps += Number(run.steps) || 0;
    if (String(run.run_id) > String(group.run_id)) {
      const title = group.task;
      const rounds = group.conversation_rounds;
      const steps = group.conversation_steps;
      Object.assign(group, run, {
        task: run.original_task || title,
        conversation_rounds: rounds,
        conversation_steps: steps,
      });
    }
  }
  return [...groups.values()].sort((a, b) => String(b.run_id).localeCompare(String(a.run_id)));
}

/** Never silently launch a fresh task against a server that ignores resume_from. */
export async function submitConversation(request, task, viewingRun) {
  const body = conversationSubmission(task, viewingRun);
  if (viewingRun) {
    const parent = await request(`/api/runs/${viewingRun}`);
    if (!parent.continuation_supported) {
      throw new Error('当前服务尚未接入对话续跑；未提交新任务，请等待入口更新');
    }
  }
  return request('/api/runs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
}
