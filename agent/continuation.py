"""Recover a decision state from immutable parent logs, including old runs."""
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import re

from agent.state import AgentState

RUN_ID = re.compile(r'run_\d{8}_\d{6}_\d+')


def ensure_unlocked(lease):
    if not lease.exists():
        return
    with lease.open('r') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('该对话仍在执行，请等待完成或先停止，再发送续跑消息') from exc
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def load_json(path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding='utf-8'))


def recover(logs_root, run_id, message):
    """Return (state, lineage, pending_call); never mutate a parent artifact."""
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
        raise ValueError('Invalid continuation run ID')
    root = Path(logs_root).resolve()
    folder = (root/run_id).resolve()
    if folder.parent != root or not folder.is_dir():
        raise ValueError('Continuation run does not exist inside the log directory')
    metadata = load_json(folder/'run.json')
    if not isinstance(metadata, dict) or not metadata.get('task'):
        raise ValueError('Continuation run has no valid task metadata')
    conversation = metadata.get('conversation_id') or run_id
    if not isinstance(conversation,str) or not RUN_ID.fullmatch(conversation):
        raise ValueError('Invalid conversation ID in parent metadata')
    ensure_unlocked(folder/'run.lock')
    ensure_unlocked(root/'.conversation_locks'/(conversation+'.lock'))
    saved = load_json(folder/'final_state.json')
    if saved is None:
        saved = load_json(folder/'state.json', {})
    if not isinstance(saved, dict):
        raise ValueError('Continuation checkpoint is not an object')
    state = AgentState(saved.get('task') or metadata.get('original_task') or metadata['task'])
    history = deepcopy(saved.get('history', []))
    if not isinstance(history, list) or any(not isinstance(entry, dict) for entry in history):
        raise ValueError('Continuation checkpoint history is malformed')
    step = 0
    for entry in history:
        if 'decision' in entry:
            step = entry.get('step', step)
        entry.setdefault('step', step)
    decisions = {entry['step'] for entry in history if 'decision' in entry}
    results = {entry['step'] for entry in history if 'tool' in entry}
    pending = None
    # A process may die after writing tool files but before its next checkpoint.
    for tool_dir in sorted((folder/'tools').glob('step_*')):
        match = re.fullmatch(r'step_(\d+)_(.+)', tool_dir.name)
        if not match:
            continue
        index, tool = int(match[1]), match[2]
        arguments = load_json(tool_dir/'input.json')
        if not isinstance(arguments, dict):
            continue
        if index not in decisions:
            history.append(dict(step=index, decision=dict(action='call_tool', tool_name=tool,
                arguments=arguments, reason='Recovered recorded tool call from the parent run.')))
            decisions.add(index)
        output = load_json(tool_dir/'output.json')
        if isinstance(output, dict) and index not in results:
            history.append(dict(step=index, tool=tool, result=output,
                                result_ref=str((tool_dir/'output.json').resolve())))
            results.add(index)
        if output is None and index not in results:
            pending = dict(step=index, tool_name=tool, arguments=arguments)
    history.sort(key=lambda entry: (entry.get('step', 0), 'tool' in entry))
    state.history = history
    state.current_step = max([saved.get('current_step', -1), *[e['step'] for e in history]], default=-1)
    state.current_stage = saved.get('current_stage', 'planning')
    state.user_messages = deepcopy(saved.get('user_messages', []))
    state.user_messages.append(dict(content=message, after_run_id=run_id))
    state.current_request = message
    state.execution_scope = saved.get('execution_scope') or metadata.get('execution_scope') or run_id
    state.resume_context = dict(parent_run_id=run_id, previous_status=saved.get('status', metadata.get('status')),
                                previous_error=saved.get('error'), inherited_history_count=len(history))
    state.pending_job = deepcopy(saved.get('pending_job')) if pending else None
    state.background_jobs = deepcopy(saved.get('background_jobs',{}))
    lineage = dict(parent_run_id=run_id,
                   conversation_id=conversation,
                   execution_scope=state.execution_scope,
                   original_task=state.task,
                   inherited_history_count=len(history))
    return state, lineage, pending
