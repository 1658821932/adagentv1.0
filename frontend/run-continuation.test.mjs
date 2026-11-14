import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, rm, symlink, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { prepareRunRequest, agentArguments, runIsAlive } from './run-continuation.mjs';
import { conversationSubmission, submitConversation, groupConversationRuns } from './conversation.js';

const parent = 'run_20261112_060801_663582';

test('follow-up includes selected parent; new conversation has no parent', () => {
  assert.deepEqual(conversationSubmission('  继续  ', parent), { task: '继续', resume_from: parent });
  assert.deepEqual(conversationSubmission('新任务', null), { task: '新任务' });
});

test('history groups all runs in one conversation under its latest status', () => {
  const child = 'run_20261112_103911_730785';
  const groups = groupConversationRuns([
    { run_id: child, task: '请继续', original_task: '原始全流程', conversation_id: parent, status: 'error', steps: 3 },
    { run_id: 'run_20261112_110000_000001', task: '另一个任务', conversation_id: 'run_20261112_110000_000001', status: 'running', steps: 1 },
    { run_id: parent, task: '原始全流程', conversation_id: parent, status: 'interrupted', steps: 11 },
  ]);
  assert.equal(groups.length, 2);
  assert.equal(groups[1].run_id, child);
  assert.equal(groups[1].task, '原始全流程');
  assert.equal(groups[1].status, 'error');
  assert.equal(groups[1].conversation_rounds, 2);
  assert.equal(groups[1].conversation_steps, 14);
});

test('old server cannot silently turn a follow-up into an unrelated fresh task', async () => {
  const calls=[];
  const oldServer=async(path,options)=>{calls.push({path,options});return {task:'fixture'};};
  await assert.rejects(submitConversation(oldServer,'继续',parent),/尚未接入/);
  assert.equal(calls.length,1);
  assert.equal(calls[0].options,undefined);
  const modern=[];
  const newServer=async(path,options)=>{modern.push({path,options});return {continuation_supported:true};};
  await submitConversation(newServer,'继续',parent);
  assert.equal(modern.length,2);
  assert.equal(JSON.parse(modern[1].options.body).resume_from,parent);
  modern.length=0;
  await submitConversation(newServer,'全新任务',null);
  assert.equal(modern.length,1);
  assert.deepEqual(JSON.parse(modern[0].options.body),{task:'全新任务'});
});

test('CLI flags preserve the exact follow-up text and cannot become extra options', () => {
  const text = '--resume-from fake; $(untrusted)\n继续';
  assert.deepEqual(agentArguments(text, parent), ['-m', 'agent.main', '--resume-from', parent, '--', text]);
  assert.deepEqual(agentArguments(text), ['-m', 'agent.main', '--', text]);
});

test('valid history can be continued, malformed or missing parents are rejected', async () => {
  const root = await mkdtemp(join(tmpdir(), 'adagent-continuation-'));
  try {
    await mkdir(join(root, parent));
    await writeFile(join(root, parent, 'run.json'), JSON.stringify({ task: 'original task' }));
    assert.deepEqual(await prepareRunRequest({ task: '继续', resume_from: parent }, root), { task: '继续', resume_from: parent });
    assert.deepEqual(await prepareRunRequest({ task: 'new' }, root), { task: 'new', resume_from: null });
    for (const value of ['../elsewhere', '/tmp/run_x', 'run_missing', 42]) {
      await assert.rejects(prepareRunRequest({ task: '继续', resume_from: value }, root));
    }
    await assert.rejects(prepareRunRequest({ task: '继续', resume_from: 'run_20200101_000000_1' }, root));
    await assert.rejects(prepareRunRequest({ task: '  ' }, root));
    const outside = await mkdtemp(join(tmpdir(), 'adagent-outside-'));
    try {
      await writeFile(join(outside, 'run.json'), JSON.stringify({ task: 'outside' }));
      await symlink(outside, join(root, 'run_20200101_000000_2'));
      await assert.rejects(prepareRunRequest({ task: '继续', resume_from: 'run_20200101_000000_2' }, root));
    } finally { await rm(outside, { recursive: true }); }
  } finally { await rm(root, { recursive: true }); }
});

test('a live parent is recognized even after frontend restart', async () => {
  const root = await mkdtemp(join(tmpdir(), 'adagent-live-'));
  const value = await readFile(`/proc/${process.pid}/stat`, 'utf8');
  const start = value.slice(value.lastIndexOf(')') + 1).trim().split(/\s+/)[19];
  const metadata = { task: 'fixture', owner_pid: process.pid, owner_process_start: start };
  try {
    await mkdir(join(root, parent));
    await writeFile(join(root, parent, 'run.json'), JSON.stringify(metadata));
    assert.equal(await runIsAlive(metadata), true);
    assert.equal(await runIsAlive({ ...metadata, owner_process_start: 'wrong' }), false);
    await assert.rejects(prepareRunRequest({ task: '继续', resume_from: parent }, root), /仍在执行/);
    assert.equal(await runIsAlive({ ...metadata, end_time: 'ended' }), false);
  } finally { await rm(root, { recursive: true }); }
});
