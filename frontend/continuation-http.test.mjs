/** Exercise both standalone and production HTTP entries -> real CLI/Agent state. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, rm, readdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';
import { conversationSubmission } from './conversation.js';

const frontend = dirname(fileURLToPath(import.meta.url));
const project = resolve(frontend, '..');

const bootstrap = `import json,os,runpy,sys,time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from agent.config import AgentConfig
from agent.executor import ToolExecutor
root=Path(os.environ['ADAGENT_PROJECT_ROOT'])
config=replace(AgentConfig.load(),logs_root=root/'logs',max_steps=3)
original_executor=ToolExecutor
class OfflineClient:
    model='offline-http-fixture'
    base_url='https://example.invalid'
    def call_functions(self,messages,tools):
        time.sleep(.2)
        context=json.loads(messages[1]['content'])
        state=context['current_state'];history=state['history']
        with (root/'observed-contexts.jsonl').open('a') as handle:
            handle.write(json.dumps(context,ensure_ascii=False)+'\\n')
        if not any(row.get('tool')=='failure_analyzer' for row in history):
            name='failure_analyzer';payload=dict(reason='offline fixture',arguments=dict(token='fixture'))
        else:
            name='finish';payload=dict(reason='offline fixture complete')
        return dict(message=dict(role='assistant',content=None,tool_calls=[
            dict(id='fixture',type='function',function=dict(name=name,arguments=json.dumps(payload)))]),finish_reason='tool_calls')
def executor_factory(**kwargs):
    executor=original_executor(**kwargs,jobs_root=root/'jobs')
    def execute(name,arguments):
        with (root/'executed-tools.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(tool=name,scope=executor.scope))+'\\n')
        return dict(status='success',tool=name,result=dict(failure_analysis=dict(failure_type=['fixture'],failure_reason='offline fixture')))
    executor.execute=execute
    return executor
sys.argv=['agent.main',*sys.argv[3:]]
with patch('agent.config.AgentConfig.load',return_value=config),patch('agent.orchestrator.LLMClient',return_value=OfflineClient()),patch('agent.executor.ToolExecutor',side_effect=executor_factory),patch('agent.prewarm.start',lambda:None):
    runpy.run_module('agent.main',run_name='__main__')
`;

// Retain the standalone fixture to verify the coordinator contract separately.
const fixtureServer = `
import http from 'node:http';
import { readFile, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { createRunCoordinator, conversationDetails } from ${JSON.stringify(join(frontend, 'run-coordinator.mjs'))};
const root=process.env.ADAGENT_PROJECT_ROOT;
const coordinator=createRunCoordinator({root,python:process.env.ADAGENT_AGENT_PYTHON});
async function json(path) {try {return JSON.parse(await readFile(path,'utf8'));}catch{return null;}}
const server=http.createServer(async(req,res)=>{
  const send=(code,body)=>{res.writeHead(code,{'Content-Type':'application/json'});res.end(JSON.stringify(body));};
  try {
    if(req.method==='POST' && req.url==='/api/runs') {
      let body='';for await(const chunk of req)body+=chunk;
      let parsed;try{parsed=JSON.parse(body);}catch{return send(400,{error:'invalid JSON'});}
      return send(200,await coordinator.submit(parsed));
    }
    const match=/^\\/api\\/runs\\/(run_\\d{8}_\\d{6}_\\d+)$/.exec(req.url);
    if(match) {
      const folder=join(root,'logs',match[1]);
      const final=await json(join(folder,'final_state.json'));
      const meta=await json(join(folder,'run.json'));
      const record=coordinator.activeRuns.get(match[1]);
      return send(200,{run_id:match[1],task:meta.task,status:final?.status||'running',
        active:!!record&&record.exitCode==null,steps:await readdir(join(folder,'tools')),
        ...await conversationDetails(folder)});
    }
    send(404,{error:'not found'});
  } catch(error) {send(error.status||500,{error:error.message});}
});
server.listen(0,'127.0.0.1',()=>console.log('adagent console: http://127.0.0.1:'+server.address().port));
`;

for (const entry of ['standalone', 'production']) {
test(`${entry}: HTTP follow-up inherits state; new chat resets; duplicate starts rejected`, { timeout: 40000 }, async t => {
  const root = await mkdtemp(join(tmpdir(), 'adagent-http-continuation-'));
  await mkdir(join(root, 'logs'));
  const launcher = join(root, 'offline-agent');
  await writeFile(launcher, `#!${join(project, 'environments/agent/.venv/bin/python')}\n${bootstrap}`, { mode: 0o700 });
  let output = '';
  const command = entry === 'production' ? [join(frontend, 'server.mjs')] : ['--input-type=module', '-e', fixtureServer];
  const server = spawn(process.execPath, command, {
    cwd: frontend, env: { ...process.env, HOST: '127.0.0.1', PORT: '0',
      ADAGENT_PROJECT_ROOT: root, ADAGENT_AGENT_PYTHON: launcher,
      PYTHONPATH: project + (process.env.PYTHONPATH ? ':' + process.env.PYTHONPATH : '') },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  server.stdout.on('data', data => { output += data; });
  server.stderr.on('data', data => { output += data; });
  t.after(async () => {
    for (const name of await readdir(join(root, 'logs')).catch(() => [])) {
      try {
        const meta = JSON.parse(await readFile(join(root, 'logs', name, 'run.json'), 'utf8'));
        if (!meta.end_time && meta.owner_pid) process.kill(meta.owner_pid, 'SIGTERM');
      } catch { /* stopped or not a run */ }
    }
    server.kill('SIGTERM');
    await new Promise(resolveDone => server.exitCode !== null ? resolveDone() : server.once('exit', resolveDone));
    await rm(root, { recursive: true, force: true });
  });
  for (let i = 0; i < 150 && !output.includes('adagent console:'); i++) {
    if (server.exitCode !== null) throw new Error(output);
    await delay(20);
  }
  const match = /http:\/\/127\.0\.0\.1:(\d+)/.exec(output);
  assert.ok(match, output);
  const base = `http://127.0.0.1:${match[1]}`;
  if (entry === 'production') {
    const app = await (await fetch(base+'/app.js')).text();
    assert.ok(app.includes('submitConversation(api, text, parentRun)'));
    assert.ok(app.includes('createApiClient'));
    assert.equal((await fetch(base+'/conversation.js')).status,200);
    assert.equal((await fetch(base+'/api/config')).status,200);
    assert.equal((await fetch(base+'/api/runs')).headers.get('cache-control'),'no-store');
  }
  const post = body => fetch(base + '/api/runs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const finish = async id => {
    for (let i = 0; i < 150; i++) {
      const detail = await (await fetch(base + '/api/runs/' + id)).json();
      if (detail.status === 'finished' && !detail.active) return detail;
      if (detail.status === 'error') throw new Error(JSON.stringify(detail));
      await delay(40);
    }
    throw new Error('offline run did not finish: ' + output);
  };
  const firstResponse = await post(conversationSubmission('原任务：离线分析一次', null));
  const first = await firstResponse.json();
  assert.equal(firstResponse.status, 200, JSON.stringify(first));
  const duplicate = await post(conversationSubmission('重复提交', null));
  assert.equal(duplicate.status, 409);
  const firstDetail = await finish(first.run_id);
  assert.equal(firstDetail.parent_run_id, null);
  const source = await readFile(join(root, 'logs', first.run_id, 'final_state.json'), 'utf8');
  const secondResponse = await post(conversationSubmission('继续解释，不重复分析', first.run_id));
  const second = await secondResponse.json();
  assert.equal(secondResponse.status, 200, JSON.stringify(second));
  const secondDetail = await finish(second.run_id);
  assert.equal(secondDetail.parent_run_id, first.run_id);
  assert.equal(secondDetail.conversation_id, first.run_id);
  assert.equal(secondDetail.original_task, '原任务：离线分析一次');
  assert.equal(secondDetail.continuation_supported, true);
  assert.equal(secondDetail.user_messages[0].content, '继续解释，不重复分析');
  assert.equal(secondDetail.steps.length, 0);
  assert.equal(await readFile(join(root, 'logs', first.run_id, 'final_state.json'), 'utf8'), source);
  const contexts = (await readFile(join(root, 'observed-contexts.jsonl'), 'utf8')).trim().split('\n').map(JSON.parse);
  const followup = contexts.find(c => c.current_state.current_request === '继续解释，不重复分析');
  assert.equal(followup.task, '原任务：离线分析一次');
  assert.ok(followup.current_state.history.some(e => e.tool === 'failure_analyzer'));
  assert.equal((await readFile(join(root, 'executed-tools.jsonl'), 'utf8')).trim().split('\n').length, 1);
  const thirdResponse = await post(conversationSubmission('新任务：从头离线分析', null));
  const third = await thirdResponse.json();
  assert.equal(thirdResponse.status, 200, JSON.stringify(third));
  const thirdDetail = await finish(third.run_id);
  assert.equal(thirdDetail.parent_run_id, null);
  assert.equal(thirdDetail.conversation_id, third.run_id);
  assert.deepEqual(thirdDetail.user_messages, []);
  assert.equal(thirdDetail.steps.length, 1);
  const invalid = await post({ task: '继续', resume_from: '../outside' });
  assert.equal(invalid.status, 400);
  const brokenJSON = await fetch(base + '/api/runs', { method: 'POST', body: '{broken' });
  assert.equal(brokenJSON.status, 400);
});
}
