/** Exercise persisted controls with detached offline processes, never a real Agent. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, mkdir, writeFile, readFile, rm } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { setTimeout as delay } from 'node:timers/promises';
import { runIsAlive, prepareRunRequest } from './run-continuation.mjs';

const frontend=dirname(fileURLToPath(import.meta.url)), project=dirname(frontend);
async function identity(pid) {
  const stat=await readFile(`/proc/${pid}/stat`,'utf8');
  return stat.slice(stat.lastIndexOf(')')+1).trim().split(/\s+/)[19];
}
test('restarted server stops the selected run and detached tool groups, preserving other scopes and completed results', {timeout:20000}, async t=>{
  const root=await mkdtemp(join(tmpdir(),'adagent-stop-http-')), logs=join(root,'logs');
  const id='run_20261115_000000_000001', folder=join(logs,id);
  await mkdir(folder,{recursive:true});
  const children=[];
  const live=()=>{const proc=spawn(process.execPath,['-e','setInterval(()=>{},1000)'],{cwd:root,detached:true,stdio:'ignore'});children.push(proc);return proc;};
  t.after(async()=>{for(const child of children){try{process.kill(-child.pid,'SIGKILL');}catch{try{child.kill('SIGKILL');}catch{}}}await rm(root,{recursive:true,force:true});});
  const owner=live(), worker=live(), child=live(), other=live();
  const workerStart=await identity(worker.pid), childStart=await identity(child.pid), otherStart=await identity(other.pid);
  const meta={status:'running',task:'offline stop fixture',start_time:new Date().toISOString(),owner_pid:owner.pid,owner_process_start:await identity(owner.pid),execution_scope:id};
  await writeFile(join(folder,'run.json'),JSON.stringify(meta));await writeFile(join(folder,'state.json'),JSON.stringify({status:'running',history:[]}));
  const job='a'.repeat(32), completed='b'.repeat(32), foreign='c'.repeat(32);
  for(const [name,scope,proc,status] of [[job,id,worker,'running'],[completed,id,null,'completed'],[foreign,'other-scope',other,'running']]){
    const dir=join(logs,'jobs',name);await mkdir(dir,{recursive:true});
    await writeFile(join(dir,'manifest.json'),JSON.stringify({status,tool:'attack_vehicle_editor',scope,pid:proc?.pid,process_start:proc?await identity(proc.pid):null}));
  }
  await writeFile(join(logs,'jobs',job,'children.json'),JSON.stringify([{pid:child.pid,process_start:await identity(child.pid)}]));
  const completedText='{"status":"success","result":{"candidate":"preserved"}}';
  await writeFile(join(logs,'jobs',completed,'result.json'),completedText);
  const server=spawn(process.execPath,[join(frontend,'server.mjs')],{cwd:frontend,detached:true,env:{...process.env,PORT:'0',HOST:'127.0.0.1',ADAGENT_PROJECT_ROOT:root,ADAGENT_AGENT_PYTHON:join(project,'environments/agent/.venv/bin/python'),PYTHONPATH:project},stdio:['ignore','pipe','pipe']});children.push(server);
  let output='';server.stdout.on('data',data=>{output+=data;});server.stderr.on('data',data=>{output+=data;});
  for(let i=0;i<100&&!/http:\/\/127\.0\.0\.1:\d+/.test(output);i++)await delay(20);
  const base=output.match(/http:\/\/127\.0\.0\.1:\d+/)?.[0];assert.ok(base,output);
  assert.equal((await (await fetch(`${base}/api/runs/${id}`)).json()).active,true);
  const response=await fetch(`${base}/api/runs/${id}/stop`,{method:'POST'});
  assert.equal(response.status,200,await response.clone().text());
  assert.equal(await runIsAlive(meta),false);
  assert.equal(await runIsAlive({owner_pid:worker.pid,owner_process_start:workerStart}),false);
  const detail=await (await fetch(`${base}/api/runs/${id}`)).json();assert.equal(detail.status,'interrupted');assert.equal(detail.active,false);
  assert.equal((await readFile(join(logs,'jobs',job,'result.json'),'utf8').then(JSON.parse)).error.type,'job_interrupted');
  assert.equal(await readFile(join(logs,'jobs',completed,'result.json'),'utf8'),completedText);
  assert.equal(await runIsAlive({owner_pid:other.pid,owner_process_start:otherStart}),true);
  assert.equal(await runIsAlive({owner_pid:child.pid,owner_process_start:childStart}),false);
  assert.equal((await prepareRunRequest({task:'continue',resume_from:id},logs)).resume_from,id);
  assert.equal((await fetch(`${base}/api/runs/${id}/stop`,{method:'POST'})).status,200);
});
