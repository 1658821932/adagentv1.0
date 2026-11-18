import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp,mkdir,writeFile,readFile,rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { resolveJobSteps } from './run-jobs.mjs';

async function fixture(t) {
  const root=await mkdtemp(join(tmpdir(),'adagent-parallel-ui-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  const stat=await readFile(`/proc/${process.pid}/stat`,'utf8');
  const start=stat.slice(stat.lastIndexOf(')')+1).trim().split(/\s+/)[19];
  const now=Date.now(),scope='run_parallel';
  async function job(id,tool,extra={},output=null) {
    const dir=join(root,'jobs',id);await mkdir(dir,{recursive:true});
    await writeFile(join(dir,'manifest.json'),JSON.stringify({tool,scope,status:'running',
      pid:process.pid,process_start:start,started_at:now/1000-30,...extra}));
    if (output) await writeFile(join(dir,'result.json'),JSON.stringify(output));
    return dir;
  }
  return {root,now,scope,start,job};
}
const a='a'.repeat(32),b='b'.repeat(32);
const step=(index,tool,id)=>({step:index,tool,status:'running',duration:.01,activities:[],
  input:{_execution:{background:true}},output:{status:'running',tool,job:{job_id:id}},
  decision:{reason:`submit ${tool}`},llm:{elapsed_seconds:1}});

test('both branches run together; collection calls merge without counting submit time',async t=>{
  const f=await fixture(t);
  await f.job(a,'ego_trajectory_editor');await f.job(b,'policy_runner');
  const steps=[step(5,'ego_trajectory_editor',a),step(6,'policy_runner',b),{
    step:7,tool:'ego_trajectory_editor',input:{_execution:{job_id:a}},activities:[],llm:{elapsed_seconds:2}}];
  let result=await resolveJobSteps({steps,logsRoot:f.root,scope:f.scope,now:f.now});
  assert.equal(result.steps.length,2);assert.equal(result.active_jobs.length,2);
  assert.deepEqual(result.steps[0].dispatch_steps,[5,7]);
  assert.equal(result.steps[0].duration,null);assert.equal(result.steps[0].elapsed_seconds,30);
  assert.equal(result.steps[0].llm.elapsed_seconds,3);
  const output={status:'success',tool:'ego_trajectory_editor',result:{expert_trajectory_path:'/scene/expert.csv'}};
  await f.job(a,'ego_trajectory_editor',{status:'completed',finished_at:f.now/1000-10},output);
  result=await resolveJobSteps({steps,logsRoot:f.root,scope:f.scope,now:f.now});
  assert.equal(result.active_jobs.length,1);assert.equal(result.steps[0].status,'success');
  assert.equal(result.steps[0].duration,20);assert.equal(result.steps[0].output.result.expert_trajectory_path,'/scene/expert.csv');
  assert.equal(result.steps[1].status,'running');
});

test('archived publication edits remain visible after the worker has finished',async t=>{
  const f=await fixture(t),base=step(5,'ego_trajectory_editor',a);
  const original={status:'success',tool:base.tool,result:{rendered_videos:{}}};
  await f.job(a,base.tool,{status:'completed',finished_at:f.now/1000},original);
  base.output={...original,job:{job_id:a},result:{rendered_videos:{expert:'/scene/later.mp4'}}};
  const result=await resolveJobSteps({steps:[base],logsRoot:f.root,scope:f.scope,now:f.now});
  assert.equal(result.steps[0].output.result.rendered_videos.expert,'/scene/later.mp4');
});

test('dead workers do not keep a stopwatch running or invent a completed result',async t=>{
  const f=await fixture(t);await f.job(a,'ego_trajectory_editor',{pid:999999999});
  const result=await resolveJobSteps({steps:[step(5,'ego_trajectory_editor',a)],logsRoot:f.root,scope:f.scope,now:f.now});
  assert.equal(result.active_jobs.length,0);assert.equal(result.steps[0].status,'interrupted');
  assert.equal(result.steps[0].duration,null);
});

test('scope and tool checks prevent job IDs from resolving another task output',async t=>{
  const f=await fixture(t);await f.job(a,'ego_trajectory_editor',{scope:'foreign'},
    {status:'success',result:{private_path:'/foreign'}});
  const result=await resolveJobSteps({steps:[step(5,'ego_trajectory_editor',a)],logsRoot:f.root,scope:f.scope});
  assert.equal(result.steps[0].job,undefined);assert.equal(result.steps[0].output.result,undefined);
  await f.job(a,'ego_trajectory_editor');
  const mixed=await resolveJobSteps({steps:[step(5,'ego_trajectory_editor',a),step(6,'policy_runner',a)],logsRoot:f.root,scope:f.scope});
  assert.equal(mixed.steps[1].job,undefined);
});

test('a surviving child keeps the branch stoppable when its worker exits',async t=>{
  const f=await fixture(t);const dir=await f.job(a,'ego_trajectory_editor',{pid:999999999});
  await writeFile(join(dir,'children.json'),JSON.stringify([{pid:process.pid,process_start:f.start}]));
  const result=await resolveJobSteps({steps:[step(5,'ego_trajectory_editor',a)],logsRoot:f.root,scope:f.scope});
  assert.equal(result.active_jobs.length,1);assert.equal(result.steps[0].status,'running');
});
