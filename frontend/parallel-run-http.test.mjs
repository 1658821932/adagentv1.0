import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp,mkdir,readFile,writeFile,rm } from 'node:fs/promises';
import { join,dirname } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';
import vm from 'node:vm';

const frontend=dirname(fileURLToPath(import.meta.url));
test('HTTP refresh resolves both jobs, live artifacts, deduplicated cards and surviving workers',{timeout:15000},async t=>{
  const root=await mkdtemp(join(tmpdir(),'adagent-parallel-http-')),logs=join(root,'logs');
  const id='run_20261116_120000_000001',run=join(logs,id),scope=id;
  await mkdir(run,{recursive:true});
  const procstat=await readFile(`/proc/${process.pid}/stat`,'utf8');
  const start=procstat.slice(procstat.lastIndexOf(')')+1).trim().split(/\s+/)[19];
  const now=Date.now()/1000;
  const meta={status:'running',start_time:new Date((now-40)*1000).toISOString(),execution_scope:scope,
    owner_pid:process.pid,owner_process_start:start};
  await writeFile(join(run,'run.json'),JSON.stringify(meta));
  await writeFile(join(run,'state.json'),JSON.stringify({execution_scope:scope}));
  const a='a'.repeat(32),b='b'.repeat(32);
  for (const [job,tool,step] of [[a,'ego_trajectory_editor',5],[b,'policy_runner',6]]) {
    const folder=join(logs,'jobs',job);await mkdir(folder,{recursive:true});
    await writeFile(join(folder,'manifest.json'),JSON.stringify({scope,tool,status:'running',pid:process.pid,
      process_start:start,started_at:now-30}));
    const dir=join(run,'tools',`step_0${step}_${tool}`);await mkdir(dir,{recursive:true});
    await writeFile(join(dir,'input.json'),JSON.stringify({_execution:{background:true}}));
    await writeFile(join(dir,'output.json'),JSON.stringify({status:'running',tool,job:{job_id:job}}));
    await writeFile(join(dir,'timing.json'),JSON.stringify({elapsed_seconds:.001}));
  }
  const collection=join(run,'tools/step_07_ego_trajectory_editor');await mkdir(collection,{recursive:true});
  await writeFile(join(collection,'input.json'),JSON.stringify({_execution:{job_id:a}}));
  const server=spawn(process.execPath,[join(frontend,'server.mjs')],{cwd:frontend,detached:true,
    env:{...process.env,PORT:'0',HOST:'127.0.0.1',ADAGENT_PROJECT_ROOT:root},stdio:['ignore','pipe','pipe']});
  t.after(async()=>{try{process.kill(-server.pid,'SIGKILL');}catch{}await rm(root,{recursive:true,force:true});});
  let output='';server.stdout.on('data',v=>{output+=v;});server.stderr.on('data',v=>{output+=v;});
  for(let i=0;i<100&&!/http:\/\/127\.0\.0\.1:\d+/.test(output);i++)await delay(20);
  const base=output.match(/http:\/\/127\.0\.0\.1:\d+/)?.[0];assert.ok(base,output);
  const get=async()=>{const response=await fetch(`${base}/api/runs/${id}`);assert.equal(response.status,200);return response.json();};
  let detail=await get();assert.equal(detail.steps.length,2);assert.equal(detail.active_jobs.length,2);
  assert.equal(detail.dispatch_count,3);assert.equal(detail.steps[0].duration,null);
  assert.ok(detail.steps[0].elapsed_seconds>=30);
  const artifact=join(root,'outputs/expert.mp4');await mkdir(dirname(artifact),{recursive:true});await writeFile(artifact,'offline-media-presence-fixture');
  const manifest=join(logs,'jobs',a,'manifest.json');const job=JSON.parse(await readFile(manifest,'utf8'));
  await writeFile(manifest,JSON.stringify({...job,status:'completed',finished_at:now-5}));
  await writeFile(join(logs,'jobs',a,'result.json'),JSON.stringify({status:'success',tool:'ego_trajectory_editor',
    result:{rendered_videos:{manual_explained:artifact}}}));
  detail=await get();assert.equal(detail.steps.length,2);assert.equal(detail.steps[0].status,'success');
  assert.equal(detail.steps[0].duration,25);assert.equal(detail.steps[1].status,'running');
  assert.ok(detail.steps[0].artifacts.some(item=>item.path===artifact));
  assert.ok(detail.steps[0].media.some(item=>item.kind==='video'));
  await writeFile(join(run,'run.json'),JSON.stringify({...meta,status:'error',owner_pid:999999999,end_time:new Date().toISOString()}));
  await writeFile(join(run,'final_state.json'),JSON.stringify({status:'error',error:'main interrupted'}));
  detail=await get();assert.equal(detail.status,'error');assert.equal(detail.active,true);
  assert.equal(detail.background_active,true);assert.equal(detail.active_jobs.length,1);
});

test('phase narrative describes both branches rather than only the latest step',async()=>{
  const source=await readFile(join(frontend,'app.js'),'utf8');
  const context=vm.createContext({nameOf:name=>({'ego_trajectory_editor':'专家编辑','policy_runner':'策略评测'}[name])});
  vm.runInContext(source.slice(source.indexOf('function phaseNarrative('),source.indexOf('function statusLabel(')),context);
  const steps=[{tool:'ego_trajectory_editor',status:'running'},{tool:'policy_runner',status:'running'}];
  assert.equal(context.phaseNarrative({status:'running',steps}),'并行执行：专家编辑、策略评测');
  steps[1].status='success';assert.equal(context.phaseNarrative({status:'running',steps}),'正在执行：专家编辑');
  assert.equal(context.phaseNarrative({status:'error',background_active:true,steps}),'后台任务仍在执行');
});
