/** Real server, isolated config/log directory, no Agent/model requests. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, mkdir, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

test('production config read/write, module MIME and stopped-state HTTP contracts', {timeout:15000}, async()=>{
  const frontend=dirname(fileURLToPath(import.meta.url));
  const root=await mkdtemp(join(tmpdir(),'adagent-config-http-'));
  const cfg=join(root,'configs');await mkdir(cfg);
  const run='run_20261114_100000_000001', runDir=join(root,'logs',run);
  await mkdir(join(runDir,'tools/step_00_attack_vehicle_editor'),{recursive:true});
  await writeFile(join(runDir,'run.json'),JSON.stringify({task:'offline fixture',status:'running',
    start_time:'2026-11-14T10:00:00Z',owner_pid:999999999,owner_process_start:'1'}));
  await writeFile(join(runDir,'tools/step_00_attack_vehicle_editor/input.json'),'{}');
  const api={active_profile:'one',one:{model:'offline-one',base_url:'https://example.invalid/v1',api_key:'test-secret-sentinel'},
    two:{model:'offline-two',base_url:'https://example.invalid/v1'},tool_profiles:{attack_editor:'active_profile'}};
  const agent={agent:{max_steps:20},orchestrator:{decision_correction_attempts:2},tools:{modes:{attack_vehicle_editor:'real'}}};
  await writeFile(join(cfg,'api.json'),JSON.stringify(api));await writeFile(join(cfg,'agent.json'),JSON.stringify(agent));
  const child=spawn(process.execPath,[join(frontend,'server.mjs')],{cwd:frontend,
    env:{...process.env,HOST:'127.0.0.1',PORT:'0',ADAGENT_PROJECT_ROOT:root},stdio:['ignore','pipe','pipe']});
  let output='';child.stderr.on('data',chunk=>{output+=chunk;});
  const exited=new Promise(resolve=>child.once('exit',resolve));
  try {
    const base=await new Promise((resolve,reject)=>{
      const timeout=setTimeout(()=>reject(new Error('Server did not start: '+output)),5000);
      child.once('exit',code=>{clearTimeout(timeout);reject(new Error('Server exited '+code+': '+output));});
      child.stdout.on('data',chunk=>{
        output+=chunk;const match=output.match(/http:\/\/127\.0\.0\.1:\d+/);
        if(match){clearTimeout(timeout);resolve(match[0]);}
      });
    });
    let response=await fetch(base+'/api/config');assert.equal(response.status,200);
    assert.equal(response.headers.get('cache-control'),'no-store');
    const current=await response.json();assert.equal(current.active_profile,'one');
    assert.equal(current.profiles.length,2);assert.equal(JSON.stringify(current).includes('test-secret-sentinel'),false);
    response=await fetch(base+'/config-panel-controller.mjs');assert.equal(response.status,200);
    assert.match(response.headers.get('content-type'),/javascript/);assert.match(await response.text(),/createConfigPanelController/);
    response=await fetch(base+'/app.js');assert.match(await response.text(),/createConfigPanelController/);
    response=await fetch(base+'/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
      active_profile:'two',reasoning_effort:'high',max_steps:25,decision_correction_attempts:3,
      tool_profiles:{attack_editor:'one'},tool_modes:{attack_vehicle_editor:'mock'},
    })});assert.equal(response.status,200);
    const updated=await (await fetch(base+'/api/config')).json();
    assert.equal(updated.active_profile,'two');assert.equal(updated.max_steps,25);
    assert.equal(updated.tool_profiles.attack_editor,'one');assert.equal(updated.tool_modes.attack_vehicle_editor,'mock');
    assert.equal(JSON.parse(await readFile(join(cfg,'api.json'),'utf8')).one.api_key,'test-secret-sentinel');
    const before=await readFile(join(cfg,'api.json'),'utf8');
    response=await fetch(base+'/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:'{"active_profile":"missing"}'});
    assert.equal(response.status,400);assert.equal(await readFile(join(cfg,'api.json'),'utf8'),before);
    const history=await (await fetch(base+'/api/runs')).json();
    assert.equal(history.runs[0].status,'interrupted');assert.equal(history.runs[0].duration_seconds,null);
    const detail=await (await fetch(base+'/api/runs/'+run)).json();
    assert.equal(detail.status,'interrupted');assert.equal(detail.active,false);
    assert.equal(detail.steps[0].status,'interrupted');assert.equal(detail.steps[0].elapsed_seconds,null);
    await writeFile(join(cfg,'agent.json'),'{invalid');
    response=await fetch(base+'/api/config');assert.equal(response.status,500);
    assert.match((await response.json()).error,/配置文件/);
  } finally {
    child.kill('SIGTERM');await exited;await rm(root,{recursive:true,force:true});
  }
});
