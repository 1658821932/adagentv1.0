import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile, rm, mkdir, readdir } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { tmpdir } from 'node:os';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { setTimeout as delay } from 'node:timers/promises';
import { videoCacheIdentity } from './video-cache.mjs';
import { resolveFfmpeg } from './video-tools.mjs';

test('same path with rewritten same-size content gets a new playback generation', async t => {
  const root=await mkdtemp(join(tmpdir(),'video-generation-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  const source=join(root,'camera.mp4');await writeFile(source,'aaaa');
  const first=await videoCacheIdentity(source);
  assert.equal(await videoCacheIdentity(source),first);
  await delay(5);await writeFile(source,'bbbb');
  assert.notEqual(await videoCacheIdentity(source),first);
  await writeFile(source,'');await assert.rejects(()=>videoCacheIdentity(source),/为空/);
});

test('HTTP playback refreshes when a native video overwrites its previous filename', {timeout:30000}, async t => {
  const frontend=dirname(fileURLToPath(import.meta.url));const project=dirname(frontend);
  const root=await mkdtemp(join(tmpdir(),'video-cache-http-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  const ffmpeg=resolveFfmpeg(project);
  const source=join(root,'camera.mp4');
  async function encode(color){
    await new Promise((resolve,reject)=>{
      const child=spawn(ffmpeg,['-v','error','-nostdin','-y','-f','lavfi','-i',`color=c=${color}:s=160x90:r=2:d=0.5`,'-c:v','libx264','-pix_fmt','yuv420p',source]);
      let error='';child.stderr.on('data',v=>{error+=v});child.on('error',reject);
      child.on('exit',code=>code===0?resolve():reject(new Error(error)));
    });
  }
  await encode('red');
  const server=spawn(process.execPath,[join(frontend,'server.mjs')],{
    cwd:frontend,env:{...process.env,PORT:'0',HOST:'127.0.0.1',ADAGENT_PROJECT_ROOT:root,ADAGENT_FFMPEG_BIN:ffmpeg},
    stdio:['ignore','pipe','pipe'],
  });
  t.after(async()=>{server.kill('SIGTERM');if(server.exitCode===null)await new Promise(r=>server.once('exit',r));});
  let text='';server.stdout.on('data',v=>{text+=v});server.stderr.on('data',v=>{text+=v});
  for(let i=0;i<100&&!/http:\/\/127.0.0.1:\d+/.test(text);i++)await delay(20);
  const base=text.match(/http:\/\/127.0.0.1:\d+/)?.[0];assert.ok(base,text);
  const url=base+'/api/video?path='+encodeURIComponent(source);
  async function read(){const response=await fetch(url);assert.equal(response.status,200);return Buffer.from(await response.arrayBuffer());}
  const first=await read();
  await encode('blue');const second=await read();
  assert.notEqual(createHash('sha256').update(first).digest('hex'),createHash('sha256').update(second).digest('hex'));
  assert.equal((await readdir(join(root,'frontend/.cache/eightcam'))).filter(p=>p.endsWith('_playable.mp4')).length,2);
});
