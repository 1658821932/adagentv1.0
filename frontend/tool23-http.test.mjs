/** Exercise the production API entry and current fixtures without running an Agent. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn, execFileSync } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';

const frontend = dirname(fileURLToPath(import.meta.url));
const demo = JSON.parse(await readFile(join(frontend, 'demo/run.json'), 'utf8'));

test('current tool02/tool03 demo outputs match their backend contracts', () => {
  const python = join(frontend, '../environments/agent/.venv/bin/python');
  execFileSync(python, ['-c', `import json,sys
from tools.tool02_scene_instance_generator.schemas import Input as Input2, Output as Output2
from tools.tool03_video_retriever.schemas import Input as Input3, Output as Output3
for step in json.load(sys.stdin)['steps']:
    if step['tool']=='scene_instance_generator':
        Input2.model_validate(step['input']); Output2.model_validate(step['output']['result'])
    elif step['tool']=='video_retriever':
        Input3.model_validate(step['input']); Output3.model_validate(step['output']['result'])
`], { cwd: resolve(frontend, '..'), input: JSON.stringify(demo), encoding: 'utf8' });
});

test('production run API preserves both contracts and serves the matching preview', { timeout: 15000 }, async t => {
  const root = await mkdtemp(join(tmpdir(), 'adagent-tool23-http-'));
  let server;
  t.after(async () => {
    if (server) {
      server.kill('SIGTERM');
      if (server.exitCode === null) await new Promise(done => server.once('exit', done));
    }
    await rm(root, { recursive: true, force: true });
  });
  const id = 'run_20261118_tool23';
  const run = join(root, 'logs', id);
  await mkdir(run, { recursive: true });
  await writeFile(join(run, 'run.json'), JSON.stringify({ status: 'success', task: 'offline tool23 UI fixture' }));
  await writeFile(join(run, 'final_state.json'), JSON.stringify({ status: 'success' }));
  const lossless = join(root, 'scene_rank01_score0.900000_lossless.mp4');
  const compressed = join(root, 'scene_rank01_score0.900000_compressed.mp4');
  await writeFile(lossless, 'unused lossless fixture');
  await writeFile(compressed, Buffer.from('00000000ftypisom0123456789abcdef0123456789'));
  const sample = join(root, 'sample');
  await mkdir(join(sample, 'preview'), { recursive: true });
  await writeFile(join(sample, 'preview/original_eightcam_2hz.mp4'), 'grid fixture');
  const scenario = demo.steps.find(step => step.tool === 'scene_instance_generator');
  const retrieval = {
    tool: 'video_retriever', input: { requirement: scenario.output.result.candidates[1].retrieval_query, candidate_id: 'cand_2' },
    output: { status: 'success', tool: 'video_retriever', result: {
      candidate_id: 'cand_2', mode: 'local_hit_reranked', match_score: .9,
      video_path: lossless, source_file: join(root, 'source.mp4'),
      scene_ref: { scene_token: 'fixture-token', sample_root: sample, matched_camera: 'CAM_R0' },
    } },
  };
  for (const [index, step] of [scenario, retrieval].entries()) {
    const dir = join(run, 'tools', `step_0${index + 1}_${step.tool}`);
    await mkdir(dir, { recursive: true });
    await writeFile(join(dir, 'input.json'), JSON.stringify(step.input));
    await writeFile(join(dir, 'output.json'), JSON.stringify(step.output));
  }
  let output = '';
  server = spawn(process.execPath, [join(frontend, 'server.mjs')], {
    cwd: frontend, env: { ...process.env, HOST: '127.0.0.1', PORT: '0', ADAGENT_PROJECT_ROOT: root },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  server.stdout.on('data', chunk => { output += chunk; });
  server.stderr.on('data', chunk => { output += chunk; });
  for (let i = 0; i < 200 && !output.includes('adagent console:'); i++) {
    if (server.exitCode !== null) throw new Error(output);
    await delay(20);
  }
  const port = /http:\/\/127\.0\.0\.1:(\d+)/.exec(output)?.[1];
  assert.ok(port, output);
  const base = `http://127.0.0.1:${port}`;
  const response = await fetch(`${base}/api/runs/${id}`);
  assert.equal(response.status, 200);
  const detail = await response.json();
  assert.deepEqual(detail.steps[0].output.result, scenario.output.result);
  assert.equal(detail.steps[0].output.result.candidates.length, 3);
  const retrieved = detail.steps[1];
  assert.equal(retrieved.output.result.candidate_id, 'cand_2');
  assert.equal(retrieved.evidence.length, 2);
  assert.equal(retrieved.evidence[0].src, '/api/file?path=' + encodeURIComponent(compressed));
  assert.match(retrieved.evidence[0].title, /CAM_R0/);
  assert.match(retrieved.evidence[1].src, /^\/api\/video\?/);
  const video = await fetch(base + retrieved.evidence[0].src, { headers: { Range: 'bytes=0-15' } });
  assert.equal(video.status, 206);
  assert.equal(video.headers.get('content-type'), 'video/mp4');
  assert.equal((await video.arrayBuffer()).byteLength, 16);
  const module = await fetch(base + '/scenario-results.mjs');
  assert.equal(module.status, 200);
  assert.match(module.headers.get('content-type'), /javascript/);
});
