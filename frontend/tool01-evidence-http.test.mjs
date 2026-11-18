/** The real run API must serve tool 01 evidence images, including failed runs. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';

const frontend = dirname(fileURLToPath(import.meta.url));
const project = resolve(frontend, '..');
const python = join(project, 'environments/agent/.venv/bin/python');
const RUN = 'run_20260101_000000_1';

const MAKE_EVIDENCE = `
import json, sys
from pathlib import Path
from unittest import mock
from PIL import Image
root = Path(sys.argv[1])
sys.path.insert(0, ${JSON.stringify(project)})
from tools.tool01_failure_analyzer import real, visual_check
cam = root / "sensor_blobs" / "scene"
for camera in visual_check.CAMERAS:
    (cam / camera).mkdir(parents=True, exist_ok=True)
    for index in range(12):
        Image.new("RGB", (64, 36), (index, 80, 140)).save(cam / camera / ("%02d.jpg" % index), format="JPEG")
frames = [{"index": index, "time_s": index * 0.5,
           "images": {camera: str(cam / camera / ("%02d.jpg" % index)) for camera in visual_check.CAMERAS}}
          for index in range(12)]
facts = dict(archive="archive", token="scene-abc", simulation_record="experiment/batch",
             map_location="map", violation_step=7,
             failure={"outcome": "collision", "interaction": "crossing"},
             attack={"map_context": "t_junction", "ego_action": "turn_right", "ego_lane_keeping": "keep",
                     "other_origin": "right_arm", "other_action": "turn_left", "other_lane_keeping": "keep"},
             environment={"time": "t"}, frame_images=frames)
class Client:
    def __init__(self, component=None): pass
    def chat(self, messages):
        return json.dumps({"map_context": "t_junction", "ego_action": "turn_right",
            "ego_lane_keeping": "keep", "other_origin": "right_arm", "other_action": "turn_left",
            "other_lane_keeping": "keep", "interaction": "crossing", "notes": [],
            "weather": "clear", "extra": ""})
run_dir = root / "logs" / "tool01" / "experiment_batch-fixture" / "4abb45d9ee1a5c5c" / "run-fixture"
with mock.patch.object(real, "failure_facts", return_value=facts), \\
     mock.patch("agent.llm_client.LLMClient", Client):
    real.analyze({"token": "4abb45d9ee1a5c5c", "simulation": "experiment/batch", "timeout": 60}, run_dir)
print(str(run_dir))
`;

function makeEvidence(root) {
  if (!existsSync(python)) return null;
  return execFileSync(python, ['-c', MAKE_EVIDENCE, root], { encoding: 'utf8' }).trim();
}

const RESULT = {
  token: '2021.09.16.19.49.00_veh-42_00990_01609-4abb45d9ee1a5c5c',
  simulation: 'experiment/batch',
  map_location: 'us-pa-pittsburgh-hazelwood',
  evidence_dir: '',
  violation_step: 8,
  failure: { outcome: 'collision', interaction: 'crossing', ego_cause: '在丁字路口右转时没有让行' },
  attack: { map_context: 't_junction', ego_action: 'turn_right', ego_lane_keeping: 'keep',
    other_origin: 'right_arm', other_action: 'turn_left', other_lane_keeping: 'keep' },
  environment: { time: '2021-09-16T20:16:41+00:00', weather: 'cloudy', extra: '傍晚，路面干燥' },
  verification: { sample_frames: [6, 8, 10], corrections: [], notes: [] },
};

test('run API serves the evidence sheets of a tool 01 step', { timeout: 40000 }, async t => {
  const root = await mkdtemp(join(tmpdir(), 'adagent-t01-http-'));
  const runDir = join(root, 'logs', RUN);
  const evidence = makeEvidence(root);
  if (!evidence) return t.skip('agent environment is not available');
  const manifest = JSON.parse(await readFile(join(evidence, 'manifest.json'), 'utf8'));
  const sheets = manifest.frames.map(frame => frame.sheet);

  async function step(name, output) {
    const dir = join(runDir, 'tools', name);
    await mkdir(dir, { recursive: true });
    await writeFile(join(dir, 'input.json'), JSON.stringify({ token: '4abb45d9ee1a5c5c' }));
    await writeFile(join(dir, 'output.json'), JSON.stringify(output));
    await writeFile(join(dir, 'execution.log'), '[2026-01-01T00:00:00Z] tool started duration_seconds=2.5\n');
  }

  await mkdir(runDir, { recursive: true });
  await writeFile(join(runDir, 'run.json'), JSON.stringify({
    task: 'fixture', start_time: '2026-01-01T00:00:00Z', end_time: '2026-01-01T00:01:00Z',
  }));
  await writeFile(join(runDir, 'final_state.json'), JSON.stringify({ status: 'success' }));
  await step('step_01_failure_analyzer', {
    status: 'success', tool: 'failure_analyzer', result: { ...RESULT, evidence_dir: evidence },
  });
  await step('step_02_failure_analyzer', {
    status: 'error', tool: 'failure_analyzer',
    error: { type: 'timeout', stage: 'vlm', message: 'analysis exceeded its 30s budget',
      evidence_dir: evidence },
  });

  let output = '';
  const server = spawn(process.execPath, [join(frontend, 'server.mjs')], {
    cwd: frontend,
    env: { ...process.env, HOST: '127.0.0.1', PORT: '0', ADAGENT_PROJECT_ROOT: root },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  server.stdout.on('data', data => { output += data; });
  server.stderr.on('data', data => { output += data; });
  t.after(async () => {
    server.kill('SIGTERM');
    await new Promise(done => (server.exitCode !== null ? done() : server.once('exit', done)));
    await rm(root, { recursive: true, force: true });
  });

  for (let i = 0; i < 200 && !output.includes('adagent console:'); i++) {
    if (server.exitCode !== null) throw new Error(output);
    await delay(20);
  }
  const match = /http:\/\/127\.0\.0\.1:(\d+)/.exec(output);
  assert.ok(match, output);
  const base = `http://127.0.0.1:${match[1]}`;

  const detail = await (await fetch(`${base}/api/runs/${RUN}`)).json();
  const steps = Object.fromEntries(detail.steps.map(entry => [entry.step, entry]));

  assert.equal(steps[1].output.result.failure.ego_cause, RESULT.failure.ego_cause);
  const success = steps[1].evidence.filter(item => item.kind === 'image');
  assert.equal(success.length, sheets.length, JSON.stringify(steps[1].evidence));
  assert.equal(success[0].src, '/api/file?path=' + encodeURIComponent(sheets[0]));
  assert.equal(steps[1].evidence_summary.manifest.frames, sheets.length);
  assert.equal(steps[1].evidence_summary.error_type, '');

  assert.equal(steps[2].output.error.stage, 'vlm');
  const failed = steps[2].evidence.filter(item => item.kind === 'image');
  assert.equal(failed.length, sheets.length, JSON.stringify(steps[2].evidence));
  assert.equal(steps[2].evidence_summary.error_type, 'timeout');
  assert.equal(steps[2].evidence_summary.stage, 'vlm');

  const image = await fetch(base + success[0].src);
  assert.equal(image.status, 200);
  assert.equal(image.headers.get('content-type'), 'image/jpeg');
  assert.ok((await image.arrayBuffer()).byteLength > 0);

  const blocked = await fetch(`${base}/api/file?path=${encodeURIComponent('/etc/passwd')}`);
  assert.equal(blocked.status, 403);

  const app = await (await fetch(base + '/app.js')).text();
  assert.ok(app.includes('tool01-result.mjs'), 'app.js must use the tool 01 result module');
  assert.ok(app.includes('failureDetailGroups'));
  assert.ok(app.includes('evidence_summary'));
});
