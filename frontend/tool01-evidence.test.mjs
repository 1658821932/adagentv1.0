/** Tool 01 evidence must come from the run directory the analysis really wrote.

The manifest used here is produced by the real backend (`real.analyze`), so the frame /
selected fields under test are the ones the frontend will actually receive.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { failureAnalyzerEvidence } from './tool01-evidence.mjs';

const frontend = dirname(fileURLToPath(import.meta.url));
const project = resolve(frontend, '..');
const python = join(project, 'environments/agent/.venv/bin/python');

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

test('manifest sheets become reachable image evidence', async t => {
  const root = await mkdtemp(join(tmpdir(), 'adagent-t01-evidence-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const evidenceDir = makeEvidence(root);
  if (!evidenceDir) return t.skip('agent environment is not available');

  const manifest = JSON.parse(await readFile(join(evidenceDir, 'manifest.json'), 'utf8'));
  assert.deepEqual(Object.keys(manifest.frames[0]).sort(), ['index', 'sheet', 'sheet_sha256']);
  assert.equal(manifest.selected[0].time_s, 3.5);
  assert.equal(manifest.frames.length, 1);
  assert.equal(manifest.selected.length, 1);

  const { summary, items } = failureAnalyzerEvidence(
    { status: 'success', result: { evidence_dir: evidenceDir } }, { projectRoot: root });
  assert.equal(items.length, 1);
  assert.deepEqual(items.map(item => item.kind), ['image']);
  assert.equal(items[0].src, '/api/file?path=' + encodeURIComponent(manifest.frames[0].sheet));
  assert.deepEqual(items.map(item => item.note), ['帧 7 · 3.5s']);
  assert.equal(items[0].primary, true);
  assert.equal(summary.images, 1);
  assert.equal(summary.manifest.frames, 1);
  assert.equal(summary.manifest.stage, 'done');
  assert.equal(summary.stage, 'done');
});

test('time comes from selected, even when frames carries none', async t => {
  const root = await mkdtemp(join(tmpdir(), 'adagent-t01-evidence-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const evidenceDir = makeEvidence(root);
  if (!evidenceDir) return t.skip('agent environment is not available');
  const path = join(evidenceDir, 'manifest.json');
  const manifest = JSON.parse(await readFile(path, 'utf8'));
  manifest.frames = manifest.frames.map(frame => ({ index: frame.index, sheet: frame.sheet }));
  await writeFile(path, JSON.stringify(manifest));
  const { items } = failureAnalyzerEvidence(
    { status: 'success', result: { evidence_dir: evidenceDir } }, { projectRoot: root });
  assert.equal(items[0].note, '帧 7 · 3.5s');
});

test('sheets still show when the manifest is missing or damaged', async t => {
  const root = await mkdtemp(join(tmpdir(), 'adagent-t01-evidence-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const evidenceDir = makeEvidence(root);
  if (!evidenceDir) return t.skip('agent environment is not available');

  const path = join(evidenceDir, 'manifest.json');
  await rm(path);
  let result = failureAnalyzerEvidence(
    { status: 'success', result: { evidence_dir: evidenceDir } }, { projectRoot: root });
  assert.equal(result.items.length, 1);
  assert.equal(result.items[0].note, '本次分析的八相机总览');
  assert.equal(result.summary.manifest, null);

  await writeFile(path, '{ not json');
  result = failureAnalyzerEvidence(
    { status: 'success', result: { evidence_dir: evidenceDir } }, { projectRoot: root });
  assert.equal(result.items.length, 1);
});

test('an unreadable sheets directory never fails the run view', async t => {
  const root = await mkdtemp(join(tmpdir(), 'adagent-t01-evidence-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const evidenceDir = makeEvidence(root);
  if (!evidenceDir) return t.skip('agent environment is not available');
  await rm(join(evidenceDir, 'sheets'), { recursive: true, force: true });
  await writeFile(join(evidenceDir, 'sheets'), 'not a directory');
  const { items, summary } = failureAnalyzerEvidence(
    { status: 'success', result: { evidence_dir: evidenceDir } }, { projectRoot: root });
  assert.deepEqual(items, []);
  assert.equal(summary.manifest.frames, 1);
});

test('failed runs keep their stage and images', async t => {
  const root = await mkdtemp(join(tmpdir(), 'adagent-t01-evidence-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const evidenceDir = makeEvidence(root);
  if (!evidenceDir) return t.skip('agent environment is not available');
  const { summary, items } = failureAnalyzerEvidence(
    { status: 'error', error: { type: 'timeout', stage: 'vlm', evidence_dir: evidenceDir } },
    { projectRoot: root });
  assert.equal(items.length, 1);
  assert.equal(summary.error_type, 'timeout');
  assert.equal(summary.stage, 'vlm');
});

test('paths outside the project are never exposed', async t => {
  const root = await mkdtemp(join(tmpdir(), 'adagent-t01-evidence-'));
  const outside = await mkdtemp(join(tmpdir(), 'adagent-t01-outside-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  t.after(() => rm(outside, { recursive: true, force: true }));
  const evidenceDir = makeEvidence(outside);
  if (!evidenceDir) return t.skip('agent environment is not available');
  const { items, summary } = failureAnalyzerEvidence(
    { status: 'success', result: { evidence_dir: evidenceDir } }, { projectRoot: root });
  assert.deepEqual(items, []);
  assert.equal(summary.evidence_dir, evidenceDir);
});

test('a manifest pointing outside the project is ignored', async t => {
  const root = await mkdtemp(join(tmpdir(), 'adagent-t01-evidence-'));
  const outside = await mkdtemp(join(tmpdir(), 'adagent-t01-outside-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  t.after(() => rm(outside, { recursive: true, force: true }));
  const outsideFile = join(outside, 'secret.jpg');
  await writeFile(outsideFile, Buffer.from([0xff, 0xd8, 0xff, 0xd9]));
  const evidenceDir = join(root, 'logs', 'tool01', 'x', 'y', 'run-1');
  await mkdir(evidenceDir, { recursive: true });
  await writeFile(join(evidenceDir, 'manifest.json'), JSON.stringify({
    status: 'success', frames: [{ index: 0, sheet: outsideFile, sheet_sha256: 'a' }], selected: [],
  }));
  const { items } = failureAnalyzerEvidence(
    { status: 'success', result: { evidence_dir: evidenceDir } }, { projectRoot: root });
  assert.deepEqual(items, []);
});

test('missing evidence directories produce no items', async () => {
  const root = await mkdtemp(join(tmpdir(), 'adagent-t01-evidence-'));
  const { items, summary } = failureAnalyzerEvidence(
    { status: 'success', result: { evidence_dir: join(root, 'nope') } }, { projectRoot: root });
  await rm(root, { recursive: true, force: true });
  assert.deepEqual(items, []);
  assert.equal(summary.images, 0);
  assert.deepEqual(failureAnalyzerEvidence(null, { projectRoot: root }).items, []);
});
