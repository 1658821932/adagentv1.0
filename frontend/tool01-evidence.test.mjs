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
src_dir = root / "tmp_facts"
src_dir.mkdir(parents=True, exist_ok=True)
traj_src = src_dir / "trajectories.json"
traj_src.write_text(json.dumps({"ego": [], "vehicles": []}))
bev_src = src_dir / "bev.png"
Image.new("RGB", (128, 128), (30, 50, 70)).save(bev_src, format="PNG")
facts = dict(
    archive="archive", token="scene-abc", simulation_record="experiment/batch",
    map_location="map", violation_step=7, outcome="collision",
    trajectories_path=str(traj_src), bev_image=str(bev_src),
    ego_kinematics={"is_stationary": False, "speed_start_mps": 4.0, "speed_violation_mps": 5.0,
                    "displacement_m": 15.0, "yaw_change_deg": 2.0, "tail_yaw_deg": 1.0},
    nearby_vehicles=[{
        "track": "veh_01", "label": "car", "size_m": [4.4, 2.0, 1.6],
        "violation_distance_m": 0.4, "relative_heading_at_violation_deg": 90.0,
        "kinematics": {"is_stationary": False, "speed_start_mps": 5.0, "speed_violation_mps": 4.5,
                       "displacement_m": 12.0, "yaw_change_deg": 80.0, "tail_yaw_deg": 0.0},
    }],
    environment={"time": "2021-09-16T20:16:41+00:00"}, frame_images=frames,
)
class Client:
    calls = 0
    def __init__(self, component=None): pass
    def chat(self, messages):
        Client.calls += 1
        if Client.calls == 1:
            return json.dumps({
                "map_context": "t_junction", "failure_driver": "collision_with_vehicle",
                "ego_behavior": "自车在丁字路口右转未让行。",
                "other_behavior": "右侧支路车辆左转进入路口。",
                "summary": "自车在丁字路口右转时与右侧支路左转车辆发生碰撞。",
            })
        return json.dumps({"weather": "clear", "extra": "傍晚路面干燥", "notes": []})
run_dir = root / "logs" / "tool01" / "experiment_batch-fixture" / "4abb45d9ee1a5c5c" / "run-fixture"
with mock.patch.object(real, "failure_facts", return_value=facts), \\
     mock.patch("agent.llm_client.LLMClient", Client):
    out = real.analyze({"token": "4abb45d9ee1a5c5c", "simulation": "experiment/batch", "timeout": 60}, run_dir)
assert out["status"] == "success", out
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
  const sheet = join(evidenceDir, 'sheets', 'step_07.jpg');
  assert.equal(manifest.bev_image, join(evidenceDir, 'bev_overview.png'));
  assert.equal(manifest.stage, 'done');

  const { summary, items } = failureAnalyzerEvidence(
    { status: 'success', result: { evidence_dir: evidenceDir } }, { projectRoot: root });
  assert.equal(items.length, 2);
  assert.deepEqual(items.map(item => item.kind), ['image', 'image']);
  assert.equal(items[0].src, '/api/file?path=' + encodeURIComponent(manifest.bev_image));
  assert.equal(items[0].primary, true);
  assert.equal(items[1].src, '/api/file?path=' + encodeURIComponent(sheet));
  assert.equal(items[1].note, '本次分析的八相机总览');
  assert.equal(items[1].primary, false);
  assert.equal(summary.images, 2);
  assert.equal(summary.manifest.bev_image, true);
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
  const sheet = join(evidenceDir, 'sheets', 'step_07.jpg');
  manifest.frames = [{ index: 7, sheet }];
  manifest.selected = [{ index: 7, time_s: 3.5 }];
  await writeFile(path, JSON.stringify(manifest));
  const { items } = failureAnalyzerEvidence(
    { status: 'success', result: { evidence_dir: evidenceDir } }, { projectRoot: root });
  assert.equal(items.find(i => i.path === sheet).note, '帧 7 · 3.5s');
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
  assert.equal(result.items.length, 2);
  assert.equal(result.items[0].title, 'BEV 俯视证据');
  assert.equal(result.items[1].note, '本次分析的八相机总览');
  assert.equal(result.summary.manifest, null);

  await writeFile(path, '{ not json');
  result = failureAnalyzerEvidence(
    { status: 'success', result: { evidence_dir: evidenceDir } }, { projectRoot: root });
  assert.equal(result.items.length, 2);
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
  assert.equal(items.length, 1);
  assert.equal(items[0].title, 'BEV 俯视证据');
  assert.equal(summary.manifest.bev_image, true);
});

test('failed runs keep their stage and images', async t => {
  const root = await mkdtemp(join(tmpdir(), 'adagent-t01-evidence-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const evidenceDir = makeEvidence(root);
  if (!evidenceDir) return t.skip('agent environment is not available');
  const { summary, items } = failureAnalyzerEvidence(
    { status: 'error', error: { type: 'timeout', stage: 'vlm', evidence_dir: evidenceDir } },
    { projectRoot: root });
  assert.equal(items.length, 2);
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
