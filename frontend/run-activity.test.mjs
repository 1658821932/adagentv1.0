import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { readSubagentActivity } from './run-activity.mjs';

async function write(path, value) {
  await mkdir(join(path, '..'), { recursive: true });
  await writeFile(path, JSON.stringify(value));
}

test('ego preparation and actor editing are separate visible stages with real state', async () => {
  const run=await mkdtemp(join(tmpdir(),'adagent-two-stages-'));
  await write(join(run,'operations/prepare.json'),{
    event_id:'prepare',name:'attack.ego_history_prepare',status:'returned',result_status:'prepared',
    context:{step:4},elapsed_seconds:.8,started_at:'2026-11-16T16:00:00Z',finished_at:'2026-11-16T16:00:01Z',
  });
  await write(join(run,'operations/edit.json'),{
    event_id:'edit',name:'attack.vehicle_editing',status:'running',context:{step:4},
    started_at:'2026-11-16T16:00:02Z',
  });
  const items=await readSubagentActivity(run);
  assert.equal(items.length,2);
  assert.equal(items[0].summary,'加速历史已生成并冻结');
  assert.equal(items[0].elapsed_seconds,.8);
  assert.equal(items[1].name,'attack.vehicle_editing');assert.equal(items[1].status,'running');
});

test('runtime tool exchanges supplement pending events without duplicating archived actions', async () => {
  const base = await mkdtemp(join(tmpdir(), 'adagent-activity-'));
  const runDir = join(base, 'run');
  const runtime = join(base, 'runtime');
  const turn1 = join(runtime, 'output/attack/trajectory_editing/tool_calls/turn_01');
  const turn2 = join(runtime, 'output/attack/trajectory_editing/tool_calls/turn_02');
  const turn3 = join(runtime, 'output/defense_agent/tool-exchanges/turn_03');
  await write(join(runDir, 'operations/object.json'), {
    event_id: 'object', name: 'attack.object_selection', status: 'returned',
    context: { step: 6 }, archive_ref: join(runtime, 'output/attack/object_selection/timing/object.json'),
  });
  await write(join(runDir, 'tool_events/first.json'), {
    context: { step: 6 }, tool_name: 'construct_attack_candidates', archive_ref: turn1,
    result: { status: 'invalid' }, elapsed_seconds: 0.006,
  });
  await write(join(turn1, 'call.json'), { function: { name: 'construct_attack_candidates' } });
  await write(join(turn1, 'result.json'), { status: 'invalid' });
  await write(join(turn2, 'call.json'), { function: { name: 'construct_attack_candidates' } });
  await write(join(turn2, 'result.json'), { status: 'fallback', candidates: [{ candidate_id: 'candidate_008' }] });
  await write(join(turn3, 'call.json'), { function: { name: 'inspect_trajectory' } });

  const activities = await readSubagentActivity(runDir);
  assert.equal(activities.filter(item => item.name === 'construct_attack_candidates').length, 2);
  assert.equal(activities.filter(item => item.archive_ref === turn1).length, 1);
  assert.equal(activities.filter(item => item.archive_ref === turn1)[0].elapsed_seconds, 0.006);
  assert.match(activities.find(item => item.archive_ref === turn2).summary, /candidate_008/);
  const pending = activities.find(item => item.archive_ref === turn3);
  assert.equal(pending.status, 'running');
  assert.equal(pending.summary, '执行中');
});
