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
