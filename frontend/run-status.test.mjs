import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolveRunStatus } from './run-status.mjs';

test('stale running metadata and unfinished steps resolve to interrupted, without invented duration', async () => {
  const result=await resolveRunStatus({status:'running',start_time:'2026-11-14T09:00:00Z',owner_pid:999999999,owner_process_start:'1'});
  assert.deepEqual(result,{status:'interrupted',active:false,duration_seconds:null});
});

test('persisted identity recognizes a live run after frontend restart', async () => {
  const stat=await readFile(`/proc/${process.pid}/stat`,'utf8');
  const start=stat.slice(stat.lastIndexOf(')')+1).trim().split(/\s+/)[19];
  const meta={status:'running',start_time:new Date().toISOString(),owner_pid:process.pid,owner_process_start:start};
  assert.equal((await resolveRunStatus(meta)).status,'running');
  assert.equal((await resolveRunStatus({...meta,owner_process_start:'wrong'})).status,'interrupted');
  const finished=await resolveRunStatus({...meta,end_time:new Date().toISOString()},{status:'finished'});
  assert.equal(finished.status,'finished'); assert.equal(finished.active,false);
});

test('final interrupted state overrides stale metadata and ended runs stop accumulating time', async () => {
  const result=await resolveRunStatus({status:'running',start_time:'2026-11-14T09:00:00Z',end_time:'2026-11-14T09:01:00Z'},{status:'interrupted'});
  assert.deepEqual(result,{status:'interrupted',active:false,duration_seconds:60});
});
