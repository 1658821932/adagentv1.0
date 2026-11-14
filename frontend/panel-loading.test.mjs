import test from 'node:test';
import assert from 'node:assert/strict';
import { panelLoadingHtml, panelErrorHtml, readPanelData } from './panel-loading.mjs';

test('loading feedback is accessible and skeletons contain no fake progress',()=>{
  const html=panelLoadingHtml('正在读取…',3);
  assert.match(html,/role="status"/);assert.equal((html.match(/<i>/g)||[]).length,3);
  assert.doesNotMatch(html,/%/);
});
test('error feedback safely escapes server text and offers a retry',()=>{
  const html=panelErrorHtml('<script>bad</script>','data-history-retry',true);
  assert.doesNotMatch(html,/<script>/);assert.match(html,/data-history-retry/);assert.match(html,/上次结果/);
});
test('a stalled read times out and aborts its GET',async()=>{
  let signal;
  await assert.rejects(readPanelData(options=>{signal=options.signal;return new Promise(()=>{});},8),/读取超时/);
  assert.equal(signal.aborted,true);
});
test('a completed read returns immediately and clears the timer',async()=>{
  let signal;
  const data=await readPanelData(options=>{signal=options.signal;return {ok:true};},20);
  assert.deepEqual(data,{ok:true});assert.equal(signal.aborted,false);
});
