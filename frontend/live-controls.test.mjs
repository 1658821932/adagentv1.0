import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

const source = await readFile(new URL('./app.js', import.meta.url), 'utf8');
test('opening a live historical run restores stop controls; terminal polling hides them', () => {
  const hidden = new Set(['hidden']);
  const context = vm.createContext({
    running:false, send:{}, prompt:{}, status:{}, stopBtn:{classList:{toggle(name, value){ value ? hidden.add(name) : hidden.delete(name); }}},
    showRunModel(){}, setWorkbenchState(){}, phaseNarrative(){ return '执行中'; }, statusLabel:d=>d.status,
    phaseEl:{}, activityEl:{}, progressEl:{classList:{contains(){return false;},add(){}},querySelector(){return null;}},
    answerEl:{}, setProgressMeta(){}, answerCard(){return '';}, toolScrollSpy:{schedule(){}},
  });
  vm.runInContext(source.slice(source.indexOf('function setRunning('),source.indexOf('function showRunModel('))
    + source.slice(source.indexOf('function updateRunView('),source.indexOf('// 运行中的计时')),context);
  context.updateRunView({status:'running',steps:[]});
  assert.equal(hidden.has('hidden'),false);
  assert.equal(context.send.disabled,true);
  assert.equal(context.prompt.disabled,true);
  context.updateRunView({status:'interrupted',steps:[]});
  assert.equal(hidden.has('hidden'),true);
  assert.equal(context.send.disabled,false);
});

test('scroll following respects a user reading older messages', () => {
  const context=vm.createContext({followLatest:false,messages:{scrollTop:40,scrollHeight:900}});
  vm.runInContext(source.slice(source.indexOf('function scroll(){'),source.indexOf('function message(')),context);
  context.scroll(); assert.equal(context.messages.scrollTop,40);
  context.followLatest=true;context.scroll();assert.equal(context.messages.scrollTop,900);
});
