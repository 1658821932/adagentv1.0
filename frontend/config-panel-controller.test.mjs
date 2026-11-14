import test from 'node:test';
import assert from 'node:assert/strict';
import { createConfigPanelController } from './config-panel-controller.mjs';

const config={active_profile:'real',profiles:[{name:'real',model:'test'}],max_steps:20,decision_correction_attempts:2};
const deferred=() => { let resolve,reject; const promise=new Promise((yes,no)=>{resolve=yes;reject=no;}); return {promise,resolve,reject}; };
const tick=async()=>{await Promise.resolve();await Promise.resolve();};
function fixture(read=async()=>config,write=async()=>({ok:true})) {
  const classes=new Set(['hidden']), message={}, saveButton={}, fills=[], timers=[], calls=[];
  let time=100, value=null, enabled=false;
  const controller=createConfigPanelController({
    modal:{classList:{remove:name=>classes.delete(name),add:name=>classes.add(name)}},message,saveButton,
    fill:cfg=>{fills.push(cfg);value=cfg.active_profile;},clear:()=>{value=null;},
    setEditable:on=>{enabled=on;},read:()=>{calls.push('GET');return read();},
    write:body=>{calls.push('POST');return write(body);},now:()=>time,later:fn=>timers.push(fn),
  });
  return {controller,classes,message,saveButton,fills,timers,calls,get value(){return value;},set value(v){value=v;},
    get enabled(){return enabled;},advance:()=>{time+=16000;}};
}

test('first open reveals panel before a slow GET resolves, and blocks save',async()=>{
  const request=deferred(),f=fixture(()=>request.promise);
  const opening=f.controller.open();
  assert.equal(f.classes.has('hidden'),false);assert.equal(f.saveButton.disabled,true);
  assert.match(f.message.textContent,/正在读取/);
  assert.equal(await f.controller.save({}),false);
  request.resolve(config);await opening;
  assert.equal(f.value,'real');assert.equal(f.enabled,true);
  assert.deepEqual(f.calls,['GET']);
});
test('fresh cache and optional prefetch avoid a GET on reopening',async()=>{
  const f=fixture();await f.controller.prefetch();await f.controller.open();
  f.controller.close();await f.controller.open();assert.deepEqual(f.calls,['GET']);
});
test('repeated clicks share one GET and preserve unsaved edits after loading',async()=>{
  const request=deferred(),f=fixture(()=>request.promise);
  const one=f.controller.open(),two=f.controller.open();await tick();
  assert.deepEqual(f.calls,['GET']);request.resolve(config);await Promise.all([one,two]);
  assert.equal(f.fills.length,1);f.value='unsaved';await f.controller.open();assert.equal(f.value,'unsaved');
});
test('stale cache is immediately visible but read-only until refresh',async()=>{
  const request=deferred();let reads=0;const f=fixture(()=>++reads===1?config:request.promise);
  await f.controller.open();f.controller.close();f.advance();const opening=f.controller.open();
  assert.equal(f.value,'real');assert.equal(f.enabled,false);
  request.resolve({...config,max_steps:30});await opening;assert.equal(f.fills.at(-1).max_steps,30);
});
test('failed refresh clears stale fields and never enables invented defaults',async()=>{
  let reads=0;const f=fixture(async()=>{if(++reads>1)throw new Error('HTTP 404');return config;});
  await f.controller.open();f.controller.close();f.advance();await f.controller.open();
  assert.equal(f.value,null);assert.equal(f.saveButton.disabled,true);assert.match(f.message.textContent,/保存已禁用/);
  assert.equal(await f.controller.save({}),false);assert.deepEqual(f.calls,['GET','GET']);
});
test('a late GET response after closing cannot change the form or enable save',async()=>{
  const request=deferred(),f=fixture(()=>request.promise);const opening=f.controller.open();
  f.controller.close();request.resolve(config);await opening;
  assert.equal(f.fills.length,0);assert.equal(f.saveButton.disabled,true);assert.equal(f.classes.has('hidden'),true);
});
test('only the latest reopen applies a shared in-flight response',async()=>{
  const request=deferred(),f=fixture(()=>request.promise);const old=f.controller.open();
  f.controller.close();const latest=f.controller.open();request.resolve(config);await Promise.all([old,latest]);
  assert.equal(f.fills.length,1);assert.deepEqual(f.calls,['GET']);assert.equal(f.enabled,true);
});
test('save deduplicates writes and invalidates cached configuration',async()=>{
  const request=deferred(),f=fixture(undefined,()=>request.promise);await f.controller.open();
  const saving=f.controller.save(()=>({max_steps:30}));assert.equal(await f.controller.save({}),false);
  request.resolve({ok:true});assert.equal(await saving,true);
  f.controller.close();await f.controller.open();assert.deepEqual(f.calls,['GET','POST','GET']);
  f.timers[0]();assert.equal(f.classes.has('hidden'),false);
});
test('failed save preserves edits for correction without retrying POST automatically',async()=>{
  const f=fixture(undefined,async()=>{throw new Error('network');});await f.controller.open();f.value='edited';
  assert.equal(await f.controller.save({}),false);assert.equal(f.value,'edited');assert.equal(f.enabled,true);
  assert.deepEqual(f.calls,['GET','POST']);
});
test('a malformed config response cannot be saved',async()=>{
  const f=fixture(async()=>({}));await f.controller.open();assert.equal(f.saveButton.disabled,true);
  assert.equal(await f.controller.save({}),false);assert.match(f.message.textContent,/缺少有效模型/);
});

test('late response from a timed-out read cannot replace a successful retry cache',async()=>{
  const old=deferred();let reads=0,value;
  const controller=createConfigPanelController({modal:{classList:{add(){},remove(){}}},message:{},saveButton:{},
    fill:cfg=>{value=cfg.active_profile;},clear(){},setEditable(){},readTimeoutMs:8,
    read:()=>++reads===1?old.promise:{...config,active_profile:'new',profiles:[{name:'new',model:'test'}]},write:async()=>({})});
  await controller.open();await controller.open();assert.equal(value,'new');
  old.resolve(config);await tick();controller.close();await controller.open();assert.equal(value,'new');assert.equal(reads,2);
});
