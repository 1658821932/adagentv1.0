/** Exercise the production app wiring with the real config controller. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { createConfigPanelController } from './config-panel-controller.mjs';
import { panelLoadingHtml, panelErrorHtml } from './panel-loading.mjs';

const source = await readFile(new URL('./app.js', import.meta.url), 'utf8');
const start = source.indexOf('function clearConfigFields(){');
const end = source.indexOf('\ndemoBtn?', start);
assert.ok(start >= 0 && end > start, 'production config wiring must remain explicit');
const config = { active_profile:'fixture', profiles:[{name:'fixture',model:'offline'}],
  reasoning_effort:'high', max_steps:30, decision_correction_attempts:2, tool_modes:{} };

function fixture(request, {timeoutMs=12000} = {}) {
  const events = {}, classes = new Set(['hidden']), calls = [], applied = [], timers = [];
  const ids=['cfg-profile','cfg-profile-info','cfg-tool-profiles','cfg-tools','cfg-max-steps',
    'cfg-corrections','cfg-effort','cfg-save','cfg-loading','cfg-fields','config-btn','config-close','cfg-cancel'];
  const elements=Object.fromEntries(ids.map(id=>['#'+id,{value:'stale',innerHTML:'stale',disabled:false,classList:{toggle(){},add(){},remove(){}},
    addEventListener:(event,handler)=>{events[id+':'+event]=handler;}}]));
  const controls=ids.filter(id=>id.startsWith('cfg-') && id!=='cfg-save').map(id=>elements['#'+id]);
  const modeRow={dataset:{tool:'attack_vehicle_editor'},querySelector:()=>({dataset:{mode:'real'}})};
  const profileRow={dataset:{component:'attack_editor'},querySelector:selector=>({value:selector==='.cfg-tool-effort'?'medium':'active_profile'})};
  const context=vm.createContext({
    document:{querySelector:selector=>elements[selector],querySelectorAll:selector=>
      selector.startsWith('#config-modal')?controls:selector==='.cfg-toggle'?[modeRow]:selector==='.cfg-tool-profile'?[profileRow]:[]},
    cfgLoading:elements['#cfg-loading'],cfgFields:elements['#cfg-fields'],panelLoadingHtml,panelErrorHtml,
    cfgSave:elements['#cfg-save'],cfgMsg:{textContent:'',className:''},
    modal:{dataset:{},setAttribute(){},classList:{remove:name=>classes.delete(name),add:name=>classes.add(name)},
      addEventListener:(event,handler)=>{events['modal:'+event]=handler;}},
    createConfigPanelController:options=>createConfigPanelController({...options,readTimeoutMs:timeoutMs,later:fn=>timers.push(fn)}),
    api:async(...args)=>{calls.push(args);return request(...args);},
    fillConfig:cfg=>{applied.push(cfg);for(const [id,key] of [['cfg-profile','active_profile'],['cfg-effort','reasoning_effort'],
      ['cfg-max-steps','max_steps'],['cfg-corrections','decision_correction_attempts']])elements['#'+id].value=String(cfg[key]);},
  });
  vm.runInContext(source.slice(start,end),context);
  return {context,elements,events,classes,calls,applied,timers,open:()=>elements['#config-btn'].onclick(),save:()=>elements['#cfg-save'].onclick()};
}

test('production click reveals panel before the GET resolves and disables saving',async()=>{
  let finish;const f=fixture(()=>new Promise(resolve=>{finish=resolve;}));
  const opening=f.open();
  assert.equal(f.classes.has('hidden'),false);assert.equal(f.elements['#cfg-save'].disabled,true);
  assert.equal(f.elements['#cfg-profile'].disabled,true);assert.match(f.context.cfgMsg.textContent,/正在读取/);
  assert.match(f.elements['#cfg-loading'].innerHTML,/panel-skeleton/);
  await Promise.resolve();assert.equal(await f.save(),false);assert.equal(f.calls.length,1);
  finish(config);await opening;assert.equal(f.elements['#cfg-save'].disabled,false);
});

test('production failure clears stale fields and cannot POST defaults',async()=>{
  const f=fixture(async()=>{throw new Error('HTTP 404');});await f.open();
  assert.equal(f.elements['#cfg-profile'].innerHTML,'');assert.equal(f.elements['#cfg-tool-profiles'].innerHTML,'');
  assert.equal(f.elements['#cfg-max-steps'].value,'');assert.match(f.context.cfgMsg.textContent,/保存已禁用/);
  assert.equal(await f.save(),false);assert.equal(f.calls.length,1);
});

test('actual save wiring preserves model, reasoning, modes and numeric fields',async()=>{
  const f=fixture(async(path,options)=>options?.method==='POST'?{ok:true}:config);
  await f.open();f.elements['#cfg-profile'].value='changed';await f.save();
  const body=JSON.parse(f.calls[1][1].body);
  assert.equal(body.active_profile,'changed');assert.equal(body.reasoning_effort,'high');
  assert.equal(body.max_steps,30);assert.equal(body.decision_correction_attempts,2);
  assert.deepEqual(body.tool_profiles,{attack_editor:'active_profile'});
  assert.deepEqual(body.tool_efforts,{attack_editor:'medium'});
  assert.deepEqual(body.tool_modes,{attack_vehicle_editor:'real'});
});

test('hover/focus prefetch and reopen share the cached GET through actual handlers',async()=>{
  const f=fixture(async()=>config);
  f.events['config-btn:pointerenter']();f.events['config-btn:focus']();await f.open();
  f.elements['#cfg-cancel'].onclick();await f.open();assert.equal(f.calls.length,1);
});

test('closing via backdrop prevents a late response from enabling the old panel',async()=>{
  let finish;const f=fixture(()=>new Promise(resolve=>{finish=resolve;}));const opening=f.open();
  f.events['modal:click']({target:f.context.modal});await Promise.resolve();finish(config);await opening;
  assert.equal(f.classes.has('hidden'),true);assert.equal(f.applied.length,0);assert.equal(await f.save(),false);
});

test('timeout replaces the skeleton with an actionable retry and then recovers',async()=>{
  let reads=0;
  const f=fixture(()=>++reads===1?new Promise(()=>{}):config,{timeoutMs:8});
  await f.open();assert.match(f.elements['#cfg-loading'].innerHTML,/data-config-retry/);
  assert.match(f.elements['#cfg-loading'].innerHTML,/读取超时/);assert.equal(f.elements['#cfg-save'].disabled,true);
  f.events['cfg-loading:click']({target:{closest:()=>({})}});
  await f.open();assert.equal(f.calls.length,2);assert.equal(f.elements['#cfg-save'].disabled,false);
});
