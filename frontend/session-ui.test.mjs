/** Run the real composer/reset handlers with DOM stubs, without a paid Agent. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { submitConversation } from './conversation.js';

const source=await readFile(new URL('./app.js',import.meta.url),'utf8');
const start=source.indexOf('async function run(text){');
const end=source.indexOf('/* ---------- 历史运行 ---------- */',start);
const clearStart=source.indexOf("document.querySelector('#clear').addEventListener('click', () => {");
const clearEnd=source.indexOf('\n});',clearStart)+4;
assert.ok(start>=0 && end>start && clearStart>=0 && clearEnd>clearStart);

function fixture(request) {
  const calls=[], views=[], messages=[], handlers={};
  const classList={add(){},remove(){},toggle(){}};
  const context=vm.createContext({
    document:{querySelector:selector=>({remove(){},addEventListener:(event,handler)=>{handlers[selector+':'+event]=handler;}})},
    prompt:{value:'',placeholder:'',focus(){}},stopBtn:{disabled:false},replayBtn:{classList},demoBtn:{classList},
    messages:{replaceChildren(){}},api:async(...args)=>{calls.push(args);return request(...args);},
    submitConversation,escape:String,message:(...args)=>messages.push(args),
    setRunning:value=>{context.running=value;},
    beginRunView:detail=>views.push({...detail,continued:false}),
    continueRunView:detail=>views.push({...detail,continued:true}),
    unmountReplayBar(){},unloadArtifactPreview(){},toolScrollSpy:{reset(){}},closeHistory(){},loadHistory(){},loadSelectedModel(){},poll(){},clearTimeout(){},resetWorkbench(){},
    playDemo:()=>{context.demoPlayed=true;},
    running:false,demoArmed:false,viewingRun:'run_20261112_060801_663582',replayToken:0,pollTimer:null,
  });
  vm.runInContext(source.slice(start,end)+'\n'+source.slice(clearStart,clearEnd),context);
  return {context,calls,views,messages,handlers};
}

test('composer sends selected parent, loads child details, and guards against a second click',async()=>{
  const child='run_20261112_100000_123456';
  const f=fixture(async(path,options)=>{
    if(options?.method==='POST') return {run_id:child,parent_run_id:'run_20261112_060801_663582'};
    return {continuation_supported:true,task:'fixture',status:'running',parent_run_id:'run_20261112_060801_663582'};
  });
  await f.context.run('修复好了，请继续');
  const post=f.calls.find(([,opts])=>opts?.method==='POST');
  assert.deepEqual(JSON.parse(post[1].body),{task:'修复好了，请继续',resume_from:'run_20261112_060801_663582'});
  assert.equal(f.context.viewingRun,child);
  assert.equal(f.views[0].parent_run_id,'run_20261112_060801_663582');
  assert.equal(f.views[0].continued,true);
  assert.equal(f.context.stopBtn.disabled,false);
  await f.context.run('重复发送');
  assert.equal(f.calls.filter(([,opts])=>opts?.method==='POST').length,1);
});

test('continuation appends an agent response without clearing or duplicating the user message',()=>{
  const start=source.indexOf('function mountAgentResponse(');
  const end=source.indexOf('/* ---------- 重放',start);
  const calls=[];
  const context=vm.createContext({
    messages:{replaceChildren(){throw new Error('continuation cleared the conversation');}},
    prompt:{placeholder:''},viewingRun:'run_20261112_100000_123456',escape:String,showRunModel(){},
    resetWorkbench(){throw new Error('continuation reset the workbench');},
    message(role,html){
      calls.push({role,html});
      return {querySelector:selector=>({selector})};
    },
  });
  vm.runInContext(source.slice(start,end),context);
  context.continueRunView({task:'继续修复'});
  assert.equal(calls.length,1);
  assert.equal(calls[0].role,'assistant');
  assert.match(calls[0].html,/保留以上执行记录/);
  assert.equal(context.streamEl.selector,'.tool-stream');
});

test('failed follow-up preserves message and parent, and does not fall back to a fresh task',async()=>{
  const f=fixture(async(path,options)=>{
    if(options?.method==='POST')throw new Error('fixture startup error');
    return {continuation_supported:true};
  });
  await f.context.run('重试攻击');
  assert.equal(f.context.prompt.value,'重试攻击');
  assert.equal(f.context.viewingRun,'run_20261112_060801_663582');
  assert.equal(f.context.running,false);
  assert.equal(f.calls.filter(([,opts])=>opts?.method==='POST').length,1);
});

test('new conversation reset omits resume_from; demo still does not submit a real job',async()=>{
  const f=fixture(async(path,options)=>options?.method==='POST'
    ? {run_id:'run_20261112_100000_123456',parent_run_id:null} : {task:'new',status:'running'});
  f.handlers['#clear:click']();
  assert.equal(f.context.viewingRun,null);
  await f.context.run('全新任务');
  assert.deepEqual(JSON.parse(f.calls[0][1].body),{task:'全新任务'});
  const demo=fixture(async()=>{throw new Error('demo must not request APIs');});
  demo.context.demoArmed=true;
  await demo.context.run('播放示例');
  assert.equal(demo.calls.length,0);
  assert.equal(demo.context.demoPlayed,true);
});

test('finished/error jobs without measured duration do not keep a live timer',()=>{
  const start=source.indexOf('function stepTimes(s){');
  const end=source.indexOf('function stepStateHtml(s){',start);
  const context=vm.createContext({fmtDuration:value=>String(value)});
  vm.runInContext(source.slice(start,end),context);
  assert.match(context.stepTimes({status:'running',elapsed_seconds:5400}),/step-time-live/);
  assert.doesNotMatch(context.stepTimes({status:'error',elapsed_seconds:5400}),/step-time-live/);
  assert.match(context.stepTimes({status:'error',elapsed_seconds:5400,output:{error:{type:'job_interrupted'}}}),/实际耗时未知/);
  assert.doesNotMatch(context.stepTimes({status:'success',duration:10,elapsed_seconds:5400}),/step-time-live/);
});

test('error card explains the native validation blocker without a continuation shortcut',()=>{
  const start=source.indexOf('function errorPresentation(');
  const end=source.indexOf('// 顶栏只讲',start);
  const context=vm.createContext({escape:String,firstSentence:text=>String(text).slice(0,140),statusLabel:()=> '运行出错'});
  vm.runInContext(source.slice(start,end),context);
  const detail='Native execution requires config/checkpoint/model/data_type';
  const html=context.answerCard({status:'error',error:detail},detail,null);
  assert.match(html,/07 策略评测未启动/);
  assert.match(html,/WorldEngine 尚未执行/);
  assert.doesNotMatch(html,/data-action="continue-error"/);
  assert.match(html,/查看详细错误/);
});

test('connection error card reports only the retries recorded for this run',()=>{
  const start=source.indexOf('function errorPresentation(');
  const end=source.indexOf('// 顶栏只讲',start);
  const context=vm.createContext({escape:String,firstSentence:text=>String(text).slice(0,140),statusLabel:()=> '运行出错'});
  vm.runInContext(source.slice(start,end),context);
  const detail='APIConnectionError: Connection error.';
  const retried=context.answerCard({status:'error',error:detail,summary:{stages:[{stage:'main_agent',retry_requests:3}]}},detail,null);
  assert.match(retried,/模型服务连接失败/);
  assert.match(retried,/已自动重试 3 次/);
  const legacy=context.answerCard({status:'error',error:detail},detail,null);
  assert.doesNotMatch(legacy,/已自动重试/);
});

test('subagent rows translate FOCI actions and expose elapsed time',()=>{
  const start=source.indexOf('const ACTIVITY_LABELS');
  const end=source.indexOf('function fillToolActivities',start);
  const context=vm.createContext({escape:String,fmtDuration:value=>`${value}s`,Date});
  vm.runInContext(source.slice(start,end),context);
  const html=context.activityRow({kind:'operation',name:'ego.foci',status:'returned',finished_at:'done',elapsed_seconds:12,summary:'生成 2 个候选'});
  assert.match(html,/FOCI 优化/);
  assert.match(html,/生成 2 个候选/);
  assert.match(html,/12s/);
});

test('subagent invalid, fallback and preview states are visually distinct',()=>{
  const start=source.indexOf('const ACTIVITY_LABELS');
  const end=source.indexOf('function fillToolActivities',start);
  const context=vm.createContext({escape:String,fmtDuration:value=>value<0.1?`${Math.round(value*1000)}ms`:`${value}s`,Date});
  vm.runInContext(source.slice(start,end),context);
  const invalid=context.activityRow({kind:'tool',name:'construct_attack_candidates',status:'invalid',elapsed_seconds:0.006,summary:'invalid'});
  assert.match(invalid,/failed/);
  assert.match(invalid,/参数无效/);
  assert.match(invalid,/6ms/);
  const fallback=context.activityRow({kind:'tool',name:'construct_attack_candidates',status:'fallback',summary:'fallback'});
  assert.match(fallback,/warning/);
  assert.match(fallback,/未完全达标/);
  const preview=context.activityRow({kind:'tool',name:'edit_speed_profile',status:'preview',summary:'preview'});
  assert.match(preview,/preview/);
  assert.match(preview,/○/);
});

test('history loader reads the complete parent chain in chronological order',async()=>{
  const start=source.indexOf('async function loadConversationChain(');
  const end=source.indexOf('async function openRun(',start);
  const context=vm.createContext({api:async path=>({
    run_id:path.split('/').at(-1),
    parent_run_id:path.endsWith('child')?'parent':null,
  })});
  vm.runInContext(source.slice(start,end),context);
  const chain=await context.loadConversationChain('child');
  assert.deepEqual(Array.from(chain,item=>item.run_id),['parent','child']);
});

test('run model label uses the actual model and reasoning snapshots',()=>{
  const start=source.indexOf('function showRunModel(');
  const end=source.indexOf('async function loadSelectedModel',start);
  const modelLabel={textContent:'',title:''};
  const context=vm.createContext({modelLabel});
  vm.runInContext(source.slice(start,end),context);
  context.showRunModel({llm_model:'kimi-k3',llm_reasoning_effort:'low'});
  assert.equal(modelLabel.textContent,'主 Agent · kimi-k3 · 思考 low');
  assert.match(modelLabel.title,/实际思考强度：low/);
  context.showRunModel({llm_model:'kimi-k3'});
  assert.match(modelLabel.textContent,/思考 未记录/);
});


test('stop waits for cleanup and refreshes the terminal run state',async()=>{
  const start=source.indexOf("stopBtn.addEventListener('click'");
  const end=source.indexOf("document.querySelector('#example')",start);
  const calls=[];
  let click;
  const context=vm.createContext({
    viewingRun:'run_fixture',stopBtn:{disabled:false,addEventListener:(name,handler)=>{click=handler;}},
    status:{textContent:''},api:async(...args)=>{calls.push(args);return {ok:true};},
    poll:async()=>{calls.push(['poll']);},
  });
  vm.runInContext(source.slice(start,end),context);
  await click();
  assert.equal(calls[0][0],'/api/runs/run_fixture/stop');
  assert.equal(calls[0][1].method,'POST');
  assert.equal(calls[1][0],'poll');
  assert.equal(context.stopBtn.disabled,false);
});
