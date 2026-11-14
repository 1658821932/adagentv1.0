import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { panelLoadingHtml, panelErrorHtml, readPanelData } from './panel-loading.mjs';
import { chooseCurrentTool } from './tool-scroll-spy.js';
import { effectiveReasoning, reasoningOptions, reasoningSelection } from './reasoning.mjs';

const appSource = await readFile(new URL('./app.js', import.meta.url), 'utf8');

test('an interrupted run is not attributed to a manual stop without evidence', () => {
  const start=appSource.indexOf('function statusLabel(');
  const end=appSource.indexOf('async function poll(',start);
  const context=vm.createContext({});
  vm.runInContext(appSource.slice(start,end),context);
  assert.equal(context.statusLabel({status:'interrupted'}),'已中断');
});

test('scroll spy chooses the card at the reading line and retains the previous card in gaps', () => {
  const viewport = { top: 100, height: 600 };
  const cards = [
    { key: 'one', top: 80, bottom: 220 },
    { key: 'two', top: 400, bottom: 700 },
    { key: 'three', top: 900, bottom: 1100 },
  ];
  assert.equal(chooseCurrentTool(viewport, []), null);
  assert.equal(chooseCurrentTool(viewport, cards), 'one');
  assert.equal(chooseCurrentTool(viewport, cards.map(card => ({ ...card, top: card.top + 300, bottom: card.bottom + 300 }))), 'one');
  assert.equal(chooseCurrentTool(viewport, [{ key: 'last', top: 700, bottom: 900 }]), 'last');
});

test('reasoning options expose official values without changing existing mappings', () => {
  const kimi = { request_dialect: 'kimi', reasoning_effort: 'high' };
  assert.deepEqual(reasoningOptions(kimi).map(option => option.native), ['low', 'high', 'max']);
  assert.equal(effectiveReasoning(kimi, 'medium'), 'high');
  assert.equal(effectiveReasoning(kimi, 'high'), 'max');
  assert.equal(effectiveReasoning(kimi, 'max'), 'max');
  assert.equal(reasoningSelection(kimi, 'high'), 'max');
  assert.equal(reasoningSelection(kimi, 'medium'), 'medium');

  const qwen = { request_dialect: 'qwen', reasoning_effort: 'medium' };
  assert.deepEqual(reasoningOptions(qwen).map(option => option.native), ['low', 'medium', 'xhigh']);
  assert.equal(effectiveReasoning(qwen, 'high'), 'xhigh');
  assert.equal(reasoningSelection(qwen, 'max'), 'high');

  const deepseek = { request_dialect: 'thinking', reasoning_effort: 'max' };
  assert.deepEqual(reasoningOptions(deepseek).map(option => option.native), ['low', 'high', 'max']);
  assert.equal(effectiveReasoning(deepseek, 'medium'), 'high');
});

test('four-level OpenAI profiles retain all native levels and unmanaged profiles stay default', () => {
  const profile = { request_dialect: 'openai', reasoning_effort: 'medium' };
  assert.deepEqual(reasoningOptions(profile).map(option => option.value), ['low', 'medium', 'high', 'max']);
  for (const level of ['low', 'medium', 'high', 'max']) assert.equal(effectiveReasoning(profile, level), level);
  assert.deepEqual(reasoningOptions({ model: 'unmanaged' }), []);
  assert.equal(reasoningSelection({ model: 'unmanaged' }, 'max'), 'profile');
});

test('history loader reuses fresh cache, deduplicates requests and preserves markup', async () => {
  const start = appSource.indexOf('let historyConversations');
  const end = appSource.indexOf('async function openRun(', start);
  let calls = 0;
  const panel = {
    dataset: {}, attributes: {}, html: '', writes: 0,
    set innerHTML(value) { this.html = value; this.writes += 1; },
    get innerHTML() { return this.html; },
    setAttribute(name, value) { this.attributes[name] = value; },
  };
  const context = vm.createContext({
    historyPanel: panel, historyCount: { textContent: '' },
    historyItems:panel,historyFeedback:{innerHTML:'',classList:{add(){},remove(){}}},
    panelLoadingHtml,panelErrorHtml,readPanelData,escape:String,
    groupConversationRuns: runs => runs,
    historyItemHtml: run => `<button>${run.run_id}</button>`,
    api: async () => { calls += 1; return { runs: [{ run_id: 'run_one', status: 'success' }] }; },
    Date,
  });
  vm.runInContext(appSource.slice(start, end), context);
  await Promise.all([context.loadHistory(), context.loadHistory()]);
  assert.equal(calls, 1);
  assert.equal(panel.writes, 1);
  await context.loadHistory();
  assert.equal(calls, 1);
  await context.loadHistory({ force: true });
  assert.equal(calls, 2);
  assert.equal(panel.writes, 1);
  assert.equal(panel.attributes['aria-busy'], 'false');
});

test('history button opens immediately while loading continues in background', () => {
  const start = appSource.indexOf("historyBtn.addEventListener('click'");
  const end = appSource.indexOf('historyPanel.addEventListener', start);
  const classes = new Set(['hidden']);
  const panel = {
    classList: {
      contains: name => classes.has(name),
      add: name => classes.add(name),
      remove: name => classes.delete(name),
    },
    innerHTML: '',
  };
  const attributes = {};
  let loads = 0;
  let click;
  const context = vm.createContext({
    historyPanel: panel,
    historyBtn: {
      setAttribute: (name, value) => { attributes[name] = value; },
      addEventListener: (name, handler) => { click = handler; },
    },
    historyConversations: null, historyMarkup: '',
    loadHistory: () => { loads += 1; return new Promise(() => {}); },
    closeHistory() {},
  });
  vm.runInContext(appSource.slice(start, end), context);
  click({ stopPropagation() {} });
  assert.equal(classes.has('hidden'), false);
  assert.equal(attributes['aria-expanded'], 'true');
  // The real loader's skeleton and pending state are covered above.
  assert.equal(panel.innerHTML, '');
  assert.equal(loads, 1);
});

test('failed history refresh keeps cached items and exposes retry feedback',async()=>{
  const start=appSource.indexOf('let historyConversations');
  const end=appSource.indexOf('async function openRun(',start);
  const feedback={innerHTML:'',classList:{add(){},remove(){}}},items={innerHTML:''};let reads=0;
  const context=vm.createContext({historyPanel:{setAttribute(){}},historyFeedback:feedback,historyItems:items,
    historyCount:{textContent:''},panelLoadingHtml,panelErrorHtml,readPanelData,escape:String,
    historyItemHtml:run=>`<button>${run.run_id}</button>`,Date,
    api:async()=>{if(++reads===2)throw new Error('network');return {runs:[{run_id:'cached'}]};}});
  vm.runInContext(appSource.slice(start,end),context);
  await context.loadHistory();await context.loadHistory({force:true});
  assert.match(items.innerHTML,/cached/);assert.match(feedback.innerHTML,/上次结果/);assert.match(feedback.innerHTML,/data-history-retry/);
  await context.loadHistory({force:true});assert.equal(reads,3);
});
