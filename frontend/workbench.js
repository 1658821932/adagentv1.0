const select = id => document.getElementById(id);
const escape = value => String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const calls = new Map();
let currentKey = null;

export function resetWorkbench(){
  calls.clear();
  currentKey = null;
  select('flow-log').replaceChildren();
  setWorkbenchState('等待任务','输入指令，开始智能体工作流','idle');
}
export function setWorkbenchState(title,description,state='running'){
  select('run-title').textContent=title;select('run-description').textContent=description;
  select('run-title').closest('.run-state').dataset.state=state;
}
export function setCurrentCall(key){
  if (currentKey === key) return;
  const previous = calls.get(currentKey);
  if (previous) {
    previous.classList.remove('is-current');
    previous.removeAttribute('aria-current');
  }
  currentKey = key;
  const item = calls.get(key);
  if (!item) return;
  item.classList.add('is-current');
  item.setAttribute('aria-current','true');
  const log = select('flow-log');
  const itemRect = item.getBoundingClientRect();
  const logRect = log.getBoundingClientRect();
  if (itemRect.top < logRect.top) log.scrollTop += itemRect.top - logRect.top - 6;
  else if (itemRect.bottom > logRect.bottom) log.scrollTop += itemRect.bottom - logRect.bottom + 6;
}
// key: `step_${序号}_${工具名}`，同一工具可被多次调用
// meta: 追加到状态小字后面的额外信息（如“思考 5.2s”）
export function updateCall(key,name,tool,state,onSelect,meta){
  let item=calls.get(key);
  if(!item){
    if(!calls.size)select('flow-log').replaceChildren();
    item=document.createElement('button');item.className='flow-item';
    item.innerHTML=`<span class="flow-marker"></span><span class="flow-item-body"><b>${escape(name)}</b><code>${escape(tool)}</code><small class="flow-item-status"></small></span>`;
    select('flow-log').append(item);calls.set(key,item);
  }
  item.dataset.state=state;
  item.querySelector('.flow-marker').textContent=state==='done'?'✓':state==='running'?'◌':state==='error'?'✕':'✧';
  const statusText={selected:'已选择工具',running:'执行中',done:'执行完成',error:'执行失败'}[state];
  item.querySelector('.flow-item-status').textContent=meta?`${statusText} · ${meta}`:statusText;
  if(onSelect)item.onclick=onSelect;
}
