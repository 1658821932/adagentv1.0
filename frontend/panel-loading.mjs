/** Shared, bounded GET loading and accessible panel feedback. */
const escape = value => String(value ?? '').replace(/[&<>"']/g,
  char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));

export function panelLoadingHtml(label, rows = 0) {
  return `<div class="panel-loading" role="status"><span class="panel-spinner" aria-hidden="true"></span>`
    + `<span>${escape(label)}</span></div>`
    + (rows ? `<div class="panel-skeleton" aria-hidden="true">${Array.from({length:rows},()=>'<i></i>').join('')}</div>` : '');
}

export function panelErrorHtml(message, retryAttribute, cached = false) {
  const attribute = ['data-history-retry','data-config-retry'].includes(retryAttribute) ? retryAttribute : 'data-panel-retry';
  return `<div class="panel-load-error" role="alert"><span>${cached ? '暂时显示上次结果。' : ''}${escape(message)}</span>`
    + `<button type="button" ${attribute}>重新加载</button></div>`;
}

export async function readPanelData(read, timeoutMs = 12000) {
  const controller = new AbortController();
  let timer;
  const deadline = new Promise((_, reject) => {
    timer = setTimeout(() => {
      reject(new Error('读取超时，请重新加载'));
      controller.abort();
    }, timeoutMs);
  });
  try {
    return await Promise.race([Promise.resolve().then(() => read({signal:controller.signal})), deadline]);
  } finally { clearTimeout(timer); }
}
