export function chooseCurrentTool(viewport, entries) {
  if (!entries.length) return null;
  const line = viewport.top + Math.min(180, viewport.height * 0.3);
  let previous = null;
  for (const entry of entries) {
    if (entry.top <= line && entry.bottom >= line) return entry.key;
    if (entry.top > line) return previous?.key || entry.key;
    previous = entry;
  }
  return previous?.key || null;
}

export function createToolScrollSpy(scroller, onCurrent) {
  const cards = new Map();
  let scheduled = false;
  let current = null;
  let suppressUntil = 0;
  let resizeObserver = null;

  const update = () => {
    scheduled = false;
    if (Date.now() < suppressUntil) return;
    const bounds = scroller.getBoundingClientRect();
    const entries = [];
    for (const [card, key] of cards) {
      if (!card.isConnected) continue;
      const rect = card.getBoundingClientRect();
      entries.push({ key, top: rect.top, bottom: rect.bottom });
    }
    const next = chooseCurrentTool({ top: bounds.top, height: bounds.height }, entries);
    if (next !== current) {
      current = next;
      onCurrent(current);
    }
  };
  const schedule = () => {
    if (scheduled) return;
    scheduled = true;
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(update);
    else setTimeout(update, 16);
  };
  const observe = (card, key) => {
    cards.set(card, key);
    resizeObserver?.observe(card);
    schedule();
  };
  const select = key => {
    suppressUntil = Date.now() + 450;
    current = key;
    onCurrent(key);
  };
  const reset = () => {
    cards.clear();
    resizeObserver?.disconnect();
    current = null;
    suppressUntil = 0;
    onCurrent(null);
  };

  scroller.addEventListener('scroll', schedule, { passive: true });
  if (typeof ResizeObserver === 'function') {
    resizeObserver = new ResizeObserver(schedule);
  }
  return { observe, select, reset, schedule };
}
