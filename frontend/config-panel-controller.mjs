/** Responsive config loading, independent of app.js and server.mjs wiring. */
import { readPanelData } from './panel-loading.mjs';
export function createConfigPanelController({
  modal, message, saveButton, fill, clear, setEditable, read, write,
  now = () => Date.now(), ttlMs = 15000, later = (fn, ms) => setTimeout(fn, ms),
  onState = () => {}, readTimeoutMs = 12000,
}) {
  let cache = null, fetchedAt = 0, pending = null;
  let version = 0, loaded = false, opened = false, saving = false;

  function editable(enabled) {
    setEditable(enabled);
    saveButton.disabled = !enabled;
  }
  function say(text, className = '') {
    message.textContent = text;
    message.className = className;
  }
  function invalidate() { cache = null; fetchedAt = 0; }
  function fresh() { return cache !== null && now() - fetchedAt < ttlMs; }
  function fetchConfig() {
    if (!pending) {
      pending = readPanelData(read, readTimeoutMs).then(cfg => {
        if (!cfg || typeof cfg !== 'object' || !Array.isArray(cfg.profiles)
          || !cfg.profiles.some(profile => profile.name === cfg.active_profile)
          || !Number.isInteger(cfg.max_steps) || !Number.isInteger(cfg.decision_correction_attempts)) {
          throw new Error('服务器配置缺少有效模型或运行参数');
        }
        cache = cfg; fetchedAt = now();
        return cfg;
      }).finally(() => { pending = null; });
    }
    return pending;
  }
  async function prefetch() {
    if (fresh()) return cache;
    return fetchConfig();
  }
  async function open() {
    if (saving || (opened && loaded)) return;
    opened = true;
    const generation = ++version;
    modal.classList.remove('hidden');
    loaded = false;
    if (cache) fill(cache); else clear();
    editable(false);
    if (fresh()) {
      loaded = true; editable(true);
      say('已加载当前配置');
      onState('ready');
      return;
    }
    say('正在读取当前配置…');
    onState('loading', { cached:cache !== null });
    try {
      const cfg = await fetchConfig();
      if (generation !== version || !opened) return;
      fill(cfg); loaded = true; editable(true);
      say('已加载当前生效配置，不修改可直接关闭');
      onState('ready');
    } catch (error) {
      if (generation !== version || !opened) return;
      invalidate(); clear(); editable(false);
      say('读取服务器配置失败，保存已禁用：' + error.message, 'err');
      onState('error', { message:error.message });
    }
  }
  function close() {
    ++version; opened = false; loaded = false;
    editable(false);
    modal.classList.add('hidden');
    onState('closed');
  }
  async function save(collect) {
    if (!opened || !loaded || saving) return false;
    saving = true;
    editable(false);
    const generation = version;
    say('保存中…');
    onState('saving');
    try {
      await write(typeof collect === 'function' ? collect() : collect);
      invalidate();
      if (generation === version && opened) {
        say('✓ 已保存，对之后启动的运行生效', 'ok');
        later(() => { if (generation === version) close(); }, 900);
      }
      return true;
    } catch (error) {
      if (generation === version && opened) say('保存失败：' + error.message, 'err');
      return false;
    } finally {
      saving = false;
      if (generation === version && opened) { editable(loaded); onState('ready'); }
    }
  }
  return { open, close, save, prefetch, invalidate };
}
