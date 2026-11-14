// Resolve API and media links beside the served app, including forwarded prefixes.
export function createApiClient(moduleUrl, fetchImpl = (...args) => fetch(...args)) {
  const base = new URL('.', moduleUrl);
  const serviceUrl = path => {
    const value = String(path ?? '');
    return /^\/?api(?:\/|\?|$)/.test(value)
      ? new URL(value.replace(/^\//, ''), base).href
      : value;
  };

  async function request(path, options = {}) {
    const destination = serviceUrl(path);
    const method = String(options.method || 'GET').toUpperCase();
    const mayStartWork = !['GET', 'HEAD', 'OPTIONS'].includes(method);
    let response;
    try {
      response = await fetchImpl(destination, { ...options, cache: 'no-store' });
    } catch (cause) {
      const target = new URL(destination, base);
      const error = new Error(`无法连接前端服务（${method} ${target.pathname}）。请强制刷新页面；若仍失败，请从 VSCode“端口”重新打开 5173。`
        + (mayStartWork ? '请求可能已经送达；请先检查历史运行，避免重复提交。' : '本次读取未发起新任务。'));
      error.cause = cause;
      error.code = 'frontend_network_error';
      error.request_method = method;
      error.request_path = target.pathname;
      error.may_have_started = mayStartWork;
      throw error;
    }
    const data = await response.json().catch(() => null);
    if (!response.ok) {
      const hint = response.status === 404 && /^\/?api\/config(?:\?|$)/.test(path)
        ? '：当前访问地址没有配置接口，请确认端口转发连接到 server.mjs 服务'
        : '';
      throw new Error(data?.error || `HTTP ${response.status}${hint}`);
    }
    if (data === null) throw new Error('接口未返回 JSON，请确认访问的是前端服务而非静态预览');
    return data;
  }

  return { request, serviceUrl };
}
