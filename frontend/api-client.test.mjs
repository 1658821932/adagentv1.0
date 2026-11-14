import assert from 'node:assert/strict';
import test from 'node:test';
import { createApiClient } from './api-client.js';

test('direct port uses the same server for config and encoded file links', () => {
  const { serviceUrl } = createApiClient('http://localhost:5173/app.js?v=1');
  assert.equal(serviceUrl('/api/config'), 'http://localhost:5173/api/config');
  assert.equal(serviceUrl('/api/file?path=%2Fa%20b.csv'), 'http://localhost:5173/api/file?path=%2Fa%20b.csv');
});

test('forwarded paths retain the prefix for config, run polling and videos', () => {
  const { serviceUrl } = createApiClient('https://host.example/workspace/proxy/5173/app.js?v=1');
  for (const path of ['/api/config', '/api/runs/run_1', '/api/wevideo?path=%2Fvideo.mp4']) {
    assert.equal(serviceUrl(path), 'https://host.example/workspace/proxy/5173/' + path.slice(1));
  }
  assert.equal(serviceUrl('https://media.example/video.mp4'), 'https://media.example/video.mp4');
});

test('config reads bypass cached responses', async () => {
  let captured;
  const { request } = createApiClient('http://localhost:5173/app.js', async (url, options) => {
    captured = { url, options };
    return Response.json({ active_profile: 'kimi' });
  });
  assert.deepEqual(await request('/api/config'), { active_profile: 'kimi' });
  assert.equal(captured.url, 'http://localhost:5173/api/config');
  assert.equal(captured.options.cache, 'no-store');
});

test('config saves preserve method, headers and body through a forwarded path', async () => {
  let captured;
  const { request } = createApiClient('https://host.example/proxy/5173/app.js', async (url, options) => {
    captured = { url, options };
    return Response.json({ ok: true });
  });
  const options = { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{"active_profile":"kimi"}' };
  await request('/api/config', options);
  assert.equal(captured.url, 'https://host.example/proxy/5173/api/config');
  assert.deepEqual(captured.options, { ...options, cache: 'no-store' });
});

test('404 static preview has a backend-address diagnostic', async () => {
  const { request } = createApiClient('http://localhost:5173/app.js', async () => new Response('Not found', { status: 404 }));
  await assert.rejects(request('/api/config'), /HTTP 404.*server\.mjs/);
});

test('HTML fallback is rejected instead of masquerading as configuration', async () => {
  const { request } = createApiClient('http://localhost:5173/app.js', async () => new Response('<html>static preview</html>'));
  await assert.rejects(request('/api/config'), /未返回 JSON/);
});

test('network failure identifies a safe continuation preflight without starting or retrying work', async () => {
  let count = 0;
  const { request } = createApiClient('https://host.example/proxy/5173/app.js', async () => {
    count += 1;
    throw new TypeError('Failed to fetch');
  });
  await assert.rejects(request('/api/runs/run_fixture'), error => {
    assert.equal(error.code, 'frontend_network_error');
    assert.equal(error.may_have_started, false);
    assert.equal(error.request_method, 'GET');
    assert.equal(error.request_path, '/proxy/5173/api/runs/run_fixture');
    assert.match(error.message, /VSCode.*5173.*读取未发起新任务/);
    return true;
  });
  assert.equal(count, 1);
});

test('network failure during POST never automatically resubmits a possibly started task', async () => {
  let count = 0;
  const { request } = createApiClient('https://host.example/proxy/5173/app.js', async () => {
    count += 1;
    throw new TypeError('Failed to fetch');
  });
  await assert.rejects(request('/api/runs', { method: 'POST', body: '{"task":"fixture"}' }), error => {
    assert.equal(error.may_have_started, true);
    assert.equal(error.request_method, 'POST');
    assert.match(error.message, /先检查历史运行.*重复提交/);
    return true;
  });
  assert.equal(count, 1);
});
