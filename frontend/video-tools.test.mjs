import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, chmod, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { resolveFfmpeg } from './video-tools.mjs';

async function binary(path, mode = 0o755) {
  await mkdir(join(path, '..'), { recursive: true });
  await writeFile(path, '#!/bin/sh\nexit 0\n');
  await chmod(path, mode);
  return path;
}

test('video transcoder resolves project-local imageio binary without a fixed Python version', async t => {
  const root = await mkdtemp(join(tmpdir(), 'video-tools-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const path = await binary(join(root, 'environments/scene_editing/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v8'));
  assert.equal(resolveFfmpeg(root, { PATH: '' }), path);
});

test('video transcoder honors explicit configuration and reports broken configuration', async t => {
  const root = await mkdtemp(join(tmpdir(), 'video-tools-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const path = await binary(join(root, 'custom tools/ffmpeg'));
  assert.equal(resolveFfmpeg(root, { ADAGENT_FFMPEG_BIN: path }), path);
  assert.equal(resolveFfmpeg(root, { IMAGEIO_FFMPEG_EXE: path }), path);
  assert.throws(() => resolveFfmpeg(root, { ADAGENT_FFMPEG_BIN: join(root, 'missing') }), /不可执行/);
});

test('video transcoder falls back to PATH and ignores files without execute permission', async t => {
  const root = await mkdtemp(join(tmpdir(), 'video-tools-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  await binary(join(root, 'environments/scene_editing/bin/ffmpeg'), 0o644);
  const path = await binary(join(root, 'system/ffmpeg'));
  assert.equal(resolveFfmpeg(root, { PATH: join(root, 'system') }), path);
  assert.throws(() => resolveFfmpeg(root, { PATH: '' }), /未找到 FFmpeg/);
});
