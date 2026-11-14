import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { artifactPreviewHtml, bindArtifactPreview, unloadArtifactPreview } from './media-preview.js';
import { previewMedia } from './run-media.mjs';

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), 'adagent-media-'));
  return {
    root,
    options: { projectRoot: root, weRoot: '/we', closedLoopRoot: join(root, 'closed') },
    async file(path, content = '') {
      await mkdir(join(path, '..'), { recursive: true });
      await writeFile(path, content);
      return path;
    },
  };
}

test('artifact preview defers videos and collapsed images', () => {
  const html = artifactPreviewHtml([
    { kind: 'video', title: '检索视频', src: '/api/video?path=a&x=1', primary: true },
    { kind: 'image', title: '首帧', src: '/api/file?path=b', primary: true },
    { kind: 'image', title: '后续帧', src: '/api/file?path=c' },
  ]);
  assert.match(html, /产物预览/);
  assert.match(html, /查看更多产物（1）/);
  assert.match(html, /preload="none"/);
  assert.doesNotMatch(html, /<video[^>]+src=/);
  assert.match(html, /data-src="\/api\/file\?path=c"/);
  assert.doesNotMatch(html, /data-src="\/api\/file\?path=c" src=/);
  assert.match(html, /loading="lazy" decoding="async"/);
  assert.match(html, /path=a&amp;x=1/);
});

function videoDom() {
  const listeners = object => {
    const handlers = new Map();
    return {
      addEventListener(name, handler) { handlers.set(name, handler); },
      dispatch(name) { handlers.get(name)?.(); },
    };
  };
  const attributes = new Map();
  const video = {
    ...listeners(), hidden: true, paused: false, readyState: 0, playCount: 0, loadCount: 0,
    getAttribute(name) { return attributes.get(name) ?? null; },
    set src(value) { attributes.set('src', value); },
    get src() { return attributes.get('src'); },
    removeAttribute(name) { attributes.delete(name); },
    load() { this.loadCount += 1; },
    play() { this.playCount += 1; return Promise.resolve(); },
    pause() { this.paused = true; },
  };
  const load = { ...listeners(), hidden: false };
  const retry = { ...listeners(), hidden: true };
  const status = { hidden: true };
  const message = { textContent: '' };
  const label = { textContent: '点击播放视频' };
  const card = {
    dataset: { videoSrc: '/api/video?path=a', state: 'idle' },
    querySelector(selector) {
      return {
        video, '[data-load-video]': load, '[data-load-label]': label,
        '.artifact-media-status': status, '[data-media-message]': message,
        '[data-retry-video]': retry,
      }[selector];
    },
  };
  const root = {
    ownerDocument: { querySelectorAll() { return [video]; } },
    querySelectorAll(selector) {
      if (selector === '.artifact-video') return [card];
      return [];
    },
  };
  return { root, card, video, load, retry, status, message };
}

test('video loads on demand and retry recovers from a media error', () => {
  const dom = videoDom();
  bindArtifactPreview(dom.root);
  assert.equal(dom.video.getAttribute('src'), null);
  dom.load.dispatch('click');
  assert.equal(dom.video.getAttribute('src'), '/api/video?path=a');
  assert.equal(dom.video.loadCount, 1);
  assert.equal(dom.video.playCount, 1);
  assert.equal(dom.status.hidden, false);
  dom.video.dispatch('canplay');
  assert.equal(dom.card.dataset.state, 'ready');
  assert.equal(dom.status.hidden, true);
  dom.video.dispatch('error');
  assert.equal(dom.card.dataset.state, 'error');
  assert.equal(dom.load.hidden, false);
  assert.equal(dom.retry.hidden, false);
  dom.retry.dispatch('click');
  assert.match(dom.video.getAttribute('src'), /_retry=1/);
  assert.equal(dom.video.playCount, 2);
  dom.video.dispatch('canplay');
  unloadArtifactPreview({ querySelectorAll: () => [dom.video] });
  assert.equal(dom.video.getAttribute('src'), null);
  assert.equal(dom.video.paused, true);
});

test('retrieval media chooses compressed clip and transcodes the source grid', async () => {
  const f = await fixture();
  const tag = 'rank01_score0.812345';
  const lossless = await f.file(join(f.root, `scene_${tag}_lossless.mp4`));
  await f.file(join(f.root, `scene_${tag}_compressed.mp4`));
  const sampleRoot = join(f.root, 'sample');
  await f.file(join(sampleRoot, 'preview', 'original_eightcam_2hz.mp4'));
  const media = previewMedia('video_retriever', { result: {
    video_path: lossless, scene_ref: { sample_root: sampleRoot },
  } }, f.options);
  assert.equal(media.length, 2);
  assert.match(media[0].path, /_compressed\.mp4$/);
  assert.match(media[0].src, /^\/api\/file/);
  assert.match(media[1].src, /^\/api\/video/);
  assert.equal(media[0].primary, true);
  assert.equal(media[1].primary, false);
});

test('planner exposes keyframe sheets with only the first expanded', async () => {
  const f = await fixture();
  const sheets = [];
  for (let index = 0; index < 3; index += 1) {
    sheets.push({ path: await f.file(join(f.root, `sheet_${index}.jpg`)), camera: 'CAM_F0', frame_indices: [index] });
  }
  const media = previewMedia('edit_planner', { result: {
    evidence: { visual_observation: { sheets } },
  } }, f.options);
  assert.equal(media.length, 3);
  assert.deepEqual(media.map(item => item.primary), [true, false, false]);
  assert.match(media[0].note, /编辑前画面/);
});

test('editing previews bind selected candidates and use the project video route', async () => {
  const f = await fixture();
  const runDir = join(f.root, 'edit-run');
  await f.file(join(runDir, 'output/attack/trajectory_editing/inspections/seed_000_01.png'));
  await f.file(join(runDir, 'output/attack/trajectory_editing/inspections/other_01.png'));
  const attack = previewMedia('attack_vehicle_editor', { result: {
    run_dir: runDir, selected_candidate_id: 'seed_000',
  } }, f.options);
  assert.equal(attack.length, 1);
  assert.match(attack[0].title, /seed_000/);

  await f.file(join(runDir, 'output/defense_agent/inspections/inspection_01.png'));
  await f.file(join(runDir, 'output/defense_agent/inspections/inspection_02.png'));
  await f.file(join(runDir, 'output/defense_agent/inspections/inspection_01.json'), JSON.stringify({
    candidate_id: 'constructor_006', path: 'inspections/inspection_01.png',
  }));
  await f.file(join(runDir, 'output/defense_agent/inspections/inspection_02.json'), JSON.stringify({
    candidate_id: 'edit_002', path: 'inspections/inspection_02.png', min_clearance_m: 8.6937,
  }));
  const rendered = await f.file(join(f.root, 'outputs/manual-explained.mp4'));
  const ego = previewMedia('ego_trajectory_editor', { result: {
    run_dir: runDir, selected_candidate_id: 'edit_002', rendered_videos: { manual_explained: rendered },
  } }, f.options);
  assert.equal(ego.length, 3);
  assert.match(ego[0].src, /^\/api\/video/);
  assert.equal(ego[1].record, undefined);
  assert.match(ego[1].title, /最终专家轨迹 · edit_002/);
  assert.equal(ego[1].primary, true);
  assert.equal(ego[2].primary, false);
});
