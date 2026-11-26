import { existsSync, readdirSync } from 'node:fs';
import { dirname, join, sep } from 'node:path';

const fileUrl = path => '/api/file?path=' + encodeURIComponent(path);
const transcodedUrl = (path, options) => {
  const route = path.startsWith(options.projectRoot + sep) ? '/api/video' : '/api/wevideo';
  return route + '?path=' + encodeURIComponent(path);
};

function image(items, path, title, note, primary = false) {
  if (typeof path !== 'string' || !existsSync(path)) return;
  items.push({ kind: 'image', title, path, src: fileUrl(path), note, primary });
}

function video(items, path, title, src, note, primary = false) {
  if (typeof path !== 'string' || !existsSync(path)) return;
  items.push({ kind: 'video', title, path, src, note, primary });
}

function eightcamSource(simulation, token, closedLoopRoot) {
  if (!simulation || !token || !closedLoopRoot) return null;
  const record = String(simulation).trim();
  const short = String(token).trim().split('-').pop();
  if (!record || !short) return null;
  const base = record.startsWith('/') ? record : join(closedLoopRoot, record, 'visualizations', 'cameras');
  const candidates = [
    join(base, `camera_view_${short}.mp4`),
    join(closedLoopRoot, record, `camera_view_${short}.mp4`),
  ];
  return candidates.find(path => existsSync(path)) || null;
}

function videoMedia(tool, result, options) {
  if (tool === 'video_retriever') return retrievalMedia(result, options);
  const items = [];
  if (tool === 'failure_analyzer') {
    const src = eightcamSource(result.simulation, result.token, options.closedLoopRoot);
    if (src) video(items, src, '失败场景关键证据', '/api/eightcam?path=' + encodeURIComponent(src), '原始仿真记录', true);
  }
  if (tool === 'policy_runner' && Array.isArray(result.videos)) {
    result.videos.slice(0, 2).forEach((path, index) => {
      video(items, path, index === 0 ? '原生仿真回放' : `原生仿真回放 ${index + 1}`,
        transcodedUrl(path, options), '实际仿真输出', index === 0);
    });
  }
  return items;
}

export function retrievalMedia(result, options) {
  const items = [];
  if (result.video_path) {
    let playable = result.video_path;
    let compressed = false;
    const tag = /(rank\d+_score[\d.]+)_lossless\.mp4$/i.exec(result.video_path);
    if (tag) {
      try {
        const hit = readdirSync(dirname(result.video_path)).find(name =>
          name.includes(tag[1]) && name.endsWith('_compressed.mp4'));
        if (hit) {
          playable = join(dirname(result.video_path), hit);
          compressed = true;
        }
      } catch { /* 转码代理仍可处理原文件 */ }
    }
    video(items, playable, `${result.mode === 'mock' ? '模拟检索预览' : '检索视频'}${result.scene_ref?.matched_camera ? ' · ' + result.scene_ref.matched_camera : ''}`,
      compressed ? fileUrl(playable) : transcodedUrl(playable, options), '原始检索结果，非编辑后画面', true);
  }
  const root = result.scene_ref?.sample_root;
  if (root) {
    const grid = join(root, 'preview', 'original_eightcam_2hz.mp4');
    video(items, grid, '原始场景八宫格', transcodedUrl(grid, options), '八相机原始记录，非编辑后画面');
  }
  return items;
}

function plannerMedia(result, items) {
  const before = items.length;
  if (typeof result.retrieved_bev_image === 'string') {
    image(items, result.retrieved_bev_image, '检索场景 BEV 俯视', '编辑前场景 BEV 拓扑', items.length === before);
  }
  const overlay = result.evidence?.self_check_overlay;
  if (typeof overlay === 'string') {
    image(items, overlay, '编辑规划自检叠加图', '攻击车起点与路径校验', items.length === before);
  }
  const sheets = result.evidence?.visual_observation?.sheets;
  if (!Array.isArray(sheets)) return;
  sheets.forEach((sheet, index) => {
    const frames = Array.isArray(sheet.frame_indices) ? sheet.frame_indices.join(' / ') : '';
    image(items, sheet.path, `原始场景关键帧 ${index + 1}/${sheets.length}`,
      [sheet.camera, frames && `帧 ${frames}`, '编辑前画面'].filter(Boolean).join(' · '), items.length === before && index === 0);
  });
}

function attackMedia(result, items) {
  const candidate = result.selected_candidate_id;
  if (!result.run_dir || !candidate) return;
  const directory = join(result.run_dir, 'output', 'attack', 'trajectory_editing', 'inspections');
  let paths = [];
  try {
    paths = readdirSync(directory)
      .filter(name => name.startsWith(`${candidate}_`) && name.endsWith('.png'))
      .sort()
      .map(name => join(directory, name));
  } catch { return; }
  paths.forEach((path, index) => {
    image(items, path, `场景编辑几何检查 · ${candidate}`,
      '几何参考，非相机渲染或物理仿真；本次编辑结果以冻结场景为准', index === paths.length - 1);
  });
}

export function previewMedia(tool, output, options) {
  if (!output?.result || typeof output.result !== 'object') return [];
  const result = output.result;
  const items = videoMedia(tool, result, options);
  if (tool === 'edit_planner') plannerMedia(result, items);
  if (tool === 'attack_vehicle_editor') attackMedia(result, items);
  return items.filter((item, index) => items.findIndex(other => other.src === item.src) === index);
}
