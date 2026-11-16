import { createHash } from 'node:crypto';
import { stat } from 'node:fs/promises';
import { resolve } from 'node:path';

// Include file generation and encoding settings, not only its reusable name.
export async function videoCacheIdentity(source) {
  const info = await stat(source, { bigint: true });
  if (!info.isFile() || info.size <= 0n) throw new Error('视频文件为空或不可读');
  const generation = [resolve(source), String(info.dev), String(info.ino),
    String(info.size), String(info.mtimeNs), String(info.ctimeNs)];
  return createHash('sha256').update(JSON.stringify([
    'h264-yuv420p-width1920-crf23-fps2-faststart-v2', ...generation,
  ])).digest('hex').slice(0, 32) + '_playable.mp4';
}
