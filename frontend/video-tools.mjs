import { accessSync, constants, readdirSync, statSync } from 'node:fs';
import { join, resolve, delimiter } from 'node:path';

function executable(path) {
  try {
    accessSync(path, constants.X_OK);
    return statSync(path).isFile();
  } catch { return false; }
}

function children(path) {
  try { return readdirSync(path); } catch { return []; }
}

export function resolveFfmpeg(projectRoot, env = process.env) {
  const system = (env.PATH || '').split(delimiter).filter(Boolean).map(dir => join(dir, 'ffmpeg'));
  const configured = env.ADAGENT_FFMPEG_BIN || env.IMAGEIO_FFMPEG_EXE;
  if (configured) {
    const candidates = configured.includes('/') ? [resolve(configured)]
      : (env.PATH || '').split(delimiter).filter(Boolean).map(dir => join(dir, configured));
    const found = candidates.find(executable);
    if (found) return found;
    throw new Error('配置的 FFmpeg 不可执行，请检查 ADAGENT_FFMPEG_BIN / IMAGEIO_FFMPEG_EXE');
  }
  // Use the project's declared environments; do not depend on another user's home directory.
  for (const name of ['scene_editing', 'agent/.venv', 'simengine']) {
    const root = join(projectRoot, 'environments', name);
    const binary = join(root, 'bin', 'ffmpeg');
    if (executable(binary)) return binary;
    for (const version of children(join(root, 'lib')).filter(name => name.startsWith('python')).sort()) {
      const folder = join(root, 'lib', version, 'site-packages', 'imageio_ffmpeg', 'binaries');
      for (const name of children(folder).filter(name => name.startsWith('ffmpeg-')).sort()) {
        const binary = join(folder, name);
        if (executable(binary)) return binary;
      }
    }
  }
  const found = system.find(executable);
  if (found) return found;
  throw new Error('未找到 FFmpeg；请设置 ADAGENT_FFMPEG_BIN 为可执行文件路径');
}
