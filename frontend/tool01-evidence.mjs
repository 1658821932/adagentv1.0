/** Evidence of tool 01: the sheets and manifest the analysis really produced.

Backend contract (tools/tool01_failure_analyzer/visual_check.py):
  manifest.frames[]   = {index, sheet, sheet_sha256}
  manifest.selected[] = {index, time_s, sources{8 cameras}}
Only paths inside the project directory that exist as files are turned into links, and a
damaged manifest or an unreadable directory never breaks the run view.
 */
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { join, resolve, sep } from 'node:path';

const IMAGE = /\.(jpe?g|png|webp)$/i;
const fileUrl = path => '/api/file?path=' + encodeURIComponent(path);

function inside(projectRoot, path) {
  if (!projectRoot || typeof path !== 'string' || !path) return false;
  const root = resolve(projectRoot);
  const target = resolve(path);
  return target === root || target.startsWith(root + sep);
}

function readableFile(projectRoot, path) {
  if (!inside(projectRoot, path)) return false;
  try {
    return statSync(path).isFile();
  } catch {
    return false;
  }
}

export function readEvidenceManifest(evidenceDir, projectRoot) {
  if (!inside(projectRoot, evidenceDir)) return null;
  const file = join(evidenceDir, 'manifest.json');
  if (!readableFile(projectRoot, file)) return null;
  try {
    const manifest = JSON.parse(readFileSync(file, 'utf8'));
    return manifest && typeof manifest === 'object' ? manifest : null;
  } catch {
    return null;
  }
}

function frameFacts(manifest) {
  const facts = new Map();
  const selected = Array.isArray(manifest?.selected) ? manifest.selected : [];
  selected.forEach(entry => {
    if (Number.isInteger(entry?.index)) facts.set(entry.index, entry);
  });
  return facts;
}

function sheetFiles(evidenceDir, manifest, projectRoot) {
  const facts = frameFacts(manifest);
  const entries = [];
  const seen = new Set();
  const listed = Array.isArray(manifest?.frames) ? manifest.frames : [];
  listed.forEach(frame => {
    const sheet = frame?.sheet;
    if (!readableFile(projectRoot, sheet)) return;
    const key = resolve(sheet);
    if (seen.has(key)) return;
    seen.add(key);
    entries.push({ path: sheet, fact: facts.get(frame?.index) || null, index: frame?.index });
  });
  const directory = join(evidenceDir, 'sheets');
  try {
    if (inside(projectRoot, directory) && existsSync(directory) && statSync(directory).isDirectory()) {
      readdirSync(directory)
        .filter(name => IMAGE.test(name))
        .sort()
        .forEach(name => {
          const path = join(directory, name);
          const key = resolve(path);
          if (seen.has(key) || !readableFile(projectRoot, path)) return;
          seen.add(key);
          entries.push({ path, fact: null, index: null });
        });
    }
  } catch {
    /* an unreadable evidence directory only means fewer pictures, never a failed request */
  }
  return entries;
}

function caption(entry, index) {
  const facts = [];
  const frameIndex = Number.isInteger(entry.fact?.index) ? entry.fact.index : entry.index;
  if (Number.isInteger(frameIndex)) facts.push(`帧 ${frameIndex}`);
  if (typeof entry.fact?.time_s === 'number') facts.push(`${entry.fact.time_s.toFixed(1)}s`);
  return {
    title: `证据截图 ${index + 1}`,
    note: facts.length ? facts.join(' · ') : '本次分析的八相机总览',
  };
}

/**
 * Evidence for one failure_analyzer step.
 * @returns {{summary: object, items: Array<object>}}
 */
export function failureAnalyzerEvidence(output, { projectRoot } = {}) {
  const result = output?.result && typeof output.result === 'object' ? output.result : {};
  const error = output?.error && typeof output.error === 'object' ? output.error : {};
  const evidenceDir = typeof result.evidence_dir === 'string' && result.evidence_dir
    ? result.evidence_dir
    : typeof error.evidence_dir === 'string' ? error.evidence_dir : '';
  const summary = {
    evidence_dir: evidenceDir,
    error_type: typeof error.type === 'string' ? error.type : '',
    stage: typeof error.stage === 'string' ? error.stage : '',
    manifest: null,
    images: 0,
  };
  if (!inside(projectRoot, evidenceDir) || !existsSync(evidenceDir)) return { summary, items: [] };

  const manifest = readEvidenceManifest(evidenceDir, projectRoot);
  if (manifest) {
    const frames = Array.isArray(manifest.frames) ? manifest.frames.length : 0;
    const selected = Array.isArray(manifest.selected) ? manifest.selected.length : 0;
    summary.manifest = {
      status: typeof manifest.status === 'string' ? manifest.status : '',
      stage: typeof manifest.stage === 'string' ? manifest.stage : '',
      simulation: typeof manifest.simulation === 'string' ? manifest.simulation : '',
      token: typeof manifest.token === 'string' ? manifest.token : '',
      run_id: typeof manifest.run_id === 'string' ? manifest.run_id : '',
      timeout_s: typeof manifest.timeout_s === 'number' ? manifest.timeout_s : null,
      frames: frames || selected,
      updated_at: typeof manifest.updated_at === 'string' ? manifest.updated_at : '',
    };
    if (!summary.stage && summary.manifest.stage) summary.stage = summary.manifest.stage;
    if (!summary.error_type && summary.manifest.status === 'failed') summary.error_type = 'analysis_failed';
  }

  const items = sheetFiles(evidenceDir, manifest, projectRoot).map((entry, index) => ({
    kind: 'image',
    ...caption(entry, index),
    path: entry.path,
    src: fileUrl(entry.path),
    primary: index === 0,
  }));
  summary.images = items.length;
  return { summary, items };
}