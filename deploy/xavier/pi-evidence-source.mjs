/** Refresh a stale search hit only when its exact passage still exists on disk. */
import { openSync, closeSync, fstatSync, readFileSync, realpathSync } from 'node:fs';
import { resolve, relative, isAbsolute } from 'node:path';

export function refreshStaleEvidence(source, workspaceRoot) {
  if (!source?.stale || typeof source.text !== 'string' || source.text.length < 32 ||
      typeof source.path !== 'string') return null;
  let fd;
  try {
    const root = realpathSync(workspaceRoot);
    const path = resolve(source.path);
    const rel = relative(root, path);
    if (!rel || rel.startsWith('..') || isAbsolute(rel) || realpathSync(path) !== path) return null;
    fd = openSync(path, 'r');
    const stat = fstatSync(fd);
    if (!stat.isFile() || stat.size > 65536) return null;
    const text = readFileSync(fd, 'utf8');
    if (Buffer.byteLength(text) > 65536) return null;
    const at = text.indexOf(source.text);
    // Do not guess a new location when the passage changed or is ambiguous.
    if (at < 0 || text.indexOf(source.text, at + 1) >= 0) return null;
    const lines = text.split('\n');
    const start = Math.max(0, text.slice(0, at).split('\n').length - 1 - 4);
    const selected = [];
    let size = 0;
    for (const line of lines.slice(start, start + 32)) {
      if (size + line.length + 1 > 2400) break;
      selected.push(line); size += line.length + 1;
    }
    if (!selected.length) return null;
    return {path, start_line:start + 1, end_line:start + selected.length,
      text:selected.join('\n'), stale:false, excerpt_kind:'live-verified',
      verification:'Exact cached passage located in current file; surrounding lines reread.'};
  } catch { return null; }
  finally { if (fd !== undefined) closeSync(fd); }
}
