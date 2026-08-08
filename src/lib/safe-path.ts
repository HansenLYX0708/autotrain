/**
 * Filesystem path safety helpers.
 *
 * Several routes build paths out of user-controlled strings (job names,
 * dataset names, project names) and then read, write, or recursively delete
 * them. Two rules keep that safe:
 *
 *   1. A user-supplied string that becomes a *path segment* must be slugified,
 *      so it cannot be `..`, an absolute path, or contain separators.
 *   2. A fully-built path must be re-checked for containment inside its
 *      intended base directory before any destructive operation.
 *
 * Note on (2): the common `resolved.startsWith(base)` check is wrong —
 * `/data/user1-backup` starts with `/data/user1`. `isInside` compares against
 * `base + sep` and treats an exact match as inside.
 */

import * as path from 'path';

/**
 * Reduce arbitrary user input to a single safe path segment.
 *
 * Strips separators, drive letters, and leading dots, so the result can never
 * escape its parent directory. Returns `fallback` when nothing usable remains.
 */
export function toSafeSlug(input: string | null | undefined, fallback: string): string {
  const slug = (input ?? '')
    .replace(/\s+/g, '_')
    .replace(/[^a-zA-Z0-9_.-]/g, '')
    .replace(/^\.+/, '')
    .toLowerCase()
    .slice(0, 100);
  return slug.length > 0 ? slug : fallback;
}

/**
 * True when `target` resolves to `base` itself or something beneath it.
 * Both arguments are resolved first, so `..` segments and symlink-free relative
 * paths are handled.
 */
export function isInside(base: string, target: string): boolean {
  const resolvedBase = path.resolve(base);
  const resolvedTarget = path.resolve(target);
  if (resolvedTarget === resolvedBase) return true;
  return resolvedTarget.startsWith(resolvedBase + path.sep);
}

/**
 * Resolve `segments` under `base` and assert the result stays inside `base`.
 * Returns null when the path would escape, so callers can fail closed.
 */
export function resolveWithin(base: string, ...segments: string[]): string | null {
  const target = path.resolve(base, ...segments);
  return isInside(base, target) ? target : null;
}
