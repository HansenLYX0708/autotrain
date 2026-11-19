import fs from 'fs/promises'
import path from 'path'

const IMAGE_EXTENSIONS = new Set(['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff', '.webp'])
const MAX_ENTRIES = 2000

export type ServerBrowseEntry = { name: string; path: string; type: 'dir' | 'image' }
export type ServerBrowseResult = {
  path: string | null
  parent: string | null
  entries: ServerBrowseEntry[]
  truncated: boolean
}

export function isServerImageName(name: string): boolean {
  return IMAGE_EXTENSIONS.has(path.extname(name).toLowerCase())
}

async function windowsDriveRoots(): Promise<string[]> {
  const roots: string[] = []
  for (let code = 65; code <= 90; code++) {
    const root = `${String.fromCharCode(code)}:\\`
    try { await fs.access(root); roots.push(root) } catch { /* drive absent */ }
  }
  return roots
}

const byName = (a: ServerBrowseEntry, b: ServerBrowseEntry) =>
  a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' })

/** Read-only directory listing for the server-path picker. `dir` null/empty lists filesystem roots. */
export async function listServerDirectory(dir: string | null | undefined, opts: { includeImages: boolean }): Promise<ServerBrowseResult> {
  const entries: ServerBrowseEntry[] = []
  if (!dir) {
    const roots = process.platform === 'win32' ? await windowsDriveRoots() : ['/']
    return { path: null, parent: null, truncated: roots.length > MAX_ENTRIES,
      entries: roots.slice(0, MAX_ENTRIES).map(root => ({ name: root, path: root, type: 'dir' as const })) }
  }
  const resolved = path.resolve(dir)
  const stat = await fs.stat(resolved).catch(() => null)
  if (!stat) throw new Error(`Path does not exist: ${resolved}`)
  if (!stat.isDirectory()) throw new Error(`Not a directory: ${resolved}`)
  const dirents = await fs.readdir(resolved, { withFileTypes: true })
  let truncated = false
  for (const dirent of dirents) {
    if (entries.length >= MAX_ENTRIES) { truncated = true; break }
    try {
      const full = path.join(resolved, dirent.name)
      if (dirent.isDirectory()) entries.push({ name: dirent.name, path: full, type: 'dir' })
      else if (opts.includeImages && dirent.isFile() && isServerImageName(dirent.name)) entries.push({ name: dirent.name, path: full, type: 'image' })
    } catch { /* permission / race: skip */ }
  }
  entries.sort((a, b) => (a.type === b.type ? 0 : a.type === 'dir' ? -1 : 1) || byName(a, b))
  const parent = path.dirname(resolved)
  return { path: resolved, parent: parent === resolved ? null : parent, entries, truncated }
}
