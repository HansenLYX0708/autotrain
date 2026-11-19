/**
 * Server-only storage for localization annotation projects.
 *
 * Projects are JSON files at `annotations/<toSafeSlug(userId)>/projects/<uuid>.json`
 * (the schema lives in `localization-annotations.ts` and is shared with the
 * Python data-prep tool — do NOT add fields to it). Lifecycle metadata that the
 * project format cannot carry lives in a sidecar `<uuid>.meta.json` next to it.
 *
 * Lifecycle rule: an annotation set is `active` while it is being worked on and
 * auto-archives 7 days after its last save (ARCHIVE_AFTER_DAYS, counted from
 * `activeSince`). Every save re-activates it; archiving never deletes anything.
 * Sets without a sidecar (everything saved before this feature) derive
 * activeSince/createdAt from the project file's mtime/birthtime, so legacy sets
 * older than 7 days auto-archive on the first listing — that is intended.
 */
import fs from 'fs/promises'
import path from 'path'
import { randomUUID } from 'crypto'
import { projectSchema, type AnnotationProject } from './localization-annotations'
import { isInside, toSafeSlug } from './safe-path'

export const ARCHIVE_AFTER_DAYS = 7

export type AnnotationSetStatus = 'active' | 'archived'
export type ArchivedReason = 'auto' | 'manual'

export interface AnnotationSetMeta {
  status: AnnotationSetStatus
  /** ISO timestamp of the last activation (save counts as an activation). */
  activeSince: string
  archivedAt?: string
  archivedReason?: ArchivedReason
  createdAt: string
}

export interface AnnotationSetSummary extends AnnotationSetMeta {
  id: string
  name: string
  images: number
  reviewedImages: number
  instances: number
  labels: number
  /** activeSince + ARCHIVE_AFTER_DAYS, present only while status is active. */
  archiveAt?: string
  /** Project file mtime — "last saved". */
  modified: string
}

export const projectsDir = (userId: string) =>
  path.join(process.cwd(), 'annotations', toSafeSlug(userId, 'user'), 'projects')

export const validId = (id: string) =>
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id)

export const projectPath = (dir: string, id: string) => path.join(dir, `${id}.json`)
const metaPath = (dir: string, id: string) => path.join(dir, `${id}.meta.json`)

/** Same tmp-then-rename pattern as the project POST: a crash mid-write cannot leave half a file. */
async function writeJsonAtomic(file: string, value: unknown) {
  const temporary = `${file}.${randomUUID()}.tmp`
  try {
    await fs.writeFile(temporary, JSON.stringify(value, null, 2), 'utf8')
    await fs.rename(temporary, file)
  } finally {
    await fs.unlink(temporary).catch(() => {})
  }
}

async function readMetaFile(dir: string, id: string): Promise<Partial<AnnotationSetMeta> | null> {
  try {
    const raw = JSON.parse(await fs.readFile(metaPath(dir, id), 'utf8'))
    return raw && typeof raw === 'object' ? raw : null
  } catch {
    return null
  }
}

/**
 * Resolve the effective meta for a set. Missing/corrupt sidecars fall back to
 * the project file's timestamps (status active), then the auto-archive rule is
 * applied lazily and persisted so the state is stable afterwards.
 */
async function effectiveMeta(
  dir: string,
  id: string,
  times: { mtime: Date; birthtime: Date },
): Promise<AnnotationSetMeta> {
  const stored = await readMetaFile(dir, id)
  let meta: AnnotationSetMeta = {
    status: stored?.status === 'archived' ? 'archived' : 'active',
    activeSince: typeof stored?.activeSince === 'string' ? stored.activeSince : times.mtime.toISOString(),
    createdAt: typeof stored?.createdAt === 'string' ? stored.createdAt : (times.birthtime || times.mtime).toISOString(),
    ...(stored?.status === 'archived' && typeof stored.archivedAt === 'string'
      ? { archivedAt: stored.archivedAt, archivedReason: stored.archivedReason === 'manual' ? 'manual' : 'auto' }
      : {}),
  }
  if (meta.status === 'active' && Date.now() - Date.parse(meta.activeSince) > ARCHIVE_AFTER_DAYS * 86400e3) {
    meta = { ...meta, status: 'archived', archivedAt: new Date().toISOString(), archivedReason: 'auto' }
    await writeJsonAtomic(metaPath(dir, id), meta)
  }
  return meta
}

/** Read a project JSON, validating it against the shared schema. */
export async function readAnnotationProject(dir: string, id: string): Promise<AnnotationProject> {
  return projectSchema.parse(JSON.parse(await fs.readFile(projectPath(dir, id), 'utf8')))
}

/** Read the project plus its effective lifecycle meta (applies lazy auto-archive). */
export async function readAnnotationSet(
  dir: string,
  id: string,
): Promise<{ project: AnnotationProject; meta: AnnotationSetMeta; modified: string; raw: string }> {
  const file = projectPath(dir, id)
  const [raw, stats] = await Promise.all([fs.readFile(file, 'utf8'), fs.stat(file)])
  return {
    project: projectSchema.parse(JSON.parse(raw)),
    meta: await effectiveMeta(dir, id, stats),
    modified: stats.mtime.toISOString(),
    raw,
  }
}

function summarize(id: string, project: AnnotationProject, meta: AnnotationSetMeta, modified: string): AnnotationSetSummary {
  return {
    id,
    name: project.name,
    images: project.images.length,
    reviewedImages: project.images.filter(image => image.reviewed).length,
    instances: project.images.reduce((total, image) => total + image.objects.length, 0),
    labels: project.labels.length,
    status: meta.status,
    activeSince: meta.activeSince,
    archiveAt: meta.status === 'active'
      ? new Date(Date.parse(meta.activeSince) + ARCHIVE_AFTER_DAYS * 86400e3).toISOString()
      : undefined,
    archivedAt: meta.archivedAt,
    archivedReason: meta.archivedReason,
    createdAt: meta.createdAt,
    modified,
  }
}

export async function listAnnotationSets(
  userId: string,
  options: { status?: AnnotationSetStatus | 'all'; q?: string } = {},
): Promise<AnnotationSetSummary[]> {
  const dir = projectsDir(userId)
  const files = await fs.readdir(dir).catch((error: NodeJS.ErrnoException) => {
    if (error.code === 'ENOENT') return []
    throw error
  })
  const ids = files
    .filter(file => file.endsWith('.json') && !file.endsWith('.meta.json'))
    .map(file => file.slice(0, -5))
    .filter(validId)
  const sets = await Promise.all(ids.map(async id => {
    const { project, meta, modified } = await readAnnotationSet(dir, id)
    return summarize(id, project, meta, modified)
  }))
  const status = options.status ?? 'active'
  const q = options.q?.trim().toLowerCase() || ''
  return sets
    .filter(set => (status === 'all' || set.status === status) && (!q || set.name.toLowerCase().includes(q)))
    .sort((a, b) => b.modified.localeCompare(a.modified))
}

/**
 * Save a project JSON atomically and mark the set active — a save restarts the
 * auto-archive countdown and re-activates an archived set.
 */
export async function saveAnnotationSet(dir: string, project: AnnotationProject) {
  const id = project.id
  const file = projectPath(dir, id)
  const previous = await fs.stat(file).then(stats => stats, () => null)
  await writeJsonAtomic(file, project)
  const stored = await readMetaFile(dir, id)
  const meta: AnnotationSetMeta = {
    status: 'active',
    activeSince: new Date().toISOString(),
    createdAt: typeof stored?.createdAt === 'string'
      ? stored.createdAt
      : (previous?.birthtime || new Date()).toISOString(),
  }
  await writeJsonAtomic(metaPath(dir, id), meta)
}

/** Manual archive / activate. Activating restarts the auto-archive countdown. */
export async function setAnnotationSetStatus(dir: string, id: string, status: AnnotationSetStatus) {
  const stats = await fs.stat(projectPath(dir, id))
  const meta = await effectiveMeta(dir, id, stats)
  const next: AnnotationSetMeta = status === 'archived'
    ? { ...meta, status, archivedAt: new Date().toISOString(), archivedReason: 'manual' }
    : { ...meta, status, archivedAt: undefined, archivedReason: undefined, activeSince: new Date().toISOString() }
  await writeJsonAtomic(metaPath(dir, id), next)
  return next
}

/** Rename rewrites the project JSON atomically (the schema owns the name). */
export async function renameAnnotationSet(dir: string, id: string, name: string) {
  const project = projectSchema.parse({
    ...JSON.parse(await fs.readFile(projectPath(dir, id), 'utf8')),
    name,
  })
  await writeJsonAtomic(projectPath(dir, id), project)
}

export async function deleteAnnotationSet(dir: string, id: string) {
  const file = projectPath(dir, id)
  if (!isInside(dir, file)) throw new Error('Invalid annotation project path')
  await fs.unlink(file)
  await fs.unlink(metaPath(dir, id)).catch(() => {})
}
