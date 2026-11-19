/**
 * Shared helpers for background upload tasks.
 *
 * Two task sources share the UploadTask table:
 *  - 'browser': the client uploads chunks into `.uploads/<id>` under the user's
 *    storage root, then `complete` assembles them into `targetDir`.
 *  - 'server': the server copies a server-local folder into `targetDir` in a
 *    background runner (see the server-import route; runners are in-memory).
 */
import fs from 'fs/promises'
import path from 'path'
import { db } from './db'
import { isInside } from './safe-path'

export const UPLOAD_FORMATS = ['COCO', 'labelme'] as const
export type UploadFormat = (typeof UPLOAD_FORMATS)[number]
export type UploadSource = 'browser' | 'server'

/** Temp chunk root for a browser upload session. */
export const uploadTempDir = (userDatabasePath: string, username: string, uploadId: string) =>
  path.join(userDatabasePath, username, '.uploads', uploadId)

/** Server-side target directory; never trust a client-supplied one. */
export const uploadTargetDir = (userDatabasePath: string, username: string, format: string, datasetName: string) =>
  path.join(userDatabasePath, username, format, datasetName)

/**
 * Validate a client-supplied per-file relative path. Rejects empties,
 * absolute paths, drive letters, `..` segments, backslash escapes and NUL —
 * anything that could make a chunk/complete write escape the upload dirs.
 * Returns the normalised forward-slash path or null.
 */
export function safeRelativePath(input: unknown): string | null {
  if (typeof input !== 'string' || !input || input.includes('\0')) return null
  const normalized = input.replace(/\\/g, '/')
  if (normalized.startsWith('/') || /^[A-Za-z]:/.test(normalized)) return null
  const segments = normalized.split('/')
  if (segments.some(s => !s || s === '.' || s === '..')) return null
  return normalized
}

/** The temp sub-dir name a file's chunks live under (matches the existing layout). */
export const fileTempDirName = (relativePath: string) => relativePath.replace(/\//g, '___')

export async function getUploadTask(id: string) {
  return db.uploadTask.findUnique({ where: { id } })
}

/** BigInt columns cannot go through JSON.stringify — convert to numbers. */
export function serializeTask(task: {
  id: string; userId: string; source: string; datasetName: string; format: string
  sourcePath: string | null; targetDir: string; status: string
  totalFiles: number; doneFiles: number; totalBytes: bigint; doneBytes: bigint
  overwrite: boolean; error: string | null; createdAt: Date; updatedAt: Date; completedAt: Date | null
}) {
  return {
    ...task,
    totalBytes: Number(task.totalBytes),
    doneBytes: Number(task.doneBytes),
  }
}

/** Common validation for both upload modes: name/format rules, quota, exists/overwrite. */
export async function validateUploadRequest(
  user: { id: string; username: string; maxStorageQuota: bigint },
  system: { userDatabasePath: string | null },
  options: { datasetName: string; format: string; totalSize: number; confirmedOverwrite?: boolean },
): Promise<
  | { ok: true; targetDir: string; overwrite: boolean }
  | { ok: false; response: NextResponseCompat }
> {
  const { datasetName, format, totalSize, confirmedOverwrite } = options

  if (!datasetName || !format) {
    return { ok: false, response: { error: 'Dataset name and format are required', status: 400 } }
  }
  if (!UPLOAD_FORMATS.includes(format as UploadFormat)) {
    return { ok: false, response: { error: 'Invalid format. Must be COCO or labelme', status: 400 } }
  }
  if (!/^[a-zA-Z0-9_-]+$/.test(datasetName)) {
    return { ok: false, response: { error: 'Dataset name can only contain letters, numbers, underscores and hyphens', status: 400 } }
  }
  if (!system.userDatabasePath) {
    return { ok: false, response: { error: 'User database path not configured', status: 500 } }
  }

  const usedStorage = await folderSize(path.join(system.userDatabasePath, user.username))
  const maxQuota = Number(user.maxStorageQuota)
  const requiredSize = totalSize || 0
  if (usedStorage + requiredSize > maxQuota) {
    return {
      ok: false,
      response: {
        error: 'Insufficient storage',
        message: `You have used ${(usedStorage / 1024 / 1024 / 1024).toFixed(2)} GB of your ${(maxQuota / 1024 / 1024 / 1024).toFixed(2)} GB quota. This upload needs ${(requiredSize / 1024 / 1024).toFixed(2)} MB. Contact an administrator to expand the quota or delete unused data.`,
        usedStorage, maxStorageQuota: maxQuota, requiredSpace: requiredSize,
        status: 403,
      },
    }
  }

  const targetDir = uploadTargetDir(system.userDatabasePath, user.username, format, datasetName)
  const userRoot = path.resolve(system.userDatabasePath, user.username)
  if (!isInside(userRoot, targetDir)) {
    return { ok: false, response: { error: 'Invalid target directory', status: 400 } }
  }
  const exists = await fs.stat(targetDir).then(s => s.isDirectory(), () => false)
  if (exists && format === 'COCO') {
    return { ok: false, response: { error: `Dataset "${datasetName}" already exists`, status: 400 } }
  }
  if (exists && format === 'labelme' && !confirmedOverwrite) {
    return {
      ok: false,
      response: {
        success: true,
        requiresConfirmation: true,
        message: `Dataset "${datasetName}" already exists. Uploading will overwrite existing files and may add new samples.`,
        status: 200,
      },
    }
  }
  return { ok: true, targetDir, overwrite: exists && format === 'labelme' }
}

export type NextResponseCompat = Record<string, unknown> & { status: number }

/** Recursive folder size in bytes (async, tolerant of races). */
export async function folderSize(root: string): Promise<number> {
  const stat = await fs.stat(root).catch(() => null)
  if (!stat) return 0
  if (stat.isFile()) return stat.size
  if (!stat.isDirectory()) return 0
  let total = 0
  const entries = await fs.readdir(root, { withFileTypes: true }).catch(() => [] as import('fs').Dirent[])
  for (const entry of entries) {
    const full = path.join(root, entry.name)
    if (entry.isDirectory()) total += await folderSize(full)
    else if (entry.isFile()) total += (await fs.stat(full).catch(() => null))?.size ?? 0
  }
  return total
}
