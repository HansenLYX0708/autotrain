/**
 * In-process runner for server-folder upload tasks.
 *
 * Copies a server-local directory into the task's `targetDir` in the
 * background, honouring cancellation, with at most one DB progress write per
 * second. The runner registry lives on `globalThis` so a dev-server HMR
 * reload does not lose track of a running copy.
 *
 * KNOWN LIMITATION: runners do not survive a server restart. A task left in
 * `uploading` with no live runner is marked "failed: interrupted by server
 * restart" by `markInterruptedServerTasks()` on the next list/get.
 */
import { promises as fs } from 'fs'
import path from 'path'
import { db } from './db'
import { isInside } from './safe-path'

type Runner = { cancelled: boolean }

const globalForRunners = globalThis as unknown as { __uploadRunners?: Map<string, Runner> }
const runners = (globalForRunners.__uploadRunners ??= new Map<string, Runner>())

const CONCURRENCY = 4
const PROGRESS_FLUSH_MS = 1000

export function cancelServerImport(taskId: string): boolean {
  const runner = runners.get(taskId)
  if (!runner) return false
  runner.cancelled = true
  return true
}

/** Fire-and-forget; the caller returns the task row immediately. */
export function startServerImport(taskId: string) {
  if (runners.has(taskId)) return
  const runner: Runner = { cancelled: false }
  runners.set(taskId, runner)
  run(taskId, runner).finally(() => runners.delete(taskId))
}

async function run(taskId: string, runner: Runner) {
  const task = await db.uploadTask.findUnique({ where: { id: taskId } })
  if (!task || !task.sourcePath) return
  const { sourcePath, targetDir } = task

  try {
    // Enumerate everything up front so totalFiles/totalBytes are real.
    const files: { src: string; rel: string; size: number }[] = []
    const walk = async (dir: string, rel: string) => {
      const entries = await fs.readdir(dir, { withFileTypes: true })
      for (const e of entries) {
        const full = path.join(dir, e.name)
        const r = rel ? `${rel}/${e.name}` : e.name
        if (e.isDirectory()) await walk(full, r)
        else if (e.isFile()) files.push({ src: full, rel: r, size: (await fs.stat(full).catch(() => null))?.size ?? 0 })
      }
    }
    await walk(sourcePath, '')
    await db.uploadTask.update({
      where: { id: taskId },
      data: { status: 'uploading', totalFiles: files.length, totalBytes: files.reduce((a, f) => a + f.size, 0) },
    })

    // labelme overwrite semantics match the browser path: the existing
    // dataset directory is emptied before the new files land.
    if (task.overwrite && task.format === 'labelme') {
      const existing = await fs.readdir(targetDir).catch(() => [] as string[])
      for (const entry of existing) {
        await fs.rm(path.join(targetDir, entry), { recursive: true, force: true }).catch(() => {})
      }
    }
    await fs.mkdir(targetDir, { recursive: true })

    let doneFiles = 0
    let doneBytes = 0
    let lastFlush = 0
    let firstError: Error | null = null
    let index = 0

    const flush = async (force = false) => {
      const now = Date.now()
      if (!force && now - lastFlush < PROGRESS_FLUSH_MS) return
      lastFlush = now
      await db.uploadTask.update({ where: { id: taskId }, data: { doneFiles, doneBytes } }).catch(() => {})
    }

    const worker = async () => {
      while (index < files.length) {
        if (runner.cancelled || firstError) return
        const file = files[index++]
        try {
          const dest = path.join(targetDir, file.rel)
          if (!isInside(targetDir, dest)) throw new Error(`Path escapes target: ${file.rel}`)
          await fs.mkdir(path.dirname(dest), { recursive: true })
          await fs.copyFile(file.src, dest)
          doneFiles += 1
          doneBytes += file.size
          await flush()
        } catch (err) {
          firstError = err instanceof Error ? err : new Error(String(err))
          return
        }
      }
    }
    await Promise.all(Array.from({ length: CONCURRENCY }, () => worker()))
    await flush(true)

    if (runner.cancelled) {
      // The task created this directory; a cancelled import leaves no partial dataset.
      await fs.rm(targetDir, { recursive: true, force: true }).catch(() => {})
      await db.uploadTask.update({
        where: { id: taskId },
        data: { status: 'cancelled', doneFiles, doneBytes },
      })
    } else if (firstError !== null) {
      await fs.rm(targetDir, { recursive: true, force: true }).catch(() => {})
      await db.uploadTask.update({
        where: { id: taskId },
        data: { status: 'failed', error: (firstError as Error).message, doneFiles, doneBytes },
      })
    } else {
      await db.uploadTask.update({
        where: { id: taskId },
        data: { status: 'completed', completedAt: new Date(), doneFiles, doneBytes },
      })
    }
  } catch (err) {
    await db.uploadTask.update({
      where: { id: taskId },
      data: { status: 'failed', error: err instanceof Error ? err.message : 'Unknown error' },
    }).catch(() => {})
  }
}

/**
 * A server task stuck in uploading/finalizing/pending with no live runner
 * means the process restarted mid-copy. Flag it failed (and drop its partial
 * targetDir) instead of leaving it green forever.
 */
export async function markInterruptedServerTasks(taskIds?: string[]) {
  const where = { source: 'server', status: { in: ['pending', 'uploading', 'finalizing'] }, ...(taskIds ? { id: { in: taskIds } } : {}) }
  const stale = await db.uploadTask.findMany({ where })
  for (const task of stale) {
    if (runners.has(task.id)) continue
    await fs.rm(task.targetDir, { recursive: true, force: true }).catch(() => {})
    await db.uploadTask.update({
      where: { id: task.id },
      data: { status: 'failed', error: 'Interrupted by server restart' },
    }).catch(() => {})
  }
}
