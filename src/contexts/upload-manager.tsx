'use client'

import React, { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react'

/**
 * Global browser-upload manager. Mounted above the page switcher so uploads
 * keep running when the user navigates between pages.
 *
 * Pipeline: init → batched chunk requests (small files grouped up to 64
 * files / 16 MB per request; large files sliced into 10 MB chunks, 4 requests
 * in flight, 3 retries) → complete. All state lives here; the Upload Tasks
 * page merges it with the server-side UploadTask list.
 */

export type LiveTaskStatus = 'uploading' | 'finalizing' | 'completed' | 'failed' | 'cancelled'

export interface LiveUploadTask {
  taskId: string
  datasetName: string
  format: string
  status: LiveTaskStatus
  totalFiles: number
  doneFiles: number
  totalBytes: number
  doneBytes: number
  speedBps: number
  etaSec: number | null
  error: string | null
  startedAt: number
  finishedAt: number | null
}

export interface StartUploadResult {
  started?: boolean
  requiresConfirmation?: boolean
  message?: string
}

interface UploadManagerValue {
  tasks: LiveUploadTask[]
  activeCount: number
  startBrowserUpload: (opts: {
    files: File[]
    datasetName: string
    format: string
    confirmedOverwrite?: boolean
  }) => Promise<StartUploadResult>
  resume: (taskId: string, files: File[]) => Promise<StartUploadResult>
  cancel: (taskId: string) => Promise<void>
  dismiss: (taskId: string) => void
}

const UploadManagerContext = createContext<UploadManagerValue | null>(null)

const CONCURRENCY = 4
const BATCH_MAX_FILES = 64
const BATCH_MAX_BYTES = 16 * 1024 * 1024
const MAX_RETRIES = 3
const PROGRESS_INTERVAL_MS = 250

interface ChunkItem {
  relativePath: string
  chunkIndex: number
  totalChunks: number
  blob: Blob
}

interface Internal {
  cancelled: boolean
  files: Map<string, File>
  items: ChunkItem[]
  doneBytes: number
  doneFiles: number
  lastBytesSample: { t: number; bytes: number }
}

export function UploadManagerProvider({ children }: { children: React.ReactNode }) {
  const [tasks, setTasks] = useState<LiveUploadTask[]>([])
  const internalsRef = useRef(new Map<string, Internal>())
  const tasksRef = useRef<LiveUploadTask[]>([])
  tasksRef.current = tasks

  const activeCount = tasks.filter(t => t.status === 'uploading' || t.status === 'finalizing').length

  // Warn before closing the tab while a browser upload is in flight.
  useEffect(() => {
    if (activeCount === 0) return
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault()
      e.returnValue = ''
    }
    window.addEventListener('beforeunload', handler)
    return () => window.removeEventListener('beforeunload', handler)
  }, [activeCount])

  const updateTask = useCallback((taskId: string, patch: Partial<LiveUploadTask>) => {
    setTasks(prev => prev.map(t => (t.taskId === taskId ? { ...t, ...patch } : t)))
  }, [])

  const reportProgress = useCallback((taskId: string, internal: Internal, force = false) => {
    const now = Date.now()
    if (!force && now - internal.lastBytesSample.t < PROGRESS_INTERVAL_MS) return
    const sample = internal.lastBytesSample
    internal.lastBytesSample = { t: now, bytes: internal.doneBytes }
    const dt = (now - sample.t) / 1000
    const speedBps = dt > 0 ? Math.max(0, (internal.doneBytes - sample.bytes) / dt) : 0
    const task = tasksRef.current.find(t => t.taskId === taskId)
    const remaining = (task?.totalBytes ?? 0) - internal.doneBytes
    updateTask(taskId, {
      doneBytes: internal.doneBytes,
      doneFiles: internal.doneFiles,
      speedBps,
      etaSec: speedBps > 0 ? remaining / speedBps : null,
    })
  }, [updateTask])

  const runPipeline = useCallback(async (taskId: string, session: any, files: Map<string, File>) => {
    const internal: Internal = {
      cancelled: false,
      files,
      items: [],
      doneBytes: 0,
      doneFiles: 0,
      lastBytesSample: { t: Date.now(), bytes: 0 },
    }
    internalsRef.current.set(taskId, internal)

    try {
      // Build the chunk work list, skipping chunks the server already has
      // (resume), and track already-uploaded bytes.
      const skippedBytes = { v: 0 }
      for (const f of session.files as any[]) {
        const file = files.get(f.relativePath)
        const uploaded = new Set<number>(f.uploadedChunks || [])
        for (let i = 0; i < f.totalChunks; i++) {
          if (uploaded.has(i)) {
            if (file) skippedBytes.v += Math.min(session.chunkSize, file.size - i * session.chunkSize)
            continue
          }
          if (!file) continue
          internal.items.push({
            relativePath: f.relativePath,
            chunkIndex: i,
            totalChunks: f.totalChunks,
            blob: file.slice(i * session.chunkSize, Math.min((i + 1) * session.chunkSize, file.size)),
          })
        }
      }
      internal.doneBytes = skippedBytes.v
      internal.doneFiles = (session.files as any[]).filter(
        f => (f.uploadedChunks || []).length >= f.totalChunks
      ).length
      internal.lastBytesSample = { t: Date.now(), bytes: internal.doneBytes }
      reportProgress(taskId, internal, true)

      // Pack single-chunk (small) files into batches; multi-chunk items go
      // one request per chunk. Each unit is {items: ChunkItem[]}.
      const units: ChunkItem[][] = []
      let batch: ChunkItem[] = []
      let batchBytes = 0
      for (const item of internal.items) {
        const single = item.totalChunks === 1
        if (!single || batch.length >= BATCH_MAX_FILES || batchBytes + item.blob.size > BATCH_MAX_BYTES) {
          if (batch.length) units.push(batch)
          batch = []
          batchBytes = 0
        }
        if (single) {
          batch.push(item)
          batchBytes += item.blob.size
        } else {
          units.push([item])
        }
      }
      if (batch.length) units.push(batch)

      let firstError: Error | null = null
      let cursor = 0
      const worker = async () => {
        while (cursor < units.length) {
          if (internal.cancelled || firstError) return
          const unit = units[cursor++]
          const form = new FormData()
          form.append('uploadId', session.uploadId)
          form.append('manifest', JSON.stringify(unit.map(i => ({
            relativePath: i.relativePath,
            chunkIndex: i.chunkIndex,
            totalChunks: i.totalChunks,
          }))))
          unit.forEach((i, idx) => form.append(`file${idx}`, i.blob, i.relativePath))

          let attempt = 0
          for (;;) {
            try {
              const res = await fetch('/api/datasets/upload/chunk', { method: 'POST', body: form })
              if (!res.ok) {
                const err = await res.json().catch(() => ({}))
                throw new Error(err.error || `Chunk upload failed: ${res.status}`)
              }
              break
            } catch (err) {
              attempt += 1
              if (attempt >= MAX_RETRIES || internal.cancelled) throw err
              await new Promise(r => setTimeout(r, 1000 * attempt))
            }
          }
          internal.doneBytes += unit.reduce((a, i) => a + i.blob.size, 0)
          internal.doneFiles += unit.filter(i => i.chunkIndex === i.totalChunks - 1).length
          reportProgress(taskId, internal)
        }
      }
      try {
        await Promise.all(Array.from({ length: CONCURRENCY }, () => worker()))
      } catch (err) {
        firstError = err instanceof Error ? err : new Error(String(err))
      }

      if (internal.cancelled) {
        updateTask(taskId, { status: 'cancelled', finishedAt: Date.now() })
        return
      }
      if (firstError) throw firstError

      // Finalize: stream-assemble server-side.
      updateTask(taskId, { status: 'finalizing' })
      const completeRes = await fetch('/api/datasets/upload/complete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          uploadId: session.uploadId,
          files: session.files,
        }),
      })
      const result = await completeRes.json().catch(() => ({}))
      if (!completeRes.ok || !result.success) {
        throw new Error(result.error || 'Failed to complete upload')
      }
      updateTask(taskId, { status: 'completed', finishedAt: Date.now(), doneBytes: internal.doneBytes, doneFiles: internal.doneFiles })
    } catch (err) {
      updateTask(taskId, {
        status: 'failed',
        error: err instanceof Error ? err.message : 'Unknown error',
        finishedAt: Date.now(),
      })
    } finally {
      internalsRef.current.delete(taskId)
    }
  }, [reportProgress, updateTask])

  const initSession = useCallback(async (body: Record<string, unknown>) => {
    const res = await fetch('/api/datasets/upload/init', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    const data = await res.json().catch(() => ({}))
    if (!res.ok) throw new Error(data.error || 'Failed to initialize upload')
    return data
  }, [])

  const startBrowserUpload = useCallback(async (opts: {
    files: File[]
    datasetName: string
    format: string
    confirmedOverwrite?: boolean
  }): Promise<StartUploadResult> => {
    const fileMap = new Map<string, File>()
    for (const file of opts.files) {
      fileMap.set((file as any).webkitRelativePath || file.name, file)
    }
    const filesMeta = opts.files.map(f => ({
      name: f.name,
      relativePath: (f as any).webkitRelativePath || f.name,
      size: f.size,
    }))
    const totalSize = opts.files.reduce((a, f) => a + f.size, 0)

    const session = await initSession({
      datasetName: opts.datasetName,
      format: opts.format,
      totalFiles: filesMeta.length,
      totalSize,
      files: filesMeta,
      confirmedOverwrite: opts.confirmedOverwrite,
    })
    if (session.requiresConfirmation) {
      return { requiresConfirmation: true, message: session.message }
    }

    setTasks(prev => [...prev, {
      taskId: session.uploadId,
      datasetName: opts.datasetName,
      format: opts.format,
      status: 'uploading',
      totalFiles: filesMeta.length,
      doneFiles: 0,
      totalBytes: totalSize,
      doneBytes: 0,
      speedBps: 0,
      etaSec: null,
      error: null,
      startedAt: Date.now(),
      finishedAt: null,
    }])
    void runPipeline(session.uploadId, session, fileMap)
    return { started: true }
  }, [initSession, runPipeline])

  const resume = useCallback(async (taskId: string, files: File[]): Promise<StartUploadResult> => {
    const fileMap = new Map<string, File>()
    for (const file of files) {
      fileMap.set((file as any).webkitRelativePath || file.name, file)
    }
    const filesMeta = files.map(f => ({
      name: f.name,
      relativePath: (f as any).webkitRelativePath || f.name,
      size: f.size,
    }))
    const session = await initSession({ resumeUploadId: taskId, files: filesMeta })
    if (session.requiresConfirmation) return { requiresConfirmation: true, message: session.message }

    const totalBytes = files.reduce((a, f) => a + f.size, 0)
    setTasks(prev => {
      const existing = prev.find(t => t.taskId === taskId)
      const entry: LiveUploadTask = existing
        ? { ...existing, status: 'uploading', error: null, finishedAt: null }
        : {
            taskId,
            datasetName: session.datasetName || 'Resumed upload',
            format: session.format || '',
            status: 'uploading',
            totalFiles: filesMeta.length,
            doneFiles: 0,
            totalBytes,
            doneBytes: 0,
            speedBps: 0,
            etaSec: null,
            error: null,
            startedAt: Date.now(),
            finishedAt: null,
          }
      return existing ? prev.map(t => (t.taskId === taskId ? entry : t)) : [...prev, entry]
    })
    void runPipeline(taskId, session, fileMap)
    return { started: true }
  }, [initSession, runPipeline])

  const cancel = useCallback(async (taskId: string) => {
    const internal = internalsRef.current.get(taskId)
    if (internal) internal.cancelled = true
    await fetch(`/api/upload-tasks/${encodeURIComponent(taskId)}/cancel`, { method: 'POST' }).catch(() => {})
    updateTask(taskId, { status: 'cancelled', finishedAt: Date.now() })
  }, [updateTask])

  const dismiss = useCallback((taskId: string) => {
    setTasks(prev => prev.filter(t => t.taskId === taskId))
  }, [])

  return (
    <UploadManagerContext.Provider value={{ tasks, activeCount, startBrowserUpload, resume, cancel, dismiss }}>
      {children}
    </UploadManagerContext.Provider>
  )
}

export function useUploadManager() {
  const ctx = useContext(UploadManagerContext)
  if (!ctx) throw new Error('useUploadManager must be used inside UploadManagerProvider')
  return ctx
}
