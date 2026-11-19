'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { RefreshCw, UploadCloud, FolderOpen, XCircle, Trash2 } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { useUploadManager } from '@/contexts/upload-manager'

interface ServerTask {
  id: string
  source: 'browser' | 'server'
  datasetName: string
  format: string
  status: string
  totalFiles: number
  doneFiles: number
  totalBytes: number
  doneBytes: number
  error: string | null
  createdAt: string
  updatedAt: string
  completedAt: string | null
}

const ACTIVE = new Set(['pending', 'uploading', 'finalizing'])

function formatBytes(bytes: number): string {
  if (bytes >= 1 << 30) return `${(bytes / (1 << 30)).toFixed(2)} GB`
  if (bytes >= 1 << 20) return `${(bytes / (1 << 20)).toFixed(1)} MB`
  if (bytes >= 1 << 10) return `${(bytes / (1 << 10)).toFixed(1)} KB`
  return `${bytes} B`
}

function formatEta(sec: number | null): string {
  if (sec == null || !isFinite(sec)) return ''
  if (sec < 60) return `~${Math.ceil(sec)}s`
  if (sec < 3600) return `~${Math.ceil(sec / 60)}m`
  return `~${(sec / 3600).toFixed(1)}h`
}

function statusBadge(status: string) {
  const variant =
    status === 'completed' ? 'default'
    : status === 'failed' ? 'destructive'
    : status === 'cancelled' ? 'secondary'
    : 'outline'
  return <Badge variant={variant as any}>{status}</Badge>
}

export function UploadTasksPage() {
  const { tasks: liveTasks, resume, cancel, dismiss } = useUploadManager()
  const [serverTasks, setServerTasks] = useState<ServerTask[]>([])
  const [loading, setLoading] = useState(true)
  const resumeInputRef = useRef<HTMLInputElement>(null)
  const resumeTargetRef = useRef<string | null>(null)
  const serverTasksRef = useRef<ServerTask[]>([])
  serverTasksRef.current = serverTasks

  const anyActive =
    liveTasks.some(t => t.status === 'uploading' || t.status === 'finalizing') ||
    serverTasks.some(t => ACTIVE.has(t.status))

  const refresh = useCallback(async () => {
    try {
      const res = await fetch('/api/upload-tasks')
      const data = await res.json()
      if (data.success) setServerTasks(data.data)
    } catch {
      // keep last known list
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
    const interval = setInterval(refresh, anyActive ? 2000 : 15000)
    return () => clearInterval(interval)
  }, [refresh, anyActive])

  const onResumePick = (taskId: string) => {
    resumeTargetRef.current = taskId
    resumeInputRef.current?.click()
  }

  const onResumeFiles = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files ? Array.from(e.target.files) : []
    e.target.value = ''
    const taskId = resumeTargetRef.current
    resumeTargetRef.current = null
    if (taskId && files.length) await resume(taskId, files)
  }

  const onCancel = async (taskId: string) => {
    // Local flag stops the pipeline instantly; the API call marks the row.
    await cancel(taskId)
    await refresh()
  }

  const onRemove = async (taskId: string) => {
    dismiss(taskId)
    await fetch(`/api/upload-tasks/${encodeURIComponent(taskId)}`, { method: 'DELETE' }).catch(() => {})
    await refresh()
  }

  const liveById = new Map(liveTasks.map(t => [t.taskId, t]))

  // Merge: live manager state wins for progress/speed; server rows without a
  // live entry that still say "uploading" are paused browser uploads.
  const rows = serverTasks.map(st => {
    const live = liveById.get(st.id)
    const paused = st.source === 'browser' && ACTIVE.has(st.status) && !live
    return { st, live, paused }
  })
  // Live tasks not yet visible in the server list (e.g. just created).
  for (const live of liveTasks) {
    if (!serverTasks.some(st => st.id === live.taskId)) {
      rows.push({
        st: {
          id: live.taskId, source: 'browser', datasetName: live.datasetName,
          format: live.format, status: live.status, totalFiles: live.totalFiles,
          doneFiles: live.doneFiles, totalBytes: live.totalBytes, doneBytes: live.doneBytes,
          error: live.error, createdAt: new Date(live.startedAt).toISOString(),
          updatedAt: '', completedAt: live.finishedAt ? new Date(live.finishedAt).toISOString() : null,
        },
        live, paused: false,
      })
    }
  }
  rows.sort((a, b) => b.st.createdAt.localeCompare(a.st.createdAt))

  return (
    <div className="flex flex-col h-full">
      <input
        ref={resumeInputRef}
        type="file"
        className="hidden"
        // @ts-expect-error non-standard attribute for folder picking
        webkitdirectory=""
        onChange={onResumeFiles}
      />
      <div className="p-6 pb-4 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <UploadCloud className="w-6 h-6" /> Upload Tasks
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Dataset uploads keep running while you work on other pages.
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={refresh}>
          <RefreshCw className="w-4 h-4 mr-2" /> Refresh
        </Button>
      </div>

      <div className="flex-1 overflow-auto px-6 pb-6">
        {loading ? (
          <p className="text-sm text-muted-foreground">Loading…</p>
        ) : rows.length === 0 ? (
          <div className="text-sm text-muted-foreground border border-dashed rounded-lg p-10 text-center">
            No upload tasks yet. Start one from the Datasets page (“Upload Dataset”).
          </div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Dataset</TableHead>
                <TableHead>Source</TableHead>
                <TableHead>Format</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="w-[28%]">Progress</TableHead>
                <TableHead>Started</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map(({ st, live, paused }) => {
                const doneFiles = live?.doneFiles ?? st.doneFiles
                const doneBytes = live?.doneBytes ?? st.doneBytes
                const pct = st.totalBytes > 0 ? Math.min(100, Math.round((doneBytes / st.totalBytes) * 100)) : 0
                const isActive = live ? live.status === 'uploading' || live.status === 'finalizing' : ACTIVE.has(st.status) && !paused
                const displayStatus = paused ? 'paused' : (live?.status ?? st.status)
                const error = live?.error ?? st.error
                return (
                  <TableRow key={st.id}>
                    <TableCell className="font-medium">{st.datasetName}</TableCell>
                    <TableCell className="capitalize">{st.source}</TableCell>
                    <TableCell>{st.format}</TableCell>
                    <TableCell>
                      {paused ? <Badge variant="secondary">paused</Badge> : statusBadge(displayStatus)}
                      {error && <div className="text-xs text-destructive mt-1 max-w-[220px] truncate" title={error}>{error}</div>}
                    </TableCell>
                    <TableCell>
                      <Progress value={paused ? pct : (displayStatus === 'completed' ? 100 : pct)} className="h-2" />
                      <div className="text-xs text-muted-foreground mt-1">
                        {doneFiles}/{st.totalFiles} files · {formatBytes(doneBytes)}/{formatBytes(st.totalBytes)}
                        {live && live.status === 'uploading' && (
                          <> · {formatBytes(live.speedBps)}/s{live.etaSec ? ` · ${formatEta(live.etaSec)}` : ''}</>
                        )}
                      </div>
                      {paused && (
                        <div className="text-xs text-muted-foreground mt-1">Paused — reselect the folder to resume.</div>
                      )}
                      {displayStatus === 'completed' && (
                        <div className="text-xs text-muted-foreground mt-1">Use “Import Dataset” on the Datasets page to load it.</div>
                      )}
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground whitespace-nowrap">
                      {new Date(st.createdAt).toLocaleString()}
                    </TableCell>
                    <TableCell className="text-right whitespace-nowrap">
                      {isActive && (
                        <Button variant="outline" size="sm" onClick={() => onCancel(st.id)}>
                          <XCircle className="w-4 h-4 mr-1" /> Cancel
                        </Button>
                      )}
                      {st.source === 'browser' && (paused || (!isActive && (st.status === 'failed' || st.status === 'cancelled'))) && (
                        <Button variant="outline" size="sm" onClick={() => onResumePick(st.id)} title="Reselect the folder to resume">
                          <FolderOpen className="w-4 h-4 mr-1" /> Resume
                        </Button>
                      )}
                      {!isActive && (
                        <Button variant="ghost" size="sm" onClick={() => onRemove(st.id)}>
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        )}
      </div>
    </div>
  )
}
