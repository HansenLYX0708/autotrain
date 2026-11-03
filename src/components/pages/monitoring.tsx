'use client'

import { useEffect, useState, useRef, useCallback } from 'react'
import { isDetection } from '@/lib/frameworks'
import { decimateMinMax } from '@/lib/chart-decimate'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { ScrollArea, ScrollBar } from '@/components/ui/scroll-area'
import { Button } from '@/components/ui/button'
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
} from 'recharts'
import {
  Activity,
  Timer,
  TrendingDown,
  Zap,
  HardDrive,
  Clock,
  Search,
  Pause,
  Play,
  AlertCircle,
  AlertTriangle,
  Download,
  Loader2,
} from 'lucide-react'

// Types
interface TrainingJob {
  id: string
  name: string
  status: string
  currentEpoch: number
  totalEpochs: number
  currentLoss: number | null
  currentLr: number | null
  bestIter: number | null
  bestMetric: number | null
  command: string | null
  trainingParams?: { doEval?: boolean }
  startedAt: string | null
  completedAt: string | null
  project: {
    id: string
    name: string
    framework: string
  }
  dataset: {
    id: string
    name: string
  }
  model: {
    id: string
    name: string
    architecture: string
  }
  config: {
    id: string
    name: string
  } | null
  _count: {
    logs: number
  }
}

interface TrainingLog {
  id: string
  epoch: number
  iteration: number
  totalIter: number
  loss: number | null
  lossCls: number | null
  lossIou: number | null
  lossDfl: number | null
  lossL1: number | null
  learningRate: number | null
  eta: string | null
  batchCost: number | null
  dataCost: number | null
  readerCost: number | null
  ips: number | null
  memReserved: number | null
  memAllocated: number | null
  // Segmentation evaluation metrics (PaddleSeg, TorchSeg)
  mIoU: number | null
  acc: number | null
  kappa: number | null
  dice: number | null
  // Detection evaluation metrics, parsed from the COCO summary block
  // (PaddleDetection, TorchDet). mAP is AP@[0.50:0.95], mAP50 is AP@0.50.
  mAP: number | null
  mAP50: number | null
  // Anomaly-detection metrics (TorchAnomaly). Pixel-level values are null when
  // the dataset has no masks for its defect images.
  imageAuroc: number | null
  imageF1: number | null
  pixelAuroc: number | null
  pixelF1: number | null
  threshold: number | null
  // JSON string: {"iou":[..],"precision":[..],"recall":[..]} — populated on
  // segmentation EVAL rows only. Kept as JSON to accommodate variable
  // num_classes without schema changes.
  classMetrics: string | null
  rawLog: string | null
  timestamp: string
}

interface LogsResponse {
  job: {
    id: string
    name: string
    status: string
  }
  data: TrainingLog[]
  pagination: {
    page: number
    limit: number
    total: number
    totalPages: number
  }
  stats: {
    totalLogs: number
    minLoss: number | null
    maxLoss: number | null
    avgLoss: number | null
    avgLearningRate: number | null
    avgBatchCost: number | null
    avgIps: number | null
  }
  epochs: number[]
}

// AMP overflow summary returned alongside console ranges (see
// `src/lib/job-console.ts`). Mirrors the server-side `AmpSummary`.
interface AmpSummary {
  count: number
  initialScale: number | null
  currentScale: number | null
  minScale: number | null
  lastLine: string | null
}

interface ConsoleRangeResponse {
  available: boolean
  path?: string
  size: number
  start: number
  end: number
  text: string
  hasMore: boolean
  amp: AmpSummary | null
}

interface ConsoleSearchResults {
  matches: { line: number; text: string }[]
  totalMatches: number
  truncated: boolean
}

// Keep at most 4 MB of console text in the browser; older lines can be
// pulled back with "Load earlier".
const CONSOLE_TEXT_LIMIT = 4 * 1024 * 1024

const byteLength = (s: string) => new TextEncoder().encode(s).length

const formatBytes = (n: number) => {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(2)} MB`
}

// Line highlighting for the console viewer: AMP overflow events are worth
// noticing but not fatal; a traceback means the run died.
const consoleLineClass = (line: string) =>
  /found inf or nan/i.test(line)
    ? 'bg-amber-500/15 text-amber-700 dark:text-amber-300'
    : /Traceback|Error|Exception/.test(line)
      ? 'bg-red-500/15 text-red-700 dark:text-red-300'
      : ''

// Chart configurations
const lossChartConfig = {
  loss: {
    label: "Total Loss",
    color: "var(--chart-1)",
  },
  lossCls: {
    label: "Cls Loss",
    color: "var(--chart-2)",
  },
  lossIou: {
    label: "IoU Loss",
    color: "var(--chart-3)",
  },
  lossDfl: {
    label: "DFL Loss",
    color: "var(--chart-4)",
  },
  lossL1: {
    label: "L1 Loss",
    color: "var(--chart-5)",
  },
} satisfies ChartConfig

const lrChartConfig = {
  learningRate: {
    label: "Learning Rate",
    color: "var(--chart-1)",
  },
} satisfies ChartConfig

// Segmentation evaluation metrics chart config (PaddleSeg, TorchSeg)
const segMetricChartConfig = {
  mIoU: {
    label: "mIoU",
    color: "var(--chart-1)",
  },
  acc: {
    label: "Acc",
    color: "var(--chart-2)",
  },
  kappa: {
    label: "Kappa",
    color: "var(--chart-3)",
  },
  dice: {
    label: "Dice",
    color: "var(--chart-4)",
  },
} satisfies ChartConfig

// Detection evaluation metrics chart config (PaddleDetection, TorchDet).
// Detection jobs previously had no evaluation chart at all: the COCO summary
// block was printed but never parsed, so only the loss curve was visible.
const detMetricChartConfig = {
  mAP: {
    label: "mAP@0.5:0.95",
    color: "var(--chart-1)",
  },
  mAP50: {
    label: "mAP@0.5",
    color: "var(--chart-2)",
  },
} satisfies ChartConfig

// Anomaly-detection metrics chart config (TorchAnomaly).
// The threshold is deliberately not plotted here: it is an anomaly *score*
// cut-off on an unbounded scale, and putting it on the same 0-1 axis as AUROC
// would either squash the curves or be clipped away.
const anomalyMetricChartConfig = {
  imageAuroc: {
    label: "Image AUROC",
    color: "var(--chart-1)",
  },
  imageF1: {
    label: "Image F1",
    color: "var(--chart-2)",
  },
  pixelAuroc: {
    label: "Pixel AUROC",
    color: "var(--chart-3)",
  },
  pixelF1: {
    label: "Pixel F1",
    color: "var(--chart-4)",
  },
} satisfies ChartConfig

// Helper functions
const getStatusBadge = (status: string) => {
  const variants: Record<string, { variant: "default" | "secondary" | "destructive" | "outline"; color: string }> = {
    running: { variant: "default", color: "bg-emerald-500 hover:bg-emerald-600" },
    completed: { variant: "secondary", color: "" },
    failed: { variant: "destructive", color: "" },
    pending: { variant: "outline", color: "" },
    stopped: { variant: "outline", color: "" },
  }
  return variants[status] || { variant: "outline", color: "" }
}

const formatTime = (date: string | null) => {
  if (!date) return 'N/A'
  return new Date(date).toLocaleString()
}

const formatDuration = (startDate: string | null, endDate: string | null) => {
  if (!startDate) return 'N/A'
  const end = endDate ? new Date(endDate) : new Date()
  const start = new Date(startDate)
  const diff = end.getTime() - start.getTime()
  
  const hours = Math.floor(diff / (1000 * 60 * 60))
  const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60))
  const seconds = Math.floor((diff % (1000 * 60)) / 1000)
  
  return `${hours}h ${minutes}m ${seconds}s`
}

const formatETA = (eta: string | null) => {
  if (!eta) return 'N/A'
  // ETA format from PaddleDetection: "1 day, 20:03:57" or "20:03:57"
  return eta
}

export function MonitoringPage() {
  // State
  const [jobs, setJobs] = useState<TrainingJob[]>([])
  const [selectedJobId, setSelectedJobId] = useState<string>('')
  const [selectedJob, setSelectedJob] = useState<TrainingJob | null>(null)
  const [logs, setLogs] = useState<TrainingLog[]>([])
  const [logsError, setLogsError] = useState<string | null>(null)
  const [stats, setStats] = useState<LogsResponse['stats'] | null>(null)
  const [loading, setLoading] = useState(true)
  const [logsLoading, setLogsLoading] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [autoScroll, setAutoScroll] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  // Console Output card. `consoleStart/End` are byte offsets into
  // autotrain-console.log bracketing the loaded `consoleText` window, so the
  // live tail and "Load earlier" can resume without re-downloading anything.
  const [consoleAvailable, setConsoleAvailable] = useState<boolean | null>(null)
  const [consoleText, setConsoleText] = useState('')
  const [consoleStart, setConsoleStart] = useState(0)
  const [consoleSize, setConsoleSize] = useState(0)
  const [consoleHasEarlier, setConsoleHasEarlier] = useState(false)
  const [consoleAmp, setConsoleAmp] = useState<AmpSummary | null>(null)
  const [consoleSearching, setConsoleSearching] = useState(false)
  const [searchResults, setSearchResults] = useState<ConsoleSearchResults | null>(null)
  // Fallback for jobs whose runs predate console capture: the parsed
  // TrainingLog rows, paged backwards through `logs?sort=desc`.
  const [fbLogs, setFbLogs] = useState<TrainingLog[]>([])
  const [fbPage, setFbPage] = useState(0)
  const [fbTotalPages, setFbTotalPages] = useState(1)
  const logContainerRef = useRef<HTMLDivElement>(null)
  const pollInFlightRef = useRef(false)
  const logsRequestRef = useRef<AbortController | null>(null)
  const logCursorRef = useRef<{ timestamp: string; id: string } | null>(null)
  const consoleStartRef = useRef(0)
  const consoleEndRef = useRef(0)
  const consoleLoadedRef = useRef(false)
  const prevStatusRef = useRef<string | undefined>(undefined)

  // Fetch training jobs
  const fetchJobs = useCallback(async () => {
    try {
      const response = await fetch('/api/training-jobs?limit=50')
      if (response.ok) {
        const data = await response.json()
        setJobs(data.data)
        // Auto-select running job or first job
        if (!selectedJobId && data.data.length > 0) {
          const runningJob = data.data.find((j: TrainingJob) => j.status === 'running')
          setSelectedJobId(runningJob?.id || data.data[0].id)
        }
      }
    } catch (error) {
      console.error('Failed to fetch jobs:', error)
    } finally {
      setLoading(false)
    }
  }, [selectedJobId])

  // Load the COMPLETE log history of a job (compact rows, no rawLog) by
  // paging forward through `logs?sort=asc`. `logs` is the single source for
  // charts, eval rows and the non-finite scan — no separate eval fetch.
  const loadFullLogs = useCallback(async (jobId: string, signal: AbortSignal) => {
    const rows: TrainingLog[] = []
    let page = 1
    let totalPages = 1
    let lastStats: LogsResponse['stats'] | null = null
    do {
      const response = await fetch(
        `/api/training-jobs/${jobId}/logs?sort=asc&limit=1000&fields=compact&page=${page}`,
        { signal },
      )
      if (!response.ok) throw new Error('Failed to load training logs')
      const data: LogsResponse = await response.json()
      rows.push(...data.data)
      totalPages = data.pagination.totalPages
      lastStats = data.stats
      page += 1
    } while (page <= totalPages)
    setLogs(rows)
    const last = rows[rows.length - 1]
    logCursorRef.current = last ? { timestamp: last.timestamp, id: last.id } : null
    setStats(lastStats)
  }, [])

  // Append rows newer than the cursor. `afterTimestamp`/`afterId` resume
  // strictly after the last known row, so incremental polls never re-read or
  // duplicate history; a full page means more rows are waiting.
  const loadIncrementalLogs = useCallback(async (jobId: string) => {
    const cursor = logCursorRef.current
    if (!cursor) {
      // No rows seen yet — e.g. the job was selected while still pending.
      // Page forward from the start (same shape as the initial load) so the
      // first rows arriving mid-poll are picked up and the cursor is set.
      const rows: TrainingLog[] = []
      let page = 1
      let totalPages = 1
      do {
        const response = await fetch(
          `/api/training-jobs/${jobId}/logs?sort=asc&limit=1000&fields=compact&page=${page}`,
        )
        if (!response.ok) return
        const data: LogsResponse = await response.json()
        rows.push(...data.data)
        totalPages = data.pagination.totalPages
        setStats(data.stats)
        page += 1
      } while (page <= totalPages)
      if (rows.length > 0) {
        setLogs((prev) => [...prev, ...rows])
        const last = rows[rows.length - 1]
        logCursorRef.current = { timestamp: last.timestamp, id: last.id }
      }
      return
    }
    let afterTs = cursor.timestamp
    let afterId = cursor.id
    const added: TrainingLog[] = []
    while (true) {
      const response = await fetch(
        `/api/training-jobs/${jobId}/logs?sort=asc&limit=1000&fields=compact` +
          `&afterTimestamp=${encodeURIComponent(afterTs)}&afterId=${encodeURIComponent(afterId)}`,
      )
      if (!response.ok) return
      const data: LogsResponse = await response.json()
      setStats(data.stats)
      if (data.data.length === 0) break
      added.push(...data.data)
      const last = data.data[data.data.length - 1]
      afterTs = last.timestamp
      afterId = last.id
      logCursorRef.current = { timestamp: afterTs, id: afterId }
      if (data.data.length < 1000) break
    }
    if (added.length > 0) setLogs((prev) => [...prev, ...added])
  }, [])

  const loadFallbackLogs = useCallback(async (jobId: string, page: number) => {
    const response = await fetch(`/api/training-jobs/${jobId}/logs?sort=desc&limit=1000&page=${page}`)
    if (!response.ok) return
    const data: LogsResponse = await response.json()
    const ascending = [...data.data].reverse()
    setFbTotalPages(data.pagination.totalPages)
    setFbPage(page)
    // Desc pages arrive newest-first; older pages are prepended so `fbLogs`
    // stays oldest→newest.
    setFbLogs((prev) => (page === 1 ? ascending : [...ascending, ...prev]))
  }, [])

  // Initial console load: tail of the file. `available: false` means the run
  // predates console capture — fall back to the parsed rows.
  const loadConsoleTail = useCallback(async (jobId: string, signal?: AbortSignal) => {
    try {
      const response = await fetch(`/api/training-jobs/${jobId}/console`, { signal })
      if (!response.ok) return
      const r: ConsoleRangeResponse = await response.json()
      setConsoleAvailable(r.available)
      setConsoleAmp(r.amp ?? null)
      if (!r.available) {
        setConsoleText('')
        consoleLoadedRef.current = false
        await loadFallbackLogs(jobId, 1)
        return
      }
      consoleLoadedRef.current = true
      setConsoleSize(r.size)
      setConsoleText(r.text)
      consoleStartRef.current = r.start
      consoleEndRef.current = r.end
      setConsoleStart(r.start)
      setConsoleHasEarlier(r.start > 0)
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') return
      console.error('Failed to load console output:', error)
    }
  }, [loadFallbackLogs])

  // Live tail while running: fetch from the last known end offset. The server
  // never returns a partial line for a running job, so appending is safe.
  const appendConsole = useCallback(async (jobId: string) => {
    try {
      const response = await fetch(
        `/api/training-jobs/${jobId}/console?offset=${consoleEndRef.current}`,
      )
      if (!response.ok) return
      const r: ConsoleRangeResponse = await response.json()
      if (!r.available) return
      setConsoleAmp(r.amp ?? null)
      setConsoleSize(r.size)
      if (!consoleLoadedRef.current) {
        // The file appeared after the job was selected (e.g. selected while
        // pending, then started): load the tail and drop the fallback rows.
        setFbLogs([])
        setFbPage(0)
        setFbTotalPages(1)
        await loadConsoleTail(jobId)
        return
      }
      if (r.size < consoleEndRef.current) {
        // File was truncated or rewritten — resync from the tail.
        consoleEndRef.current = 0
        await loadConsoleTail(jobId)
        return
      }
      if (r.text) {
        setConsoleText((prev) => {
          let text = prev + r.text
          let start = consoleStartRef.current
          // Cap the in-memory window at 4 MB by dropping the oldest complete
          // lines; start moves forward so "Load earlier" can still reach them.
          while (byteLength(text) > CONSOLE_TEXT_LIMIT) {
            const nl = text.indexOf('\n')
            if (nl === -1) break
            start += byteLength(text.slice(0, nl + 1))
            text = text.slice(nl + 1)
          }
          consoleStartRef.current = start
          setConsoleStart(start)
          setConsoleHasEarlier(start > 0)
          return text
        })
      }
      consoleEndRef.current = r.end
      setConsoleHasEarlier(consoleStartRef.current > 0)
    } catch (error) {
      console.error('Failed to append console output:', error)
    }
  }, [loadConsoleTail])

  // "Load earlier": prepend the chunk ending at the current start offset.
  const loadConsoleEarlier = useCallback(async (jobId: string) => {
    try {
      const response = await fetch(
        `/api/training-jobs/${jobId}/console?before=${consoleStartRef.current}`,
      )
      if (!response.ok) return
      const r: ConsoleRangeResponse = await response.json()
      if (!r.available || !r.text) {
        setConsoleHasEarlier(false)
        return
      }
      setConsoleText((prev) => r.text + prev)
      consoleStartRef.current = r.start
      setConsoleStart(r.start)
      setConsoleHasEarlier(r.hasMore)
    } catch (error) {
      console.error('Failed to load earlier console output:', error)
    }
  }, [])

  const runConsoleSearch = useCallback(async (jobId: string, query: string) => {
    const q = query.trim()
    if (!q) {
      setSearchResults(null)
      return
    }
    setConsoleSearching(true)
    try {
      const response = await fetch(
        `/api/training-jobs/${jobId}/console?q=${encodeURIComponent(q)}`,
      )
      if (response.ok) setSearchResults(await response.json())
    } finally {
      setConsoleSearching(false)
    }
  }, [])

  // Initial fetch
  useEffect(() => {
    fetchJobs()
  }, [fetchJobs])

  // Keep selectedJob in sync with the polled jobs list — a plain derivation,
  // it must not reload logs.
  useEffect(() => {
    setSelectedJob(jobs.find((job) => job.id === selectedJobId) || null)
  }, [jobs, selectedJobId])

  // Full reload on job selection only.
  useEffect(() => {
    if (!selectedJobId) return
    logsRequestRef.current?.abort()
    const controller = new AbortController()
    logsRequestRef.current = controller

    setLogs([])
    setStats(null)
    setLogsError(null)
    logCursorRef.current = null
    setConsoleAvailable(null)
    setConsoleText('')
    setConsoleAmp(null)
    setConsoleSize(0)
    consoleStartRef.current = 0
    consoleEndRef.current = 0
    consoleLoadedRef.current = false
    setConsoleStart(0)
    setConsoleHasEarlier(false)
    setSearchResults(null)
    setSearchQuery('')
    setFbLogs([])
    setFbPage(0)
    setFbTotalPages(1)
    prevStatusRef.current = undefined

    setLogsLoading(true)
    ;(async () => {
      try {
        await loadFullLogs(selectedJobId, controller.signal)
        if (controller.signal.aborted) return
        await loadConsoleTail(selectedJobId, controller.signal)
        if (controller.signal.aborted) return
        setLogsError(null)
      } catch (error) {
        if (!controller.signal.aborted) {
          setLogsError(error instanceof Error ? error.message : 'Failed to load logs')
        }
      } finally {
        if (!controller.signal.aborted) setLogsLoading(false)
      }
    })()
    return () => controller.abort()
  }, [selectedJobId, loadFullLogs, loadConsoleTail])

  // Polling while the selected job is running (or pending, so a job started
  // from elsewhere transitions into live monitoring without a reload): job
  // list + incremental logs + console append. The in-flight ref keeps 5s
  // ticks from overlapping when a fetch is slow.
  useEffect(() => {
    const pollable = selectedJob?.status === 'running' || selectedJob?.status === 'pending'
    if (!autoRefresh || !selectedJobId || !pollable) return
    const tick = async () => {
      if (pollInFlightRef.current) return
      pollInFlightRef.current = true
      try {
        await fetchJobs()
        await loadIncrementalLogs(selectedJobId)
        await appendConsole(selectedJobId)
      } catch {
        // Keep polling — a dropped request should not stop live updates.
      } finally {
        pollInFlightRef.current = false
      }
    }
    const interval = setInterval(tick, 5000)
    return () => clearInterval(interval)
  }, [autoRefresh, selectedJobId, selectedJob?.status, fetchJobs, loadIncrementalLogs, appendConsole])

  // When the job leaves `running`, do one final incremental fetch so the last
  // lines and the run footer are not missed between polls.
  useEffect(() => {
    const prev = prevStatusRef.current
    const current = selectedJob?.status
    prevStatusRef.current = current
    if (prev === 'running' && current && current !== 'running' && selectedJobId) {
      ;(async () => {
        await loadIncrementalLogs(selectedJobId)
        await appendConsole(selectedJobId)
      })()
    }
  }, [selectedJob?.status, selectedJobId, loadIncrementalLogs, appendConsole])

  // Auto-scroll to bottom
  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
    }
  }, [consoleText, autoScroll])

  // EVAL rows are part of `logs` now (complete history, one fetch path), so
  // evaluation charts just filter for the rows carrying eval metrics.
  const evaluationLogs = logs.filter(
    (log) =>
      log.mAP !== null || log.mAP50 !== null || log.mIoU !== null ||
      log.imageAuroc !== null || log.imageF1 !== null ||
      log.pixelAuroc !== null || log.pixelF1 !== null,
  )

  // Prepare chart data. Very long runs are min/max-decimated so tens of
  // thousands of points don't stall recharts — peaks are preserved per bucket.
  const lossChartFull = logs
    .filter(log => log.loss !== null)
    .map(log => ({
      iteration: log.iteration,
      epoch: log.epoch,
      loss: log.loss,
      lossCls: log.lossCls,
      lossIou: log.lossIou,
      lossDfl: log.lossDfl,
      lossL1: log.lossL1,
    }))
  const lossChartData = decimateMinMax(lossChartFull, (r) => r.loss, 4000)
  const lossDecimated = lossChartData.length < lossChartFull.length

  const lrChartFull = logs
    .filter(log => log.learningRate !== null)
    .map(log => ({
      iteration: log.iteration,
      epoch: log.epoch,
      learningRate: log.learningRate,
    }))
  const lrChartData = decimateMinMax(lrChartFull, (r) => r.learningRate, 4000)
  const lrDecimated = lrChartData.length < lrChartFull.length

  // Segmentation evaluation metrics over successive evaluations
  const segMetricChartData = evaluationLogs
    .filter(log => log.mIoU !== null)
    .map((log, i) => ({
      step: i + 1,
      mIoU: log.mIoU,
      acc: log.acc,
      kappa: log.kappa,
      dice: log.dice,
    }))

  // Detection evaluation metrics over successive evaluations
  const detMetricChartData = evaluationLogs
    .filter(log => log.mAP !== null || log.mAP50 !== null)
    .map((log, i) => ({
      step: i + 1,
      epoch: log.epoch,
      mAP: log.mAP,
      mAP50: log.mAP50,
    }))

  // Anomaly-detection metrics over successive evaluations. `iteration` is kept
  // as the x-axis label because an anomaly run is measured in steps, and the
  // adapter records the step each evaluation happened at.
  const anomalyMetricChartData = evaluationLogs
    .filter(log => log.imageAuroc !== null || log.pixelAuroc !== null)
    .map((log, i) => ({
      step: i + 1,
      iteration: log.iteration,
      imageAuroc: log.imageAuroc,
      imageF1: log.imageF1,
      pixelAuroc: log.pixelAuroc,
      pixelF1: log.pixelF1,
      threshold: log.threshold,
    }))
  const latestAnomalyEvalLog = anomalyMetricChartData[anomalyMetricChartData.length - 1]

  // Rows whose parsed classMetrics JSON carries a `nonFinite` list (written by
  // the server when a metric printed NaN/Inf). These rows are omitted from the
  // charts because their numeric fields are null — surface that as a warning
  // rather than letting the curve silently skip them. `logs` holds the
  // complete history, so scanning it alone suffices.
  const nonFiniteRows = logs.flatMap((log) => {
    if (!log.classMetrics) return []
    try {
      const parsed = JSON.parse(log.classMetrics)
      return Array.isArray(parsed?.nonFinite) && parsed.nonFinite.length > 0
        ? [{ log, names: parsed.nonFinite as string[] }]
        : []
    } catch {
      return []
    }
  })
  const nonFiniteNames = [...new Set(nonFiniteRows.flatMap((r) => r.names))]
  const firstNonFinite = nonFiniteRows[0]

  // Get latest log for current metrics
  const latestLog = logs[logs.length - 1]

  // Latest segmentation EVAL row (has mIoU + optional per-class metrics). Kept
  // separate from `latestLog` because EVAL rows are interleaved with far
  // more numerous TRAIN rows; the last TRAIN row won't have mIoU set.
  const latestEvalLog = [...evaluationLogs].reverse().find((l) => l.mIoU !== null)

  // Same idea for detection: the newest row carrying COCO metrics.
  const latestDetEvalLog = [...evaluationLogs].reverse().find((l) => l.mAP !== null || l.mAP50 !== null)

  // Parse latest per-class metrics JSON blob into arrays (safe: catch bad
  // JSON so a corrupt row doesn't take down the page).
  const latestClassMetrics: {
    iou?: (number | null)[]
    precision?: (number | null)[]
    recall?: (number | null)[]
  } | null = (() => {
    if (!latestEvalLog?.classMetrics) return null
    try {
      return JSON.parse(latestEvalLog.classMetrics)
    } catch {
      return null
    }
  })()

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[600px]">
        <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Training Monitor</h1>
          <p className="text-muted-foreground">
            Real-time training job monitoring and log visualization
          </p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Label htmlFor="auto-refresh" className="text-sm">Auto Refresh</Label>
            <Switch
              id="auto-refresh"
              checked={autoRefresh}
              onCheckedChange={setAutoRefresh}
            />
          </div>
        </div>
      </div>

      {/* Job Selection */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">Select Training Job</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <Select value={selectedJobId} onValueChange={setSelectedJobId}>
              <SelectTrigger className="w-[400px]">
                <SelectValue placeholder="Select a training job" />
              </SelectTrigger>
              <SelectContent>
                {jobs.length === 0 ? (
                  <SelectItem value="none" disabled>No training jobs available</SelectItem>
                ) : (
                  jobs.map(job => (
                    <SelectItem key={job.id} value={job.id}>
                      <div className="flex items-center gap-2">
                        <span>{job.name}</span>
                        <Badge 
                          variant={getStatusBadge(job.status).variant}
                          className={getStatusBadge(job.status).color}
                        >
                          {job.status}
                        </Badge>
                      </div>
                    </SelectItem>
                  ))
                )}
              </SelectContent>
            </Select>
          </div>

          {/* Job Info */}
          {selectedJob && (
            <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-muted-foreground">Project:</span>
                <span className="ml-2 font-medium">{selectedJob.project.name}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Dataset:</span>
                <span className="ml-2 font-medium">{selectedJob.dataset.name}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Model:</span>
                <span className="ml-2 font-medium">{selectedJob.model.name}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Started:</span>
                <span className="ml-2 font-medium">{formatTime(selectedJob.startedAt)}</span>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {selectedJob ? (
        <>
          {/* Progress Panel */}
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Epoch Progress</CardTitle>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {selectedJob.currentEpoch} / {selectedJob.totalEpochs}
                </div>
                <Progress 
                  value={(selectedJob.currentEpoch / selectedJob.totalEpochs) * 100} 
                  className="mt-2"
                />
                <p className="text-xs text-muted-foreground mt-2">
                  {((selectedJob.currentEpoch / selectedJob.totalEpochs) * 100).toFixed(1)}% complete
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Current Loss</CardTitle>
                <TrendingDown className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {selectedJob.currentLoss?.toFixed(4) || 'N/A'}
                </div>
                {stats && (
                  <p className="text-xs text-muted-foreground mt-2">
                    Avg: {stats.avgLoss?.toFixed(4) || 'N/A'}
                  </p>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Learning Rate</CardTitle>
                <Zap className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {selectedJob.currentLr?.toExponential(2) || 'N/A'}
                </div>
                {stats && (
                  <p className="text-xs text-muted-foreground mt-2">
                    Avg: {stats.avgLearningRate?.toExponential(2) || 'N/A'}
                  </p>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">ETA</CardTitle>
                <Timer className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {formatETA(latestLog?.eta || null)}
                </div>
                <p className="text-xs text-muted-foreground mt-2">
                  Elapsed: {formatDuration(selectedJob.startedAt, selectedJob.completedAt)}
                </p>
              </CardContent>
            </Card>
          </div>

          {/* Charts */}
          {logsError && <p role="alert" className="text-sm text-destructive">{logsError}</p>}
          {nonFiniteRows.length > 0 && firstNonFinite && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Non-finite values detected</AlertTitle>
              <AlertDescription>
                Non-finite values (NaN/Inf) detected in {nonFiniteRows.length} log line
                {nonFiniteRows.length === 1 ? '' : 's'}, first at epoch {firstNonFinite.log.epoch}{' '}
                iter {firstNonFinite.log.iteration}: {nonFiniteNames.join(', ')}. These points are
                omitted from the charts — training has likely diverged (try lower learning rate,
                disable AMP, or check labels).
              </AlertDescription>
            </Alert>
          )}
          {consoleAmp && consoleAmp.count > 0 && (
            <Alert
              variant={consoleAmp.currentScale !== null && consoleAmp.currentScale <= 1 || consoleAmp.count >= 10 ? 'destructive' : 'default'}
              className={consoleAmp.currentScale !== null && consoleAmp.currentScale <= 1 || consoleAmp.count >= 10 ? '' : 'border-amber-500/60 bg-amber-500/10 text-amber-700 dark:text-amber-300'}
            >
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>AMP loss-scale overflow</AlertTitle>
              <AlertDescription>
                Found inf/nan in gradients {consoleAmp.count} time(s) in the current run; loss
                scale fell from {consoleAmp.initialScale ?? '?'} to {consoleAmp.currentScale ?? '?'}{' '}
                (each event skips that step). A few early events are normal with AMP; repeated
                drops mean gradients keep overflowing — lower the learning rate, check
                labels/boxes, or disable AMP.
              </AlertDescription>
            </Alert>
          )}
          <div className="grid gap-6 lg:grid-cols-2">
            {/* Loss Chart */}
            <Card>
              <CardHeader>
                <CardTitle>Loss Over Time</CardTitle>
                <CardDescription>
                  Training loss metrics across iterations
                </CardDescription>
              </CardHeader>
              <CardContent>
                {lossChartData.length > 0 ? (
                  <ChartContainer config={lossChartConfig} className="h-[300px] w-full">
                    <LineChart data={lossChartData}>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                      <XAxis
                        dataKey="iteration"
                        className="text-xs"
                        tickFormatter={(value) => value.toLocaleString()}
                      />
                      <YAxis className="text-xs" />
                      <ChartTooltip content={<ChartTooltipContent />} />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="loss"
                        stroke="var(--chart-1)"
                        strokeWidth={2}
                        dot={false}
                        name="Total Loss"
                      />
                      <Line
                        type="monotone"
                        dataKey="lossCls"
                        stroke="var(--chart-2)"
                        strokeWidth={1.5}
                        dot={false}
                        name="Cls Loss"
                      />
                      <Line
                        type="monotone"
                        dataKey="lossIou"
                        stroke="var(--chart-3)"
                        strokeWidth={1.5}
                        dot={false}
                        name="IoU Loss"
                      />
                      <Line
                        type="monotone"
                        dataKey="lossDfl"
                        stroke="var(--chart-4)"
                        strokeWidth={1.5}
                        dot={false}
                        name="DFL Loss"
                      />
                      <Line
                        type="monotone"
                        dataKey="lossL1"
                        stroke="var(--chart-5)"
                        strokeWidth={1.5}
                        dot={false}
                        name="L1 Loss"
                      />
                    </LineChart>
                  </ChartContainer>
                ) : (
                  <div className="flex items-center justify-center h-[300px] text-muted-foreground">
                    No loss data available
                  </div>
                )}
                {lossDecimated && (
                  <p className="mt-2 text-xs text-muted-foreground">
                    Showing {lossChartData.length.toLocaleString()} of {lossChartFull.length.toLocaleString()} points (peaks preserved)
                  </p>
                )}
              </CardContent>
            </Card>

            {/* Learning Rate Chart */}
            <Card>
              <CardHeader>
                <CardTitle>Learning Rate Schedule</CardTitle>
                <CardDescription>
                  Learning rate changes during training
                </CardDescription>
              </CardHeader>
              <CardContent>
                {lrChartData.length > 0 ? (
                  <ChartContainer config={lrChartConfig} className="h-[300px] w-full">
                    <LineChart data={lrChartData}>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                      <XAxis
                        dataKey="iteration"
                        className="text-xs"
                        tickFormatter={(value) => value.toLocaleString()}
                      />
                      <YAxis
                        className="text-xs"
                        tickFormatter={(value) => value.toExponential(1)}
                      />
                      <ChartTooltip content={<ChartTooltipContent />} />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="learningRate"
                        stroke="var(--chart-1)"
                        strokeWidth={2}
                        dot={false}
                        name="Learning Rate"
                      />
                    </LineChart>
                  </ChartContainer>
                ) : (
                  <div className="flex items-center justify-center h-[300px] text-muted-foreground">
                    No learning rate data available
                  </div>
                )}
                {lrDecimated && (
                  <p className="mt-2 text-xs text-muted-foreground">
                    Showing {lrChartData.length.toLocaleString()} of {lrChartFull.length.toLocaleString()} points (peaks preserved)
                  </p>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Detection Evaluation Metrics (PaddleDetection, TorchDet) */}
          {isDetection(selectedJob.project.framework) && (
            <Card>
              <CardHeader>
                <CardTitle>Detection Metrics</CardTitle>
                <CardDescription>
                  Validation COCO mAP by training epoch
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {detMetricChartData.length === 0 ? (
                  <p className="py-8 text-center text-sm text-muted-foreground">
                    {logsLoading ? 'Loading validation results…'
                      : selectedJob.trainingParams?.doEval === false ||
                        (selectedJob.command && !/\s--(?:do_eval|eval)(?:\s|=|$)/.test(selectedJob.command))
                        ? 'Validation is disabled for this job. Enable “Validate during training” when creating a new job to record val metrics.'
                        : 'No validation results yet. Results appear after the first evaluation at the configured validation interval.'}
                  </p>
                ) : (
                <>
                <div className="grid gap-4 md:grid-cols-3">
                  <div className="p-4 rounded-lg border bg-muted/50">
                    <div className="text-sm text-muted-foreground mb-1">Best mAP@0.5:0.95</div>
                    <div className="text-2xl font-bold">
                      {selectedJob?.bestMetric?.toFixed(4) ?? 'N/A'}
                    </div>
                  </div>
                  <div className="p-4 rounded-lg border bg-muted/50">
                    <div className="text-sm text-muted-foreground mb-1">Best Checkpoint</div>
                    <div className="text-2xl font-bold">
                      epoch {selectedJob?.bestIter ?? '—'}
                    </div>
                  </div>
                  <div className="p-4 rounded-lg border bg-muted/50">
                    <div className="text-sm text-muted-foreground mb-1">Latest mAP@0.5</div>
                    <div className="text-2xl font-bold">
                      {latestDetEvalLog?.mAP50?.toFixed(4) ?? 'N/A'}
                    </div>
                  </div>
                </div>

                <ChartContainer config={detMetricChartConfig} className="h-[300px] w-full">
                  <LineChart data={detMetricChartData}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                    <XAxis dataKey="epoch" className="text-xs" label={{ value: 'Epoch', position: 'insideBottomRight', offset: -5 }} />
                    <YAxis className="text-xs" domain={[0, 1]} />
                    <ChartTooltip content={<ChartTooltipContent />} />
                    <Legend />
                    <Line type="monotone" dataKey="mAP" stroke="var(--chart-1)" strokeWidth={2} dot={{ r: 3 }} connectNulls name="mAP@0.5:0.95" />
                    <Line type="monotone" dataKey="mAP50" stroke="var(--chart-2)" strokeWidth={1.5} dot={{ r: 3 }} connectNulls name="mAP@0.5" />
                  </LineChart>
                </ChartContainer>
                </>
                )}
              </CardContent>
            </Card>
          )}

          {/* Anomaly Detection Metrics (TorchAnomaly) */}
          {anomalyMetricChartData.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Anomaly Detection Metrics</CardTitle>
                <CardDescription>
                  Image / pixel AUROC and F1 across evaluations. Pixel metrics need masks
                  for the defect images; they stay empty without them.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid gap-4 md:grid-cols-4">
                  <div className="p-4 rounded-lg border bg-muted/50">
                    <div className="text-sm text-muted-foreground mb-1">Best monitored metric</div>
                    <div className="text-2xl font-bold">
                      {selectedJob?.bestMetric?.toFixed(4) ?? 'N/A'}
                    </div>
                  </div>
                  <div className="p-4 rounded-lg border bg-muted/50">
                    <div className="text-sm text-muted-foreground mb-1">Best Checkpoint</div>
                    <div className="text-2xl font-bold">
                      iter {selectedJob?.bestIter ?? '—'}
                    </div>
                  </div>
                  <div className="p-4 rounded-lg border bg-muted/50">
                    <div className="text-sm text-muted-foreground mb-1">Latest Image AUROC</div>
                    <div className="text-2xl font-bold">
                      {latestAnomalyEvalLog?.imageAuroc?.toFixed(4) ?? 'N/A'}
                    </div>
                  </div>
                  <div className="p-4 rounded-lg border bg-muted/50">
                    {/* The score threshold is what a deployed model actually
                        compares against, so it is worth surfacing even though it
                        is not plotted. */}
                    <div className="text-sm text-muted-foreground mb-1">Score threshold</div>
                    <div className="text-2xl font-bold">
                      {latestAnomalyEvalLog?.threshold?.toFixed(4) ?? 'N/A'}
                    </div>
                  </div>
                </div>

                <ChartContainer config={anomalyMetricChartConfig} className="h-[300px] w-full">
                  <LineChart data={anomalyMetricChartData}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                    <XAxis dataKey="step" className="text-xs" />
                    <YAxis className="text-xs" domain={[0, 1]} />
                    <ChartTooltip content={<ChartTooltipContent />} />
                    <Legend />
                    <Line type="monotone" dataKey="imageAuroc" stroke="var(--chart-1)" strokeWidth={2} dot={false} name="Image AUROC" />
                    <Line type="monotone" dataKey="imageF1" stroke="var(--chart-2)" strokeWidth={1.5} dot={false} name="Image F1" />
                    <Line type="monotone" dataKey="pixelAuroc" stroke="var(--chart-3)" strokeWidth={1.5} dot={false} name="Pixel AUROC" />
                    <Line type="monotone" dataKey="pixelF1" stroke="var(--chart-4)" strokeWidth={1.5} dot={false} name="Pixel F1" />
                  </LineChart>
                </ChartContainer>
              </CardContent>
            </Card>
          )}

          {/* Segmentation Evaluation Metrics (PaddleSeg, TorchSeg) */}
          {segMetricChartData.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Segmentation Metrics</CardTitle>
                <CardDescription>
                  mIoU / Acc / Kappa / Dice across evaluations
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Best checkpoint summary — sourced from EVAL "best mIoU (X)
                    was saved at iter N" line. Only shown for segmentation jobs
                    that have recorded at least one eval. */}
                {(selectedJob?.bestMetric !== null && selectedJob?.bestMetric !== undefined) && (
                  <div className="grid gap-4 md:grid-cols-3">
                    <div className="p-4 rounded-lg border bg-muted/50">
                      <div className="text-sm text-muted-foreground mb-1">Best mIoU</div>
                      <div className="text-2xl font-bold">
                        {selectedJob.bestMetric.toFixed(4)}
                      </div>
                    </div>
                    <div className="p-4 rounded-lg border bg-muted/50">
                      <div className="text-sm text-muted-foreground mb-1">Best Checkpoint</div>
                      <div className="text-2xl font-bold">
                        iter {selectedJob.bestIter ?? '—'}
                      </div>
                    </div>
                    <div className="p-4 rounded-lg border bg-muted/50">
                      <div className="text-sm text-muted-foreground mb-1">Latest mIoU</div>
                      <div className="text-2xl font-bold">
                        {latestEvalLog?.mIoU?.toFixed(4) ?? 'N/A'}
                      </div>
                    </div>
                  </div>
                )}

                <ChartContainer config={segMetricChartConfig} className="h-[300px] w-full">
                  <LineChart data={segMetricChartData}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                    <XAxis dataKey="step" className="text-xs" />
                    <YAxis className="text-xs" domain={[0, 1]} />
                    <ChartTooltip content={<ChartTooltipContent />} />
                    <Legend />
                    <Line type="monotone" dataKey="mIoU" stroke="var(--chart-1)" strokeWidth={2} dot={false} name="mIoU" />
                    <Line type="monotone" dataKey="acc" stroke="var(--chart-2)" strokeWidth={1.5} dot={false} name="Acc" />
                    <Line type="monotone" dataKey="kappa" stroke="var(--chart-3)" strokeWidth={1.5} dot={false} name="Kappa" />
                    <Line type="monotone" dataKey="dice" stroke="var(--chart-4)" strokeWidth={1.5} dot={false} name="Dice" />
                  </LineChart>
                </ChartContainer>

                {/* Per-class IoU / Precision / Recall table for the most
                    recent eval. Class labels are unknown at this layer
                    (they'd require a fetch to the dataset's `labels.txt`);
                    we show them as `Class 0..N` and let the shape speak. */}
                {latestClassMetrics && (
                  <div className="overflow-x-auto">
                    <div className="text-sm font-medium mb-2">
                      Latest per-class metrics (eval @ iter {latestEvalLog?.iteration || '—'})
                    </div>
                    <table className="w-full text-sm border-collapse">
                      <thead>
                        <tr className="border-b">
                          <th className="text-left py-2 px-3 font-medium text-muted-foreground">Metric</th>
                          {(latestClassMetrics.iou ||
                            latestClassMetrics.precision ||
                            latestClassMetrics.recall ||
                            []).map((_, i) => (
                            <th key={i} className="text-right py-2 px-3 font-medium text-muted-foreground">
                              Class {i}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {(['iou', 'precision', 'recall'] as const).map((key) => {
                          const arr = latestClassMetrics[key]
                          if (!arr) return null
                          return (
                            <tr key={key} className="border-b last:border-0">
                              <td className="py-2 px-3 font-medium capitalize">{key}</td>
                              {arr.map((v, i) => (
                                <td key={i} className="text-right py-2 px-3 font-mono">
                                  {typeof v === 'number' ? v.toFixed(4) : '—'}
                                </td>
                              ))}
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Performance Metrics */}
          <Card>
            <CardHeader>
              <CardTitle>Performance Metrics</CardTitle>
              <CardDescription>
                Training speed and resource utilization
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                <div className="p-4 rounded-lg border bg-muted/50">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground mb-1">
                    <Clock className="h-4 w-4" />
                    Batch Cost
                  </div>
                  <div className="text-xl font-bold">
                    {latestLog?.batchCost?.toFixed(4) || 'N/A'} s
                  </div>
                  {stats && (
                    <div className="text-xs text-muted-foreground">
                      Avg: {stats.avgBatchCost?.toFixed(4) || 'N/A'} s
                    </div>
                  )}
                </div>

                <div className="p-4 rounded-lg border bg-muted/50">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground mb-1">
                    <Clock className="h-4 w-4" />
                    Data Cost
                  </div>
                  <div className="text-xl font-bold">
                    {latestLog?.dataCost?.toFixed(4) || 'N/A'} s
                  </div>
                </div>

                <div className="p-4 rounded-lg border bg-muted/50">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground mb-1">
                    <Zap className="h-4 w-4" />
                    IPS (Images/sec)
                  </div>
                  <div className="text-xl font-bold">
                    {latestLog?.ips?.toFixed(2) || 'N/A'}
                  </div>
                  {stats && (
                    <div className="text-xs text-muted-foreground">
                      Avg: {stats.avgIps?.toFixed(2) || 'N/A'}
                    </div>
                  )}
                </div>

                <div className="p-4 rounded-lg border bg-muted/50">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground mb-1">
                    <HardDrive className="h-4 w-4" />
                    GPU Memory
                  </div>
                  <div className="text-xl font-bold">
                    {latestLog?.memReserved 
                      ? `${(latestLog.memReserved / 1024).toFixed(1)} GB`
                      : 'N/A'}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Allocated: {latestLog?.memAllocated 
                      ? `${(latestLog.memAllocated / 1024).toFixed(1)} GB`
                      : 'N/A'}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Console Output — verbatim stdout+stderr captured to
              autotrain-console.log next to the job's checkpoints. Older runs
              (no console file) fall back to parsed TrainingLog rows. */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Console Output</CardTitle>
                  <CardDescription>
                    {consoleAvailable
                      ? `${formatBytes(consoleSize)} · ${
                          consoleStart > 0 ? `showing from byte ${consoleStart.toLocaleString()}` : 'full file'
                        }`
                      : consoleAvailable === false
                        ? 'Parsed training log entries'
                        : 'Loading console output…'}
                  </CardDescription>
                </div>
                <div className="flex items-center gap-4">
                  {consoleAvailable && (
                    <>
                      <div className="relative">
                        <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                        <Input
                          placeholder="Search console..."
                          value={searchQuery}
                          onChange={(e) => setSearchQuery(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') {
                              if (searchQuery.trim()) {
                                runConsoleSearch(selectedJob.id, searchQuery)
                              } else {
                                setSearchResults(null)
                              }
                            }
                          }}
                          className="pl-8 w-[200px]"
                        />
                      </div>
                      {searchResults ? (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => { setSearchResults(null); setSearchQuery('') }}
                        >
                          Clear
                        </Button>
                      ) : (
                        consoleHasEarlier && (
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => loadConsoleEarlier(selectedJob.id)}
                          >
                            Load earlier
                          </Button>
                        )
                      )}
                      <a
                        href={`/api/training-jobs/${selectedJob.id}/console?download=1`}
                        title="Download full log"
                        className="text-muted-foreground hover:text-foreground"
                      >
                        <Download className="h-4 w-4" />
                      </a>
                    </>
                  )}
                  <div className="flex items-center gap-2">
                    <Label htmlFor="auto-scroll" className="text-sm">Auto Scroll</Label>
                    <Switch
                      id="auto-scroll"
                      checked={autoScroll}
                      onCheckedChange={setAutoScroll}
                    />
                  </div>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[400px] w-full rounded-md border bg-muted/30" ref={logContainerRef}>
                {consoleAvailable === false ? (
                  <div className="p-4 font-mono text-xs space-y-1">
                    <p className="font-sans text-sm text-muted-foreground mb-3">
                      Console output is recorded for runs started after this update.
                    </p>
                    {fbPage < fbTotalPages && (
                      <Button
                        variant="outline"
                        size="sm"
                        className="mb-2"
                        onClick={() => loadFallbackLogs(selectedJob.id, fbPage + 1)}
                      >
                        Load earlier
                      </Button>
                    )}
                    {fbLogs.length === 0 ? (
                      <div className="text-muted-foreground text-center py-8">No logs available</div>
                    ) : (
                      fbLogs.map((log) => (
                        <div key={log.id} className="hover:bg-muted/50 px-2 py-1 rounded">
                          <span className="text-muted-foreground mr-2">
                            [Epoch {log.epoch}] [Iter {log.iteration}/{log.totalIter}]
                          </span>
                          {log.rawLog || (
                            <span>
                              loss: {log.loss?.toFixed(6) || 'N/A'} |
                              lr: {log.learningRate?.toExponential(2) || 'N/A'} |
                              ips: {log.ips?.toFixed(2) || 'N/A'}
                            </span>
                          )}
                        </div>
                      ))
                    )}
                  </div>
                ) : searchResults ? (
                  <div className="p-4 font-mono text-xs space-y-1">
                    <div className="text-muted-foreground mb-2">
                      {searchResults.totalMatches} match(es)
                      {searchResults.truncated ? ` — showing first ${searchResults.matches.length}` : ''}
                    </div>
                    {searchResults.matches.length === 0 ? (
                      <div className="text-muted-foreground text-center py-8">No matches</div>
                    ) : (
                      searchResults.matches.map((m) => (
                        <div key={m.line} className={`px-2 py-1 rounded ${consoleLineClass(m.text)}`}>
                          <span className="text-muted-foreground mr-2">line {m.line}:</span>
                          {m.text}
                        </div>
                      ))
                    )}
                  </div>
                ) : (
                  <pre className="whitespace-pre-wrap break-all p-4 font-mono text-xs">
                    {consoleSearching ? (
                      <div className="text-muted-foreground text-center py-8">Searching…</div>
                    ) : consoleText.length === 0 ? (
                      <div className="text-muted-foreground text-center py-8">
                        {consoleAvailable === null ? 'Loading…' : 'No console output yet'}
                      </div>
                    ) : (
                      consoleText.split('\n').map((line, i) => (
                        <div key={i} className={`px-2 rounded ${consoleLineClass(line)}`}>
                          {line || ' '}
                        </div>
                      ))
                    )}
                  </pre>
                )}
                <ScrollBar orientation="vertical" />
              </ScrollArea>
            </CardContent>
          </Card>
        </>
      ) : (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <AlertCircle className="w-12 h-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium">No Training Job Selected</h3>
            <p className="text-sm text-muted-foreground mt-1">
              Select a training job from the dropdown above to view monitoring data
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
