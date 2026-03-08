'use client'

import { useEffect, useState, useRef, useCallback } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
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
  ResponsiveContainer,
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
  ips: number | null
  memReserved: number | null
  memAllocated: number | null
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

// Chart configurations
const lossChartConfig = {
  loss: {
    label: "Total Loss",
    color: "hsl(var(--chart-1))",
  },
  lossCls: {
    label: "Cls Loss",
    color: "hsl(var(--chart-2))",
  },
  lossIou: {
    label: "IoU Loss",
    color: "hsl(var(--chart-3))",
  },
  lossDfl: {
    label: "DFL Loss",
    color: "hsl(var(--chart-4))",
  },
  lossL1: {
    label: "L1 Loss",
    color: "hsl(var(--chart-5))",
  },
} satisfies ChartConfig

const lrChartConfig = {
  learningRate: {
    label: "Learning Rate",
    color: "hsl(var(--chart-1))",
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
  const [stats, setStats] = useState<LogsResponse['stats'] | null>(null)
  const [loading, setLoading] = useState(true)
  const [logsLoading, setLogsLoading] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [autoScroll, setAutoScroll] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const logContainerRef = useRef<HTMLDivElement>(null)
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null)

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

  // Fetch job logs
  const fetchJobLogs = useCallback(async (jobId: string) => {
    if (!jobId) return
    
    setLogsLoading(true)
    try {
      const response = await fetch(`/api/training-jobs/${jobId}/logs?limit=500&sort=asc`)
      if (response.ok) {
        const data: LogsResponse = await response.json()
        setLogs(data.data)
        setStats(data.stats)
        
        // Update selected job from the jobs list
        const job = jobs.find(j => j.id === jobId)
        if (job) {
          setSelectedJob(job)
        }
      }
    } catch (error) {
      console.error('Failed to fetch logs:', error)
    } finally {
      setLogsLoading(false)
    }
  }, [jobs])

  // Initial fetch
  useEffect(() => {
    fetchJobs()
  }, [fetchJobs])

  // Fetch logs when job is selected
  useEffect(() => {
    if (selectedJobId) {
      fetchJobLogs(selectedJobId)
    }
  }, [selectedJobId, fetchJobLogs])

  // Polling for real-time updates
  useEffect(() => {
    if (autoRefresh && selectedJobId && selectedJob?.status === 'running') {
      pollingIntervalRef.current = setInterval(() => {
        fetchJobLogs(selectedJobId)
        fetchJobs() // Refresh job list to get status updates
      }, 5000)
    }

    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current)
      }
    }
  }, [autoRefresh, selectedJobId, selectedJob?.status, fetchJobLogs, fetchJobs])

  // Auto-scroll to bottom
  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
    }
  }, [logs, autoScroll])

  // Prepare chart data
  const lossChartData = logs
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

  const lrChartData = logs
    .filter(log => log.learningRate !== null)
    .map(log => ({
      iteration: log.iteration,
      epoch: log.epoch,
      learningRate: log.learningRate,
    }))

  // Filter logs for raw log viewer
  const filteredLogs = logs.filter(log => {
    if (!searchQuery) return true
    const query = searchQuery.toLowerCase()
    return (
      log.rawLog?.toLowerCase().includes(query) ||
      log.epoch.toString().includes(query) ||
      log.iteration.toString().includes(query)
    )
  })

  // Get latest log for current metrics
  const latestLog = logs[logs.length - 1]

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
                    <ResponsiveContainer width="100%" height="100%">
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
                          stroke="hsl(var(--chart-1))"
                          strokeWidth={2}
                          dot={false}
                          name="Total Loss"
                        />
                        <Line
                          type="monotone"
                          dataKey="lossCls"
                          stroke="hsl(var(--chart-2))"
                          strokeWidth={1.5}
                          dot={false}
                          name="Cls Loss"
                        />
                        <Line
                          type="monotone"
                          dataKey="lossIou"
                          stroke="hsl(var(--chart-3))"
                          strokeWidth={1.5}
                          dot={false}
                          name="IoU Loss"
                        />
                        <Line
                          type="monotone"
                          dataKey="lossDfl"
                          stroke="hsl(var(--chart-4))"
                          strokeWidth={1.5}
                          dot={false}
                          name="DFL Loss"
                        />
                        <Line
                          type="monotone"
                          dataKey="lossL1"
                          stroke="hsl(var(--chart-5))"
                          strokeWidth={1.5}
                          dot={false}
                          name="L1 Loss"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </ChartContainer>
                ) : (
                  <div className="flex items-center justify-center h-[300px] text-muted-foreground">
                    No loss data available
                  </div>
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
                    <ResponsiveContainer width="100%" height="100%">
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
                        <ChartTooltip 
                          content={<ChartTooltipContent />}
                          formatter={(value: number) => value.toExponential(4)}
                        />
                        <Legend />
                        <Line
                          type="monotone"
                          dataKey="learningRate"
                          stroke="hsl(var(--chart-1))"
                          strokeWidth={2}
                          dot={false}
                          name="Learning Rate"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </ChartContainer>
                ) : (
                  <div className="flex items-center justify-center h-[300px] text-muted-foreground">
                    No learning rate data available
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

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

          {/* Raw Log Viewer */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Raw Log Output</CardTitle>
                  <CardDescription>
                    Training log entries ({filteredLogs.length} of {logs.length} logs)
                  </CardDescription>
                </div>
                <div className="flex items-center gap-4">
                  <div className="relative">
                    <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                    <Input
                      placeholder="Search logs..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-8 w-[200px]"
                    />
                  </div>
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
                <div className="p-4 font-mono text-xs space-y-1">
                  {filteredLogs.length === 0 ? (
                    <div className="text-muted-foreground text-center py-8">
                      {searchQuery ? 'No logs match your search' : 'No logs available'}
                    </div>
                  ) : (
                    filteredLogs.map((log) => (
                      <div 
                        key={log.id}
                        className="hover:bg-muted/50 px-2 py-1 rounded"
                      >
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
