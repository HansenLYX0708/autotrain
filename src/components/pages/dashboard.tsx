'use client'

import { useEffect, useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import {
  FolderKanban,
  Database,
  Cpu,
  PlayCircle,
  Activity,
  Cpu as GpuIcon,
  Thermometer,
  HardDrive,
  Settings,
  TrendingUp,
  AlertCircle,
  RefreshCw,
  Monitor,
  CheckCircle,
  XCircle,
  Terminal,
  AlertTriangle,
} from 'lucide-react'
import {
  ChartConfig,
  ChartContainer,
} from "@/components/ui/chart"
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from 'recharts'

interface DashboardStats {
  projects: number
  datasets: number
  models: number
  runningJobs: number
  completedJobs: number
}

interface GpuInfo {
  id: number
  name: string
  utilization: number
  memoryUsed: number
  memoryTotal: number
  temperature: number
}

interface GpuEnvironmentCheck {
  gpuId: number
  pythonPath: string
  exists: boolean
  version: string | null
  isValid: boolean
  error?: string
}

interface SystemConfig {
  pythonPath: string
  paddleDetectionPath: string
  paddleClasPath: string
}

interface EnvironmentCheck {
  paddleDetection: {
    exists: boolean
    version: string | null
    isValid: boolean
    error?: string
  }
  gpuEnvironments: GpuEnvironmentCheck[]
  totalGpus: number
  configuredGpus: number
  validGpus: number
}

interface StorageInfo {
  maxStorageQuota: number
  usedStorage: number
  availableStorage: number
  userDatabasePath: string | null
}

const chartConfig = {
  gpu: {
    label: "GPU Utilization",
    color: "hsl(var(--chart-1))",
  },
  memory: {
    label: "Memory",
    color: "hsl(var(--chart-2))",
  },
} satisfies ChartConfig

export function DashboardPage() {
  const [stats, setStats] = useState<DashboardStats>({
    projects: 0,
    datasets: 0,
    models: 0,
    runningJobs: 0,
    completedJobs: 0,
  })
  const [gpuInfo, setGpuInfo] = useState<GpuInfo[]>([])
  const [hasNvidia, setHasNvidia] = useState(false)
  const [platform, setPlatform] = useState('')
  const [systemConfig, setSystemConfig] = useState<SystemConfig>({
    pythonPath: '',
    paddleDetectionPath: '',
    paddleClasPath: '',
  })
  const [gpuHistory, setGpuHistory] = useState<{ time: string; gpu: number; memory: number }[]>([])
  const [loading, setLoading] = useState(true)
  const [gpuLoading, setGpuLoading] = useState(true)
  const [environmentCheck, setEnvironmentCheck] = useState<EnvironmentCheck | null>(null)
  const [envCheckLoading, setEnvCheckLoading] = useState(false)
  const [occupiedGpuIds, setOccupiedGpuIds] = useState<number[]>([])
  const [gpuUsageMap, setGpuUsageMap] = useState<Map<number, string[]>>(new Map())
  const [storageInfo, setStorageInfo] = useState<StorageInfo | null>(null)
  const [storageLoading, setStorageLoading] = useState(true)

  // Fetch GPU usage info from running jobs
  const fetchGpuUsage = async () => {
    try {
      const response = await fetch('/api/system/gpu-usage')
      if (response.ok) {
        const result = await response.json()
        if (result.success && result.data) {
          setOccupiedGpuIds(result.data.occupiedGpuIds || [])
          // Build map of GPU ID to job names
          const usageMap = new Map<number, string[]>()
          for (const usage of result.data.gpuUsage || []) {
            usageMap.set(usage.gpuId, usage.jobNames || [])
          }
          setGpuUsageMap(usageMap)
        }
      }
    } catch (error) {
      console.error('Failed to fetch GPU usage:', error)
    }
  }

  // Fetch GPU info
  const fetchGpuInfo = async () => {
    try {
      const response = await fetch('/api/system/gpu')
      if (response.ok) {
        const result = await response.json()
        const data = result.data
        setGpuInfo(data.gpus || [])
        setHasNvidia(data.hasNvidia || false)
        setPlatform(data.platform || 'unknown')
        
        // Update history
        setGpuHistory(prev => {
          const newEntry = {
            time: new Date().toLocaleTimeString(),
            gpu: data.gpus?.[0]?.utilization || 0,
            memory: data.gpus?.[0] ? (data.gpus[0].memoryUsed / data.gpus[0].memoryTotal) * 100 : 0,
          }
          const newData = [...prev.slice(-19), newEntry]
          return newData
        })
      }
    } catch (error) {
      console.error('Failed to fetch GPU info:', error)
    } finally {
      setGpuLoading(false)
    }
  }

  // Fetch storage info
  const fetchStorageInfo = async () => {
    try {
      setStorageLoading(true)
      const response = await fetch('/api/users/storage')
      if (response.ok) {
        const result = await response.json()
        if (result.success && result.data) {
          setStorageInfo({
            maxStorageQuota: Number(result.data.maxStorageQuota),
            usedStorage: result.data.usedStorage,
            availableStorage: result.data.availableStorage,
            userDatabasePath: result.data.userDatabasePath,
          })
        }
      }
    } catch (error) {
      console.error('Failed to fetch storage info:', error)
    } finally {
      setStorageLoading(false)
    }
  }

  useEffect(() => {
    // Fetch dashboard stats
    const fetchStats = async () => {
      try {
        const response = await fetch('/api/dashboard/stats')
        if (response.ok) {
          const result = await response.json()
          const data = result.data || result
          setStats({
            projects: data.totalProjects || 0,
            datasets: data.totalDatasets || 0,
            models: data.totalModels || 0,
            runningJobs: data.runningTrainingJobs || 0,
            completedJobs: data.completedTrainingJobs || 0,
          })
        }
      } catch (error) {
        console.error('Failed to fetch stats:', error)
      }
    }

    // Fetch system config
    const fetchConfig = async () => {
      try {
        const response = await fetch('/api/settings')
        if (response.ok) {
          const result = await response.json()
          const data = result.data || result
          if (data) {
            setSystemConfig({
              pythonPath: data.pythonPath || '',
              paddleDetectionPath: data.paddleDetectionPath || '',
              paddleClasPath: data.paddleClasPath || '',
            })
          }
        }
      } catch (error) {
        console.error('Failed to fetch config:', error)
      } finally {
        setLoading(false)
      }
    }

    // Fetch environment check
    const fetchEnvironmentCheck = async () => {
      setEnvCheckLoading(true)
      try {
        const response = await fetch('/api/system/environment-check')
        if (response.ok) {
          const result = await response.json()
          if (result.success && result.data) {
            setEnvironmentCheck(result.data)
          }
        }
      } catch (error) {
        console.error('Failed to fetch environment check:', error)
      } finally {
        setEnvCheckLoading(false)
      }
    }

    fetchStats()
    fetchConfig()
    fetchEnvironmentCheck()
    fetchGpuInfo()
    fetchGpuUsage()
    fetchStorageInfo()

    // Poll GPU info every 30 seconds
    const interval = setInterval(() => {
      fetchGpuInfo()
      fetchGpuUsage()
    }, 30000)

    return () => clearInterval(interval)
  }, [])

  const statCards = [
    { name: 'Projects', value: stats.projects, icon: FolderKanban, color: 'text-blue-500', bgColor: 'bg-blue-500/10' },
    { name: 'Datasets', value: stats.datasets, icon: Database, color: 'text-emerald-500', bgColor: 'bg-emerald-500/10' },
    { name: 'Models', value: stats.models, icon: Cpu, color: 'text-purple-500', bgColor: 'bg-purple-500/10' },
    { name: 'Running Jobs', value: stats.runningJobs, icon: PlayCircle, color: 'text-orange-500', bgColor: 'bg-orange-500/10' },
  ]

  const formatMemory = (mb: number) => {
    if (mb >= 1024) {
      return `${(mb / 1024).toFixed(1)} GB`
    }
    return `${mb} MB`
  }

  const formatBytes = (bytes: number) => {
    if (bytes >= 1024 * 1024 * 1024 * 1024) {
      return `${(bytes / 1024 / 1024 / 1024 / 1024).toFixed(2)} TB`
    }
    if (bytes >= 1024 * 1024 * 1024) {
      return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`
    }
    if (bytes >= 1024 * 1024) {
      return `${(bytes / 1024 / 1024).toFixed(2)} MB`
    }
    return `${bytes} B`
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
          <p className="text-muted-foreground">
            Overview of your training platform
          </p>
        </div>
        <Button onClick={() => { fetchGpuInfo(); fetchGpuUsage(); fetchStorageInfo(); }}>
          <RefreshCw className="w-4 h-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {statCards.map((stat) => (
          <Card key={stat.name}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">{stat.name}</CardTitle>
              <div className={`p-2 rounded-lg ${stat.bgColor}`}>
                <stat.icon className={`h-4 w-4 ${stat.color}`} />
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stat.value}</div>
              {stat.name === 'Running Jobs' && stats.runningJobs > 0 && (
                <p className="text-xs text-muted-foreground flex items-center gap-1 mt-1">
                  <Activity className="w-3 h-3 text-green-500 animate-pulse" />
                  Training in progress
                </p>
              )}
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Main Content Grid */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* GPU Monitoring */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  {hasNvidia ? <GpuIcon className="w-5 h-5" /> : <Monitor className="w-5 h-5" />}
                  {hasNvidia ? 'GPU Monitoring' : 'System Monitoring'}
                </CardTitle>
                <CardDescription>
                  {hasNvidia ? 'Real-time GPU utilization and memory' : 'System resource usage'}
                  {platform && <span className="ml-2 text-xs">({platform})</span>}
                </CardDescription>
              </div>
              <Badge variant="outline" className="text-emerald-500 border-emerald-500/50">
                {gpuLoading ? 'Loading...' : 'Live'}
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            {/* GPU Cards */}
            {gpuInfo.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-8 text-center">
                <AlertCircle className="w-12 h-12 text-muted-foreground mb-4" />
                <h3 className="text-lg font-medium">No GPU detected</h3>
                <p className="text-sm text-muted-foreground mt-1">
                  System will use CPU for training
                </p>
              </div>
            ) : (
              <div className="grid gap-4 md:grid-cols-2 mb-6">
                {gpuInfo.map((gpu) => {
                  // Determine GPU status
                  const highMemory = gpu.memoryTotal > 0 && (gpu.memoryUsed / gpu.memoryTotal) >= 0.5
                  const highUtil = gpu.utilization >= 30
                  const isUsedBySystem = occupiedGpuIds.includes(gpu.id)
                  const isOccupied = highMemory || highUtil || isUsedBySystem
                  const isIdle = !isOccupied
                  
                  const statusConfig = {
                    idle: {
                      badge: 'bg-emerald-500/10 text-emerald-600 border-emerald-500/20',
                      label: 'Idle',
                      dot: 'bg-emerald-500',
                    },
                    occupied: {
                      badge: 'bg-orange-500/10 text-orange-600 border-orange-500/20',
                      label: 'Occupied',
                      dot: 'bg-orange-500',
                    },
                    system: {
                      badge: 'bg-blue-500/10 text-blue-600 border-blue-500/20',
                      label: 'In Training',
                      dot: 'bg-blue-500 animate-pulse',
                    },
                  }
                  const status = isIdle ? statusConfig.idle : (isUsedBySystem ? statusConfig.system : statusConfig.occupied)
                  const jobNames = gpuUsageMap.get(gpu.id) || []
                  
                  return (
                    <div key={gpu.id} className="p-4 rounded-lg border border-border bg-muted/50">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-sm truncate" title={gpu.name}>{gpu.name}</span>
                          <div className={`px-2 py-0.5 rounded-full text-xs font-medium border ${status.badge} flex items-center gap-1`}>
                            <div className={`w-1.5 h-1.5 rounded-full ${status.dot}`} />
                            {status.label}
                          </div>
                        </div>
                        <Badge variant="secondary">{hasNvidia ? `GPU ${gpu.id}` : 'CPU'}</Badge>
                      </div>
                      
                      <div className="space-y-3">
                        {jobNames.length > 0 && (
                          <div className="text-xs text-muted-foreground">
                            <span className="font-medium">Training Jobs: </span>
                            {jobNames.join(', ')}
                          </div>
                        )}
                        <div>
                          <div className="flex items-center justify-between text-sm mb-1">
                            <span className="text-muted-foreground flex items-center gap-1">
                              <Activity className="w-3 h-3" />
                              Utilization
                            </span>
                            <span className="font-medium">{gpu.utilization.toFixed(1)}%</span>
                          </div>
                          <Progress value={gpu.utilization} className="h-2" />
                        </div>
                        
                        <div>
                          <div className="flex items-center justify-between text-sm mb-1">
                            <span className="text-muted-foreground flex items-center gap-1">
                              <HardDrive className="w-3 h-3" />
                              Memory
                            </span>
                            <span className="font-medium">{formatMemory(gpu.memoryUsed)} / {formatMemory(gpu.memoryTotal)}</span>
                          </div>
                          <Progress value={gpu.memoryTotal > 0 ? (gpu.memoryUsed / gpu.memoryTotal) * 100 : 0} className="h-2" />
                        </div>
                        
                        {gpu.temperature > 0 && (
                          <div className="flex items-center justify-between text-sm">
                            <span className="text-muted-foreground flex items-center gap-1">
                              <Thermometer className="w-3 h-3" />
                              Temperature
                            </span>
                            <span className={`font-medium ${gpu.temperature > 80 ? 'text-red-500' : gpu.temperature > 70 ? 'text-yellow-500' : 'text-green-500'}`}>
                              {gpu.temperature}°C
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            )}

            {/* Chart */}
            {gpuHistory.length > 0 && (
              <ChartContainer config={chartConfig} className="h-[200px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={gpuHistory}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                    <XAxis 
                      dataKey="time" 
                      tickFormatter={(value) => value?.slice(0, 5) || ''}
                      className="text-xs"
                    />
                    <YAxis className="text-xs" />
                    <Area
                      type="monotone"
                      dataKey="gpu"
                      stackId="1"
                      stroke="hsl(var(--chart-1))"
                      fill="hsl(var(--chart-1))"
                      fillOpacity={0.3}
                    />
                    <Area
                      type="monotone"
                      dataKey="memory"
                      stackId="2"
                      stroke="hsl(var(--chart-2))"
                      fill="hsl(var(--chart-2))"
                      fillOpacity={0.3}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </ChartContainer>
            )}
          </CardContent>
        </Card>

        {/* System Configuration & Storage */}
        <div className="space-y-6">
          {/* Storage Usage Card */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <HardDrive className="w-5 h-5" />
                Storage Usage
              </CardTitle>
              <CardDescription>Your dataset storage quota</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {storageLoading ? (
                <div className="flex items-center justify-center py-4">
                  <RefreshCw className="w-5 h-5 animate-spin text-muted-foreground" />
                </div>
              ) : storageInfo ? (
                <>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Used</span>
                      <span className="font-medium">{formatBytes(storageInfo.usedStorage)}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Total</span>
                      <span className="font-medium">{formatBytes(storageInfo.maxStorageQuota)}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Available</span>
                      <span className="font-medium text-emerald-600">
                        {formatBytes(storageInfo.availableStorage)}
                      </span>
                    </div>
                  </div>
                  
                  <div className="space-y-1">
                    <Progress 
                      value={storageInfo.maxStorageQuota > 0 
                        ? (storageInfo.usedStorage / storageInfo.maxStorageQuota) * 100 
                        : 0
                      } 
                      className="h-2"
                    />
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span>
                        {storageInfo.maxStorageQuota > 0 
                          ? ((storageInfo.usedStorage / storageInfo.maxStorageQuota) * 100).toFixed(1) 
                          : 0}% used
                      </span>
                      <span className={storageInfo.availableStorage < 10 * 1024 * 1024 * 1024 ? 'text-red-500 font-medium' : ''}>
                        {storageInfo.availableStorage < 10 * 1024 * 1024 * 1024 && 'Low space warning'}
                      </span>
                    </div>
                  </div>

                  {storageInfo.userDatabasePath && (
                    <div className="pt-2 border-t">
                      <div className="text-xs text-muted-foreground">Storage Path</div>
                      <div className="text-xs font-mono truncate text-muted-foreground" title={storageInfo.userDatabasePath}>
                        {storageInfo.userDatabasePath}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="flex flex-col items-center justify-center py-4 text-center">
                  <AlertCircle className="w-8 h-8 text-muted-foreground mb-2" />
                  <p className="text-sm text-muted-foreground">Failed to load storage info</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* System Configuration */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Terminal className="w-5 h-5" />
                System Environment
              </CardTitle>
              <CardDescription>Version compatibility check</CardDescription>
            </CardHeader>
          <CardContent className="space-y-4">
            {/* GPU Python Environments */}
            <div className="space-y-3">
              <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                GPU Python Environments
              </div>
              {environmentCheck?.gpuEnvironments && environmentCheck.gpuEnvironments.length > 0 ? (
                environmentCheck.gpuEnvironments.map((gpuEnv) => (
                  <div key={gpuEnv.gpuId} className={`p-3 rounded-lg border ${gpuEnv.isValid ? 'border-emerald-500/30 bg-emerald-50/30' : gpuEnv.exists ? 'border-red-500/30 bg-red-50/30' : 'border-yellow-500/30 bg-yellow-50/30'}`}>
                    <div className="flex items-start gap-3">
                      <div className="mt-0.5">
                        {gpuEnv.isValid ? (
                          <CheckCircle className="w-5 h-5 text-emerald-500" />
                        ) : gpuEnv.exists ? (
                          <XCircle className="w-5 h-5 text-red-500" />
                        ) : (
                          <AlertTriangle className="w-5 h-5 text-yellow-500" />
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-medium text-sm">GPU {gpuEnv.gpuId}</span>
                          {gpuEnv.version && (
                            <Badge variant="outline" className="text-xs">
                              {gpuEnv.version}
                            </Badge>
                          )}
                        </div>
                        <div className="text-xs text-muted-foreground mt-1">
                          {gpuEnv.isValid ? (
                            <span className="text-emerald-600">Python {gpuEnv.version} ready</span>
                          ) : gpuEnv.error ? (
                            <span className="text-red-600">{gpuEnv.error}</span>
                          ) : (
                            <span>Not configured</span>
                          )}
                        </div>
                        <div className="text-xs font-mono truncate text-muted-foreground mt-1" title={gpuEnv.pythonPath}>
                          {gpuEnv.pythonPath || 'No path configured'}
                        </div>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="p-3 rounded-lg border border-yellow-500/30 bg-yellow-50/30">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="w-5 h-5 text-yellow-500 mt-0.5" />
                    <div className="flex-1 min-w-0">
                      <span className="font-medium text-sm">No GPU Environments Configured</span>
                      <p className="text-xs text-muted-foreground mt-1">
                        Configure GPU Python mappings in Settings to enable training
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* PaddleDetection Check */}
            <div className={`p-3 rounded-lg border ${environmentCheck?.paddleDetection?.isValid ? 'border-emerald-500/30 bg-emerald-50/30' : environmentCheck?.paddleDetection?.exists ? 'border-red-500/30 bg-red-50/30' : 'border-yellow-500/30 bg-yellow-50/30'}`}>
              <div className="flex items-start gap-3">
                <div className="mt-0.5">
                  {environmentCheck?.paddleDetection?.isValid ? (
                    <CheckCircle className="w-5 h-5 text-emerald-500" />
                  ) : environmentCheck?.paddleDetection?.exists ? (
                    <XCircle className="w-5 h-5 text-red-500" />
                  ) : (
                    <AlertTriangle className="w-5 h-5 text-yellow-500" />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-medium text-sm">PaddleDetection</span>
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    {environmentCheck?.paddleDetection?.isValid ? (
                      <span className="text-emerald-600">Path valid</span>
                    ) : environmentCheck?.paddleDetection?.error ? (
                      <span className="text-red-600">{environmentCheck.paddleDetection.error}</span>
                    ) : envCheckLoading ? (
                      <span>Checking...</span>
                    ) : (
                      <span>Not configured</span>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Path Info */}
            <div className="space-y-2 pt-2">
              <div className="text-xs text-muted-foreground">PaddleDetection Path</div>
              <div className="text-xs font-mono truncate text-muted-foreground" title={systemConfig.paddleDetectionPath}>
                {systemConfig.paddleDetectionPath || 'Not configured'}
              </div>
            </div>

            {/* Overall Status */}
            <div className="pt-2 border-t">
              <div className="flex items-center gap-2 text-sm">
                <div className={`w-2 h-2 rounded-full ${
                  environmentCheck?.validGpus && environmentCheck?.validGpus > 0 && environmentCheck?.paddleDetection?.isValid
                    ? 'bg-emerald-500' 
                    : environmentCheck?.configuredGpus && environmentCheck?.configuredGpus > 0 
                      ? 'bg-red-500' 
                      : 'bg-yellow-500'
                }`} />
                <span>
                  {environmentCheck?.validGpus && environmentCheck?.validGpus > 0 && environmentCheck?.paddleDetection?.isValid
                    ? `${environmentCheck.validGpus} GPU${environmentCheck.validGpus > 1 ? 's' : ''} Ready` 
                    : environmentCheck?.configuredGpus && environmentCheck?.configuredGpus > 0 
                      ? `${environmentCheck.configuredGpus - (environmentCheck.validGpus || 0)} GPU${(environmentCheck.configuredGpus - (environmentCheck.validGpus || 0)) > 1 ? 's' : ''} Have Issues` 
                      : 'Configure GPU Environments in Settings'}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
</div>
      {/* Recent Activity */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Activity</CardTitle>
          <CardDescription>Your latest training activities</CardDescription>
        </CardHeader>
        <CardContent>
          {stats.runningJobs === 0 && stats.completedJobs === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <AlertCircle className="w-12 h-12 text-muted-foreground mb-4" />
              <h3 className="text-lg font-medium">No activity yet</h3>
              <p className="text-sm text-muted-foreground mt-1">
                Create a project and start training to see activity here
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {/* Activity items would go here */}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
