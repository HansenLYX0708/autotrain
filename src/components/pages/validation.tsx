'use client'

import { useEffect, useState, useCallback } from 'react'
import { useToast } from '@/hooks/use-toast'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
  Cell,
} from 'recharts'
import {
  CheckCircle2,
  Play,
  FileSearch,
  Loader2,
  Image as ImageIcon,
  BarChart3,
  FolderOpen,
  RefreshCw,
  AlertCircle,
  Trash2,
  ChevronDown,
  ChevronUp,
  Download,
  FileDown,
} from 'lucide-react'

// Types
interface TrainingJob {
  id: string
  name: string
  status: string
  evalCommand: string | null
  inferCommand: string | null
  configPath: string | null
  absoluteConfigPath?: string | null
  outputDir: string | null
  yamlConfig: string | null
  trainingParams?: string | null
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
}

interface GpuPythonMapping {
  gpuId: string
  pythonPath: string
}

interface SystemConfig {
  paddleDetectionPath: string
  gpuPythonMappings?: GpuPythonMapping[] | Record<string, { pythonPath: string }>
}

interface ValidationJob {
  id: string
  name: string
  type: string
  status: string
  command: string | null
  configPath: string | null
  weightsPath: string | null
  trainingJobId: string | null
  inferInputPath: string | null
  inferOutputPath: string | null
  resultJson: string | null
  resultPath: string | null
  outputLog: string | null
  startedAt: string | null
  completedAt: string | null
  createdAt: string
  project: {
    id: string
    name: string
    framework: string
  }
}

interface Checkpoint {
  name: string
  path: string
  relativePath: string
  size: number
  mtime: string
  epoch?: number
  exportedFiles?: string[]
}

// Eval metrics interface
interface EvalMetrics {
  samplesCount?: number | null
  mAP?: number | null
  mAP50?: number | null
  mAP75?: number | null
  mAP_small?: number | null
  mAP_medium?: number | null
  mAP_large?: number | null
  AR_1?: number | null
  AR_10?: number | null
  AR_100?: number | null
  AR_small?: number | null
  AR_medium?: number | null
  AR_large?: number | null
}

// Chart configurations
const apChartConfig = {
  mAP: { label: "mAP@0.5:0.95", color: "var(--chart-1)" },
  mAP50: { label: "mAP@0.5", color: "var(--chart-2)" },
  mAP75: { label: "mAP@0.75", color: "var(--chart-3)" },
} satisfies ChartConfig

const arChartConfig = {
  AR_1: { label: "AR@1", color: "var(--chart-1)" },
  AR_10: { label: "AR@10", color: "var(--chart-2)" },
  AR_100: { label: "AR@100", color: "var(--chart-3)" },
} satisfies ChartConfig

const areaChartConfig = {
  mAP: { label: "mAP", color: "var(--chart-1)" },
  AR: { label: "AR", color: "var(--chart-2)" },
} satisfies ChartConfig

// Helper functions
const getStatusBadge = (status: string) => {
  const variants: Record<string, { variant: 'default' | 'secondary' | 'destructive' | 'outline'; color: string }> = {
    running: { variant: 'default', color: 'bg-emerald-500 hover:bg-emerald-600' },
    completed: { variant: 'secondary', color: '' },
    failed: { variant: 'destructive', color: '' },
    pending: { variant: 'outline', color: '' },
    stopped: { variant: 'outline', color: '' },
  }
  return variants[status] || { variant: 'outline', color: '' }
}

const formatTime = (date: string | null) => {
  if (!date) return 'N/A'
  return new Date(date).toLocaleString()
}

const formatSize = (bytes: number) => {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

// Quote path if it contains spaces
const quotePath = (path: string): string => {
  if (path.includes(' ')) {
    return `"${path}"`
  }
  return path
}

// Check if path is likely a file (has image extension)
const isImageFile = (path: string): boolean => {
  const imageExtensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
  const ext = path.split('.').pop()?.toLowerCase() || ''
  return imageExtensions.includes(`.${ext}`)
}

// Get image URL for displaying
const getImageUrl = (imagePath: string): string => {
  return `/api/images?path=${encodeURIComponent(imagePath)}`
}

// Metrics Display Component
function MetricsDisplay({ result, compact = false }: { result: EvalMetrics; compact?: boolean }) {
  const apChartData = [
    { name: 'mAP@0.5:0.95', value: result.mAP ?? 0, fill: 'var(--chart-1)' },
    { name: 'mAP@0.5', value: result.mAP50 ?? 1, fill: 'var(--chart-2)' },
    { name: 'mAP@0.75', value: result.mAP75 ?? 1, fill: 'var(--chart-3)' },
  ]

  const arChartData = [
    { name: 'AR@1', value: result.AR_1 ?? 1, fill: 'var(--chart-1)' },
    { name: 'AR@10', value: result.AR_10 ?? 1, fill: 'var(--chart-2)' },
    { name: 'AR@100', value: result.AR_100 ?? 1, fill: 'var(--chart-3)' },
  ]

  const areaChartData = [
    { name: 'Small', mAP: (result.mAP_small ?? 0) * 100, AR: (result.AR_small ?? 0) * 100 },
    { name: 'Medium', mAP: (result.mAP_medium ?? 0) * 100, AR: (result.AR_medium ?? 0) * 100 },
    { name: 'Large', mAP: (result.mAP_large ?? 0) * 100, AR: (result.AR_large ?? 0) * 100 },
  ]

  if (compact) {
    return (
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <div className="font-medium mb-1">Average Precision</div>
          <div className="space-y-1">
            <div className="flex justify-between">
              <span className="text-muted-foreground">mAP@0.5:0.95:</span>
              <span className="font-bold text-emerald-600">{((result.mAP ?? 0) * 100).toFixed(2)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">mAP@0.5:</span>
              <span className="font-bold text-blue-600">{((result.mAP50 ?? 0) * 100).toFixed(2)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">mAP@0.75:</span>
              <span className="font-bold text-purple-600">{((result.mAP75 ?? 0) * 100).toFixed(2)}%</span>
            </div>
          </div>
        </div>
        <div>
          <div className="font-medium mb-1">Average Recall</div>
          <div className="space-y-1">
            <div className="flex justify-between">
              <span className="text-muted-foreground">AR@1:</span>
              <span className="font-bold text-emerald-600">{((result.AR_1 ?? 0) * 100).toFixed(2)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">AR@10:</span>
              <span className="font-bold text-blue-600">{((result.AR_10 ?? 0) * 100).toFixed(2)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">AR@100:</span>
              <span className="font-bold text-purple-600">{((result.AR_100 ?? 0) * 100).toFixed(2)}%</span>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Sample count */}
      {result.samplesCount && (
        <div className="p-3 rounded-lg border bg-muted/50">
          <span className="text-muted-foreground">Validation Samples:</span>
          <span className="ml-2 font-bold text-lg">{result.samplesCount}</span>
        </div>
      )}

      {/* mAP Chart */}
      <div>
        <div className="text-sm font-medium mb-2">Average Precision (AP)</div>
        <ChartContainer config={apChartConfig} className="h-[200px] w-full">
          <BarChart data={apChartData}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
            <XAxis dataKey="name" className="text-xs" />
            <YAxis className="text-xs" domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
            <ChartTooltip content={<ChartTooltipContent />} />
            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
              {apChartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ChartContainer>
        <div className="grid grid-cols-3 gap-2 mt-2 text-center text-xs">
          <div>
            <div className="font-bold text-emerald-600">{((result.mAP ?? 0) * 100).toFixed(2)}%</div>
            <div className="text-muted-foreground">mAP@0.5:0.95</div>
          </div>
          <div>
            <div className="font-bold text-blue-600">{((result.mAP50 ?? 0) * 100).toFixed(2)}%</div>
            <div className="text-muted-foreground">mAP@0.5</div>
          </div>
          <div>
            <div className="font-bold text-purple-600">{((result.mAP75 ?? 0) * 100).toFixed(2)}%</div>
            <div className="text-muted-foreground">mAP@0.75</div>
          </div>
        </div>
      </div>

      {/* AR Chart */}
      <div>
        <div className="text-sm font-medium mb-2">Average Recall (AR)</div>
        <ChartContainer config={arChartConfig} className="h-[200px] w-full">
          <BarChart data={arChartData}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
            <XAxis dataKey="name" className="text-xs" />
            <YAxis className="text-xs" domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
            <ChartTooltip content={<ChartTooltipContent />} />
            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
              {arChartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ChartContainer>
        <div className="grid grid-cols-3 gap-2 mt-2 text-center text-xs">
          <div>
            <div className="font-bold text-emerald-600">{((result.AR_1 ?? 0) * 100).toFixed(2)}%</div>
            <div className="text-muted-foreground">AR@1</div>
          </div>
          <div>
            <div className="font-bold text-blue-600">{((result.AR_10 ?? 0) * 100).toFixed(2)}%</div>
            <div className="text-muted-foreground">AR@10</div>
          </div>
          <div>
            <div className="font-bold text-purple-600">{((result.AR_100 ?? 0) * 100).toFixed(2)}%</div>
            <div className="text-muted-foreground">AR@100</div>
          </div>
        </div>
      </div>

      {/* Area-based Chart */}
      <div>
        <div className="text-sm font-medium mb-2">Performance by Object Size</div>
        <ChartContainer config={areaChartConfig} className="h-[200px] w-full">
          <BarChart data={areaChartData}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
            <XAxis dataKey="name" className="text-xs" />
            <YAxis className="text-xs" tickFormatter={(v) => `${v.toFixed(1)}%`} />
            <ChartTooltip content={<ChartTooltipContent />} />
            <Legend />
            <Bar dataKey="mAP" fill="var(--chart-1)" radius={[4, 4, 0, 0]} />
            <Bar dataKey="AR" fill="var(--chart-2)" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ChartContainer>
      </div>
    </div>
  )
}

export function ValidationPage() {
  // Toast hook
  const { toast } = useToast()

  // State
  const [trainingJobs, setTrainingJobs] = useState<TrainingJob[]>([])
  const [validationJobs, setValidationJobs] = useState<ValidationJob[]>([])
  const [selectedJobId, setSelectedJobId] = useState<string>('')
  const [selectedJob, setSelectedJob] = useState<TrainingJob | null>(null)
  const [loading, setLoading] = useState(true)

  // Checkpoints state
  const [saveDir, setSaveDir] = useState<string>('')
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([])
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>('')
  const [checkpointsLoading, setCheckpointsLoading] = useState(false)

  // Eval state
  const [evalRunning, setEvalRunning] = useState(false)
  const [evalLog, setEvalLog] = useState<string>('')
  const [evalResult, setEvalResult] = useState<EvalMetrics | null>(null)

  // Infer state
  const [inferInputPath, setInferInputPath] = useState('')
  const [inferOutputPath, setInferOutputPath] = useState('')
  const [inferRunning, setInferRunning] = useState(false)
  const [inferLog, setInferLog] = useState<string>('')
  const [inferImages, setInferImages] = useState<string[]>([])

  // Result dialog
  const [resultDialogOpen, setResultDialogOpen] = useState(false)
  const [selectedImage, setSelectedImage] = useState<string>('')

  // Delete dialog
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [jobToDelete, setJobToDelete] = useState<ValidationJob | null>(null)
  const [deleting, setDeleting] = useState(false)

  // Export state
  const [exportRunning, setExportRunning] = useState(false)
  const [exportedFiles, setExportedFiles] = useState<string[]>([])
  const [systemConfig, setSystemConfig] = useState<SystemConfig | null>(null)

  // Fetch system config for GPU Python mappings
  const fetchSystemConfig = useCallback(async () => {
    try {
      const response = await fetch('/api/settings')
      if (response.ok) {
        const data = await response.json()
        setSystemConfig(data)
      }
    } catch (error) {
      console.error('Failed to fetch system config:', error)
    }
  }, [])

  // Get Python path for a job based on its GPU configuration
  const getPythonPathForJob = (job: TrainingJob | null): string => {
    if (!job) return 'python'

    const gpuMappings = systemConfig?.gpuPythonMappings
    if (!gpuMappings) return 'python'

    // Parse trainingParams to get GPU IDs
    let gpuIds = '0'
    if (job.trainingParams) {
      try {
        const params = JSON.parse(job.trainingParams)
        gpuIds = params.gpuIds || '0'
      } catch {
        // Use default
      }
    }

    const primaryGpuId = String(gpuIds).split(',')[0].trim()

    // Handle array format
    if (Array.isArray(gpuMappings)) {
      const mapping = gpuMappings.find(m => String(m.gpuId) === primaryGpuId)
      if (mapping?.pythonPath) return mapping.pythonPath
      // Fallback to first mapping
      const fallback = gpuMappings[0]
      if (fallback?.pythonPath) return fallback.pythonPath
      return 'python'
    }

    // Handle object format
    if (typeof gpuMappings === 'object') {
      const mapping = gpuMappings[primaryGpuId] ||
                      gpuMappings[parseInt(primaryGpuId)] ||
                      gpuMappings['0'] ||
                      gpuMappings[0]
      if (mapping?.pythonPath) return mapping.pythonPath
    }

    return 'python'
  }

  // Generate export TRT command with absolute paths
  const generateExportCommand = () => {
    if (!selectedJob?.absoluteConfigPath || !selectedCheckpoint) return null
    const pythonPath = getPythonPathForJob(selectedJob)
    const configPath = selectedJob.absoluteConfigPath
    const checkpointPath = selectedCheckpoint
    return `${pythonPath} tools/export_model.py -c "${configPath}" -o weights="${checkpointPath}" trt=True --output_dir "${saveDir}/export_model"`
  }

  // Expanded jobs in history
  const [expandedJobs, setExpandedJobs] = useState<Set<string>>(new Set())

  // Fetch training jobs
  const fetchTrainingJobs = useCallback(async () => {
    try {
      const response = await fetch('/api/training-jobs?limit=50')
      if (response.ok) {
        const data = await response.json()
        setTrainingJobs(data.data)
      }
    } catch (error) {
      console.error('Failed to fetch training jobs:', error)
    } finally {
      setLoading(false)
    }
  }, [])

  // Fetch validation jobs
  const fetchValidationJobs = useCallback(async () => {
    try {
      const response = await fetch('/api/validation-jobs?limit=50')
      if (response.ok) {
        const data = await response.json()
        setValidationJobs(data.data)
      }
    } catch (error) {
      console.error('Failed to fetch validation jobs:', error)
    }
  }, [])

  // Initial fetch
  useEffect(() => {
    fetchTrainingJobs()
    fetchValidationJobs()
    fetchSystemConfig()
  }, [fetchTrainingJobs, fetchValidationJobs, fetchSystemConfig])

  // Update selected job
  useEffect(() => {
    if (selectedJobId) {
      const job = trainingJobs.find(j => j.id === selectedJobId)
      setSelectedJob(job || null)
      if (job) {
        fetchCheckpoints(job.id)
      }
    } else {
      setSelectedJob(null)
      setCheckpoints([])
      setSelectedCheckpoint('')
      setSaveDir('')
    }
  }, [selectedJobId, trainingJobs])

  // Fetch checkpoints for a training job
  const fetchCheckpoints = async (jobId: string, dir?: string) => {
    setCheckpointsLoading(true)
    try {
      let url = `/api/checkpoints?jobId=${jobId}&checkExported=true`
      if (dir) {
        url = `/api/checkpoints?dir=${encodeURIComponent(dir)}&checkExported=true`
      }
      
      const response = await fetch(url)
      if (response.ok) {
        const data = await response.json()
        setSaveDir(data.saveDir || '')
        setCheckpoints(data.checkpoints || [])
        if (data.checkpoints && data.checkpoints.length > 0) {
          const firstCheckpoint = data.checkpoints[0]
          setSelectedCheckpoint(firstCheckpoint.relativePath)
          // Auto-populate exported files if they exist for the first checkpoint
          if (firstCheckpoint.exportedFiles && firstCheckpoint.exportedFiles.length > 0) {
            setExportedFiles(firstCheckpoint.exportedFiles)
          } else {
            setExportedFiles([])
          }
        } else {
          setSelectedCheckpoint('')
          setExportedFiles([])
        }
      }
    } catch (error) {
      console.error('Failed to fetch checkpoints:', error)
    } finally {
      setCheckpointsLoading(false)
    }
  }

  // Handle checkpoint selection change
  const handleCheckpointChange = (value: string) => {
    setSelectedCheckpoint(value)
    // Find the selected checkpoint and check if it has exported files
    const selectedCp = checkpoints.find(cp => cp.relativePath === value)
    if (selectedCp?.exportedFiles && selectedCp.exportedFiles.length > 0) {
      setExportedFiles(selectedCp.exportedFiles)
    } else {
      setExportedFiles([])
    }
  }
  const generateEvalCommand = () => {
    if (!selectedJob?.configPath || !selectedCheckpoint) return null
    const pythonPath = getPythonPathForJob(selectedJob)
    const configPath = selectedJob.absoluteConfigPath || selectedJob.configPath
    const weightsPath = selectedCheckpoint
    return `${pythonPath} tools/eval.py -c ${configPath} -o weights=${weightsPath}`
  }

  // Generate infer command - paths will be handled by backend
  const generateInferCommand = () => {
    if (!selectedJob?.configPath || !selectedCheckpoint || !inferInputPath) return null
    const pythonPath = getPythonPathForJob(selectedJob)
    const configPath = selectedJob.absoluteConfigPath || selectedJob.configPath
    const weightsPath = selectedCheckpoint
    const inputPath = inferInputPath

    // Use --infer_img for single image files, --infer_dir for directories
    const inputParam = isImageFile(inferInputPath) ? '--infer_img' : '--infer_dir'

    let cmd = `${pythonPath} tools/infer.py -c ${configPath} -o weights=${weightsPath} ${inputParam}=${inputPath}`
    if (inferOutputPath) {
      cmd += ` --output_dir=${inferOutputPath}`
    }
    return cmd
  }

  // Run evaluation
  const runEval = async () => {
    if (!selectedJob || !selectedCheckpoint) return

    setEvalRunning(true)
    setEvalLog('')
    setEvalResult(null)

    try {
      const response = await fetch('/api/validation-jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          projectId: selectedJob.project.id,
          trainingJobId: selectedJob.id,
          name: `Eval: ${selectedJob.name}`,
          type: 'eval',
          configPath: selectedJob.absoluteConfigPath || selectedJob.configPath,
          weightsPath: selectedCheckpoint,
          saveDir: saveDir,
          customCommand: generateEvalCommand(),
          runImmediately: true,
        }),
      })

      if (response.ok) {
        const data = await response.json()
        pollValidationStatus(data.data.id, 'eval')
      } else {
        const error = await response.json()
        setEvalLog(`Error: ${error.error}`)
        setEvalRunning(false)
      }
    } catch (error) {
      console.error('Failed to run eval:', error)
      setEvalRunning(false)
    }
  }

  // Run inference
  const runInfer = async () => {
    if (!selectedJob || !selectedCheckpoint || !inferInputPath) return

    setInferRunning(true)
    setInferLog('')
    setInferImages([])

    try {
      const response = await fetch('/api/validation-jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          projectId: selectedJob.project.id,
          trainingJobId: selectedJob.id,
          name: `Infer: ${selectedJob.name}`,
          type: 'infer',
          configPath: selectedJob.absoluteConfigPath || selectedJob.configPath,
          weightsPath: selectedCheckpoint,
          inferInputPath,
          inferOutputPath: inferOutputPath || undefined,
          customCommand: generateInferCommand(),
          runImmediately: true,
        }),
      })

      if (response.ok) {
        const data = await response.json()
        pollValidationStatus(data.data.id, 'infer')
      } else {
        const error = await response.json()
        setInferLog(`Error: ${error.error}`)
        setInferRunning(false)
      }
    } catch (error) {
      console.error('Failed to run inference:', error)
      setInferRunning(false)
    }
  }

  // Poll validation status
  const pollValidationStatus = async (validationId: string, type: 'eval' | 'infer') => {
    const poll = async () => {
      try {
        const response = await fetch(`/api/validation-jobs/${validationId}`)
        if (response.ok) {
          const data = await response.json()
          const job = data.data

          if (type === 'eval') {
            setEvalLog(job.outputLog || '')
            if (job.status === 'completed') {
              setEvalRunning(false)
              if (job.resultJson) {
                try {
                  setEvalResult(JSON.parse(job.resultJson))
                } catch {
                  // Ignore parse errors
                }
              }
              fetchValidationJobs()
            } else if (job.status === 'failed') {
              setEvalRunning(false)
              fetchValidationJobs()
            }
          } else {
            setInferLog(job.outputLog || '')
            if (job.status === 'completed') {
              setInferRunning(false)
              // Parse inference result images from resultJson
              if (job.resultJson) {
                try {
                  const result = JSON.parse(job.resultJson)
                  if (result.images && Array.isArray(result.images)) {
                    setInferImages(result.images)
                  }
                } catch {
                  // Ignore parse errors
                }
              } else if (job.resultPath) {
                // Fallback to resultPath if resultJson not available
                setInferImages([job.resultPath])
              }
              fetchValidationJobs()
            } else if (job.status === 'failed') {
              setInferRunning(false)
              fetchValidationJobs()
            }
          }

          if (job.status === 'running') {
            setTimeout(poll, 2000)
          }
        }
      } catch (error) {
        console.error('Failed to poll status:', error)
        if (type === 'eval') {
          setEvalRunning(false)
        } else {
          setInferRunning(false)
        }
      }
    }

    poll()
  }

  // Delete validation job
  const deleteValidationJob = async () => {
    if (!jobToDelete) return
    
    setDeleting(true)
    try {
      const response = await fetch(`/api/validation-jobs/${jobToDelete.id}`, {
        method: 'DELETE',
      })
      
      if (response.ok) {
        setValidationJobs(prev => prev.filter(j => j.id !== jobToDelete.id))
        setDeleteDialogOpen(false)
        setJobToDelete(null)
      } else {
        const error = await response.json()
        console.error('Failed to delete:', error)
      }
    } catch (error) {
      console.error('Failed to delete validation job:', error)
    } finally {
      setDeleting(false)
    }
  }

  // Toggle job expansion
  const toggleJobExpansion = (jobId: string) => {
    setExpandedJobs(prev => {
      const next = new Set(prev)
      if (next.has(jobId)) {
        next.delete(jobId)
      } else {
        next.add(jobId)
      }
      return next
    })
  }

  // Export checkpoint to TensorRT
  const exportCheckpoint = async () => {
    if (!selectedJob || !selectedCheckpoint) return

    setExportRunning(true)
    setExportedFiles([])

    try {
      const response = await fetch('/api/checkpoints/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jobId: selectedJob.id,
          checkpointPath: selectedCheckpoint,
          checkpointName: checkpoints.find(cp => cp.relativePath === selectedCheckpoint)?.name || 'model',
          configPath: selectedJob.absoluteConfigPath,
          outputDir: saveDir,
        }),
      })

      if (response.ok) {
        const data = await response.json()
        setExportedFiles(data.exportedFiles || [])
        toast({
          title: 'Export completed',
          description: `Model exported to ${data.outputDir}`,
        })
      } else {
        const error = await response.json()
        toast({
          title: 'Export failed',
          description: error.error || 'Unknown error',
          variant: 'destructive',
        })
      }
    } catch (error) {
      console.error('Failed to export checkpoint:', error)
      toast({
        title: 'Export failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      })
    } finally {
      setExportRunning(false)
    }
  }

  // Download exported model folder as zip
  const downloadExportedFile = async (folderPath: string) => {
    try {
      const response = await fetch(`/api/checkpoints/export?path=${encodeURIComponent(folderPath)}&folder=true`)
      
      if (!response.ok) {
        const error = await response.json()
        toast({
          title: 'Download failed',
          description: error.error || 'Failed to download model',
          variant: 'destructive',
        })
        return
      }
      
      // Get filename from Content-Disposition header or use default
      const contentDisposition = response.headers.get('Content-Disposition')
      let filename = 'model.zip'
      if (contentDisposition) {
        const match = contentDisposition.match(/filename="([^"]+)"/)
        if (match) {
          filename = match[1]
        }
      }
      
      // Create blob from response
      const blob = await response.blob()
      
      // Create download link
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      
      // Clean up
      window.URL.revokeObjectURL(url)
      
      toast({
        title: 'Download started',
        description: `Downloading ${filename}`,
      })
    } catch (error) {
      console.error('Download error:', error)
      toast({
        title: 'Download failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      })
    }
  }

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
          <h1 className="text-3xl font-bold tracking-tight">Validation</h1>
          <p className="text-muted-foreground">
            Evaluate trained models and run inference on images
          </p>
        </div>
        <Button variant="outline" onClick={() => { fetchTrainingJobs(); fetchValidationJobs(); }}>
          <RefreshCw className="w-4 h-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Job Selection */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">Select Training Job</CardTitle>
          <CardDescription>
            Choose a training job to evaluate or run inference
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <div>
              <Label>Training Job</Label>
              <Select value={selectedJobId} onValueChange={setSelectedJobId}>
                <SelectTrigger className="mt-1.5">
                  <SelectValue placeholder="Select a training job" />
                </SelectTrigger>
                <SelectContent>
                  {trainingJobs.length === 0 ? (
                    <SelectItem value="none" disabled>No training jobs available</SelectItem>
                  ) : (
                    trainingJobs.map(job => (
                      <SelectItem key={job.id} value={job.id}>
                        <div className="flex items-center gap-2">
                          <span>{job.name}</span>
                          <Badge variant={getStatusBadge(job.status).variant} className={getStatusBadge(job.status).color}>
                            {job.status}
                          </Badge>
                        </div>
                      </SelectItem>
                    ))
                  )}
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label>Config Path</Label>
              <Input
                className="mt-1.5 bg-muted"
                value={selectedJob?.configPath || ''}
                readOnly
                placeholder="No config"
              />
            </div>

            <div className="lg:col-span-2">
              <Label>Save Directory (from YAML)</Label>
              <div className="flex gap-2 mt-1.5">
                <Input
                  value={saveDir}
                  onChange={(e) => setSaveDir(e.target.value)}
                  placeholder="Auto-detected from yaml or enter manually"
                  className="flex-1"
                />
                <Button variant="outline" size="icon" onClick={() => fetchCheckpoints(selectedJobId, saveDir)}>
                  <RefreshCw className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </div>

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
                <span className="text-muted-foreground">Status:</span>
                <Badge variant={getStatusBadge(selectedJob.status).variant} className={getStatusBadge(selectedJob.status).color + ' ml-2'}>
                  {selectedJob.status}
                </Badge>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Checkpoints Selection */}
      {selectedJob && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <FolderOpen className="w-5 h-5" />
              Model Checkpoints
            </CardTitle>
            <CardDescription>
              Select a model checkpoint for evaluation or inference
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-3">
              <div className="md:col-span-2">
                <Label>Select Checkpoint</Label>
                <Select 
                  value={selectedCheckpoint} 
                  onValueChange={handleCheckpointChange}
                  disabled={checkpointsLoading || checkpoints.length === 0}
                >
                  <SelectTrigger className="mt-1.5">
                    <SelectValue placeholder={checkpointsLoading ? "Loading..." : checkpoints.length === 0 ? "No checkpoints found" : "Select checkpoint"} />
                  </SelectTrigger>
                  <SelectContent>
                    {checkpoints.map(cp => (
                      <SelectItem key={cp.path} value={cp.relativePath}>
                        <div className="flex items-center gap-2">
                          <span>{cp.name}</span>
                          {cp.epoch !== undefined && (
                            <Badge variant="outline" className="text-xs">Epoch {cp.epoch}</Badge>
                          )}
                          <span className="text-muted-foreground text-xs">
                            {formatSize(cp.size)}
                          </span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Checkpoint Path</Label>
                <Input
                  className="mt-1.5 bg-muted text-xs"
                  value={selectedCheckpoint}
                  readOnly
                  placeholder="No checkpoint selected"
                />
              </div>
            </div>

            {checkpoints.length === 0 && !checkpointsLoading && (
              <div className="mt-4 p-4 rounded-lg border bg-muted/50 text-sm">
                <div className="flex items-center gap-2 text-muted-foreground">
                  <AlertCircle className="w-4 h-4" />
                  <span>No checkpoints found in &quot;{saveDir}&quot;. Make sure training has produced model files.</span>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Eval and Infer Tabs */}
      <Tabs defaultValue="eval" className="space-y-4">
        <TabsList>
          <TabsTrigger value="eval" className="flex items-center gap-2">
            <BarChart3 className="w-4 h-4" />
            Evaluation
          </TabsTrigger>
          <TabsTrigger value="infer" className="flex items-center gap-2">
            <ImageIcon className="w-4 h-4" />
            Inference
          </TabsTrigger>
          <TabsTrigger value="history" className="flex items-center gap-2">
            <FileSearch className="w-4 h-4" />
            History
          </TabsTrigger>
        </TabsList>

        {/* Evaluation Tab */}
        <TabsContent value="eval">
          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5" />
                  Run Evaluation
                </CardTitle>
                <CardDescription>
                  Evaluate model performance on the validation dataset
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="p-4 rounded-lg border bg-muted/50">
                  <div className="text-sm font-medium mb-2">Generated Command</div>
                  <code className="text-xs break-all block">
                    {generateEvalCommand() || 'Select a training job and checkpoint first'}
                  </code>
                </div>

                <Button
                  className="w-full"
                  onClick={runEval}
                  disabled={!selectedJob || !selectedCheckpoint || evalRunning}
                >
                  {evalRunning ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Running Evaluation...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 mr-2" />
                      Run Evaluation
                    </>
                  )}
                </Button>

                <div className="p-4 rounded-lg border bg-muted/50">
                  <div className="text-sm font-medium mb-2">Export TRT Command</div>
                  <code className="text-xs break-all block">
                    {generateExportCommand() || 'Select a training job and checkpoint first'}
                  </code>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <Button
                    variant="outline"
                    onClick={exportCheckpoint}
                    disabled={!selectedJob || !selectedCheckpoint || exportRunning}
                  >
                    {exportRunning ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Exporting...
                      </>
                    ) : (
                      <>
                        <Download className="w-4 h-4 mr-2" />
                        Export
                      </>
                    )}
                  </Button>
                  <Button
                    variant="secondary"
                    onClick={() => {
                      if (exportedFiles.length > 0) {
                        downloadExportedFile(exportedFiles[0])
                      }
                    }}
                    disabled={exportedFiles.length === 0}
                  >
                    <FileDown className="w-4 h-4 mr-2" />
                    Download
                  </Button>
                </div>
                {exportedFiles.length === 0 && selectedCheckpoint && (
                  <div className="p-3 rounded-lg border bg-amber-50 border-amber-200">
                    <div className="flex items-center gap-2 text-sm text-amber-700">
                      <AlertCircle className="w-4 h-4 flex-shrink-0" />
                      <span>No exported TRT model found. Click "Export" to generate it first.</span>
                    </div>
                  </div>
                )}
                {exportedFiles.length > 0 && (
                  <div className="p-2 rounded text-xs text-muted-foreground">
                    <span>File will download to your browser&apos;s default download folder. </span>
                    <span className="text-blue-600 cursor-pointer hover:underline" onClick={() => {
                      // Open browser download settings
                      window.open('chrome://settings/downloads', '_blank')
                    }}>
                      Change in browser settings
                    </span>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Evaluation Results</CardTitle>
                <CardDescription>
                  mAP and AR metrics from the evaluation
                </CardDescription>
              </CardHeader>
              <CardContent>
                {evalResult ? (
                  <MetricsDisplay result={evalResult} />
                ) : (
                  <div className="flex flex-col items-center justify-center py-8 text-center">
                    <BarChart3 className="w-12 h-12 text-muted-foreground mb-4" />
                    <p className="text-muted-foreground">
                      Run evaluation to see results
                    </p>
                  </div>
                )}

                {evalLog && (
                  <div className="mt-4">
                    <div className="text-sm font-medium mb-2">Output Log</div>
                    <ScrollArea className="h-[200px] w-full rounded border bg-muted/30 p-2">
                      <pre className="text-xs whitespace-pre-wrap font-mono">{evalLog}</pre>
                    </ScrollArea>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Inference Tab */}
        <TabsContent value="infer">
          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <ImageIcon className="w-5 h-5" />
                  Run Inference
                </CardTitle>
                <CardDescription>
                  Run inference on images or video
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label>Input Path (Image or Folder)</Label>
                  <div className="flex gap-2 mt-1.5">
                    <Input
                      placeholder="/path/to/image.jpg or /path/to/folder"
                      value={inferInputPath}
                      onChange={(e) => setInferInputPath(e.target.value)}
                    />
                    <Button variant="outline" size="icon" type="button">
                      <FolderOpen className="w-4 h-4" />
                    </Button>
                  </div>
                  {inferInputPath && (
                    <div className="mt-2 flex items-center gap-2">
                      <Badge variant={isImageFile(inferInputPath) ? 'default' : 'secondary'}>
                        {isImageFile(inferInputPath) ? 'Single Image (--infer_img)' : 'Directory (--infer_dir)'}
                      </Badge>
                      <span className="text-xs text-muted-foreground">
                        {isImageFile(inferInputPath) 
                          ? 'Will process single image file'
                          : 'Will process all images in directory'}
                      </span>
                    </div>
                  )}
                </div>

                <div>
                  <Label>Output Directory (Optional)</Label>
                  <div className="flex gap-2 mt-1.5">
                    <Input
                      placeholder="output/infer_results"
                      value={inferOutputPath}
                      onChange={(e) => setInferOutputPath(e.target.value)}
                    />
                    <Button variant="outline" size="icon" type="button">
                      <FolderOpen className="w-4 h-4" />
                    </Button>
                  </div>
                </div>

                <div className="p-4 rounded-lg border bg-muted/50">
                  <div className="text-sm font-medium mb-2">Generated Command</div>
                  <code className="text-xs break-all block">
                    {generateInferCommand() || 'Select a training job, checkpoint, and input path first'}
                  </code>
                </div>

                <Button
                  className="w-full"
                  onClick={runInfer}
                  disabled={!selectedJob || !selectedCheckpoint || !inferInputPath || inferRunning}
                >
                  {inferRunning ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Running Inference...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 mr-2" />
                      Run Inference
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Inference Results</CardTitle>
                <CardDescription>
                  Detection results with bounding boxes
                </CardDescription>
              </CardHeader>
              <CardContent>
                {inferImages.length > 0 ? (
                  <div className="space-y-4">
                    <div className="text-sm text-muted-foreground">
                      {inferImages.length} result image{inferImages.length > 1 ? 's' : ''} found
                    </div>
                    <div className="grid gap-3 grid-cols-2 sm:grid-cols-3">
                      {inferImages.slice(0, 12).map((img, idx) => {
                        const imageName = img.split(/[/\\]/).pop() || `Image ${idx + 1}`
                        return (
                          <div
                            key={idx}
                            className="group relative aspect-square rounded-lg border overflow-hidden cursor-pointer hover:ring-2 hover:ring-primary transition-all"
                            onClick={() => {
                              setSelectedImage(img)
                              setResultDialogOpen(true)
                            }}
                          >
                            <img
                              src={getImageUrl(img)}
                              alt={imageName}
                              className="w-full h-full object-cover"
                              onError={(e) => {
                                const target = e.target as HTMLImageElement
                                target.style.display = 'none'
                                target.parentElement!.innerHTML = `
                                  <div class="w-full h-full flex flex-col items-center justify-center bg-muted/50">
                                    <svg class="w-8 h-8 text-muted-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
                                    </svg>
                                    <span class="text-xs text-muted-foreground mt-1">${imageName}</span>
                                  </div>
                                `
                              }}
                            />
                            <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                              <span className="text-white text-xs px-2 py-1 bg-black/50 rounded truncate max-w-full">
                                {imageName}
                              </span>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                    {inferImages.length > 12 && (
                      <div className="text-sm text-muted-foreground text-center">
                        Showing 12 of {inferImages.length} images. Click to preview.
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-8 text-center">
                    <ImageIcon className="w-12 h-12 text-muted-foreground mb-4" />
                    <p className="text-muted-foreground">
                      Run inference to see results
                    </p>
                  </div>
                )}

                {inferLog && (
                  <div className="mt-4">
                    <div className="text-sm font-medium mb-2">Output Log</div>
                    <ScrollArea className="h-[200px] w-full rounded border bg-muted/30 p-2">
                      <pre className="text-xs whitespace-pre-wrap font-mono">{inferLog}</pre>
                    </ScrollArea>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* History Tab */}
        <TabsContent value="history">
          <Card>
            <CardHeader>
              <CardTitle>Validation History</CardTitle>
              <CardDescription>
                {selectedJob 
                  ? `Validation jobs for: ${selectedJob.name}`
                  : 'Select a training job above to view its validation history'}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {!selectedJob ? (
                <div className="flex flex-col items-center justify-center py-8 text-center">
                  <AlertCircle className="w-12 h-12 text-muted-foreground mb-4" />
                  <h3 className="text-lg font-medium">No training job selected</h3>
                  <p className="text-sm text-muted-foreground mt-1">
                    Select a training job above to view its validation history
                  </p>
                </div>
              ) : validationJobs.filter(job => job.trainingJobId === selectedJobId).length === 0 ? (
                <div className="flex flex-col items-center justify-center py-8 text-center">
                  <AlertCircle className="w-12 h-12 text-muted-foreground mb-4" />
                  <h3 className="text-lg font-medium">No validation jobs yet</h3>
                  <p className="text-sm text-muted-foreground mt-1">
                    Run your first validation for this job to see results here
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  {validationJobs
                    .filter(job => job.trainingJobId === selectedJobId)
                    .map((job) => {
                    const isExpanded = expandedJobs.has(job.id)
                    let result: EvalMetrics | null = null
                    if (job.resultJson && job.type === 'eval') {
                      try {
                        result = JSON.parse(job.resultJson)
                      } catch {
                        // Ignore parse errors
                      }
                    }

                    return (
                      <div key={job.id} className="rounded-lg border">
                        {/* Header row */}
                        <div 
                          className="flex items-center justify-between p-4 cursor-pointer hover:bg-muted/50 transition-colors"
                          onClick={() => toggleJobExpansion(job.id)}
                        >
                          <div className="flex items-center gap-4">
                            <Button variant="ghost" size="icon" className="h-6 w-6">
                              {isExpanded ? (
                                <ChevronUp className="h-4 w-4" />
                              ) : (
                                <ChevronDown className="h-4 w-4" />
                              )}
                            </Button>
                            <div>
                              <div className="font-medium">{job.name}</div>
                              <div className="text-sm text-muted-foreground">
                                {job.type === 'eval' ? 'Evaluation' : 'Inference'} • {formatTime(job.createdAt)}
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center gap-4">
                            {result && (
                              <div className="text-sm hidden sm:block">
                                <span className="font-bold text-emerald-600">{((result.mAP ?? 0) * 100).toFixed(2)}%</span>
                                <span className="text-muted-foreground ml-1">mAP</span>
                              </div>
                            )}
                            <Badge variant={getStatusBadge(job.status).variant} className={getStatusBadge(job.status).color}>
                              {job.status}
                            </Badge>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-8 w-8 text-destructive hover:text-destructive hover:bg-destructive/10"
                              onClick={(e) => {
                                e.stopPropagation()
                                setJobToDelete(job)
                                setDeleteDialogOpen(true)
                              }}
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </div>
                        </div>

                        {/* Expanded content */}
                        {isExpanded && (
                          <div className="border-t px-4 py-4">
                            {/* Job details */}
                            <div className="grid gap-2 text-sm mb-4">
                              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                <div>
                                  <span className="text-muted-foreground">Weights:</span>
                                  <span className="ml-2 font-mono text-xs">{job.weightsPath || 'N/A'}</span>
                                </div>
                                <div>
                                  <span className="text-muted-foreground">Started:</span>
                                  <span className="ml-2">{formatTime(job.startedAt)}</span>
                                </div>
                                <div>
                                  <span className="text-muted-foreground">Completed:</span>
                                  <span className="ml-2">{formatTime(job.completedAt)}</span>
                                </div>
                                <div>
                                  <span className="text-muted-foreground">Duration:</span>
                                  <span className="ml-2">
                                    {job.startedAt && job.completedAt
                                      ? (() => {
                                          const diff = new Date(job.completedAt).getTime() - new Date(job.startedAt).getTime()
                                          const mins = Math.floor(diff / 60000)
                                          const secs = Math.floor((diff % 60000) / 1000)
                                          return `${mins}m ${secs}s`
                                        })()
                                      : 'N/A'}
                                  </span>
                                </div>
                              </div>
                            </div>

                            {/* Metrics for eval jobs */}
                            {result && job.type === 'eval' && (
                              <div className="mb-4">
                                <MetricsDisplay result={result} compact />
                              </div>
                            )}

                            {/* Inference images for infer jobs */}
                            {job.type === 'infer' && job.resultJson && (() => {
                              try {
                                const inferResult = JSON.parse(job.resultJson)
                                if (inferResult.images && Array.isArray(inferResult.images) && inferResult.images.length > 0) {
                                  return (
                                    <div className="mb-4">
                                      <div className="text-sm font-medium mb-2">
                                        Inference Results ({inferResult.images.length} images)
                                      </div>
                                      <div className="grid gap-2 grid-cols-3 sm:grid-cols-4 md:grid-cols-6">
                                        {inferResult.images.slice(0, 12).map((img: string, idx: number) => {
                                          const imageName = img.split(/[/\\]/).pop() || `Image ${idx + 1}`
                                          return (
                                            <div
                                              key={idx}
                                              className="group relative aspect-square rounded border overflow-hidden cursor-pointer hover:ring-2 hover:ring-primary transition-all"
                                              onClick={() => {
                                                setSelectedImage(img)
                                                setResultDialogOpen(true)
                                              }}
                                            >
                                              <img
                                                src={getImageUrl(img)}
                                                alt={imageName}
                                                className="w-full h-full object-cover"
                                                onError={(e) => {
                                                  const target = e.target as HTMLImageElement
                                                  target.style.display = 'none'
                                                }}
                                              />
                                              <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                                                <ImageIcon className="w-4 h-4 text-white" />
                                              </div>
                                            </div>
                                          )
                                        })}
                                      </div>
                                      {inferResult.images.length > 12 && (
                                        <div className="text-xs text-muted-foreground mt-2">
                                          Showing 12 of {inferResult.images.length} images. Click to preview.
                                        </div>
                                      )}
                                    </div>
                                  )
                                }
                              } catch {
                                // Ignore parse errors
                              }
                              return null
                            })()}

                            {/* Output log */}
                            {job.outputLog && (
                              <div>
                                <div className="text-sm font-medium mb-2">Output Log</div>
                                <ScrollArea className="h-[200px] w-full rounded border bg-muted/30 p-2">
                                  <pre className="text-xs whitespace-pre-wrap font-mono">{job.outputLog}</pre>
                                </ScrollArea>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Image Preview Dialog */}
      <Dialog open={resultDialogOpen} onOpenChange={setResultDialogOpen}>
        <DialogContent className="max-w-4xl max-h-[90vh]">
          <DialogHeader>
            <DialogTitle>Detection Result</DialogTitle>
            <DialogDescription>
              {selectedImage ? (
                <span className="truncate block max-w-full">
                  {selectedImage.split(/[/\\]/).pop() || selectedImage}
                </span>
              ) : (
                'Image with detected objects and bounding boxes'
              )}
            </DialogDescription>
          </DialogHeader>
          <div className="relative rounded-lg border overflow-hidden bg-muted/30">
            {selectedImage ? (
              <img
                src={getImageUrl(selectedImage)}
                alt="Detection result"
                className="w-full h-auto max-h-[70vh] object-contain"
                onError={(e) => {
                  const target = e.target as HTMLImageElement
                  target.style.display = 'none'
                  target.parentElement!.innerHTML = `
                    <div class="flex flex-col items-center justify-center py-16">
                      <svg class="w-16 h-16 text-muted-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
                      </svg>
                      <p class="text-muted-foreground mt-2">Unable to load image</p>
                      <p class="text-xs text-muted-foreground mt-1">${selectedImage}</p>
                    </div>
                  `
                }}
              />
            ) : (
              <div className="flex flex-col items-center justify-center py-16">
                <ImageIcon className="w-16 h-16 text-muted-foreground" />
                <p className="text-muted-foreground mt-2">No image selected</p>
              </div>
            )}
          </div>
          {selectedImage && (
            <div className="text-xs text-muted-foreground break-all">
              Path: {selectedImage}
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Validation Job</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete &quot;{jobToDelete?.name}&quot;? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setDeleteDialogOpen(false)}
              disabled={deleting}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={deleteValidationJob}
              disabled={deleting}
            >
              {deleting ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Deleting...
                </>
              ) : (
                <>
                  <Trash2 className="w-4 h-4 mr-2" />
                  Delete
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
