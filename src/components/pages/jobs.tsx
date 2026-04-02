'use client'

import { useEffect, useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import {
  Play,
  AlertCircle,
  Trash2,
  MoreHorizontal,
  Pause,
  RotateCcw,
  Copy,
  List,
  ChevronDown,
  ChevronRight,
  Terminal,
  Settings2,
  Plus,
  Loader2,
  Cpu,
  Filter,
  X,
} from 'lucide-react'
import { toast } from '@/hooks/use-toast'

interface Project {
  id: string
  name: string
  framework: string
}

interface Dataset {
  id: string
  name: string
  projectId: string
  numClasses: number
}

interface Model {
  id: string
  name: string
  projectId: string
  architecture: string
  yamlConfig: string | null
}

interface TrainingConfig {
  id: string
  name: string
  epoch: number
  batchSize: number
  baseLr: number
}

interface GPUInfo {
  id: number
  name: string
  utilization: number
  memoryUsed: number
  memoryTotal: number
  temperature: number
  status?: 'idle' | 'occupied'
}

interface TrainingJob {
  id: string
  name: string
  status: string
  command: string | null
  configPath: string | null
  errorMessage: string | null
  currentEpoch: number
  totalEpochs: number
  currentLoss: number | null
  currentLr: number | null
  outputDir: string | null
  weightsPath: string | null
  startedAt: string | null
  completedAt: string | null
  createdAt: string
  project: { id: string; name: string; framework: string }
  dataset: { id: string; name: string; format: string }
  model: { id: string; name: string; architecture: string }
  config: { id: string; name: string } | null
  trainingParams?: Record<string, unknown>
  yamlConfig?: string | null
  _count?: { logs: number }
}

const defaultFormData = {
  name: '',
  projectId: '',
  datasetId: '',
  modelId: '',
  configId: '__none__',
  gpuIds: '0',
  useAmp: true,
  useVdl: false,
}

const defaultFilterData = {
  projectId: '__all__',
  datasetId: '__all__',
  modelId: '__all__',
  configId: '__all__',
}

export function JobsPage() {
  const [jobs, setJobs] = useState<TrainingJob[]>([])
  const [projects, setProjects] = useState<Project[]>([])
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [models, setModels] = useState<Model[]>([])
  const [configs, setConfigs] = useState<TrainingConfig[]>([])
  const [loading, setLoading] = useState(true)
  const [expandedJobs, setExpandedJobs] = useState<Set<string>>(new Set())
  const [dialogOpen, setDialogOpen] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [formData, setFormData] = useState(defaultFormData)
  const [filterData, setFilterData] = useState(defaultFilterData)
  const [allDatasets, setAllDatasets] = useState<Dataset[]>([])
  const [allModels, setAllModels] = useState<Model[]>([])
  const [totalJobs, setTotalJobs] = useState(0)
  const [showAll, setShowAll] = useState(false)
  const [gpus, setGpus] = useState<GPUInfo[]>([])

  useEffect(() => {
    fetchJobs()
    fetchProjects()
    fetchAllDatasets()
    fetchAllModels()
    fetchGPUs()
  }, [])

  useEffect(() => {
    if (formData.projectId) {
      fetchDatasets(formData.projectId)
      fetchModels(formData.projectId)
    } else {
      setDatasets([])
      setModels([])
    }
  }, [formData.projectId])

  useEffect(() => {
    fetchConfigs()
  }, [])

  // Re-fetch jobs when filter changes
  useEffect(() => {
    fetchJobs()
  }, [filterData, showAll])

  const fetchJobs = async () => {
    try {
      setLoading(true)
      const params = new URLSearchParams()
      
      // If showAll is false, limit to 5 records
      if (!showAll) {
        params.append('limit', '5')
      } else {
        params.append('limit', '100')
      }
      
      // Add filter parameters (only if not '__all__')
      if (filterData.projectId && filterData.projectId !== '__all__') {
        params.append('projectId', filterData.projectId)
      }
      if (filterData.datasetId && filterData.datasetId !== '__all__') {
        params.append('datasetId', filterData.datasetId)
      }
      if (filterData.modelId && filterData.modelId !== '__all__') {
        params.append('modelId', filterData.modelId)
      }
      if (filterData.configId && filterData.configId !== '__all__') {
        params.append('configId', filterData.configId)
      }
      
      const response = await fetch(`/api/training-jobs?${params.toString()}`)
      if (response.ok) {
        const data = await response.json()
        setJobs(data.data || [])
        setTotalJobs(data.pagination?.total || 0)
      }
    } catch (error) {
      console.error('Failed to fetch jobs:', error)
    } finally {
      setLoading(false)
    }
  }

  const fetchProjects = async () => {
    try {
      const response = await fetch('/api/projects')
      if (response.ok) {
        const data = await response.json()
        setProjects(data.data || data)
      }
    } catch (error) {
      console.error('Failed to fetch projects:', error)
    }
  }

  const fetchAllDatasets = async () => {
    try {
      const response = await fetch('/api/datasets')
      if (response.ok) {
        const result = await response.json()
        setAllDatasets(Array.isArray(result.data) ? result.data : (Array.isArray(result) ? result : []))
      }
    } catch (error) {
      console.error('Failed to fetch all datasets:', error)
      setAllDatasets([])
    }
  }

  const fetchAllModels = async () => {
    try {
      const response = await fetch('/api/models')
      if (response.ok) {
        const result = await response.json()
        setAllModels(Array.isArray(result.data) ? result.data : (Array.isArray(result) ? result : []))
      }
    } catch (error) {
      console.error('Failed to fetch all models:', error)
      setAllModels([])
    }
  }

  const fetchDatasets = async (projectId: string) => {
    try {
      const response = await fetch(`/api/datasets?projectId=${projectId}`)
      if (response.ok) {
        const result = await response.json()
        setDatasets(Array.isArray(result.data) ? result.data : (Array.isArray(result) ? result : []))
      }
    } catch (error) {
      console.error('Failed to fetch datasets:', error)
      setDatasets([])
    }
  }

  const fetchModels = async (projectId: string) => {
    try {
      const response = await fetch(`/api/models?projectId=${projectId}`)
      if (response.ok) {
        const result = await response.json()
        setModels(Array.isArray(result.data) ? result.data : (Array.isArray(result) ? result : []))
      }
    } catch (error) {
      console.error('Failed to fetch models:', error)
      setModels([])
    }
  }

  const fetchGPUs = async () => {
    try {
      const response = await fetch('/api/system/gpu')
      if (response.ok) {
        const data = await response.json()
        if (data.success && data.data && Array.isArray(data.data)) {
          // Determine GPU status based on utilization/memory usage
          const gpusWithStatus = data.data.map((gpu: GPUInfo) => {
            const memoryUsagePercent = (gpu.memoryUsed / gpu.memoryTotal) * 100
            const isOccupied = memoryUsagePercent >= 50 || gpu.utilization >= 30
            return {
              ...gpu,
              status: isOccupied ? 'occupied' : 'idle' as 'idle' | 'occupied'
            }
          })
          setGpus(gpusWithStatus)
        }
      }
    } catch (error) {
      console.error('Failed to fetch GPUs:', error)
    }
  }

  const fetchConfigs = async () => {
    try {
      const response = await fetch('/api/training-configs')
      if (response.ok) {
        const data = await response.json()
        setConfigs(data.data || data)
      }
    } catch (error) {
      console.error('Failed to fetch configs:', error)
    }
  }

  const generateCommand = () => {
    if (!formData.name || !formData.projectId) return ''
    
    const jobName = formData.name.replace(/\s+/g, '_').replace(/[^a-zA-Z0-9_.-]/g, '').toLowerCase()
    const configPath = `configs/autotrain/jobs/${jobName}.yml`
    
    const parts: string[] = []
    
    // GPU selection - always use paddle.distributed.launch
    const gpuIds = formData.gpuIds || '0'
    parts.push(`python -m paddle.distributed.launch --gpus ${gpuIds} tools/train.py`)

    // Config file - use merged job config
    parts.push(`-c ${configPath}`)

    // AMP
    if (formData.useAmp) {
      parts.push('--amp')
    }

    // VDL
    if (formData.useVdl) {
      parts.push('--use_vdl=true')
      const project = projects.find(p => p.id === formData.projectId)
      parts.push(`--vdl_log_dir=output/${project?.name || 'default'}/${jobName}/vdl`)
    }

    return parts.join(' ')
  }
  
  const generateCommandDisplay = () => {
    const cmd = generateCommand()
    if (!cmd) return ''
    
    const gpuIds = formData.gpuIds || '0'
    const isMultiGpu = gpuIds.includes(',')
    const isWindows = typeof window !== 'undefined' && window.navigator.platform.toLowerCase().includes('win')
    
    let display = cmd
    
    if (isMultiGpu && isWindows) {
      display = `# WARNING: Multi-GPU training is only supported on Linux.\n# On Windows, please use single GPU or WSL2.\n${cmd}`
    }
    
    return display
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!formData.projectId || !formData.datasetId || !formData.modelId || !formData.name) {
      toast({
        title: 'Missing required fields',
        description: 'Please select project, dataset, model and enter job name',
        variant: 'destructive',
      })
      return
    }

    setSubmitting(true)
    try {
      const response = await fetch('/api/training-jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          projectId: formData.projectId,
          datasetId: formData.datasetId,
          modelId: formData.modelId,
          configId: formData.configId === '__none__' ? null : formData.configId,
          name: formData.name,
          gpuIds: formData.gpuIds,
          useAmp: formData.useAmp,
          useVdl: formData.useVdl,
        }),
      })

      if (response.ok) {
        const result = await response.json()
        toast({ 
          title: 'Job created successfully',
          description: `Config saved to ${result.configPath}`,
        })
        setDialogOpen(false)
        resetForm()
        fetchJobs()
      } else {
        const error = await response.json()
        throw new Error(error.error || 'Failed to create job')
      }
    } catch (error) {
      toast({
        title: 'Error creating job',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      })
    } finally {
      setSubmitting(false)
    }
  }

  const handleDeleteJob = async (jobId: string) => {
    if (!confirm('Are you sure you want to delete this training job?')) {
      return
    }
    try {
      const response = await fetch(`/api/training-jobs/${jobId}`, {
        method: 'DELETE',
      })
      if (response.ok) {
        toast({ title: 'Job deleted successfully' })
        fetchJobs()
      } else {
        throw new Error('Failed to delete job')
      }
    } catch (error) {
      toast({ title: 'Error deleting job', variant: 'destructive' })
    }
  }

  const handleUpdateJobStatus = async (jobId: string, status: string) => {
    try {
      const response = await fetch(`/api/training-jobs/${jobId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status }),
      })
      if (response.ok) {
        const result = await response.json()
        // Show the actual status from server response
        const actualStatus = result.status || status
        toast({ 
          title: `Job ${actualStatus}`,
          description: actualStatus === 'failed' && result.errorMessage 
            ? result.errorMessage.slice(0, 100) 
            : undefined,
          variant: actualStatus === 'failed' ? 'destructive' : 'default'
        })
        fetchJobs()
      } else {
        const error = await response.json()
        throw new Error(error.error || 'Failed to update job')
      }
    } catch (error) {
      toast({ 
        title: 'Error updating job', 
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive' 
      })
    }
  }

  const toggleJobExpand = (jobId: string) => {
    setExpandedJobs(prev => {
      const newSet = new Set(prev)
      if (newSet.has(jobId)) {
        newSet.delete(jobId)
      } else {
        newSet.add(jobId)
      }
      return newSet
    })
  }

  const resetForm = () => {
    setFormData(defaultFormData)
  }

  const getStatusBadge = (status: string) => {
    const variants: Record<string, { variant: 'default' | 'secondary' | 'destructive' | 'outline'; className: string }> = {
      pending: { variant: 'secondary', className: '' },
      running: { variant: 'default', className: 'bg-blue-500' },
      completed: { variant: 'default', className: 'bg-emerald-500' },
      failed: { variant: 'destructive', className: '' },
      stopped: { variant: 'outline', className: '' },
    }
    const config = variants[status] || { variant: 'secondary', className: '' }
    return <Badge variant={config.variant} className={config.className}>{status}</Badge>
  }

  const formatTrainingParams = (params: Record<string, unknown> | undefined) => {
    if (!params) return []
    return [
      { label: 'Epochs', value: params.epochs },
      { label: 'Batch Size', value: params.batchSize },
      { label: 'Learning Rate', value: params.baseLr },
      { label: 'GPU IDs', value: params.gpuIds },
      { label: 'AMP', value: params.useAmp ? 'Enabled' : 'Disabled' },
      { label: 'VDL', value: params.useVdl ? 'Enabled' : 'Disabled' },
    ].filter(p => p.value !== null && p.value !== undefined)
  }

  const resetFilter = () => {
    setFilterData(defaultFilterData)
    setShowAll(false)
  }

  const hasActiveFilters = 
    (filterData.projectId && filterData.projectId !== '__all__') ||
    (filterData.datasetId && filterData.datasetId !== '__all__') ||
    (filterData.modelId && filterData.modelId !== '__all__') ||
    (filterData.configId && filterData.configId !== '__all__')

  const command = generateCommand()
  const commandDisplay = generateCommandDisplay()

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Jobs</h1>
          <p className="text-muted-foreground">
            Manage your training jobs
          </p>
        </div>
        <div className="flex gap-2">
          <Dialog open={dialogOpen} onOpenChange={(open) => {
            setDialogOpen(open)
            if (!open) resetForm()
          }}>
            <DialogTrigger asChild>
              <Button>
                <Plus className="w-4 h-4 mr-2" />
                Create Job
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
              <DialogHeader>
                <DialogTitle>Create Training Job</DialogTitle>
                <DialogDescription>
                  Configure and create a new training job
                </DialogDescription>
              </DialogHeader>
              <form onSubmit={handleSubmit}>
                <div className="space-y-4 py-4">
                  {/* Basic Settings */}
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="jobName">Job Name</Label>
                        <Input
                          id="jobName"
                          value={formData.name}
                          onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                          placeholder="My Training Job"
                          required
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="project">Project</Label>
                        <Select
                          value={formData.projectId}
                          onValueChange={(value) => setFormData({ ...formData, projectId: value, datasetId: '', modelId: '', configId: '__none__' })}
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="Select project" />
                          </SelectTrigger>
                          <SelectContent>
                            {projects.map((project) => (
                              <SelectItem key={project.id} value={project.id}>
                                {project.name}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                    <div className="grid grid-cols-3 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="dataset">Dataset</Label>
                        <Select
                          value={formData.datasetId}
                          onValueChange={(value) => setFormData({ ...formData, datasetId: value })}
                          disabled={!formData.projectId}
                        >
                          <SelectTrigger className="w-full">
                            <SelectValue placeholder="Select dataset" className="truncate" />
                          </SelectTrigger>
                          <SelectContent>
                            {datasets.map((dataset) => (
                              <SelectItem key={dataset.id} value={dataset.id}>
                                {dataset.name}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="model">Model</Label>
                        <Select
                          value={formData.modelId}
                          onValueChange={(value) => setFormData({ ...formData, modelId: value })}
                          disabled={!formData.projectId}
                        >
                          <SelectTrigger className="w-full">
                            <SelectValue placeholder="Select model" className="truncate" />
                          </SelectTrigger>
                          <SelectContent>
                            {models.map((model) => (
                              <SelectItem key={model.id} value={model.id}>
                                {model.name}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="config">Training Config</Label>
                        <Select
                          value={formData.configId}
                          onValueChange={(value) => setFormData({ ...formData, configId: value })}
                          disabled={!formData.projectId}
                        >
                          <SelectTrigger className="w-full">
                            <SelectValue placeholder="Select config (optional)" className="truncate" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="__none__">None</SelectItem>
                            {configs.map((config) => (
                              <SelectItem key={config.id} value={config.id}>
                                {config.name}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                  </div>

                  {/* Runtime Settings */}
                  <div className="space-y-4 pt-4 border-t">
                    <div className="flex items-center gap-2 text-sm font-medium">
                      <Cpu className="w-4 h-4" />
                      Runtime Settings
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="gpuIds">GPU IDs</Label>
                        <Select
                          value={formData.gpuIds}
                          onValueChange={(value) => setFormData({ ...formData, gpuIds: value })}
                          disabled={gpus.length === 0}
                        >
                          <SelectTrigger className="w-full">
                            <SelectValue placeholder={gpus.length === 0 ? "No GPUs available" : "Select GPU"} />
                          </SelectTrigger>
                          <SelectContent>
                            {gpus.map((gpu) => {
                              const isIdle = gpu.status === 'idle'
                              return (
                                <SelectItem 
                                  key={gpu.id} 
                                  value={String(gpu.id)}
                                  disabled={!isIdle}
                                >
                                  <div className="flex items-center justify-between w-full gap-4">
                                    <span>GPU {gpu.id} - {gpu.name || 'Unknown'}</span>
                                    <Badge 
                                      variant={gpu.status === 'idle' ? 'secondary' : 'destructive'} 
                                      className="text-[10px]"
                                    >
                                      {gpu.status === 'idle' ? 'Idle' : 'Occupied'}
                                    </Badge>
                                  </div>
                                </SelectItem>
                              )
                            })}
                          </SelectContent>
                        </Select>
                        <p className="text-xs text-muted-foreground">
                          Select GPU for training (Occupied GPUs are disabled)
                        </p>
                      </div>
                      <div className="flex flex-col justify-end gap-4">
                        <div className="flex items-center justify-between">
                          <div className="space-y-0.5">
                            <Label htmlFor="useAmp">Use AMP</Label>
                            <p className="text-xs text-muted-foreground">
                              Automatic Mixed Precision
                            </p>
                          </div>
                          <Switch
                            id="useAmp"
                            checked={formData.useAmp}
                            onCheckedChange={(checked) => setFormData({ ...formData, useAmp: checked })}
                          />
                        </div>
                        <div className="flex items-center justify-between">
                          <div className="space-y-0.5">
                            <Label htmlFor="useVdl">Use VDL</Label>
                            <p className="text-xs text-muted-foreground">
                              VisualDL logging
                            </p>
                          </div>
                          <Switch
                            id="useVdl"
                            checked={formData.useVdl}
                            onCheckedChange={(checked) => setFormData({ ...formData, useVdl: checked })}
                          />
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Command Preview */}
                  {commandDisplay && (
                    <div className="space-y-2 pt-4 border-t">
                      <Label className="flex items-center gap-2">
                        <Terminal className="w-4 h-4" />
                        Generated Command
                      </Label>
                      <pre className="p-3 rounded-lg bg-muted/50 text-xs overflow-auto max-h-[150px] font-mono whitespace-pre-wrap break-all">
                        {commandDisplay}
                      </pre>
                    </div>
                  )}
                </div>
                <DialogFooter>
                  <Button type="button" variant="outline" onClick={() => setDialogOpen(false)}>Cancel</Button>
                  <Button type="submit" disabled={submitting}>
                    {submitting ? (
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    ) : (
                      <Play className="w-4 h-4 mr-2" />
                    )}
                    Create Job
                  </Button>
                </DialogFooter>
              </form>
            </DialogContent>
          </Dialog>

          <Button variant="outline" onClick={() => fetchJobs()} disabled={loading}>
            <RotateCcw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Filter Section */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2 text-base">
              <Filter className="w-4 h-4" />
              Filter Jobs
            </CardTitle>
            <div className="flex items-center gap-2">
              {hasActiveFilters && (
                <Button variant="ghost" size="sm" onClick={resetFilter}>
                  <X className="w-4 h-4 mr-1" />
                  Clear Filters
                </Button>
              )}
              <Button 
                variant={showAll ? "default" : "outline"} 
                size="sm"
                onClick={() => setShowAll(!showAll)}
              >
                {showAll ? 'Show Latest 5' : `Show All (${totalJobs})`}
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Project Filter */}
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground">Project</Label>
              <Select
                value={filterData.projectId}
                onValueChange={(value) => setFilterData({ ...filterData, projectId: value })}
              >
                <SelectTrigger>
                  <SelectValue placeholder="All Projects" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__all__">All Projects</SelectItem>
                  {projects.map((project) => (
                    <SelectItem key={project.id} value={project.id}>
                      {project.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Dataset Filter */}
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground">Dataset</Label>
              <Select
                value={filterData.datasetId}
                onValueChange={(value) => setFilterData({ ...filterData, datasetId: value })}
              >
                <SelectTrigger>
                  <SelectValue placeholder="All Datasets" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__all__">All Datasets</SelectItem>
                  {allDatasets.map((dataset) => (
                    <SelectItem key={dataset.id} value={dataset.id}>
                      {dataset.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Model Filter */}
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground">Model</Label>
              <Select
                value={filterData.modelId}
                onValueChange={(value) => setFilterData({ ...filterData, modelId: value })}
              >
                <SelectTrigger>
                  <SelectValue placeholder="All Models" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__all__">All Models</SelectItem>
                  {allModels.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      {model.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Config Filter */}
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground">Config</Label>
              <Select
                value={filterData.configId}
                onValueChange={(value) => setFilterData({ ...filterData, configId: value })}
              >
                <SelectTrigger>
                  <SelectValue placeholder="All Configs" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__all__">All Configs</SelectItem>
                  <SelectItem value="__none__">No Config</SelectItem>
                  {configs.map((config) => (
                    <SelectItem key={config.id} value={config.id}>
                      {config.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Jobs List */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <List className="w-5 h-5" />
            Training Jobs
          </CardTitle>
          <CardDescription>
            View and manage all training jobs
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <RotateCcw className="w-12 h-12 text-muted-foreground mb-4 animate-spin" />
              <p className="text-sm text-muted-foreground">Loading jobs...</p>
            </div>
          ) : jobs.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <AlertCircle className="w-12 h-12 text-muted-foreground mb-4" />
              <h3 className="text-lg font-medium">No training jobs</h3>
              <p className="text-sm text-muted-foreground mt-1">
                Click "Create Job" to start a new training job
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {jobs.map((job) => (
                <Collapsible
                  key={job.id}
                  open={expandedJobs.has(job.id)}
                  onOpenChange={() => toggleJobExpand(job.id)}
                >
                  <div className="rounded-lg border bg-card hover:bg-accent/50 transition-colors">
                    {/* Job Header - Always visible */}
                    <div className="flex items-center justify-between p-4">
                      <div className="flex items-center gap-3 flex-1 min-w-0">
                        <CollapsibleTrigger asChild>
                          <Button variant="ghost" size="icon" className="h-6 w-6">
                            {expandedJobs.has(job.id) ? (
                              <ChevronDown className="w-4 h-4" />
                            ) : (
                              <ChevronRight className="w-4 h-4" />
                            )}
                          </Button>
                        </CollapsibleTrigger>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="font-medium truncate">{job.name}</span>
                            {getStatusBadge(job.status)}
                          </div>
                          <div className="flex items-center gap-4 text-sm text-muted-foreground">
                            <span>{job.project?.name || '-'}</span>
                            <span>•</span>
                            <span>{job.model?.name || '-'}</span>
                            <span>•</span>
                            <span>{job.dataset?.name || '-'}</span>
                          </div>
                          {job.status === 'running' && (
                            <div className="mt-2">
                              <div className="flex items-center justify-between text-xs mb-1">
                                <span>Progress</span>
                                <span>{job.currentEpoch} / {job.totalEpochs} epochs</span>
                              </div>
                              <Progress value={(job.currentEpoch / job.totalEpochs) * 100} className="h-1.5" />
                            </div>
                          )}
                          {job.currentLoss && (
                            <div className="mt-1 text-xs text-muted-foreground">
                              Loss: {job.currentLoss.toFixed(4)}
                              {job.currentLr && ` | LR: ${job.currentLr.toFixed(6)}`}
                            </div>
                          )}
                          {job.status === 'failed' && job.errorMessage && (
                            <div className="mt-2 p-2 rounded bg-destructive/10 border border-destructive/20">
                              <div className="flex items-start gap-2">
                                  <AlertCircle className="w-4 h-4 text-destructive mt-0.5 flex-shrink-0" />
                                  <div className="flex-1 min-w-0">
                                      <p className="text-xs font-medium text-destructive">Error</p>
                                      <p className="text-xs text-destructive/80 mt-0.5 break-words">{job.errorMessage}</p>
                                  </div>
                              </div>
                          </div>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center gap-2 ml-4">
                        {job.status === 'pending' && (
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => handleUpdateJobStatus(job.id, 'running')}
                          >
                            <Play className="w-4 h-4" />
                          </Button>
                        )}
                        {job.status === 'running' && (
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => handleUpdateJobStatus(job.id, 'stopped')}
                          >
                            <Pause className="w-4 h-4" />
                          </Button>
                        )}
                        {(job.status === 'stopped' || job.status === 'failed') && (
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => handleUpdateJobStatus(job.id, 'pending')}
                          >
                            <RotateCcw className="w-4 h-4" />
                          </Button>
                        )}
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="icon">
                              <MoreHorizontal className="w-4 h-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem onClick={() => {
                              if (job.command) {
                                navigator.clipboard.writeText(job.command)
                                toast({ title: 'Command copied' })
                              }
                            }}>
                              <Copy className="w-4 h-4 mr-2" />
                              Copy Command
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              className="text-destructive"
                              onClick={() => handleDeleteJob(job.id)}
                            >
                              <Trash2 className="w-4 h-4 mr-2" />
                              Delete
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    </div>

                    {/* Expanded Content - Training Details */}
                    <CollapsibleContent>
                      <div className="border-t px-4 py-4 bg-muted/30">
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                          {/* Training Parameters */}
                          <div className="space-y-3">
                            <div className="flex items-center gap-2 text-sm font-medium">
                              <Settings2 className="w-4 h-4" />
                              Training Parameters
                            </div>
                            <div className="grid grid-cols-2 gap-2 text-sm">
                              {formatTrainingParams(job.trainingParams as Record<string, unknown>).map((param, idx) => (
                                <div key={idx} className="flex items-center justify-between p-2 rounded bg-background/50">
                                  <span className="text-muted-foreground">{param.label}</span>
                                  <Badge variant="secondary" className="font-mono text-xs">
                                    {String(param.value)}
                                  </Badge>
                                </div>
                              ))}
                              {(!job.trainingParams || formatTrainingParams(job.trainingParams as Record<string, unknown>).length === 0) && (
                                <div className="col-span-2 text-sm text-muted-foreground text-center py-4">
                                  No training parameters available
                                </div>
                              )}
                            </div>
                          </div>

                          {/* Training Command */}
                          <div className="space-y-3">
                            <div className="flex items-center gap-2 text-sm font-medium">
                              <Terminal className="w-4 h-4" />
                              Training Command
                            </div>
                            {job.command ? (
                              <div className="relative">
                                <pre className="p-3 rounded-lg bg-background/80 text-xs overflow-auto max-h-[200px] font-mono whitespace-pre-wrap break-all border">
                                  {job.command}
                                </pre>
                                <Button
                                  variant="outline"
                                  size="sm"
                                  className="absolute top-2 right-2 h-6"
                                  onClick={() => {
                                    if (job.command) {
                                      navigator.clipboard.writeText(job.command)
                                      toast({ title: 'Command copied' })
                                    }
                                  }}
                                >
                                  <Copy className="w-3 h-3" />
                                </Button>
                              </div>
                            ) : (
                              <div className="text-sm text-muted-foreground text-center py-4 bg-background/50 rounded border">
                                No command available
                              </div>
                            )}
                          </div>
                        </div>

                        {/* Output Information */}
                        {(job.configPath || job.outputDir || job.weightsPath || job.startedAt || job.completedAt) && (
                          <div className="mt-4 pt-4 border-t">
                            <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
                              {job.configPath && (
                                <div>
                                  <span className="font-medium">Config:</span> {job.configPath}
                                </div>
                              )}
                              {job.outputDir && (
                                <div>
                                  <span className="font-medium">Output:</span> {job.outputDir}
                                </div>
                              )}
                              {job.weightsPath && (
                                <div>
                                  <span className="font-medium">Weights:</span> {job.weightsPath}
                                </div>
                              )}
                              {job.startedAt && (
                                <div>
                                  <span className="font-medium">Started:</span> {new Date(job.startedAt).toLocaleString()}
                                </div>
                              )}
                              {job.completedAt && (
                                <div>
                                  <span className="font-medium">Completed:</span> {new Date(job.completedAt).toLocaleString()}
                                </div>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    </CollapsibleContent>
                  </div>
                </Collapsible>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
