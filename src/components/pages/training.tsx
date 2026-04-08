'use client'

import { useEffect, useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
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
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Slider } from '@/components/ui/slider'
import {
  Plus,
  Search,
  MoreHorizontal,
  Pencil,
  Trash2,
  Settings2,
  Download,
  Copy,
  Upload,
  FileText,
  Loader2,
  Code,
  Filter,
  X,
} from 'lucide-react'
import { toast } from '@/hooks/use-toast'

interface Project {
  id: string
  name: string
  framework: string
}

interface TrainingConfig {
  id: string
  name: string
  projectId: string
  epoch: number
  batchSize: number
  baseLr: number
  momentum: number
  weightDecay: number
  scheduler: string
  warmupEpochs: number
  maxEpochs: number
  workerNum: number
  evalHeight: number
  evalWidth: number
  snapshotEpoch: number
  useGpu: boolean
  logIter: number
  saveDir: string | null
  outputDir: string | null
  weights: string | null
  pretrainWeights: string | null
  createdAt: string
  project?: {
    id: string
    name: string
    framework: string
  }
}

interface ConfigFile {
  name: string
  path: string
  content: string
}

const SCHEDULERS = [
  'CosineDecay',
  'LinearWarmup',
  'PiecewiseDecay',
  'ExpDecay',
  'ConstLr',
]

const defaultFormData = {
  name: '',
  projectId: '',
  // Optimizer params
  epochs: 10,
  batchSize: 8,
  baseLr: 0.001,
  momentum: 0.9,
  weightDecay: 0.0005,
  // Scheduler
  scheduler: 'CosineDecay',
  warmupEpochs: 5,
  maxEpochs: 96,
  // Reader settings
  workerNum: 4,
  imageWidth: 640,
  imageHeight: 640,
  trainBatchSize: 8,
  evalBatchSize: 2,
  // Runtime
  useGpu: true,
  logIter: 20,
  saveDir: '',
  snapshotEpoch: 1,
  // Output
  outputDir: '',
  weights: '',
  pretrainWeights: '',
}

export function TrainingPage() {
  const [projects, setProjects] = useState<Project[]>([])
  const [configs, setConfigs] = useState<TrainingConfig[]>([])
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [dialogOpen, setDialogOpen] = useState(false)
  const [importDialogOpen, setImportDialogOpen] = useState(false)
  const [importing, setImporting] = useState(false)
  const [filterProjectId, setFilterProjectId] = useState<string>('__all__')
  const [selectedConfig, setSelectedConfig] = useState<TrainingConfig | null>(null)
  const [defaultConfigs, setDefaultConfigs] = useState<ConfigFile[]>([])
  const [userConfigs, setUserConfigs] = useState<ConfigFile[]>([])
  const [formData, setFormData] = useState(defaultFormData)
  const [importForm, setImportForm] = useState({
    projectId: '',
    name: '',
    configSource: 'default' as 'default' | 'user' | 'custom',
    selectedConfig: '',
    customYaml: '',
  })

  useEffect(() => {
    fetchProjects()
    fetchConfigs()
  }, [])



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

  const fetchConfigs = async (projectId?: string) => {
    try {
      const params = new URLSearchParams()
      if (projectId && projectId !== '__all__') {
        params.append('projectId', projectId)
      }
      const url = `/api/training-configs${params.toString() ? `?${params.toString()}` : ''}`
      const response = await fetch(url)
      if (response.ok) {
        const data = await response.json()
        setConfigs(Array.isArray(data.data) ? data.data : [])
      }
    } catch (error) {
      console.error('Failed to fetch configs:', error)
      setConfigs([])
    } finally {
      setLoading(false)
    }
  }

  const fetchConfigFiles = async (projectId: string) => {
    try {
      const response = await fetch(`/api/training-configs/import?projectId=${projectId}`)
      if (response.ok) {
        const result = await response.json()
        if (result.success) {
          setDefaultConfigs(result.data.defaultConfigs || [])
          setUserConfigs(result.data.userConfigs || [])
        }
      }
    } catch (error) {
      console.error('Failed to fetch config files:', error)
    }
  }

  const generateYaml = () => {
    return `# Training Configuration - Generated by AutoTrain
epoch: ${formData.epochs}

LearningRate:
  base_lr: ${formData.baseLr}
  schedulers:
    - name: ${formData.scheduler}
      max_epochs: ${formData.maxEpochs}
    - name: LinearWarmup
      start_factor: 0.
      epochs: ${formData.warmupEpochs}

OptimizerBuilder:
  optimizer:
    momentum: ${formData.momentum}
    type: Momentum
  regularizer:
    factor: ${formData.weightDecay}
    type: L2

# Reader settings
worker_num: ${formData.workerNum}
eval_height: &eval_height ${formData.imageHeight}
eval_width: &eval_width ${formData.imageWidth}
eval_size: &eval_size [*eval_height, *eval_width]

TrainReader:
  sample_transforms:
    - Decode: {}
    - RandomDistort: {}
    - RandomExpand: {fill_value: [123.675, 116.28, 103.53]}
    - RandomCrop: {}
    - RandomFlip: {}
  batch_transforms:
    - BatchRandomResize: {target_size: [320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768], random_size: True, random_interp: True, keep_ratio: False}
    - NormalizeImage: {mean: [0., 0., 0.], std: [1., 1., 1.], norm_type: none}
    - Permute: {}
    - PadGT: {}
  batch_size: ${formData.trainBatchSize}
  shuffle: true
  drop_last: true
  use_shared_memory: true
  collate_batch: true

EvalReader:
  sample_transforms:
    - Decode: {}
    - Resize: {target_size: *eval_size, keep_ratio: False, interp: 2}
    - NormalizeImage: {mean: [0., 0., 0.], std: [1., 1., 1.], norm_type: none}
    - Permute: {}
  batch_size: ${formData.evalBatchSize}

TestReader:
  inputs_def:
    image_shape: [3, *eval_height, *eval_width]
  sample_transforms:
    - Decode: {}
    - Resize: {target_size: *eval_size, keep_ratio: False, interp: 2}
    - NormalizeImage: {mean: [0., 0., 0.], std: [1., 1., 1.], norm_type: none}
    - Permute: {}
  batch_size: 1

# Runtime settings
use_gpu: ${formData.useGpu}
log_iter: ${formData.logIter}
${formData.saveDir ? `save_dir: ${formData.saveDir}` : ''}
snapshot_epoch: ${formData.snapshotEpoch}
print_flops: false
print_params: false

# Export settings
export:
  post_process: True
  nms: True
  benchmark: False
  fuse_conv_bn: False
${formData.outputDir ? `output_dir: ${formData.outputDir}` : ''}${formData.weights ? `\nweights: ${formData.weights}` : ''}${formData.pretrainWeights ? `\npretrain_weights: ${formData.pretrainWeights}` : ''}
`
  }

  const handleImportConfig = async () => {
    if (!importForm.projectId || !importForm.name) {
      toast({
        title: 'Missing required fields',
        description: 'Please select a project and enter a config name',
        variant: 'destructive',
      })
      return
    }

    if (importForm.configSource !== 'custom' && !importForm.selectedConfig) {
      toast({
        title: 'No config selected',
        description: 'Please select a configuration file',
        variant: 'destructive',
      })
      return
    }

    setImporting(true)
    try {
      const configs = importForm.configSource === 'default' ? defaultConfigs : userConfigs
      const selectedConfigData = configs.find(c => c.name === importForm.selectedConfig)
      
      const response = await fetch('/api/training-configs/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          projectId: importForm.projectId,
          name: importForm.name,
          yamlContent: importForm.configSource === 'custom' ? importForm.customYaml : selectedConfigData?.content,
          isDefault: importForm.configSource === 'default',
          configPath: importForm.configSource !== 'custom' ? selectedConfigData?.path : null,
        }),
      })

      const result = await response.json()
      
      if (result.success) {
        toast({
          title: 'Config imported successfully',
          description: `Config saved to ${result.data.configPath}`,
        })
        fetchConfigs()
        setImportDialogOpen(false)
        setImportForm({
          projectId: '',
          name: '',
          configSource: 'default',
          selectedConfig: '',
          customYaml: '',
        })
      } else {
        throw new Error(result.error || 'Failed to import config')
      }
    } catch (error) {
      toast({
        title: 'Error importing config',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      })
    } finally {
      setImporting(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    const yamlContent = generateYaml()
    
    try {
      const response = await fetch('/api/training-configs/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          projectId: formData.projectId,
          name: formData.name,
          yamlContent: yamlContent,
          isDefault: false,
          configPath: null,
          trainingParams: {
            epochs: formData.epochs,
            batchSize: formData.trainBatchSize,
            baseLr: formData.baseLr,
            momentum: formData.momentum,
            weightDecay: formData.weightDecay,
            scheduler: formData.scheduler,
            warmupEpochs: formData.warmupEpochs,
            maxEpochs: formData.maxEpochs,
            workerNum: formData.workerNum,
            imageWidth: formData.imageWidth,
            imageHeight: formData.imageHeight,
            useGpu: formData.useGpu,
            logIter: formData.logIter,
            snapshotEpoch: formData.snapshotEpoch,
            saveDir: formData.saveDir || null,
            outputDir: formData.outputDir || null,
            weights: formData.weights || null,
            pretrainWeights: formData.pretrainWeights || null,
          },
        }),
      })
      
      if (response.ok) {
        const result = await response.json()
        toast({ 
          title: 'Config created successfully',
          description: `Config saved to ${result.data.configPath}`,
        })
        fetchConfigs()
        setDialogOpen(false)
        resetForm()
      } else {
        const error = await response.json()
        throw new Error(error.error || 'Failed to create config')
      }
    } catch (error) {
      toast({ 
        title: 'Error saving config', 
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive' 
      })
    }
  }

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this config?')) {
      return
    }
    
    try {
      const response = await fetch(`/api/training-configs/${id}`, {
        method: 'DELETE',
      })
      if (response.ok) {
        toast({ title: 'Config deleted successfully' })
        fetchConfigs()
        if (selectedConfig?.id === id) {
          setSelectedConfig(null)
        }
      }
    } catch (error) {
      toast({ title: 'Error deleting config', variant: 'destructive' })
    }
  }

  const resetForm = () => {
    setFormData(defaultFormData)
  }

  const copyYamlToClipboard = () => {
    const yaml = selectedConfig ? generateYamlFromConfig(selectedConfig) : generateYaml()
    navigator.clipboard.writeText(yaml)
    toast({ title: 'YAML copied to clipboard' })
  }

  const generateYamlFromConfig = (config: TrainingConfig) => {
    return `# Training Configuration
epoch: ${config.epoch}

LearningRate:
  base_lr: ${config.baseLr}
  schedulers:
    - name: ${config.scheduler}
      max_epochs: ${config.maxEpochs}
    - name: LinearWarmup
      start_factor: 0.
      epochs: ${config.warmupEpochs}

OptimizerBuilder:
  optimizer:
    momentum: ${config.momentum}
    type: Momentum
  regularizer:
    factor: ${config.weightDecay}
    type: L2

# Reader settings
worker_num: ${config.workerNum}
eval_height: &eval_height ${config.evalHeight}
eval_width: &eval_width ${config.evalWidth}
eval_size: &eval_size [*eval_height, *eval_width]

TrainReader:
  sample_transforms:
    - Decode: {}
    - RandomDistort: {}
    - RandomExpand: {fill_value: [123.675, 116.28, 103.53]}
    - RandomCrop: {}
    - RandomFlip: {}
  batch_transforms:
    - BatchRandomResize: {target_size: [320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768], random_size: True, random_interp: True, keep_ratio: False}
    - NormalizeImage: {mean: [0., 0., 0.], std: [1., 1., 1.], norm_type: none}
    - Permute: {}
    - PadGT: {}
  batch_size: ${config.batchSize}
  shuffle: true
  drop_last: true
  use_shared_memory: true
  collate_batch: true

EvalReader:
  sample_transforms:
    - Decode: {}
    - Resize: {target_size: *eval_size, keep_ratio: False, interp: 2}
    - NormalizeImage: {mean: [0., 0., 0.], std: [1., 1., 1.], norm_type: none}
    - Permute: {}
  batch_size: 2

TestReader:
  inputs_def:
    image_shape: [3, *eval_height, *eval_width]
  sample_transforms:
    - Decode: {}
    - Resize: {target_size: *eval_size, keep_ratio: False, interp: 2}
    - NormalizeImage: {mean: [0., 0., 0.], std: [1., 1., 1.], norm_type: none}
    - Permute: {}
  batch_size: 1

# Runtime settings
use_gpu: ${config.useGpu}
log_iter: ${config.logIter}
${config.saveDir ? `save_dir: ${config.saveDir}` : ''}
snapshot_epoch: ${config.snapshotEpoch}
print_flops: false
print_params: false

# Export settings
export:
  post_process: True
  nms: True
  benchmark: False
  fuse_conv_bn: False
${config.outputDir ? `output_dir: ${config.outputDir}` : ''}${config.weights ? `\nweights: ${config.weights}` : ''}${config.pretrainWeights ? `\npretrain_weights: ${config.pretrainWeights}` : ''}
`
  }

  const filteredConfigs = configs.filter(config => {
    const matchesSearch = config.name.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesProject = filterProjectId === '__all__' || config.projectId === filterProjectId
    return matchesSearch && matchesProject
  })

  const resetFilter = () => {
    setFilterProjectId('__all__')
    setSearchQuery('')
  }

  const hasActiveFilters = filterProjectId !== '__all__' || searchQuery !== ''

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Training Configs</h1>
          <p className="text-muted-foreground">
            Configure training parameters and generate YAML files
          </p>
        </div>
        <div className="flex gap-2">
          <Dialog open={importDialogOpen} onOpenChange={(open) => {
            setImportDialogOpen(open)
            if (!open) {
              setImportForm({
                projectId: '',
                name: '',
                configSource: 'default',
                selectedConfig: '',
                customYaml: '',
              })
            }
          }}>
            <DialogTrigger asChild>
              <Button variant="outline">
                <Upload className="w-4 h-4 mr-2" />
                Import Config
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
              <DialogHeader>
                <DialogTitle>Import Training Configuration</DialogTitle>
                <DialogDescription>
                  Import an existing training config from autotrain/training/default or autotrain/training/user folder, or paste custom YAML
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4 py-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="importProject">Project</Label>
                    <Select
                      value={importForm.projectId}
                      onValueChange={(value) => {
                        setImportForm({ ...importForm, projectId: value, selectedConfig: '' })
                        fetchConfigFiles(value)
                      }}
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
                  <div className="space-y-2">
                    <Label htmlFor="importName">Config Name</Label>
                    <Input
                      id="importName"
                      value={importForm.name}
                      onChange={(e) => setImportForm({ ...importForm, name: e.target.value })}
                      placeholder="My Training Config"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>Configuration Source</Label>
                  <div className="flex gap-2">
                    <Button
                      type="button"
                      variant={importForm.configSource === 'default' ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => setImportForm({ ...importForm, configSource: 'default', selectedConfig: '' })}
                    >
                      Default Configs
                    </Button>
                    <Button
                      type="button"
                      variant={importForm.configSource === 'user' ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => setImportForm({ ...importForm, configSource: 'user', selectedConfig: '' })}
                    >
                      User Configs
                    </Button>
                    <Button
                      type="button"
                      variant={importForm.configSource === 'custom' ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => setImportForm({ ...importForm, configSource: 'custom' })}
                    >
                      Custom YAML
                    </Button>
                  </div>
                </div>

                {importForm.configSource !== 'custom' ? (
                  <div className="space-y-2">
                    <Label>Select Configuration</Label>
                    <Select
                      value={importForm.selectedConfig}
                      onValueChange={(value) => setImportForm({ ...importForm, selectedConfig: value })}
                      disabled={!importForm.projectId}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder={importForm.configSource === 'default' ? "Select from autotrain/training/default" : "Select from autotrain/training/user"} />
                      </SelectTrigger>
                      <SelectContent>
                        {(importForm.configSource === 'default' ? defaultConfigs : userConfigs).map((config) => (
                          <SelectItem key={config.name} value={config.name}>
                            {config.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                ) : (
                  <div className="space-y-2">
                    <Label>Paste YAML Configuration</Label>
                    <Textarea
                      value={importForm.customYaml}
                      onChange={(e) => setImportForm({ ...importForm, customYaml: e.target.value })}
                      placeholder="Paste your YAML configuration here..."
                      rows={12}
                      className="font-mono text-xs"
                    />
                  </div>
                )}
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={() => setImportDialogOpen(false)}>Cancel</Button>
                <Button onClick={handleImportConfig} disabled={importing}>
                  {importing ? (
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  ) : (
                    <FileText className="w-4 h-4 mr-2" />
                  )}
                  Import Config
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>

          <Dialog open={dialogOpen} onOpenChange={(open) => {
            setDialogOpen(open)
            if (!open) {
              resetForm()
            }
          }}>
            <DialogTrigger asChild>

            </DialogTrigger>
            <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
              <DialogHeader>
                <DialogTitle>Create Training Configuration</DialogTitle>
                <DialogDescription>
                  Configure training parameters and generate YAML config
                </DialogDescription>
              </DialogHeader>
              <form onSubmit={handleSubmit}>
                <Tabs defaultValue="basic" className="w-full">
                  <TabsList className="grid w-full grid-cols-4">
                    <TabsTrigger value="basic">Basic</TabsTrigger>
                    <TabsTrigger value="optimizer">Optimizer</TabsTrigger>
                    <TabsTrigger value="reader">Reader</TabsTrigger>
                    <TabsTrigger value="runtime">Runtime</TabsTrigger>
                  </TabsList>

                  {/* Basic Tab */}
                  <TabsContent value="basic" className="space-y-4 mt-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="name">Config Name</Label>
                        <Input
                          id="name"
                          value={formData.name}
                          onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                          placeholder="My Training Config"
                          required
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="project">Project</Label>
                        <Select
                          value={formData.projectId}
                          onValueChange={(value) => setFormData({ ...formData, projectId: value })}
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
                    <div className="space-y-2">
                      <Label>Epochs: {formData.epochs}</Label>
                      <Slider
                        value={[formData.epochs]}
                        onValueChange={(value) => setFormData({ ...formData, epochs: value[0] })}
                        min={1}
                        max={500}
                        step={1}
                      />
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="trainBatchSize">Train Batch Size</Label>
                        <Input
                          id="trainBatchSize"
                          type="number"
                          min={1}
                          max={64}
                          value={formData.trainBatchSize}
                          onChange={(e) => setFormData({ ...formData, trainBatchSize: parseInt(e.target.value) || 8 })}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="saveDir">Output Directory (optional)</Label>
                        <Input
                          id="saveDir"
                          value={formData.saveDir}
                          onChange={(e) => setFormData({ ...formData, saveDir: e.target.value })}
                          placeholder="/path/to/output"
                        />
                      </div>
                    </div>
                  </TabsContent>

                  {/* Optimizer Tab */}
                  <TabsContent value="optimizer" className="space-y-4 mt-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="baseLr">Base Learning Rate</Label>
                        <Input
                          id="baseLr"
                          type="number"
                          step={0.0001}
                          value={formData.baseLr}
                          onChange={(e) => setFormData({ ...formData, baseLr: parseFloat(e.target.value) || 0.001 })}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="momentum">Momentum</Label>
                        <Input
                          id="momentum"
                          type="number"
                          step={0.01}
                          min={0}
                          max={1}
                          value={formData.momentum}
                          onChange={(e) => setFormData({ ...formData, momentum: parseFloat(e.target.value) || 0.9 })}
                        />
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="weightDecay">Weight Decay</Label>
                        <Input
                          id="weightDecay"
                          type="number"
                          step={0.0001}
                          value={formData.weightDecay}
                          onChange={(e) => setFormData({ ...formData, weightDecay: parseFloat(e.target.value) || 0.0005 })}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label>Scheduler Type</Label>
                        <Select
                          value={formData.scheduler}
                          onValueChange={(value) => setFormData({ ...formData, scheduler: value })}
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="Select scheduler" />
                          </SelectTrigger>
                          <SelectContent>
                            {SCHEDULERS.map((scheduler) => (
                              <SelectItem key={scheduler} value={scheduler}>
                                {scheduler}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label>Warmup Epochs: {formData.warmupEpochs}</Label>
                        <Slider
                          value={[formData.warmupEpochs]}
                          onValueChange={(value) => setFormData({ ...formData, warmupEpochs: value[0] })}
                          min={0}
                          max={20}
                          step={1}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="maxEpochs">Max Epochs</Label>
                        <Input
                          id="maxEpochs"
                          type="number"
                          min={1}
                          value={formData.maxEpochs}
                          onChange={(e) => setFormData({ ...formData, maxEpochs: parseInt(e.target.value) || 100 })}
                        />
                      </div>
                    </div>
                  </TabsContent>

                  {/* Reader Tab */}
                  <TabsContent value="reader" className="space-y-4 mt-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="workerNum">Worker Num</Label>
                        <Input
                          id="workerNum"
                          type="number"
                          min={1}
                          max={16}
                          value={formData.workerNum}
                          onChange={(e) => setFormData({ ...formData, workerNum: parseInt(e.target.value) || 4 })}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="evalBatchSize">Eval Batch Size</Label>
                        <Input
                          id="evalBatchSize"
                          type="number"
                          min={1}
                          max={32}
                          value={formData.evalBatchSize}
                          onChange={(e) => setFormData({ ...formData, evalBatchSize: parseInt(e.target.value) || 2 })}
                        />
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="imageWidth">Image Width</Label>
                        <Input
                          id="imageWidth"
                          type="number"
                          step={32}
                          value={formData.imageWidth}
                          onChange={(e) => setFormData({ ...formData, imageWidth: parseInt(e.target.value) || 640 })}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="imageHeight">Image Height</Label>
                        <Input
                          id="imageHeight"
                          type="number"
                          step={32}
                          value={formData.imageHeight}
                          onChange={(e) => setFormData({ ...formData, imageHeight: parseInt(e.target.value) || 640 })}
                        />
                      </div>
                    </div>
                  </TabsContent>

                  {/* Runtime Tab */}
                  <TabsContent value="runtime" className="space-y-4 mt-4">
                    <div className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        id="useGpu"
                        checked={formData.useGpu}
                        onChange={(e) => setFormData({ ...formData, useGpu: e.target.checked })}
                        className="h-4 w-4"
                      />
                      <Label htmlFor="useGpu">Use GPU</Label>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="snapshotEpoch">Snapshot Epoch Interval</Label>
                        <Input
                          id="snapshotEpoch"
                          type="number"
                          min={1}
                          value={formData.snapshotEpoch}
                          onChange={(e) => setFormData({ ...formData, snapshotEpoch: parseInt(e.target.value) || 1 })}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="logIter">Log Iteration</Label>
                        <Input
                          id="logIter"
                          type="number"
                          min={1}
                          value={formData.logIter}
                          onChange={(e) => setFormData({ ...formData, logIter: parseInt(e.target.value) || 20 })}
                        />
                      </div>
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="saveDir">Save Directory</Label>
                      <Input
                        id="saveDir"
                        value={formData.saveDir}
                        onChange={(e) => setFormData({ ...formData, saveDir: e.target.value })}
                        placeholder="outputs/your_job"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="outputDir">Output Directory</Label>
                      <Input
                        id="outputDir"
                        value={formData.outputDir}
                        onChange={(e) => setFormData({ ...formData, outputDir: e.target.value })}
                        placeholder="/absolute/path/to/output"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="weights">Weights Path</Label>
                      <Input
                        id="weights"
                        value={formData.weights}
                        onChange={(e) => setFormData({ ...formData, weights: e.target.value })}
                        placeholder="output/model/model_final"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="pretrainWeights">Pretrain Weights URL</Label>
                      <Input
                        id="pretrainWeights"
                        value={formData.pretrainWeights}
                        onChange={(e) => setFormData({ ...formData, pretrainWeights: e.target.value })}
                        placeholder="https://paddledet.bj.bcebos.com/models/pretrained/..."
                      />
                    </div>
                  </TabsContent>
                </Tabs>
                <DialogFooter className="mt-6">
                  <Button type="submit">Create Config</Button>
                </DialogFooter>
              </form>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      {/* Main Content */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Config List */}
        <div className="lg:col-span-2 space-y-4">
          {/* Filter Section */}
          <Card>
            <CardContent className="pt-4">
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Filter className="w-4 h-4" />
                  <span>Filter:</span>
                </div>
                <Select
                  value={filterProjectId}
                  onValueChange={(value) => setFilterProjectId(value)}
                >
                  <SelectTrigger className="w-[200px]">
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
                <div className="relative flex-1 max-w-md">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                  <Input
                    placeholder="Search training configs..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-9"
                  />
                </div>
                {hasActiveFilters && (
                  <Button variant="ghost" size="sm" onClick={resetFilter}>
                    <X className="w-4 h-4 mr-1" />
                    Clear
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>

          {loading ? (
            <div className="grid gap-4">
              {[1, 2].map((i) => (
                <Card key={i} className="animate-pulse">
                  <CardHeader>
                    <div className="h-5 bg-muted rounded w-1/3"></div>
                  </CardHeader>
                </Card>
              ))}
            </div>
          ) : filteredConfigs.length === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Settings2 className="w-12 h-12 text-muted-foreground mb-4" />
                <h3 className="text-lg font-medium">No training configs found</h3>
                <p className="text-sm text-muted-foreground mt-1">
                  {searchQuery ? 'Try a different search term' : 'Create your first training configuration'}
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-3">
              {filteredConfigs.map((config) => (
                <Card
                  key={config.id}
                  className={`cursor-pointer transition-all hover:shadow-md ${
                    selectedConfig?.id === config.id ? 'ring-2 ring-primary' : ''
                  }`}
                  onClick={() => setSelectedConfig(config)}
                >
                  <CardHeader className="pb-2">
                    <div className="flex items-start justify-between">
                      <div>
                        <CardTitle className="text-lg">{config.name}</CardTitle>
                        <CardDescription>
                          {config.scheduler} scheduler
                        </CardDescription>
                      </div>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
                          <Button variant="ghost" size="icon">
                            <MoreHorizontal className="w-4 h-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem
                            className="text-destructive"
                            onClick={(e) => {
                              e.stopPropagation()
                              handleDelete(config.id)
                            }}
                          >
                            <Trash2 className="w-4 h-4 mr-2" />
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center gap-4 text-sm text-muted-foreground">
                      <Badge variant="outline">{config.epoch} epochs</Badge>
                      <span>lr: {config.baseLr}</span>
                      <span>•</span>
                      <span>batch: {config.batchSize}</span>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>

        {/* YAML Preview */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Code className="w-5 h-5" />
                YAML Configuration
              </CardTitle>
              <CardDescription>
                {selectedConfig ? selectedConfig.name : 'Select a config to view YAML'}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {selectedConfig ? (
                <div className="space-y-4">
                  <div className="relative">
                    <pre className="p-4 rounded-lg bg-muted/50 text-xs overflow-auto max-h-[400px] font-mono">
                      {generateYamlFromConfig(selectedConfig)}
                    </pre>
                  </div>
                  <div className="flex gap-2">
                    <Button variant="outline" className="flex-1" onClick={copyYamlToClipboard}>
                      <Copy className="w-4 h-4 mr-2" />
                      Copy
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-8 text-center">
                  <Settings2 className="w-12 h-12 text-muted-foreground mb-4" />
                  <p className="text-sm text-muted-foreground">
                    Click on a config to view its YAML configuration
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
