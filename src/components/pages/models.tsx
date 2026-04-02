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
import { Switch } from '@/components/ui/switch'
import { Slider } from '@/components/ui/slider'
import {
  Plus,
  Search,
  MoreHorizontal,
  Pencil,
  Trash2,
  Cpu,
  Code,
  Settings2,
  Download,
  Copy,
  Upload,
  FileText,
  Loader2,
  Filter,
  X,
} from 'lucide-react'
import { toast } from '@/hooks/use-toast'

interface Project {
  id: string
  name: string
}

interface Model {
  id: string
  name: string
  description: string | null
  projectId: string
  architecture: string
  backbone: string
  neck: string
  head: string
  numClasses: number
  normType: string
  useEma: boolean
  emaDecay: number
  depthMult: number
  widthMult: number
  pretrainWeights: string | null
  yamlConfig: string | null
  createdAt: string
  project?: {
    id: string
    name: string
  }
}

const BACKBONES = ['CSPResNet', 'MobileNetV3', 'ResNet']
const NECKS = ['CustomCSPPAN', 'FPN', 'YOLOv3FPN', 'CenterNetDLAFPN','HybridEncoder']
const HEADS = ['PPYOLOEHead', 'RetinaHead', 'YOLOv3Head', 'CenterNetHead', 'DINOHead']

interface ConfigFile {
  name: string
  path: string
  content: string
}

export function ModelsPage() {
  const [models, setModels] = useState<Model[]>([])
  const [projects, setProjects] = useState<Project[]>([])
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [dialogOpen, setDialogOpen] = useState(false)
  const [importDialogOpen, setImportDialogOpen] = useState(false)
  const [editingModel, setEditingModel] = useState<Model | null>(null)
  const [selectedModel, setSelectedModel] = useState<Model | null>(null)
  const [importing, setImporting] = useState(false)
  const [filterProjectId, setFilterProjectId] = useState<string>('__all__')
  const [defaultConfigs, setDefaultConfigs] = useState<ConfigFile[]>([])
  const [userConfigs, setUserConfigs] = useState<ConfigFile[]>([])
  const [importForm, setImportForm] = useState({
    projectId: '',
    name: '',
    description: '',
    configSource: 'default' as 'default' | 'user' | 'custom',
    selectedConfig: '',
    customYaml: '',
  })
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    projectId: '',
    architecture: 'YOLOv3',
    backbone: 'CSPResNet',
    neck: 'CustomCSPPAN',
    head: 'PPYOLOEHead',
    numClasses: 1,
    normType: 'sync_bn',
    useEma: true,
    emaDecay: 0.9998,
    depthMult: 0.33,
    widthMult: 0.50,
    pretrainWeights: '',
    yamlConfig: '',
  })

  useEffect(() => {
    fetchModels()
    fetchProjects()
  }, [])



  const fetchModels = async () => {
    try {
      const response = await fetch('/api/models')
      if (response.ok) {
        const data = await response.json()
        setModels(data.data || data)
      }
    } catch (error) {
      console.error('Failed to fetch models:', error)
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

  const fetchConfigs = async (projectId: string) => {
    try {
      const response = await fetch(`/api/models/import?projectId=${projectId}`)
      if (response.ok) {
        const result = await response.json()
        if (result.success) {
          setDefaultConfigs(result.data.defaultConfigs || [])
          setUserConfigs(result.data.userConfigs || [])
        }
      }
    } catch (error) {
      console.error('Failed to fetch configs:', error)
    }
  }

  const handleImportModel = async () => {
    if (!importForm.projectId || !importForm.name) {
      toast({
        title: 'Missing required fields',
        description: 'Please select a project and enter a model name',
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
      
      const response = await fetch('/api/models/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          projectId: importForm.projectId,
          name: importForm.name,
          description: importForm.description,
          yamlContent: importForm.configSource === 'custom' ? importForm.customYaml : selectedConfigData?.content,
          isDefault: importForm.configSource === 'default',
          configPath: importForm.configSource !== 'custom' ? selectedConfigData?.path : null,
        }),
      })

      const result = await response.json()
      
      if (result.success) {
        toast({
          title: 'Model imported successfully',
          description: `Config saved to ${result.data.configPath}`,
        })
        fetchModels()
        setImportDialogOpen(false)
        setImportForm({
          projectId: '',
          name: '',
          description: '',
          configSource: 'default',
          selectedConfig: '',
          customYaml: '',
        })
      } else {
        throw new Error(result.error || 'Failed to import model')
      }
    } catch (error) {
      toast({
        title: 'Error importing model',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      })
    } finally {
      setImporting(false)
    }
  }

  const generateYaml = () => {
    return `# Model Configuration - Generated by AutoTrain
architecture: ${formData.architecture}
norm_type: ${formData.normType}
use_ema: ${formData.useEma}
ema_decay: ${formData.emaDecay}
ema_black_list: ['proj_conv.weight']
custom_black_list: ['reduce_mean']

${formData.architecture}:
  backbone: ${formData.backbone}
  neck: ${formData.neck}
  yolo_head: ${formData.head}
  post_process: ~

${formData.backbone}:
  layers: [3, 6, 6, 3]
  channels: [64, 128, 256, 512, 1024]
  return_idx: [1, 2, 3]
  use_large_stem: True
  use_alpha: True

${formData.neck}:
  out_channels: [768, 384, 192]
  stage_num: 1
  block_num: 3
  act: 'swish'
  spp: true

${formData.head}:
  fpn_strides: [32, 16, 8]
  grid_cell_scale: 5.0
  grid_cell_offset: 0.5
  static_assigner_epoch: 30
  use_varifocal_loss: True
  loss_weight: {class: 1.0, iou: 2.5, dfl: 0.5}

depth_mult: ${formData.depthMult}
width_mult: ${formData.widthMult}
${formData.pretrainWeights ? `pretrain_weights: ${formData.pretrainWeights}` : ''}
`
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    const yamlConfig = generateYaml()
    
    try {
      if (editingModel) {
        const response = await fetch(`/api/models/${editingModel.id}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ...formData, yamlConfig }),
        })
        if (response.ok) {
          toast({ title: 'Model updated successfully' })
          fetchModels()
          setDialogOpen(false)
          setEditingModel(null)
        }
      } else {
        // Create new model - first save YAML file, then create model record
        const response = await fetch('/api/models/import', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            projectId: formData.projectId,
            name: formData.name,
            description: formData.description,
            yamlContent: yamlConfig,
            isDefault: false, // New model configs go to autotrain/models/user
            configPath: null,
          }),
        })
        
        if (response.ok) {
          const result = await response.json()
          toast({ 
            title: 'Model created successfully',
            description: `Config saved to ${result.data.configPath}`,
          })
          fetchModels()
          setDialogOpen(false)
          resetForm()
        } else {
          const error = await response.json()
          throw new Error(error.error || 'Failed to create model')
        }
      }
    } catch (error) {
      toast({ 
        title: 'Error saving model', 
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive' 
      })
    }
  }

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this model?')) {
      return
    }
    
    try {
      const response = await fetch(`/api/models/${id}`, {
        method: 'DELETE',
      })
      if (response.ok) {
        toast({ title: 'Model deleted successfully' })
        fetchModels()
        if (selectedModel?.id === id) {
          setSelectedModel(null)
        }
      }
    } catch (error) {
      toast({ title: 'Error deleting model', variant: 'destructive' })
    }
  }

  const openEditDialog = (model: Model) => {
    setEditingModel(model)
    setFormData({
      name: model.name,
      description: model.description || '',
      projectId: model.projectId,
      architecture: model.architecture,
      backbone: model.backbone,
      neck: model.neck,
      head: model.head,
      numClasses: model.numClasses,
      normType: model.normType,
      useEma: model.useEma,
      emaDecay: model.emaDecay,
      depthMult: model.depthMult,
      widthMult: model.widthMult,
      pretrainWeights: model.pretrainWeights || '',
      yamlConfig: model.yamlConfig || '',
    })
    setDialogOpen(true)
  }

  const resetForm = () => {
    setFormData({
      name: '',
      description: '',
      projectId: '',
      architecture: 'YOLOv3',
      backbone: 'CSPResNet',
      neck: 'CustomCSPPAN',
      head: 'PPYOLOEHead',
      numClasses: 1,
      normType: 'sync_bn',
      useEma: true,
      emaDecay: 0.9998,
      depthMult: 0.33,
      widthMult: 0.50,
      pretrainWeights: '',
      yamlConfig: '',
    })
  }

  const copyYamlToClipboard = () => {
    if (selectedModel?.yamlConfig) {
      navigator.clipboard.writeText(selectedModel.yamlConfig)
      toast({ title: 'YAML copied to clipboard' })
    }
  }

  const filteredModels = models.filter(model => {
    const matchesSearch = model.name.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesProject = filterProjectId === '__all__' || model.projectId === filterProjectId
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
          <h1 className="text-3xl font-bold tracking-tight">Models</h1>
          <p className="text-muted-foreground">
            Configure your model architectures
          </p>
        </div>
        <div className="flex gap-2">
          <Dialog open={importDialogOpen} onOpenChange={(open) => {
            setImportDialogOpen(open)
            if (!open) {
              setImportForm({
                projectId: '',
                name: '',
                description: '',
                configSource: 'default',
                selectedConfig: '',
                customYaml: '',
              })
            }
          }}>
            <DialogTrigger asChild>
              <Button variant="outline">
                <Upload className="w-4 h-4 mr-2" />
                Import Model
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
              <DialogHeader>
                <DialogTitle>Import Model Configuration</DialogTitle>
                <DialogDescription>
                  Import an existing model configuration from autotrain/models/default or autotrain/models/user folder, or paste custom YAML
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
                        fetchConfigs(value)
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
                    <Label htmlFor="importName">Model Name</Label>
                    <Input
                      id="importName"
                      value={importForm.name}
                      onChange={(e) => setImportForm({ ...importForm, name: e.target.value })}
                      placeholder="PPYOLOE-Plus-S"
                    />
                  </div>
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="importDesc">Description</Label>
                  <Textarea
                    id="importDesc"
                    value={importForm.description}
                    onChange={(e) => setImportForm({ ...importForm, description: e.target.value })}
                    placeholder="Model description..."
                    rows={2}
                  />
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
                        <SelectValue placeholder={importForm.configSource === 'default' ? "Select from autotrain/models/default" : "Select from autotrain/models/user"} />
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
                <Button onClick={handleImportModel} disabled={importing}>
                  {importing ? (
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  ) : (
                    <FileText className="w-4 h-4 mr-2" />
                  )}
                  Import Model
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
          
          <Dialog open={dialogOpen} onOpenChange={(open) => {
            setDialogOpen(open)
            if (!open) {
              setEditingModel(null)
              resetForm()
            }
          }}>
            <DialogTrigger asChild>
              
            </DialogTrigger>
            <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>{editingModel ? 'Edit Model' : 'Create Model Configuration'}</DialogTitle>
              <DialogDescription>
                Configure your model architecture and parameters
              </DialogDescription>
            </DialogHeader>
            <form onSubmit={handleSubmit}>
              <Tabs defaultValue="basic" className="w-full">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="basic">Basic</TabsTrigger>
                  <TabsTrigger value="architecture">Architecture</TabsTrigger>
                  <TabsTrigger value="advanced">Advanced</TabsTrigger>
                </TabsList>
                
                <TabsContent value="basic" className="space-y-4 mt-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="name">Model Name</Label>
                      <Input
                        id="name"
                        value={formData.name}
                        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                        placeholder="PPYOLOE-Plus-S"
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
                    <Label htmlFor="description">Description</Label>
                    <Textarea
                      id="description"
                      value={formData.description}
                      onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                      placeholder="Describe your model..."
                      rows={2}
                    />
                  </div>
                </TabsContent>

                <TabsContent value="architecture" className="space-y-4 mt-4">
                  <div className="space-y-2">
                    <Label>Architecture</Label>
                    <Select
                      value={formData.architecture}
                      onValueChange={(value) => setFormData({ ...formData, architecture: value })}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select architecture" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="YOLOv3">YOLOv3</SelectItem>
                        <SelectItem value="CenterNet">CenterNet</SelectItem>
                        <SelectItem value="RetinaNet">RetinaNet</SelectItem>
                        <SelectItem value="DETR">DETR</SelectItem>
                        <SelectItem value="FasterRCNN">FasterRCNN</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="space-y-2">
                      <Label>Backbone</Label>
                      <Select
                        value={formData.backbone}
                        onValueChange={(value) => setFormData({ ...formData, backbone: value })}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select" />
                        </SelectTrigger>
                        <SelectContent>
                          {BACKBONES.map((b) => (
                            <SelectItem key={b} value={b}>{b}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label>Neck</Label>
                      <Select
                        value={formData.neck}
                        onValueChange={(value) => setFormData({ ...formData, neck: value })}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select" />
                        </SelectTrigger>
                        <SelectContent>
                          {NECKS.map((n) => (
                            <SelectItem key={n} value={n}>{n}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label>Head</Label>
                      <Select
                        value={formData.head}
                        onValueChange={(value) => setFormData({ ...formData, head: value })}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select" />
                        </SelectTrigger>
                        <SelectContent>
                          {HEADS.map((h) => (
                            <SelectItem key={h} value={h}>{h}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="pretrainWeights">Pretrained Weights URL</Label>
                    <Input
                      id="pretrainWeights"
                      value={formData.pretrainWeights}
                      onChange={(e) => setFormData({ ...formData, pretrainWeights: e.target.value })}
                      placeholder="https://paddledet.bj.bcebos.com/models/..."
                    />
                  </div>
                </TabsContent>

                <TabsContent value="advanced" className="space-y-4 mt-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Normalization Type</Label>
                      <Select
                        value={formData.normType}
                        onValueChange={(value) => setFormData({ ...formData, normType: value })}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="sync_bn">Sync BatchNorm</SelectItem>
                          <SelectItem value="bn">BatchNorm</SelectItem>
                          <SelectItem value="gn">GroupNorm</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="flex items-center justify-between pt-8">
                      <Label htmlFor="useEma">Use EMA</Label>
                      <Switch
                        id="useEma"
                        checked={formData.useEma}
                        onCheckedChange={(checked) => setFormData({ ...formData, useEma: checked })}
                      />
                    </div>
                  </div>
                  <div className="space-y-2">
                    <Label>Depth Multiplier: {formData.depthMult.toFixed(2)}</Label>
                    <Slider
                      value={[formData.depthMult * 100]}
                      onValueChange={(value) => setFormData({ ...formData, depthMult: value[0] / 100 })}
                      min={10}
                      max={100}
                      step={1}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Width Multiplier: {formData.widthMult.toFixed(2)}</Label>
                    <Slider
                      value={[formData.widthMult * 100]}
                      onValueChange={(value) => setFormData({ ...formData, widthMult: value[0] / 100 })}
                      min={10}
                      max={100}
                      step={1}
                    />
                  </div>
                </TabsContent>
              </Tabs>
              <DialogFooter className="mt-6">
                <Button type="submit">{editingModel ? 'Update' : 'Create'}</Button>
              </DialogFooter>
            </form>
          </DialogContent>
        </Dialog>
      </div>
      </div>

      {/* Main Content */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Model List */}
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
                    placeholder="Search models..."
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
          ) : filteredModels.length === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Cpu className="w-12 h-12 text-muted-foreground mb-4" />
                <h3 className="text-lg font-medium">No models found</h3>
                <p className="text-sm text-muted-foreground mt-1">
                  {searchQuery ? 'Try a different search term' : 'Create your first model configuration'}
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-3">
              {filteredModels.map((model) => (
                <Card
                  key={model.id}
                  className={`cursor-pointer transition-all hover:shadow-md ${
                    selectedModel?.id === model.id ? 'ring-2 ring-primary' : ''
                  }`}
                  onClick={() => setSelectedModel(model)}
                >
                  <CardHeader className="pb-2">
                    <div className="flex items-start justify-between">
                      <div>
                        <CardTitle className="text-lg">{model.name}</CardTitle>
                        <CardDescription>
                          {model.description || 'No description'}
                        </CardDescription>
                      </div>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
                          <Button variant="ghost" size="icon">
                            <MoreHorizontal className="w-4 h-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem onClick={(e) => {
                            e.stopPropagation()
                            openEditDialog(model)
                          }}>
                            <Pencil className="w-4 h-4 mr-2" />
                            Edit
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            className="text-destructive"
                            onClick={(e) => {
                              e.stopPropagation()
                              handleDelete(model.id)
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
                      <Badge variant="outline">{model.architecture}</Badge>
                      <span>{model.backbone}</span>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>

        {/* Model Details */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Code className="w-5 h-5" />
                YAML Configuration
              </CardTitle>
              <CardDescription>
                {selectedModel ? selectedModel.name : 'Select a model to view config'}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {selectedModel ? (
                <div className="space-y-4">
                  <div className="relative">
                    <pre className="p-4 rounded-lg bg-muted/50 text-xs overflow-auto max-h-[400px] font-mono">
                      {selectedModel.yamlConfig || generateYaml()}
                    </pre>
                  </div>
                  <div className="flex gap-2">
                    <Button variant="outline" className="flex-1" onClick={copyYamlToClipboard}>
                      <Copy className="w-4 h-4 mr-2" />
                      Copy
                    </Button>
                    <Button variant="outline" className="flex-1">
                      <Download className="w-4 h-4 mr-2" />
                      Download
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-8 text-center">
                  <Settings2 className="w-12 h-12 text-muted-foreground mb-4" />
                  <p className="text-sm text-muted-foreground">
                    Click on a model to view its YAML configuration
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
