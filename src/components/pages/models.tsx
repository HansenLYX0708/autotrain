'use client'

import { useEffect, useMemo, useState } from 'react'
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
  Copy,
  Upload,
  FileText,
  Loader2,
  Filter,
  X,
} from 'lucide-react'
import { toast } from '@/hooks/use-toast'
import { ConfigYamlPane, yamlSyntaxError } from '@/components/config-yaml-pane'
import { asConfigFramework, configSchemaOf, type ConfigFramework } from '@/lib/training-yaml'
import {
  ANOMALY_MODEL_SIZES,
  ANOMALY_PRESETS,
  ANOMALY_PRESET_KEYS,
  CLAS_ARCHITECTURES,
  DETECTION_PRESETS,
  DETECTION_PRESET_KEYS,
  SEG_LOSS_TYPES,
  anomalyPreset,
  TORCH_DET_PRESETS,
  TORCH_DET_PRESET_KEYS,
  TORCH_PRETRAIN_OPTIONS,
  segArchitecturesFor,
  defaultModelParams,
  detectionPresetFor,
  generateModelYaml,
  parseModelParams,
  reconcileSegLoss,
  segArchitecture,
  validateModelParams,
  type ModelParams,
} from '@/lib/model-yaml'

interface Project {
  id: string
  name: string
  framework?: string
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
  /** Authoritative config content. Everything above is a display cache. */
  yamlConfig: string | null
  createdAt: string
  project?: {
    id: string
    name: string
    framework?: string
  }
}

interface ConfigFile {
  name: string
  path: string
  content: string
}

export function ModelsPage() {
  const [models, setModels] = useState<Model[]>([])
  const [projects, setProjects] = useState<Project[]>([])
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [dialogOpen, setDialogOpen] = useState(false)
  const [importDialogOpen, setImportDialogOpen] = useState(false)
  const [editingModel, setEditingModel] = useState<Model | null>(null)
  const [selectedModel, setSelectedModel] = useState<Model | null>(null)
  const [importing, setImporting] = useState(false)
  const [filterProjectId, setFilterProjectId] = useState<string>('__all__')
  const [defaultConfigs, setDefaultConfigs] = useState<ConfigFile[]>([])
  const [userConfigs, setUserConfigs] = useState<ConfigFile[]>([])

  // ---- Create / edit dialog state ----------------------------------------
  const [modelName, setModelName] = useState('')
  const [modelDescription, setModelDescription] = useState('')
  const [modelProjectId, setModelProjectId] = useState('')
  const [detectionPreset, setDetectionPreset] = useState('PP-YOLOE')
  const [params, setParams] = useState<ModelParams>(defaultModelParams('PaddleDetection'))
  const [manualYaml, setManualYaml] = useState(false)
  const [manualYamlText, setManualYamlText] = useState('')

  const [importForm, setImportForm] = useState({
    projectId: '',
    name: '',
    description: '',
    configSource: 'default' as 'default' | 'user' | 'custom',
    selectedConfig: '',
    customYaml: '',
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

  // ---- Derived dialog values ---------------------------------------------

  const frameworkOf = (projectId: string): ConfigFramework =>
    asConfigFramework(projects.find((p) => p.id === projectId)?.framework)

  const dialogFramework = frameworkOf(modelProjectId)
  const isSeg = configSchemaOf(dialogFramework) === 'segmentation'
  const isClas = dialogFramework === 'PaddleClas'
  const isTorchDet = dialogFramework === 'TorchDet'
  const isAnomalyModel = dialogFramework === 'TorchAnomaly'
  const anomalyPresetInfo = anomalyPreset(params.architecture)
  const preset = DETECTION_PRESETS[detectionPreset] ?? DETECTION_PRESETS['PP-YOLOE']
  // Architecture vocabularies are framework-specific: TorchSeg offers torchvision
  // networks (UNet / DeepLabV3+ / FCN / LR-ASPP), PaddleSeg offers Paddle's.
  const segArchitectures = segArchitecturesFor(dialogFramework)
  const segArch = segArchitecture(params.architecture, dialogFramework)
  const torchDetPreset = TORCH_DET_PRESETS[params.architecture] ?? TORCH_DET_PRESETS.FasterRCNN

  const generatedYaml = useMemo(
    () => generateModelYaml(dialogFramework, params, modelName || 'Model Config'),
    [dialogFramework, params, modelName],
  )
  const effectiveYaml = manualYaml ? manualYamlText : generatedYaml
  const syntaxError = yamlSyntaxError(effectiveYaml)
  const issues = useMemo(
    () => (manualYaml ? [] : validateModelParams(dialogFramework, params)),
    [manualYaml, dialogFramework, params],
  )
  const blockingIssue = issues.some((i) => i.level === 'error')

  const setParam = <K extends keyof ModelParams>(key: K, value: ModelParams[K]) =>
    setParams((prev) => ({ ...prev, [key]: value }))

  const handleProjectChange = (projectId: string) => {
    const nextFramework = frameworkOf(projectId)
    setModelProjectId(projectId)
    if (nextFramework === dialogFramework) return
    // Architecture vocabularies do not overlap across frameworks, so start from
    // that framework's defaults but keep num_classes, which always transfers.
    const next = defaultModelParams(nextFramework)
    next.numClasses = params.numClasses
    setParams(next)
    setDetectionPreset('PP-YOLOE')
  }

  /** Applying a preset rewires backbone/neck/head to a combination that builds. */
  const applyDetectionPreset = (key: string) => {
    const p = DETECTION_PRESETS[key]
    if (!p) return
    setDetectionPreset(key)
    setParams((prev) => ({
      ...prev,
      architecture: p.architecture,
      backbone: p.backbones.includes(prev.backbone) ? prev.backbone : p.backbones[0],
      neck: p.necks.includes(prev.neck) ? prev.neck : p.necks[0],
      head: p.heads.includes(prev.head) ? prev.head : p.heads[0],
    }))
  }

  /**
   * Changing segmentation architecture must resize `loss.types` / `loss.coef`,
   * because both PaddleSeg and TorchSeg hard-fail when their length differs from
   * the number of logits the network emits.
   */
  const applySegArchitecture = (value: string) => {
    setParams((prev) => {
      const arch = segArchitecture(value, dialogFramework)
      const { segLossTypes, segLossCoef } = reconcileSegLoss(
        value, prev.segLossTypes, prev.segLossCoef, dialogFramework,
      )
      return {
        ...prev,
        architecture: value,
        backbone: arch?.needsBackbone ? (arch.backbones.includes(prev.backbone) ? prev.backbone : arch.backbones[0]) : '',
        segLossTypes,
        segLossCoef,
      }
    })
  }

  /**
   * TorchAnomaly: switching algorithm resets the fields that are
   * algorithm-specific, because carrying them over produces invalid configs —
   * PatchCore's layers are not EfficientAD's (which has none at all), and a
   * backbone name from one table is not in the other's.
   */
  const applyAnomalyArchitecture = (value: string) => {
    const p = ANOMALY_PRESETS[value]
    if (!p) return
    setParams((prev) => ({
      ...prev,
      architecture: value,
      backbone: p.backbones[0] ?? '',
      adLayers: [...p.layers],
      adImageWidth: p.imageSize[0],
      adImageHeight: p.imageSize[1],
      // The paper recipe for PatchCore crops 256 -> 224; nothing else uses it.
      adCenterCrop: 0,
    }))
  }

  /** TorchDet: architecture + backbone fully determine the torchvision builder. */
  const applyTorchDetArchitecture = (value: string) => {
    const p = TORCH_DET_PRESETS[value]
    if (!p) return
    setParams((prev) => ({
      ...prev,
      architecture: p.architecture,
      backbone: p.backbones.includes(prev.backbone) ? prev.backbone : p.backbones[0],
      neck: '',
      head: '',
    }))
  }

  const resetDialog = () => {
    setEditingModel(null)
    setModelName('')
    setModelDescription('')
    setModelProjectId('')
    setDetectionPreset('PP-YOLOE')
    setParams(defaultModelParams('PaddleDetection'))
    setManualYaml(false)
    setManualYamlText('')
  }

  const openCreateDialog = () => {
    resetDialog()
    setDialogOpen(true)
  }

  /**
   * Editing starts from the stored YAML in manual mode. The previous editor
   * regenerated YAML from the form on save, which silently destroyed any
   * imported or hand-written config the moment a user opened it to rename it.
   */
  const openEditDialog = (model: Model) => {
    const framework = asConfigFramework(model.project?.framework)
    const parsed = parseModelParams(framework, model.yamlConfig)
    const next: ModelParams = {
      ...defaultModelParams(framework),
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
      ...parsed,
    }
    if (configSchemaOf(framework) === 'segmentation') {
      const reconciled = reconcileSegLoss(
        next.architecture, next.segLossTypes, next.segLossCoef, framework,
      )
      next.segLossTypes = reconciled.segLossTypes
      next.segLossCoef = reconciled.segLossCoef
    }

    setEditingModel(model)
    setModelName(model.name)
    setModelDescription(model.description || '')
    setModelProjectId(model.projectId)
    setDetectionPreset(detectionPresetFor(next.architecture, next.head))
    setParams(next)
    if (model.yamlConfig) {
      setManualYaml(true)
      setManualYamlText(model.yamlConfig)
    } else {
      setManualYaml(false)
      setManualYamlText('')
    }
    setDialogOpen(true)
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
      const sourceConfigs = importForm.configSource === 'default' ? defaultConfigs : userConfigs
      const selectedConfigData = sourceConfigs.find((c) => c.name === importForm.selectedConfig)

      const response = await fetch('/api/models/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          projectId: importForm.projectId,
          name: importForm.name,
          description: importForm.description,
          yamlContent:
            importForm.configSource === 'custom' ? importForm.customYaml : selectedConfigData?.content,
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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!modelName.trim() || !modelProjectId) {
      toast({
        title: 'Missing required fields',
        description: 'A model name and project are required.',
        variant: 'destructive',
      })
      return
    }
    if (syntaxError) {
      toast({ title: 'Invalid YAML', description: syntaxError, variant: 'destructive' })
      return
    }
    if (blockingIssue) {
      toast({
        title: 'Configuration is not valid',
        description: issues.find((i) => i.level === 'error')?.message,
        variant: 'destructive',
      })
      return
    }

    setSaving(true)
    try {
      const response = editingModel
        ? await fetch(`/api/models/${editingModel.id}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              name: modelName.trim(),
              description: modelDescription,
              yamlConfig: effectiveYaml,
            }),
          })
        : await fetch('/api/models/import', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              projectId: modelProjectId,
              name: modelName.trim(),
              description: modelDescription,
              yamlContent: effectiveYaml,
              isDefault: false,
              configPath: null,
            }),
          })

      const result = await response.json().catch(() => ({}))
      if (!response.ok || result.success === false) {
        throw new Error(result.error || result.message || 'Failed to save model')
      }

      toast({
        title: editingModel ? 'Model updated' : 'Model created',
        description: result?.data?.configPath ? `Saved to ${result.data.configPath}` : undefined,
      })
      await fetchModels()
      if (editingModel && selectedModel?.id === editingModel.id) {
        setSelectedModel(null)
      }
      setDialogOpen(false)
      resetDialog()
    } catch (error) {
      toast({
        title: 'Error saving model',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      })
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this model?')) {
      return
    }

    try {
      const response = await fetch(`/api/models/${id}`, { method: 'DELETE' })
      const result = await response.json().catch(() => ({}))
      if (!response.ok) {
        throw new Error(result.error || 'Failed to delete model')
      }
      toast({ title: 'Model deleted successfully' })
      fetchModels()
      if (selectedModel?.id === id) {
        setSelectedModel(null)
      }
    } catch (error) {
      toast({
        title: 'Error deleting model',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      })
    }
  }

  const copyYamlToClipboard = () => {
    if (!selectedModel?.yamlConfig) return
    navigator.clipboard.writeText(selectedModel.yamlConfig)
    toast({ title: 'YAML copied to clipboard' })
  }

  const filteredModels = models.filter((model) => {
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
          <p className="text-muted-foreground">Configure your model architectures</p>
        </div>
        <div className="flex gap-2">
          <Dialog
            open={importDialogOpen}
            onOpenChange={(open) => {
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
            }}
          >
            <DialogTrigger asChild>
              <Button variant="outline">
                <Upload className="w-4 h-4 mr-2" />
                Import Model
              </Button>
            </DialogTrigger>
            {/* `sm:` prefix required to override DialogContent's own sm:max-w-lg. */}
            <DialogContent className="sm:max-w-2xl max-h-[90vh] overflow-y-auto">
              <DialogHeader>
                <DialogTitle>Import Model Configuration</DialogTitle>
                <DialogDescription>
                  Import a preset model config for this project&apos;s framework, one of your own saved
                  configs, or paste custom YAML.
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
                      placeholder="My Model"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="importDescription">Description</Label>
                  <Textarea
                    id="importDescription"
                    value={importForm.description}
                    onChange={(e) => setImportForm({ ...importForm, description: e.target.value })}
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
                      onClick={() =>
                        setImportForm({ ...importForm, configSource: 'default', selectedConfig: '' })
                      }
                    >
                      Default Configs
                    </Button>
                    <Button
                      type="button"
                      variant={importForm.configSource === 'user' ? 'default' : 'outline'}
                      size="sm"
                      onClick={() =>
                        setImportForm({ ...importForm, configSource: 'user', selectedConfig: '' })
                      }
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
                        <SelectValue
                          placeholder={
                            importForm.configSource === 'default'
                              ? 'Select a preset model config'
                              : 'Select one of your saved configs'
                          }
                        />
                      </SelectTrigger>
                      <SelectContent>
                        {(importForm.configSource === 'default' ? defaultConfigs : userConfigs).map(
                          (config) => (
                            <SelectItem key={config.name} value={config.name}>
                              {config.name}
                            </SelectItem>
                          ),
                        )}
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
                <Button variant="outline" onClick={() => setImportDialogOpen(false)}>
                  Cancel
                </Button>
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

          <Dialog
            open={dialogOpen}
            onOpenChange={(open) => {
              setDialogOpen(open)
              if (!open) resetDialog()
            }}
          >
            <DialogTrigger asChild>
              <Button onClick={openCreateDialog}>
                <Plus className="w-4 h-4 mr-2" />
                New Model
              </Button>
            </DialogTrigger>
            {/* The `sm:` prefix is required: DialogContent's own base class list
                ends with `sm:max-w-lg`, and an unprefixed `max-w-*` here is not
                treated as a conflict by tailwind-merge, so both ship and the
                responsive one wins above 640px — pinning the dialog to 32rem. */}
            <DialogContent className="sm:max-w-[min(1500px,94vw)] max-h-[92vh] overflow-y-auto">
              <DialogHeader>
                <DialogTitle>
                  {editingModel ? `Edit "${editingModel.name}"` : 'Create Model Configuration'}
                </DialogTitle>
                <DialogDescription>
                  {modelProjectId
                    ? `Framework: ${dialogFramework}. Architecture options are restricted to combinations this framework can build.`
                    : 'Select a project first — the available architectures depend on its framework.'}
                </DialogDescription>
              </DialogHeader>

              <form onSubmit={handleSubmit}>
                {/* minmax(0,…) rather than a bare 1fr: grid tracks default to
                    min-content, so the nested field grids would otherwise push
                    the columns wider than the dialog. */}
                <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
                  {/* ---------------- Form side ---------------- */}
                  <div className="space-y-4 min-w-0">
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
                              value={modelName}
                              onChange={(e) => setModelName(e.target.value)}
                              placeholder="My model"
                              required
                            />
                          </div>
                          <div className="space-y-2">
                            <Label htmlFor="project">Project</Label>
                            <Select
                              value={modelProjectId}
                              onValueChange={handleProjectChange}
                              disabled={!!editingModel}
                            >
                              <SelectTrigger>
                                <SelectValue placeholder="Select project" />
                              </SelectTrigger>
                              <SelectContent>
                                {projects.map((project) => (
                                  <SelectItem key={project.id} value={project.id}>
                                    {project.name} · {project.framework ?? 'PaddleDetection'}
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
                            value={modelDescription}
                            onChange={(e) => setModelDescription(e.target.value)}
                            placeholder="Describe your model..."
                            rows={2}
                          />
                        </div>
                        {/* Anomaly detection has no classes: it learns what
                            "normal" looks like and scores deviation from it. */}
                        {isAnomalyModel ? (
                          <p className="text-sm text-muted-foreground">
                            Unsupervised: the model is trained only on normal images and has no
                            classes. Defect images are used for evaluation and for choosing the
                            anomaly-score threshold.
                          </p>
                        ) : (
                        <div className="space-y-2">
                          <Label htmlFor="numClasses">Number of Classes</Label>
                          <Input
                            id="numClasses"
                            type="number"
                            min={1}
                            value={params.numClasses}
                            onChange={(e) => setParam('numClasses', Number(e.target.value) || 1)}
                          />
                          <p className="text-xs text-muted-foreground">
                            {isSeg
                              ? 'Include the background class.'
                              : 'Must match the number of categories in the dataset.'}
                          </p>
                        </div>
                        )}
                      </TabsContent>

                      <TabsContent value="architecture" className="space-y-4 mt-4">
                        {isSeg ? (
                          <>
                            <div className="space-y-2">
                              <Label>Architecture</Label>
                              <Select value={params.architecture} onValueChange={applySegArchitecture}>
                                <SelectTrigger>
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  {segArchitectures.map((a) => (
                                    <SelectItem key={a.value} value={a.value}>
                                      {a.label} · {a.logits} logit{a.logits > 1 ? 's' : ''}
                                    </SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                            </div>
                            {segArch?.needsBackbone && (
                              <div className="space-y-2">
                                <Label>Backbone</Label>
                                <Select
                                  value={params.backbone}
                                  onValueChange={(value) => setParam('backbone', value)}
                                >
                                  <SelectTrigger>
                                    <SelectValue />
                                  </SelectTrigger>
                                  <SelectContent>
                                    {segArch.backbones.map((b) => (
                                      <SelectItem key={b} value={b}>
                                        {b}
                                      </SelectItem>
                                    ))}
                                  </SelectContent>
                                </Select>
                              </div>
                            )}
                            <div className="space-y-3 rounded-lg border p-3">
                              <div>
                                <Label>Loss</Label>
                                <p className="text-xs text-muted-foreground mt-1">
                                  {params.architecture} emits {segArch?.logits ?? '?'} logit
                                  {(segArch?.logits ?? 1) > 1 ? 's' : ''} during training.{' '}
                                  {dialogFramework} requires exactly one loss entry per logit.
                                  {dialogFramework === 'TorchSeg' && (segArch?.logits ?? 1) > 1
                                    ? ' The second entry is the torchvision auxiliary head.'
                                    : ''}
                                </p>
                              </div>
                              {params.segLossTypes.map((type, index) => (
                                <div key={index} className="grid grid-cols-[1fr_100px] gap-2 items-end">
                                  <div className="space-y-1">
                                    <Label className="text-xs font-normal">
                                      {index === 0 ? 'Main head' : `Aux head ${index}`}
                                    </Label>
                                    <Select
                                      value={type}
                                      onValueChange={(value) => {
                                        const next = [...params.segLossTypes]
                                        next[index] = value
                                        setParam('segLossTypes', next)
                                      }}
                                    >
                                      <SelectTrigger>
                                        <SelectValue />
                                      </SelectTrigger>
                                      <SelectContent>
                                        {SEG_LOSS_TYPES.map((t) => (
                                          <SelectItem key={t} value={t}>
                                            {t}
                                          </SelectItem>
                                        ))}
                                      </SelectContent>
                                    </Select>
                                  </div>
                                  <div className="space-y-1">
                                    <Label className="text-xs font-normal">coef</Label>
                                    <Input
                                      type="number"
                                      step={0.1}
                                      value={params.segLossCoef[index] ?? 1}
                                      onChange={(e) => {
                                        const next = [...params.segLossCoef]
                                        next[index] = Number(e.target.value)
                                        setParam('segLossCoef', next)
                                      }}
                                    />
                                  </div>
                                </div>
                              ))}
                            </div>
                          </>
                        ) : isClas ? (
                          <div className="space-y-2">
                            <Label>Architecture</Label>
                            <Select
                              value={params.architecture}
                              onValueChange={(value) => setParam('architecture', value)}
                            >
                              <SelectTrigger>
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                {CLAS_ARCHITECTURES.map((a) => (
                                  <SelectItem key={a} value={a}>
                                    {a}
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          </div>
                        ) : isAnomalyModel ? (
                          <>
                            {/* Unsupervised: no classes, no head. The algorithm
                                choice is the whole decision, and it determines
                                which of the remaining controls mean anything. */}
                            <div className="space-y-2">
                              <Label>Algorithm</Label>
                              <Select value={params.architecture} onValueChange={applyAnomalyArchitecture}>
                                <SelectTrigger>
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  {ANOMALY_PRESET_KEYS.map((key) => (
                                    <SelectItem key={key} value={key}>
                                      {ANOMALY_PRESETS[key].label}
                                    </SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                              <p className="text-xs text-muted-foreground">{anomalyPresetInfo.notes}</p>
                            </div>

                            {anomalyPresetInfo.backbones.length > 0 && (
                              <div className="space-y-2">
                                <Label>Backbone</Label>
                                <Select
                                  value={params.backbone}
                                  onValueChange={(value) => setParam('backbone', value)}
                                >
                                  <SelectTrigger>
                                    <SelectValue />
                                  </SelectTrigger>
                                  <SelectContent>
                                    {anomalyPresetInfo.backbones.map((b) => (
                                      <SelectItem key={b} value={b}>{b}</SelectItem>
                                    ))}
                                  </SelectContent>
                                </Select>
                                <p className="text-xs text-muted-foreground">
                                  Feature extractor, frozen and pretrained on ImageNet. Nothing is
                                  trained on the defect images.
                                </p>
                              </div>
                            )}

                            <div className="grid grid-cols-3 gap-4">
                              <div className="space-y-2">
                                <Label htmlFor="adImageWidth">Input Width</Label>
                                <Input
                                  id="adImageWidth"
                                  type="number"
                                  step={32}
                                  min={32}
                                  value={params.adImageWidth}
                                  onChange={(e) => setParam('adImageWidth', Number(e.target.value) || 32)}
                                />
                              </div>
                              <div className="space-y-2">
                                <Label htmlFor="adImageHeight">Input Height</Label>
                                <Input
                                  id="adImageHeight"
                                  type="number"
                                  step={32}
                                  min={32}
                                  value={params.adImageHeight}
                                  onChange={(e) => setParam('adImageHeight', Number(e.target.value) || 32)}
                                />
                              </div>
                              <div className="space-y-2">
                                <Label htmlFor="adCenterCrop">Centre Crop (0 = off)</Label>
                                <Input
                                  id="adCenterCrop"
                                  type="number"
                                  min={0}
                                  step={16}
                                  value={params.adCenterCrop}
                                  onChange={(e) => setParam('adCenterCrop', Number(e.target.value) || 0)}
                                />
                              </div>
                            </div>
                            <p className="text-xs text-muted-foreground">
                              The whole image is resized to this before the model sees it, so a defect
                              smaller than a couple of pixels at this scale becomes invisible. For
                              small defects on a large image, enable tiling in the training config
                              instead of raising this — and note that EfficientAD cannot tile.
                            </p>

                            {params.architecture === 'Patchcore' && (
                              <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                  <Label htmlFor="adCoresetRatio">Coreset Ratio</Label>
                                  <Input
                                    id="adCoresetRatio"
                                    type="number"
                                    step={0.01}
                                    min={0.001}
                                    max={1}
                                    value={params.adCoresetRatio}
                                    onChange={(e) => setParam('adCoresetRatio', Number(e.target.value) || 0.1)}
                                  />
                                  <p className="text-xs text-muted-foreground">
                                    Fraction of patch features kept in the memory bank. Lower it if
                                    training runs out of memory.
                                  </p>
                                </div>
                                <div className="space-y-2">
                                  <Label htmlFor="adNumNeighbors">Neighbours (k)</Label>
                                  <Input
                                    id="adNumNeighbors"
                                    type="number"
                                    min={1}
                                    value={params.adNumNeighbors}
                                    onChange={(e) => setParam('adNumNeighbors', Number(e.target.value) || 1)}
                                  />
                                </div>
                              </div>
                            )}

                            {params.architecture === 'EfficientAd' && (
                              <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                  <Label>Model Size</Label>
                                  <Select
                                    value={params.adModelSize}
                                    onValueChange={(value) => setParam('adModelSize', value)}
                                  >
                                    <SelectTrigger>
                                      <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                      {ANOMALY_MODEL_SIZES.map((size) => (
                                        <SelectItem key={size} value={size}>{size}</SelectItem>
                                      ))}
                                    </SelectContent>
                                  </Select>
                                </div>
                                <div className="space-y-2">
                                  <Label htmlFor="adLr">Learning Rate</Label>
                                  <Input
                                    id="adLr"
                                    type="number"
                                    step={0.00001}
                                    value={params.adLr}
                                    onChange={(e) => setParam('adLr', Number(e.target.value))}
                                  />
                                  <p className="text-xs text-muted-foreground">
                                    Lives here, not in the training config: anomalib takes it as a
                                    model constructor argument.
                                  </p>
                                </div>
                              </div>
                            )}
                          </>
                        ) : isTorchDet ? (
                          <>
                            {/* torchvision builds the full detector from
                                architecture + backbone, so there is no neck/head
                                to choose (and none is emitted). */}
                            <div className="space-y-2">
                              <Label>Architecture</Label>
                              <Select
                                value={params.architecture}
                                onValueChange={applyTorchDetArchitecture}
                              >
                                <SelectTrigger>
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  {TORCH_DET_PRESET_KEYS.map((key) => (
                                    <SelectItem key={key} value={key}>
                                      {TORCH_DET_PRESETS[key].label}
                                    </SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                            </div>
                            <div className="space-y-2">
                              <Label>Backbone</Label>
                              <Select
                                value={params.backbone}
                                onValueChange={(value) => setParam('backbone', value)}
                              >
                                <SelectTrigger>
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  {torchDetPreset.backbones.map((b) => (
                                    <SelectItem key={b} value={b}>{b}</SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                              <p className="text-xs text-muted-foreground">
                                Builds <code>{torchDetPreset.architecture.toLowerCase()}_
                                {params.backbone.toLowerCase().replace(/-/g, '_')}</code> from
                                torchvision. No neck/head blocks are emitted.
                              </p>
                            </div>
                            <div className="space-y-2">
                              <Label>Initial Weights</Label>
                              <Select
                                value={
                                  TORCH_PRETRAIN_OPTIONS.some((o) => o.value === params.pretrainWeights)
                                    ? params.pretrainWeights
                                    : 'custom'
                                }
                                onValueChange={(value) =>
                                  setParam('pretrainWeights', value === 'custom' ? params.pretrainWeights : value)
                                }
                              >
                                <SelectTrigger>
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  {TORCH_PRETRAIN_OPTIONS.map((o) => (
                                    <SelectItem key={o.value || 'none'} value={o.value}>
                                      {o.label}
                                    </SelectItem>
                                  ))}
                                  <SelectItem value="custom">Checkpoint path…</SelectItem>
                                </SelectContent>
                              </Select>
                              <p className="text-xs text-muted-foreground">
                                {torchDetPreset.supportsCocoTransfer
                                  ? 'COCO loads a pretrained detector and replaces its classifier to match num_classes — usually the single biggest accuracy win on small datasets.'
                                  : 'SSD does not support COCO head replacement; a COCO choice falls back to ImageNet backbone weights.'}
                              </p>
                            </div>
                          </>
                        ) : (
                          <>
                            <div className="space-y-2">
                              <Label>Model Family</Label>
                              <Select value={detectionPreset} onValueChange={applyDetectionPreset}>
                                <SelectTrigger>
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  {DETECTION_PRESET_KEYS.map((key) => (
                                    <SelectItem key={key} value={key}>
                                      {DETECTION_PRESETS[key].label}
                                    </SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                              <p className="text-xs text-muted-foreground">
                                Sets <code>architecture: {preset.architecture}</code> and wires the head
                                under <code>{preset.headKey}</code>.
                              </p>
                            </div>
                            <div className="grid grid-cols-3 gap-4">
                              <div className="space-y-2">
                                <Label>Backbone</Label>
                                <Select
                                  value={params.backbone}
                                  onValueChange={(value) => setParam('backbone', value)}
                                >
                                  <SelectTrigger>
                                    <SelectValue />
                                  </SelectTrigger>
                                  <SelectContent>
                                    {preset.backbones.map((b) => (
                                      <SelectItem key={b} value={b}>
                                        {b}
                                      </SelectItem>
                                    ))}
                                  </SelectContent>
                                </Select>
                              </div>
                              <div className="space-y-2">
                                <Label>Neck</Label>
                                <Select
                                  value={params.neck}
                                  onValueChange={(value) => setParam('neck', value)}
                                >
                                  <SelectTrigger>
                                    <SelectValue />
                                  </SelectTrigger>
                                  <SelectContent>
                                    {preset.necks.map((n) => (
                                      <SelectItem key={n} value={n}>
                                        {n}
                                      </SelectItem>
                                    ))}
                                  </SelectContent>
                                </Select>
                              </div>
                              <div className="space-y-2">
                                <Label>Head</Label>
                                <Select
                                  value={params.head}
                                  onValueChange={(value) => setParam('head', value)}
                                >
                                  <SelectTrigger>
                                    <SelectValue />
                                  </SelectTrigger>
                                  <SelectContent>
                                    {preset.heads.map((h) => (
                                      <SelectItem key={h} value={h}>
                                        {h}
                                      </SelectItem>
                                    ))}
                                  </SelectContent>
                                </Select>
                              </div>
                            </div>
                          </>
                        )}

                        <div className="space-y-2">
                          <Label htmlFor="pretrainWeights">
                            {isSeg ? 'Pretrained Backbone Weights' : 'Pretrained Weights'}
                          </Label>
                          <Input
                            id="pretrainWeights"
                            value={params.pretrainWeights}
                            onChange={(e) => setParam('pretrainWeights', e.target.value)}
                            placeholder="https://paddledet.bj.bcebos.com/models/..."
                          />
                        </div>
                      </TabsContent>

                      <TabsContent value="advanced" className="space-y-4 mt-4">
                        {isSeg ? (
                          <div className="flex items-center justify-between rounded-lg border p-3">
                            <div>
                              <Label htmlFor="alignCorners">align_corners</Label>
                              <p className="text-xs text-muted-foreground mt-1">
                                Enable when your label masks were produced with corner-aligned
                                interpolation.
                              </p>
                            </div>
                            <Switch
                              id="alignCorners"
                              checked={params.segAlignCorners}
                              onCheckedChange={(v) => setParam('segAlignCorners', v)}
                            />
                          </div>
                        ) : isClas ? (
                          <p className="text-sm text-muted-foreground">
                            PaddleClas model configs carry no additional structural options here. Use
                            the YAML editor for architecture-specific settings.
                          </p>
                        ) : isAnomalyModel ? (
                          <div className="space-y-3">
                            <p className="text-sm text-muted-foreground">
                              Every remaining knob for this algorithm is on the Architecture tab.
                              Normalization, EMA and depth/width multipliers are PaddleDetection
                              concepts and are not emitted.
                            </p>
                            {anomalyPresetInfo.downloadsAssets && (
                              <p className="text-sm text-amber-600">
                                First run downloads {anomalyPresetInfo.downloadsAssets}. Pre-warm the
                                cache on machines without internet access.
                              </p>
                            )}
                            {!anomalyPresetInfo.hasLoss && (
                              <p className="text-sm text-muted-foreground">
                                This algorithm does no gradient descent, so the training loss curve
                                stays flat at zero — that is expected, not a stalled run. Progress is
                                reported as memory-bank fill.
                              </p>
                            )}
                          </div>
                        ) : isTorchDet ? (
                          <div className="space-y-3">
                            <p className="text-sm text-muted-foreground">
                              TorchDet models have no additional structural options: normalization,
                              EMA and depth/width multipliers are PaddleDetection concepts with no
                              torchvision equivalent, so they are not emitted.
                            </p>
                            <div className="space-y-2">
                              <Label htmlFor="torchWeightsPath">Checkpoint Path (optional)</Label>
                              <Input
                                id="torchWeightsPath"
                                value={params.pretrainWeights}
                                onChange={(e) => setParam('pretrainWeights', e.target.value)}
                                placeholder="COCO, ImageNet, or H:\\...\\best_model\\model.pt"
                              />
                              <p className="text-xs text-muted-foreground">
                                Written as <code>pretrain_weights</code>. A <code>.pt</code> path
                                resumes from a checkpoint produced by this platform.
                              </p>
                            </div>
                          </div>
                        ) : (
                          <>
                            <div className="grid grid-cols-2 gap-4">
                              <div className="space-y-2">
                                <Label>Normalization Type</Label>
                                <Select
                                  value={params.normType}
                                  onValueChange={(value) => setParam('normType', value)}
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
                                  checked={params.useEma}
                                  onCheckedChange={(checked) => setParam('useEma', checked)}
                                />
                              </div>
                            </div>
                            {params.useEma && (
                              <div className="space-y-2">
                                <Label htmlFor="emaDecay">EMA Decay</Label>
                                <Input
                                  id="emaDecay"
                                  type="number"
                                  step={0.0001}
                                  value={params.emaDecay}
                                  onChange={(e) => setParam('emaDecay', Number(e.target.value))}
                                />
                              </div>
                            )}
                            {preset.usesMultipliers ? (
                              <>
                                <div className="space-y-2">
                                  <Label>Depth Multiplier: {params.depthMult.toFixed(2)}</Label>
                                  <Slider
                                    value={[params.depthMult * 100]}
                                    onValueChange={(value) => setParam('depthMult', value[0] / 100)}
                                    min={10}
                                    max={200}
                                    step={1}
                                  />
                                </div>
                                <div className="space-y-2">
                                  <Label>Width Multiplier: {params.widthMult.toFixed(2)}</Label>
                                  <Slider
                                    value={[params.widthMult * 100]}
                                    onValueChange={(value) => setParam('widthMult', value[0] / 100)}
                                    min={10}
                                    max={200}
                                    step={1}
                                  />
                                </div>
                              </>
                            ) : (
                              <p className="text-sm text-muted-foreground">
                                {preset.label} does not use depth/width multipliers.
                              </p>
                            )}
                          </>
                        )}
                      </TabsContent>
                    </Tabs>
                  </div>

                  {/* ---------------- YAML side ---------------- */}
                  <div className="min-w-0">
                    <ConfigYamlPane
                      yaml={effectiveYaml}
                      manual={manualYaml}
                      onManualYamlChange={setManualYamlText}
                      onEnterManual={() => {
                        setManualYamlText(generatedYaml)
                        setManualYaml(true)
                      }}
                      onLeaveManual={() => setManualYaml(false)}
                      issues={issues}
                      rows={30}
                    />
                  </div>
                </div>

                <DialogFooter className="mt-6">
                  <Button type="button" variant="outline" onClick={() => setDialogOpen(false)}>
                    Cancel
                  </Button>
                  <Button type="submit" disabled={saving || !!syntaxError || blockingIssue}>
                    {saving && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                    {editingModel ? 'Save Changes' : 'Create Model'}
                  </Button>
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
                <Select value={filterProjectId} onValueChange={(value) => setFilterProjectId(value)}>
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
                {!searchQuery && (
                  <Button className="mt-4" onClick={openCreateDialog}>
                    <Plus className="w-4 h-4 mr-2" />
                    New Model
                  </Button>
                )}
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
                        <CardDescription>{model.description || 'No description'}</CardDescription>
                      </div>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
                          <Button variant="ghost" size="icon">
                            <MoreHorizontal className="w-4 h-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem
                            onClick={(e) => {
                              e.stopPropagation()
                              openEditDialog(model)
                            }}
                          >
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
                      {model.backbone && <span>{model.backbone}</span>}
                      <span>•</span>
                      <span>{model.numClasses} classes</span>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>

        {/* Model Details — shows the stored config verbatim. */}
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
                selectedModel.yamlConfig ? (
                  <div className="space-y-4">
                    <pre className="p-4 rounded-lg bg-muted/50 text-xs overflow-auto max-h-[400px] font-mono whitespace-pre">
                      {selectedModel.yamlConfig}
                    </pre>
                    <div className="flex gap-2">
                      <Button variant="outline" className="flex-1" onClick={copyYamlToClipboard}>
                        <Copy className="w-4 h-4 mr-2" />
                        Copy
                      </Button>
                      <Button
                        variant="outline"
                        className="flex-1"
                        onClick={() => openEditDialog(selectedModel)}
                      >
                        <Pencil className="w-4 h-4 mr-2" />
                        Edit
                      </Button>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-8 text-center">
                    <Settings2 className="w-10 h-10 text-muted-foreground mb-3" />
                    <p className="text-sm text-muted-foreground">
                      This model has no stored YAML. Edit it to generate one.
                    </p>
                    <Button className="mt-4" variant="outline" onClick={() => openEditDialog(selectedModel)}>
                      <Pencil className="w-4 h-4 mr-2" />
                      Edit
                    </Button>
                  </div>
                )
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
