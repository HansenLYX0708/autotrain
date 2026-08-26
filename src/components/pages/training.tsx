'use client'

import { useEffect, useMemo, useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Switch } from '@/components/ui/switch'
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
import {
  Plus,
  Search,
  MoreHorizontal,
  Pencil,
  Trash2,
  Settings2,
  Copy,
  Upload,
  FileText,
  Loader2,
  Code,
  Filter,
  X,
} from 'lucide-react'
import { toast } from '@/hooks/use-toast'
import { ConfigYamlPane, yamlSyntaxError } from '@/components/config-yaml-pane'
import {
  asConfigFramework,
  countsIterations,
  defaultTrainingParams,
  generateTrainingYaml,
  parseTrainingParams,
  supportsField,
  ANOMALY_BEST_METRICS,
  OPTIMIZER_OPTIONS,
  SCHEDULER_OPTIONS,
  TRAINING_FIELD_SUPPORT,
  type ConfigFramework,
  type TrainingParams,
} from '@/lib/training-yaml'

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
  iters: number | null
  saveInterval: number | null
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
  /** Authoritative config content. Everything above is a display cache. */
  yamlConfig: string | null
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

/** Comma-separated numeric list <-> number[] for the free-text list inputs. */
function parseNumberList(value: string): number[] {
  return value
    .split(',')
    .map((s) => Number(s.trim()))
    .filter((n) => Number.isFinite(n))
}

export function TrainingPage() {
  const [projects, setProjects] = useState<Project[]>([])
  const [configs, setConfigs] = useState<TrainingConfig[]>([])
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [dialogOpen, setDialogOpen] = useState(false)
  const [importDialogOpen, setImportDialogOpen] = useState(false)
  const [importing, setImporting] = useState(false)
  const [filterProjectId, setFilterProjectId] = useState<string>('__all__')
  const [selectedConfig, setSelectedConfig] = useState<TrainingConfig | null>(null)
  const [defaultConfigs, setDefaultConfigs] = useState<ConfigFile[]>([])
  const [userConfigs, setUserConfigs] = useState<ConfigFile[]>([])

  // ---- Create / edit dialog state ----------------------------------------
  const [editingConfig, setEditingConfig] = useState<TrainingConfig | null>(null)
  const [configName, setConfigName] = useState('')
  const [configProjectId, setConfigProjectId] = useState('')
  const [params, setParams] = useState<TrainingParams>(defaultTrainingParams('PaddleDetection'))
  /** When true the YAML is hand-written and the form no longer drives it. */
  const [manualYaml, setManualYaml] = useState(false)
  const [manualYamlText, setManualYamlText] = useState('')

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

  // ---- Derived dialog values ---------------------------------------------

  const frameworkOf = (projectId: string): ConfigFramework =>
    asConfigFramework(projects.find((p) => p.id === projectId)?.framework)

  const dialogFramework = frameworkOf(configProjectId)
  const isSeg = countsIterations(dialogFramework)
  // Several controls are shared with segmentation but need different wording:
  // for an anomaly run `iters` is `trainer.max_steps` and there is no optimizer.
  const isAnomalyConfig = dialogFramework === 'TorchAnomaly'
  const shows = (field: Parameters<typeof supportsField>[1]) => supportsField(dialogFramework, field)

  const generatedYaml = useMemo(
    () => generateTrainingYaml(dialogFramework, params, configName || 'Training Config'),
    [dialogFramework, params, configName],
  )
  const effectiveYaml = manualYaml ? manualYamlText : generatedYaml
  const syntaxError = yamlSyntaxError(effectiveYaml)

  const setParam = <K extends keyof TrainingParams>(key: K, value: TrainingParams[K]) =>
    setParams((prev) => ({ ...prev, [key]: value }))

  /**
   * Switching project can switch framework. Rebuild on the new framework's
   * defaults but carry over every knob both frameworks understand, so changing
   * project does not silently discard a learning rate the user just typed.
   */
  const handleProjectChange = (projectId: string) => {
    const nextFramework = frameworkOf(projectId)
    setConfigProjectId(projectId)
    if (nextFramework === dialogFramework) return

    const next = defaultTrainingParams(nextFramework)
    for (const field of TRAINING_FIELD_SUPPORT[nextFramework]) {
      if (TRAINING_FIELD_SUPPORT[dialogFramework].has(field)) {
        Object.assign(next, { [field]: params[field] })
      }
    }
    setParams(next)
  }

  const resetDialog = () => {
    setEditingConfig(null)
    setConfigName('')
    setConfigProjectId('')
    setParams(defaultTrainingParams('PaddleDetection'))
    setManualYaml(false)
    setManualYamlText('')
  }

  const openCreateDialog = () => {
    resetDialog()
    setDialogOpen(true)
  }

  /**
   * Editing loads the stored YAML (the source of truth) and back-fills the form
   * from it. The config starts in manual mode so that opening an imported or
   * hand-written config and pressing Save cannot silently replace it with a
   * regenerated template — which is exactly how the old model editor destroyed
   * custom YAML.
   */
  const openEditDialog = (config: TrainingConfig) => {
    const framework = asConfigFramework(config.project?.framework)
    const parsed = parseTrainingParams(framework, config.yamlConfig)
    setEditingConfig(config)
    setConfigName(config.name)
    setConfigProjectId(config.projectId)
    setParams({ ...defaultTrainingParams(framework), ...parsed })
    if (config.yamlConfig) {
      setManualYaml(true)
      setManualYamlText(config.yamlConfig)
    } else {
      setManualYaml(false)
      setManualYamlText('')
    }
    setDialogOpen(true)
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
      const sourceConfigs = importForm.configSource === 'default' ? defaultConfigs : userConfigs
      const selectedConfigData = sourceConfigs.find((c) => c.name === importForm.selectedConfig)

      const response = await fetch('/api/training-configs/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          projectId: importForm.projectId,
          name: importForm.name,
          yamlContent:
            importForm.configSource === 'custom' ? importForm.customYaml : selectedConfigData?.content,
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

    if (!configName.trim() || !configProjectId) {
      toast({
        title: 'Missing required fields',
        description: 'A config name and project are required.',
        variant: 'destructive',
      })
      return
    }
    if (syntaxError) {
      toast({ title: 'Invalid YAML', description: syntaxError, variant: 'destructive' })
      return
    }

    setSaving(true)
    try {
      const body = {
        projectId: configProjectId,
        name: configName.trim(),
        yamlContent: effectiveYaml,
      }

      const response = editingConfig
        ? await fetch(`/api/training-configs/${editingConfig.id}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: body.name, yamlConfig: effectiveYaml }),
          })
        : await fetch('/api/training-configs/import', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ...body, isDefault: false, configPath: null }),
          })

      const result = await response.json().catch(() => ({}))
      if (!response.ok || result.success === false) {
        throw new Error(result.error || result.message || 'Failed to save config')
      }

      toast({
        title: editingConfig ? 'Config updated' : 'Config created',
        description: result?.data?.configPath ? `Saved to ${result.data.configPath}` : undefined,
      })
      await fetchConfigs()
      if (editingConfig && selectedConfig?.id === editingConfig.id) {
        setSelectedConfig(null)
      }
      setDialogOpen(false)
      resetDialog()
    } catch (error) {
      toast({
        title: 'Error saving config',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      })
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this config?')) {
      return
    }

    try {
      const response = await fetch(`/api/training-configs/${id}`, { method: 'DELETE' })
      const result = await response.json().catch(() => ({}))
      if (!response.ok) {
        throw new Error(result.error || 'Failed to delete config')
      }
      toast({ title: 'Config deleted successfully' })
      fetchConfigs()
      if (selectedConfig?.id === id) {
        setSelectedConfig(null)
      }
    } catch (error) {
      toast({
        title: 'Error deleting config',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      })
    }
  }

  const copyYamlToClipboard = () => {
    if (!selectedConfig?.yamlConfig) return
    navigator.clipboard.writeText(selectedConfig.yamlConfig)
    toast({ title: 'YAML copied to clipboard' })
  }

  const filteredConfigs = configs.filter((config) => {
    const matchesSearch = config.name.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesProject = filterProjectId === '__all__' || config.projectId === filterProjectId
    return matchesSearch && matchesProject
  })

  const resetFilter = () => {
    setFilterProjectId('__all__')
    setSearchQuery('')
  }

  const hasActiveFilters = filterProjectId !== '__all__' || searchQuery !== ''

  /** Short summary line for a config card, in the unit its framework uses. */
  const configSummary = (config: TrainingConfig) => {
    const framework = asConfigFramework(config.project?.framework)
    return countsIterations(framework)
      ? `${config.iters ?? config.epoch} iters`
      : `${config.epoch} epochs`
  }

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
          <Dialog
            open={importDialogOpen}
            onOpenChange={(open) => {
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
            }}
          >
            <DialogTrigger asChild>
              <Button variant="outline">
                <Upload className="w-4 h-4 mr-2" />
                Import Config
              </Button>
            </DialogTrigger>
            {/* `sm:` prefix required to override DialogContent's own sm:max-w-lg. */}
            <DialogContent className="sm:max-w-2xl max-h-[90vh] overflow-y-auto">
              <DialogHeader>
                <DialogTitle>Import Training Configuration</DialogTitle>
                <DialogDescription>
                  Import a preset training config for this project&apos;s framework, one of your own
                  saved configs, or paste custom YAML.
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
                              ? 'Select a preset config'
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
                New Config
              </Button>
            </DialogTrigger>
            {/* The `sm:` prefix is required: DialogContent's own base class list
                ends with `sm:max-w-lg`, and an unprefixed `max-w-*` here is not
                treated as a conflict by tailwind-merge, so both ship and the
                responsive one wins above 640px — pinning the dialog to 32rem. */}
            <DialogContent className="sm:max-w-[min(1500px,94vw)] max-h-[92vh] overflow-y-auto">
              <DialogHeader>
                <DialogTitle>
                  {editingConfig ? `Edit "${editingConfig.name}"` : 'Create Training Configuration'}
                </DialogTitle>
                <DialogDescription>
                  {configProjectId
                    ? `Framework: ${dialogFramework}. Only parameters this framework honours are shown.`
                    : 'Select a project first — the available parameters depend on its framework.'}
                </DialogDescription>
              </DialogHeader>

              <form onSubmit={handleSubmit}>
                {/* minmax(0,…) rather than a bare 1fr: grid tracks default to
                    min-content, so the nested two-column field grids would
                    otherwise push the columns wider than the dialog. */}
                <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
                  {/* ---------------- Form side ---------------- */}
                  <div className="space-y-4 min-w-0">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="name">Config Name</Label>
                        <Input
                          id="name"
                          value={configName}
                          onChange={(e) => setConfigName(e.target.value)}
                          placeholder="My Training Config"
                          required
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="project">Project</Label>
                        <Select
                          value={configProjectId}
                          onValueChange={handleProjectChange}
                          disabled={!!editingConfig}
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="Select project" />
                          </SelectTrigger>
                          <SelectContent>
                            {projects.map((project) => (
                              <SelectItem key={project.id} value={project.id}>
                                {project.name} · {project.framework}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </div>

                    <Tabs defaultValue="schedule" className="w-full">
                      <TabsList className={`grid w-full ${isAnomalyConfig ? 'grid-cols-3' : 'grid-cols-4'}`}>
                        <TabsTrigger value="schedule">Schedule</TabsTrigger>
                        {!isAnomalyConfig && <TabsTrigger value="optimizer">Optimizer</TabsTrigger>}
                        <TabsTrigger value="data">Data</TabsTrigger>
                        <TabsTrigger value="runtime">Runtime</TabsTrigger>
                      </TabsList>

                      {/* ---- Schedule ---- */}
                      <TabsContent value="schedule" className="space-y-4 mt-4">
                        <div className="grid grid-cols-2 gap-4">
                          {shows('iters') && (
                            <div className="space-y-2">
                              <Label htmlFor="iters">{isAnomalyConfig ? 'Max steps' : 'Iterations'}</Label>
                              <Input
                                id="iters"
                                type="number"
                                min={1}
                                step={100}
                                value={params.iters}
                                onChange={(e) => setParam('iters', Number(e.target.value) || 1)}
                              />
                              <p className="text-xs text-muted-foreground">
                                {isAnomalyConfig
                                  ? 'trainer.max_steps — the real training length. Ignored by PatchCore and PaDiM, which do a single pass over the normal images.'
                                  : 'PaddleSeg trains by iteration, not epoch.'}
                              </p>
                            </div>
                          )}
                          {shows('epochs') && (
                            <div className="space-y-2">
                              <Label htmlFor="epochs">{isAnomalyConfig ? 'Max epochs (ceiling)' : 'Epochs'}</Label>
                              <Input
                                id="epochs"
                                type="number"
                                min={1}
                                value={params.epochs}
                                onChange={(e) => setParam('epochs', Number(e.target.value) || 1)}
                              />
                              {isAnomalyConfig && (
                                <p className="text-xs text-muted-foreground">
                                  Whichever limit is reached first stops training. Memory-bank models
                                  override this with 1.
                                </p>
                              )}
                            </div>
                          )}
                          {shows('saveInterval') && (
                            <div className="space-y-2">
                              <Label htmlFor="saveInterval">Save Interval (iters)</Label>
                              <Input
                                id="saveInterval"
                                type="number"
                                min={1}
                                step={50}
                                value={params.saveInterval}
                                onChange={(e) => setParam('saveInterval', Number(e.target.value) || 1)}
                              />
                            </div>
                          )}
                          {shows('adValInterval') && (
                            <div className="space-y-2">
                              <Label htmlFor="adValInterval">Validate every N steps</Label>
                              <Input
                                id="adValInterval"
                                type="number"
                                min={1}
                                step={50}
                                value={params.adValInterval}
                                onChange={(e) => setParam('adValInterval', Number(e.target.value) || 1)}
                              />
                              <p className="text-xs text-muted-foreground">
                                Each validation emits one metrics row and may update the best
                                checkpoint. The adapter converts this into safe Lightning
                                arguments, including for single-epoch models.
                              </p>
                            </div>
                          )}
                          {shows('adBestMetric') && (
                            <div className="space-y-2">
                              <Label htmlFor="adBestMetric">Best-checkpoint metric</Label>
                              <Select
                                value={params.adBestMetric}
                                onValueChange={(v) => setParam('adBestMetric', v)}
                              >
                                <SelectTrigger id="adBestMetric">
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  {ANOMALY_BEST_METRICS.map((metric) => (
                                    <SelectItem key={metric} value={metric}>{metric}</SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                              <p className="text-xs text-muted-foreground">
                                Pixel metrics require masks for the defect images; without them the
                                run falls back to image_AUROC.
                              </p>
                            </div>
                          )}
                          {shows('snapshotEpoch') && (
                            <div className="space-y-2">
                              <Label htmlFor="snapshotEpoch">Snapshot Every (epochs)</Label>
                              <Input
                                id="snapshotEpoch"
                                type="number"
                                min={1}
                                value={params.snapshotEpoch}
                                onChange={(e) => setParam('snapshotEpoch', Number(e.target.value) || 1)}
                              />
                            </div>
                          )}
                        </div>

                        {/* No LR schedule for anomaly detection: each anomalib
                            model builds its own optimizer and scheduler inside
                            `configure_optimizers`, and half of them do no
                            gradient descent at all. An empty dropdown here would
                            imply a choice that does not exist. */}
                        {!isAnomalyConfig && (
                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-2">
                            <Label>LR Scheduler</Label>
                            <Select
                              value={params.scheduler}
                              onValueChange={(value) => setParam('scheduler', value)}
                            >
                              <SelectTrigger>
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                {SCHEDULER_OPTIONS[dialogFramework].map((s) => (
                                  <SelectItem key={s} value={s}>
                                    {s}
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          </div>
                          {shows('maxEpochs') && (
                            <div className="space-y-2">
                              <Label htmlFor="maxEpochs">Decay Max Epochs</Label>
                              <Input
                                id="maxEpochs"
                                type="number"
                                min={1}
                                value={params.maxEpochs}
                                onChange={(e) => setParam('maxEpochs', Number(e.target.value) || 1)}
                              />
                            </div>
                          )}
                          {shows('power') && (
                            <div className="space-y-2">
                              <Label htmlFor="power">Poly Power</Label>
                              <Input
                                id="power"
                                type="number"
                                step={0.05}
                                value={params.power}
                                onChange={(e) => setParam('power', Number(e.target.value))}
                              />
                            </div>
                          )}
                          {shows('endLr') && (
                            <div className="space-y-2">
                              <Label htmlFor="endLr">End LR</Label>
                              <Input
                                id="endLr"
                                type="number"
                                step={0.0001}
                                value={params.endLr}
                                onChange={(e) => setParam('endLr', Number(e.target.value))}
                              />
                            </div>
                          )}
                          {shows('gamma') && (
                            <div className="space-y-2">
                              <Label htmlFor="gamma">Decay Gamma</Label>
                              <Input
                                id="gamma"
                                type="number"
                                step={0.01}
                                value={params.gamma}
                                onChange={(e) => setParam('gamma', Number(e.target.value))}
                              />
                            </div>
                          )}
                          {shows('milestones') && (
                            <div className="space-y-2">
                              <Label htmlFor="milestones">Milestones (epochs)</Label>
                              <Input
                                id="milestones"
                                value={params.milestones.join(', ')}
                                onChange={(e) => setParam('milestones', parseNumberList(e.target.value))}
                                placeholder="60, 72"
                              />
                            </div>
                          )}
                        </div>
                        )}

                        {!isAnomalyConfig && (
                        <div className="grid grid-cols-2 gap-4">
                          {shows('warmupEpochs') && (
                            <div className="space-y-2">
                              <Label htmlFor="warmupEpochs">Warmup Epochs (0 = off)</Label>
                              <Input
                                id="warmupEpochs"
                                type="number"
                                min={0}
                                value={params.warmupEpochs}
                                onChange={(e) => setParam('warmupEpochs', Number(e.target.value) || 0)}
                              />
                            </div>
                          )}
                          {shows('warmupIters') && (
                            <div className="space-y-2">
                              <Label htmlFor="warmupIters">Warmup Iters (0 = off)</Label>
                              <Input
                                id="warmupIters"
                                type="number"
                                min={0}
                                value={params.warmupIters}
                                onChange={(e) => setParam('warmupIters', Number(e.target.value) || 0)}
                              />
                            </div>
                          )}
                          <div className="space-y-2">
                            <Label htmlFor="warmupStartLr">Warmup Start Factor / LR</Label>
                            <Input
                              id="warmupStartLr"
                              type="number"
                              step={0.0001}
                              value={params.warmupStartLr}
                              onChange={(e) => setParam('warmupStartLr', Number(e.target.value))}
                            />
                          </div>
                        </div>
                        )}

                        {isAnomalyConfig && (
                          <p className="text-xs text-muted-foreground">
                            The optimizer and learning rate are not configured here: in anomalib they
                            are constructor arguments of the model, so they live in the model config.
                            PatchCore and PaDiM have no optimizer at all.
                          </p>
                        )}
                      </TabsContent>

                      {/* ---- Optimizer ---- */}
                      {!isAnomalyConfig && (
                      <TabsContent value="optimizer" className="space-y-4 mt-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-2">
                            <Label>Optimizer</Label>
                            <Select
                              value={params.optimizerType}
                              onValueChange={(value) => setParam('optimizerType', value)}
                            >
                              <SelectTrigger>
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                {OPTIMIZER_OPTIONS[dialogFramework].map((o) => (
                                  <SelectItem key={o} value={o}>
                                    {o}
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          </div>
                          <div className="space-y-2">
                            <Label htmlFor="baseLr">Base Learning Rate</Label>
                            <Input
                              id="baseLr"
                              type="number"
                              step={0.0001}
                              value={params.baseLr}
                              onChange={(e) => setParam('baseLr', Number(e.target.value))}
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
                              value={params.momentum}
                              onChange={(e) => setParam('momentum', Number(e.target.value))}
                              disabled={!['Momentum', 'SGD'].includes(params.optimizerType)}
                            />
                          </div>
                          <div className="space-y-2">
                            <Label htmlFor="weightDecay">Weight Decay</Label>
                            <Input
                              id="weightDecay"
                              type="number"
                              step={0.00001}
                              value={params.weightDecay}
                              onChange={(e) => setParam('weightDecay', Number(e.target.value))}
                            />
                          </div>
                          {shows('regularizerType') && (
                            <div className="space-y-2">
                              <Label>Regularizer</Label>
                              <Select
                                value={params.regularizerType}
                                onValueChange={(value) => setParam('regularizerType', value)}
                              >
                                <SelectTrigger>
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="L2">L2</SelectItem>
                                  <SelectItem value="L1">L1</SelectItem>
                                </SelectContent>
                              </Select>
                            </div>
                          )}
                          {shows('clipGradByNorm') && (
                            <div className="space-y-2">
                              <Label htmlFor="clipGrad">Clip Grad By Norm (empty = off)</Label>
                              <Input
                                id="clipGrad"
                                type="number"
                                step={1}
                                value={params.clipGradByNorm ?? ''}
                                onChange={(e) =>
                                  setParam(
                                    'clipGradByNorm',
                                    e.target.value === '' ? null : Number(e.target.value),
                                  )
                                }
                              />
                            </div>
                          )}
                        </div>
                      </TabsContent>
                      )}

                      {/* ---- Data ---- */}
                      <TabsContent value="data" className="space-y-4 mt-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-2">
                            <Label htmlFor="trainBatchSize">Train Batch Size</Label>
                            <Input
                              id="trainBatchSize"
                              type="number"
                              min={1}
                              value={params.trainBatchSize}
                              onChange={(e) => setParam('trainBatchSize', Number(e.target.value) || 1)}
                            />
                          </div>
                          {shows('evalBatchSize') && (
                            <div className="space-y-2">
                              <Label htmlFor="evalBatchSize">Eval Batch Size</Label>
                              <Input
                                id="evalBatchSize"
                                type="number"
                                min={1}
                                value={params.evalBatchSize}
                                onChange={(e) => setParam('evalBatchSize', Number(e.target.value) || 1)}
                              />
                            </div>
                          )}
                          <div className="space-y-2">
                            <Label htmlFor="workerNum">Dataloader Workers</Label>
                            <Input
                              id="workerNum"
                              type="number"
                              min={0}
                              value={params.workerNum}
                              onChange={(e) => setParam('workerNum', Number(e.target.value) || 0)}
                            />
                          </div>
                          {/* Anomaly runs set the input size in the MODEL config:
                              anomalib resizes inside the model's PreProcessor and
                              each algorithm has its own recommended size (and, for
                              EfficientAD, a hard rule against normalising). */}
                          {shows('imageWidth') && (
                            <div className="space-y-2">
                              <Label htmlFor="imageWidth">Image Width</Label>
                              <Input
                                id="imageWidth"
                                type="number"
                                step={32}
                                min={32}
                                value={params.imageWidth}
                                onChange={(e) => setParam('imageWidth', Number(e.target.value) || 32)}
                              />
                            </div>
                          )}
                          {shows('imageHeight') && (
                            <div className="space-y-2">
                              <Label htmlFor="imageHeight">Image Height</Label>
                              <Input
                                id="imageHeight"
                                type="number"
                                step={32}
                                min={32}
                                value={params.imageHeight}
                                onChange={(e) => setParam('imageHeight', Number(e.target.value) || 32)}
                              />
                            </div>
                          )}
                        </div>
                        {isAnomalyConfig && (
                          <p className="text-xs text-muted-foreground">
                            Input size and normalisation belong to the model config for anomaly
                            detection — anomalib applies them inside the model&apos;s pre-processor.
                          </p>
                        )}

                        {shows('multiScaleTrain') && (
                          <div className="space-y-3 rounded-lg border p-3">
                            <div className="flex items-center justify-between">
                              <Label htmlFor="multiScale">Multi-scale training</Label>
                              <Switch
                                id="multiScale"
                                checked={params.multiScaleTrain}
                                onCheckedChange={(v) => setParam('multiScaleTrain', v)}
                              />
                            </div>
                            {params.multiScaleTrain && (
                              <div className="space-y-2">
                                <Label htmlFor="multiScaleSizes">Scales</Label>
                                <Input
                                  id="multiScaleSizes"
                                  value={params.multiScaleSizes.join(', ')}
                                  onChange={(e) =>
                                    setParam('multiScaleSizes', parseNumberList(e.target.value))
                                  }
                                  placeholder="320, 384, 448, 512, 576, 640"
                                />
                              </div>
                            )}
                          </div>
                        )}

                        {shows('augRandomFlip') && (
                          <div className="space-y-3 rounded-lg border p-3">
                            <Label>Augmentation</Label>
                            <div className="grid grid-cols-2 gap-3">
                              {(
                                [
                                  ['augRandomDistort', 'Random Distort'],
                                  ['augRandomExpand', 'Random Expand'],
                                  ['augRandomCrop', 'Random Crop'],
                                  ['augRandomFlip', 'Random Flip'],
                                ] as const
                              ).map(([key, label]) => (
                                <div key={key} className="flex items-center justify-between">
                                  <Label htmlFor={key} className="font-normal">
                                    {label}
                                  </Label>
                                  <Switch
                                    id={key}
                                    checked={params[key]}
                                    onCheckedChange={(v) => setParam(key, v)}
                                  />
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {shows('adTileEnabled') && (
                          <div className="space-y-3 rounded-lg border p-3">
                            <div className="flex items-center justify-between">
                              <div>
                                <Label htmlFor="adTiling">Input tiling</Label>
                                <p className="text-xs text-muted-foreground mt-1">
                                  Feed the model overlapping crops instead of a downscaled whole
                                  image — the only way small defects survive on a large picture.
                                  Supported by PatchCore, PaDiM and STFPM; <b>not</b> by EfficientAD.
                                </p>
                              </div>
                              <Switch
                                id="adTiling"
                                checked={params.adTileEnabled}
                                onCheckedChange={(v) => setParam('adTileEnabled', v)}
                              />
                            </div>
                            {params.adTileEnabled && (
                              <div className="grid grid-cols-2 gap-3 pt-2">
                                <div className="space-y-2">
                                  <Label htmlFor="adTileSize">Tile size</Label>
                                  <Input
                                    id="adTileSize"
                                    type="number"
                                    min={32}
                                    step={32}
                                    value={params.adTileSize}
                                    onChange={(e) => setParam('adTileSize', Number(e.target.value) || 32)}
                                  />
                                </div>
                                <div className="space-y-2">
                                  <Label htmlFor="adTileStride">Stride</Label>
                                  <Input
                                    id="adTileStride"
                                    type="number"
                                    min={1}
                                    step={16}
                                    value={params.adTileStride}
                                    onChange={(e) => setParam('adTileStride', Number(e.target.value) || 1)}
                                  />
                                  <p className="text-xs text-muted-foreground">
                                    A stride below the tile size overlaps tiles, which costs compute
                                    but avoids missing a defect that straddles a tile border.
                                  </p>
                                </div>
                              </div>
                            )}
                          </div>
                        )}

                        {shows('segOverrideTransforms') && (
                          <div className="space-y-3 rounded-lg border p-3">
                            <div className="flex items-center justify-between">
                              <div>
                                <Label htmlFor="segOverride">Override dataset transforms</Label>
                                <p className="text-xs text-muted-foreground mt-1">
                                  The dataset config normally owns the transform pipeline. Enable this
                                  to control image size and augmentation from here instead.
                                </p>
                              </div>
                              <Switch
                                id="segOverride"
                                checked={params.segOverrideTransforms}
                                onCheckedChange={(v) => setParam('segOverrideTransforms', v)}
                              />
                            </div>
                            {params.segOverrideTransforms && (
                              <div className="grid grid-cols-2 gap-3 pt-2">
                                {(
                                  [
                                    ['segAugFlipHorizontal', 'Horizontal Flip'],
                                    ['segAugFlipVertical', 'Vertical Flip'],
                                    ['segAugDistort', 'Random Distort'],
                                    ['segAugScaleAspect', 'Scale Jitter'],
                                    ['segAugBlur', 'Random Blur'],
                                  ] as const
                                ).map(([key, label]) => (
                                  <div key={key} className="flex items-center justify-between">
                                    <Label htmlFor={key} className="font-normal">
                                      {label}
                                    </Label>
                                    <Switch
                                      id={key}
                                      checked={params[key]}
                                      onCheckedChange={(v) => setParam(key, v)}
                                    />
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        )}

                        <div className="space-y-3 rounded-lg border p-3">
                          <div className="grid grid-cols-2 gap-4">
                            <div className="space-y-2">
                              <Label>Normalization</Label>
                              <Select
                                value={params.normalizeType}
                                onValueChange={(value) =>
                                  setParam('normalizeType', value as 'none' | 'mean_std')
                                }
                                disabled={isSeg}
                              >
                                <SelectTrigger>
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="mean_std">mean / std</SelectItem>
                                  <SelectItem value="none">none (raw 0-255)</SelectItem>
                                </SelectContent>
                              </Select>
                            </div>
                            <div className="space-y-2">
                              <Label htmlFor="normMean">Mean</Label>
                              <Input
                                id="normMean"
                                value={params.normMean.join(', ')}
                                onChange={(e) => setParam('normMean', parseNumberList(e.target.value))}
                              />
                            </div>
                            <div className="space-y-2">
                              <Label htmlFor="normStd">Std</Label>
                              <Input
                                id="normStd"
                                value={params.normStd.join(', ')}
                                onChange={(e) => setParam('normStd', parseNumberList(e.target.value))}
                              />
                            </div>
                          </div>
                        </div>
                      </TabsContent>

                      {/* ---- Runtime ---- */}
                      <TabsContent value="runtime" className="space-y-4 mt-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div className="flex items-center justify-between rounded-lg border p-3">
                            <Label htmlFor="useGpu">Use GPU</Label>
                            <Switch
                              id="useGpu"
                              checked={params.useGpu}
                              onCheckedChange={(v) => setParam('useGpu', v)}
                            />
                          </div>
                          {shows('useAmp') && (
                            <div className="flex items-center justify-between rounded-lg border p-3">
                              <Label htmlFor="useAmp">Mixed precision (AMP)</Label>
                              <Switch
                                id="useAmp"
                                checked={params.useAmp}
                                onCheckedChange={(v) => setParam('useAmp', v)}
                              />
                            </div>
                          )}
                        </div>
                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-2">
                            <Label htmlFor="logIter">Log Every N Iters</Label>
                            <Input
                              id="logIter"
                              type="number"
                              min={1}
                              value={params.logIter}
                              onChange={(e) => setParam('logIter', Number(e.target.value) || 1)}
                            />
                          </div>
                          {shows('saveDir') && (
                            <div className="space-y-2">
                              <Label htmlFor="saveDir">Save Dir (optional)</Label>
                              <Input
                                id="saveDir"
                                value={params.saveDir}
                                onChange={(e) => setParam('saveDir', e.target.value)}
                                placeholder="left blank = managed by the job"
                              />
                            </div>
                          )}
                          {shows('outputDir') && (
                            <div className="space-y-2">
                              <Label htmlFor="outputDir">Output Dir (optional)</Label>
                              <Input
                                id="outputDir"
                                value={params.outputDir}
                                onChange={(e) => setParam('outputDir', e.target.value)}
                              />
                            </div>
                          )}
                          {shows('weights') && (
                            <div className="space-y-2">
                              <Label htmlFor="weights">Weights (resume/eval)</Label>
                              <Input
                                id="weights"
                                value={params.weights}
                                onChange={(e) => setParam('weights', e.target.value)}
                                placeholder="output/model_final"
                              />
                            </div>
                          )}
                        </div>
                        {shows('pretrainWeights') && (
                          <div className="space-y-2">
                            <Label htmlFor="pretrainWeights">Pretrained Weights</Label>
                            <Input
                              id="pretrainWeights"
                              value={params.pretrainWeights}
                              onChange={(e) => setParam('pretrainWeights', e.target.value)}
                              placeholder="https://paddledet.bj.bcebos.com/models/pretrained/..."
                            />
                          </div>
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
                      rows={30}
                    />
                  </div>
                </div>

                <DialogFooter className="mt-6">
                  <Button type="button" variant="outline" onClick={() => setDialogOpen(false)}>
                    Cancel
                  </Button>
                  <Button type="submit" disabled={saving || !!syntaxError}>
                    {saving && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                    {editingConfig ? 'Save Changes' : 'Create Config'}
                  </Button>
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
                {!searchQuery && (
                  <Button className="mt-4" onClick={openCreateDialog}>
                    <Plus className="w-4 h-4 mr-2" />
                    New Config
                  </Button>
                )}
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
                          {config.project?.framework ?? 'PaddleDetection'} · {config.scheduler} scheduler
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
                            onClick={(e) => {
                              e.stopPropagation()
                              openEditDialog(config)
                            }}
                          >
                            <Pencil className="w-4 h-4 mr-2" />
                            Edit
                          </DropdownMenuItem>
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
                      <Badge variant="outline">{configSummary(config)}</Badge>
                      <span>lr: {config.baseLr}</span>
                      <span>•</span>
                      <span>batch: {config.batchSize}</span>
                      <span>•</span>
                      <span>
                        {config.evalWidth}×{config.evalHeight}
                      </span>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>

        {/* YAML Preview — shows the stored config verbatim. */}
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
                  {selectedConfig.yamlConfig ? (
                    <>
                      <pre className="p-4 rounded-lg bg-muted/50 text-xs overflow-auto max-h-[400px] font-mono whitespace-pre">
                        {selectedConfig.yamlConfig}
                      </pre>
                      <div className="flex gap-2">
                        <Button variant="outline" className="flex-1" onClick={copyYamlToClipboard}>
                          <Copy className="w-4 h-4 mr-2" />
                          Copy
                        </Button>
                        <Button
                          variant="outline"
                          className="flex-1"
                          onClick={() => openEditDialog(selectedConfig)}
                        >
                          <Pencil className="w-4 h-4 mr-2" />
                          Edit
                        </Button>
                      </div>
                    </>
                  ) : (
                    <div className="flex flex-col items-center justify-center py-8 text-center">
                      <Settings2 className="w-10 h-10 text-muted-foreground mb-3" />
                      <p className="text-sm text-muted-foreground">
                        This config has no stored YAML. Edit it to generate one.
                      </p>
                      <Button className="mt-4" variant="outline" onClick={() => openEditDialog(selectedConfig)}>
                        <Pencil className="w-4 h-4 mr-2" />
                        Edit
                      </Button>
                    </div>
                  )}
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
