'use client'

import { useEffect, useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
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
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import {
  Plus,
  Search,
  MoreHorizontal,
  Pencil,
  Trash2,
  Database,
  BarChart3,
  Image as ImageIcon,
  Tag,
  Download,
  FolderOpen,
  RefreshCw,
  Filter,
  X,
  FileJson,
} from 'lucide-react'
import { toast } from '@/hooks/use-toast'
import {
  ChartConfig,
  ChartContainer,
} from "@/components/ui/chart"

interface Project {
  id: string
  name: string
}

interface Dataset {
  id: string
  name: string
  description: string | null
  format: string
  projectId: string
  trainImagePath: string | null
  trainAnnoPath: string | null
  evalImagePath: string | null
  evalAnnoPath: string | null
  datasetDir: string | null
  numClasses: number
  numAnnotations: number
  numTrainImages: number
  numEvalImages: number
  classStats: string | null
  yamlConfig: string | null
  createdAt: string
  project?: {
    id: string
    name: string
  }
}

const chartConfig = {
  count: {
    label: "Annotations",
  },
  imageCount: {
    label: "Images",
  },
} satisfies ChartConfig

export function DatasetsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [projects, setProjects] = useState<Project[]>([])
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [dialogOpen, setDialogOpen] = useState(false)
  const [labelmeDialogOpen, setLabelmeDialogOpen] = useState(false)
  const [converting, setConverting] = useState(false)
  const [labelmeFormData, setLabelmeFormData] = useState({
    name: '',
    description: '',
    projectId: '',
    labelmeImagesPath: '',
    labelmeAnnotationsPath: '',
    outputDatasetDir: '',
    trainRatio: 0.7,
    valRatio: 0.2,
    testRatio: 0.1,
  })
  const [editingDataset, setEditingDataset] = useState<Dataset | null>(null)
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null)
  const [parsing, setParsing] = useState(false)
  const [filterProjectId, setFilterProjectId] = useState<string>('__all__')
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    projectId: '',
    format: 'COCO',
    numClasses: 1,
    trainImagePath: '',
    trainAnnoPath: '',
    evalImagePath: '',
    evalAnnoPath: '',
    datasetDir: '',
  })

  useEffect(() => {
    fetchDatasets()
    fetchProjects()
  }, [])

  const fetchDatasets = async () => {
    try {
      const response = await fetch('/api/datasets')
      if (response.ok) {
        const data = await response.json()
        setDatasets(data.data || data)
      }
    } catch (error) {
      console.error('Failed to fetch datasets:', error)
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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    try {
      if (editingDataset) {
        const response = await fetch(`/api/datasets/${editingDataset.id}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(formData),
        })
        if (response.ok) {
          toast({ title: 'Dataset updated successfully' })
          fetchDatasets()
          setDialogOpen(false)
          setEditingDataset(null)
        }
      } else {
        const response = await fetch('/api/datasets', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(formData),
        })
        if (response.ok) {
          toast({ title: 'Dataset created successfully' })
          fetchDatasets()
          setDialogOpen(false)
          setFormData({
            name: '',
            description: '',
            projectId: '',
            format: 'COCO',
            numClasses: 1,
            trainImagePath: '',
            trainAnnoPath: '',
            evalImagePath: '',
            evalAnnoPath: '',
            datasetDir: '',
          })
        }
      }
    } catch (error) {
      toast({ title: 'Error saving dataset', variant: 'destructive' })
    }
  }

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this dataset?')) {
      return
    }
    
    try {
      const response = await fetch(`/api/datasets/${id}`, {
        method: 'DELETE',
      })
      if (response.ok) {
        toast({ title: 'Dataset deleted successfully' })
        fetchDatasets()
        if (selectedDataset?.id === id) {
          setSelectedDataset(null)
        }
      }
    } catch (error) {
      toast({ title: 'Error deleting dataset', variant: 'destructive' })
    }
  }

  const openEditDialog = (dataset: Dataset) => {
    setEditingDataset(dataset)
    setFormData({
      name: dataset.name,
      description: dataset.description || '',
      projectId: dataset.projectId,
      format: dataset.format,
      numClasses: dataset.numClasses || 1,
      trainImagePath: dataset.trainImagePath || '',
      trainAnnoPath: dataset.trainAnnoPath || '',
      evalImagePath: dataset.evalImagePath || '',
      evalAnnoPath: dataset.evalAnnoPath || '',
      datasetDir: dataset.datasetDir || '',
    })
    setDialogOpen(true)
  }

  const getClassStats = (classStats: string | null) => {
    if (!classStats) return { train: [], eval: [] }
    try {
      const parsed = JSON.parse(classStats)
      // Support both old format (array) and new format (object with train/eval)
      if (Array.isArray(parsed)) {
        return { train: parsed, eval: [] }
      }
      return { train: parsed.train || [], eval: parsed.eval || [] }
    } catch {
      return { train: [], eval: [] }
    }
  }

  const handleParseDataset = async (datasetId: string) => {
    setParsing(true)
    try {
      const response = await fetch('/api/datasets/parse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ datasetId }),
      })
      const result = await response.json()
      
      if (result.success) {
        toast({ title: 'Dataset parsed successfully' })
        // Update the selected dataset with new stats
        setSelectedDataset(result.data)
        fetchDatasets()
      } else {
        toast({ 
          title: 'Failed to parse dataset', 
          description: result.error || 'Check if the annotation file exists',
          variant: 'destructive' 
        })
      }
    } catch (error) {
      toast({ title: 'Error parsing dataset', variant: 'destructive' })
    } finally {
      setParsing(false)
    }
  }

  const handleLabelmeConvert = async (e: React.FormEvent) => {
    e.preventDefault()
    
    // Validate ratios sum to 1
    const total = labelmeFormData.trainRatio + labelmeFormData.valRatio + labelmeFormData.testRatio
    if (Math.abs(total - 1.0) > 0.001) {
      toast({ 
        title: 'Invalid ratios', 
        description: `Train + Val + Test must equal 1.0 (current: ${total.toFixed(2)})`,
        variant: 'destructive' 
      })
      return
    }

    setConverting(true)
    try {
      const response = await fetch('/api/datasets/labelme-to-coco', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(labelmeFormData),
      })
      const result = await response.json()
      
      if (result.success) {
        toast({ 
          title: 'Conversion successful', 
          description: `Created dataset with ${result.stats.trainCount} train, ${result.stats.valCount} val, ${result.stats.testCount} test images` 
        })
        fetchDatasets()
        setLabelmeDialogOpen(false)
        setLabelmeFormData({
          name: '',
          description: '',
          projectId: '',
          labelmeImagesPath: '',
          labelmeAnnotationsPath: '',
          outputDatasetDir: '',
          trainRatio: 0.7,
          valRatio: 0.2,
          testRatio: 0.1,
        })
      } else {
        toast({ 
          title: 'Conversion failed', 
          description: result.error || 'Check the paths and try again',
          variant: 'destructive' 
        })
      }
    } catch (error) {
      toast({ title: 'Error converting dataset', variant: 'destructive' })
    } finally {
      setConverting(false)
    }
  }

  const filteredDatasets = datasets.filter(dataset => {
    const matchesSearch = dataset.name.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesProject = filterProjectId === '__all__' || dataset.projectId === filterProjectId
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
          <h1 className="text-3xl font-bold tracking-tight">Datasets</h1>
          <p className="text-muted-foreground">
            Manage your training datasets
          </p>
        </div>
        <div className="flex items-center gap-2">
          {/* Labelme to COCO Dialog */}
          <Dialog open={labelmeDialogOpen} onOpenChange={setLabelmeDialogOpen}>
            <DialogTrigger asChild>
              <Button variant="outline">
                <FileJson className="w-4 h-4 mr-2" />
                Labelme → COCO
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-2xl">
              <DialogHeader>
                <DialogTitle>Convert Labelme to COCO</DialogTitle>
                <DialogDescription>
                  Convert Labelme format dataset to COCO format with train/val/test split.
                </DialogDescription>
              </DialogHeader>
              <form onSubmit={handleLabelmeConvert}>
                <div className="grid grid-cols-2 gap-4 py-4">
                  <div className="space-y-2">
                    <Label htmlFor="lm-name">Dataset Name</Label>
                    <Input
                      id="lm-name"
                      value={labelmeFormData.name}
                      onChange={(e) => setLabelmeFormData({ ...labelmeFormData, name: e.target.value })}
                      placeholder="My Dataset"
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="lm-project">Project</Label>
                    <Select
                      value={labelmeFormData.projectId}
                      onValueChange={(value) => setLabelmeFormData({ ...labelmeFormData, projectId: value })}
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
                  <div className="col-span-2 space-y-2">
                    <Label htmlFor="lm-description">Description</Label>
                    <Textarea
                      id="lm-description"
                      value={labelmeFormData.description}
                      onChange={(e) => setLabelmeFormData({ ...labelmeFormData, description: e.target.value })}
                      placeholder="Describe your dataset..."
                      rows={2}
                    />
                  </div>
                  <div className="col-span-2 space-y-2">
                    <Label htmlFor="lm-images-path">Labelme Images Path</Label>
                    <Input
                      id="lm-images-path"
                      value={labelmeFormData.labelmeImagesPath}
                      onChange={(e) => setLabelmeFormData({ ...labelmeFormData, labelmeImagesPath: e.target.value })}
                      placeholder="path/to/labelme/images"
                      required
                    />
                    <p className="text-xs text-muted-foreground">
                      Absolute path or relative to PaddleDetection root
                    </p>
                  </div>
                  <div className="col-span-2 space-y-2">
                    <Label htmlFor="lm-annotations-path">Labelme Annotations Path</Label>
                    <Input
                      id="lm-annotations-path"
                      value={labelmeFormData.labelmeAnnotationsPath}
                      onChange={(e) => setLabelmeFormData({ ...labelmeFormData, labelmeAnnotationsPath: e.target.value })}
                      placeholder="path/to/labelme/annotations"
                      required
                    />
                    <p className="text-xs text-muted-foreground">
                      Directory containing .json annotation files
                    </p>
                  </div>
                  <div className="col-span-2 space-y-2">
                    <Label htmlFor="lm-output-dir">Output Dataset Directory (Optional)</Label>
                    <Input
                      id="lm-output-dir"
                      value={labelmeFormData.outputDatasetDir}
                      onChange={(e) => setLabelmeFormData({ ...labelmeFormData, outputDatasetDir: e.target.value })}
                      placeholder="dataset/my_converted_dataset"
                    />
                    <p className="text-xs text-muted-foreground">
                      Leave empty to auto-generate based on dataset name
                    </p>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="lm-train-ratio">Train Ratio</Label>
                    <Input
                      id="lm-train-ratio"
                      type="number"
                      step="0.1"
                      min="0"
                      max="1"
                      value={labelmeFormData.trainRatio}
                      onChange={(e) => setLabelmeFormData({ ...labelmeFormData, trainRatio: parseFloat(e.target.value) || 0 })}
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="lm-val-ratio">Val Ratio</Label>
                    <Input
                      id="lm-val-ratio"
                      type="number"
                      step="0.1"
                      min="0"
                      max="1"
                      value={labelmeFormData.valRatio}
                      onChange={(e) => setLabelmeFormData({ ...labelmeFormData, valRatio: parseFloat(e.target.value) || 0 })}
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="lm-test-ratio">Test Ratio</Label>
                    <Input
                      id="lm-test-ratio"
                      type="number"
                      step="0.1"
                      min="0"
                      max="1"
                      value={labelmeFormData.testRatio}
                      onChange={(e) => setLabelmeFormData({ ...labelmeFormData, testRatio: parseFloat(e.target.value) || 0 })}
                      required
                    />
                  </div>
                  <div className="col-span-2">
                    <p className={`text-sm ${Math.abs(labelmeFormData.trainRatio + labelmeFormData.valRatio + labelmeFormData.testRatio - 1.0) < 0.001 ? 'text-green-600' : 'text-red-600'}`}>
                      Sum: {(labelmeFormData.trainRatio + labelmeFormData.valRatio + labelmeFormData.testRatio).toFixed(2)} 
                      {Math.abs(labelmeFormData.trainRatio + labelmeFormData.valRatio + labelmeFormData.testRatio - 1.0) < 0.001 ? ' ✓' : ' (must equal 1.0)'}
                    </p>
                  </div>
                </div>
                <DialogFooter>
                  <Button type="submit" disabled={converting}>
                    {converting ? 'Converting...' : 'Convert to COCO'}
                  </Button>
                </DialogFooter>
              </form>
            </DialogContent>
          </Dialog>
          <Dialog open={dialogOpen} onOpenChange={(open) => {
          setDialogOpen(open)
          if (!open) {
            setEditingDataset(null)
            setFormData({
              name: '',
              description: '',
              projectId: '',
              format: 'COCO',
              numClasses: 1,
              trainImagePath: '',
              trainAnnoPath: '',
              evalImagePath: '',
              evalAnnoPath: '',
              datasetDir: '',
            })
          }
        }}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="w-4 h-4 mr-2" />
              Import Dataset
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-2xl">
            <DialogHeader>
              <DialogTitle>{editingDataset ? 'Edit Dataset' : 'Import COCO Dataset'}</DialogTitle>
              <DialogDescription>
                {editingDataset ? 'Update dataset configuration.' : 'Import a COCO format dataset by specifying the paths.'}
              </DialogDescription>
            </DialogHeader>
            <form onSubmit={handleSubmit}>
              <div className="grid grid-cols-2 gap-4 py-4">
                <div className="space-y-2">
                  <Label htmlFor="name">Dataset Name</Label>
                  <Input
                    id="name"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    placeholder="My Dataset"
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
                <div className="col-span-2 space-y-2">
                  <Label htmlFor="description">Description</Label>
                  <Textarea
                    id="description"
                    value={formData.description}
                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    placeholder="Describe your dataset..."
                    rows={2}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="format">Dataset Format</Label>
                  <Select
                    value={formData.format}
                    onValueChange={(value) => setFormData({ ...formData, format: value })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="COCO">COCO</SelectItem>
                      <SelectItem value="VOC">VOC</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="numClasses">Number of Classes</Label>
                  <Input
                    id="numClasses"
                    type="number"
                    min={1}
                    value={formData.numClasses}
                    onChange={(e) => setFormData({ ...formData, numClasses: parseInt(e.target.value) || 1 })}
                    placeholder="1"
                  />
                </div>
                <div className="col-span-2 space-y-2">
                  <Label htmlFor="datasetDir">Dataset Directory</Label>
                  <Input
                    id="datasetDir"
                    value={formData.datasetDir}
                    onChange={(e) => setFormData({ ...formData, datasetDir: e.target.value })}
                    placeholder="dataset/my_dataset"
                  />
                  <p className="text-xs text-muted-foreground">
                    Relative path from PaddleDetection root
                  </p>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="trainImagePath">Train Images Path</Label>
                  <Input
                    id="trainImagePath"
                    value={formData.trainImagePath}
                    onChange={(e) => setFormData({ ...formData, trainImagePath: e.target.value })}
                    placeholder="images/train"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="trainAnnoPath">Train Annotations Path</Label>
                  <Input
                    id="trainAnnoPath"
                    value={formData.trainAnnoPath}
                    onChange={(e) => setFormData({ ...formData, trainAnnoPath: e.target.value })}
                    placeholder="annotations/train.json"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="evalImagePath">Eval Images Path</Label>
                  <Input
                    id="evalImagePath"
                    value={formData.evalImagePath}
                    onChange={(e) => setFormData({ ...formData, evalImagePath: e.target.value })}
                    placeholder="images/val"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="evalAnnoPath">Eval Annotations Path</Label>
                  <Input
                    id="evalAnnoPath"
                    value={formData.evalAnnoPath}
                    onChange={(e) => setFormData({ ...formData, evalAnnoPath: e.target.value })}
                    placeholder="annotations/val.json"
                  />
                </div>
                

              </div>
              <DialogFooter>
                <Button type="submit">{editingDataset ? 'Update' : 'Import'}</Button>
              </DialogFooter>
            </form>
          </DialogContent>
        </Dialog>
      </div>
    </div>
      {/* Main Content */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Dataset List */}
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
                    placeholder="Search datasets..."
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
          ) : filteredDatasets.length === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Database className="w-12 h-12 text-muted-foreground mb-4" />
                <h3 className="text-lg font-medium">No datasets found</h3>
                <p className="text-sm text-muted-foreground mt-1">
                  {searchQuery ? 'Try a different search term' : 'Import your first dataset to get started'}
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-3">
              {filteredDatasets.map((dataset) => (
                <Card
                  key={dataset.id}
                  className={`cursor-pointer transition-all hover:shadow-md ${
                    selectedDataset?.id === dataset.id ? 'ring-2 ring-primary' : ''
                  }`}
                  onClick={() => setSelectedDataset(dataset)}
                >
                  <CardHeader className="pb-2">
                    <div className="flex items-start justify-between">
                      <div>
                        <CardTitle className="text-lg">{dataset.name}</CardTitle>
                        <CardDescription>
                          {dataset.description || 'No description'}
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
                            openEditDialog(dataset)
                          }}>
                            <Pencil className="w-4 h-4 mr-2" />
                            Edit
                          </DropdownMenuItem>
                          <DropdownMenuItem onClick={(e) => {
                            e.stopPropagation()
                            handleParseDataset(dataset.id)
                          }}>
                            <RefreshCw className="w-4 h-4 mr-2" />
                            Parse Statistics
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            className="text-destructive"
                            onClick={(e) => {
                              e.stopPropagation()
                              handleDelete(dataset.id)
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
                    <div className="flex items-center gap-6 text-sm text-muted-foreground">
                      <div className="flex items-center gap-1">
                        <Tag className="w-4 h-4" />
                        <span>{dataset.numClasses} classes</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <ImageIcon className="w-4 h-4" />
                        <span>{dataset.numTrainImages} train / {dataset.numEvalImages} val</span>
                      </div>
                      <Badge variant="secondary">{dataset.format}</Badge>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>

        {/* Dataset Statistics */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="w-5 h-5" />
                Dataset Statistics
              </CardTitle>
              <CardDescription>
                {selectedDataset ? selectedDataset.name : 'Select a dataset to view statistics'}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {selectedDataset ? (
                <div className="space-y-6">
                  {/* Summary Stats */}
                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-3 rounded-lg bg-muted/50">
                      <div className="text-2xl font-bold">{selectedDataset.numClasses}</div>
                      <div className="text-xs text-muted-foreground">Classes</div>
                    </div>
                    <div className="p-3 rounded-lg bg-muted/50">
                      <div className="text-2xl font-bold">{selectedDataset.numAnnotations}</div>
                      <div className="text-xs text-muted-foreground">Annotations</div>
                    </div>
                    <div className="p-3 rounded-lg bg-muted/50">
                      <div className="text-2xl font-bold">{selectedDataset.numTrainImages}</div>
                      <div className="text-xs text-muted-foreground">Train Images</div>
                    </div>
                    <div className="p-3 rounded-lg bg-muted/50">
                      <div className="text-2xl font-bold">{selectedDataset.numEvalImages}</div>
                      <div className="text-xs text-muted-foreground">Val Images</div>
                    </div>
                  </div>

                  {/* Class Distribution Chart */}
                  {selectedDataset.classStats && (() => {
                    const stats = getClassStats(selectedDataset.classStats)
                    return (
                      <div className="space-y-4">
                        {/* Train Chart */}
                        {stats.train.length > 0 && (
                          <div>
                            <h4 className="text-sm font-medium mb-3">Train Set Distribution</h4>
                            <ChartContainer config={chartConfig} className="h-[180px] w-full">
                              <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={stats.train}>
                                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                                  <XAxis dataKey="name" className="text-xs" />
                                  <YAxis className="text-xs" />
                                  <Tooltip />
                                  <Bar dataKey="count" name="Annotations" fill="#e07b39" radius={[4, 4, 0, 0]} />
                                  <Bar dataKey="imageCount" name="Images" fill="#4caf50" radius={[4, 4, 0, 0]} />
                                </BarChart>
                              </ResponsiveContainer>
                            </ChartContainer>
                          </div>
                        )}
                        {/* Val Chart */}
                        {stats.eval.length > 0 && (
                          <div>
                            <h4 className="text-sm font-medium mb-3">Val Set Distribution</h4>
                            <ChartContainer config={chartConfig} className="h-[180px] w-full">
                              <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={stats.eval}>
                                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                                  <XAxis dataKey="name" className="text-xs" />
                                  <YAxis className="text-xs" />
                                  <Tooltip />
                                  <Bar dataKey="count" name="Annotations" fill="#2196f3" radius={[4, 4, 0, 0]} />
                                  <Bar dataKey="imageCount" name="Images" fill="#9c27b0" radius={[4, 4, 0, 0]} />
                                </BarChart>
                              </ResponsiveContainer>
                            </ChartContainer>
                          </div>
                        )}
                      </div>
                    )
                  })()}

                  {/* Parse Button */}
                  <Button 
                    variant="outline" 
                    className="w-full"
                    onClick={() => handleParseDataset(selectedDataset.id)}
                    disabled={parsing}
                  >
                    <RefreshCw className={`w-4 h-4 mr-2 ${parsing ? 'animate-spin' : ''}`} />
                    {parsing ? 'Parsing...' : 'Parse Dataset Statistics'}
                  </Button>

                  {/* Download Chart */}
                  <Button variant="outline" className="w-full">
                    <Download className="w-4 h-4 mr-2" />
                    Download Statistics
                  </Button>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-8 text-center">
                  <FolderOpen className="w-12 h-12 text-muted-foreground mb-4" />
                  <p className="text-sm text-muted-foreground">
                    Click on a dataset to view its statistics
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
