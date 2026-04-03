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
  Eye,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Upload,
  AlertCircle,
  FileArchive,
} from 'lucide-react'
import { toast } from '@/hooks/use-toast'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription } from '@/components/ui/alert'
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

interface SampleImage {
  id: number
  fileName: string
  width: number
  height: number
  imagePath: string
  annotations: {
    id: number
    categoryId: number
    categoryName: string
    bbox: number[]
    area: number
  }[]
}

interface Category {
  id: number
  name: string
  supercategory?: string
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
  const [detectingClasses, setDetectingClasses] = useState(false)
  const [filterProjectId, setFilterProjectId] = useState<string>('__all__')
  const [previewDialogOpen, setPreviewDialogOpen] = useState(false)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewSamples, setPreviewSamples] = useState<SampleImage[]>([])
  const [previewCategories, setPreviewCategories] = useState<Category[]>([])
  const [selectedCategory, setSelectedCategory] = useState<string>('__all__')
  const [detailDialogOpen, setDetailDialogOpen] = useState(false)
  const [selectedSample, setSelectedSample] = useState<SampleImage | null>(null)
  const [zoom, setZoom] = useState(1)
  // Upload dataset dialog state
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false)
  const [uploadFormData, setUploadFormData] = useState({
    name: '',
    description: '',
    projectId: '',
    format: 'COCO' as 'COCO' | 'labelme',
  })
  const [uploadFiles, setUploadFiles] = useState<FileList | null>(null)
  const [uploadFolderMode, setUploadFolderMode] = useState(false)
  const [availableDatasets, setAvailableDatasets] = useState<Array<{
    name: string;
    path: string;
    hasTrain: boolean;
    hasVal: boolean;
    hasAnnotations: boolean;
    trainAnnotations: string[];
    valAnnotations: string[];
  }>>([])
  const [loadingAvailableDatasets, setLoadingAvailableDatasets] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadError, setUploadError] = useState<string | null>(null)
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

  // Auto-detect numClasses when trainAnnoPath changes
  useEffect(() => {
    const detectClasses = async () => {
      if (!formData.trainAnnoPath || formData.trainAnnoPath.trim() === '') {
        return
      }
      
      setDetectingClasses(true)
      try {
        const response = await fetch(`/api/datasets/parse?path=${encodeURIComponent(formData.trainAnnoPath)}&datasetDir=${encodeURIComponent(formData.datasetDir)}`)
        const result = await response.json()
        
        if (result.success && result.data.numClasses > 0) {
          setFormData(prev => ({ ...prev, numClasses: result.data.numClasses }))
        }
      } catch (error) {
        console.error('Failed to detect classes:', error)
      } finally {
        setDetectingClasses(false)
      }
    }

    const timeoutId = setTimeout(detectClasses, 500)
    return () => clearTimeout(timeoutId)
  }, [formData.trainAnnoPath])

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

  const fetchAvailableDatasets = async () => {
    setLoadingAvailableDatasets(true)
    try {
      const response = await fetch('/api/datasets/available')
      if (response.ok) {
        const data = await response.json()
        if (data.success) {
          setAvailableDatasets(data.datasets || [])
        }
      }
    } catch (error) {
      console.error('Failed to fetch available datasets:', error)
    } finally {
      setLoadingAvailableDatasets(false)
    }
  }

  // Auto-fill form when dataset folder is selected
  const handleDatasetSelect = (datasetName: string) => {
    const selectedDataset = availableDatasets.find(d => d.name === datasetName)
    if (selectedDataset) {
      setFormData(prev => ({
        ...prev,
        name: datasetName,
        datasetDir: `COCO/${datasetName}`,
        trainImagePath: selectedDataset.hasTrain ? 'data/train' : '',
        evalImagePath: selectedDataset.hasVal ? 'data/val' : '',
        trainAnnoPath: selectedDataset.trainAnnotations.length > 0 
          ? selectedDataset.trainAnnotations[0] 
          : '',
        evalAnnoPath: selectedDataset.valAnnotations.length > 0 
          ? selectedDataset.valAnnotations[0] 
          : '',
      }))
    } else {
      setFormData(prev => ({ ...prev, name: datasetName }))
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

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!uploadFiles || uploadFiles.length === 0) {
      toast({ 
        title: 'No files selected', 
        description: 'Please select files to upload',
        variant: 'destructive'
      })
      return
    }

    if (!uploadFormData.name.trim()) {
      toast({ 
        title: 'Dataset name required', 
        variant: 'destructive'
      })
      return
    }

    setUploading(true)
    setUploadProgress(0)
    setUploadError(null)

    try {
      const formData = new FormData()
      formData.append('datasetName', uploadFormData.name)
      formData.append('format', uploadFormData.format)
      formData.append('projectId', uploadFormData.projectId)
      formData.append('description', uploadFormData.description)

      // Calculate total size for progress tracking
      let totalSize = 0
      for (let i = 0; i < uploadFiles.length; i++) {
        totalSize += uploadFiles[i].size
        formData.append('files', uploadFiles[i])
      }

      const response = await fetch('/api/datasets/upload', {
        method: 'POST',
        body: formData,
      })

      // Simulate progress (since fetch doesn't support progress natively)
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) return prev
          return prev + 10
        })
      }, 500)

      const result = await response.json()
      clearInterval(progressInterval)

      if (response.ok && result.success) {
        setUploadProgress(100)
        toast({ 
          title: 'Upload successful', 
          description: `${result.message || `Uploaded ${result.data.files.length} files`}. Use "Import Dataset" to load this data.`
        })
        // Reset form - note: we don't call fetchDatasets() since no dataset record was created
        setUploadFormData({ name: '', description: '', projectId: '', format: 'COCO' })
        setUploadFiles(null)
        setUploadDialogOpen(false)
      } else {
        setUploadError(result.error || result.message || 'Upload failed')
        toast({ 
          title: 'Upload failed', 
          description: result.error || result.message || 'Unknown error',
          variant: 'destructive'
        })
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error'
      setUploadError(errorMsg)
      toast({ 
        title: 'Upload error', 
        description: errorMsg,
        variant: 'destructive'
      })
    } finally {
      setUploading(false)
    }
  }

  const filteredDatasets = datasets.filter(dataset => {
    const matchesSearch = dataset.name.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesProject = filterProjectId === '__all__' || dataset.projectId === filterProjectId
    return matchesSearch && matchesProject
  })

  const fetchSamples = async (categoryId?: string) => {
    if (!selectedDataset) return
    
    setPreviewLoading(true)
    try {
      const url = new URL('/api/datasets/samples', window.location.origin)
      url.searchParams.append('datasetId', selectedDataset.id)
      url.searchParams.append('limit', '10')
      if (categoryId && categoryId !== '__all__') {
        url.searchParams.append('categoryId', categoryId)
      }
      
      const response = await fetch(url.toString())
      const result = await response.json()
      
      if (result.success) {
        setPreviewSamples(result.data.samples)
        setPreviewCategories(result.data.categories)
      } else {
        toast({
          title: 'Failed to load samples',
          description: result.error || 'Could not load dataset samples',
          variant: 'destructive',
        })
      }
    } catch (error) {
      console.error('Error fetching samples:', error)
      toast({
        title: 'Error loading samples',
        variant: 'destructive',
      })
    } finally {
      setPreviewLoading(false)
    }
  }

  const openPreview = () => {
    setPreviewDialogOpen(true)
    setSelectedCategory('__all__')
    fetchSamples()
  }

  const openDetailView = (sample: SampleImage) => {
    setSelectedSample(sample)
    setDetailDialogOpen(true)
    setZoom(1)
  }

  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev + 0.25, 3))
  }

  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev - 0.25, 0.5))
  }

  const handleZoomReset = () => {
    setZoom(1)
  }

  const handleCategoryChange = (categoryId: string) => {
    setSelectedCategory(categoryId)
    fetchSamples(categoryId)
  }

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
          <Dialog open={uploadDialogOpen} onOpenChange={(open) => {
            setUploadDialogOpen(open)
            if (!open) {
              setUploadFormData({ name: '', description: '', projectId: '', format: 'COCO' })
              setUploadFiles(null)
              setUploadProgress(0)
              setUploadError(null)
              setUploadFolderMode(false)
            }
          }}>
            <DialogTrigger asChild>
              <Button variant="outline">
                <Upload className="w-4 h-4 mr-2" />
                Upload Dataset
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-2xl">
              <DialogHeader>
                <DialogTitle>Upload Dataset</DialogTitle>
                <DialogDescription>
                  Upload dataset files to your database directory. Supports COCO and Labelme formats.
                </DialogDescription>
              </DialogHeader>
              <form onSubmit={handleUpload}>
                <div className="grid grid-cols-2 gap-4 py-4">
                  <div className="space-y-2">
                    <Label htmlFor="upload-name">Dataset Name</Label>
                    <Input
                      id="upload-name"
                      value={uploadFormData.name}
                      onChange={(e) => setUploadFormData({ ...uploadFormData, name: e.target.value })}
                      placeholder="my_dataset"
                      required
                      pattern="[a-zA-Z0-9_-]+"
                      title="Only letters, numbers, underscores and hyphens allowed"
                    />
                    <p className="text-xs text-muted-foreground">
                      Folder will be created: {uploadFormData.format}/{uploadFormData.name || '[name]'}/data/
                    </p>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="upload-format">Data Format</Label>
                    <Select
                      value={uploadFormData.format}
                      onValueChange={(value: 'COCO' | 'labelme') => setUploadFormData({ ...uploadFormData, format: value })}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="COCO">COCO</SelectItem>
                        <SelectItem value="labelme">Labelme</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="upload-project">Project (Optional)</Label>
                    <Select
                      value={uploadFormData.projectId}
                      onValueChange={(value) => setUploadFormData({ ...uploadFormData, projectId: value })}
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
                    <Label htmlFor="upload-description">Description</Label>
                    <Textarea
                      id="upload-description"
                      value={uploadFormData.description}
                      onChange={(e) => setUploadFormData({ ...uploadFormData, description: e.target.value })}
                      placeholder="Describe your dataset..."
                      rows={2}
                    />
                  </div>
                  <div className="col-span-2 space-y-2">
                    <div className="flex items-center justify-between">
                      <Label htmlFor="upload-files">
                        {uploadFolderMode ? 'Select Folder' : 'Select Files'}
                      </Label>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                          setUploadFolderMode(!uploadFolderMode)
                          setUploadFiles(null)
                        }}
                      >
                        {uploadFolderMode ? 'Switch to Files' : 'Switch to Folder'}
                      </Button>
                    </div>
                    <Input
                      id="upload-files"
                      type="file"
                      {...(uploadFolderMode ? { webkitdirectory: '', directory: '' } : { multiple: true })}
                      onChange={(e) => setUploadFiles(e.target.files)}
                      disabled={uploading}
                    />
                    <p className="text-xs text-muted-foreground">
                      {uploadFolderMode
                        ? 'Select a folder containing your dataset files (images, annotations, etc.)'
                        : 'You can select multiple files (images, annotations, etc.)'}
                    </p>
                  </div>
                  {uploadFiles && uploadFiles.length > 0 && (
                    <div className="col-span-2">
                      <p className="text-sm text-muted-foreground">
                        Selected {uploadFiles.length} files, total size {((Array.from(uploadFiles).reduce((acc, f) => acc + f.size, 0)) / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  )}
                  {uploadError && (
                    <div className="col-span-2">
                      <Alert variant="destructive">
                        <AlertCircle className="h-4 w-4" />
                        <AlertDescription>{uploadError}</AlertDescription>
                      </Alert>
                    </div>
                  )}
                  {uploading && (
                    <div className="col-span-2 space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Upload Progress</span>
                        <span>{uploadProgress}%</span>
                      </div>
                      <Progress value={uploadProgress} />
                    </div>
                  )}
                </div>
                <DialogFooter>
                  <Button type="submit" disabled={uploading || !uploadFiles || uploadFiles.length === 0}>
                    {uploading ? 'Uploading...' : 'Start Upload'}
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
            <Button onClick={() => fetchAvailableDatasets()}>
              <Plus className="w-4 h-4 mr-2" />
              Import Dataset
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-2xl">
            <DialogHeader>
              <DialogTitle>{editingDataset ? 'Edit Dataset' : 'Import COCO Dataset'}</DialogTitle>
              <DialogDescription>
                {editingDataset ? 'Update dataset configuration.' : 'Select a dataset folder from your COCO directory.'}
              </DialogDescription>
            </DialogHeader>
            <form onSubmit={handleSubmit}>
              <div className="grid grid-cols-2 gap-4 py-4">
                {!editingDataset && (
                  <div className="space-y-2">
                    <Label htmlFor="name">Dataset Folder</Label>
                    <Select
                      value={formData.name}
                      onValueChange={handleDatasetSelect}
                      disabled={loadingAvailableDatasets}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder={loadingAvailableDatasets ? "Loading..." : "Select folder"} />
                      </SelectTrigger>
                      <SelectContent>
                        {availableDatasets.length === 0 ? (
                          <SelectItem value="__empty__" disabled>
                            No folders found in COCO directory
                          </SelectItem>
                        ) : (
                          availableDatasets.map((dataset) => (
                            <SelectItem key={dataset.name} value={dataset.name}>
                              {dataset.name} {!dataset.hasTrain && !dataset.hasVal && !dataset.hasAnnotations ? '(invalid)' : ''}
                            </SelectItem>
                          ))
                        )}
                      </SelectContent>
                    </Select>
                    {availableDatasets.length === 0 && !loadingAvailableDatasets && (
                      <p className="text-xs text-amber-600">
                        No dataset folders found. Please upload a dataset first.
                      </p>
                    )}
                  </div>
                )}
                {editingDataset && (
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
                )}
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
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="datasetDir">Dataset Directory</Label>
                  <Input
                    id="datasetDir"
                    value={formData.datasetDir}
                    disabled
                    className="bg-muted"
                  />
                  <p className="text-xs text-muted-foreground">
                    COCO/{formData.name}/
                  </p>
                </div>
                {/* Train Set Paths */}
                <div className="col-span-2 pt-2 border-t">
                  <h4 className="text-sm font-medium text-muted-foreground mb-2">Train Set</h4>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="trainImagePath">Train Images Path</Label>
                  <Input
                    id="trainImagePath"
                    value={formData.trainImagePath}
                    disabled
                    className="bg-muted"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="trainAnnoPath">Train Annotations Path</Label>
                  <Select
                    value={formData.trainAnnoPath}
                    onValueChange={(value) => setFormData({ ...formData, trainAnnoPath: value })}
                    disabled={availableDatasets.length === 0}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder={loadingAvailableDatasets ? "Loading..." : "Select annotation file"} />
                    </SelectTrigger>
                    <SelectContent>
                      {availableDatasets.find(d => d.name === formData.name)?.trainAnnotations.map((file) => (
                        <SelectItem key={file} value={file}>
                          {file}
                        </SelectItem>
                      )) || (
                        <SelectItem value="__empty__" disabled>
                          No annotation files found
                        </SelectItem>
                      )}
                    </SelectContent>
                  </Select>
                </div>
                {/* Eval Set Paths */}
                <div className="col-span-2 pt-2 border-t">
                  <h4 className="text-sm font-medium text-muted-foreground mb-2">Validation Set</h4>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="evalImagePath">Eval Images Path</Label>
                  <Input
                    id="evalImagePath"
                    value={formData.evalImagePath}
                    disabled
                    className="bg-muted"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="evalAnnoPath">Eval Annotations Path</Label>
                  <Select
                    value={formData.evalAnnoPath}
                    onValueChange={(value) => setFormData({ ...formData, evalAnnoPath: value })}
                    disabled={availableDatasets.length === 0}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder={loadingAvailableDatasets ? "Loading..." : "Select annotation file"} />
                    </SelectTrigger>
                    <SelectContent>
                      {availableDatasets.find(d => d.name === formData.name)?.valAnnotations.map((file) => (
                        <SelectItem key={file} value={file}>
                          {file}
                        </SelectItem>
                      )) || (
                        <SelectItem value="__empty__" disabled>
                          No annotation files found
                        </SelectItem>
                      )}
                    </SelectContent>
                  </Select>
                </div>
                {/* Detected Classes Display */}
                {formData.numClasses > 0 && (
                  <div className="col-span-2 pt-2">
                    <p className="text-sm text-green-600">
                      Detected {formData.numClasses} classes from train annotations
                    </p>
                  </div>
                )}
              </div>
              <DialogFooter>
                <Button type="submit" disabled={!editingDataset && availableDatasets.length === 0}>
                  {editingDataset ? 'Update' : 'Import'}
                </Button>
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

                  {/* Preview Samples Button */}
                  <Button 
                    variant="outline" 
                    className="w-full"
                    onClick={openPreview}
                    disabled={previewLoading}
                  >
                    <Eye className="w-4 h-4 mr-2" />
                    {previewLoading ? 'Loading...' : 'Preview Samples'}
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

      {/* Preview Samples Dialog */}
      <Dialog open={previewDialogOpen} onOpenChange={setPreviewDialogOpen}>
        <DialogContent className="!max-w-none w-[95vw] max-h-[90vh] overflow-hidden flex flex-col">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <ImageIcon className="w-5 h-5" />
              Sample Preview - {selectedDataset?.name}
            </DialogTitle>
            <DialogDescription>
              Preview dataset images with annotations. Select a category to filter samples.
            </DialogDescription>
          </DialogHeader>
          
          <div className="flex items-center gap-4 py-2">
            <Label htmlFor="category-filter">Filter by Category:</Label>
            <Select
              value={selectedCategory}
              onValueChange={handleCategoryChange}
            >
              <SelectTrigger className="w-[200px]" id="category-filter">
                <SelectValue placeholder="All Categories" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__all__">All Categories</SelectItem>
                {previewCategories.map((cat) => (
                  <SelectItem key={cat.id} value={String(cat.id)}>
                    {cat.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <span className="text-sm text-muted-foreground">
              {previewSamples.length} images
            </span>
          </div>

          <div className="flex-1 overflow-y-auto py-4">
            {previewLoading ? (
              <div className="flex items-center justify-center py-12">
                <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground" />
                <span className="ml-2 text-muted-foreground">Loading samples...</span>
              </div>
            ) : previewSamples.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <ImageIcon className="w-12 h-12 text-muted-foreground mb-4" />
                <p className="text-muted-foreground">No samples found for the selected category</p>
              </div>
            ) : (
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                {previewSamples.slice(0, 10).map((sample) => (
                  <Card 
                    key={sample.id} 
                    className="overflow-hidden cursor-pointer hover:ring-2 hover:ring-primary transition-all"
                    onDoubleClick={() => openDetailView(sample)}
                  >
                    <div className="relative aspect-square bg-muted">
                      <img
                        src={`/api/datasets/image?path=${encodeURIComponent(sample.imagePath)}`}
                        alt={sample.fileName}
                        className="w-full h-full object-contain"
                        onError={(e) => {
                          (e.target as HTMLImageElement).style.display = 'none'
                        }}
                      />
                      {/* Annotation overlays */}
                      <svg className="absolute inset-0 w-full h-full pointer-events-none">
                        {sample.annotations.map((ann, annIndex) => (
                          <g key={ann.id}>
                            <rect
                              x={(ann.bbox[0] / sample.width) * 100 + '%'}
                              y={(ann.bbox[1] / sample.height) * 100 + '%'}
                              width={(ann.bbox[2] / sample.width) * 100 + '%'}
                              height={(ann.bbox[3] / sample.height) * 100 + '%'}
                              fill="none"
                              stroke="#ef4444"
                              strokeWidth="2"
                            />
                          </g>
                        ))}
                      </svg>
                      {/* Annotation count badge */}
                      <Badge className="absolute top-2 right-2 bg-primary/80">
                        {sample.annotations.length} annotations
                      </Badge>
                    </div>
                    <CardContent className="p-2">
                      <p className="text-xs truncate text-muted-foreground" title={sample.fileName}>
                        {sample.fileName}
                      </p>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {Array.from(new Set(sample.annotations.map(a => a.categoryName))).slice(0, 3).map((catName, i) => (
                          <Badge key={i} variant="secondary" className="text-[10px]">
                            {catName}
                          </Badge>
                        ))}
                        {new Set(sample.annotations.map(a => a.categoryName)).size > 3 && (
                          <Badge variant="secondary" className="text-[10px]">
                            +{new Set(sample.annotations.map(a => a.categoryName)).size - 3}
                          </Badge>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setPreviewDialogOpen(false)}>
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Detail View Dialog with Zoom */}
      <Dialog open={detailDialogOpen} onOpenChange={setDetailDialogOpen}>
        <DialogContent className="!max-w-none w-[95vw] h-[95vh] max-h-none overflow-hidden flex flex-col p-0">
          <DialogHeader className="px-6 pt-6 pb-2">
            <DialogTitle className="flex items-center gap-2">
              <ImageIcon className="w-5 h-5" />
              {selectedSample?.fileName}
            </DialogTitle>
            <DialogDescription>
              Double-click image to zoom in, use controls to zoom in/out
            </DialogDescription>
          </DialogHeader>
          
          {/* Zoom Controls */}
          <div className="flex items-center justify-center gap-2 px-6 py-2 border-b">
            <Button variant="outline" size="sm" onClick={handleZoomOut} disabled={zoom <= 0.5}>
              <ZoomOut className="w-4 h-4" />
            </Button>
            <span className="text-sm min-w-[60px] text-center">{Math.round(zoom * 100)}%</span>
            <Button variant="outline" size="sm" onClick={handleZoomIn} disabled={zoom >= 3}>
              <ZoomIn className="w-4 h-4" />
            </Button>
            <Button variant="outline" size="sm" onClick={handleZoomReset}>
              <RotateCcw className="w-4 h-4" />
            </Button>
          </div>

          {/* Image Container */}
          <div className="flex-1 overflow-auto p-4 bg-muted/50">
            {selectedSample && (
              <div 
                className="relative inline-block min-w-full min-h-full"
                style={{ 
                  transform: `scale(${zoom})`, 
                  transformOrigin: 'top left',
                  transition: 'transform 0.2s ease'
                }}
              >
                <img
                  src={`/api/datasets/image?path=${encodeURIComponent(selectedSample.imagePath)}`}
                  alt={selectedSample.fileName}
                  className="max-w-none"
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.display = 'none'
                  }}
                  onDoubleClick={handleZoomIn}
                />
                {/* SVG Overlay for Annotations */}
                <svg 
                  className="absolute top-0 left-0 pointer-events-none"
                  style={{ width: selectedSample.width, height: selectedSample.height }}
                >
                  {selectedSample.annotations.map((ann) => (
                    <g key={ann.id}>
                      <rect
                        x={ann.bbox[0]}
                        y={ann.bbox[1]}
                        width={ann.bbox[2]}
                        height={ann.bbox[3]}
                        fill="none"
                        stroke="#ef4444"
                        strokeWidth="2"
                      />
                      <text
                        x={ann.bbox[0]}
                        y={ann.bbox[1] - 4}
                        fill="#ef4444"
                        fontSize="12"
                        fontWeight="bold"
                      >
                        {ann.categoryName}
                      </text>
                    </g>
                  ))}
                </svg>
              </div>
            )}
          </div>

          {/* Annotation List */}
          {selectedSample && selectedSample.annotations.length > 0 && (
            <div className="px-6 py-2 border-t max-h-[150px] overflow-y-auto">
              <h4 className="text-sm font-medium mb-2">Annotations ({selectedSample.annotations.length})</h4>
              <div className="flex flex-wrap gap-2">
                {selectedSample.annotations.map((ann) => (
                  <Badge key={ann.id} variant="secondary" className="text-xs">
                    {ann.categoryName} ({ann.bbox.map(n => Math.round(n)).join(', ')})
                  </Badge>
                ))}
              </div>
            </div>
          )}

          <DialogFooter className="px-6 py-4">
            <Button variant="outline" onClick={() => setDetailDialogOpen(false)}>
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
