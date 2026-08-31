'use client'

import { useEffect, useState, useRef } from 'react'
import { frameworkMeta, isAnomaly, isSegmentation } from '@/lib/frameworks'

/** Dataset layouts the importer understands; mirrors `FrameworkMeta.datasetFormat`. */
type DatasetFormat = 'COCO' | 'PaddleSeg' | 'AnomalyFolder'

/**
 * Blank import form. Extracted so the three places that reset the dialog cannot
 * drift apart — and so adding a field is one edit rather than four.
 */
const EMPTY_DATASET_FORM = {
  name: '',
  description: '',
  projectId: '',
  format: 'COCO' as DatasetFormat,
  numClasses: 1,
  trainImagePath: '',
  trainAnnoPath: '',
  evalImagePath: '',
  evalAnnoPath: '',
  datasetDir: '',
  // AnomalyFolder only: directory names relative to datasetDir. Pre-filled with
  // the layout the docs describe, since typing four paths from scratch is the
  // most tedious part of creating an anomaly dataset.
  normalDir: 'train_good',
  normalTestDir: 'test_good',
  abnormalDir: 'test_ng',
  maskDir: 'test_ng_mask',
}
import { useAuth } from '@/contexts/auth-context'
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
  Cell,
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
  Layers,
} from 'lucide-react'
import { toast } from '@/hooks/use-toast'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription } from '@/components/ui/alert'
import {
  ChartConfig,
  ChartContainer,
} from "@/components/ui/chart"
import { Slider } from '@/components/ui/slider'
import { SegSampleView } from '@/components/seg-sample-view'
import { getSegColorMap, rgbToCss } from '@/lib/seg-colors'

interface Project {
  id: string
  name: string
  framework?: string
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
  // AnomalyFolder only; null for every other format.
  normalDir?: string | null
  normalTestDir?: string | null
  abnormalDir?: string | null
  maskDir?: string | null
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
    framework?: string
  }
}

interface SampleImage {
  id: number
  fileName: string
  width: number
  height: number
  imagePath: string
  maskPath?: string
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
  color?: number[]
}

const chartConfig = {
  count: {
    label: "Annotations",
  },
  imageCount: {
    label: "Images",
  },
} satisfies ChartConfig

interface LabelmeMergePreview {
  canMerge: boolean
  labelme: { name: string; categories: string[]; totalJson: number; validSamples: number; invalidSamples: string[] }
  coco: { name: string; categories: string[]; trainCategories: string[]; valCategories: string[] }
  differences: { matched: string[]; onlyInLabelme: string[]; onlyInCoco: string[]; missingFromTrain: string[]; missingFromVal: string[] }
  before: { trainImages: number; valImages: number; trainAnnotations: number; valAnnotations: number }
  planned: { trainAdd: number; valAdd: number; skipped: number; trainImages: number; valImages: number }
}

export function DatasetsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [projects, setProjects] = useState<Project[]>([])
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [dialogOpen, setDialogOpen] = useState(false)
  const [labelmeDialogOpen, setLabelmeDialogOpen] = useState(false)
  const [converting, setConverting] = useState(false)
  const [mergeDialogOpen, setMergeDialogOpen] = useState(false)
  const [verifyingMerge, setVerifyingMerge] = useState(false)
  const [mergingDatasets, setMergingDatasets] = useState(false)
  const [mergePreview, setMergePreview] = useState<LabelmeMergePreview | null>(null)
  const [mergeFormData, setMergeFormData] = useState({ labelmeDataset: '', cocoDataset: '', trainRatio: 0.7, valRatio: 0.2 })
  const [labelmeFormData, setLabelmeFormData] = useState({
    name: '',
    description: '',
    selectedLabelmeDataset: '',
    labelmeImagesPath: '',
    labelmeAnnotationsPath: '',
    outputDatasetDir: '',
    trainRatio: 0.7,
    valRatio: 0.2,
    testRatio: 0.1,
    targetFramework: 'COCO' as 'COCO' | 'PaddleSeg',
  })
  const [editingDataset, setEditingDataset] = useState<Dataset | null>(null)
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null)
  const [parsing, setParsing] = useState(false)
  const [detectingClasses, setDetectingClasses] = useState(false)
  const [deletingFolderDataset, setDeletingFolderDataset] = useState<string | null>(null)
  const [manageDatasetsDialogOpen, setManageDatasetsDialogOpen] = useState(false)
  const { user } = useAuth()
  const [filterProjectId, setFilterProjectId] = useState<string>('__all__')
  const [previewDialogOpen, setPreviewDialogOpen] = useState(false)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewSamples, setPreviewSamples] = useState<SampleImage[]>([])
  const [previewCategories, setPreviewCategories] = useState<Category[]>([])
  const [selectedCategory, setSelectedCategory] = useState<string>('__all__')
  const [detailDialogOpen, setDetailDialogOpen] = useState(false)
  const [selectedSample, setSelectedSample] = useState<SampleImage | null>(null)
  const [zoom, setZoom] = useState(1)
  const [previewType, setPreviewType] = useState<'detection' | 'segmentation'>('detection')
  const [overlayOpacity, setOverlayOpacity] = useState(0.5)
  const [hideBackground, setHideBackground] = useState(true)
  // Abort controller ref for cancelling upload
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
    format: 'COCO' | 'labelme' | 'PaddleSeg';
    hasTrain: boolean;
    hasVal: boolean;
    hasAnnotations: boolean;
    hasImgs: boolean;
    hasJsons: boolean;
    trainAnnotations: string[];
    valAnnotations: string[];
    images: string[];
    jsons: string[];
  }>>([])
  const [loadingAvailableDatasets, setLoadingAvailableDatasets] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadError, setUploadError] = useState<string | null>(null)
  // Chunked upload state
  const [uploadSession, setUploadSession] = useState<{
    uploadId: string;
    targetDir: string;
    tempDir: string;
    files: Array<{
      relativePath: string;
      size: number;
      totalChunks: number;
      uploadedChunks: number;
      progress: number;
    }>;
    chunkSize: number;
    overallProgress: number;
  } | null>(null)
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'initializing' | 'uploading' | 'completing' | 'completed' | 'error'>('idle')
  const [currentFile, setCurrentFile] = useState<string>('')
  const abortControllerRef = useRef<AbortController | null>(null)
  // Labelme overwrite confirmation state
  const [confirmOverwriteOpen, setConfirmOverwriteOpen] = useState(false)
  const [confirmOverwriteMessage, setConfirmOverwriteMessage] = useState('')
  const [pendingUploadSession, setPendingUploadSession] = useState<any>(null)
  const [formData, setFormData] = useState({ ...EMPTY_DATASET_FORM })

  // PaddleSeg datasets use list files (train.txt/val.txt) + num_classes instead of COCO annotations.
  const isSegDataset = isSegmentation(projects.find(p => p.id === formData.projectId)?.framework)

  // Anomaly datasets are folders of images with no annotation file at all, so
  // they bypass the annotation pickers entirely.
  const isAnomalyDataset = isAnomaly(projects.find(p => p.id === formData.projectId)?.framework)

  // Whether the dataset currently selected for stats/preview is PaddleSeg.
  const isSegSelected = isSegmentation(selectedDataset?.project?.framework)

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

  // Auto-fill form when dataset folder is selected. Fields differ per format:
  //   COCO      -> datasetDir=COCO/{name}, image paths = data/train, data/val;
  //                trainAnnoPath/evalAnnoPath = data/annotations/instance_*.json
  //   PaddleSeg -> datasetDir=PaddleSeg/{name}/data (the PaddleSeg dataset_root,
  //                which contains train.txt, val.txt, class_names.txt,
  //                JPEGImages/, Annotations/); image paths are unused for seg.
  const handleDatasetSelect = (datasetName: string) => {
    const selected = availableDatasets.find(d => d.name === datasetName)
    if (!selected) {
      setFormData(prev => ({ ...prev, name: datasetName }))
      return
    }
    if (selected.format === 'PaddleSeg') {
      setFormData(prev => ({
        ...prev,
        name: datasetName,
        format: 'PaddleSeg',
        datasetDir: `PaddleSeg/${datasetName}/data`,
        trainImagePath: '',
        evalImagePath: '',
        trainAnnoPath: selected.trainAnnotations[0] || 'train.txt',
        evalAnnoPath: selected.valAnnotations[0] || 'val.txt',
      }))
    } else {
      setFormData(prev => ({
        ...prev,
        name: datasetName,
        format: 'COCO',
        datasetDir: `COCO/${datasetName}`,
        trainImagePath: selected.hasTrain ? 'data/train' : '',
        evalImagePath: selected.hasVal ? 'data/val' : '',
        trainAnnoPath: selected.trainAnnotations[0] || '',
        evalAnnoPath: selected.valAnnotations[0] || '',
      }))
    }
  }

  // Switching project may change the required dataset framework (COCO vs
  // PaddleSeg). If the currently-selected dataset folder doesn't match the
  // new project's framework, wipe the auto-filled paths so the user gets a
  // clean slate to reselect from the freshly-filtered dropdown.
  const handleProjectChange = (projectId: string) => {
    const newProject = projects.find(p => p.id === projectId)
    // Derived from the framework registry rather than a local conditional, so a
    // new framework's dataset layout is declared in exactly one place.
    const expectedFormat = frameworkMeta(newProject?.framework).datasetFormat as DatasetFormat
    setFormData(prev => {
      const currentDs = availableDatasets.find(d => d.name === prev.name)
      const compatible = !currentDs || currentDs.format === expectedFormat
      if (compatible) {
        return { ...prev, projectId, format: expectedFormat }
      }
      return {
        ...prev,
        projectId,
        format: expectedFormat,
        name: '',
        datasetDir: '',
        trainImagePath: '',
        evalImagePath: '',
        trainAnnoPath: '',
        evalAnnoPath: '',
      }
    })
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
        } else {
          const err = await response.json().catch(() => ({}))
          toast({
            title: 'Failed to update dataset',
            description: err.message || err.error || `HTTP ${response.status}`,
            variant: 'destructive',
          })
        }
      } else {
        // Include username when importing dataset
        const submitData = {
          ...formData,
          username: user?.username,
        }
        const response = await fetch('/api/datasets', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(submitData),
        })
        if (response.ok) {
          toast({ title: 'Dataset created successfully' })
          fetchDatasets()
          setDialogOpen(false)
          setFormData({ ...EMPTY_DATASET_FORM })
        } else {
          const err = await response.json().catch(() => ({}))
          toast({
            title: 'Failed to import dataset',
            description: err.message || err.error || `HTTP ${response.status}`,
            variant: 'destructive',
          })
        }
      }
    } catch (error) {
      toast({
        title: 'Error saving dataset',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      })
    }
  }

  const handleDeleteFolderDataset = async (datasetName: string) => {
    if (!confirm(`Are you sure you want to delete the dataset folder "${datasetName}"?\n\nThis will permanently delete all files in this folder.`)) {
      return
    }

    setDeletingFolderDataset(datasetName)
    try {
      const response = await fetch(`/api/datasets/available?name=${encodeURIComponent(datasetName)}`, {
        method: 'DELETE',
      })
      const result = await response.json()

      if (result.success) {
        toast({ title: `Dataset "${datasetName}" deleted successfully` })
        // Refresh the available datasets list
        fetchAvailableDatasets()
      } else {
        toast({ 
          title: 'Failed to delete dataset', 
          description: result.error || 'Unknown error',
          variant: 'destructive' 
        })
      }
    } catch (error) {
      toast({ 
        title: 'Error deleting dataset', 
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive' 
      })
    } finally {
      setDeletingFolderDataset(null)
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
      ...EMPTY_DATASET_FORM,
      name: dataset.name,
      description: dataset.description || '',
      projectId: dataset.projectId,
      format: (['PaddleSeg', 'AnomalyFolder'].includes(dataset.format)
        ? dataset.format
        : 'COCO') as DatasetFormat,
      numClasses: dataset.numClasses || 1,
      trainImagePath: dataset.trainImagePath || '',
      trainAnnoPath: dataset.trainAnnoPath || '',
      evalImagePath: dataset.evalImagePath || '',
      evalAnnoPath: dataset.evalAnnoPath || '',
      datasetDir: dataset.datasetDir || '',
      normalDir: dataset.normalDir || EMPTY_DATASET_FORM.normalDir,
      normalTestDir: dataset.normalTestDir ?? '',
      abnormalDir: dataset.abnormalDir ?? '',
      maskDir: dataset.maskDir ?? '',
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

  // Handle labelme dataset selection - auto-fill paths
  const handleLabelmeDatasetSelect = (datasetName: string) => {
    const selectedDataset = availableDatasets.find(
      d => d.name === datasetName && d.format === 'labelme'
    )
    if (selectedDataset) {
      // Extract base path from dataset path (remove /data suffix if exists)
      const basePath = selectedDataset.path.replace(/[/\\]data$/, '')
      setLabelmeFormData(prev => ({
        ...prev,
        selectedLabelmeDataset: datasetName,
        labelmeImagesPath: `${basePath}/data/imgs`,
        labelmeAnnotationsPath: `${basePath}/data/jsons`,
        // Auto-generate output directory based on dataset name and target framework
        outputDatasetDir: prev.name
          ? `${basePath.replace(/[/\\]labelme[/\\][^/\\]+$/, '')}/${prev.targetFramework}/${prev.name}`
          : '',
      }))
    } else {
      setLabelmeFormData(prev => ({
        ...prev,
        selectedLabelmeDataset: datasetName,
      }))
    }
  }

  // Update output directory when name changes if a labelme dataset is selected
  const handleLabelmeNameChange = (name: string) => {
    const selectedDataset = availableDatasets.find(
      d => d.name === labelmeFormData.selectedLabelmeDataset && d.format === 'labelme'
    )
    setLabelmeFormData(prev => {
      let outputDir = ''
      if (selectedDataset && name) {
        const basePath = selectedDataset.path.replace(/[/\\]data$/, '')
        outputDir = `${basePath.replace(/[/\\]labelme[/\\][^/\\]+$/, '')}/${prev.targetFramework}/${name}`
      }
      return { ...prev, name, outputDatasetDir: outputDir }
    })
  }

  // Switch target framework (COCO / PaddleSeg) and recompute the auto-
  // generated output directory so it points at the matching top-level folder.
  const handleTargetFrameworkChange = (target: 'COCO' | 'PaddleSeg') => {
    const selectedDataset = availableDatasets.find(
      d => d.name === labelmeFormData.selectedLabelmeDataset && d.format === 'labelme'
    )
    setLabelmeFormData(prev => {
      let outputDir = ''
      if (selectedDataset && prev.name) {
        const basePath = selectedDataset.path.replace(/[/\\]data$/, '')
        outputDir = `${basePath.replace(/[/\\]labelme[/\\][^/\\]+$/, '')}/${target}/${prev.name}`
      }
      return { ...prev, targetFramework: target, outputDatasetDir: outputDir }
    })
  }

  const handleLabelmeConvert = async (e: React.FormEvent) => {
    e.preventDefault()

    const isSeg = labelmeFormData.targetFramework === 'PaddleSeg'

    // Validate ratios: for PaddleSeg only train/val (must sum to 1);
    // for COCO the classic train/val/test triple applies.
    if (isSeg) {
      const total = labelmeFormData.trainRatio + labelmeFormData.valRatio
      if (Math.abs(total - 1.0) > 0.001) {
        toast({
          title: 'Invalid ratios',
          description: `Train + Val must equal 1.0 (current: ${total.toFixed(2)})`,
          variant: 'destructive'
        })
        return
      }
    } else {
      const total = labelmeFormData.trainRatio + labelmeFormData.valRatio + labelmeFormData.testRatio
      if (Math.abs(total - 1.0) > 0.001) {
        toast({
          title: 'Invalid ratios',
          description: `Train + Val + Test must equal 1.0 (current: ${total.toFixed(2)})`,
          variant: 'destructive'
        })
        return
      }
    }

    // Check if a dataset with the same name already exists in the target format.
    const existingDataset = availableDatasets.find(
      d => d.name === labelmeFormData.name && d.format === labelmeFormData.targetFramework
    )
    if (existingDataset) {
      toast({
        title: 'Dataset name already exists',
        description: `A ${labelmeFormData.targetFramework} dataset named "${labelmeFormData.name}" already exists. Please choose a different name.`,
        variant: 'destructive'
      })
      return
    }

    const endpoint = isSeg ? '/api/datasets/labelme-to-paddleseg' : '/api/datasets/labelme-to-coco'

    setConverting(true)
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(labelmeFormData),
      })
      const result = await response.json()
      
      if (result.success) {
        const desc = isSeg
          ? `Created PaddleSeg dataset with ${result.stats.trainCount} train, ${result.stats.valCount} val images (${result.stats.numClasses} classes: ${result.stats.classNames?.join(', ') ?? ''})`
          : `Created dataset with ${result.stats.trainCount} train, ${result.stats.valCount} val, ${result.stats.testCount} test images`
        toast({
          title: 'Conversion successful',
          description: desc,
        })
        fetchDatasets()
        setLabelmeDialogOpen(false)
        setLabelmeFormData({
          name: '',
          description: '',
          selectedLabelmeDataset: '',
          labelmeImagesPath: '',
          labelmeAnnotationsPath: '',
          outputDatasetDir: '',
          trainRatio: 0.7,
          valRatio: 0.2,
          testRatio: 0.1,
          targetFramework: 'COCO',
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

  // Chunked upload functions
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

    // Start chunked upload
    await startChunkedUpload()
  }

  const startChunkedUpload = async () => {
    if (!uploadFiles) return

    setUploading(true)
    setUploadError(null)
    setUploadStatus('initializing')
    abortControllerRef.current = new AbortController()

    try {
      // Build file list with metadata
      const files = Array.from(uploadFiles).map(file => {
        const relativePath = (file as any).webkitRelativePath || file.name
        return {
          name: file.name,
          relativePath: relativePath,
          size: file.size,
        }
      })

      const totalSize = files.reduce((acc, f) => acc + f.size, 0)

      // Step 1: Initialize upload session
      const initResponse = await fetch('/api/datasets/upload/init', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          datasetName: uploadFormData.name,
          format: uploadFormData.format,
          totalFiles: files.length,
          totalSize: totalSize,
          files: files,
        }),
        signal: abortControllerRef.current.signal,
      })

      if (!initResponse.ok) {
        const error = await initResponse.json()
        throw new Error(error.error || 'Failed to initialize upload')
      }

      const session = await initResponse.json()

      // Check if requires confirmation (labelme overwrite)
      if (session.requiresConfirmation) {
        setConfirmOverwriteMessage(session.message)
        setPendingUploadSession(session)
        setConfirmOverwriteOpen(true)
        setUploading(false)
        setUploadStatus('idle')
        return
      }

      await executeChunkUpload(session)
    } catch (error) {
      if ((error as Error).name === 'AbortError') {
        toast({ 
          title: 'Upload cancelled', 
          description: 'Upload was cancelled by user',
        })
      } else {
        const errorMsg = error instanceof Error ? error.message : 'Unknown error'
        setUploadError(errorMsg)
        setUploadStatus('error')
        toast({ 
          title: 'Upload error', 
          description: errorMsg,
          variant: 'destructive'
        })
      }
    } finally {
      setUploading(false)
      abortControllerRef.current = null
    }
  }

  // Execute file chunk upload after confirmation
  const executeChunkUpload = async (session: any) => {
    if (!uploadFiles) return

    setUploadSession(session)
    setUploadStatus('uploading')
    setUploading(true)

    try {
      // Step 2: Upload chunks for each file
      let totalUploadedChunks = 0
      const totalExpectedChunks = session.files.reduce((acc: number, f: any) => acc + f.totalChunks, 0)

      for (const fileInfo of session.files) {
        if (abortControllerRef.current?.signal.aborted) {
          throw new Error('Upload cancelled')
        }

        setCurrentFile(fileInfo.relativePath)
        
        // Find the actual File object
        const file = Array.from(uploadFiles).find(f => 
          ((f as any).webkitRelativePath || f.name) === fileInfo.relativePath
        )
        
        if (!file) continue

        // Upload missing chunks
        const uploadedChunks = new Set(fileInfo.uploadedChunks)
        
        for (let chunkIndex = 0; chunkIndex < fileInfo.totalChunks; chunkIndex++) {
          if (abortControllerRef.current?.signal.aborted) {
            throw new Error('Upload cancelled')
          }

          // Skip already uploaded chunks (resume support)
          if (uploadedChunks.has(chunkIndex)) {
            totalUploadedChunks++
            continue
          }

          // Calculate chunk boundaries
          const start = chunkIndex * session.chunkSize
          const end = Math.min(start + session.chunkSize, file.size)
          const chunk = file.slice(start, end)

          // Upload chunk
          const chunkFormData = new FormData()
          chunkFormData.append('uploadId', session.uploadId)
          chunkFormData.append('relativePath', fileInfo.relativePath)
          chunkFormData.append('chunkIndex', chunkIndex.toString())
          chunkFormData.append('totalChunks', fileInfo.totalChunks.toString())
          chunkFormData.append('chunk', chunk)

          let retries = 0
          const maxRetries = 3
          
          while (retries < maxRetries) {
            try {
              const response = await fetch('/api/datasets/upload/chunk', {
                method: 'POST',
                body: chunkFormData,
                signal: abortControllerRef.current?.signal,
              })

              if (!response.ok) {
                throw new Error(`Chunk upload failed: ${response.status}`)
              }

              break // Success, exit retry loop
            } catch (err) {
              retries++
              if (retries >= maxRetries) throw err
              await new Promise(r => setTimeout(r, 1000 * retries)) // Exponential backoff
            }
          }

          totalUploadedChunks++
          const progress = Math.round((totalUploadedChunks / totalExpectedChunks) * 100)
          setUploadProgress(progress)

          // Update file progress in session
          setUploadSession(prev => {
            if (!prev) return prev
            return {
              ...prev,
              files: prev.files.map(f => 
                f.relativePath === fileInfo.relativePath
                  ? { ...f, uploadedChunks: f.uploadedChunks + 1, progress: Math.round(((f.uploadedChunks + 1) / f.totalChunks) * 100) }
                  : f
              ),
              overallProgress: progress,
            }
          })
        }
      }

      // Step 3: Complete upload and merge chunks
      setUploadStatus('completing')
      setCurrentFile('Merging files...')

      const completeResponse = await fetch('/api/datasets/upload/complete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          uploadId: session.uploadId,
          targetDir: session.targetDir,
          files: session.files,
          format: uploadFormData.format,
          datasetName: uploadFormData.name,
          overwrite: session.requiresConfirmation || false,
        }),
        signal: abortControllerRef.current?.signal,
      })

      if (!completeResponse.ok) {
        const error = await completeResponse.json()
        throw new Error(error.error || 'Failed to complete upload')
      }

      const result = await completeResponse.json()

      if (result.success) {
        setUploadStatus('completed')
        setUploadProgress(100)
        toast({ 
          title: 'Upload successful', 
          description: `${result.message || `Uploaded ${result.data.files.length} files`}. Use "Import Dataset" to load this data.`
        })
        // Reset form
        setUploadFormData({ name: '', description: '', projectId: '', format: 'COCO' })
        setUploadFiles(null)
        setUploadSession(null)
        setUploadDialogOpen(false)
      } else {
        throw new Error(result.error || 'Upload failed')
      }

    } catch (error) {
      if ((error as Error).name === 'AbortError') {
        toast({ 
          title: 'Upload cancelled', 
          description: 'Upload was cancelled by user',
        })
      } else {
        const errorMsg = error instanceof Error ? error.message : 'Unknown error'
        setUploadError(errorMsg)
        setUploadStatus('error')
        toast({ 
          title: 'Upload error', 
          description: errorMsg,
          variant: 'destructive'
        })
      }
    } finally {
      setUploading(false)
      abortControllerRef.current = null
    }
  }

  const cancelUpload = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
  }

  // Handle user confirmation for labelme overwrite - re-init with confirmedOverwrite flag
  const handleConfirmOverwrite = async () => {
    setConfirmOverwriteOpen(false)
    if (!uploadFiles) return

    setUploading(true)
    setUploadError(null)
    setUploadStatus('initializing')
    abortControllerRef.current = new AbortController()

    try {
      // Build file list with metadata
      const files = Array.from(uploadFiles).map(file => {
        const relativePath = (file as any).webkitRelativePath || file.name
        return {
          name: file.name,
          relativePath: relativePath,
          size: file.size,
        }
      })

      const totalSize = files.reduce((acc, f) => acc + f.size, 0)

      // Re-initialize upload session with confirmedOverwrite flag
      const initResponse = await fetch('/api/datasets/upload/init', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          datasetName: uploadFormData.name,
          format: uploadFormData.format,
          totalFiles: files.length,
          totalSize: totalSize,
          files: files,
          confirmedOverwrite: true,
        }),
        signal: abortControllerRef.current?.signal,
      })

      if (!initResponse.ok) {
        const error = await initResponse.json()
        throw new Error(error.error || 'Failed to initialize upload')
      }

      const session = await initResponse.json()
      await executeChunkUpload(session)
      setPendingUploadSession(null)
    } catch (error) {
      if ((error as Error).name === 'AbortError') {
        toast({
          title: 'Upload cancelled',
          description: 'Upload was cancelled by user',
        })
      } else {
        const errorMsg = error instanceof Error ? error.message : 'Unknown error'
        setUploadError(errorMsg)
        setUploadStatus('error')
        toast({
          title: 'Upload error',
          description: errorMsg,
          variant: 'destructive'
        })
      }
    } finally {
      setUploading(false)
      abortControllerRef.current = null
    }
  }

  // Handle user cancellation for labelme overwrite
  const handleCancelOverwrite = () => {
    setConfirmOverwriteOpen(false)
    setPendingUploadSession(null)
    setConfirmOverwriteMessage('')
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
        setPreviewType(result.data.type === 'segmentation' ? 'segmentation' : 'detection')
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
          <Dialog open={labelmeDialogOpen} onOpenChange={(open) => {
              setLabelmeDialogOpen(open)
              if (open) {
                fetchAvailableDatasets()
              }
            }}>
            <DialogTrigger asChild>
              <Button variant="outline">
                <FileJson className="w-4 h-4 mr-2" />
                Convert Labelme
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-2xl">
              <DialogHeader>
                <DialogTitle>
                  {labelmeFormData.targetFramework === 'PaddleSeg'
                    ? 'Convert Labelme to PaddleSeg'
                    : 'Convert Labelme to COCO'}
                </DialogTitle>
                <DialogDescription>
                  {labelmeFormData.targetFramework === 'PaddleSeg'
                    ? 'Convert Labelme polygon annotations to PaddleSeg (grayscale label masks + train/val list files).'
                    : 'Convert Labelme format dataset to COCO format with train/val/test split.'}
                </DialogDescription>
              </DialogHeader>
              <form onSubmit={handleLabelmeConvert}>
                <div className="grid grid-cols-2 gap-4 py-4">
                  <div className="col-span-2 space-y-2">
                    <Label>Target Framework</Label>
                    <div className="flex gap-2">
                      <Button
                        type="button"
                        variant={labelmeFormData.targetFramework === 'COCO' ? 'default' : 'outline'}
                        onClick={() => handleTargetFrameworkChange('COCO')}
                        className="flex-1"
                      >
                        COCO (Detection)
                      </Button>
                      <Button
                        type="button"
                        variant={labelmeFormData.targetFramework === 'PaddleSeg' ? 'default' : 'outline'}
                        onClick={() => handleTargetFrameworkChange('PaddleSeg')}
                        className="flex-1"
                      >
                        PaddleSeg (Segmentation)
                      </Button>
                    </div>
                  </div>
                  <div className="col-span-2 space-y-2">
                    <Label htmlFor="lm-dataset-select">Select Labelme Dataset</Label>
                    <Select
                      value={labelmeFormData.selectedLabelmeDataset}
                      onValueChange={handleLabelmeDatasetSelect}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Choose an uploaded labelme dataset" />
                      </SelectTrigger>
                      <SelectContent>
                        {availableDatasets
                          .filter(d => d.format === 'labelme')
                          .map((dataset) => (
                            <SelectItem key={dataset.name} value={dataset.name}>
                              {dataset.name} ({dataset.images.length} images, {dataset.jsons.length} jsons)
                            </SelectItem>
                          ))}
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-muted-foreground">
                      Selecting a dataset will auto-fill the paths below
                    </p>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="lm-name">Dataset Name</Label>
                    <Input
                      id="lm-name"
                      value={labelmeFormData.name}
                      onChange={(e) => handleLabelmeNameChange(e.target.value)}
                      placeholder="My Dataset"
                      required
                    />
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
                  {labelmeFormData.targetFramework !== 'PaddleSeg' && (
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
                  )}
                  <div className="col-span-2">
                    {labelmeFormData.targetFramework === 'PaddleSeg' ? (
                      <p className={`text-sm ${Math.abs(labelmeFormData.trainRatio + labelmeFormData.valRatio - 1.0) < 0.001 ? 'text-green-600' : 'text-red-600'}`}>
                        Sum: {(labelmeFormData.trainRatio + labelmeFormData.valRatio).toFixed(2)}
                        {Math.abs(labelmeFormData.trainRatio + labelmeFormData.valRatio - 1.0) < 0.001 ? ' ✓' : ' (train + val must equal 1.0)'}
                      </p>
                    ) : (
                      <p className={`text-sm ${Math.abs(labelmeFormData.trainRatio + labelmeFormData.valRatio + labelmeFormData.testRatio - 1.0) < 0.001 ? 'text-green-600' : 'text-red-600'}`}>
                        Sum: {(labelmeFormData.trainRatio + labelmeFormData.valRatio + labelmeFormData.testRatio).toFixed(2)}
                        {Math.abs(labelmeFormData.trainRatio + labelmeFormData.valRatio + labelmeFormData.testRatio - 1.0) < 0.001 ? ' ✓' : ' (must equal 1.0)'}
                      </p>
                    )}
                  </div>
                </div>
                <DialogFooter>
                  <Button type="submit" disabled={converting}>
                    {converting
                      ? 'Converting...'
                      : labelmeFormData.targetFramework === 'PaddleSeg'
                        ? 'Convert to PaddleSeg'
                        : 'Convert to COCO'}
                  </Button>
                </DialogFooter>
              </form>
            </DialogContent>
          </Dialog>
          {/* Labelme Overwrite Confirmation Dialog */}
          <Dialog open={confirmOverwriteOpen} onOpenChange={setConfirmOverwriteOpen}>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Confirm Overwrite</DialogTitle>
                <DialogDescription>
                  {confirmOverwriteMessage}
                </DialogDescription>
              </DialogHeader>
              <DialogFooter>
                <Button variant="outline" onClick={handleCancelOverwrite}>
                  Cancel
                </Button>
                <Button onClick={handleConfirmOverwrite}>
                  Confirm Upload
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>

          <Dialog open={uploadDialogOpen} onOpenChange={(open) => {
            setUploadDialogOpen(open)
            if (!open) {
              setUploadFormData({ name: '', description: '', projectId: '', format: 'COCO' })
              setUploadFiles(null)
              setUploadProgress(0)
              setUploadError(null)
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
                    <Label htmlFor="upload-files">Select Folder</Label>
                    <input
                      id="upload-files"
                      type="file"
                      {...({ webkitdirectory: '', directory: '' } as any)}
                      onChange={(e) => setUploadFiles(e.target.files)}
                      disabled={uploading}
                      className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                    />
                    <p className="text-xs text-muted-foreground">
                      Select a folder containing your dataset files (images, annotations, etc.)
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
                    <div className="col-span-2 space-y-3">
                      <div className="flex justify-between text-sm">
                        <span className="font-medium">
                          {uploadStatus === 'initializing' && 'Initializing upload...'}
                          {uploadStatus === 'uploading' && 'Uploading chunks...'}
                          {uploadStatus === 'completing' && 'Merging files...'}
                          {uploadStatus === 'completed' && 'Upload complete!'}
                          {uploadStatus === 'error' && 'Upload failed'}
                        </span>
                        <span className="font-mono">{uploadProgress}%</span>
                      </div>
                      <Progress value={uploadProgress} className="h-2" />
                      {currentFile && (
                        <p className="text-xs text-muted-foreground truncate">
                          Current: {currentFile}
                        </p>
                      )}
                      {uploadSession && uploadSession.files.length > 0 && (
                        <div className="text-xs text-muted-foreground">
                          {uploadSession.files.filter(f => f.progress === 100).length} / {uploadSession.files.length} files completed
                        </div>
                      )}
                    </div>
                  )}
                </div>
                <DialogFooter className="gap-2">
                  {uploading && (
                    <Button type="button" variant="destructive" onClick={cancelUpload}>
                      Cancel
                    </Button>
                  )}
                  <Button type="submit" disabled={uploading || !uploadFiles || uploadFiles.length === 0}>
                    {uploading ? 'Uploading...' : 'Start Upload'}
                  </Button>
                </DialogFooter>
              </form>
            </DialogContent>
          </Dialog>
          <Dialog open={manageDatasetsDialogOpen} onOpenChange={(open) => {
            setManageDatasetsDialogOpen(open)
            if (open) {
              fetchAvailableDatasets()
            }
          }}>
            <DialogTrigger asChild>
              <Button variant="outline">
                <FolderOpen className="w-4 h-4 mr-2" />
                Manage Uploaded
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-2xl">
              <DialogHeader>
                <DialogTitle>Manage Uploaded Datasets</DialogTitle>
                <DialogDescription>
                  Delete dataset folders from your COCO directory. This action cannot be undone.
                </DialogDescription>
              </DialogHeader>
              <div className="py-4">
                {loadingAvailableDatasets ? (
                  <div className="flex items-center justify-center py-8">
                    <RefreshCw className="w-6 h-6 animate-spin text-muted-foreground" />
                    <span className="ml-2 text-muted-foreground">Loading datasets...</span>
                  </div>
                ) : availableDatasets.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-8 text-center">
                    <FolderOpen className="w-12 h-12 text-muted-foreground mb-4" />
                    <p className="text-muted-foreground">No dataset folders found in COCO directory</p>
                  </div>
                ) : (
                  <div className="space-y-2 max-h-[400px] overflow-y-auto">
                    {availableDatasets.map((dataset) => (
                      <div key={dataset.name} className="flex items-center justify-between p-3 border rounded-lg">
                        <div className="flex items-center gap-3">
                          <Database className="w-5 h-5 text-muted-foreground" />
                          <div>
                            <p className="font-medium">{dataset.name}</p>
                            <p className="text-xs text-muted-foreground">
                              {dataset.hasTrain && <span className="mr-2">Train</span>}
                              {dataset.hasVal && <span className="mr-2">Val</span>}
                              {dataset.hasAnnotations && <span>Annotations</span>}
                            </p>
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-destructive hover:text-destructive hover:bg-destructive/10"
                          onClick={() => handleDeleteFolderDataset(dataset.name)}
                          disabled={deletingFolderDataset === dataset.name}
                        >
                          {deletingFolderDataset === dataset.name ? (
                            <RefreshCw className="w-4 h-4 animate-spin mr-1" />
                          ) : (
                            <Trash2 className="w-4 h-4 mr-1" />
                          )}
                          Delete
                        </Button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={() => setManageDatasetsDialogOpen(false)}>
                  Close
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
          <Dialog open={dialogOpen} onOpenChange={(open) => {
          setDialogOpen(open)
          if (!open) {
            setEditingDataset(null)
            setFormData({ ...EMPTY_DATASET_FORM })
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
              <DialogTitle>{editingDataset ? 'Edit Dataset' : (isSegDataset ? 'Import Segmentation Dataset' : 'Import COCO Dataset')}</DialogTitle>
              <DialogDescription>
                {editingDataset ? 'Update dataset configuration.' : (isSegDataset ? 'Configure a PaddleSeg dataset using list files and number of classes.' : 'Select a dataset folder from your COCO directory.')}
              </DialogDescription>
            </DialogHeader>
            <form onSubmit={handleSubmit}>
              <div className="grid grid-cols-2 gap-4 py-4">
                {!editingDataset && (() => {
                  // Filter the dataset dropdown by the project's framework so
                  // PaddleSeg projects only see PaddleSeg datasets and vice
                  // versa. When no project is selected yet we default to COCO
                  // for backwards compatibility.
                  const expected: 'COCO' | 'PaddleSeg' = isSegDataset ? 'PaddleSeg' : 'COCO'
                  const filtered = availableDatasets.filter(d => d.format === expected)
                  return (
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
                          {filtered.length === 0 ? (
                            <SelectItem value="__empty__" disabled>
                              No {expected} folders found
                            </SelectItem>
                          ) : (
                            filtered.map((dataset) => (
                              <SelectItem key={dataset.name} value={dataset.name}>
                                {dataset.name} {!dataset.hasTrain && !dataset.hasVal && !dataset.hasAnnotations ? '(invalid)' : ''}
                              </SelectItem>
                            ))
                          )}
                        </SelectContent>
                      </Select>
                      {filtered.length === 0 && !loadingAvailableDatasets && (
                        <p className="text-xs text-amber-600">
                          {expected === 'PaddleSeg'
                            ? 'No PaddleSeg dataset folders found. Convert a labelme dataset first.'
                            : 'No dataset folders found. Please upload a dataset first.'}
                        </p>
                      )}
                    </div>
                  )
                })()}
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
                    onValueChange={handleProjectChange}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select project" />
                    </SelectTrigger>
                    <SelectContent>
                      {projects.map((project) => (
                        <SelectItem key={project.id} value={project.id}>
                          {project.name} <span className="text-xs text-muted-foreground ml-1">({project.framework})</span>
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
                    onValueChange={(value) => setFormData({ ...formData, format: value as DatasetFormat })}
                    disabled
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="COCO">COCO</SelectItem>
                      <SelectItem value="PaddleSeg">PaddleSeg</SelectItem>
                      <SelectItem value="AnomalyFolder">AnomalyFolder</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">Determined by project framework.</p>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="datasetDir">Dataset Directory</Label>
                  <Input
                    id="datasetDir"
                    value={formData.datasetDir}
                    onChange={(e) => setFormData({ ...formData, datasetDir: e.target.value })}
                    disabled={!isSegDataset && !isAnomalyDataset}
                    className={isSegDataset || isAnomalyDataset ? '' : 'bg-muted'}
                    placeholder={
                      isAnomalyDataset
                        ? 'D:/datasets/slider_v1'
                        : isSegDataset
                          ? 'D:/datasets/my_seg_dataset'
                          : undefined
                    }
                  />
                  <p className="text-xs text-muted-foreground">
                    {isAnomalyDataset
                      ? 'Absolute path to the folder that contains the directories below.'
                      : isSegDataset
                        ? 'Absolute path to the segmentation dataset root (contains images, masks and list files).'
                        : `COCO/${formData.name}/`}
                  </p>
                </div>
                {/* Anomaly datasets: four directories, no annotation files.
                    Only `normalDir` is trained on; the rest exist so the run can
                    be evaluated and a score threshold can be chosen at all. */}
                {isAnomalyDataset && (
                  <>
                    <div className="col-span-2 pt-2 border-t">
                      <h4 className="text-sm font-medium text-muted-foreground mb-2">
                        Image directories (relative to the dataset directory)
                      </h4>
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="normalDir">Normal images (training)</Label>
                      <Input
                        id="normalDir"
                        value={formData.normalDir}
                        onChange={(e) => setFormData({ ...formData, normalDir: e.target.value })}
                        placeholder="train_good"
                        required
                      />
                      <p className="text-xs text-muted-foreground">
                        Required. The only directory the model trains on.
                      </p>
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="normalTestDir">Normal images (evaluation)</Label>
                      <Input
                        id="normalTestDir"
                        value={formData.normalTestDir}
                        onChange={(e) => setFormData({ ...formData, normalTestDir: e.target.value })}
                        placeholder="test_good"
                      />
                      <p className="text-xs text-muted-foreground">
                        Optional. Held-out OK images; without them a slice of the training
                        images is used instead.
                      </p>
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="abnormalDir">Defect images (evaluation only)</Label>
                      <Input
                        id="abnormalDir"
                        value={formData.abnormalDir}
                        onChange={(e) => setFormData({ ...formData, abnormalDir: e.target.value })}
                        placeholder="test_ng"
                      />
                      <p className="text-xs text-muted-foreground">
                        Never trained on. Needed to compute AUROC and to pick the threshold.
                      </p>
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="maskDir">Defect masks</Label>
                      <Input
                        id="maskDir"
                        value={formData.maskDir}
                        onChange={(e) => setFormData({ ...formData, maskDir: e.target.value })}
                        placeholder="test_ng_mask"
                      />
                      <p className="text-xs text-muted-foreground">
                        Optional, one per defect image with the same file name. Required for
                        pixel-level metrics.
                      </p>
                    </div>
                    <div className="col-span-2 rounded-md border border-amber-500/40 bg-amber-500/5 p-3">
                      <p className="text-xs text-muted-foreground">
                        <span className="font-medium text-foreground">How the split works.</span>{' '}
                        anomalib builds the test set from the directories above and then takes
                        half of it as the validation set, which is what the anomaly-score
                        threshold is fitted on. Numbers reported on that half are therefore
                        optimistic. Keep a separate batch of defect images out of this dataset
                        and run a validation job against it for an unbiased figure.
                      </p>
                    </div>
                  </>
                )}

                {/* Train Set Paths */}
                {!isAnomalyDataset && (
                <div className="col-span-2 pt-2 border-t">
                  <h4 className="text-sm font-medium text-muted-foreground mb-2">Train Set</h4>
                </div>
                )}
                {!isSegDataset && !isAnomalyDataset && (
                  <div className="space-y-2">
                    <Label htmlFor="trainImagePath">Train Images Path</Label>
                    <Input
                      id="trainImagePath"
                      value={formData.trainImagePath}
                      disabled
                      className="bg-muted"
                    />
                  </div>
                )}
                {isAnomalyDataset ? null : isSegDataset ? (
                  <>
                    <div className="space-y-2">
                      <Label htmlFor="trainListPath">Train List File</Label>
                      <Input
                        id="trainListPath"
                        value={formData.trainAnnoPath}
                        onChange={(e) => setFormData({ ...formData, trainAnnoPath: e.target.value })}
                        placeholder="train.txt"
                      />
                      <p className="text-xs text-muted-foreground">Relative to dataset root, e.g. train.txt</p>
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="segNumClasses">Number of Classes</Label>
                      <Input
                        id="segNumClasses"
                        type="number"
                        min={1}
                        value={formData.numClasses}
                        onChange={(e) => setFormData({ ...formData, numClasses: parseInt(e.target.value, 10) || 1 })}
                      />
                      <p className="text-xs text-muted-foreground">Includes background as class 0.</p>
                    </div>
                  </>
                ) : (
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
                )}
                {/* Eval Set Paths */}
                {!isAnomalyDataset && (
                <div className="col-span-2 pt-2 border-t">
                  <h4 className="text-sm font-medium text-muted-foreground mb-2">Validation Set</h4>
                </div>
                )}
                {!isSegDataset && !isAnomalyDataset && (
                  <div className="space-y-2">
                    <Label htmlFor="evalImagePath">Eval Images Path</Label>
                    <Input
                      id="evalImagePath"
                      value={formData.evalImagePath}
                      disabled
                      className="bg-muted"
                    />
                  </div>
                )}
                {isAnomalyDataset ? null : isSegDataset ? (
                  <div className="space-y-2">
                    <Label htmlFor="valListPath">Validation List File</Label>
                    <Input
                      id="valListPath"
                      value={formData.evalAnnoPath}
                      onChange={(e) => setFormData({ ...formData, evalAnnoPath: e.target.value })}
                      placeholder="val.txt"
                    />
                    <p className="text-xs text-muted-foreground">Relative to dataset root, e.g. val.txt</p>
                  </div>
                ) : (
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
                )}
                {/* Detected Classes Display */}
                {!isSegDataset && !isAnomalyDataset && formData.numClasses > 0 && (
                  <div className="col-span-2 pt-2">
                    <p className="text-sm text-green-600">
                      Detected {formData.numClasses} classes from train annotations
                    </p>
                  </div>
                )}
              </div>
              <DialogFooter>
                {/* The "no folders scanned" guard only applies to COCO, which
                    needs an annotation file picked from the scan. Segmentation
                    and anomaly datasets are described by hand-typed paths. */}
                <Button
                  type="submit"
                  disabled={
                    !editingDataset && !isSegDataset && !isAnomalyDataset && availableDatasets.length === 0
                  }
                >
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
                      {isSegSelected ? (
                        <>
                          <div className="text-2xl font-bold">{selectedDataset.numTrainImages + selectedDataset.numEvalImages}</div>
                          <div className="text-xs text-muted-foreground">Total Images</div>
                        </>
                      ) : (
                        <>
                          <div className="text-2xl font-bold">{selectedDataset.numAnnotations}</div>
                          <div className="text-xs text-muted-foreground">Annotations</div>
                        </>
                      )}
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

                  {/* Class Distribution Chart (COCO) */}
                  {!isSegSelected && selectedDataset.classStats && (() => {
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

                  {/* Class Legend + Distribution (PaddleSeg) */}
                  {isSegSelected && (() => {
                    const segStats = getClassStats(selectedDataset.classStats).train as Array<{ id: number; name: string; count: number; imageCount: number }>
                    const classList = segStats.length > 0
                      ? segStats
                      : Array.from({ length: selectedDataset.numClasses || 0 }, (_, i) => ({ id: i, name: `class_${i}`, count: 0, imageCount: 0 }))
                    const colors = getSegColorMap(Math.max(classList.length, 1))
                    const hasDistribution = classList.some(c => (c.imageCount || 0) > 0 || (c.count || 0) > 0)
                    let sampledMasks = 0
                    try { sampledMasks = JSON.parse(selectedDataset.classStats || '{}').sampledMasks || 0 } catch { /* ignore */ }
                    return (
                      <div className="space-y-4">
                        <div>
                          <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
                            <Layers className="w-4 h-4" /> Classes ({classList.length})
                          </h4>
                          <div className="flex flex-wrap gap-2">
                            {classList.map((c, i) => (
                              <div key={c.id} className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-muted/50">
                                <span
                                  className="w-3 h-3 rounded-sm border border-border"
                                  style={{ backgroundColor: rgbToCss(colors[i]) }}
                                />
                                <span className="text-xs">{c.name}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                        {hasDistribution ? (
                          <div>
                            <h4 className="text-sm font-medium mb-1">Class Distribution</h4>
                            <p className="text-xs text-muted-foreground mb-3">
                              Images containing each class{sampledMasks ? ` (sampled from ${sampledMasks} masks)` : ''}
                            </p>
                            <ChartContainer config={chartConfig} className="h-[220px] w-full">
                              <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={classList} layout="vertical" margin={{ left: 12, right: 12 }}>
                                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                                  <XAxis type="number" allowDecimals={false} className="text-xs" />
                                  <YAxis type="category" dataKey="name" width={90} className="text-xs" />
                                  <Tooltip />
                                  <Bar dataKey="imageCount" name="Images" radius={[0, 4, 4, 0]}>
                                    {classList.map((c, i) => (
                                      <Cell key={c.id} fill={rgbToCss(colors[i])} />
                                    ))}
                                  </Bar>
                                </BarChart>
                              </ResponsiveContainer>
                            </ChartContainer>
                          </div>
                        ) : (
                          <p className="text-xs text-muted-foreground">
                            Click &quot;Compute Class Distribution&quot; below to analyze the per-class pixel/image distribution from a sample of masks.
                          </p>
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
                    {parsing
                      ? (isSegSelected ? 'Analyzing masks...' : 'Parsing...')
                      : (isSegSelected ? 'Compute Class Distribution' : 'Parse Dataset Statistics')}
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
              {previewType === 'segmentation'
                ? 'Preview images with their segmentation mask overlay.'
                : 'Preview dataset images with annotations. Select a category to filter samples.'}
            </DialogDescription>
          </DialogHeader>
          
          {previewType === 'segmentation' ? (
            <div className="flex flex-col gap-3 py-2">
              <div className="flex items-center gap-4 flex-wrap">
                <span className="text-sm text-muted-foreground">{previewSamples.length} samples</span>
                <div className="flex items-center gap-2">
                  <Label className="text-xs whitespace-nowrap">Mask opacity</Label>
                  <Slider
                    value={[Math.round(overlayOpacity * 100)]}
                    min={0}
                    max={100}
                    step={5}
                    onValueChange={(v) => setOverlayOpacity((v[0] ?? 50) / 100)}
                    className="w-[140px]"
                  />
                  <span className="text-xs w-9 text-right">{Math.round(overlayOpacity * 100)}%</span>
                </div>
                <Button
                  variant={hideBackground ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setHideBackground(v => !v)}
                >
                  {hideBackground ? 'Background hidden' : 'Background shown'}
                </Button>
              </div>
              <div className="flex flex-wrap gap-2">
                {previewCategories.map((cat) => (
                  <div key={cat.id} className="flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-muted/50">
                    <span
                      className="w-3 h-3 rounded-sm border border-border"
                      style={{ backgroundColor: rgbToCss(getSegColorMap(Math.max(previewCategories.length, 1))[cat.id]) }}
                    />
                    <span className="text-xs">{cat.name}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
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
          )}

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
                      {previewType === 'segmentation' && sample.maskPath ? (
                        <SegSampleView
                          imagePath={sample.imagePath}
                          maskPath={sample.maskPath}
                          numClasses={selectedDataset?.numClasses}
                          opacity={overlayOpacity}
                          hideBackground={hideBackground}
                          maxSize={400}
                          className="w-full h-full object-contain"
                        />
                      ) : (
                        <>
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
                        </>
                      )}
                    </div>
                    <CardContent className="p-2">
                      <p className="text-xs truncate text-muted-foreground" title={sample.fileName}>
                        {sample.fileName}
                      </p>
                      {previewType !== 'segmentation' && (
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
                      )}
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
            {previewType === 'segmentation' && (
              <>
                <div className="w-px h-6 bg-border mx-2" />
                <Label className="text-xs whitespace-nowrap">Mask</Label>
                <Slider
                  value={[Math.round(overlayOpacity * 100)]}
                  min={0}
                  max={100}
                  step={5}
                  onValueChange={(v) => setOverlayOpacity((v[0] ?? 50) / 100)}
                  className="w-[120px]"
                />
                <span className="text-xs w-9 text-right">{Math.round(overlayOpacity * 100)}%</span>
                <Button
                  variant={hideBackground ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setHideBackground(v => !v)}
                >
                  {hideBackground ? 'BG hidden' : 'BG shown'}
                </Button>
              </>
            )}
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
                {previewType === 'segmentation' && selectedSample.maskPath ? (
                  <SegSampleView
                    imagePath={selectedSample.imagePath}
                    maskPath={selectedSample.maskPath}
                    numClasses={selectedDataset?.numClasses}
                    opacity={overlayOpacity}
                    hideBackground={hideBackground}
                    maxSize={2048}
                    className="max-w-none"
                  />
                ) : (
                  <>
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
                  </>
                )}
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
