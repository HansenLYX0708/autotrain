'use client'

import { useState, useRef, useCallback, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Upload,
  Trash2,
  Download,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Save,
  Plus,
  FolderOpen,
  Image as ImageIcon,
  ChevronLeft,
  ChevronRight,
  Check,
  Loader2,
} from 'lucide-react'
import { toast } from '@/hooks/use-toast'
import { useAuth } from '@/contexts/auth-context'

interface BoundingBox {
  id: string
  label: string
  x1: number
  y1: number
  x2: number
  y2: number
}

interface ImageFile {
  name: string
  dataUrl: string
  width: number
  height: number
  boxes: BoundingBox[]
  hasUnsavedChanges: boolean
}

interface LabelMeAnnotation {
  version: string
  flags: Record<string, never>
  shapes: Array<{
    label: string
    points: [number, number][]
    group_id: null
    shape_type: string
    flags: Record<string, never>
  }>
  imagePath: string
  imageData: string | null
  imageHeight: number
  imageWidth: number
}

const defaultLabels: string[] = []

// Helper to get initial labels from localStorage
function getInitialLabels(): string[] {
  if (typeof window === 'undefined') return []
  const savedLabels = localStorage.getItem('annotationLabels')
  if (savedLabels) {
    try {
      const parsed = JSON.parse(savedLabels)
      return Array.isArray(parsed) ? parsed : []
    } catch {
      return []
    }
  }
  return []
}

export function AnnotationPage() {
  const { user, isAuthenticated, isLoading: authLoading } = useAuth()
  const router = useRouter()

  // Check if user is admin
  useEffect(() => {
    if (!authLoading && isAuthenticated && user?.role !== 'admin') {
      toast({
        title: 'Access Denied',
        description: 'Only administrators can access the annotation page.',
        variant: 'destructive',
      })
      router.push('/')
    }
  }, [authLoading, isAuthenticated, user, router])

  // Image list state
  const [images, setImages] = useState<ImageFile[]>([])
  const [currentIndex, setCurrentIndex] = useState(-1)
  const [folderPath, setFolderPath] = useState<string>('')
  
  // Annotation state
  const [labels, setLabels] = useState<string[]>(getInitialLabels)
  const [selectedLabel, setSelectedLabel] = useState<string>('')
  const [isDrawing, setIsDrawing] = useState(false)
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null)
  const [currentBox, setCurrentBox] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null)
  const [selectedBoxId, setSelectedBoxId] = useState<string | null>(null)
  
  // View state
  const [zoom, setZoom] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const [isPanning, setIsPanning] = useState(false)
  const [panStart, setPanStart] = useState<{ x: number; y: number } | null>(null)
  
  // Dialog state
  const [newLabelDialog, setNewLabelDialog] = useState(false)
  const [newLabelName, setNewLabelName] = useState('')
  const [saving, setSaving] = useState(false)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const imageRef = useRef<HTMLImageElement | null>(null)

  // Current image
  const currentImage = currentIndex >= 0 ? images[currentIndex] : null

  // Save labels to localStorage
  useEffect(() => {
    localStorage.setItem('annotationLabels', JSON.stringify(labels))
  }, [labels])

  // Handle folder selection
  const handleFolderSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return

    const imageFiles: ImageFile[] = []
    const jsonFiles: Map<string, File> = new Map()

    // Separate image and JSON files
    for (let i = 0; i < files.length; i++) {
      const file = files[i]
      const ext = file.name.toLowerCase()
      if (ext.endsWith('.json')) {
        jsonFiles.set(file.name.replace('.json', ''), file)
      }
    }

    // Process image files
    for (let i = 0; i < files.length; i++) {
      const file = files[i]
      const ext = file.name.toLowerCase()
      if (!ext.match(/\.(jpg|jpeg|png|bmp|gif|webp)$/)) continue

      try {
        const dataUrl = await readFileAsDataURL(file)
        const dims = await getImageDimensions(dataUrl)
        
        // Check for existing JSON annotation
        const baseName = file.name.replace(/\.[^.]+$/, '')
        let boxes: BoundingBox[] = []
        
        const jsonFile = jsonFiles.get(baseName)
        if (jsonFile) {
          try {
            const jsonContent = await readFileAsText(jsonFile)
            const annotation = JSON.parse(jsonContent) as LabelMeAnnotation
            boxes = annotation.shapes.map((shape, idx) => ({
              id: `box-${idx}-${Date.now()}`,
              label: shape.label,
              x1: Math.min(shape.points[0][0], shape.points[2][0]),
              y1: Math.min(shape.points[0][1], shape.points[2][1]),
              x2: Math.max(shape.points[0][0], shape.points[2][0]),
              y2: Math.max(shape.points[0][1], shape.points[2][1]),
            }))
          } catch {
            console.warn(`Failed to parse JSON for ${file.name}`)
          }
        }

        imageFiles.push({
          name: file.name,
          dataUrl,
          width: dims.width,
          height: dims.height,
          boxes,
          hasUnsavedChanges: false,
        })
      } catch (err) {
        console.error(`Failed to load ${file.name}:`, err)
      }
    }

    // Sort by name
    imageFiles.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }))

    setImages(imageFiles)
    setFolderPath(files[0].webkitRelativePath.split('/')[0] || 'folder')
    setCurrentIndex(imageFiles.length > 0 ? 0 : -1)
    setZoom(1)
    setOffset({ x: 0, y: 0 })
    setSelectedBoxId(null)

    const loadedCount = imageFiles.filter(f => f.boxes.length > 0).length
    toast({
      title: 'Folder loaded',
      description: `${imageFiles.length} images found, ${loadedCount} with annotations`,
    })
  }, [])

  // Helper functions
  function readFileAsDataURL(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(reader.result as string)
      reader.onerror = reject
      reader.readAsDataURL(file)
    })
  }

  function readFileAsText(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(reader.result as string)
      reader.onerror = reject
      reader.readAsText(file)
    })
  }

  function getImageDimensions(dataUrl: string): Promise<{ width: number; height: number }> {
    return new Promise((resolve, reject) => {
      const img = new Image()
      img.onload = () => resolve({ width: img.width, height: img.height })
      img.onerror = reject
      img.src = dataUrl
    })
  }

  // Load image to canvas
  useEffect(() => {
    if (!currentImage) {
      imageRef.current = null
      return
    }

    const img = new Image()
    img.onload = () => {
      imageRef.current = img
    }
    img.src = currentImage.dataUrl
  }, [currentImage])

  // Canvas rendering
  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx || !currentImage) return

    const container = containerRef.current
    if (!container) return

    canvas.width = container.clientWidth
    canvas.height = container.clientHeight

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.save()

    // Apply zoom and offset
    ctx.translate(offset.x, offset.y)
    ctx.scale(zoom, zoom)

    // Draw image
    if (imageRef.current) {
      ctx.drawImage(imageRef.current, 0, 0)
    }

    // Draw existing boxes
    currentImage.boxes.forEach((box) => {
      const isSelected = box.id === selectedBoxId
      ctx.strokeStyle = isSelected ? '#00ff00' : '#ff0000'
      ctx.lineWidth = isSelected ? 3 / zoom : 2 / zoom
      ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1)

      // Draw label background
      ctx.fillStyle = isSelected ? '#00ff00' : '#ff0000'
      const labelText = box.label
      const labelWidth = ctx.measureText(labelText).width + 8
      const labelHeight = 18
      ctx.fillRect(box.x1, box.y1 - labelHeight, labelWidth, labelHeight)

      // Draw label text
      ctx.fillStyle = '#ffffff'
      ctx.font = `${12 / zoom}px sans-serif`
      ctx.fillText(labelText, box.x1 + 4, box.y1 - 5)
    })

    // Draw current box being drawn
    if (currentBox && isDrawing) {
      ctx.strokeStyle = '#00ff00'
      ctx.lineWidth = 2 / zoom
      ctx.setLineDash([5 / zoom, 5 / zoom])
      ctx.strokeRect(
        Math.min(currentBox.x1, currentBox.x2),
        Math.min(currentBox.y1, currentBox.y2),
        Math.abs(currentBox.x2 - currentBox.x1),
        Math.abs(currentBox.y2 - currentBox.y1)
      )
      ctx.setLineDash([])
    }

    ctx.restore()
  }, [currentImage, currentBox, isDrawing, zoom, offset, selectedBoxId])

  // Get image coordinates from mouse event
  const getImageCoords = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current
    if (!canvas) return null

    const rect = canvas.getBoundingClientRect()
    const x = (e.clientX - rect.left - offset.x) / zoom
    const y = (e.clientY - rect.top - offset.y) / zoom
    return { x, y }
  }, [zoom, offset])

  // Mouse handlers
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (!currentImage) return

    const coords = getImageCoords(e)
    if (!coords) return

    if (e.button === 1 || (e.button === 0 && e.altKey)) {
      setIsPanning(true)
      setPanStart({ x: e.clientX - offset.x, y: e.clientY - offset.y })
      return
    }

    if (e.button === 0) {
      // First check if clicking on an existing box
      for (let i = currentImage.boxes.length - 1; i >= 0; i--) {
        const box = currentImage.boxes[i]
        if (coords.x >= box.x1 && coords.x <= box.x2 && coords.y >= box.y1 && coords.y <= box.y2) {
          setSelectedBoxId(box.id === selectedBoxId ? null : box.id)
          return
        }
      }

      // No box clicked, check if label is selected for drawing
      if (!selectedLabel) {
        toast({
          title: 'No label selected',
          description: 'Please add and select a label before drawing',
          variant: 'destructive'
        })
        return
      }
      setIsDrawing(true)
      setDrawStart(coords)
      setCurrentBox({ x1: coords.x, y1: coords.y, x2: coords.x, y2: coords.y })
      setSelectedBoxId(null)
    }
  }, [currentImage, getImageCoords, offset, selectedLabel, selectedBoxId])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (isPanning && panStart) {
      setOffset({
        x: e.clientX - panStart.x,
        y: e.clientY - panStart.y,
      })
      return
    }

    if (!isDrawing || !drawStart) return

    const coords = getImageCoords(e)
    if (!coords) return

    setCurrentBox({
      x1: drawStart.x,
      y1: drawStart.y,
      x2: coords.x,
      y2: coords.y,
    })
  }, [isDrawing, drawStart, isPanning, panStart, getImageCoords])

  const handleMouseUp = useCallback(() => {
    if (isPanning) {
      setIsPanning(false)
      setPanStart(null)
      return
    }

    if (!isDrawing || !currentBox || !currentImage) {
      setIsDrawing(false)
      setDrawStart(null)
      setCurrentBox(null)
      return
    }

    const width = Math.abs(currentBox.x2 - currentBox.x1)
    const height = Math.abs(currentBox.y2 - currentBox.y1)

    if (width > 5 && height > 5) {
      const newBox: BoundingBox = {
        id: `box-${Date.now()}`,
        label: selectedLabel,
        x1: Math.min(currentBox.x1, currentBox.x2),
        y1: Math.min(currentBox.y1, currentBox.y2),
        x2: Math.max(currentBox.x1, currentBox.x2),
        y2: Math.max(currentBox.y1, currentBox.y2),
      }

      updateCurrentImage({
        ...currentImage,
        boxes: [...currentImage.boxes, newBox],
        hasUnsavedChanges: true,
      })
    }

    setIsDrawing(false)
    setDrawStart(null)
    setCurrentBox(null)
  }, [isDrawing, currentBox, currentImage, isPanning, selectedLabel])

  // Update current image helper
  const updateCurrentImage = useCallback((updatedImage: ImageFile) => {
    setImages(prev => prev.map((img, idx) => idx === currentIndex ? updatedImage : img))
  }, [currentIndex])

  // Handle box selection (for click outside boxes)
  const handleCanvasClick = useCallback((e: React.MouseEvent) => {
    // Selection is now handled in handleMouseDown
    // This is kept for potential future use
  }, [])

  // Delete selected box
  const deleteSelectedBox = useCallback(() => {
    if (!currentImage || !selectedBoxId) return

    updateCurrentImage({
      ...currentImage,
      boxes: currentImage.boxes.filter(b => b.id !== selectedBoxId),
      hasUnsavedChanges: true,
    })
    setSelectedBoxId(null)
    toast({ title: 'Box deleted' })
  }, [currentImage, selectedBoxId, updateCurrentImage])

  // Change label of selected box
  const changeSelectedBoxLabel = useCallback((newLabel: string) => {
    if (!currentImage || !selectedBoxId) return

    updateCurrentImage({
      ...currentImage,
      boxes: currentImage.boxes.map(b => b.id === selectedBoxId ? { ...b, label: newLabel } : b),
      hasUnsavedChanges: true,
    })
  }, [currentImage, selectedBoxId, updateCurrentImage])

  // Navigation
  const goToImage = useCallback((index: number) => {
    if (index < 0 || index >= images.length) return
    setCurrentIndex(index)
    setZoom(1)
    setOffset({ x: 0, y: 0 })
    setSelectedBoxId(null)
  }, [images.length])

  const goToPrev = () => goToImage(currentIndex - 1)
  const goToNext = () => goToImage(currentIndex + 1)

  // Add new label
  const handleAddLabel = useCallback(() => {
    if (!newLabelName.trim()) return
    if (labels.includes(newLabelName.trim())) {
      toast({ title: 'Label already exists', variant: 'destructive' })
      return
    }
    setLabels([...labels, newLabelName.trim()])
    setSelectedLabel(newLabelName.trim())
    setNewLabelName('')
    setNewLabelDialog(false)
    toast({ title: 'Label added' })
  }, [newLabelName, labels])

  // Save current annotation to file
  const saveAnnotation = useCallback(async () => {
    if (!currentImage) return

    setSaving(true)
    try {
      const shapes = currentImage.boxes.map(box => ({
        label: box.label,
        points: [
          [box.x1, box.y1],
          [box.x1, box.y2],
          [box.x2, box.y2],
          [box.x2, box.y1],
        ] as [number, number][],
        group_id: null,
        shape_type: 'rectangle',
        flags: {},
      }))

      const base64Data = currentImage.dataUrl.split(',')[1] || null

      const labelMeJson: LabelMeAnnotation = {
        version: '5.0.1',
        flags: {},
        shapes,
        imagePath: currentImage.name,
        imageData: base64Data,
        imageHeight: currentImage.height,
        imageWidth: currentImage.width,
      }

      // Save via API
      const response = await fetch('/api/annotation/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          fileName: currentImage.name.replace(/\.[^.]+$/, '.json'),
          annotation: labelMeJson,
        }),
      })

      if (response.ok) {
        updateCurrentImage({ ...currentImage, hasUnsavedChanges: false })
        toast({ title: 'Annotation saved', description: `${currentImage.name.replace(/\.[^.]+$/, '.json')}` })
      } else {
        // Fallback to download
        const blob = new Blob([JSON.stringify(labelMeJson, null, 2)], { type: 'application/json' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = currentImage.name.replace(/\.[^.]+$/, '.json')
        a.click()
        URL.revokeObjectURL(url)
        updateCurrentImage({ ...currentImage, hasUnsavedChanges: false })
        toast({ title: 'Annotation downloaded' })
      }
    } catch (err) {
      toast({ title: 'Error saving annotation', variant: 'destructive' })
    } finally {
      setSaving(false)
    }
  }, [currentImage, updateCurrentImage])

  // Save all annotations
  const saveAllAnnotations = useCallback(async () => {
    setSaving(true)
    let saved = 0
    for (const image of images) {
      if (image.boxes.length > 0) {
        try {
          const shapes = image.boxes.map(box => ({
            label: box.label,
            points: [
              [box.x1, box.y1],
              [box.x1, box.y2],
              [box.x2, box.y2],
              [box.x2, box.y1],
            ] as [number, number][],
            group_id: null,
            shape_type: 'rectangle',
            flags: {},
          }))

          const base64Data = image.dataUrl.split(',')[1] || null

          const labelMeJson: LabelMeAnnotation = {
            version: '5.0.1',
            flags: {},
            shapes,
            imagePath: image.name,
            imageData: base64Data,
            imageHeight: image.height,
            imageWidth: image.width,
          }

          const response = await fetch('/api/annotation/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              fileName: image.name.replace(/\.[^.]+$/, '.json'),
              annotation: labelMeJson,
            }),
          })

          if (response.ok) saved++
        } catch {
          // Skip on error
        }
      }
    }
    
    // Clear unsaved flags
    setImages(prev => prev.map(img => ({ ...img, hasUnsavedChanges: false })))
    setSaving(false)
    toast({ title: 'Saved', description: `${saved} annotations saved` })
  }, [images])

  // Zoom handlers
  const handleZoomIn = () => setZoom(z => Math.min(z * 1.2, 5))
  const handleZoomOut = () => setZoom(z => Math.max(z / 1.2, 0.1))
  const handleResetView = () => {
    setZoom(1)
    setOffset({ x: 0, y: 0 })
  }

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (selectedBoxId && !newLabelDialog) {
          e.preventDefault()
          deleteSelectedBox()
        }
      }
      if (e.key === 'Escape') {
        setSelectedBoxId(null)
      }
      if (e.key === 'ArrowLeft' && !newLabelDialog) {
        e.preventDefault()
        goToPrev()
      }
      if (e.key === 'ArrowRight' && !newLabelDialog) {
        e.preventDefault()
        goToNext()
      }
      if ((e.key === 's' || e.key === 'S') && (e.ctrlKey || e.metaKey)) {
        e.preventDefault()
        saveAnnotation()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [selectedBoxId, deleteSelectedBox, newLabelDialog, goToPrev, goToNext, saveAnnotation])

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Annotation</h1>
          <p className="text-muted-foreground">
            {images.length > 0 
              ? `${folderPath} - ${images.length} images` 
              : 'Load a folder to start annotating'}
          </p>
        </div>
        <div className="flex gap-2">
          {images.length > 0 && (
            <>
              <Button variant="outline" onClick={saveAnnotation} disabled={saving}>
                {saving ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Save className="w-4 h-4 mr-2" />}
                Save (Ctrl+S)
              </Button>
              <Button variant="outline" onClick={saveAllAnnotations} disabled={saving}>
                <Download className="w-4 h-4 mr-2" />
                Save All
              </Button>
            </>
          )}
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-5">
        {/* Left Sidebar - Image List */}
        <div className="space-y-4">
          {/* Folder Upload */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">Folder</CardTitle>
            </CardHeader>
            <CardContent>
              <label className="flex items-center justify-center w-full h-16 border-2 border-dashed rounded-lg cursor-pointer hover:bg-accent transition-colors">
                <div className="text-center">
                  <FolderOpen className="w-5 h-5 mx-auto mb-1 text-muted-foreground" />
                  <span className="text-xs text-muted-foreground">
                    {images.length > 0 ? 'Change Folder' : 'Select Folder'}
                  </span>
                </div>
                <input
                  type="file"
                  /* @ts-expect-error webkitdirectory is not in types */
                  webkitdirectory=""
                  directory=""
                  multiple
                  className="hidden"
                  onChange={handleFolderSelect}
                />
              </label>
            </CardContent>
          </Card>

          {/* Image List */}
          {images.length > 0 && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Images ({currentIndex + 1}/{images.length})</CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                <ScrollArea className="h-[300px]">
                  <div className="space-y-1 p-3 pt-0">
                    {images.map((img, idx) => (
                      <button
                        key={idx}
                        onClick={() => goToImage(idx)}
                        className={`w-full text-left px-2 py-1.5 rounded text-xs flex items-center gap-2 transition-colors ${
                          idx === currentIndex
                            ? 'bg-primary text-primary-foreground'
                            : 'hover:bg-accent'
                        }`}
                      >
                        <ImageIcon className="w-3 h-3 flex-shrink-0" />
                        <span className="truncate flex-1">{img.name}</span>
                        <div className="flex items-center gap-1 flex-shrink-0">
                          {img.boxes.length > 0 && (
                            <Badge variant={idx === currentIndex ? 'secondary' : 'outline'} className="h-4 px-1 text-[10px]">
                              {img.boxes.length}
                            </Badge>
                          )}
                          {img.hasUnsavedChanges && (
                            <div className="w-2 h-2 rounded-full bg-orange-500" />
                          )}
                        </div>
                      </button>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          )}

          {/* Navigation */}
          {images.length > 0 && (
            <Card>
              <CardContent className="p-3">
                <div className="flex items-center justify-between">
                  <Button variant="outline" size="sm" onClick={goToPrev} disabled={currentIndex <= 0}>
                    <ChevronLeft className="w-4 h-4" />
                  </Button>
                  <span className="text-xs text-muted-foreground">
                    {currentIndex + 1} / {images.length}
                  </span>
                  <Button variant="outline" size="sm" onClick={goToNext} disabled={currentIndex >= images.length - 1}>
                    <ChevronRight className="w-4 h-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Center - Canvas */}
        <div className="lg:col-span-3">
          <Card className="h-[calc(100vh-220px)]">
            <CardContent className="p-0 h-full">
              {currentImage ? (
                <div
                  ref={containerRef}
                  className="relative w-full h-full overflow-hidden bg-gray-900"
                >
                  <canvas
                    ref={canvasRef}
                    className="w-full h-full"
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onClick={handleCanvasClick}
                    onContextMenu={e => e.preventDefault()}
                    style={{ cursor: isPanning ? 'grabbing' : 'crosshair' }}
                  />
                  {/* Status bar */}
                  <div className="absolute bottom-0 left-0 right-0 bg-black/50 text-white text-xs p-2 flex justify-between">
                    <span>{currentImage.name}</span>
                    <span>{currentImage.width} x {currentImage.height}</span>
                    <span>{currentImage.boxes.length} boxes</span>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
                  <FolderOpen className="w-16 h-16 mb-4 opacity-50" />
                  <p className="text-lg mb-2">No folder loaded</p>
                  <p className="text-sm">Select a folder with images to start</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Right Sidebar - Tools */}
        <div className="space-y-4">
          {/* Labels */}
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm">Labels</CardTitle>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 w-6 p-0"
                  onClick={() => setNewLabelDialog(true)}
                >
                  <Plus className="w-4 h-4" />
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              {labels.length === 0 ? (
                <div className="text-center py-4">
                  <p className="text-sm text-muted-foreground mb-3">No labels yet</p>
                  <Button variant="outline" size="sm" onClick={() => setNewLabelDialog(true)}>
                    <Plus className="w-4 h-4 mr-1" />
                    Add Label
                  </Button>
                </div>
              ) : (
                <>
                  <Select value={selectedLabel} onValueChange={setSelectedLabel}>
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder="Select a label" />
                    </SelectTrigger>
                    <SelectContent>
                      {labels.map(label => (
                        <SelectItem key={label} value={label}>{label}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>

                  {/* Label stats */}
                  {currentImage && (
                    <div className="pt-2 border-t">
                      <div className="text-xs text-muted-foreground mb-2">Current Image:</div>
                      <div className="space-y-1">
                        {labels.map(label => {
                          const count = currentImage.boxes.filter(b => b.label === label).length
                          if (count === 0) return null
                          return (
                            <div key={label} className="flex items-center justify-between text-xs">
                              <span>{label}</span>
                              <Badge variant="secondary" className="h-5">{count}</Badge>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  )}
                </>
              )}
            </CardContent>
          </Card>

          {/* View Controls */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">View</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex gap-2">
                <Button variant="outline" size="sm" onClick={handleZoomIn}>
                  <ZoomIn className="w-4 h-4" />
                </Button>
                <Button variant="outline" size="sm" onClick={handleZoomOut}>
                  <ZoomOut className="w-4 h-4" />
                </Button>
                <Button variant="outline" size="sm" onClick={handleResetView}>
                  <RotateCcw className="w-4 h-4" />
                </Button>
              </div>
              <div className="text-xs text-muted-foreground">
                Zoom: {(zoom * 100).toFixed(0)}%
              </div>
              <div className="text-xs text-muted-foreground space-y-1">
                <p>• Drag to draw box</p>
                <p>• Alt+drag to pan</p>
                <p>• Click box to select</p>
                <p>• Delete to remove</p>
                <p>• ←/→ to navigate</p>
                <p>• Ctrl+S to save</p>
              </div>
            </CardContent>
          </Card>

          {/* Selected Box */}
          {selectedBoxId && currentImage && (
            <Card className="border-primary">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm">Selected Box</CardTitle>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 w-6 p-0 text-destructive"
                    onClick={deleteSelectedBox}
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-2">
                  <Label className="text-xs">Change Label</Label>
                  <Select
                    value={currentImage.boxes.find(b => b.id === selectedBoxId)?.label}
                    onValueChange={changeSelectedBoxLabel}
                  >
                    <SelectTrigger className="w-full">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {labels.map(label => (
                        <SelectItem key={label} value={label}>{label}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                {(() => {
                  const box = currentImage.boxes.find(b => b.id === selectedBoxId)
                  if (!box) return null
                  return (
                    <div className="text-xs text-muted-foreground space-y-1">
                      <p>Position: ({box.x1.toFixed(0)}, {box.y1.toFixed(0)})</p>
                      <p>Size: {(box.x2 - box.x1).toFixed(0)} x {(box.y2 - box.y1).toFixed(0)}</p>
                    </div>
                  )
                })()}
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* New Label Dialog */}
      <Dialog open={newLabelDialog} onOpenChange={setNewLabelDialog}>
        <DialogContent className="max-w-sm">
          <DialogHeader>
            <DialogTitle>Add New Label</DialogTitle>
            <DialogDescription>Enter a new label name</DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <Input
              value={newLabelName}
              onChange={e => setNewLabelName(e.target.value)}
              placeholder="Enter label name"
              onKeyDown={e => e.key === 'Enter' && handleAddLabel()}
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setNewLabelDialog(false)}>Cancel</Button>
            <Button onClick={handleAddLabel}>Add</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
