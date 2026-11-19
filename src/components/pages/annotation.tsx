'use client'

import { useCallback, useEffect, useLayoutEffect, useReducer, useRef, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import { useAuth } from '@/contexts/auth-context'
import { resolveUploadedLabelmeImage, type ImageNameIndex } from '@/lib/labelme-image-match'
import { toast } from '@/hooks/use-toast'
import { Download, Save, Undo2, Redo2, Trash2, ZoomIn, ZoomOut, FolderOpen, Crosshair, MousePointer2, Square, RotateCw, Archive, ArchiveRestore, Pencil, Plus } from 'lucide-react'
import {
  type AnnotationImage, type AnnotationObject, type AnnotationProject, type Point, type Split,
  boxCorners, boxFromThreePoints, containsPoint, fromLabelme, imageIssues, insideImage,
  newProject, projectIssues, projectSchema, toLabelme,
  resizeBox, rotateBox,
} from '@/lib/localization-annotations'
import { predictionsSchema, predictionToImage } from '@/lib/localization-predictions'
import { uuid } from '@/lib/browser-safe'

type History = { present: AnnotationProject; past: AnnotationProject[]; future: AnnotationProject[] }
type Action = { type: 'load' | 'edit'; project: AnnotationProject } | { type: 'undo' | 'redo' }
function reducer(state: History, action: Action): History {
  if (action.type === 'load') return { present: action.project, past: [], future: [] }
  if (action.type === 'edit') return { present: action.project, past: [...state.past.slice(-49), state.present], future: [] }
  if (action.type === 'undo' && state.past.length) return {
    present: state.past[state.past.length - 1], past: state.past.slice(0, -1), future: [state.present, ...state.future],
  }
  if (action.type === 'redo' && state.future.length) return {
    present: state.future[0], past: [...state.past, state.present], future: state.future.slice(1),
  }
  return state
}

function download(name: string, value: unknown) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }))
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = name
  anchor.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
function errorMessage(error: unknown) { return error instanceof Error ? error.message : String(error) }
function report(error: unknown) { toast({ title: 'Action failed', description: errorMessage(error), variant: 'destructive' }) }
const selectClass = 'h-9 w-full rounded-md border bg-background px-2 text-sm'
const splitNames = { unassigned: 'Unassigned', train: 'Train', val: 'Val', test: 'Test' }
type Tool = 'select' | 'rectangle' | 'obb' | 'point'
type Asset = { url: string; width: number; height: number }

type AnnotationSetRow = {
  id: string; name: string; images: number; reviewedImages: number; instances: number; labels: number
  status: 'active' | 'archived'; activeSince: string; archiveAt?: string
  archivedAt?: string; archivedReason?: 'auto' | 'manual'; createdAt: string; modified: string
}

const clampZoom = (value: number) => Math.min(20, Math.max(0.02, value))
const SETS_PAGE_SIZE = 20

/**
 * "Annotation sets" management tab: lists saved sets (active / archived / all),
 * auto-archive countdown, and the rename / archive / export / delete actions.
 */
function AnnotationSetsPanel({ refresh, onOpen, onNew }: { refresh: number; onOpen: (id: string) => Promise<void>; onNew: () => void }) {
  const [status, setStatus] = useState<'active' | 'archived' | 'all'>('active')
  const [search, setSearch] = useState('')
  const [query, setQuery] = useState('')
  const [page, setPage] = useState(1)
  const [sets, setSets] = useState<AnnotationSetRow[] | null>(null)
  const [renaming, setRenaming] = useState<AnnotationSetRow | null>(null)
  const [draft, setDraft] = useState('')
  const [deleting, setDeleting] = useState<AnnotationSetRow | null>(null)
  const [working, setWorking] = useState(false)

  useEffect(() => {
    const timer = setTimeout(() => setQuery(search.trim()), 300)
    return () => clearTimeout(timer)
  }, [search])
  useEffect(() => { setPage(1) }, [status, query])

  const load = useCallback(async () => {
    try {
      const params = new URLSearchParams({ status })
      if (query) params.set('q', query)
      const response = await fetch(`/api/annotation/projects?${params}`)
      const result = await response.json()
      if (!response.ok) throw new Error(result.error)
      setSets(result.data)
    } catch (error) { report(error) }
  }, [status, query])
  useEffect(() => { void load() }, [load, refresh])

  async function patch(id: string, body: Record<string, unknown>) {
    setWorking(true)
    try {
      const response = await fetch(`/api/annotation/projects/${id}`, {
        method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      })
      const result = await response.json()
      if (!response.ok) throw new Error(result.error)
      await load()
    } catch (error) { report(error) } finally { setWorking(false) }
  }
  async function remove(id: string) {
    setWorking(true)
    try {
      const response = await fetch(`/api/annotation/projects/${id}`, { method: 'DELETE' })
      const result = await response.json()
      if (!response.ok) throw new Error(result.error)
      toast({ title: 'Annotation set deleted' })
      setDeleting(null)
      await load()
    } catch (error) { report(error) } finally { setWorking(false) }
  }
  async function exportSet(id: string, name: string) {
    try {
      const response = await fetch(`/api/annotation/projects?id=${encodeURIComponent(id)}`)
      const result = await response.json()
      if (!response.ok) throw new Error(result.error)
      download(`${name.replace(/[^\w.-]+/g, '_') || 'annotation-project'}.json`, result.data)
    } catch (error) { report(error) }
  }

  const total = sets?.length ?? 0
  const pageCount = Math.max(1, Math.ceil(total / SETS_PAGE_SIZE))
  const current = Math.min(page, pageCount)
  const rows = (sets ?? []).slice((current - 1) * SETS_PAGE_SIZE, current * SETS_PAGE_SIZE)

  return <div className="space-y-4">
    <div className="flex flex-wrap items-center justify-between gap-3">
      <div>
        <h1 className="text-2xl font-bold">Annotation sets</h1>
        <p className="text-sm text-muted-foreground">Sets not saved for 7 days are archived automatically. Import for training uses active sets only.</p>
      </div>
      <Button onClick={onNew}><Plus className="mr-1 size-4" />New annotation set</Button>
    </div>
    <div className="flex flex-wrap items-center gap-3">
      <div className="flex items-center gap-1 rounded-md border p-1">
        {(['active', 'archived', 'all'] as const).map(value => (
          <Button key={value} size="sm" variant={status === value ? 'default' : 'ghost'} onClick={() => setStatus(value)}>
            {value === 'active' ? 'Active' : value === 'archived' ? 'Archived' : 'All'}
          </Button>
        ))}
      </div>
      <Input className="max-w-xs" placeholder="Search by name..." value={search} onChange={event => setSearch(event.target.value)} />
      <p className="ml-auto text-sm text-muted-foreground">{sets ? `${total} set${total === 1 ? '' : 's'}` : 'Loading…'}</p>
    </div>
    <div className="rounded border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Name</TableHead>
            <TableHead>Images (reviewed)</TableHead>
            <TableHead>Instances</TableHead>
            <TableHead>Classes</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Last saved</TableHead>
            <TableHead>Archiving</TableHead>
            <TableHead className="text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map(set => <TableRow key={set.id}>
            <TableCell className="max-w-56"><span className="block truncate font-medium" title={set.name}>{set.name}</span><span className="text-xs text-muted-foreground">{set.id.slice(0, 8)}</span></TableCell>
            <TableCell>{set.reviewedImages}/{set.images}</TableCell>
            <TableCell>{set.instances}</TableCell>
            <TableCell>{set.labels}</TableCell>
            <TableCell><Badge variant={set.status === 'active' ? 'default' : 'secondary'}>{set.status}</Badge></TableCell>
            <TableCell className="whitespace-nowrap text-xs">{new Date(set.modified).toLocaleString()}</TableCell>
            <TableCell className="text-xs text-muted-foreground">
              {set.status === 'active' && set.archiveAt
                ? `auto-archives in ${Math.max(0, Math.ceil((Date.parse(set.archiveAt) - Date.now()) / 86400e3))} days`
                : `archived ${set.archivedAt ? new Date(set.archivedAt).toLocaleDateString() : ''} (${set.archivedReason ?? 'manual'})`}
            </TableCell>
            <TableCell className="whitespace-nowrap text-right">
              <div className="flex justify-end gap-1">
                <Button size="sm" variant="outline" onClick={() => void onOpen(set.id)}><FolderOpen className="mr-1 size-4" />Open</Button>
                <Button size="sm" variant="outline" aria-label="Rename" onClick={() => { setRenaming(set); setDraft(set.name) }}><Pencil className="size-4" /></Button>
                {set.status === 'active'
                  ? <Button size="sm" variant="outline" aria-label="Archive" disabled={working} onClick={() => void patch(set.id, { status: 'archived' })}><Archive className="size-4" /></Button>
                  : <Button size="sm" variant="outline" aria-label="Activate" disabled={working} onClick={() => void patch(set.id, { status: 'active' })}><ArchiveRestore className="size-4" /></Button>}
                <Button size="sm" variant="outline" aria-label="Export JSON" onClick={() => void exportSet(set.id, set.name)}><Download className="size-4" /></Button>
                <Button size="sm" variant="outline" aria-label="Delete" className="text-destructive" onClick={() => setDeleting(set)}><Trash2 className="size-4" /></Button>
              </div>
            </TableCell>
          </TableRow>)}
          {sets && rows.length === 0 && <TableRow><TableCell colSpan={8} className="py-8 text-center text-sm text-muted-foreground">
            {status === 'active' ? 'No active annotation sets. Create one, or check Archived.' : 'No annotation sets match.'}
          </TableCell></TableRow>}
        </TableBody>
      </Table>
    </div>
    {pageCount > 1 && <div className="flex items-center justify-center gap-2 text-sm">
      <Button size="sm" variant="outline" disabled={current <= 1} onClick={() => setPage(current - 1)}>Previous</Button>
      <span className="text-muted-foreground">Page {current} of {pageCount}</span>
      <Button size="sm" variant="outline" disabled={current >= pageCount} onClick={() => setPage(current + 1)}>Next</Button>
    </div>}
    <Dialog open={renaming !== null} onOpenChange={open => !open && setRenaming(null)}>
      <DialogContent><DialogHeader><DialogTitle>Rename annotation set</DialogTitle><DialogDescription>The name is stored inside the project file.</DialogDescription></DialogHeader>
        <div className="space-y-3">
          <Input value={draft} onChange={event => setDraft(event.target.value)} maxLength={100} />
          <Button disabled={working || !draft.trim()} onClick={async () => { if (renaming) { await patch(renaming.id, { name: draft.trim() }); setRenaming(null) } }}>Rename</Button>
        </div>
      </DialogContent>
    </Dialog>
    <AlertDialog open={deleting !== null} onOpenChange={open => !open && setDeleting(null)}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Delete "{deleting?.name}"?</AlertDialogTitle>
          <AlertDialogDescription>The annotation project file is removed permanently. Source images on your machine are not deleted.</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction disabled={working} onClick={() => deleting && void remove(deleting.id)}>Delete</AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  </div>
}

export function AnnotationPage() {
  const { user } = useAuth()
  const [history, dispatch] = useReducer(reducer, undefined, () => ({ present: newProject(), past: [], future: [] }))
  const project = history.present
  const [saved, setSaved] = useState<AnnotationProject | null>(null)
  const dirty = project !== saved && project.images.length > 0
  const [assets, setAssets] = useState<Record<string, Asset>>({})
  const assetsRef = useRef(assets)
  const [requestedIndex, setIndex] = useState(0)
  const index = Math.max(0, Math.min(requestedIndex, project.images.length - 1))
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [label, setLabel] = useState('slider_abs')
  const [tool, setTool] = useState<Tool>('obb')
  const [keypointName, setKeypointName] = useState('datum_a')
  const [pending, setPending] = useState<Point[]>([])
  const [hover, setHover] = useState<Point | null>(null)
  const [zoom, setZoom] = useState(1)
  const zoomRef = useRef(1)
  const [tab, setTab] = useState<'sets' | 'editor'>('sets')
  const [setsVersion, setSetsVersion] = useState(0)
  const zoomAnchor = useRef<{ px: number; py: number; ox: number; oy: number } | null>(null)
  const [busy, setBusy] = useState(false)
  const [preview, setPreview] = useState<AnnotationObject | null>(null)
  const drag = useRef<{ start: Point; object: AnnotationObject; key?: string; handle?: string } | null>(null)
  const pan = useRef<{ x: number; y: number; left: number; top: number } | null>(null)
  const viewport = useRef<HTMLDivElement>(null)
  const svg = useRef<SVGSVGElement>(null)
  const [settings, setSettings] = useState(false)
  const [labelsDraft, setLabelsDraft] = useState('')
  const [pointsDraft, setPointsDraft] = useState('')
  const [serverProjects, setServerProjects] = useState<Array<{ id: string; name: string; images: number; modified: string }> | null>(null)
  const [validation, setValidation] = useState<string[] | null>(null)
  const [showHelp, setShowHelp] = useState(false)
  const [predictionOpen, setPredictionOpen] = useState(false)
  const [models, setModels] = useState<{ id: string; name: string; framework: string; checkpoint?: string }[]>([])
  const [predictionJob, setPredictionJob] = useState('')
  const [predictionScope, setPredictionScope] = useState('empty')
  const [confidence, setConfidence] = useState(0.25)
  const [predictionProgress, setPredictionProgress] = useState('')
  const stopPrediction = useRef(false)
  const image = project.images[index]
  const selected = image?.objects.find(o => o.id === selectedId)
  const activeLabel = project.labels.includes(label) ? label : project.labels[0]
  const activeKey = project.keypointNames.includes(keypointName) ? keypointName : project.keypointNames[0]
  const asset = image && assets[image.name]
  const edit = (next: AnnotationProject) => dispatch({ type: 'edit', project: next })
  const editImage = (next: AnnotationImage) => edit({ ...project, images: project.images.map((v, i) => i === index ? next : v) })
  const editObject = (next: AnnotationObject) => {
    if (image) editImage({ ...image, reviewed: false, objects: image.objects.map(v => v.id === next.id ? next : v) })
  }

  useEffect(() => { assetsRef.current = assets }, [assets])
  useEffect(() => () => Object.values(assetsRef.current).forEach(a => URL.revokeObjectURL(a.url)), [])
  useEffect(() => {
    if (!dirty) return
    const warn = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = '' }
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [dirty])

  function cancelGesture() { setPending([]); setHover(null); setPreview(null); drag.current = null; pan.current = null }
  function navigate(next: number) {
    cancelGesture()
    setSelectedId(null)
    setIndex(Math.max(0, Math.min(project.images.length - 1, next)))
  }
  function fit(width = image?.width, height = image?.height) {
    if (width && height && viewport.current) {
      commitZoom(Math.min((viewport.current.clientWidth - 24) / width, (viewport.current.clientHeight - 24) / height, 2))
      viewport.current.scrollTo(0, 0)
    }
  }
  // zoomRef is updated synchronously so bursts of wheel events between renders
  // compound correctly instead of all reading the same stale render value.
  function commitZoom(next: number) {
    zoomRef.current = next
    setZoom(next)
  }
  /**
   * Zoom keeping the image point under (ox, oy) — viewport coordinates — fixed.
   * The scroll correction runs in a layout effect once the zoom state commits.
   * Without an anchor the viewport centre is used.
   */
  function zoomBy(factor: number, ox?: number, oy?: number) {
    const el = viewport.current
    const current = zoomRef.current
    const next = clampZoom(current * factor)
    if (next === current) return
    if (el) {
      // Flush a pending anchor immediately: several wheel events between renders
      // would otherwise anchor the later steps to an uncorrected scroll offset.
      if (zoomAnchor.current) {
        el.scrollLeft = zoomAnchor.current.px * current - zoomAnchor.current.ox
        el.scrollTop = zoomAnchor.current.py * current - zoomAnchor.current.oy
        zoomAnchor.current = null
      }
      const x = ox ?? el.clientWidth / 2, y = oy ?? el.clientHeight / 2
      zoomAnchor.current = { px: (el.scrollLeft + x) / current, py: (el.scrollTop + y) / current, ox: x, oy: y }
    }
    commitZoom(next)
  }
  useLayoutEffect(() => {
    const anchor = zoomAnchor.current
    const el = viewport.current
    if (!anchor || !el) return
    el.scrollLeft = anchor.px * zoom - anchor.ox
    el.scrollTop = anchor.py * zoom - anchor.oy
    zoomAnchor.current = null
  }, [zoom])
  // Ctrl/⌘ + wheel zooms anchored at the cursor. Must be a NATIVE non-passive
  // listener: React's onWheel is passive, so preventDefault there cannot stop
  // the browser's page zoom.
  useEffect(() => {
    const el = viewport.current
    if (!el) return
    function wheel(event: WheelEvent) {
      if (!event.ctrlKey && !event.metaKey) return
      event.preventDefault()
      const rect = el!.getBoundingClientRect()
      zoomBy(Math.exp(-event.deltaY * 0.0015), event.clientX - rect.left, event.clientY - rect.top)
    }
    el.addEventListener('wheel', wheel, { passive: false })
    return () => el.removeEventListener('wheel', wheel)
  })
  const save = useCallback(async () => {
    if (busy || !project.images.length) return
    setBusy(true)
    try {
      const snapshot = projectSchema.parse(project)
      const response = await fetch('/api/annotation/projects', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(snapshot),
      })
      const result = await response.json()
      if (!response.ok) throw new Error(result.error || 'Save failed')
      setSaved(project)
      setSetsVersion(v => v + 1)
      toast({ title: 'Project saved', description: 'Annotations, templates and splits were saved to the server; source images stay in your local folder.' })
    } catch (error) { report(error) } finally { setBusy(false) }
  }, [busy, project])

  function removeSelected() {
    if (!image || !selected) return
    editImage({ ...image, reviewed: false, objects: image.objects.filter(o => o.id !== selected.id) })
    setSelectedId(null)
  }
  useEffect(() => {
    function keydown(event: KeyboardEvent) {
      if (event.target instanceof HTMLElement && event.target.closest('input, textarea, select, [contenteditable="true"], [role="dialog"]')) return
      if (busy || settings || serverProjects || validation || showHelp || tab !== 'editor') return
      const modifier = event.ctrlKey || event.metaKey
      if (modifier && event.key.toLowerCase() === 's') { event.preventDefault(); void save(); return }
      if (modifier && event.key.toLowerCase() === 'z') { event.preventDefault(); cancelGesture(); dispatch({ type: event.shiftKey ? 'redo' : 'undo' }); return }
      if (modifier && event.key.toLowerCase() === 'y') { event.preventDefault(); cancelGesture(); dispatch({ type: 'redo' }); return }
      // Zoom shortcuts. The Ctrl/⌘ variants must preventDefault or the browser
      // zooms the whole page instead of the canvas.
      if (modifier && ['=', '+', '-', '0'].includes(event.key)) {
        event.preventDefault()
        if (event.key === '0') fit()
        else zoomBy(event.key === '-' ? 1 / 1.25 : 1.25)
        return
      }
      if (!modifier && !event.altKey) {
        if (event.key === '=' || event.key === '+') { zoomBy(1.25); return }
        if (event.key === '-') { zoomBy(1 / 1.25); return }
        if (event.key === '0') { fit(); return }
        if (event.key === '1') { zoomBy(1 / zoomRef.current); return }
      }
      if (event.key === 'Escape') cancelGesture()
      if (event.key === 'Delete') { event.preventDefault(); removeSelected() }
      if (event.key === 'ArrowLeft') { event.preventDefault(); navigate(index - 1) }
      if (event.key === 'ArrowRight') { event.preventDefault(); navigate(index + 1) }
    }
    window.addEventListener('keydown', keydown)
    return () => window.removeEventListener('keydown', keydown)
  })

  async function loadFolder(files: File[]) {
    if (!files.length) return
    setBusy(true)
    const loaded: Record<string, Asset> = {}
    try {
      const relative = (file: File) => file.webkitRelativePath ? file.webkitRelativePath.split('/').slice(1).join('/') : file.name
      const imageFiles = files.filter(f => /\.(png|jpe?g|bmp|webp)$/i.test(f.name))
      const imageIndex: ImageNameIndex = { byName: new Map(), byStem: new Map() }
      for (const file of imageFiles) {
        for (const [map, key] of [[imageIndex.byName, file.name.toLowerCase()], [imageIndex.byStem, file.name.replace(/\.[^.]+$/, '').toLowerCase()]] as const) {
          map.set(key, [...(map.get(key) ?? []), relative(file)])
        }
      }
      const imports = new Map<string, ReturnType<typeof fromLabelme>>()
      for (const file of files.filter(f => /\.json$/i.test(f.name))) {
        const raw = JSON.parse((await file.text()).replace(/^\uFEFF/, ''))
        // Project backups, COCO documents and unrelated JSON are not Labelme files.
        if (!raw || !Array.isArray(raw.shapes)) continue
        const resolved = resolveUploadedLabelmeImage(imageIndex, relative(file), raw.imagePath)
        if ('error' in resolved) throw new Error(`${relative(file)}: ${resolved.error}`)
        const key = resolved.path.toLowerCase()
        if (imports.has(key)) throw new Error(`Multiple JSON files match the same image: ${resolved.path}`)
        imports.set(key, fromLabelme(raw, resolved.path))
      }
      const records: AnnotationImage[] = []
      const importedNames = [...imports.values()].flatMap(v => v.keypointNames)
      const importedLabels = [...imports.values()].flatMap(v => v.image.objects.map(o => o.label))
      const labels = new Set(!project.images.length && importedLabels.length ? importedLabels : project.labels)
      const names = new Set(!project.images.length && importedNames.length ? importedNames : project.keypointNames)
      const existing = new Map(project.images.map(i => [i.name.toLowerCase(), i]))
      for (const file of imageFiles) {
        const name = relative(file)
        if (Object.keys(loaded).some(k => k.toLowerCase() === name.toLowerCase())) throw new Error(`Duplicate file name: ${name}`)
        const url = URL.createObjectURL(file)
        loaded[name] = { url, width: 0, height: 0 }
        const dims = await new Promise<{ width: number; height: number }>((resolve, reject) => {
          const img = new Image()
          img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight })
          img.onerror = () => reject(new Error(`Could not decode image: ${name}`))
          img.src = url
        })
        loaded[name] = { url, ...dims }
        let record = existing.get(name.toLowerCase())
        const imported = imports.get(name.toLowerCase())
        if (!record && imported) {
          record = imported.image
          imported.keypointNames.forEach(n => names.add(n))
          record.objects.forEach(o => labels.add(o.label))
        }
        if (record && (record.width !== dims.width || record.height !== dims.height)) throw new Error(`${name}: image dimensions differ from the saved annotation; mismatched images cannot be annotated`)
        records.push(record ? { ...record, name } : {
          name, ...dims, objects: [], reviewed: false, split: 'unassigned',
          group: name.includes('/') ? name.slice(0, name.lastIndexOf('/')) : '',
        })
      }
      if (!records.length) throw new Error('No PNG/JPEG/BMP/WebP images found; convert TIFF to PNG without scaling first')
      for (const old of project.images) {
        if (!records.some(v => v.name.toLowerCase() === old.name.toLowerCase())) records.push(old)
      }
      records.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }))
      const next = projectSchema.parse({ ...project, labels: [...labels], keypointNames: [...names], images: records })
      Object.values(assetsRef.current).forEach(a => URL.revokeObjectURL(a.url))
      assetsRef.current = loaded
      setAssets(loaded)
      if (JSON.stringify(next) !== JSON.stringify(project)) edit(next)
      navigate(0)
      fit(records[0].width, records[0].height)
      toast({ title: 'Images loaded', description: `${Object.keys(loaded).length} images; existing annotations were kept. Link the same image root when reopening this project.` })
    } catch (error) {
      Object.values(loaded).forEach(a => URL.revokeObjectURL(a.url))
      report(error)
    } finally { setBusy(false) }
  }

  function loadProject(raw: unknown): boolean {
    const next = projectSchema.parse(raw)
    if (dirty && !window.confirm('The current project has unsaved changes. Open another project anyway?')) return false
    cancelGesture()
    Object.values(assetsRef.current).forEach(a => URL.revokeObjectURL(a.url))
    assetsRef.current = {}
    setAssets({})
    dispatch({ type: 'load', project: next })
    setSaved(next)
    setIndex(0)
    setSelectedId(null)
    setLabel(next.labels[0])
    setKeypointName(next.keypointNames[0] ?? '')
    setServerProjects(null)
    toast({ title: 'Annotations restored', description: 'Click "Link image folder" to choose the source image root.' })
    return true
  }

  async function openSavedSet(id: string) {
    try {
      const response = await fetch(`/api/annotation/projects?id=${encodeURIComponent(id)}`)
      const result = await response.json()
      if (!response.ok) throw new Error(result.error)
      if (loadProject(result.data)) setTab('editor')
    } catch (error) { report(error) }
  }
  function newAnnotationSet() {
    if (loadProject(newProject())) setTab('editor')
  }

  function coords(event: React.PointerEvent<SVGSVGElement>): Point | null {
    const matrix = svg.current?.getScreenCTM()
    if (!matrix) return null
    const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse())
    return { x: point.x, y: point.y }
  }
  function pointerDown(event: React.PointerEvent<SVGSVGElement>) {
    if (!image || !asset || busy) return
    if (event.button === 1 || (event.button === 0 && event.altKey)) {
      if (viewport.current) {
        event.preventDefault()
        pan.current = { x: event.clientX, y: event.clientY, left: viewport.current.scrollLeft, top: viewport.current.scrollTop }
        event.currentTarget.setPointerCapture(event.pointerId)
      }
      return
    }
    if (event.button !== 0) return
    const p = coords(event)
    if (!p || !insideImage(p, image)) return
    if (tool === 'select') {
      const target = event.target as Element
      const key = target.getAttribute('data-key') ?? undefined
      const id = target.closest('[data-instance]')?.getAttribute('data-instance')
      const object = image.objects.find(o => o.id === id) ?? [...image.objects].reverse().find(o => containsPoint(o, p))
      setSelectedId(object?.id ?? null)
      if (object) {
        drag.current = { start: p, object, key, handle: target.getAttribute('data-handle') ?? undefined }
        event.currentTarget.setPointerCapture(event.pointerId)
      }
      return
    }
    if (tool === 'point') {
      if (!selected || !activeKey) { report(new Error('Select an instance and a keypoint name first')); return }
      editObject({ ...selected, keypoints: { ...selected.keypoints, [activeKey]: { ...p, visibility: selected.keypoints[activeKey]?.visibility === 1 ? 1 : 2 } } })
      const next = project.keypointNames.find(k => k !== activeKey && !selected.keypoints[k])
      if (next) setKeypointName(next)
      return
    }
    const points = [...pending, p]
    const needed = tool === 'obb' ? 3 : 2
    if (points.length < needed) { setPending(points); return }
    try {
      const geometry = tool === 'obb' ? boxFromThreePoints(points[0], points[1], points[2]) : {
        cx: (points[0].x + p.x) / 2, cy: (points[0].y + p.y) / 2,
        width: Math.abs(p.x - points[0].x), height: Math.abs(p.y - points[0].y), angle: 0,
      }
      if (geometry.width < 1 || geometry.height < 1) throw new Error('Box width and height must be at least 1 px')
      if (!boxCorners(geometry).every(q => insideImage(q, image))) throw new Error('Rotated box extends outside the image; adjust the third point')
      const object: AnnotationObject = { ...geometry, id: uuid(), label: activeLabel, kind: tool, keypoints: {} }
      editImage({ ...image, reviewed: false, objects: [...image.objects, object] })
      setSelectedId(object.id)
      cancelGesture()
      setTool('select')
    } catch (error) { report(error) }
  }
  function pointerMove(event: React.PointerEvent<SVGSVGElement>) {
    if (pan.current && viewport.current) {
      viewport.current.scrollTo(pan.current.left - event.clientX + pan.current.x, pan.current.top - event.clientY + pan.current.y)
      return
    }
    const p = coords(event)
    if (!p || !image) return
    setHover(p)
    const d = drag.current
    if (!d) return
    if (d.handle) {
      const resized = d.handle === 'rotate' ? rotateBox(d.object, d.start, p, event.shiftKey) :
        resizeBox(d.object, p, ...d.handle.split(',').map(Number) as [number, number])
      if (boxCorners(resized).every(point => insideImage(point, image))) setPreview(resized)
    } else if (d.key && d.object.keypoints[d.key]) {
      const point = { ...d.object.keypoints[d.key], x: Math.max(0, Math.min(image.width, p.x)), y: Math.max(0, Math.min(image.height, p.y)) }
      setPreview({ ...d.object, keypoints: { ...d.object.keypoints, [d.key]: point } })
    } else {
      const positions = [...boxCorners(d.object), ...Object.values(d.object.keypoints).filter(k => k.visibility > 0)]
      const dx = Math.max(-Math.min(...positions.map(q => q.x)), Math.min(image.width - Math.max(...positions.map(q => q.x)), p.x - d.start.x))
      const dy = Math.max(-Math.min(...positions.map(q => q.y)), Math.min(image.height - Math.max(...positions.map(q => q.y)), p.y - d.start.y))
      setPreview({ ...d.object, cx: d.object.cx + dx, cy: d.object.cy + dy,
        keypoints: Object.fromEntries(Object.entries(d.object.keypoints).map(([name, point]) => [name, point.visibility ? { ...point, x: point.x + dx, y: point.y + dy } : point])),
      })
    }
  }
  function pointerUp(event: React.PointerEvent<SVGSVGElement>) {
    if (drag.current && preview) editObject(preview)
    drag.current = null
    pan.current = null
    setPreview(null)
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
  }
  function draftCorners(): Point[] {
    if (!hover || !pending.length) return pending
    if (tool === 'rectangle') {
      return boxCorners({ cx: (pending[0].x + hover.x) / 2, cy: (pending[0].y + hover.y) / 2, width: Math.abs(hover.x - pending[0].x), height: Math.abs(hover.y - pending[0].y), angle: 0 })
    }
    if (pending.length === 2) {
      try { return boxCorners(boxFromThreePoints(pending[0], pending[1], hover)) } catch { return pending }
    }
    return [...pending, hover]
  }
  function changeTemplate() {
    try {
      const parseNames = (text: string) => text.split(/\r?\n/).map(v => v.trim()).filter(Boolean)
      const next = projectSchema.parse({ ...project, labels: parseNames(labelsDraft), keypointNames: parseNames(pointsDraft),
        images: project.images.map(i => ({ ...i, reviewed: false })),
      })
      edit(next)
      setSettings(false)
      cancelGesture()
    } catch (error) { report(error) }
  }
  function exportProject() {
    try {
      download('localization-project.json', projectSchema.parse(project))
      toast({ title: 'Project exported', description: 'Includes all annotations and templates, but no source images. Back it up together with the images.' })
    } catch (error) { report(error) }
  }
  async function openPrediction() {
    try {
      const response = await fetch('/api/annotation/predict')
      const result = await response.json()
      if (!response.ok) throw new Error(result.error)
      setModels(result.data)
      setPredictionJob(result.data[0]?.id || '')
      setPredictionProgress('')
      setPredictionOpen(true)
    } catch (error) { report(error) }
  }
  async function predict() {
    if (!predictionJob || busy) return
    const targets = predictionScope === 'current' ? [image].filter(Boolean) : project.images.filter(item => !item.objects.length && !item.reviewed)
    if (!targets.length) { report(new Error('No images match the selected scope')); return }
    if (targets.some(item => !assets[item.name])) { report(new Error('Link the source images for every image to predict first')); return }
    if (predictionScope === 'current' && image?.objects.length && !window.confirm('This prediction replaces the instances and keypoints of the current image (undoable). Continue?')) return
    setBusy(true)
    stopPrediction.current = false
    const changed = new Map<string, AnnotationImage>()
    let nextLabels = project.labels, nextNames = project.keypointNames
    try {
      for (let offset = 0; offset < targets.length && !stopPrediction.current; offset += 8) {
        const batch = targets.slice(offset, offset + 8)
        setPredictionProgress(`Predicting ${offset + 1}–${offset + batch.length} / ${targets.length}`)
        const data = new FormData()
        data.append('jobId', predictionJob); data.append('confidence', String(confidence))
        data.append('names', JSON.stringify(batch.map(item => item.name)))
        for (const item of batch) data.append('images', await (await fetch(assets[item.name].url)).blob(), item.name.split('/').pop()!)
        const response = await fetch('/api/annotation/predict', { method: 'POST', body: data })
        const raw = await response.json()
        if (!response.ok) throw new Error(raw.error)
        const result = predictionsSchema.parse(raw.data)
        if (result.task === 'pose' && project.keypointNames.length && JSON.stringify(project.keypointNames) !== JSON.stringify(result.keypointNames)) throw new Error('The model keypoint names or order differ from the project template; use a matching template.')
        nextLabels = [...new Set([...project.labels, ...result.labels])]
        if (result.task === 'pose') nextNames = result.keypointNames
        if (result.images.length !== batch.length) throw new Error('The model did not return every image in this batch')
        for (const predicted of result.images) {
          const source = batch.find(item => item.name === predicted.source)
          if (!source) throw new Error('The model returned an unknown image')
          changed.set(source.name, predictionToImage(source, result, predicted))
        }
      }
    } catch (error) { report(error) }
    finally {
      if (changed.size) {
        edit(projectSchema.parse({ ...project, labels: nextLabels, keypointNames: nextNames, images: project.images.map(item => changed.get(item.name) || item) }))
        navigate(project.images.findIndex(item => changed.has(item.name)))
        setTool('select')
        toast({ title: 'Predictions shown, pending review', description: `${changed.size} images. Adjust boxes and keypoints, then review each image; unreviewed predictions cannot be imported for training.` })
      }
      setBusy(false); setPredictionOpen(false); setPredictionProgress('')
    }
  }
  if (user?.role !== 'admin') return <p className="text-muted-foreground">Sign in with an admin account to use the annotation tool.</p>
  const issues = image ? imageIssues(image, project.keypointNames) : []

  return (
    <Tabs value={tab} onValueChange={value => setTab(value as 'sets' | 'editor')} className="space-y-4">
      <TabsList>
        <TabsTrigger value="sets">Annotation sets</TabsTrigger>
        <TabsTrigger value="editor">Editor</TabsTrigger>
      </TabsList>
      <TabsContent value="sets">
        <AnnotationSetsPanel refresh={setsVersion} onOpen={openSavedSet} onNew={newAnnotationSet} />
      </TabsContent>
      <TabsContent value="editor">
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold">Localization annotation · OBB + keypoints</h1>
          <p className="text-sm text-muted-foreground">{project.name} · {project.images.length} images · {dirty ? 'Unsaved changes' : 'Saved'}</p>
          {image?.prediction && <p className={`text-sm ${image.reviewed ? 'text-green-600' : 'text-amber-600'}`}>{image.reviewed ? 'Model-assisted annotation · human-reviewed' : 'Model prediction · pending review, not usable for training yet'}</p>}
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" onClick={() => setShowHelp(true)}>Annotation & training guide</Button>
          <Button variant="outline" disabled={busy} onClick={async () => {
            try {
              const response = await fetch('/api/annotation/projects')
              const result = await response.json()
              if (!response.ok) throw new Error(result.error)
              setServerProjects(result.data)
            } catch (error) { report(error) }
          }}><FolderOpen className="mr-1 size-4" />Open saved project</Button>
          <Button disabled={busy || !image} onClick={save}><Save className="mr-1 size-4" />{busy ? 'Working…' : 'Save project Ctrl+S'}</Button>
          <Button variant="outline" disabled={!image} onClick={exportProject}><Download className="mr-1 size-4" />Export project</Button>
        </div>
      </div>
      <div className="grid gap-4 xl:grid-cols-[230px_minmax(0,1fr)_290px]" inert={busy}>
        <div className="space-y-3">
          <Card><CardHeader className="pb-2"><CardTitle className="text-sm">Project & images</CardTitle></CardHeader><CardContent className="space-y-3">
            <Input aria-label="Project name" value={project.name} onChange={e => edit({ ...project, name: e.target.value })} />
            <label className="block cursor-pointer rounded border border-dashed p-3 text-center text-sm">
              Link image folder
              <input type="file" className="hidden" multiple {...{ webkitdirectory: '', directory: '' }}
                disabled={busy} onChange={e => { void loadFolder(Array.from(e.target.files ?? [])); e.target.value = '' }} />
            </label>
            <label className="block cursor-pointer text-sm underline">Import project JSON
              <input type="file" accept=".json" className="hidden" disabled={busy} onChange={async e => {
                const file = e.target.files?.[0]
                e.target.value = ''
                if (!file) return
                try { loadProject(JSON.parse((await file.text()).replace(/^\uFEFF/, ''))) } catch (error) { report(error) }
              }} />
            </label>
            <Button className="w-full" variant="outline" disabled={busy || !project.images.length} onClick={openPrediction}>Model pre-labeling</Button>
            <Button className="w-full" variant="outline" onClick={() => loadProject(newProject())}>New project</Button>
            <Button className="w-full" variant="outline" onClick={() => { setLabelsDraft(project.labels.join('\n')); setPointsDraft(project.keypointNames.join('\n')); setSettings(true) }}>Classes & keypoint template</Button>
            <p className="text-xs text-muted-foreground">Source images stay local. To resume, open the project first, then link the image root. Labelme JSON files sharing the image name in the same directory are supported.</p>
          </CardContent></Card>
          <div className="max-h-[420px] space-y-1 overflow-auto rounded border p-1" aria-label="Image list">
            {project.images.map((item, i) => <button key={item.name} className={`block w-full rounded p-2 text-left text-xs ${i === index ? 'bg-accent' : 'hover:bg-muted'}`} onClick={() => navigate(i)}>
              <span className="break-all">{item.reviewed ? '✓ ' : '○ '}{item.name}</span>
              <span className="block text-muted-foreground">{item.objects.length} instances · {splitNames[item.split]}{!assets[item.name] && ' · image missing'}</span>
            </button>)}
          </div>
        </div>
        <div className="min-w-0 space-y-3">
          <div className="flex flex-wrap items-center gap-2 rounded border p-2">
            {([
              ['select', 'Select / drag', MousePointer2], ['rectangle', 'Axis-aligned box (2 points)', Square],
              ['obb', 'Rotated box (3 points)', RotateCw], ['point', 'Keypoint', Crosshair],
            ] as const).map(([value, text, Icon]) => <Button key={value} size="sm" variant={tool === value ? 'default' : 'outline'} onClick={() => { cancelGesture(); setTool(value) }}>
              <Icon className="mr-1 size-4" />{text}
            </Button>)}
            <Button size="icon" variant="outline" aria-label="Undo" disabled={!history.past.length} onClick={() => { cancelGesture(); dispatch({ type: 'undo' }) }}><Undo2 className="size-4" /></Button>
            <Button size="icon" variant="outline" aria-label="Redo" disabled={!history.future.length} onClick={() => { cancelGesture(); dispatch({ type: 'redo' }) }}><Redo2 className="size-4" /></Button>
            <Button size="icon" variant="outline" aria-label="Zoom in" onClick={() => zoomBy(1.25)}><ZoomIn className="size-4" /></Button>
            <Button size="icon" variant="outline" aria-label="Zoom out" onClick={() => zoomBy(1 / 1.25)}><ZoomOut className="size-4" /></Button>
            <Button variant="outline" size="sm" onClick={() => fit()}>Fit {Math.round(zoom * 100)}%</Button>
          </div>
          <p className="text-xs text-muted-foreground">
            {tool === 'obb' ? `Click both ends of the first edge, then click the opposite side (${pending.length}/3); Esc cancels.` :
              tool === 'rectangle' ? 'Click two opposite corners; Esc cancels.' :
                tool === 'point' ? `Place ${activeKey || 'keypoint'} on the selected instance; click again to adjust.` : 'Drag the box to move; drag square handles to resize and the rotation handle to rotate (Shift snaps 15°); drag keypoints to reposition.'}
            {' '}Alt + drag or middle-button to pan; +/− zoom, 0 fit, 1 100%, Ctrl+wheel zooms at the cursor; use the scrollbars when zoomed in.
          </p>
          <div ref={viewport} className="relative h-[600px] overflow-auto rounded border bg-slate-950">
            {image && asset ? <svg ref={svg} width={image.width * zoom} height={image.height * zoom} viewBox={`0 0 ${image.width} ${image.height}`}
              className="block touch-none select-none" style={{ maxWidth: 'none', cursor: tool === 'select' ? 'default' : 'crosshair' }}
              aria-label="Annotation canvas" onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={pointerUp}
              onPointerCancel={cancelGesture} onContextMenu={e => e.preventDefault()}>
              <image href={asset.url} width={image.width} height={image.height} />
              {image.objects.map((object, i) => {
                const shown = preview?.id === object.id ? preview : object
                const corners = boxCorners(shown)
                const active = selectedId === object.id
                const color = active ? '#4ade80' : '#fb923c'
                return <g key={object.id} data-instance={object.id}>
                  <polygon points={corners.map(p => `${p.x},${p.y}`).join(' ')} fill={active ? '#4ade8014' : 'transparent'} stroke={color} strokeWidth={2 / zoom} />
                  <text x={corners[0].x} y={corners[0].y - 5 / zoom} fill={color} stroke="#000" strokeWidth={2 / zoom} paintOrder="stroke" fontSize={13 / zoom}>{i + 1}: {object.label}</text>
                  {active && tool === 'select' && <>
                    {([[-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0]] as const).map(([sx, sy]) => {
                      const angle = shown.angle * Math.PI / 180
                      const x = shown.cx + sx * shown.width / 2 * Math.cos(angle) - sy * shown.height / 2 * Math.sin(angle)
                      const y = shown.cy + sx * shown.width / 2 * Math.sin(angle) + sy * shown.height / 2 * Math.cos(angle)
                      return <rect key={`${sx},${sy}`} data-handle={`${sx},${sy}`} aria-label={`Resize handle ${sx},${sy}`} x={x - 5 / zoom} y={y - 5 / zoom} width={10 / zoom} height={10 / zoom} fill="#fff" stroke={color} strokeWidth={1.5 / zoom} style={{ cursor: 'nwse-resize' }} />
                    })}
                    {(() => {
                      const angle = shown.angle * Math.PI / 180
                      const x = Math.max(9 / zoom, Math.min(image.width - 9 / zoom, shown.cx + (shown.height / 2 + 30 / zoom) * Math.sin(angle)))
                      const y = Math.max(9 / zoom, Math.min(image.height - 9 / zoom, shown.cy - (shown.height / 2 + 30 / zoom) * Math.cos(angle)))
                      return <><line x1={(corners[0].x + corners[1].x) / 2} y1={(corners[0].y + corners[1].y) / 2} x2={x} y2={y} stroke={color} strokeWidth={1 / zoom} pointerEvents="none" /><circle data-handle="rotate" aria-label="Rotation handle" cx={x} cy={y} r={6 / zoom} fill="#facc15" stroke="#000" strokeWidth={1 / zoom} style={{ cursor: 'grab' }} /></>
                    })()}
                  </>}
                  {Object.entries(shown.keypoints).filter(([, p]) => p.visibility > 0).map(([name, p]) => <g key={name}>
                    <circle data-key={name} cx={p.x} cy={p.y} r={5 / zoom} fill={p.visibility === 1 ? 'transparent' : color} stroke={color} strokeWidth={2 / zoom} strokeDasharray={p.visibility === 1 ? `${3 / zoom} ${2 / zoom}` : undefined} />
                    <text pointerEvents="none" x={p.x + 8 / zoom} y={p.y - 5 / zoom} fontSize={12 / zoom} fill={color} stroke="#000" strokeWidth={2 / zoom} paintOrder="stroke">{name}</text>
                  </g>)}
                </g>
              })}
              {pending.length > 0 && <polyline points={draftCorners().map(p => `${p.x},${p.y}`).join(' ')} fill="none" stroke="#38bdf8" strokeWidth={2 / zoom} strokeDasharray={`${5 / zoom} ${4 / zoom}`} pointerEvents="none" />}
            </svg> : <div className="flex h-full items-center justify-center p-8 text-center text-sm text-slate-300">{image ? 'Annotations loaded; link the image folder.' : 'Choose an image folder to start. Subdirectories per capture batch are recommended.'}</div>}
          </div>
          <div className="flex items-center justify-between gap-2">
            <Button variant="outline" disabled={index <= 0} onClick={() => navigate(index - 1)}>Prev ←</Button>
            <span className="truncate text-xs">{image ? `${index + 1}/${project.images.length} · ${image.name} · ${image.width}×${image.height}` : 'Not loaded'}</span>
            <Button variant="outline" disabled={index >= project.images.length - 1} onClick={() => navigate(index + 1)}>Next →</Button>
          </div>
          {image && <Card><CardContent className="space-y-3 pt-4">
            <div className="grid gap-3 sm:grid-cols-3">
              <label className="space-y-1 text-xs">Capture batch (consecutive shots of the same workpiece share a batch)
                <Input value={image.group} onChange={e => editImage({ ...image, group: e.target.value })} placeholder="e.g. lot_01_camera_A" />
              </label>
              <label className="space-y-1 text-xs">Split
                <select className={selectClass} value={image.split} onChange={e => editImage({ ...image, split: e.target.value as Split })}>
                  {Object.entries(splitNames).map(([value, text]) => <option key={value} value={value}>{text}</option>)}
                </select>
              </label>
              <Button className="self-end" variant="outline" disabled={!image.group} onClick={() => {
                edit({ ...project, images: project.images.map(v => v.group === image.group ? { ...v, split: image.split } : v) })
              }}>Apply split to the same batch</Button>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <Button variant={image.reviewed ? 'secondary' : 'default'} disabled={!asset} onClick={() => editImage({ ...image, reviewed: !image.reviewed })}>
                {image.reviewed ? '✓ Reviewed (click to unmark)' : image.objects.length ? 'Mark image reviewed' : 'Mark image reviewed (no targets)'}
              </Button>
              <Button variant="outline" onClick={() => setValidation(projectIssues(project, false))}>Check OBB data</Button>
              <Button variant="outline" onClick={() => setValidation(projectIssues(project, true))}>Check Pose data</Button>
            </div>
            {issues.length > 0 && <p className="text-xs text-amber-600">{issues.slice(0, 3).join('; ')}{issues.length > 3 ? `; ${issues.length - 3} more` : ''}</p>}
          </CardContent></Card>}
        </div>
        <div className="space-y-3">
          <Card><CardHeader className="pb-2"><CardTitle className="text-sm">Instances</CardTitle></CardHeader><CardContent className="space-y-3">
            <label className="text-xs">New box class
              <select className={selectClass} value={activeLabel} onChange={e => setLabel(e.target.value)}>{project.labels.map(v => <option key={v}>{v}</option>)}</select>
            </label>
            <div className="max-h-40 space-y-1 overflow-auto">
              {image?.objects.map((o, i) => <button key={o.id} className={`block w-full rounded p-2 text-left text-xs ${selectedId === o.id ? 'bg-accent' : 'hover:bg-muted'}`} onClick={() => { cancelGesture(); setSelectedId(o.id); setTool('select') }}>
                #{i + 1} {o.label} · {o.kind === 'obb' ? 'rotated box' : 'box'} · {Object.keys(o.keypoints).length}/{project.keypointNames.length} pts
              </button>)}
            </div>
            {selected && <>
              <select aria-label="Selected instance class" className={selectClass} value={selected.label} onChange={e => editObject({ ...selected, label: e.target.value })}>{project.labels.map(v => <option key={v}>{v}</option>)}</select>
              <div className="grid grid-cols-2 gap-2">
                {([['cx', 'Center X'], ['cy', 'Center Y'], ['width', 'Width'], ['height', 'Height'], ['angle', 'Clockwise angle °']] as const).map(([field, text]) => <label key={`${selected.id}-${field}-${selected[field]}`} className="text-xs">{text}
                  <Input type="number" step="0.1" defaultValue={Number(selected[field].toFixed(3))} onBlur={e => {
                    const value = Number(e.target.value)
                    if (!e.target.value || !Number.isFinite(value) || ((field === 'width' || field === 'height') && value < 1) || (field === 'angle' && Math.abs(value) > 180)) { e.target.value = String(selected[field]); report(new Error('Enter a valid number; width/height ≥ 1, angle between -180 and 180')); return }
                    if (value !== selected[field]) editObject({ ...selected, [field]: value, kind: field === 'angle' ? 'obb' : selected.kind })
                  }} />
                </label>)}
              </div>
              <p className="text-xs text-muted-foreground">Resize/rotation handles change the box only; keypoints keep their image positions. Moving the box moves its keypoints too. Every adjustment is undoable.</p>
              <Button variant="destructive" size="sm" onClick={removeSelected}><Trash2 className="mr-1 size-4" />Delete instance & keypoints</Button>
            </>}
          </CardContent></Card>
          <Card><CardHeader className="pb-2"><CardTitle className="text-sm">Keypoints · bound to selected instance</CardTitle></CardHeader><CardContent className="space-y-3">
            <p className="text-xs text-muted-foreground">Pick a name, then click the image to place it. 0 = cannot determine; 1 = occluded but position known; 2 = visible. Blank means not yet handled.</p>
            {!selected && <p className="rounded bg-muted p-3 text-sm">Select a workpiece on the canvas or in the instance list first.</p>}
            {selected && project.keypointNames.map(name => {
              const point = selected?.keypoints[name]
              return <div key={name} className="space-y-1 rounded border p-2">
                <button className={`text-sm ${tool === 'point' && activeKey === name ? 'font-bold text-primary' : ''}`} disabled={!selected} onClick={() => { cancelGesture(); setKeypointName(name); setTool('point') }}>{name} {tool === 'point' && activeKey === name ? '← placing' : ''}</button>
                <select aria-label={`${name} visibility`} className={selectClass} value={point?.visibility ?? ''} disabled={!selected} onChange={e => {
                  if (!selected) return
                  const points = { ...selected.keypoints }
                  if (e.target.value === '') delete points[name]
                  else if (e.target.value === '0') points[name] = { x: 0, y: 0, visibility: 0 }
                  else if (!point || point.visibility === 0) { setKeypointName(name); setTool('point'); report(new Error('Place the keypoint on the image before setting its visibility')); return }
                  else points[name] = { ...point, visibility: Number(e.target.value) as 1 | 2 }
                  editObject({ ...selected, keypoints: points })
                }}>
                  <option value="">Pending</option><option value="2">2 · Visible</option><option value="1">1 · Occluded, position known</option><option value="0">0 · Cannot determine / N/A</option>
                </select>
                {point && point.visibility > 0 && <p className="text-xs text-muted-foreground">({point.x.toFixed(2)}, {point.y.toFixed(2)}) px</p>}
              </div>
            })}
            <Button className="w-full" variant="outline" disabled={!image} onClick={() => image && download(image.name.split('/').pop()!.replace(/\.[^.]+$/, '.json'), toLabelme(image, project.keypointNames))}>Export current image as Labelme</Button>
          </CardContent></Card>
        </div>
      </div>
      <Dialog open={predictionOpen} onOpenChange={open => { if (!busy) setPredictionOpen(open) }}><DialogContent className="sm:max-w-xl"><DialogHeader><DialogTitle>Model pre-labeling</DialogTitle><DialogDescription>Results appear on the canvas as "pending review"; each image must be reviewed before it can be imported as ground truth for training.</DialogDescription></DialogHeader>
        <fieldset disabled={busy} className="space-y-3">
          <label className="block text-sm">Trained model<select aria-label="Pre-labeling model" className={selectClass} value={predictionJob} onChange={event => setPredictionJob(event.target.value)}><option value="">Select model</option>{models.map(model => <option key={model.id} value={model.id}>{model.name} · {model.framework}{model.checkpoint ? ` · ${model.checkpoint}` : ''}</option>)}</select></label>
          {!models.length && <p className="text-sm text-muted-foreground">No PaddleDetection/OBB/Pose checkpoints available yet. Finish a training job in Jobs first.</p>}
          <label className="block text-sm">Scope<select aria-label="Prediction scope" className={selectClass} value={predictionScope} onChange={event => setPredictionScope(event.target.value)}><option value="empty">Unannotated, unreviewed images (keep human labels)</option><option value="current">Current image (replace instances, undoable)</option></select></label>
          <label className="block text-sm">Confidence threshold<Input aria-label="Prediction confidence" type="number" min={0.01} max={0.99} step={0.05} value={confidence} onChange={event => setConfidence(Number(event.target.value))} /></label>
          <p className="text-xs text-muted-foreground">Detection models produce axis-aligned boxes; pose keypoints below 0.5 confidence stay pending — that does not mean "invisible". Images are uploaded to a temporary prediction directory and cleaned up afterwards.</p>
          <Button disabled={!predictionJob || !Number.isFinite(confidence) || confidence < 0.01 || confidence > 0.99} onClick={predict}>Start prediction</Button>
        </fieldset>
        {busy && <div className="space-y-2"><p>{predictionProgress}</p><Button variant="outline" onClick={() => { stopPrediction.current = true; setPredictionProgress('Stopping after the current batch; completed results are kept') }}>Stop remaining images</Button></div>}
      </DialogContent></Dialog>
      <Dialog open={settings} onOpenChange={setSettings}><DialogContent className="sm:max-w-2xl"><DialogHeader><DialogTitle>Classes & keypoint template</DialogTitle><DialogDescription>One name per line; the order defines the training class / keypoint order. The template applies to every class in the project; annotate workpieces with different structures in separate projects.</DialogDescription></DialogHeader>
        <div className="grid gap-3 sm:grid-cols-2">
          <label className="text-sm">Classes<textarea className="mt-1 h-48 w-full rounded border bg-background p-2" value={labelsDraft} onChange={e => setLabelsDraft(e.target.value)} /></label>
          <label className="text-sm">Keypoints<textarea className="mt-1 h-48 w-full rounded border bg-background p-2" value={pointsDraft} onChange={e => setPointsDraft(e.target.value)} /></label>
        </div>
        <p className="text-xs text-muted-foreground">datum_a/b/c are placeholders — replace them with meaningful names (e.g. hole, notch) before annotating. Name by workpiece structure, not by image orientation. Names already in use cannot be deleted or renamed directly; after changing the template all images must be re-reviewed.</p>
        <Button onClick={changeTemplate}>Apply template</Button>
      </DialogContent></Dialog>
      <Dialog open={serverProjects !== null} onOpenChange={open => !open && setServerProjects(null)}><DialogContent className="sm:max-w-xl"><DialogHeader><DialogTitle>Open my annotation project</DialogTitle><DialogDescription>After restoring, link the image folder again to resume annotating.</DialogDescription></DialogHeader>
        <div className="max-h-96 space-y-2 overflow-auto">{serverProjects?.length === 0 && <p className="text-sm">No saved projects yet.</p>}
          {serverProjects?.map(p => <Button key={p.id} variant="outline" className="h-auto w-full justify-start whitespace-normal text-left" onClick={async () => {
            try {
              const response = await fetch(`/api/annotation/projects?id=${encodeURIComponent(p.id)}`)
              const result = await response.json()
              if (!response.ok) throw new Error(result.error)
              loadProject(result.data)
            } catch (error) { report(error) }
          }}>{p.name} · {p.images} images · {new Date(p.modified).toLocaleString()}</Button>)}
        </div>
      </DialogContent></Dialog>
      <Dialog open={validation !== null} onOpenChange={open => !open && setValidation(null)}><DialogContent className="sm:max-w-2xl"><DialogHeader><DialogTitle>Training data check</DialogTitle><DialogDescription>Drafts can be saved; these issues must be resolved before preparing training data.</DialogDescription></DialogHeader>
        <div className="max-h-96 overflow-auto text-sm">{validation?.length === 0 ? <p>All checks passed. Export the project, then run the data preparation command to generate the training directory.</p> : <ul className="list-disc space-y-1 pl-5">{validation?.slice(0, 150).map((issue, i) => <li key={i}>{issue}</li>)}</ul>}
          {(validation?.length ?? 0) > 150 && <p>{validation?.length} issues total; showing the first 150.</p>}
        </div>
      </DialogContent></Dialog>
      <Dialog open={showHelp} onOpenChange={setShowHelp}><DialogContent className="sm:max-w-2xl"><DialogHeader><DialogTitle>From annotation to training</DialogTitle><DialogDescription>Rotated boxes localize workpieces; semantic keypoints locate holes, notches and pose.</DialogDescription></DialogHeader>
        <ol className="list-decimal space-y-3 pl-5 text-sm">
          <li>Define keypoint names and criteria first, then load images by capture batch; symmetric workpieces need keypoints that distinguish head from tail.</li>
          <li>Draw one box per instance and set each keypoint's visibility. Mark true image positions under perspective; an OBB is not a perspective quadrilateral.</li>
          <li>Keep one split per batch and prepare train, val and test separately. Review every image, including negatives with no targets.</li>
          <li>Save and export the project and keep the source images. Run the command below in the repo directory; see docs/LOCALIZATION_WORKFLOW.md for training, validation and inference steps.</li>
        </ol>
        <pre className="overflow-auto rounded bg-muted p-3 text-xs">python tools/localization/prepare.py --project localization-project.json --images D:/images --output D:/localization-data --task both</pre>
        <p className="text-sm text-muted-foreground">In Datasets click "Import localization annotations" to create the dataset and configs; start TorchOBB / TorchPose training in Jobs. Return here afterwards for model pre-labeling. Projects do not include source images — save before closing the browser.</p>
        <p className="text-sm">Shortcuts: Ctrl+S save · Ctrl+Z / Ctrl+Y undo/redo · ←/→ previous/next image · Delete removes the selected instance · +/− zoom · 0 fit · 1 100% · Ctrl+wheel zooms at the cursor · Alt+drag or middle-button pans.</p>
      </DialogContent></Dialog>
    </div>
      </TabsContent>
    </Tabs>
  )
}
