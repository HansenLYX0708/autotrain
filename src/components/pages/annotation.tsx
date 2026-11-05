'use client'

import { useCallback, useEffect, useReducer, useRef, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'
import { useAuth } from '@/contexts/auth-context'
import { resolveUploadedLabelmeImage, type ImageNameIndex } from '@/lib/labelme-image-match'
import { toast } from '@/hooks/use-toast'
import { Download, Save, Undo2, Redo2, Trash2, ZoomIn, ZoomOut, FolderOpen, Crosshair, MousePointer2, Square, RotateCw } from 'lucide-react'
import {
  type AnnotationImage, type AnnotationObject, type AnnotationProject, type Point, type Split,
  boxCorners, boxFromThreePoints, containsPoint, fromLabelme, imageIssues, insideImage,
  newProject, projectIssues, projectSchema, toLabelme,
} from '@/lib/localization-annotations'

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
function report(error: unknown) { toast({ title: '操作未完成', description: errorMessage(error), variant: 'destructive' }) }
const selectClass = 'h-9 w-full rounded-md border bg-background px-2 text-sm'
const splitNames = { unassigned: '未划分', train: '训练 train', val: '验证 val', test: '测试 test' }
type Tool = 'select' | 'rectangle' | 'obb' | 'point'
type Asset = { url: string; width: number; height: number }

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
  const [busy, setBusy] = useState(false)
  const [preview, setPreview] = useState<AnnotationObject | null>(null)
  const drag = useRef<{ start: Point; object: AnnotationObject; key?: string } | null>(null)
  const pan = useRef<{ x: number; y: number; left: number; top: number } | null>(null)
  const viewport = useRef<HTMLDivElement>(null)
  const svg = useRef<SVGSVGElement>(null)
  const [settings, setSettings] = useState(false)
  const [labelsDraft, setLabelsDraft] = useState('')
  const [pointsDraft, setPointsDraft] = useState('')
  const [serverProjects, setServerProjects] = useState<Array<{ id: string; name: string; images: number; modified: string }> | null>(null)
  const [validation, setValidation] = useState<string[] | null>(null)
  const [showHelp, setShowHelp] = useState(false)
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
      setZoom(Math.min((viewport.current.clientWidth - 24) / width, (viewport.current.clientHeight - 24) / height, 2))
      viewport.current.scrollTo(0, 0)
    }
  }
  const save = useCallback(async () => {
    if (busy || !project.images.length) return
    setBusy(true)
    try {
      const snapshot = projectSchema.parse(project)
      const response = await fetch('/api/annotation/projects', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(snapshot),
      })
      const result = await response.json()
      if (!response.ok) throw new Error(result.error || '保存失败')
      setSaved(project)
      toast({ title: '项目已保存', description: '标注、模板和划分已保存到服务器；原图仍在你的本地文件夹。' })
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
      if (busy || settings || serverProjects || validation || showHelp) return
      const modifier = event.ctrlKey || event.metaKey
      if (modifier && event.key.toLowerCase() === 's') { event.preventDefault(); void save(); return }
      if (modifier && event.key.toLowerCase() === 'z') { event.preventDefault(); cancelGesture(); dispatch({ type: event.shiftKey ? 'redo' : 'undo' }); return }
      if (modifier && event.key.toLowerCase() === 'y') { event.preventDefault(); cancelGesture(); dispatch({ type: 'redo' }); return }
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
        if ('error' in resolved) throw new Error(`${relative(file)}：${resolved.error}`)
        const key = resolved.path.toLowerCase()
        if (imports.has(key)) throw new Error(`多个 JSON 对应同一图片：${resolved.path}`)
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
        if (Object.keys(loaded).some(k => k.toLowerCase() === name.toLowerCase())) throw new Error(`重复文件名：${name}`)
        const url = URL.createObjectURL(file)
        loaded[name] = { url, width: 0, height: 0 }
        const dims = await new Promise<{ width: number; height: number }>((resolve, reject) => {
          const img = new Image()
          img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight })
          img.onerror = () => reject(new Error(`无法解码图片：${name}`))
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
        if (record && (record.width !== dims.width || record.height !== dims.height)) throw new Error(`${name}：图片尺寸与标注不一致，禁止错位续标`)
        records.push(record ? { ...record, name } : {
          name, ...dims, objects: [], reviewed: false, split: 'unassigned',
          group: name.includes('/') ? name.slice(0, name.lastIndexOf('/')) : '',
        })
      }
      if (!records.length) throw new Error('没有找到 PNG/JPEG/BMP/WebP 图片；TIFF 请先无缩放转换为 PNG')
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
      toast({ title: '图片已载入', description: `${Object.keys(loaded).length} 张图片；原有标注优先保留。重新打开项目时请关联同一个图片根目录。` })
    } catch (error) {
      Object.values(loaded).forEach(a => URL.revokeObjectURL(a.url))
      report(error)
    } finally { setBusy(false) }
  }

  function loadProject(raw: unknown) {
    const next = projectSchema.parse(raw)
    if (dirty && !window.confirm('当前项目有未保存修改。仍要打开另一个项目吗？')) return
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
    toast({ title: '标注已恢复', description: '请点击“关联图片文件夹”选择原图根目录。' })
  }

  function coords(event: React.PointerEvent<SVGSVGElement>): Point | null {
    const matrix = svg.current?.getScreenCTM()
    if (!matrix) return null
    const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse())
    return { x: point.x, y: point.y }
  }
  function pointerDown(event: React.PointerEvent<SVGSVGElement>) {
    if (!image || !asset) return
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
        drag.current = { start: p, object, key }
        event.currentTarget.setPointerCapture(event.pointerId)
      }
      return
    }
    if (tool === 'point') {
      if (!selected || !activeKey) { report(new Error('先选择一个实例和关键点名称')); return }
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
      if (geometry.width < 1 || geometry.height < 1) throw new Error('框的宽和高必须至少为 1 像素')
      if (!boxCorners(geometry).every(q => insideImage(q, image))) throw new Error('旋转框超出图片，请调整第三个点')
      const object: AnnotationObject = { ...geometry, id: crypto.randomUUID(), label: activeLabel, kind: tool, keypoints: {} }
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
    if (d.key && d.object.keypoints[d.key]) {
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
      toast({ title: '项目已导出', description: '包含所有标注和模板，不包含原图。请与原图一起备份。' })
    } catch (error) { report(error) }
  }
  if (user?.role !== 'admin') return <p className="text-muted-foreground">请使用管理员账号访问标注工具。</p>
  const issues = image ? imageIssues(image, project.keypointNames) : []

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold">定位标注 · OBB + 关键点</h1>
          <p className="text-sm text-muted-foreground">{project.name} · {project.images.length} 张 · {dirty ? '有未保存修改' : '已保存 / 未修改'}</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" onClick={() => setShowHelp(true)}>标注与训练指南</Button>
          <Button variant="outline" disabled={busy} onClick={async () => {
            try {
              const response = await fetch('/api/annotation/projects')
              const result = await response.json()
              if (!response.ok) throw new Error(result.error)
              setServerProjects(result.data)
            } catch (error) { report(error) }
          }}><FolderOpen className="mr-1 size-4" />打开已保存项目</Button>
          <Button disabled={busy || !image} onClick={save}><Save className="mr-1 size-4" />{busy ? '处理中…' : '保存项目 Ctrl+S'}</Button>
          <Button variant="outline" disabled={!image} onClick={exportProject}><Download className="mr-1 size-4" />导出项目</Button>
        </div>
      </div>
      <div className="grid gap-4 xl:grid-cols-[230px_minmax(0,1fr)_290px]" inert={busy}>
        <div className="space-y-3">
          <Card><CardHeader className="pb-2"><CardTitle className="text-sm">项目与图片</CardTitle></CardHeader><CardContent className="space-y-3">
            <Input aria-label="项目名称" value={project.name} onChange={e => edit({ ...project, name: e.target.value })} />
            <label className="block cursor-pointer rounded border border-dashed p-3 text-center text-sm">
              关联图片文件夹
              <input type="file" className="hidden" multiple {...{ webkitdirectory: '', directory: '' }}
                disabled={busy} onChange={e => { void loadFolder(Array.from(e.target.files ?? [])); e.target.value = '' }} />
            </label>
            <label className="block cursor-pointer text-sm underline">导入项目 JSON
              <input type="file" accept=".json" className="hidden" disabled={busy} onChange={async e => {
                const file = e.target.files?.[0]
                e.target.value = ''
                if (!file) return
                try { loadProject(JSON.parse((await file.text()).replace(/^\uFEFF/, ''))) } catch (error) { report(error) }
              }} />
            </label>
            <Button className="w-full" variant="outline" onClick={() => loadProject(newProject())}>新建项目</Button>
            <Button className="w-full" variant="outline" onClick={() => { setLabelsDraft(project.labels.join('\n')); setPointsDraft(project.keypointNames.join('\n')); setSettings(true) }}>类别与关键点模板</Button>
            <p className="text-xs text-muted-foreground">原图保留在本地。续标时先打开项目，再关联原图根目录。支持同目录同名 Labelme JSON。</p>
          </CardContent></Card>
          <div className="max-h-[420px] space-y-1 overflow-auto rounded border p-1" aria-label="图片列表">
            {project.images.map((item, i) => <button key={item.name} className={`block w-full rounded p-2 text-left text-xs ${i === index ? 'bg-accent' : 'hover:bg-muted'}`} onClick={() => navigate(i)}>
              <span className="break-all">{item.reviewed ? '✓ ' : '○ '}{item.name}</span>
              <span className="block text-muted-foreground">{item.objects.length} 个实例 · {splitNames[item.split]}{!assets[item.name] && ' · 缺少原图'}</span>
            </button>)}
          </div>
        </div>
        <div className="min-w-0 space-y-3">
          <div className="flex flex-wrap items-center gap-2 rounded border p-2">
            {([
              ['select', '选择/拖动', MousePointer2], ['rectangle', '水平框(2点)', Square],
              ['obb', '旋转框(3点)', RotateCw], ['point', '关键点', Crosshair],
            ] as const).map(([value, text, Icon]) => <Button key={value} size="sm" variant={tool === value ? 'default' : 'outline'} onClick={() => { cancelGesture(); setTool(value) }}>
              <Icon className="mr-1 size-4" />{text}
            </Button>)}
            <Button size="icon" variant="outline" aria-label="撤销" disabled={!history.past.length} onClick={() => { cancelGesture(); dispatch({ type: 'undo' }) }}><Undo2 className="size-4" /></Button>
            <Button size="icon" variant="outline" aria-label="重做" disabled={!history.future.length} onClick={() => { cancelGesture(); dispatch({ type: 'redo' }) }}><Redo2 className="size-4" /></Button>
            <Button size="icon" variant="outline" aria-label="放大" onClick={() => setZoom(v => Math.min(20, v * 1.25))}><ZoomIn className="size-4" /></Button>
            <Button size="icon" variant="outline" aria-label="缩小" onClick={() => setZoom(v => Math.max(0.02, v / 1.25))}><ZoomOut className="size-4" /></Button>
            <Button variant="outline" size="sm" onClick={() => fit()}>适应 {Math.round(zoom * 100)}%</Button>
          </div>
          <p className="text-xs text-muted-foreground">
            {tool === 'obb' ? `依次点击第一条边的两端，再点击对边位置（${pending.length}/3）；Esc 取消。` :
              tool === 'rectangle' ? '依次点击对角两点；Esc 取消。' :
                tool === 'point' ? `在选中实例上放置 ${activeKey || '关键点'}；可重复点击修正。` : '点击选择实例；拖动框整体移动，拖动圆点修改关键点。'}
            {' '}Alt + 拖动或鼠标中键平移；滚动条浏览放大后的图片。
          </p>
          <div ref={viewport} className="relative h-[600px] overflow-auto rounded border bg-slate-950">
            {image && asset ? <svg ref={svg} width={image.width * zoom} height={image.height * zoom} viewBox={`0 0 ${image.width} ${image.height}`}
              className="block touch-none select-none" style={{ maxWidth: 'none', cursor: tool === 'select' ? 'default' : 'crosshair' }}
              aria-label="标注画布" onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={pointerUp}
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
                  {Object.entries(shown.keypoints).filter(([, p]) => p.visibility > 0).map(([name, p]) => <g key={name}>
                    <circle data-key={name} cx={p.x} cy={p.y} r={5 / zoom} fill={p.visibility === 1 ? 'transparent' : color} stroke={color} strokeWidth={2 / zoom} strokeDasharray={p.visibility === 1 ? `${3 / zoom} ${2 / zoom}` : undefined} />
                    <text pointerEvents="none" x={p.x + 8 / zoom} y={p.y - 5 / zoom} fontSize={12 / zoom} fill={color} stroke="#000" strokeWidth={2 / zoom} paintOrder="stroke">{name}</text>
                  </g>)}
                </g>
              })}
              {pending.length > 0 && <polyline points={draftCorners().map(p => `${p.x},${p.y}`).join(' ')} fill="none" stroke="#38bdf8" strokeWidth={2 / zoom} strokeDasharray={`${5 / zoom} ${4 / zoom}`} pointerEvents="none" />}
            </svg> : <div className="flex h-full items-center justify-center p-8 text-center text-sm text-slate-300">{image ? '标注已载入，请关联原图文件夹。' : '选择图片文件夹开始。建议按采集批次建立子目录。'}</div>}
          </div>
          <div className="flex items-center justify-between gap-2">
            <Button variant="outline" disabled={index <= 0} onClick={() => navigate(index - 1)}>上一张 ←</Button>
            <span className="truncate text-xs">{image ? `${index + 1}/${project.images.length} · ${image.name} · ${image.width}×${image.height}` : '未载入'}</span>
            <Button variant="outline" disabled={index >= project.images.length - 1} onClick={() => navigate(index + 1)}>下一张 →</Button>
          </div>
          {image && <Card><CardContent className="space-y-3 pt-4">
            <div className="grid gap-3 sm:grid-cols-3">
              <label className="space-y-1 text-xs">采集批次（同工件连续照片用同一批次）
                <Input value={image.group} onChange={e => editImage({ ...image, group: e.target.value })} placeholder="例如 lot_01_camera_A" />
              </label>
              <label className="space-y-1 text-xs">数据划分
                <select className={selectClass} value={image.split} onChange={e => editImage({ ...image, split: e.target.value as Split })}>
                  {Object.entries(splitNames).map(([value, text]) => <option key={value} value={value}>{text}</option>)}
                </select>
              </label>
              <Button className="self-end" variant="outline" disabled={!image.group} onClick={() => {
                edit({ ...project, images: project.images.map(v => v.group === image.group ? { ...v, split: image.split } : v) })
              }}>将划分应用到同批次</Button>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <Button variant={image.reviewed ? 'secondary' : 'default'} disabled={!asset} onClick={() => editImage({ ...image, reviewed: !image.reviewed })}>
                {image.reviewed ? '✓ 已确认（点击撤回）' : image.objects.length ? '确认本图标注完成' : '确认本图没有目标'}
              </Button>
              <Button variant="outline" onClick={() => setValidation(projectIssues(project, false))}>检查 OBB 数据</Button>
              <Button variant="outline" onClick={() => setValidation(projectIssues(project, true))}>检查 Pose 数据</Button>
            </div>
            {issues.length > 0 && <p className="text-xs text-amber-600">{issues.slice(0, 3).join('；')}{issues.length > 3 ? `；另有 ${issues.length - 3} 项` : ''}</p>}
          </CardContent></Card>}
        </div>
        <div className="space-y-3">
          <Card><CardHeader className="pb-2"><CardTitle className="text-sm">实例</CardTitle></CardHeader><CardContent className="space-y-3">
            <label className="text-xs">新框类别
              <select className={selectClass} value={activeLabel} onChange={e => setLabel(e.target.value)}>{project.labels.map(v => <option key={v}>{v}</option>)}</select>
            </label>
            <div className="max-h-40 space-y-1 overflow-auto">
              {image?.objects.map((o, i) => <button key={o.id} className={`block w-full rounded p-2 text-left text-xs ${selectedId === o.id ? 'bg-accent' : 'hover:bg-muted'}`} onClick={() => { cancelGesture(); setSelectedId(o.id); setTool('select') }}>
                #{i + 1} {o.label} · {o.kind === 'obb' ? '旋转框' : '水平框'} · {Object.keys(o.keypoints).length}/{project.keypointNames.length} 点
              </button>)}
            </div>
            {selected && <>
              <select aria-label="选中实例类别" className={selectClass} value={selected.label} onChange={e => editObject({ ...selected, label: e.target.value })}>{project.labels.map(v => <option key={v}>{v}</option>)}</select>
              <div className="grid grid-cols-2 gap-2">
                {([['cx', '中心 X'], ['cy', '中心 Y'], ['width', '宽'], ['height', '高'], ['angle', '顺时针角度 °']] as const).map(([field, text]) => <label key={`${selected.id}-${field}-${selected[field]}`} className="text-xs">{text}
                  <Input type="number" step="0.1" defaultValue={Number(selected[field].toFixed(3))} onBlur={e => {
                    const value = Number(e.target.value)
                    if (!e.target.value || !Number.isFinite(value) || ((field === 'width' || field === 'height') && value < 1) || (field === 'angle' && Math.abs(value) > 180)) { e.target.value = String(selected[field]); report(new Error('请输入有效数值；宽高 ≥ 1，角度在 -180 到 180 之间')); return }
                    if (value !== selected[field]) editObject({ ...selected, [field]: value, kind: field === 'angle' ? 'obb' : selected.kind })
                  }} />
                </label>)}
              </div>
              <p className="text-xs text-muted-foreground">几何数值修改只调整框，不改变特征点的图像位置。拖动框会整体移动框和关键点。</p>
              <Button variant="destructive" size="sm" onClick={removeSelected}><Trash2 className="mr-1 size-4" />删除实例和关联点</Button>
            </>}
          </CardContent></Card>
          <Card><CardHeader className="pb-2"><CardTitle className="text-sm">关键点 · 绑定选中实例</CardTitle></CardHeader><CardContent className="space-y-3">
            <p className="text-xs text-muted-foreground">先选名称，再点击图像放点。0 = 无法确定；1 = 遮挡但位置可确定；2 = 可见。空白表示尚未处理。</p>
            {!selected && <p className="rounded bg-muted p-3 text-sm">请先在图中或实例列表中选择一个工件。</p>}
            {selected && project.keypointNames.map(name => {
              const point = selected?.keypoints[name]
              return <div key={name} className="space-y-1 rounded border p-2">
                <button className={`text-sm ${tool === 'point' && activeKey === name ? 'font-bold text-primary' : ''}`} disabled={!selected} onClick={() => { cancelGesture(); setKeypointName(name); setTool('point') }}>{name} {tool === 'point' && activeKey === name ? '← 放置中' : ''}</button>
                <select aria-label={`${name} 可见性`} className={selectClass} value={point?.visibility ?? ''} disabled={!selected} onChange={e => {
                  if (!selected) return
                  const points = { ...selected.keypoints }
                  if (e.target.value === '') delete points[name]
                  else if (e.target.value === '0') points[name] = { x: 0, y: 0, visibility: 0 }
                  else if (!point || point.visibility === 0) { setKeypointName(name); setTool('point'); report(new Error('请先在图像上放置该点，再设置可见性')); return }
                  else points[name] = { ...point, visibility: Number(e.target.value) as 1 | 2 }
                  editObject({ ...selected, keypoints: points })
                }}>
                  <option value="">待标注</option><option value="2">2 · 可见</option><option value="1">1 · 遮挡，位置确定</option><option value="0">0 · 无法确定 / 不适用</option>
                </select>
                {point && point.visibility > 0 && <p className="text-xs text-muted-foreground">({point.x.toFixed(2)}, {point.y.toFixed(2)}) px</p>}
              </div>
            })}
            <Button className="w-full" variant="outline" disabled={!image} onClick={() => image && download(image.name.split('/').pop()!.replace(/\.[^.]+$/, '.json'), toLabelme(image, project.keypointNames))}>导出当前图 Labelme</Button>
          </CardContent></Card>
        </div>
      </div>
      <Dialog open={settings} onOpenChange={setSettings}><DialogContent className="sm:max-w-2xl"><DialogHeader><DialogTitle>类别与关键点模板</DialogTitle><DialogDescription>每行一个名称，顺序就是训练类别 / 关键点顺序。模板应用于项目内所有类别；结构不同的工件建议分项目标注。</DialogDescription></DialogHeader>
        <div className="grid gap-3 sm:grid-cols-2">
          <label className="text-sm">类别<textarea className="mt-1 h-48 w-full rounded border bg-background p-2" value={labelsDraft} onChange={e => setLabelsDraft(e.target.value)} /></label>
          <label className="text-sm">关键点<textarea className="mt-1 h-48 w-full rounded border bg-background p-2" value={pointsDraft} onChange={e => setPointsDraft(e.target.value)} /></label>
        </div>
        <p className="text-xs text-muted-foreground">datum_a/b/c 是占位名，请在开始标注前替换为明确的孔位、缺口等名称。命名按工件自身结构，不随图像上下左右改变。已使用的名称不能直接删除或改名；修改模板后所有图片需重新确认。</p>
        <Button onClick={changeTemplate}>应用模板</Button>
      </DialogContent></Dialog>
      <Dialog open={serverProjects !== null} onOpenChange={open => !open && setServerProjects(null)}><DialogContent className="sm:max-w-xl"><DialogHeader><DialogTitle>打开我的标注项目</DialogTitle><DialogDescription>恢复标注后，重新关联原图文件夹即可续标。</DialogDescription></DialogHeader>
        <div className="max-h-96 space-y-2 overflow-auto">{serverProjects?.length === 0 && <p className="text-sm">还没有保存的项目。</p>}
          {serverProjects?.map(p => <Button key={p.id} variant="outline" className="h-auto w-full justify-start whitespace-normal text-left" onClick={async () => {
            try {
              const response = await fetch(`/api/annotation/projects?id=${encodeURIComponent(p.id)}`)
              const result = await response.json()
              if (!response.ok) throw new Error(result.error)
              loadProject(result.data)
            } catch (error) { report(error) }
          }}>{p.name} · {p.images} 张 · {new Date(p.modified).toLocaleString()}</Button>)}
        </div>
      </DialogContent></Dialog>
      <Dialog open={validation !== null} onOpenChange={open => !open && setValidation(null)}><DialogContent className="sm:max-w-2xl"><DialogHeader><DialogTitle>训练数据检查</DialogTitle><DialogDescription>草稿可以保存；准备训练数据时，以下问题必须解决。</DialogDescription></DialogHeader>
        <div className="max-h-96 overflow-auto text-sm">{validation?.length === 0 ? <p>检查通过。导出项目后，使用数据准备命令生成训练目录。</p> : <ul className="list-disc space-y-1 pl-5">{validation?.slice(0, 150).map((issue, i) => <li key={i}>{issue}</li>)}</ul>}
          {(validation?.length ?? 0) > 150 && <p>共 {validation?.length} 项，显示前 150 项。</p>}
        </div>
      </DialogContent></Dialog>
      <Dialog open={showHelp} onOpenChange={setShowHelp}><DialogContent className="sm:max-w-2xl"><DialogHeader><DialogTitle>从标注到训练</DialogTitle><DialogDescription>旋转框定位工件；语义关键点定位孔位、缺口与姿态。</DialogDescription></DialogHeader>
        <ol className="list-decimal space-y-3 pl-5 text-sm">
          <li>先定义关键点名称和判定标准，再按采集批次载入图片；对称工件需要可区分头尾的特征点。</li>
          <li>每个实例绘制一个框，逐点处理可见性。透视下仍标真实图像位置；OBB 不是透视四边形。</li>
          <li>同批次保持同一划分，分别准备 train、val、test。确认每张图片，包含没有目标的负样本。</li>
          <li>保存并导出项目，保留原图。在仓库目录执行下面的命令；具体训练、验证、推理步骤见 docs/LOCALIZATION_WORKFLOW.md。</li>
        </ol>
        <pre className="overflow-auto rounded bg-muted p-3 text-xs">python tools/localization/prepare.py --project localization-project.json --images D:/images --output D:/localization-data --task both</pre>
        <p className="text-sm text-muted-foreground">本页提供标注与训练数据准备。OBB / Pose 使用独立 Ultralytics 环境和配套命令行训练入口；当前 Training Jobs 尚未接入这两种任务。标注项目不包含原图，浏览器关闭前请保存。</p>
      </DialogContent></Dialog>
    </div>
  )
}
