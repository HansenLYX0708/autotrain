'use client'

import { useEffect, useState } from 'react'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogTrigger } from './ui/dialog'
import { toast } from '@/hooks/use-toast'
import { frameworkMeta } from '@/lib/frameworks'

type DatasetRow = {
  id: string; name: string; format: string; projectId: string
  project?: { id: string; name: string; framework?: string }
}
type Preview = {
  format: string; framework: string; canMerge: boolean; blockers: string[]; warnings: string[];
  sources: { id: string; name: string; prefix: string; classes: string[]; splits: Record<string, number>; fileCount: number; bytes: number }[];
  classes?: { union: string[]; mismatched: boolean };
  duplicates: { first: string; second: string }[];
  needsAllowClassMismatch?: boolean;
}

const FORMAT_NAME: Record<string, string> = {
  COCO: 'COCO detection', PaddleSeg: 'PaddleSeg segmentation', YOLOOBB: 'YOLOOBB rotated box',
  YOLOPose: 'YOLOPose keypoints', AnomalyFolder: 'AnomalyFolder anomaly detection',
}

export function MergeDatasetsDialog({ datasets, onMerged }: { datasets: DatasetRow[]; onMerged: () => void }) {
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState(false)
  const [selected, setSelected] = useState<string[]>([])
  const [name, setName] = useState('')
  const [projectId, setProjectId] = useState('')
  const [allowMismatch, setAllowMismatch] = useState(false)
  const [preview, setPreview] = useState<Preview | null>(null)

  const framework = (d: DatasetRow) => d.project?.framework ?? ''
  const effectiveFormat = (d: DatasetRow) => {
    if (d.format) return d.format
    try { return framework(d) ? frameworkMeta(framework(d) as 'TorchOBB').datasetFormat : '' } catch { return '' }
  }
  const first = datasets.find(d => d.id === selected[0])
  const compatible = (d: DatasetRow) =>
    !selected.length || (!selected.includes(d.id) && effectiveFormat(d) === (first ? effectiveFormat(first) : '') && framework(d) === (first ? framework(first) : ''))
  const reason = (d: DatasetRow) =>
    effectiveFormat(d) !== (first ? effectiveFormat(first) : '') ? 'different format' : framework(d) !== (first ? framework(first) : '') ? 'different framework' : ''
  const projects = [...new Map(datasets.filter(d => framework(d) === (first ? framework(first) : '')).map(d => [d.projectId, d.project?.name ?? d.projectId])).entries()]

  useEffect(() => {
    if (!open) { setSelected([]); setName(''); setProjectId(''); setPreview(null); setAllowMismatch(false) }
  }, [open])
  useEffect(() => { setPreview(null) }, [selected, name, projectId])

  const toggle = (id: string) =>
    setSelected(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id])

  const submit = async (action: 'preview' | 'merge') => {
    setBusy(true)
    try {
      const response = await fetch('/api/datasets/merge', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, datasetIds: selected, name, projectId: projectId || undefined, allowClassMismatch: allowMismatch }),
      })
      const result = await response.json()
      if (!response.ok) {
        if (result.data?.blockers) setPreview(result.data)
        throw new Error(result.error || 'Merge failed')
      }
      if (action === 'preview') setPreview(result.data)
      else {
        toast({ title: 'Datasets merged', description: `New dataset ${result.data.dataset.name} created; source datasets unchanged.` })
        setOpen(false)
        onMerged()
      }
    } catch (error) {
      toast({ title: action === 'preview' ? 'Preview failed' : 'Merge failed', description: error instanceof Error ? error.message : 'Unknown error', variant: 'destructive' })
    } finally { setBusy(false) }
  }

  return <Dialog open={open} onOpenChange={value => { if (!busy) setOpen(value) }}>
    <DialogTrigger asChild><Button variant="outline">Merge datasets</Button></DialogTrigger>
    <DialogContent className="sm:max-w-3xl max-h-[90vh] overflow-y-auto">
      <DialogHeader>
        <DialogTitle>Merge datasets of the same type</DialogTitle>
        <DialogDescription>Merge multiple datasets with the same format and framework into a new dataset; each source keeps its own train/val/test split, file names get the source prefix, and sources are not modified.</DialogDescription>
      </DialogHeader>
      <fieldset disabled={busy} className="space-y-4">
        <div className="space-y-1">
          <Label>Select datasets (the first one defines the class order)</Label>
          <div className="max-h-56 overflow-y-auto rounded border p-2 space-y-1">
            {datasets.filter(d => d.format !== 'labelme').map(d => {
              const picked = selected.includes(d.id)
              const ok = picked || compatible(d)
              return <label key={d.id} className={`flex items-center gap-2 text-sm ${ok ? '' : 'opacity-50'}`}>
                <input type="checkbox" checked={picked} disabled={!ok} onChange={() => toggle(d.id)} />
                <span className="flex-1">{d.name}</span>
                <span className="text-xs text-muted-foreground">{FORMAT_NAME[effectiveFormat(d)] ?? effectiveFormat(d)}{!ok && !picked ? ` · ${reason(d)}` : ''}</span>
              </label>
            })}
          </div>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1"><Label htmlFor="merge-name">New dataset name</Label>
            <Input id="merge-name" value={name} onChange={e => setName(e.target.value)} placeholder="merged-dataset" /></div>
          <div className="space-y-1"><Label htmlFor="merge-project">Target project</Label>
            <select id="merge-project" className="h-10 w-full rounded border bg-background px-3" value={projectId} onChange={e => setProjectId(e.target.value)}>
              <option value="">Default: first source's project</option>
              {projects.map(([id, pname]) => <option key={id} value={id}>{pname}</option>)}
            </select></div>
        </div>
        <Button type="button" onClick={() => submit('preview')} disabled={selected.length < 2}>Preview</Button>
      </fieldset>
      {preview && <div className="space-y-3 rounded border p-4 text-sm">
        {preview.blockers.map((b, i) => <p key={i} className="text-destructive">{b}</p>)}
        {preview.warnings.map((w, i) => <p key={i} className="text-amber-600">{w}</p>)}
        {preview.duplicates.length > 0 && <div>
          <p className="font-medium text-destructive">Duplicate files (identical content):</p>
          <ul className="max-h-24 overflow-y-auto text-xs">{preview.duplicates.map((d, i) => <li key={i}>{d.first} ≡ {d.second}</li>)}</ul>
        </div>}
        <div>
          <p className="font-medium">Source datasets</p>
          <ul className="text-xs space-y-1">{preview.sources.map(s => <li key={s.id}>
            <span className="font-mono">{s.prefix}__</span> {s.name}: {Object.entries(s.splits).map(([k, v]) => `${k} ${v}`).join(', ')}; {s.fileCount} files
            {s.classes.length > 0 && `; classes: ${s.classes.join(', ')}`}
          </li>)}</ul>
        </div>
        {preview.classes && <p className="text-xs">Merged classes ({preview.classes.union.length}): {preview.classes.union.join(', ') || '—'}</p>}
        {preview.needsAllowClassMismatch && <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={allowMismatch} onChange={e => setAllowMismatch(e.target.checked)} />
          Class sets differ — merge anyway (batches missing a class may contain unannotated instances)
        </label>}
        <Button type="button" onClick={() => submit('merge')}
          disabled={busy || !preview.canMerge && !(preview.needsAllowClassMismatch && allowMismatch && preview.blockers.length === 0) || !name.trim()}>
          Merge
        </Button>
      </div>}
    </DialogContent>
  </Dialog>
}
