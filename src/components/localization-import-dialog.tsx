'use client'

import { useEffect, useState } from 'react'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogTrigger } from './ui/dialog'
import { toast } from '@/hooks/use-toast'

type ImportReport = {
  sourceRevision: string; sourceImages: number; includedImages: number; excludedImages: string[]; emptyImages: string[]; unreviewedImages: string[];
  instances: number; labels: string[]; groupAssumption: string; errors: string[]; rotatedConverted?: number;
  splits: Record<string, { images: number; instances: number; groups: string[]; excludedUnreviewed: number }>
}

export function LocalizationImportDialog({ onImported }: { onImported: () => void }) {
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState(false)
  const [projects, setProjects] = useState<{ id: string; name: string; images: number }[]>([])
  const [form, setForm] = useState({
    annotationProjectId: '', imagesRoot: '', name: '', task: 'obb', splitMode: 'existing',
    valRatio: 0.2, testRatio: 0.2, seed: 42, createConfigs: 'dataset', detectFramework: 'TorchDet',
  })
  const [report, setReport] = useState<ImportReport | null>(null)
  const [created, setCreated] = useState<string | null>(null)
  useEffect(() => {
    if (!open) return
    fetch('/api/annotation/projects').then(async response => {
      const result = await response.json()
      if (!response.ok) throw new Error(result.error || 'Could not load saved annotations')
      setProjects(result.data)
    }).catch(error => toast({ title: 'Could not load saved annotations', description: String(error.message), variant: 'destructive' }))
  }, [open])
  const change = (value: Partial<typeof form>) => { setForm(previous => ({ ...previous, ...value })); setReport(null); setCreated(null) }
  const submit = async (preview: boolean) => {
    setBusy(true)
    try {
      const response = await fetch('/api/datasets/import-localization', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...form, preview, sourceRevision: report?.sourceRevision }),
      })
      const result = await response.json()
      if (!response.ok) throw new Error(result.error || 'Import failed')
      setReport(result.data.report)
      if (!preview) {
        setCreated(result.data.project.name)
        onImported()
        toast({ title: 'Localization data imported', description: form.createConfigs === 'dataset'
          ? 'Project and dataset config were saved. Create the model and training configurations in Models / Training Configurations, then create and start the job in Jobs.'
          : 'Project, dataset, model and training configs were saved. Set local pre-trained weights in Models, then create and start the job in Jobs.' })
      }
    } catch (error) {
      toast({ title: preview ? 'Check failed' : 'Import failed', description: error instanceof Error ? error.message : 'Unknown error', variant: 'destructive' })
    } finally { setBusy(false) }
  }
  return <Dialog open={open} onOpenChange={value => { if (!busy) setOpen(value) }}>
    <DialogTrigger asChild><Button variant="outline">Import localization annotations</Button></DialogTrigger>
    <DialogContent className="sm:max-w-3xl max-h-[90vh] overflow-y-auto">
      <DialogHeader>
        <DialogTitle>Import OBB / Pose / Detection localization data</DialogTitle>
        <DialogDescription>Generate training data from annotations saved on the server. Only reviewed images with instances are imported; images without instances or with unreviewed labels (including model pre-labels) are excluded. Source images and annotations are kept.</DialogDescription>
      </DialogHeader>
      <fieldset disabled={busy || Boolean(created)} className="space-y-4">
        <div className="space-y-1"><Label htmlFor="localization-source">Active annotation set</Label>
          <select id="localization-source" className="h-10 w-full rounded border bg-background px-3" value={form.annotationProjectId} onChange={event => change({ annotationProjectId: event.target.value })}>
            <option value="">Select annotation project</option>
            {projects.map(project => <option key={project.id} value={project.id}>{project.name} · {project.images} images · {project.id.slice(0, 8)}</option>)}
          </select>
          <p className="text-xs text-muted-foreground">Only active sets can be imported. Archived sets are reactivated on the Annotation page.</p>
        </div>
        <div className="space-y-1"><Label htmlFor="localization-root">Source image root on the server</Label><Input id="localization-root" value={form.imagesRoot} onChange={event => change({ imagesRoot: event.target.value })} placeholder="G:\datasets\sliderlocation\gen5\corner\abs" /></div>
        <div className="space-y-1"><Label htmlFor="localization-name">New project / dataset name</Label><Input id="localization-name" value={form.name} onChange={event => change({ name: event.target.value })} /></div>
        <div className="grid grid-cols-2 gap-4">
          <div><Label htmlFor="localization-task">Training task</Label><select id="localization-task" className="h-10 w-full rounded border bg-background px-3" value={form.task} onChange={event => change({ task: event.target.value })}><option value="obb">OBB rotated box</option><option value="pose">Pose structural keypoints</option><option value="detect">Axis-aligned box detection (COCO)</option></select></div>
          <div><Label htmlFor="localization-split">Data split</Label><select id="localization-split" className="h-10 w-full rounded border bg-background px-3" value={form.splitMode} onChange={event => change({ splitMode: event.target.value })}><option value="existing">Use batches and splits from the annotation project</option><option value="filename-index">Group by trailing filename index, e.g. Cal_LB-1</option><option value="random">Random split (ratio and seed)</option></select></div>
        </div>
        {form.task === 'detect' && <>
          <div className="space-y-1"><Label htmlFor="localization-detect-framework">Detection framework</Label>
            <select id="localization-detect-framework" className="h-10 w-full rounded border bg-background px-3" value={form.detectFramework} onChange={event => change({ detectFramework: event.target.value })}>
              <option value="TorchDet">TorchDet (bundled torchtrain)</option>
              <option value="PaddleDetection">PaddleDetection</option>
            </select>
          </div>
          <p className="text-sm text-muted-foreground">Axis-aligned rectangles are exported as-is; rotated boxes become their enclosing upright bounding box, and keypoints are ignored. The output is a standard COCO dataset (data/train, data/val, data/test + instance_*.json).</p>
        </>}
        <div className="space-y-1"><Label htmlFor="localization-configs">Configs to create</Label>
          <select id="localization-configs" className="h-10 w-full rounded border bg-background px-3" value={form.createConfigs} onChange={event => change({ createConfigs: event.target.value })}>
            <option value="dataset">Dataset config only</option>
            <option value="all">Dataset + model + training params (three configs)</option>
          </select>
        </div>
        {form.splitMode !== 'existing' && <>
          <p className="text-sm text-muted-foreground">{form.splitMode === 'random'
            ? 'Only reviewed images are split randomly, ignoring the train / val / test set on the annotation page. Images sharing a capture batch stay in one split; images without a batch are split independently, so similar consecutive shots may land in both train and validation and inflate metrics.'
            : 'Different views sharing a trailing index go into the same split. Use this only when the index means one capture, and check the grouping below.'}</p>
          <div className="grid grid-cols-3 gap-3">{(['valRatio', 'testRatio', 'seed'] as const).map((key, index) => <div key={key}><Label htmlFor={`localization-${key}`}>{['Val ratio', 'Test ratio', 'Seed'][index]}</Label><Input id={`localization-${key}`} type="number" step={key === 'seed' ? 1 : 0.05} value={form[key]} onChange={event => change({ [key]: Number(event.target.value) })} /></div>)}</div>
        </>}
        <Button type="button" onClick={() => submit(true)} disabled={!form.annotationProjectId || !form.name || !form.imagesRoot}>Check samples & splits</Button>
      </fieldset>
      {report && <div className="space-y-2 rounded border p-4 text-sm">
        <p>{report.sourceImages} source images → {report.includedImages} included, {report.instances} instances; {report.excludedImages.length} excluded ({report.emptyImages.length} unannotated, {report.unreviewedImages.length} labeled but unreviewed). Classes: {report.labels.join(', ')}</p>
        {report.unreviewedImages.length > 0 && <details className="text-muted-foreground"><summary>Unreviewed images (not imported)</summary>{report.unreviewedImages.map(name => <p key={name}>{name}</p>)}</details>}
        {(report.rotatedConverted ?? 0) > 0 && <p className="text-amber-600">⚠ {report.rotatedConverted} rotated box(es) will be converted to their enclosing axis-aligned bounding boxes.</p>}
        {Object.entries(report.splits).map(([split, value]) => <p key={split}>{split}: {value.images} images / {value.instances} instances{value.excludedUnreviewed > 0 && ` (${value.excludedUnreviewed} more marked ${split} but unreviewed, excluded)`}; groups: {form.splitMode === 'random' && value.groups.every(group => group.startsWith('image:')) ? 'per-image' : value.groups.map(group => group.startsWith('image:') ? '(single)' : group).filter((group, index, all) => all.indexOf(group) === index).join(', ') || 'none'}</p>)}
        {report.errors.length > 0 && <div className="max-h-48 overflow-y-auto text-destructive"><p>{report.errors.length} issues to fix:</p>{report.errors.map((error, index) => <p key={index}>{error}</p>)}</div>}
        {created ? <p>Created {created}. {form.createConfigs === 'dataset' ? 'Create the model and training configurations in Models / Training Configurations, then create the job in Jobs.' : 'Continue configuring in Models, Training Configurations and Jobs.'}</p> : <Button onClick={() => submit(false)} disabled={busy || report.errors.length > 0}>{busy ? 'Importing…' : form.createConfigs === 'dataset' ? 'Import & create dataset config' : 'Import & create all three configs'}</Button>}
      </div>}
    </DialogContent>
  </Dialog>
}
