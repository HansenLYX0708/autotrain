'use client'

import { useEffect, useState } from 'react'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogTrigger } from './ui/dialog'
import { toast } from '@/hooks/use-toast'

type DatasetRow = { id: string; name: string; format: string }
type Preview = {
  dataset: { id: string; name: string }; trainImages: number; trainInstances: number;
  copies: number; newImages: number; totalImages: number; note: string;
}

export function RotateAugmentDialog({ datasets, onCreated }: { datasets: DatasetRow[]; onCreated: () => void }) {
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState(false)
  const [datasetId, setDatasetId] = useState('')
  const [name, setName] = useState('')
  const [degrees, setDegrees] = useState(10)
  const [copies, setCopies] = useState(2)
  const [seed, setSeed] = useState(42)
  const [preview, setPreview] = useState<Preview | null>(null)

  const sources = datasets.filter(d => d.format === 'YOLOOBB' || d.format === '')
  useEffect(() => {
    if (!open) { setDatasetId(''); setName(''); setPreview(null) }
  }, [open])
  useEffect(() => { setPreview(null) }, [datasetId, degrees, copies, seed])

  const submit = async (action: 'preview' | 'create') => {
    setBusy(true)
    try {
      const response = await fetch('/api/datasets/rotate-augment', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, datasetId, name, degrees, copies, seed }),
      })
      const result = await response.json()
      if (!response.ok) throw new Error(result.error || 'Operation failed')
      if (action === 'preview') setPreview(result.data)
      else {
        const t = result.data.report.totals
        toast({ title: 'Rotated-augmented dataset created', description: `Kept ${t.kept}, clipped ${t.shrunk}, dropped ${t.dropped} boxes.` })
        setOpen(false)
        onCreated()
      }
    } catch (error) {
      toast({ title: action === 'preview' ? 'Preview failed' : 'Create failed', description: error instanceof Error ? error.message : 'Unknown error', variant: 'destructive' })
    } finally { setBusy(false) }
  }

  return <Dialog open={open} onOpenChange={value => { if (!busy) setOpen(value) }}>
    <DialogTrigger asChild><Button variant="outline" disabled={!sources.length}>Create rotated-augmented dataset</Button></DialogTrigger>
    <DialogContent className="sm:max-w-xl max-h-[90vh] overflow-y-auto">
      <DialogHeader>
        <DialogTitle>Create rotated-augmented dataset (OBB)</DialogTitle>
        <DialogDescription>Apply inscribed-rotation augmentation to the training images (no gray borders; clipped boxes keep exact angles), producing a new dataset; val/test are copied as-is and the source is not modified.</DialogDescription>
      </DialogHeader>
      <fieldset disabled={busy} className="space-y-4">
        <div className="space-y-1"><Label>Source dataset</Label>
          <select className="h-10 w-full rounded border bg-background px-3" value={datasetId} onChange={e => setDatasetId(e.target.value)}>
            <option value="">Select a YOLOOBB dataset</option>
            {sources.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
          </select></div>
        <div className="space-y-1"><Label>New dataset name</Label>
          <Input value={name} onChange={e => setName(e.target.value)} placeholder="slider-rot10" /></div>
        <div className="grid grid-cols-3 gap-3">
          <div><Label>±Rotation degrees</Label><Input type="number" min={1} max={45} step={1} value={degrees} onChange={e => setDegrees(Number(e.target.value))} /></div>
          <div><Label>Copies per image</Label><Input type="number" min={1} max={20} step={1} value={copies} onChange={e => setCopies(Number(e.target.value))} /></div>
          <div><Label>Seed</Label><Input type="number" min={0} step={1} value={seed} onChange={e => setSeed(Number(e.target.value))} /></div>
        </div>
        <Button type="button" onClick={() => submit('preview')} disabled={!datasetId}>Preview</Button>
      </fieldset>
      {preview && <div className="space-y-2 rounded border p-4 text-sm">
        <p>{preview.trainImages} training images ({preview.trainInstances} instances) × {preview.copies} copies = {preview.newImages} new images, {preview.totalImages} total after merge.</p>
        <p className="text-amber-600 text-xs">{preview.note}</p>
        <Button type="button" onClick={() => submit('create')} disabled={busy || !name.trim()}>Create</Button>
      </div>}
    </DialogContent>
  </Dialog>
}
