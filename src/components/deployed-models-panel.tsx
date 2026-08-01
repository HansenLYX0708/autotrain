'use client'

/**
 * DeployedModelsPanel
 *
 * A drop-in card for pages that need "one-click switching" between multiple
 * trained models registered per product label. Kept self-contained so it can
 * be embedded on the Validation page or any future inference page.
 *
 * Interactions with parent:
 *   - onLoad(deployed)    : parent should populate its own selectedJob /
 *                           selectedCheckpoint / saveDir state from the record
 *                           so downstream Eval/Infer commands use the exact
 *                           configPath + weightsPath of the deployed model.
 *   - registerContext     : the currently-picked checkpoint on the parent
 *                           (framework, jobId, jobName, configPath, weightsPath,
 *                           exportedDir, metrics). Enables the "Register as
 *                           production" flow without a manual re-entry.
 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import {
  Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle,
} from '@/components/ui/dialog'
import { useToast } from '@/hooks/use-toast'
import { CheckCircle2, PackagePlus, Rocket, Trash2, RefreshCw, Zap } from 'lucide-react'

export interface DeployedModel {
  id: string
  name: string
  product: string
  framework: string
  architecture?: string | null
  configPath: string
  weightsPath: string
  exportedDir?: string | null
  metrics?: string | null
  notes?: string | null
  isActive: boolean
  projectId?: string | null
  trainingJobId?: string | null
  project?: { id: string; name: string; framework: string } | null
  trainingJob?: { id: string; name: string; status: string } | null
  createdAt: string
  updatedAt: string
}

export interface RegisterContext {
  framework: string
  projectId?: string | null
  trainingJobId?: string | null
  jobName?: string | null
  configPath?: string | null
  weightsPath?: string | null
  exportedDir?: string | null
  architecture?: string | null
  metrics?: Record<string, unknown> | null
}

interface Props {
  onLoad: (m: DeployedModel) => void
  registerContext?: RegisterContext | null
  /** Restrict the panel to a single framework (e.g. current project's). */
  frameworkFilter?: string
}

export function DeployedModelsPanel({ onLoad, registerContext, frameworkFilter }: Props) {
  const { toast } = useToast()

  const [rows, setRows] = useState<DeployedModel[]>([])
  const [products, setProducts] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedProduct, setSelectedProduct] = useState<string>('__all__')
  const [selectedId, setSelectedId] = useState<string>('')

  // Register dialog state
  const [regOpen, setRegOpen] = useState(false)
  const [regName, setRegName] = useState('')
  const [regProduct, setRegProduct] = useState('')
  const [regNotes, setRegNotes] = useState('')
  const [regActivate, setRegActivate] = useState(true)
  const [regBusy, setRegBusy] = useState(false)

  const fetchList = useCallback(async () => {
    setLoading(true)
    try {
      const qs = new URLSearchParams()
      if (frameworkFilter) qs.set('framework', frameworkFilter)
      const res = await fetch(`/api/deployed-models?${qs.toString()}`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setRows(data.data || [])
      setProducts(data.products || [])
    } catch (e) {
      console.error('Failed to load deployed models:', e)
    } finally {
      setLoading(false)
    }
  }, [frameworkFilter])

  useEffect(() => { fetchList() }, [fetchList])

  const filtered = useMemo(() => {
    if (selectedProduct === '__all__') return rows
    return rows.filter(r => r.product === selectedProduct)
  }, [rows, selectedProduct])

  const activeInProduct = useMemo(() => {
    if (selectedProduct === '__all__') return null
    return rows.find(r => r.product === selectedProduct && r.isActive) || null
  }, [rows, selectedProduct])

  const selected = useMemo(
    () => rows.find(r => r.id === selectedId) || null,
    [rows, selectedId],
  )

  // Auto-pick the active model of the chosen product on filter change.
  useEffect(() => {
    if (selectedProduct === '__all__') {
      if (!selectedId && rows.length > 0) setSelectedId(rows[0].id)
      return
    }
    const act = rows.find(r => r.product === selectedProduct && r.isActive)
    const first = rows.find(r => r.product === selectedProduct)
    setSelectedId((act || first)?.id || '')
  }, [selectedProduct, rows]) // eslint-disable-line react-hooks/exhaustive-deps

  // --- Actions ---
  const activate = async (id: string) => {
    try {
      const res = await fetch(`/api/deployed-models/${id}/activate`, { method: 'POST' })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.error || `HTTP ${res.status}`)
      }
      toast({ title: 'Activated', description: 'This model is now the active version for its product.' })
      await fetchList()
    } catch (e) {
      toast({ title: 'Activation failed', description: e instanceof Error ? e.message : String(e), variant: 'destructive' })
    }
  }

  const remove = async (id: string) => {
    if (!confirm('Delete this deployed model entry? (Underlying weight files are NOT deleted.)')) return
    try {
      const res = await fetch(`/api/deployed-models/${id}`, { method: 'DELETE' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      toast({ title: 'Deleted' })
      await fetchList()
    } catch (e) {
      toast({ title: 'Delete failed', description: e instanceof Error ? e.message : String(e), variant: 'destructive' })
    }
  }

  const openRegister = () => {
    if (!registerContext?.configPath || !registerContext?.weightsPath) {
      toast({
        title: 'Nothing to register',
        description: 'Select a training job and checkpoint above first.',
        variant: 'destructive',
      })
      return
    }
    // Prefill sensible defaults from context.
    const stamp = new Date().toISOString().slice(0, 10).replace(/-/g, '')
    const jn = (registerContext.jobName || 'model').replace(/[^A-Za-z0-9_.-]/g, '_')
    setRegName(`${jn}-${stamp}`)
    setRegProduct(selectedProduct !== '__all__' ? selectedProduct : '')
    setRegNotes('')
    setRegActivate(true)
    setRegOpen(true)
  }

  const submitRegister = async () => {
    if (!registerContext) return
    if (!regName.trim() || !regProduct.trim()) {
      toast({ title: 'Missing fields', description: 'Name and Product are required.', variant: 'destructive' })
      return
    }
    setRegBusy(true)
    try {
      const body = {
        name: regName.trim(),
        product: regProduct.trim(),
        framework: registerContext.framework,
        architecture: registerContext.architecture || undefined,
        configPath: registerContext.configPath,
        weightsPath: registerContext.weightsPath,
        exportedDir: registerContext.exportedDir || undefined,
        projectId: registerContext.projectId || undefined,
        trainingJobId: registerContext.trainingJobId || undefined,
        metrics: registerContext.metrics || undefined,
        notes: regNotes || undefined,
        activate: regActivate,
      }
      const res = await fetch('/api/deployed-models', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data?.error || `HTTP ${res.status}`)
      const warnings: string[] = data?.warnings || []
      toast({
        title: 'Registered',
        description: warnings.length ? `With warnings:\n${warnings.join('\n')}` : `${regName} → ${regProduct}`,
      })
      setRegOpen(false)
      await fetchList()
      setSelectedProduct(regProduct.trim())
      setSelectedId(data.data.id)
    } catch (e) {
      toast({ title: 'Register failed', description: e instanceof Error ? e.message : String(e), variant: 'destructive' })
    } finally {
      setRegBusy(false)
    }
  }

  const parseMetrics = (s?: string | null): string => {
    if (!s) return ''
    try {
      const o = JSON.parse(s) as Record<string, unknown>
      const parts: string[] = []
      if (typeof o.mIoU === 'number') parts.push(`mIoU=${(o.mIoU as number).toFixed(4)}`)
      if (typeof o.mAP === 'number') parts.push(`mAP=${(o.mAP as number).toFixed(4)}`)
      if (typeof o.bestIter === 'number') parts.push(`iter=${o.bestIter}`)
      if (typeof o.epoch === 'number') parts.push(`epoch=${o.epoch}`)
      return parts.join(' · ')
    } catch { return '' }
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg flex items-center gap-2">
              <Rocket className="w-5 h-5" />
              Deployed Models
            </CardTitle>
            <CardDescription>
              Switch quickly between production models registered per product
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={fetchList} disabled={loading}>
              <RefreshCw className={`w-4 h-4 mr-1 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
            <Button size="sm" onClick={openRegister} disabled={!registerContext?.weightsPath}>
              <PackagePlus className="w-4 h-4 mr-1" />
              Register current checkpoint
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent>
        <div className="grid gap-4 md:grid-cols-3">
          <div>
            <Label>Product</Label>
            <Select value={selectedProduct} onValueChange={setSelectedProduct}>
              <SelectTrigger className="mt-1.5">
                <SelectValue placeholder="All products" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__all__">All products</SelectItem>
                {products.map(p => (
                  <SelectItem key={p} value={p}>{p}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="md:col-span-2">
            <Label>Model {activeInProduct ? <Badge variant="outline" className="ml-2">active: {activeInProduct.name}</Badge> : null}</Label>
            <Select value={selectedId} onValueChange={setSelectedId} disabled={filtered.length === 0}>
              <SelectTrigger className="mt-1.5">
                <SelectValue placeholder={filtered.length === 0 ? 'No models registered yet' : 'Select a deployed model'} />
              </SelectTrigger>
              <SelectContent>
                {filtered.map(m => (
                  <SelectItem key={m.id} value={m.id}>
                    <div className="flex items-center gap-2">
                      {m.isActive ? <CheckCircle2 className="w-3.5 h-3.5 text-green-600" /> : null}
                      <span className="font-medium">{m.name}</span>
                      <span className="text-muted-foreground text-xs">· {m.product}</span>
                      <span className="text-muted-foreground text-xs">· {m.framework}</span>
                      {parseMetrics(m.metrics) && (
                        <span className="text-muted-foreground text-xs">· {parseMetrics(m.metrics)}</span>
                      )}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {selected && (
          <div className="mt-4 rounded-lg border bg-muted/40 p-3 text-xs space-y-1">
            <div className="flex flex-wrap gap-x-4 gap-y-1">
              <span><span className="text-muted-foreground">config:</span> <code className="break-all">{selected.configPath}</code></span>
              <span><span className="text-muted-foreground">weights:</span> <code className="break-all">{selected.weightsPath}</code></span>
              {selected.exportedDir && (
                <span><span className="text-muted-foreground">export:</span> <code className="break-all">{selected.exportedDir}</code></span>
              )}
              {selected.trainingJob && (
                <span><span className="text-muted-foreground">job:</span> {selected.trainingJob.name}</span>
              )}
              {selected.notes && (
                <span><span className="text-muted-foreground">notes:</span> {selected.notes}</span>
              )}
            </div>
            <div className="flex items-center gap-2 pt-2">
              <Button size="sm" onClick={() => onLoad(selected)}>
                <Zap className="w-4 h-4 mr-1" />
                Load into validator
              </Button>
              {!selected.isActive && (
                <Button size="sm" variant="outline" onClick={() => activate(selected.id)}>
                  <CheckCircle2 className="w-4 h-4 mr-1" />
                  Set as active for &quot;{selected.product}&quot;
                </Button>
              )}
              <Button size="sm" variant="ghost" className="text-destructive hover:text-destructive" onClick={() => remove(selected.id)}>
                <Trash2 className="w-4 h-4 mr-1" />
                Delete entry
              </Button>
            </div>
          </div>
        )}
      </CardContent>

      {/* Register dialog */}
      <Dialog open={regOpen} onOpenChange={setRegOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Register as production model</DialogTitle>
            <DialogDescription>
              Create a switchable production entry for the currently selected checkpoint. Underlying weight files are referenced, not copied.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-3">
            <div>
              <Label>Product <span className="text-destructive">*</span></Label>
              <Input
                value={regProduct}
                onChange={(e) => setRegProduct(e.target.value)}
                placeholder="e.g. line13, waferA, TEM-1300kx"
                list="deployed-products-datalist"
              />
              <datalist id="deployed-products-datalist">
                {products.map(p => <option key={p} value={p} />)}
              </datalist>
            </div>
            <div>
              <Label>Version name <span className="text-destructive">*</span></Label>
              <Input value={regName} onChange={(e) => setRegName(e.target.value)} placeholder="e.g. v3-2026-07" />
            </div>
            <div>
              <Label>Notes</Label>
              <Textarea rows={2} value={regNotes} onChange={(e) => setRegNotes(e.target.value)} placeholder="Optional" />
            </div>
            <div className="flex items-center gap-2 text-sm">
              <input
                id="reg-activate"
                type="checkbox"
                checked={regActivate}
                onChange={(e) => setRegActivate(e.target.checked)}
              />
              <label htmlFor="reg-activate">Set as active for this product</label>
            </div>

            {registerContext && (
              <div className="rounded border bg-muted/40 p-2 text-xs space-y-1">
                <div><span className="text-muted-foreground">framework:</span> {registerContext.framework}</div>
                <div><span className="text-muted-foreground">config:</span> <code className="break-all">{registerContext.configPath}</code></div>
                <div><span className="text-muted-foreground">weights:</span> <code className="break-all">{registerContext.weightsPath}</code></div>
                {registerContext.exportedDir && (
                  <div><span className="text-muted-foreground">export:</span> <code className="break-all">{registerContext.exportedDir}</code></div>
                )}
              </div>
            )}
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setRegOpen(false)} disabled={regBusy}>Cancel</Button>
            <Button onClick={submitRegister} disabled={regBusy}>
              {regBusy ? 'Registering…' : 'Register'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  )
}
