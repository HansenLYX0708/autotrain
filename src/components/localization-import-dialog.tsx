'use client'

import { useEffect, useState } from 'react'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogTrigger } from './ui/dialog'
import { toast } from '@/hooks/use-toast'

type ImportReport = {
  sourceRevision: string; sourceImages: number; includedImages: number; excludedImages: string[];
  instances: number; labels: string[]; groupAssumption: string; errors: string[];
  splits: Record<string, { images: number; instances: number; groups: string[] }>
}

export function LocalizationImportDialog({ onImported }: { onImported: () => void }) {
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState(false)
  const [projects, setProjects] = useState<{ id: string; name: string; images: number }[]>([])
  const [form, setForm] = useState({
    annotationProjectId: '', imagesRoot: '', name: '', task: 'obb', splitMode: 'existing',
    valRatio: 0.2, testRatio: 0.2, seed: 42,
  })
  const [report, setReport] = useState<ImportReport | null>(null)
  const [created, setCreated] = useState<string | null>(null)
  useEffect(() => {
    if (!open) return
    fetch('/api/annotation/projects').then(async response => {
      const result = await response.json()
      if (!response.ok) throw new Error(result.error || 'Could not load saved annotations')
      setProjects(result.data)
    }).catch(error => toast({ title: '无法读取已保存标注', description: String(error.message), variant: 'destructive' }))
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
        toast({ title: '定位数据已导入', description: '项目、数据集、模型配置与训练参数已分别保存。请在 Models 中设置本地预训练权重，然后在 Jobs 创建并启动任务。' })
      }
    } catch (error) {
      toast({ title: preview ? '检查失败' : '导入失败', description: error instanceof Error ? error.message : 'Unknown error', variant: 'destructive' })
    } finally { setBusy(false) }
  }
  return <Dialog open={open} onOpenChange={value => { if (!busy) setOpen(value) }}>
    <DialogTrigger asChild><Button variant="outline">导入定位标注</Button></DialogTrigger>
    <DialogContent className="sm:max-w-3xl max-h-[90vh] overflow-y-auto">
      <DialogHeader>
        <DialogTitle>导入 OBB / Pose 定位数据</DialogTitle>
        <DialogDescription>从服务器已保存的标注生成训练数据。没有实例的图片会排除；原始图片和标注保留。</DialogDescription>
      </DialogHeader>
      <fieldset disabled={busy || Boolean(created)} className="space-y-4">
        <div className="space-y-1"><Label htmlFor="localization-source">已保存的标注项目</Label>
          <select id="localization-source" className="h-10 w-full rounded border bg-background px-3" value={form.annotationProjectId} onChange={event => change({ annotationProjectId: event.target.value })}>
            <option value="">选择标注项目</option>
            {projects.map(project => <option key={project.id} value={project.id}>{project.name} · {project.images} 张 · {project.id.slice(0, 8)}</option>)}
          </select>
        </div>
        <div className="space-y-1"><Label htmlFor="localization-root">服务器上的原图根目录</Label><Input id="localization-root" value={form.imagesRoot} onChange={event => change({ imagesRoot: event.target.value })} placeholder="G:\datasets\sliderlocation\gen5\corner\abs" /></div>
        <div className="space-y-1"><Label htmlFor="localization-name">新项目 / 数据集名称</Label><Input id="localization-name" value={form.name} onChange={event => change({ name: event.target.value })} /></div>
        <div className="grid grid-cols-2 gap-4">
          <div><Label htmlFor="localization-task">训练任务</Label><select id="localization-task" className="h-10 w-full rounded border bg-background px-3" value={form.task} onChange={event => change({ task: event.target.value })}><option value="obb">OBB 旋转框</option><option value="pose">Pose 结构关键点</option></select></div>
          <div><Label htmlFor="localization-split">数据划分</Label><select id="localization-split" className="h-10 w-full rounded border bg-background px-3" value={form.splitMode} onChange={event => change({ splitMode: event.target.value })}><option value="existing">采用标注项目中的批次和划分</option><option value="filename-index">按文件末尾编号分组，例如 Cal_LB-1</option></select></div>
        </div>
        {form.splitMode === 'filename-index' && <>
          <p className="text-sm text-muted-foreground">同一末尾编号的不同视角会放入同一个集合。仅当编号代表同次采集时使用，并检查下方的分组结果。</p>
          <div className="grid grid-cols-3 gap-3">{(['valRatio', 'testRatio', 'seed'] as const).map((key, index) => <div key={key}><Label htmlFor={`localization-${key}`}>{['验证比例', '测试比例', '随机种子'][index]}</Label><Input id={`localization-${key}`} type="number" step={key === 'seed' ? 1 : 0.05} value={form[key]} onChange={event => change({ [key]: Number(event.target.value) })} /></div>)}</div>
        </>}
        <Button type="button" onClick={() => submit(true)} disabled={!form.annotationProjectId || !form.name || !form.imagesRoot}>检查样本与划分</Button>
      </fieldset>
      {report && <div className="space-y-2 rounded border p-4 text-sm">
        <p>{report.sourceImages} 张原图 → 纳入 {report.includedImages} 张、{report.instances} 个实例；排除 {report.excludedImages.length} 张。类别：{report.labels.join(', ')}</p>
        {Object.entries(report.splits).map(([split, value]) => <p key={split}>{split}: {value.images} 张 / {value.instances} 个实例；组：{value.groups.join(', ') || '无'}</p>)}
        {report.errors.length > 0 && <div className="max-h-48 overflow-y-auto text-destructive"><p>需要修正 {report.errors.length} 项：</p>{report.errors.map((error, index) => <p key={index}>{error}</p>)}</div>}
        {created ? <p>已创建 {created}。在 Models、Training Configurations 和 Jobs 中可继续配置训练。</p> : <Button onClick={() => submit(false)} disabled={busy || report.errors.length > 0}>{busy ? '正在导入…' : '导入并创建三段配置'}</Button>}
      </div>}
    </DialogContent>
  </Dialog>
}
