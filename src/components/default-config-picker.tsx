'use client'

import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { yamlSyntaxError } from '@/components/config-yaml-pane'

interface DefaultConfig { name: string; path: string; content: string }

/** Loads a complete template into the editor without discarding custom YAML fields. */
interface PickerProps {
  projectId: string
  kind: 'models' | 'training-configs'
  onLoad: (config: DefaultConfig) => void
}

export function DefaultConfigPicker(props: PickerProps) {
  return <ConfigPicker key={`${props.kind}:${props.projectId}`} {...props} />
}

function ConfigPicker({ projectId, kind, onLoad }: PickerProps) {
  const [configs, setConfigs] = useState<DefaultConfig[]>([])
  const [directory, setDirectory] = useState('')
  const [selected, setSelected] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(!!projectId)
  const [refresh, setRefresh] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    if (!projectId) return () => controller.abort()
    fetch(`/api/${kind}/import?projectId=${encodeURIComponent(projectId)}`, { signal: controller.signal })
      .then(async response => {
        const result = await response.json()
        if (!response.ok || !result.success) throw new Error(result.error || 'Failed to load defaults')
        if (controller.signal.aborted) return
        setConfigs(result.data.defaultConfigs || [])
        setDirectory(result.data.defaultConfigDir || '')
      })
      .catch(error => { if (!controller.signal.aborted) setError(String(error.message || error)) })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [projectId, kind, refresh])

  return (
    <div className="space-y-2 rounded-md border p-3">
      <Label>Import default {kind === 'models' ? 'model' : 'training'} configuration</Label>
      <div className="flex gap-2">
        <Select value={selected} disabled={!projectId || loading || !configs.length} onValueChange={value => {
          const config = configs.find(item => item.path === value)
          if (!config) return
          const problem = yamlSyntaxError(config.content)
          if (problem) { setError(problem); return }
          setSelected(value)
          setError('')
          onLoad(config)
        }}>
          <SelectTrigger aria-label={`Default ${kind === 'models' ? 'model' : 'training'} configuration`}>
            <SelectValue placeholder={!projectId ? 'Select a project first' : loading ? 'Loading...' : configs.length ? 'Select a default configuration' : 'No default configurations'} />
          </SelectTrigger>
          <SelectContent>
            {configs.map(config => <SelectItem key={config.path} value={config.path}>{config.name}</SelectItem>)}
          </SelectContent>
        </Select>
        <Button type="button" variant="outline" disabled={!projectId || loading} onClick={() => {
          setLoading(true)
          setError('')
          setSelected('')
          setRefresh(value => value + 1)
        }}>Refresh</Button>
      </div>
      {directory && <p className="text-xs text-muted-foreground break-all">Add .yml or .yaml files to: {directory}</p>}
      {selected && <p className="text-xs text-muted-foreground">Template loaded into the YAML editor. Edit it there to preserve all template settings.</p>}
      {error && <p role="alert" className="text-xs text-destructive">{error}</p>}
    </div>
  )
}
