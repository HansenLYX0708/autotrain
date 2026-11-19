'use client'

import { useCallback, useEffect, useState } from 'react'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from './ui/dialog'
import { Folder, FolderOpen, Image as ImageIcon, ArrowUp, Loader2 } from 'lucide-react'

type Entry = { name: string; path: string; type: 'dir' | 'image' }
type BrowseData = { path: string | null; parent: string | null; entries: Entry[]; truncated: boolean }

const IMAGE_RE = /\.(jpe?g|png|bmp|gif|tiff?|webp)$/i

export function ServerPathPicker({ value, onSelect, mode, title }: {
  value: string
  onSelect: (path: string) => void
  mode: 'file-or-dir' | 'dir'
  title: string
}) {
  const [open, setOpen] = useState(false)
  const [data, setData] = useState<BrowseData | null>(null)
  const [currentPath, setCurrentPath] = useState('')
  const [selected, setSelected] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const browse = useCallback(async (dir: string | null, preselect: string | null = null, fallbackToRoots = false) => {
    setLoading(true); setError(null)
    try {
      const params = new URLSearchParams()
      if (dir) params.set('path', dir)
      if (mode === 'file-or-dir') params.set('images', '1')
      const response = await fetch(`/api/fs/browse?${params}`)
      const result = await response.json()
      if (!response.ok || !result.success) throw new Error(result.error || 'Browse failed')
      setData(result.data)
      setCurrentPath(result.data.path ?? '')
      setSelected(preselect)
    } catch (err) {
      if (fallbackToRoots && dir) return browse(null)
      setError(err instanceof Error ? err.message : 'Browse failed')
    } finally { setLoading(false) }
  }, [mode])

  useEffect(() => {
    if (!open) return
    setData(null); setSelected(null); setError(null)
    const initial = value.trim()
    if (initial && IMAGE_RE.test(initial)) {
      // Keep the separator for drive roots: "D:" alone resolves to the server's cwd on that drive.
      const parent = initial.replace(/[/\\][^/\\]+$/, match => (/^[A-Za-z]:$/.test(initial.slice(0, initial.length - match.length)) ? match[0] : ''))
      void browse(parent || null, initial, true)
    } else {
      void browse(initial || null, null, true)
    }
  }, [open])

  const confirm = (path: string) => { onSelect(path); setOpen(false) }
  const atRoots = data !== null && data.path === null

  return <Dialog open={open} onOpenChange={setOpen}>
    <DialogTrigger asChild>
      <Button variant="outline" size="icon" type="button"><FolderOpen className="w-4 h-4" /></Button>
    </DialogTrigger>
    <DialogContent className="sm:max-w-2xl">
      <DialogHeader><DialogTitle>{title}</DialogTitle></DialogHeader>
      <div className="flex gap-2">
        <Button variant="outline" type="button" disabled={atRoots && !currentPath}
          onClick={() => browse(data?.parent ?? null)} title="Up one level">
          <ArrowUp className="w-4 h-4" />
        </Button>
        <Input value={currentPath} onChange={event => setCurrentPath(event.target.value)}
          onKeyDown={event => { if (event.key === 'Enter') browse(currentPath || null) }}
          placeholder="Enter a server path; empty lists roots" />
        <Button variant="outline" type="button" onClick={() => browse(currentPath || null)}>Go</Button>
      </div>
      <div className="h-72 overflow-y-auto rounded border p-2 text-sm">
        {loading && <div className="flex items-center gap-2 p-4 text-muted-foreground"><Loader2 className="w-4 h-4 animate-spin" />Loading…</div>}
        {!loading && error && <p className="p-4 text-destructive">{error}</p>}
        {!loading && !error && data && data.entries.length === 0 && <p className="p-4 text-muted-foreground">(empty directory)</p>}
        {!loading && !error && data?.entries.map(entry => (
          <button key={entry.path} type="button"
            className={`flex w-full items-center gap-2 rounded px-2 py-1 text-left hover:bg-accent ${selected === entry.path ? 'bg-accent font-medium' : ''}`}
            onClick={() => entry.type === 'dir' ? browse(entry.path) : setSelected(entry.path)}
            onDoubleClick={() => { if (entry.type === 'image') confirm(entry.path) }}>
            {entry.type === 'dir' ? <Folder className="w-4 h-4 shrink-0 text-amber-500" /> : <ImageIcon className="w-4 h-4 shrink-0 text-blue-500" />}
            <span className="truncate">{entry.name}</span>
          </button>
        ))}
        {data?.truncated && <p className="p-2 text-xs text-muted-foreground">Showing the first 2000 entries only</p>}
      </div>
      <div className="flex justify-end gap-2">
        {mode === 'file-or-dir' && <Button type="button" disabled={!selected || !IMAGE_RE.test(selected)}
          onClick={() => selected && confirm(selected)}>Select this image</Button>}
        <Button type="button" variant={mode === 'file-or-dir' ? 'outline' : 'default'} disabled={!data?.path}
          onClick={() => data?.path && confirm(data.path)}>Select current folder</Button>
      </div>
    </DialogContent>
  </Dialog>
}
