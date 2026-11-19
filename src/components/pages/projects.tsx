'use client'

import { useEffect, useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  FRAMEWORK_LIST,
  FRAMEWORK_META,
  isDetection,
  isSegmentation,
  isTorch,
  normalizeFramework,
  resolveTask,
} from '@/lib/frameworks'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  Plus,
  Search,
  MoreHorizontal,
  Pencil,
  Trash2,
  FolderKanban,
  Database,
  Cpu,
  Calendar,
} from 'lucide-react'
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from '@/components/ui/pagination'
import { toast } from '@/hooks/use-toast'

interface Project {
  id: string
  name: string
  description: string | null
  framework: string
  task?: string
  status: string
  createdAt: string
  updatedAt: string
  _count?: {
    datasets: number
    models: number
    trainingJobs: number
  }
}

export function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [view, setView] = useState<'recent' | 'all'>('recent')
  const [page, setPage] = useState(1)
  const [searchInput, setSearchInput] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [dialogOpen, setDialogOpen] = useState(false)
  const [editingProject, setEditingProject] = useState<Project | null>(null)
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    framework: 'PaddleDetection',
    task: 'detection',
  })

  const PAGE_SIZE = 12

  // Debounce the search input ~300 ms before it hits the server.
  useEffect(() => {
    const timer = setTimeout(() => setSearchQuery(searchInput.trim()), 300)
    return () => clearTimeout(timer)
  }, [searchInput])

  // A new view or query always starts on page 1.
  useEffect(() => {
    setPage(1)
  }, [view, searchQuery])

  useEffect(() => {
    void fetchProjects()
  }, [view, page, searchQuery])

  const fetchProjects = async () => {
    setLoading(true)
    try {
      const params = new URLSearchParams()
      if (searchQuery) params.set('q', searchQuery)
      if (view === 'recent') {
        params.set('limit', '5')
      } else {
        params.set('page', String(page))
        params.set('pageSize', String(PAGE_SIZE))
      }
      const response = await fetch(`/api/projects?${params}`)
      if (response.ok) {
        const data = await response.json()
        setProjects(data.data || [])
        setTotal(data.total ?? 0)
        // If the current page no longer exists (deletes/shrinking results), step back.
        const pageSize = data.pageSize || PAGE_SIZE
        const lastPage = Math.max(1, Math.ceil((data.total ?? 0) / pageSize))
        if (view === 'all' && page > lastPage) setPage(lastPage)
      }
    } catch (error) {
      console.error('Failed to fetch projects:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    try {
      if (editingProject) {
        const response = await fetch(`/api/projects/${editingProject.id}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(formData),
        })
        if (response.ok) {
          toast({ title: 'Project updated successfully' })
          fetchProjects()
          setDialogOpen(false)
          setEditingProject(null)
        }
      } else {
        const response = await fetch('/api/projects', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(formData),
        })
        if (response.ok) {
          toast({ title: 'Project created successfully' })
          fetchProjects()
          setDialogOpen(false)
          setFormData({ name: '', description: '', framework: 'PaddleDetection', task: 'detection' })
        }
      }
    } catch (error) {
      toast({ title: 'Error saving project', variant: 'destructive' })
    }
  }

  const handleDelete = async (project: Project) => {
    const relatedData = [
      project._count?.datasets ? `${project._count.datasets} datasets` : null,
      project._count?.models ? `${project._count.models} models` : null,
      project._count?.trainingJobs ? `${project._count.trainingJobs} training jobs` : null,
    ].filter(Boolean)

    const warningMessage = relatedData.length > 0
      ? `This project contains:\n${relatedData.map(item => `• ${item}`).join('\n')}\n\nDeleting it will permanently delete all related data. This cannot be undone. Continue?`
      : 'Delete this project? This cannot be undone.'

    if (!confirm(warningMessage)) {
      return
    }
    
    try {
      const response = await fetch(`/api/projects/${project.id}`, {
        method: 'DELETE',
      })
      if (response.ok) {
        toast({ title: 'Project deleted successfully' })
        fetchProjects()
      }
    } catch (error) {
      toast({ title: 'Error deleting project', variant: 'destructive' })
    }
  }

  const openEditDialog = (project: Project) => {
    setEditingProject(project)
    setFormData({
      name: project.name,
      description: project.description || '',
      framework: project.framework,
      task: project.task || 'detection',
    })
    setDialogOpen(true)
  }

  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE))
  const rangeStart = projects.length === 0 ? 0 : (view === 'all' ? (page - 1) * PAGE_SIZE + 1 : 1)
  const rangeEnd = view === 'all' ? (page - 1) * PAGE_SIZE + projects.length : projects.length

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Projects</h1>
          <p className="text-muted-foreground">
            Manage your training projects
          </p>
        </div>
        <Dialog open={dialogOpen} onOpenChange={(open) => {
          setDialogOpen(open)
          if (!open) {
            setEditingProject(null)
            setFormData({ name: '', description: '', framework: 'PaddleDetection', task: 'detection' })
          }
        }}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="w-4 h-4 mr-2" />
              New Project
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>{editingProject ? 'Edit Project' : 'Create New Project'}</DialogTitle>
              <DialogDescription>
                {editingProject ? 'Update your project details.' : 'Create a new project to organize your datasets and models.'}
              </DialogDescription>
            </DialogHeader>
            <form onSubmit={handleSubmit}>
              <div className="space-y-4 py-4">
                <div className="space-y-2">
                  <Label htmlFor="name">Project Name</Label>
                  <Input
                    id="name"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    placeholder="My Object Detection Project"
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="description">Description</Label>
                  <Textarea
                    id="description"
                    value={formData.description}
                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    placeholder="Describe your project..."
                    rows={3}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="framework">Training Framework</Label>
                  <Select
                    value={formData.framework}
                    onValueChange={(value) => setFormData({
                      ...formData,
                      framework: value,
                      // Task is derived from the framework registry, so a switch can
                      // never leave a project on a task its framework cannot run.
                      task: resolveTask(value, formData.task),
                    })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select framework" />
                    </SelectTrigger>
                    <SelectContent>
                      {/* Grouped by runtime: the two families need different
                          Python environments and different repository paths. */}
                      <SelectGroup>
                        <SelectLabel>PaddlePaddle</SelectLabel>
                        {FRAMEWORK_LIST.filter((f) => FRAMEWORK_META[f].family === 'paddle').map((f) => (
                          <SelectItem key={f} value={f}>{FRAMEWORK_META[f].label}</SelectItem>
                        ))}
                      </SelectGroup>
                      <SelectGroup>
                        <SelectLabel>PyTorch</SelectLabel>
                        {FRAMEWORK_LIST.filter((f) => FRAMEWORK_META[f].family === 'torch').map((f) => (
                          <SelectItem key={f} value={f}>{FRAMEWORK_META[f].label}</SelectItem>
                        ))}
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                </div>
                {isSegmentation(formData.framework) && (
                  <p className="text-xs text-muted-foreground">
                    {FRAMEWORK_META[normalizeFramework(formData.framework)].label} projects use
                    semantic segmentation (per-pixel masks) and train by iteration.
                  </p>
                )}
                {isTorch(formData.framework) && (
                  <p className="text-xs text-muted-foreground">
                    Runs on the bundled <code>torchtrain</code> trainer. Make sure Settings has a
                    Framework Python environment for {normalizeFramework(formData.framework)} that
                    has PyTorch installed.
                  </p>
                )}
                {isDetection(formData.framework) && (
                  <div className="space-y-2">
                    <Label htmlFor="task">Task Type</Label>
                    <Select
                      value={formData.task}
                      onValueChange={(value) => setFormData({ ...formData, task: value })}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select task type" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="detection">Object Detection</SelectItem>
                        <SelectItem
                          value="instance_segmentation"
                          // TorchDet's torchvision detectors are box-only here, so
                          // offering polygon masks would promise something the
                          // trainer does not deliver.
                          disabled={isTorch(formData.framework)}
                        >
                          Instance Segmentation
                          {isTorch(formData.framework) ? ' (PaddleDetection only)' : ''}
                        </SelectItem>
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-muted-foreground">
                      Instance Segmentation enables polygon masks (e.g. Mask R-CNN, SOLOv2).
                    </p>
                  </div>
                )}
              </div>
              <DialogFooter>
                <Button type="submit">{editingProject ? 'Update' : 'Create'}</Button>
              </DialogFooter>
            </form>
          </DialogContent>
        </Dialog>
      </div>

      {/* Search + view toggle */}
      <div className="flex flex-wrap items-center gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search projects..."
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            className="pl-9"
          />
        </div>
        <div className="flex items-center gap-1 rounded-md border p-1">
          {(['recent', 'all'] as const).map((value) => (
            <Button
              key={value}
              size="sm"
              variant={view === value ? 'default' : 'ghost'}
              onClick={() => setView(value)}
            >
              {value === 'recent' ? 'Recent 5' : 'All projects'}
            </Button>
          ))}
        </div>
        {!loading && (
          <p className="text-sm text-muted-foreground ml-auto">
            {view === 'recent' && !searchQuery
              ? `Showing ${projects.length} most recent of ${total}`
              : `Showing ${rangeStart}–${rangeEnd} of ${total}`}
          </p>
        )}
      </div>

      {/* Projects Grid */}
      {loading ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3].map((i) => (
            <Card key={i} className="animate-pulse">
              <CardHeader>
                <div className="h-5 bg-muted rounded w-1/2"></div>
                <div className="h-4 bg-muted rounded w-3/4"></div>
              </CardHeader>
              <CardContent>
                <div className="h-4 bg-muted rounded w-full"></div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : projects.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <FolderKanban className="w-12 h-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium">No projects found</h3>
            <p className="text-sm text-muted-foreground mt-1">
              {searchQuery ? 'Try a different search term' : view === 'recent' ? 'Create your first project to get started' : 'No projects on this page'}
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {projects.map((project) => (
            <Card key={project.id} className="hover:shadow-md transition-shadow">
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                  <div>
                    <CardTitle className="text-lg">{project.name}</CardTitle>
                    <CardDescription className="mt-1">
                      {project.description || 'No description'}
                    </CardDescription>
                  </div>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="icon">
                        <MoreHorizontal className="w-4 h-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem onClick={() => openEditDialog(project)}>
                        <Pencil className="w-4 h-4 mr-2" />
                        Edit
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        className="text-destructive"
                        onClick={() => handleDelete(project)}
                      >
                        <Trash2 className="w-4 h-4 mr-2" />
                        Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-4 text-sm text-muted-foreground">
                  <div className="flex items-center gap-1">
                    <Database className="w-4 h-4" />
                    <span>{project._count?.datasets || 0} datasets</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Cpu className="w-4 h-4" />
                    <span>{project._count?.models || 0} models</span>
                  </div>
                </div>
                <div className="flex items-center justify-between mt-4 pt-3 border-t">
                  <div className="flex items-center gap-2 flex-wrap">
                    <Badge variant={project.status === 'active' ? 'default' : 'secondary'}>
                      {project.status}
                    </Badge>
                    {isDetection(project.framework) && project.task === 'instance_segmentation' && (
                      <Badge variant="outline" className="border-purple-500/40 text-purple-600 dark:text-purple-400">
                        Instance Seg
                      </Badge>
                    )}
                  </div>
                  <div className="flex items-center gap-1 text-xs text-muted-foreground">
                    <Calendar className="w-3 h-3" />
                    {new Date(project.createdAt).toLocaleDateString()}
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Pagination — only meaningful in "All projects" mode */}
      {!loading && view === 'all' && total > 0 && pageCount > 1 && (
        <Pagination>
          <PaginationContent>
            <PaginationItem>
              <PaginationPrevious
                aria-disabled={page <= 1}
                className={page <= 1 ? 'pointer-events-none opacity-50' : 'cursor-pointer'}
                onClick={(e) => { e.preventDefault(); if (page > 1) setPage(page - 1) }}
              />
            </PaginationItem>
            {Array.from({ length: pageCount }, (_, i) => i + 1)
              .filter((n) => n === 1 || n === pageCount || Math.abs(n - page) <= 1 || (page <= 3 && n <= 4) || (page >= pageCount - 2 && n >= pageCount - 3))
              .reduce<(number | 'ellipsis')[]>((acc, n) => {
                const prev = acc[acc.length - 1]
                if (typeof prev === 'number' && n - prev > 1) acc.push('ellipsis')
                acc.push(n)
                return acc
              }, [])
              .map((item, i) => (
                <PaginationItem key={`${item}-${i}`}>
                  {item === 'ellipsis' ? (
                    <PaginationLink className="pointer-events-none">…</PaginationLink>
                  ) : (
                    <PaginationLink
                      href="#"
                      isActive={item === page}
                      onClick={(e) => { e.preventDefault(); setPage(item) }}
                    >
                      {item}
                    </PaginationLink>
                  )}
                </PaginationItem>
              ))}
            <PaginationItem>
              <PaginationNext
                aria-disabled={page >= pageCount}
                className={page >= pageCount ? 'pointer-events-none opacity-50' : 'cursor-pointer'}
                onClick={(e) => { e.preventDefault(); if (page < pageCount) setPage(page + 1) }}
              />
            </PaginationItem>
          </PaginationContent>
        </Pagination>
      )}
    </div>
  )
}
