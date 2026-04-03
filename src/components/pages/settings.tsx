'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { 
  Settings as SettingsIcon, 
  Save, 
  RefreshCw,
  FolderOpen,
  Terminal,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Cpu,
  Shield,
  Plus,
  Trash2,
  X
} from 'lucide-react'
import { useAuth } from '@/contexts/auth-context'
import { toast } from '@/hooks/use-toast'

interface GpuPythonMapping {
  gpuId: number
  pythonPath: string
}

interface SystemConfig {
  pythonPath: string
  condaEnv: string
  condaPath: string
  pythonEnvsBasePath: string
  gpuPythonMappings: GpuPythonMapping[]
  paddleDetectionPath: string
  paddleClasPath: string
  defaultGpu: number
  defaultFramework: string
}

export function SettingsPage() {
  const { user, isAuthenticated, isLoading: authLoading } = useAuth()
  const router = useRouter()
  const [config, setConfig] = useState<SystemConfig>({
    pythonPath: '',
    condaEnv: '',
    condaPath: '',
    pythonEnvsBasePath: '',
    gpuPythonMappings: [],
    paddleDetectionPath: '',
    paddleClasPath: '',
    defaultGpu: 0,
    defaultFramework: 'PaddleDetection',
  })
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [saveStatus, setSaveStatus] = useState<'idle' | 'success' | 'error'>('idle')
  const [availableGpus, setAvailableGpus] = useState<number[]>([0, 1, 2, 3])

  // Check if user is admin
  useEffect(() => {
    if (!authLoading && isAuthenticated && user?.role !== 'admin') {
      toast({
        title: 'Access Denied',
        description: 'Only administrators can access the settings page.',
        variant: 'destructive',
      })
      router.push('/')
    }
  }, [authLoading, isAuthenticated, user, router, toast])

  // Fetch GPU info to show available GPUs
  useEffect(() => {
    const fetchGpuInfo = async () => {
      try {
        const response = await fetch('/api/system/gpu')
        if (response.ok) {
          const result = await response.json()
          const gpus = result.data?.gpus || []
          if (Array.isArray(gpus) && gpus.length > 0) {
            setAvailableGpus(gpus.map((g: {id: number}) => g.id).sort((a: number, b: number) => a - b))
          }
        }
      } catch (error) {
        console.error('Failed to fetch GPU info:', error)
      }
    }
    fetchGpuInfo()
  }, [])

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const response = await fetch('/api/settings')
        if (response.ok) {
          const result = await response.json()
          const data = result.data || result
          if (data) {
            // Parse gpuPythonMappings from JSON string
            let mappings: GpuPythonMapping[] = []
            if (data.gpuPythonMappings) {
              try {
                const parsed = JSON.parse(data.gpuPythonMappings)
                mappings = Object.entries(parsed).map(([gpuId, path]) => ({
                  gpuId: parseInt(gpuId, 10),
                  pythonPath: path as string
                })).sort((a, b) => a.gpuId - b.gpuId)
              } catch {
                mappings = []
              }
            }
            
            setConfig({
              pythonPath: data.pythonPath || '',
              condaEnv: data.condaEnv || '',
              condaPath: data.condaPath || '',
              pythonEnvsBasePath: data.pythonEnvsBasePath || '',
              gpuPythonMappings: mappings,
              paddleDetectionPath: data.paddleDetectionPath || '',
              paddleClasPath: data.paddleClasPath || '',
              defaultGpu: data.defaultGpu ?? 0,
              defaultFramework: data.defaultFramework || 'PaddleDetection',
            })
          }
        } else if (response.status === 403) {
          toast({
            title: 'Access Denied',
            description: 'Only administrators can access settings.',
            variant: 'destructive',
          })
          router.push('/')
        }
      } catch (error) {
        console.error('Failed to fetch config:', error)
      } finally {
        setLoading(false)
      }
    }

    if (user?.role === 'admin') {
      fetchConfig()
    }
  }, [user, router, toast])

  const handleSave = async () => {
    setSaving(true)
    setSaveStatus('idle')
    
    try {
      // Convert gpuPythonMappings to JSON string
      const mappingsObject: Record<string, string> = {}
      config.gpuPythonMappings.forEach(mapping => {
        if (mapping.pythonPath.trim()) {
          mappingsObject[mapping.gpuId.toString()] = mapping.pythonPath.trim()
        }
      })
      
      const payload = {
        ...config,
        gpuPythonMappings: JSON.stringify(mappingsObject)
      }
      
      const response = await fetch('/api/settings', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      })
      
      if (response.ok) {
        setSaveStatus('success')
        setTimeout(() => setSaveStatus('idle'), 3000)
      } else if (response.status === 403) {
        toast({
          title: 'Access Denied',
          description: 'Only administrators can modify settings.',
          variant: 'destructive',
        })
        setSaveStatus('error')
      } else {
        setSaveStatus('error')
      }
    } catch (error) {
      console.error('Failed to save config:', error)
      setSaveStatus('error')
    } finally {
      setSaving(false)
    }
  }

  const addGpuMapping = () => {
    // Find next available GPU ID
    const usedGpuIds = config.gpuPythonMappings.map(m => m.gpuId)
    const nextGpuId = availableGpus.find(id => !usedGpuIds.includes(id))
    
    if (nextGpuId === undefined) {
      toast({
        title: 'No available GPU',
        description: 'All detected GPUs already have mappings configured.',
        variant: 'destructive',
      })
      return
    }
    
    setConfig(prev => ({
      ...prev,
      gpuPythonMappings: [...prev.gpuPythonMappings, { gpuId: nextGpuId, pythonPath: '' }]
        .sort((a, b) => a.gpuId - b.gpuId)
    }))
  }

  const removeGpuMapping = (gpuId: number) => {
    setConfig(prev => ({
      ...prev,
      gpuPythonMappings: prev.gpuPythonMappings.filter(m => m.gpuId !== gpuId)
    }))
  }

  const updateGpuMapping = (gpuId: number, pythonPath: string) => {
    setConfig(prev => ({
      ...prev,
      gpuPythonMappings: prev.gpuPythonMappings.map(m => 
        m.gpuId === gpuId ? { ...m, pythonPath } : m
      )
    }))
  }

  // Show loading while checking auth
  if (authLoading || loading) {
    return (
      <div className="flex items-center justify-center h-[400px]">
        <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  // Show access denied for non-admin users
  if (user?.role !== 'admin') {
    return (
      <div className="flex flex-col items-center justify-center h-[400px] space-y-4">
        <Shield className="w-16 h-16 text-muted-foreground" />
        <div className="text-center">
          <h2 className="text-xl font-semibold">Access Denied</h2>
          <p className="text-muted-foreground">
            Only administrators can access the settings page.
          </p>
        </div>
        <Button onClick={() => router.push('/')}>Go to Dashboard</Button>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Settings</h1>
          <p className="text-muted-foreground">
            Configure your training environment (Admin Only)
          </p>
        </div>
        <Button onClick={handleSave} disabled={saving}>
          {saving ? (
            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
          ) : (
            <Save className="w-4 h-4 mr-2" />
          )}
          Save Changes
        </Button>
      </div>

      {/* Save Status */}
      {saveStatus === 'success' && (
        <div className="flex items-center gap-2 text-sm text-emerald-600 bg-emerald-500/10 border border-emerald-500/20 rounded-lg p-3">
          <CheckCircle2 className="w-4 h-4" />
          Settings saved successfully
        </div>
      )}
      {saveStatus === 'error' && (
        <div className="flex items-center gap-2 text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-lg p-3">
          <AlertCircle className="w-4 h-4" />
          Failed to save settings
        </div>
      )}

      {/* Environment Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Terminal className="w-5 h-5" />
            Environment Configuration
          </CardTitle>
          <CardDescription>
            Configure Python environment and GPU settings
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Python/Conda Configuration */}
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="pythonPath">Python Path</Label>
              <div className="flex gap-2">
                <Input
                  id="pythonPath"
                  value={config.pythonPath}
                  onChange={(e) => setConfig({ ...config, pythonPath: e.target.value })}
                  placeholder="python or C:\\Python39\\python.exe"
                />
                <Button variant="outline" size="icon">
                  <FolderOpen className="w-4 h-4" />
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                Path to Python executable. Leave empty if using conda.
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="defaultGpu">Default GPU</Label>
              <Input
                id="defaultGpu"
                type="number"
                value={config.defaultGpu}
                onChange={(e) => setConfig({ ...config, defaultGpu: parseInt(e.target.value) || 0 })}
                min={0}
              />
              <p className="text-xs text-muted-foreground">
                GPU ID to use by default
              </p>
            </div>
          </div>

          {/* Python Environments Base Path */}
          <div className="border-t pt-4">
            <div className="flex items-center gap-2 mb-4">
              <Badge variant="secondary">GPU Python Environments</Badge>
            </div>
            
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="pythonEnvsBasePath">Python Environments Base Path</Label>
                <div className="flex gap-2">
                  <Input
                    id="pythonEnvsBasePath"
                    value={config.pythonEnvsBasePath}
                    onChange={(e) => setConfig({ ...config, pythonEnvsBasePath: e.target.value })}
                    placeholder="C:\\envs or /opt/envs"
                  />
                  <Button variant="outline" size="icon">
                    <FolderOpen className="w-4 h-4" />
                  </Button>
                </div>
                <p className="text-xs text-muted-foreground">
                  Base directory where multiple Python environments are stored. Each GPU can have its own Python environment.
                </p>
              </div>

              {/* GPU Python Mappings */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label className="text-sm">GPU-Specific Python Paths</Label>
                  <Button type="button" variant="outline" size="sm" onClick={addGpuMapping}>
                    <Plus className="w-4 h-4 mr-1" />
                    Add GPU Mapping
                  </Button>
                </div>
                
                {config.gpuPythonMappings.length === 0 ? (
                  <p className="text-sm text-muted-foreground italic">
                    No GPU-specific Python mappings configured. Click "Add GPU Mapping" to configure Python paths for specific GPUs.
                  </p>
                ) : (
                  <div className="space-y-2">
                    {config.gpuPythonMappings.map((mapping) => (
                      <div key={mapping.gpuId} className="flex items-center gap-2 p-2 rounded-lg border bg-muted/30">
                        <div className="flex items-center gap-2 min-w-[80px]">
                          <Cpu className="w-4 h-4 text-muted-foreground" />
                          <span className="font-medium text-sm">GPU {mapping.gpuId}</span>
                        </div>
                        <Input
                          value={mapping.pythonPath}
                          onChange={(e) => updateGpuMapping(mapping.gpuId, e.target.value)}
                          placeholder={`${config.pythonEnvsBasePath || 'C:\\envs'}\\gpu${mapping.gpuId}\\python.exe`}
                          className="flex-1"
                        />
                        <Button 
                          variant="ghost" 
                          size="icon" 
                          onClick={() => removeGpuMapping(mapping.gpuId)}
                          className="text-destructive hover:text-destructive"
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                    ))}
                  </div>
                )}
                
                <p className="text-xs text-muted-foreground">
                  Configure Python executable paths for each GPU. When training starts, the system will use the Python path corresponding to the selected GPU(s).
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Framework Paths */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FolderOpen className="w-5 h-5" />
            Framework Paths
          </CardTitle>
          <CardDescription>
            Configure paths to PaddleDetection and PaddleClas repositories
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="paddleDetectionPath">PaddleDetection Path</Label>
              <div className="flex gap-2">
                <Input
                  id="paddleDetectionPath"
                  value={config.paddleDetectionPath}
                  onChange={(e) => setConfig({ ...config, paddleDetectionPath: e.target.value })}
                  placeholder="C:\\workspace\\PaddleDetection"
                />
                <Button variant="outline" size="icon">
                  <FolderOpen className="w-4 h-4" />
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                Path to PaddleDetection repository
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Default Framework */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <SettingsIcon className="w-5 h-5" />
            Default Settings
          </CardTitle>
          <CardDescription>
            Configure default framework and preferences
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Default Framework</Label>
              <div className="flex gap-3">
                <Button
                  variant={config.defaultFramework === 'PaddleDetection' ? 'default' : 'outline'}
                  onClick={() => setConfig({ ...config, defaultFramework: 'PaddleDetection' })}
                >
                  PaddleDetection
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                Framework to use by default for new projects
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* System Status */}
      <Card>
        <CardHeader>
          <CardTitle>System Status</CardTitle>
          <CardDescription>
            Current system health and availability
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="flex items-center gap-3 p-3 rounded-lg border bg-muted/50">
              <div className="w-2 h-2 rounded-full bg-emerald-500" />
              <div>
                <div className="font-medium text-sm">Python</div>
                <div className="text-xs text-muted-foreground">Ready</div>
              </div>
            </div>
            <div className="flex items-center gap-3 p-3 rounded-lg border bg-muted/50">
              <div className="w-2 h-2 rounded-full bg-emerald-500" />
              <div>
                <div className="font-medium text-sm">PaddleDetection</div>
                <div className="text-xs text-muted-foreground">Ready</div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
