'use client'

import { useEffect, useState } from 'react'
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
  Loader2
} from 'lucide-react'

interface SystemConfig {
  pythonPath: string
  paddleDetectionPath: string
  paddleClasPath: string
  defaultGpu: number
  defaultFramework: string
}

export function SettingsPage() {
  const [config, setConfig] = useState<SystemConfig>({
    pythonPath: '',
    paddleDetectionPath: '',
    paddleClasPath: '',
    defaultGpu: 0,
    defaultFramework: 'PaddleDetection',
  })
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [saveStatus, setSaveStatus] = useState<'idle' | 'success' | 'error'>('idle')

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const response = await fetch('/api/settings')
        if (response.ok) {
          const result = await response.json()
          const data = result.data || result
          if (data) {
            setConfig({
              pythonPath: data.pythonPath || '',
              paddleDetectionPath: data.paddleDetectionPath || '',
              paddleClasPath: data.paddleClasPath || '',
              defaultGpu: data.defaultGpu ?? 0,
              defaultFramework: data.defaultFramework || 'PaddleDetection',
            })
          }
        }
      } catch (error) {
        console.error('Failed to fetch config:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchConfig()
  }, [])

  const handleSave = async () => {
    setSaving(true)
    setSaveStatus('idle')
    
    try {
      const response = await fetch('/api/settings', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      })
      
      if (response.ok) {
        setSaveStatus('success')
        setTimeout(() => setSaveStatus('idle'), 3000)
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

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[400px]">
        <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
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
            Configure your training environment
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
            Configure paths to Python and PaddlePaddle frameworks
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
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
                Path to Python executable (e.g., python, python3, or full path)
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

            <div className="space-y-2">
              <Label htmlFor="paddleClasPath">PaddleClas Path</Label>
              <div className="flex gap-2">
                <Input
                  id="paddleClasPath"
                  value={config.paddleClasPath}
                  onChange={(e) => setConfig({ ...config, paddleClasPath: e.target.value })}
                  placeholder="C:\\workspace\\PaddleClas"
                />
                <Button variant="outline" size="icon">
                  <FolderOpen className="w-4 h-4" />
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                Path to PaddleClas repository
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
                <Button
                  variant={config.defaultFramework === 'PaddleClas' ? 'default' : 'outline'}
                  onClick={() => setConfig({ ...config, defaultFramework: 'PaddleClas' })}
                >
                  PaddleClas
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
            <div className="flex items-center gap-3 p-3 rounded-lg border bg-muted/50">
              <div className="w-2 h-2 rounded-full bg-emerald-500" />
              <div>
                <div className="font-medium text-sm">PaddleClas</div>
                <div className="text-xs text-muted-foreground">Ready</div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
