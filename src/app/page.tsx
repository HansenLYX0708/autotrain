'use client'

import { useState, useSyncExternalStore, useEffect } from 'react'
import { cn } from '@/lib/utils'
import {
  LayoutDashboard,
  FolderKanban,
  Database,
  Cpu,
  PlayCircle,
  ListTodo,
  Activity,
  CheckCircle2,
  Settings,
  Moon,
  Sun,
  Bot,
  ChevronLeft,
  ChevronRight,
  PencilRuler,
  Users,
  User,
  LogOut,
  Lock,
  AlertCircle,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { useTheme } from 'next-themes'
import { useAuth } from '@/contexts/auth-context'
import { LoginPage } from '@/components/pages/login'
import { DashboardPage } from '@/components/pages/dashboard'
import { ProjectsPage } from '@/components/pages/projects'
import { DatasetsPage } from '@/components/pages/datasets'
import { ModelsPage } from '@/components/pages/models'
import { TrainingPage } from '@/components/pages/training'
import { JobsPage } from '@/components/pages/jobs'
import { MonitoringPage } from '@/components/pages/monitoring'
import { ValidationPage } from '@/components/pages/validation'
import { SettingsPage } from '@/components/pages/settings'
import { AnnotationPage } from '@/components/pages/annotation'
import { UserManagementPage } from '@/components/pages/users'
import { ChangePasswordDialog } from '@/components/change-password-dialog'
import { AuthProvider } from '@/contexts/auth-context'

// 基础导航（所有用户可见）
const navigation = [
  { id: 'dashboard', name: 'Dashboard', icon: LayoutDashboard },
  { id: 'projects', name: 'Projects', icon: FolderKanban },
  { id: 'datasets', name: 'Datasets', icon: Database },
  { id: 'models', name: 'Models', icon: Cpu },
  { id: 'training', name: 'Configurations', icon: PlayCircle },
  { id: 'jobs', name: 'Jobs', icon: ListTodo },
  { id: 'monitoring', name: 'Monitoring', icon: Activity },
  { id: 'validation', name: 'Validation', icon: CheckCircle2 },
]

// 管理员专属导航
const adminNavigation = [
  { id: 'annotation', name: 'Annotation', icon: PencilRuler },
  { id: 'users', name: 'User Management', icon: Users },
]

const pageComponents: Record<string, React.ComponentType> = {
  dashboard: DashboardPage,
  projects: ProjectsPage,
  datasets: DatasetsPage,
  models: ModelsPage,
  training: TrainingPage,
  jobs: JobsPage,
  monitoring: MonitoringPage,
  validation: ValidationPage,
  annotation: AnnotationPage,
  settings: SettingsPage,
  users: UserManagementPage,
}

function AppContent() {
  const [currentPage, setCurrentPage] = useState('dashboard')
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [changePasswordOpen, setChangePasswordOpen] = useState(false)
  const { setTheme, theme } = useTheme()
  const { user, isLoading, isAuthenticated, logout } = useAuth()
  
  // System status states
  const [systemStatus, setSystemStatus] = useState<'ready' | 'occupied' | 'busy' | 'error' | 'partial'>('ready')
  const [runningJobs, setRunningJobs] = useState(0)
  const [gpuInfo, setGpuInfo] = useState<{ memoryUsed: number; memoryTotal: number; utilization: number }[]>([])
  const [envCheck, setEnvCheck] = useState({ pythonValid: false, paddleValid: false })

  // Fetch system status data
  useEffect(() => {
    const fetchSystemStatus = async () => {
      try {
        // Fetch running jobs
        const jobsResponse = await fetch('/api/training-jobs?status=running')
        let jobsCount = 0
        if (jobsResponse.ok) {
          const jobsData = await jobsResponse.json()
          jobsCount = jobsData.data?.length || 0
        }
        setRunningJobs(jobsCount)

        // Fetch GPU info
        const gpuResponse = await fetch('/api/system/gpu')
        let currentGpuInfo: typeof gpuInfo = []
        if (gpuResponse.ok) {
          const gpuResult = await gpuResponse.json()
          currentGpuInfo = gpuResult.data?.gpus || []
          setGpuInfo(currentGpuInfo)
        }

        // Fetch environment check
        const envResponse = await fetch('/api/system/environment-check')
        let currentEnvCheck = { pythonValid: false, paddleValid: false }
        if (envResponse.ok) {
          const envData = await envResponse.json()
          if (envData.success && envData.data) {
            // New API: check validGpus count and paddleDetection validity
            currentEnvCheck = {
              pythonValid: (envData.data.validGpus || 0) > 0,
              paddleValid: envData.data.paddleDetection?.isValid || false,
            }
            setEnvCheck(currentEnvCheck)
          }
        }

        // Determine system status using current data
        const hasEnvError = !currentEnvCheck.pythonValid || !currentEnvCheck.paddleValid
        
        if (hasEnvError) {
          setSystemStatus('error')
        } else if (jobsCount > 0) {
          setSystemStatus('busy')
        } else {
          // Check GPU usage across all GPUs
          const hasGpu = currentGpuInfo.length > 0
          if (hasGpu) {
            // Count GPUs by status
            const gpuStats = currentGpuInfo.map(g => {
              const highMemory = g.memoryTotal > 0 && (g.memoryUsed / g.memoryTotal) >= 0.5
              const highUtil = g.utilization >= 30
              return { highMemory, highUtil }
            })
            
            const occupiedGpus = gpuStats.filter(g => g.highMemory || g.highUtil).length
            const idleGpus = gpuStats.length - occupiedGpus
            
            if (occupiedGpus === 0) {
              setSystemStatus('ready')
            } else if (idleGpus === 0) {
              setSystemStatus('occupied')
            } else {
              // Partial - some GPUs occupied, some idle
              setSystemStatus('partial')
            }
          } else {
            setSystemStatus('ready')
          }
        }
      } catch (error) {
        console.error('Failed to fetch system status:', error)
        setSystemStatus('error')
      }
    }

    fetchSystemStatus()
    const interval = setInterval(fetchSystemStatus, 30000)
    return () => clearInterval(interval)
  }, [])

  const mounted = useSyncExternalStore(
    () => () => {},
    () => true,
    () => false
  )

  // Show loading while checking auth
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary to-primary/80 flex items-center justify-center animate-pulse">
            <Bot className="w-5 h-5 text-primary-foreground" />
          </div>
          <span className="text-lg font-semibold">Loading...</span>
        </div>
      </div>
    )
  }

  // Show login page if not authenticated
  if (!isAuthenticated) {
    return <LoginPage />
  }

  const isAdmin = user?.role === 'admin'

  // 合并导航：管理员看到所有，普通用户只看到基础导航
  const allNavigation = isAdmin
    ? [...navigation, ...adminNavigation]
    : navigation

  const PageComponent = pageComponents[currentPage]

  return (
    <TooltipProvider delayDuration={0}>
      <div className="min-h-screen bg-background flex">
        {/* Sidebar */}
        <aside
          className={cn(
            'fixed left-0 top-0 z-40 h-screen bg-card border-r border-border transition-all duration-300 flex flex-col',
            sidebarCollapsed ? 'w-16' : 'w-64'
          )}
        >
          {/* Logo */}
          <div className="flex items-center h-16 px-4 border-b border-border">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center">
                <Bot className="w-5 h-5 text-white" />
              </div>
              {!sidebarCollapsed && (
                <span className="font-semibold text-lg">Hawkeye+</span>
              )}
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-2 py-4 space-y-1 overflow-y-auto">
            {allNavigation.map((item) => {
              const isActive = currentPage === item.id

              return (
                <Tooltip key={item.id}>
                  <TooltipTrigger asChild>
                    <button
                      onClick={() => setCurrentPage(item.id)}
                      className={cn(
                        'w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors',
                        isActive
                          ? 'bg-primary text-primary-foreground'
                          : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                      )}
                    >
                      <item.icon className="w-5 h-5 flex-shrink-0" />
                      {!sidebarCollapsed && <span>{item.name}</span>}
                    </button>
                  </TooltipTrigger>
                  {sidebarCollapsed && (
                    <TooltipContent side="right">
                      {item.name}
                    </TooltipContent>
                  )}
                </Tooltip>
              )
            })}
          </nav>

          {/* Settings & Collapse */}
          <div className="border-t border-border p-2 space-y-1">
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  onClick={() => setCurrentPage('settings')}
                  className={cn(
                    'w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-muted-foreground hover:bg-accent hover:text-accent-foreground transition-colors',
                    currentPage === 'settings' && 'bg-accent text-accent-foreground'
                  )}
                >
                  <Settings className="w-5 h-5 flex-shrink-0" />
                  {!sidebarCollapsed && <span>Settings</span>}
                </button>
              </TooltipTrigger>
              {sidebarCollapsed && (
                <TooltipContent side="right">
                  Settings
                </TooltipContent>
              )}
            </Tooltip>
            
            <Button
              variant="ghost"
              size="sm"
              className="w-full justify-start gap-3 px-3"
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            >
              {sidebarCollapsed ? (
                <ChevronRight className="w-5 h-5" />
              ) : (
                <>
                  <ChevronLeft className="w-5 h-5" />
                  <span>Collapse</span>
                </>
              )}
            </Button>
          </div>
        </aside>

        {/* Main Content */}
        <div
          className={cn(
            'flex-1 transition-all duration-300 flex flex-col min-h-screen',
            sidebarCollapsed ? 'ml-16' : 'ml-64'
          )}
        >
          {/* Header */}
          <header className="h-16 border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-30 shrink-0">
            <div className="flex items-center justify-between h-full px-6">
              {/* Page Title */}
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-semibold capitalize">
                  {allNavigation.find(n => n.id === currentPage)?.name || 'Settings'}
                </h1>
              </div>

              {/* Right side */}
              <div className="flex items-center gap-3">
                {/* User Role Badge */}
                {isAdmin && (
                  <Badge variant="default" className="bg-primary/20 text-primary border-primary/30">
                    Admin
                  </Badge>
                )}

                {/* Status indicator */}
                {(() => {
                  const statusConfig = {
                    ready: {
                      bg: 'bg-emerald-500/10',
                      border: 'border-emerald-500/20',
                      dot: 'bg-emerald-500',
                      text: 'text-emerald-600 dark:text-emerald-400',
                      label: 'System Ready',
                    },
                    occupied: {
                      bg: 'bg-orange-500/10',
                      border: 'border-orange-500/20',
                      dot: 'bg-orange-500',
                      text: 'text-orange-600 dark:text-orange-400',
                      label: 'GPU Occupied',
                    },
                    partial: {
                      bg: 'bg-yellow-500/10',
                      border: 'border-yellow-500/20',
                      dot: 'bg-yellow-500',
                      text: 'text-yellow-600 dark:text-yellow-400',
                      label: 'GPU Partial',
                    },
                    busy: {
                      bg: 'bg-blue-500/10',
                      border: 'border-blue-500/20',
                      dot: 'bg-blue-500',
                      text: 'text-blue-600 dark:text-blue-400',
                      label: `Training (${runningJobs})`,
                    },
                    error: {
                      bg: 'bg-red-500/10',
                      border: 'border-red-500/20',
                      dot: 'bg-red-500',
                      text: 'text-red-600 dark:text-red-400',
                      label: 'Env Error',
                    },
                  }
                  const config = statusConfig[systemStatus]
                  
                  return (
                    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${config.bg} border ${config.border}`}>
                      <div className={`w-2 h-2 rounded-full ${config.dot} animate-pulse`} />
                      <span className={`text-xs font-medium ${config.text}`}>
                        {config.label}
                      </span>
                    </div>
                  )
                })()}

                {/* Theme toggle */}
                {mounted && (
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="icon">
                        {theme === 'dark' ? (
                          <Moon className="w-5 h-5" />
                        ) : (
                          <Sun className="w-5 h-5" />
                        )}
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem onClick={() => setTheme('light')}>
                        Light
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => setTheme('dark')}>
                        Dark
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => setTheme('system')}>
                        System
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                )}

                {/* User Menu */}
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" size="icon" className="rounded-full">
                      <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                        <User className="w-4 h-4 text-primary" />
                      </div>
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-56">
                    <div className="px-3 py-2">
                      <p className="text-sm font-medium">{user?.username}</p>
                      <p className="text-xs text-muted-foreground capitalize">{user?.role}</p>
                    </div>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem onClick={() => setChangePasswordOpen(true)}>
                      <Lock className="mr-2 h-4 w-4" />
                      Change Password
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem onClick={logout} className="text-destructive focus:text-destructive">
                      <LogOut className="mr-2 h-4 w-4" />
                      Logout
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </div>
          </header>

          {/* Page Content */}
          <main className="flex-1 p-6 overflow-auto">
            {PageComponent && <PageComponent />}
          </main>

          {/* Footer */}
          <footer className="h-10 border-t border-border bg-card/50 flex items-center justify-between px-6 text-xs text-muted-foreground shrink-0">
            <div>HawkeyePlus v1.0.0</div>
            <div className="flex items-center gap-4">
              <span>AutoTrain Platform</span>
            </div>
          </footer>
        </div>
      </div>

      {/* Change Password Dialog */}
      <ChangePasswordDialog open={changePasswordOpen} onOpenChange={setChangePasswordOpen} />
    </TooltipProvider>
  )
}

export default function Home() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  )
}
