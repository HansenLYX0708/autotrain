'use client'

import { useState, useSyncExternalStore } from 'react'
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
  Bell,
  Search,
  Bot,
  ChevronLeft,
  ChevronRight,
  PencilRuler,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
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
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { useTheme } from 'next-themes'
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

const navigation = [
  {
    id: 'dashboard',
    name: 'Dashboard',
    icon: LayoutDashboard,
  },
  {
    id: 'projects',
    name: 'Projects',
    icon: FolderKanban,
  },
  {
    id: 'datasets',
    name: 'Datasets',
    icon: Database,
  },
  {
    id: 'models',
    name: 'Models',
    icon: Cpu,
  },
  {
    id: 'training',
    name: 'Configurations',
    icon: PlayCircle,
  },
  {
    id: 'jobs',
    name: 'Jobs',
    icon: ListTodo,
  },
  {
    id: 'monitoring',
    name: 'Monitoring',
    icon: Activity,
  },
  {
    id: 'validation',
    name: 'Validation',
    icon: CheckCircle2,
  },
  {
    id: 'annotation',
    name: 'Annotation',
    icon: PencilRuler,
  },
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
}

export default function Home() {
  const [currentPage, setCurrentPage] = useState('dashboard')
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const { setTheme, theme } = useTheme()
  
  // Use useSyncExternalStore to avoid setState in effect
  const mounted = useSyncExternalStore(
    () => () => {},
    () => true,
    () => false
  )

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
                <span className="font-semibold text-lg">HawkeyePlus</span>
              )}
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-2 py-4 space-y-1 overflow-y-auto">
            {navigation.map((item) => {
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
              {/* Search */}
              <div className="flex items-center gap-4 flex-1 max-w-md">
                <div className="relative w-full">

                </div>
              </div>

              {/* Right side */}
              <div className="flex items-center gap-3">
                {/* Status indicator */}
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20">
                  <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                  <span className="text-xs font-medium text-emerald-600 dark:text-emerald-400">
                    System Ready
                  </span>
                </div>


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
              </div>
            </div>
          </header>

          {/* Page Content */}
          <main className="flex-1 p-6 overflow-auto">
            {currentPage === 'settings' ? <SettingsPage /> : <PageComponent />}
          </main>

          {/* Footer */}
          <footer className="h-10 border-t border-border bg-card/50 flex items-center justify-between px-6 text-xs text-muted-foreground shrink-0">
            <div>HawkeyePlus v1.0.0</div>
            <div className="flex items-center gap-4">
              <span>PaddleDetection Ready</span>
              <span>•</span>
              <span>PaddleClas Ready</span>
            </div>
          </footer>
        </div>
      </div>
    </TooltipProvider>
  )
}
