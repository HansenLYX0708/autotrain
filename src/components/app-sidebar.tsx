'use client'

import { usePathname } from 'next/navigation'
import Link from 'next/link'
import { cn } from '@/lib/utils'
import {
  LayoutDashboard,
  FolderKanban,
  Database,
  Cpu,
  PlayCircle,
  Activity,
  CheckCircle2,
  Settings,
  ChevronLeft,
  ChevronRight,
  Bot
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useState } from 'react'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'

const navigation = [
  {
    name: 'Dashboard',
    href: '/',
    icon: LayoutDashboard,
  },
  {
    name: 'Projects',
    href: '/projects',
    icon: FolderKanban,
  },
  {
    name: 'Datasets',
    href: '/datasets',
    icon: Database,
  },
  {
    name: 'Models',
    href: '/models',
    icon: Cpu,
  },
  {
    name: 'Training',
    href: '/training',
    icon: PlayCircle,
  },
  {
    name: 'Monitoring',
    href: '/monitoring',
    icon: Activity,
  },
  {
    name: 'Validation',
    href: '/validation',
    icon: CheckCircle2,
  },
]

export function AppSidebar() {
  const pathname = usePathname()
  const [collapsed, setCollapsed] = useState(false)

  return (
    <TooltipProvider delayDuration={0}>
      <aside
        className={cn(
          'fixed left-0 top-0 z-40 h-screen bg-card border-r border-border transition-all duration-300',
          collapsed ? 'w-16' : 'w-64'
        )}
      >
        {/* Logo */}
        <div className="flex items-center h-16 px-4 border-b border-border">
          <Link href="/" className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center">
              <Bot className="w-5 h-5 text-white" />
            </div>
            {!collapsed && (
              <span className="font-semibold text-lg">HawkeyePlus</span>
            )}
          </Link>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-2 py-4 space-y-1 overflow-y-auto">
          {navigation.map((item) => {
            const isActive = item.href === '/' 
              ? pathname === '/' 
              : pathname.startsWith(item.href)
            
            return (
              <Tooltip key={item.name}>
                <TooltipTrigger asChild>
                  <Link
                    href={item.href}
                    className={cn(
                      'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors',
                      isActive
                        ? 'bg-primary text-primary-foreground'
                        : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                    )}
                  >
                    <item.icon className="w-5 h-5 flex-shrink-0" />
                    {!collapsed && <span>{item.name}</span>}
                  </Link>
                </TooltipTrigger>
                {collapsed && (
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
              <Link
                href="/settings"
                className={cn(
                  'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-muted-foreground hover:bg-accent hover:text-accent-foreground transition-colors'
                )}
              >
                <Settings className="w-5 h-5 flex-shrink-0" />
                {!collapsed && <span>Settings</span>}
              </Link>
            </TooltipTrigger>
            {collapsed && (
              <TooltipContent side="right">
                Settings
              </TooltipContent>
            )}
          </Tooltip>
          
          <Button
            variant="ghost"
            size="sm"
            className="w-full justify-start gap-3 px-3"
            onClick={() => setCollapsed(!collapsed)}
          >
            {collapsed ? (
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
    </TooltipProvider>
  )
}
