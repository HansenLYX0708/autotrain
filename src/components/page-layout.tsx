'use client'

import { AppSidebar } from '@/components/app-sidebar'
import { Header } from '@/components/header'
import { cn } from '@/lib/utils'

interface PageLayoutProps {
  children: React.ReactNode
  className?: string
}

export function PageLayout({ children, className }: PageLayoutProps) {
  return (
    <div className="min-h-screen bg-background">
      <AppSidebar />
      <div className="ml-64 transition-all duration-300">
        <Header />
        <main className={cn('p-6', className)}>
          {children}
        </main>
      </div>
    </div>
  )
}
