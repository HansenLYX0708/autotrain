import { NextRequest, NextResponse } from 'next/server'
import { requireAdmin } from '@/lib/auth'
import { importLocalizationDataset, localizationImportSchema } from '@/lib/localization-import'
import { logActivity } from '@/lib/activity-log'

export const runtime = 'nodejs'
export const maxDuration = 600

export async function POST(request: NextRequest) {
  const auth = await requireAdmin(request)
  if (auth instanceof NextResponse) return auth
  try {
    const options = localizationImportSchema.parse(await request.json())
    const data = await importLocalizationDataset(auth.userId, options)
    if (!data.preview) await logActivity(auth.userId, {
      action: 'import_dataset', entityType: 'dataset',
      entityName: options.name, details: { task: options.task, sourceRevision: data.report.sourceRevision },
    })
    return NextResponse.json({ success: true, data })
  } catch (error) {
    return NextResponse.json({ success: false, error: error instanceof Error ? error.message : 'Localization import failed' }, { status: 400 })
  }
}
