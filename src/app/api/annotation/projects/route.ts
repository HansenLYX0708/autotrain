import { NextRequest, NextResponse } from 'next/server'
import { mkdir } from 'fs/promises'
import { requireAdmin } from '@/lib/auth'
import { projectSchema } from '@/lib/localization-annotations'
import {
  listAnnotationSets, projectsDir, readAnnotationProject, saveAnnotationSet, validId,
} from '@/lib/annotation-store'

export async function GET(request: NextRequest) {
  const auth = await requireAdmin(request)
  if (auth instanceof NextResponse) return auth
  const root = projectsDir(auth.userId)
  const params = request.nextUrl.searchParams
  const id = params.get('id')
  if (id && !validId(id)) return NextResponse.json({ success: false, error: 'Invalid project ID' }, { status: 400 })
  try {
    if (id) {
      // Single-project fetch works for archived sets too (the editor can reopen them).
      const project = await readAnnotationProject(root, id)
      return NextResponse.json({ success: true, data: project })
    }
    const statusParam = params.get('status')
    const status = statusParam === 'archived' || statusParam === 'all' ? statusParam : 'active'
    const data = await listAnnotationSets(auth.userId, { status, q: params.get('q') ?? undefined })
    return NextResponse.json({ success: true, data })
  } catch (error) {
    const missing = (error as NodeJS.ErrnoException).code === 'ENOENT'
    return NextResponse.json({ success: false, error: missing ? 'Project not found' : 'Could not read annotation projects' }, { status: missing ? 404 : 500 })
  }
}

export async function POST(request: NextRequest) {
  const auth = await requireAdmin(request)
  if (auth instanceof NextResponse) return auth
  let raw: unknown
  try {
    const text = await request.text()
    if (Buffer.byteLength(text, 'utf8') > 20 * 1024 * 1024) return NextResponse.json({ success: false, error: 'Project exceeds 20 MB' }, { status: 413 })
    raw = JSON.parse(text)
  } catch {
    return NextResponse.json({ success: false, error: 'Invalid JSON' }, { status: 400 })
  }
  const parsed = projectSchema.safeParse(raw)
  if (!parsed.success) return NextResponse.json({ success: false, error: parsed.error.issues[0]?.message }, { status: 400 })
  const root = projectsDir(auth.userId)
  try {
    await mkdir(root, { recursive: true })
    // Every save marks the set active and restarts the auto-archive countdown.
    await saveAnnotationSet(root, parsed.data)
    return NextResponse.json({ success: true, data: { id: parsed.data.id } })
  } catch {
    return NextResponse.json({ success: false, error: 'Could not save annotation project' }, { status: 500 })
  }
}
