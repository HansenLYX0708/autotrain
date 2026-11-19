import { NextRequest, NextResponse } from 'next/server'
import { z } from 'zod'
import { requireAdmin } from '@/lib/auth'
import { isInside } from '@/lib/safe-path'
import {
  deleteAnnotationSet, projectPath, projectsDir, renameAnnotationSet, setAnnotationSetStatus, validId,
} from '@/lib/annotation-store'

const patchSchema = z.object({
  // Same bounds as the shared projectSchema name field.
  name: z.string().trim().min(1).max(100).optional(),
  status: z.enum(['active', 'archived']).optional(),
}).refine(v => v.name !== undefined || v.status !== undefined, 'Nothing to update')

/** Confine every operation to the caller's own annotation storage. */
function resolveFile(dir: string, id: string) {
  const file = projectPath(dir, id)
  return validId(id) && isInside(dir, file) ? file : null
}

async function guard(request: NextRequest, id: string) {
  const auth = await requireAdmin(request)
  if (auth instanceof NextResponse) return { error: auth }
  const dir = projectsDir(auth.userId)
  if (!resolveFile(dir, id)) {
    return { error: NextResponse.json({ success: false, error: 'Invalid project ID' }, { status: 400 }) }
  }
  return { dir }
}

// PATCH /api/annotation/projects/[id] - rename and/or change active/archived status
export async function PATCH(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params
  const access = await guard(request, id)
  if (access.error) return access.error
  let body: unknown
  try {
    body = await request.json()
  } catch {
    return NextResponse.json({ success: false, error: 'Invalid JSON' }, { status: 400 })
  }
  const parsed = patchSchema.safeParse(body)
  if (!parsed.success) return NextResponse.json({ success: false, error: parsed.error.issues[0]?.message }, { status: 400 })
  try {
    if (parsed.data.name !== undefined) await renameAnnotationSet(access.dir, id, parsed.data.name)
    const meta = parsed.data.status !== undefined
      ? await setAnnotationSetStatus(access.dir, id, parsed.data.status)
      : undefined
    return NextResponse.json({ success: true, data: { id, name: parsed.data.name, ...(meta ?? {}) } })
  } catch (error) {
    const missing = (error as NodeJS.ErrnoException).code === 'ENOENT'
    return NextResponse.json({ success: false, error: missing ? 'Project not found' : 'Could not update annotation project' }, { status: missing ? 404 : 500 })
  }
}

// DELETE /api/annotation/projects/[id] - remove the project JSON and its metadata sidecar
export async function DELETE(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params
  const access = await guard(request, id)
  if (access.error) return access.error
  try {
    await deleteAnnotationSet(access.dir, id)
    return NextResponse.json({ success: true, data: { id } })
  } catch (error) {
    const missing = (error as NodeJS.ErrnoException).code === 'ENOENT'
    return NextResponse.json({ success: false, error: missing ? 'Project not found' : 'Could not delete annotation project' }, { status: missing ? 404 : 500 })
  }
}

