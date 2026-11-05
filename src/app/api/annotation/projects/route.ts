import { NextRequest, NextResponse } from 'next/server'
import { mkdir, readFile, readdir, rename, stat, unlink, writeFile } from 'fs/promises'
import path from 'path'
import { randomUUID } from 'crypto'
import { requireAdmin } from '@/lib/auth'
import { projectSchema } from '@/lib/localization-annotations'
import { toSafeSlug } from '@/lib/safe-path'

const storage = (userId: string) => path.join(process.cwd(), 'annotations', toSafeSlug(userId, 'user'), 'projects')
const validId = (id: string) => /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id)

export async function GET(request: NextRequest) {
  const auth = await requireAdmin(request)
  if (auth instanceof NextResponse) return auth
  const root = storage(auth.userId)
  const id = request.nextUrl.searchParams.get('id')
  if (id && !validId(id)) return NextResponse.json({ success: false, error: 'Invalid project ID' }, { status: 400 })
  try {
    if (id) {
      const project = projectSchema.parse(JSON.parse(await readFile(path.join(root, `${id}.json`), 'utf8')))
      return NextResponse.json({ success: true, data: project })
    }
    const files = await readdir(root).catch((error: NodeJS.ErrnoException) => {
      if (error.code === 'ENOENT') return []
      throw error
    })
    const projects = await Promise.all(files.filter(f => f.endsWith('.json') && validId(f.slice(0, -5))).map(async file => {
      const full = path.join(root, file)
      const project = projectSchema.parse(JSON.parse(await readFile(full, 'utf8')))
      return { id: project.id, name: project.name, images: project.images.length, modified: (await stat(full)).mtime.toISOString() }
    }))
    return NextResponse.json({ success: true, data: projects.sort((a, b) => b.modified.localeCompare(a.modified)) })
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
  const root = storage(auth.userId)
  const file = path.join(root, `${toSafeSlug(parsed.data.id, 'invalid')}.json`)
  const temporary = `${file}.${randomUUID()}.tmp`
  try {
    await mkdir(root, { recursive: true })
    await writeFile(temporary, JSON.stringify(parsed.data, null, 2), 'utf8')
    await rename(temporary, file)
    return NextResponse.json({ success: true, data: { id: parsed.data.id } })
  } catch {
    return NextResponse.json({ success: false, error: 'Could not save annotation project' }, { status: 500 })
  } finally {
    await unlink(temporary).catch(() => {})
  }
}
