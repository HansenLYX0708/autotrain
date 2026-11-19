import { NextRequest, NextResponse } from 'next/server'
import { requireAuth } from '@/lib/auth'
import { listServerDirectory } from '@/lib/server-browse'

export const runtime = 'nodejs'

export async function GET(request: NextRequest) {
  const auth = await requireAuth(request)
  if (auth instanceof NextResponse) return auth
  try {
    const params = request.nextUrl.searchParams
    const data = await listServerDirectory(params.get('path'), { includeImages: params.get('images') === '1' })
    return NextResponse.json({ success: true, data })
  } catch (error) {
    return NextResponse.json({ success: false, error: error instanceof Error ? error.message : 'Browse failed' }, { status: 400 })
  }
}
