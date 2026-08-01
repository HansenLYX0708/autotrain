import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { requireAuth } from '@/lib/auth';

async function loadOwned(id: string, userId: string, role: string) {
  if (role === 'admin') {
    return db.deployedModel.findUnique({ where: { id } });
  }
  return db.deployedModel.findFirst({ where: { id, userId } });
}

// GET /api/deployed-models/:id
export async function GET(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  try {
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;
    const { userId, role } = auth;
    const { id } = await params;

    const row = await loadOwned(id, userId, role);
    if (!row) return NextResponse.json({ success: false, error: 'Not found' }, { status: 404 });

    // Enrich with relations
    const full = await db.deployedModel.findUnique({
      where: { id },
      include: {
        project: { select: { id: true, name: true, framework: true } },
        trainingJob: { select: { id: true, name: true, status: true } },
      },
    });
    return NextResponse.json({ success: true, data: full });
  } catch (err) {
    console.error('deployed-models GET item failed:', err);
    return NextResponse.json({ success: false, error: 'Failed to load' }, { status: 500 });
  }
}

// PATCH /api/deployed-models/:id
// Body: subset of { name, product, notes, exportedDir, metrics, architecture, configPath, weightsPath }
export async function PATCH(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  try {
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;
    const { userId, role } = auth;
    const { id } = await params;

    const existing = await loadOwned(id, userId, role);
    if (!existing) return NextResponse.json({ success: false, error: 'Not found' }, { status: 404 });

    const body = await request.json();
    const patch: Record<string, unknown> = {};
    const allowed = ['name', 'product', 'notes', 'exportedDir', 'architecture', 'configPath', 'weightsPath'] as const;
    for (const k of allowed) {
      if (typeof body[k] === 'string') patch[k] = body[k];
    }
    if (body.metrics != null) {
      patch.metrics = typeof body.metrics === 'string' ? body.metrics : JSON.stringify(body.metrics);
    }

    const updated = await db.deployedModel.update({ where: { id }, data: patch });
    return NextResponse.json({ success: true, data: updated });
  } catch (err) {
    console.error('deployed-models PATCH failed:', err);
    return NextResponse.json({ success: false, error: 'Failed to update' }, { status: 500 });
  }
}

// DELETE /api/deployed-models/:id
export async function DELETE(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  try {
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;
    const { userId, role } = auth;
    const { id } = await params;

    const existing = await loadOwned(id, userId, role);
    if (!existing) return NextResponse.json({ success: false, error: 'Not found' }, { status: 404 });

    await db.deployedModel.delete({ where: { id } });
    return NextResponse.json({ success: true });
  } catch (err) {
    console.error('deployed-models DELETE failed:', err);
    return NextResponse.json({ success: false, error: 'Failed to delete' }, { status: 500 });
  }
}
