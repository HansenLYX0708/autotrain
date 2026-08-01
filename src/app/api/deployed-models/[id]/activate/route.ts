import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { requireAuth } from '@/lib/auth';

// POST /api/deployed-models/:id/activate
// Marks the target as active and deactivates every other entry with the same
// (userId, product, framework) tuple, atomically. Admin activation still keys
// on the original owner's namespace so behaviour matches list-scoping.
export async function POST(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  try {
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;
    const { userId, role } = auth;
    const { id } = await params;

    const target = role === 'admin'
      ? await db.deployedModel.findUnique({ where: { id } })
      : await db.deployedModel.findFirst({ where: { id, userId } });
    if (!target) return NextResponse.json({ success: false, error: 'Not found' }, { status: 404 });

    const scopeUserId = target.userId; // may be null; treat as its own scope
    const updated = await db.$transaction(async (tx) => {
      await tx.deployedModel.updateMany({
        where: {
          id: { not: id },
          product: target.product,
          framework: target.framework,
          userId: scopeUserId,
        },
        data: { isActive: false },
      });
      return tx.deployedModel.update({ where: { id }, data: { isActive: true } });
    });

    return NextResponse.json({ success: true, data: updated });
  } catch (err) {
    console.error('deployed-models activate failed:', err);
    return NextResponse.json({ success: false, error: 'Failed to activate' }, { status: 500 });
  }
}

// DELETE /api/deployed-models/:id/activate - clear active flag (no-op if already off)
export async function DELETE(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  try {
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;
    const { userId, role } = auth;
    const { id } = await params;

    const target = role === 'admin'
      ? await db.deployedModel.findUnique({ where: { id } })
      : await db.deployedModel.findFirst({ where: { id, userId } });
    if (!target) return NextResponse.json({ success: false, error: 'Not found' }, { status: 404 });

    const updated = await db.deployedModel.update({ where: { id }, data: { isActive: false } });
    return NextResponse.json({ success: true, data: updated });
  } catch (err) {
    console.error('deployed-models deactivate failed:', err);
    return NextResponse.json({ success: false, error: 'Failed to deactivate' }, { status: 500 });
  }
}
