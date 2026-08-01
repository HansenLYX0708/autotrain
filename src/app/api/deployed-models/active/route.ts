import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { requireAuth, buildUserFilter } from '@/lib/auth';

// GET /api/deployed-models/active?product=X&framework=Y
// Returns the single active DeployedModel for the given product (+ optional
// framework) in the caller's scope. Handy for external inference clients:
//     curl .../api/deployed-models/active?product=line13 -> { configPath, weightsPath, ... }
export async function GET(request: NextRequest) {
  try {
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;
    const { userId, role } = auth;

    const { searchParams } = new URL(request.url);
    const product = searchParams.get('product');
    const framework = searchParams.get('framework') || undefined;
    if (!product) {
      return NextResponse.json({ success: false, error: 'product is required' }, { status: 400 });
    }

    const where: Record<string, unknown> = {
      ...buildUserFilter(userId, role, 'userId'),
      product,
      isActive: true,
    };
    if (framework) where.framework = framework;

    const row = await db.deployedModel.findFirst({
      where,
      include: {
        project: { select: { id: true, name: true, framework: true } },
        trainingJob: { select: { id: true, name: true, status: true } },
      },
      orderBy: { updatedAt: 'desc' },
    });

    if (!row) return NextResponse.json({ success: false, error: 'No active model for product' }, { status: 404 });
    return NextResponse.json({ success: true, data: row });
  } catch (err) {
    console.error('deployed-models active GET failed:', err);
    return NextResponse.json({ success: false, error: 'Failed to load active model' }, { status: 500 });
  }
}
