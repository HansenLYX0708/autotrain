import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { requireAuth, buildUserFilter } from '@/lib/auth';
import * as fs from 'fs';

// GET /api/deployed-models
//   ?product=X&framework=Y&projectId=Z&activeOnly=true&includeInactive=true
// Lists deployed model registry entries visible to the caller.
export async function GET(request: NextRequest) {
  try {
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;
    const { userId, role } = auth;

    const { searchParams } = new URL(request.url);
    const product = searchParams.get('product');
    const framework = searchParams.get('framework');
    const projectId = searchParams.get('projectId');
    const activeOnly = searchParams.get('activeOnly') === 'true';

    const where: Record<string, unknown> = { ...buildUserFilter(userId, role, 'userId') };
    if (product) where.product = product;
    if (framework) where.framework = framework;
    if (projectId) where.projectId = projectId;
    if (activeOnly) where.isActive = true;

    const rows = await db.deployedModel.findMany({
      where,
      include: {
        project: { select: { id: true, name: true, framework: true } },
        trainingJob: { select: { id: true, name: true, status: true } },
      },
      orderBy: [{ product: 'asc' }, { isActive: 'desc' }, { updatedAt: 'desc' }],
    });

    // Distinct products list for the UI dropdown convenience
    const products = Array.from(new Set(rows.map(r => r.product))).sort();

    return NextResponse.json({ success: true, data: rows, products });
  } catch (err) {
    console.error('deployed-models GET failed:', err);
    return NextResponse.json({ success: false, error: 'Failed to list deployed models' }, { status: 500 });
  }
}

// POST /api/deployed-models
// Body: {
//   name, product, framework,
//   configPath, weightsPath,          // required
//   architecture?, exportedDir?, notes?,
//   trainingJobId?, projectId?,
//   metrics?: object|string,
//   activate?: boolean                // if true, also deactivate siblings
// }
export async function POST(request: NextRequest) {
  try {
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;
    const { userId, role } = auth;

    const body = await request.json();
    const {
      name, product, framework,
      configPath, weightsPath,
      architecture, exportedDir, notes,
      trainingJobId, projectId,
      metrics, activate,
    } = body ?? {};

    // Basic validation
    for (const [k, v] of Object.entries({ name, product, framework, configPath, weightsPath })) {
      if (!v || typeof v !== 'string') {
        return NextResponse.json({ success: false, error: `Missing or invalid field: ${k}` }, { status: 400 });
      }
    }

    // Path sanity: we allow non-existent paths (user may register before export
    // completes on a networked FS), but warn via response field.
    const warnings: string[] = [];
    if (!fs.existsSync(configPath)) warnings.push(`configPath not found on disk: ${configPath}`);
    if (!fs.existsSync(weightsPath)) warnings.push(`weightsPath not found on disk: ${weightsPath}`);
    if (exportedDir && !fs.existsSync(exportedDir)) warnings.push(`exportedDir not found on disk: ${exportedDir}`);

    // Access check on referenced project / training job (user must own them,
    // admins bypass). Silently drops IDs that don't pass the check so the
    // record can still be created as a standalone entry.
    let safeProjectId: string | null = null;
    if (projectId) {
      const p = role === 'admin'
        ? await db.project.findUnique({ where: { id: projectId }, select: { id: true } })
        : await db.project.findFirst({ where: { id: projectId, userId }, select: { id: true } });
      if (p) safeProjectId = p.id;
    }
    let safeJobId: string | null = null;
    if (trainingJobId) {
      const j = role === 'admin'
        ? await db.trainingJob.findUnique({ where: { id: trainingJobId }, select: { id: true } })
        : await db.trainingJob.findFirst({ where: { id: trainingJobId, userId }, select: { id: true } });
      if (j) safeJobId = j.id;
    }

    const metricsStr = typeof metrics === 'string'
      ? metrics
      : (metrics != null ? JSON.stringify(metrics) : null);

    // Create + optional activate atomically.
    const created = await db.$transaction(async (tx) => {
      const row = await tx.deployedModel.create({
        data: {
          name, product, framework,
          architecture: architecture || null,
          configPath, weightsPath,
          exportedDir: exportedDir || null,
          notes: notes || null,
          metrics: metricsStr,
          projectId: safeProjectId,
          trainingJobId: safeJobId,
          userId,
          isActive: !!activate,
        },
      });
      if (activate) {
        await tx.deployedModel.updateMany({
          where: {
            id: { not: row.id },
            product,
            framework,
            userId,
          },
          data: { isActive: false },
        });
      }
      return row;
    });

    return NextResponse.json({ success: true, data: created, warnings }, { status: 201 });
  } catch (err: unknown) {
    // Unique constraint (userId, product, name)
    if (err && typeof err === 'object' && 'code' in err && (err as { code: string }).code === 'P2002') {
      return NextResponse.json(
        { success: false, error: 'A deployed model with this (product, name) already exists for you.' },
        { status: 409 },
      );
    }
    console.error('deployed-models POST failed:', err);
    return NextResponse.json({ success: false, error: 'Failed to register deployed model' }, { status: 500 });
  }
}
