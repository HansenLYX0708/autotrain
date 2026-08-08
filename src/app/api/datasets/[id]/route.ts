import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { notFoundOrDenied, requireOwnedScope } from '@/lib/auth';

/**
 * Fields a client may change on a dataset.
 *
 * PATCH used to forward the whole request body straight into
 * `db.dataset.update({ data: body })`, which let a caller reassign `userId`
 * (stealing or dumping a row onto another account) or rewrite `id`.
 */
const MUTABLE_DATASET_FIELDS = [
  'name', 'description', 'format',
  'trainImagePath', 'trainAnnoPath', 'evalImagePath', 'evalAnnoPath',
  'datasetDir', 'numClasses', 'numAnnotations', 'numTrainImages',
  'numEvalImages', 'classStats', 'yamlConfig',
] as const;

function pickMutableFields(body: Record<string, unknown>): Record<string, unknown> {
  const data: Record<string, unknown> = {};
  for (const field of MUTABLE_DATASET_FIELDS) {
    if (body[field] !== undefined) data[field] = body[field];
  }
  return data;
}

/**
 * Authenticate and confirm the caller owns this dataset. These handlers
 * previously ran unauthenticated, so any client could read, rewrite, or delete
 * another user's dataset by guessing an id.
 */
async function requireDatasetAccess(request: NextRequest, id: string) {
  const scope = await requireOwnedScope(request, id);
  if (scope instanceof NextResponse) return { error: scope };

  const dataset = await db.dataset.findFirst({ where: scope.where });
  if (!dataset) return { error: notFoundOrDenied('Dataset') };

  return { scope, dataset };
}

// GET /api/datasets/[id] - Get a single dataset by ID
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    const access = await requireDatasetAccess(request, id);
    if (access.error) return access.error;

    const dataset = await db.dataset.findUnique({
      where: { id },
      include: {
        project: {
          select: {
            id: true,
            name: true,
            framework: true,
            status: true,
            description: true,
          },
        },
        trainingJobs: {
          select: {
            id: true,
            name: true,
            status: true,
            createdAt: true,
          },
          orderBy: {
            createdAt: 'desc',
          },
          take: 10,
        },
      },
    });

    if (!dataset) {
      return NextResponse.json(
        {
          success: false,
          error: 'Dataset not found',
          message: `Dataset with id ${id} does not exist`,
        },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      data: dataset,
    });
  } catch (error) {
    console.error('Error fetching dataset:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to fetch dataset',
        message: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}

// PUT /api/datasets/[id] - Update a dataset
export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    const access = await requireDatasetAccess(request, id);
    if (access.error) return access.error;
    const { scope, dataset: existingDataset } = access;

    const body = await request.json();

    // If projectId is being changed, verify the caller also owns the target
    // project — otherwise a dataset could be moved into someone else's project.
    if (body.projectId && body.projectId !== existingDataset.projectId) {
      const project = await db.project.findFirst({
        where:
          scope.role === 'admin'
            ? { id: body.projectId }
            : { id: body.projectId, OR: [{ userId: scope.userId }, { userId: null }] },
      });

      if (!project) {
        return NextResponse.json(
          {
            success: false,
            error: 'Project not found or access denied',
          },
          { status: 404 }
        );
      }
    }

    const updatedDataset = await db.dataset.update({
      where: { id },
      data: {
        name: body.name !== undefined ? body.name : undefined,
        description: body.description !== undefined ? body.description : undefined,
        projectId: body.projectId !== undefined ? body.projectId : undefined,
        format: body.format !== undefined ? body.format : undefined,
        trainImagePath: body.trainImagePath !== undefined ? body.trainImagePath : undefined,
        trainAnnoPath: body.trainAnnoPath !== undefined ? body.trainAnnoPath : undefined,
        evalImagePath: body.evalImagePath !== undefined ? body.evalImagePath : undefined,
        evalAnnoPath: body.evalAnnoPath !== undefined ? body.evalAnnoPath : undefined,
        datasetDir: body.datasetDir !== undefined ? body.datasetDir : undefined,
        numClasses: body.numClasses !== undefined ? body.numClasses : undefined,
        numAnnotations: body.numAnnotations !== undefined ? body.numAnnotations : undefined,
        numTrainImages: body.numTrainImages !== undefined ? body.numTrainImages : undefined,
        numEvalImages: body.numEvalImages !== undefined ? body.numEvalImages : undefined,
        classStats: body.classStats !== undefined ? body.classStats : undefined,
      },
      include: {
        project: {
          select: {
            id: true,
            name: true,
            framework: true,
            status: true,
          },
        },
      },
    });

    return NextResponse.json({
      success: true,
      data: updatedDataset,
      message: 'Dataset updated successfully',
    });
  } catch (error) {
    console.error('Error updating dataset:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to update dataset',
        message: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}

// DELETE /api/datasets/[id] - Delete a dataset
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    const access = await requireDatasetAccess(request, id);
    if (access.error) return access.error;

    const existingDataset = await db.dataset.findUnique({
      where: { id },
      include: {
        _count: {
          select: {
            trainingJobs: true,
          },
        },
      },
    });

    if (!existingDataset) return notFoundOrDenied('Dataset');

    // Deleting a dataset cascades to its training jobs. Make that explicit
    // instead of silently destroying training history, and never do it while a
    // job is still running.
    const runningJobs = await db.trainingJob.count({
      where: { datasetId: id, status: 'running' },
    });
    if (runningJobs > 0) {
      return NextResponse.json(
        {
          success: false,
          error: `Cannot delete: ${runningJobs} training job(s) using this dataset are still running. Stop them first.`,
        },
        { status: 409 }
      );
    }

    // Delete the dataset (cascade will handle related training jobs)
    await db.dataset.delete({
      where: { id },
    });

    return NextResponse.json({
      success: true,
      message: 'Dataset deleted successfully',
      data: {
        id,
        deletedTrainingJobs: existingDataset._count.trainingJobs,
      },
    });
  } catch (error) {
    console.error('Error deleting dataset:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to delete dataset',
        message: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}

// PATCH /api/datasets/[id] - Partial update of a dataset
export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    const access = await requireDatasetAccess(request, id);
    if (access.error) return access.error;

    const body = await request.json();
    const data = pickMutableFields(body);

    if (Object.keys(data).length === 0) {
      return NextResponse.json(
        { success: false, error: 'No updatable fields provided' },
        { status: 400 }
      );
    }

    const updatedDataset = await db.dataset.update({
      where: { id },
      data,
      include: {
        project: {
          select: {
            id: true,
            name: true,
            framework: true,
            status: true,
          },
        },
      },
    });

    return NextResponse.json({
      success: true,
      data: updatedDataset,
      message: 'Dataset updated successfully',
    });
  } catch (error) {
    console.error('Error patching dataset:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to update dataset',
        message: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}
