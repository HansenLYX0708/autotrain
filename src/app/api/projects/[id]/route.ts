import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { notFoundOrDenied, requireOwnedScope } from '@/lib/auth';

// Helper function to validate ID
function isValidId(id: string): boolean {
  return typeof id === 'string' && id.length > 0;
}

/**
 * Authenticate and confirm the caller owns this project.
 *
 * All three handlers previously ran unauthenticated, so any client could read a
 * project's full contents, rename it, switch its framework, or cascade-delete
 * every dataset/model/job under it by guessing an id.
 */
async function requireProjectAccess(request: NextRequest, id: string) {
  const scope = await requireOwnedScope(request, id);
  if (scope instanceof NextResponse) return { error: scope };

  const project = await db.project.findFirst({ where: scope.where });
  if (!project) return { error: notFoundOrDenied('Project') };

  return { scope, project };
}

// GET /api/projects/[id] - Get a single project with related data
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    if (!isValidId(id)) {
      return NextResponse.json(
        {
          success: false,
          error: 'Invalid project ID',
        },
        { status: 400 }
      );
    }

    const access = await requireProjectAccess(request, id);
    if (access.error) return access.error;

    const project = await db.project.findUnique({
      where: {
        id,
      },
      include: {
        datasets: {
          orderBy: {
            createdAt: 'desc',
          },
        },
        models: {
          orderBy: {
            createdAt: 'desc',
          },
        },
        trainingJobs: {
          orderBy: {
            createdAt: 'desc',
          },
          include: {
            dataset: {
              select: {
                id: true,
                name: true,
              },
            },
            model: {
              select: {
                id: true,
                name: true,
                architecture: true,
              },
            },
          },
        },
        validationJobs: {
          orderBy: {
            createdAt: 'desc',
          },
        },
        _count: {
          select: {
            datasets: true,
            models: true,
            trainingJobs: true,
            validationJobs: true,
          },
        },
      },
    });

    if (!project) {
      return NextResponse.json(
        {
          success: false,
          error: 'Project not found',
        },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      data: project,
    });
  } catch (error) {
    console.error('Error fetching project:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to fetch project',
      },
      { status: 500 }
    );
  }
}

// PUT /api/projects/[id] - Update a project
export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    if (!isValidId(id)) {
      return NextResponse.json(
        {
          success: false,
          error: 'Invalid project ID',
        },
        { status: 400 }
      );
    }

    const access = await requireProjectAccess(request, id);
    if (access.error) return access.error;
    const existingProject = access.project;

    const body = await request.json();
    const { name, description, framework, task, status } = body;

    const allowedTasks = ['detection', 'instance_segmentation'];

    // If name is being updated, check for duplicates within the caller's own
    // projects. Checking globally leaked the existence of other users' project
    // names and blocked legitimate renames.
    if (name && name !== existingProject.name) {
      const duplicateName = await db.project.findFirst({
        where: {
          name: name.trim(),
          id: { not: id },
          userId: existingProject.userId,
        },
      });

      if (duplicateName) {
        return NextResponse.json(
          {
            success: false,
            error: 'A project with this name already exists',
          },
          { status: 409 }
        );
      }
    }

    // Build update data object
    const updateData: {
      name?: string;
      description?: string | null;
      framework?: string;
      task?: string;
      status?: string;
    } = {};

    if (name !== undefined) {
      updateData.name = name.trim();
    }
    if (description !== undefined) {
      updateData.description = description?.trim() || null;
    }
    if (framework !== undefined) {
      updateData.framework = framework;
    }
    // Resolve task based on the effective framework.
    // PaddleSeg -> semantic_segmentation; PaddleDetection -> detection|instance_segmentation; others -> detection.
    const effectiveFramework = framework !== undefined ? framework : existingProject.framework;
    if (effectiveFramework === 'PaddleSeg') {
      updateData.task = 'semantic_segmentation';
    } else if (effectiveFramework === 'PaddleDetection') {
      if (task !== undefined && allowedTasks.includes(task)) {
        updateData.task = task;
      } else if (framework !== undefined) {
        // Switching into PaddleDetection without a valid task -> keep existing valid task or default
        updateData.task = allowedTasks.includes(existingProject.task) ? existingProject.task : 'detection';
      }
    } else if (framework !== undefined) {
      // PaddleClas or any other framework
      updateData.task = 'detection';
    }
    if (status !== undefined) {
      updateData.status = status;
    }

    // Update the project
    const updatedProject = await db.project.update({
      where: { id },
      data: updateData,
      include: {
        _count: {
          select: {
            datasets: true,
            models: true,
            trainingJobs: true,
          },
        },
      },
    });

    return NextResponse.json({
      success: true,
      data: updatedProject,
      message: 'Project updated successfully',
    });
  } catch (error) {
    console.error('Error updating project:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to update project',
      },
      { status: 500 }
    );
  }
}

// DELETE /api/projects/[id] - Delete a project (cascade delete related data)
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    if (!isValidId(id)) {
      return NextResponse.json(
        {
          success: false,
          error: 'Invalid project ID',
        },
        { status: 400 }
      );
    }

    const access = await requireProjectAccess(request, id);
    if (access.error) return access.error;

    const existingProject = await db.project.findUnique({
      where: { id },
      include: {
        _count: {
          select: {
            datasets: true,
            models: true,
            trainingJobs: true,
            validationJobs: true,
          },
        },
      },
    });

    if (!existingProject) return notFoundOrDenied('Project');

    // Refuse while training is in flight: the cascade would drop the job row
    // while its OS process keeps running and writing to disk.
    const runningJobs = await db.trainingJob.count({
      where: { projectId: id, status: 'running' },
    });
    if (runningJobs > 0) {
      return NextResponse.json(
        {
          success: false,
          error: `Cannot delete: ${runningJobs} training job(s) are still running. Stop them first.`,
        },
        { status: 409 }
      );
    }

    // Delete the project (cascade delete is handled by Prisma schema)
    await db.project.delete({
      where: { id },
    });

    return NextResponse.json({
      success: true,
      message: 'Project deleted successfully',
      data: {
        deletedCounts: existingProject._count,
      },
    });
  } catch (error) {
    console.error('Error deleting project:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to delete project',
      },
      { status: 500 }
    );
  }
}
