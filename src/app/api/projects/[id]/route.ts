import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';

// Helper function to validate ID
function isValidId(id: string): boolean {
  return typeof id === 'string' && id.length > 0;
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

    const body = await request.json();
    const { name, description, framework, status } = body;

    // Check if project exists
    const existingProject = await db.project.findUnique({
      where: { id },
    });

    if (!existingProject) {
      return NextResponse.json(
        {
          success: false,
          error: 'Project not found',
        },
        { status: 404 }
      );
    }

    // If name is being updated, check for duplicates
    if (name && name !== existingProject.name) {
      const duplicateName = await db.project.findFirst({
        where: {
          name: name.trim(),
          id: { not: id },
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

    // Check if project exists
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

    if (!existingProject) {
      return NextResponse.json(
        {
          success: false,
          error: 'Project not found',
        },
        { status: 404 }
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
