import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';

// GET /api/validation-jobs/[id] - Get a single validation job by ID
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    const validationJob = await db.validationJob.findUnique({
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
      },
    });

    if (!validationJob) {
      return NextResponse.json(
        { success: false, error: 'Validation job not found' },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      data: validationJob,
    });
  } catch (error) {
    console.error('Error fetching validation job:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to fetch validation job' },
      { status: 500 }
    );
  }
}

// PUT /api/validation-jobs/[id] - Update a validation job
export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const body = await request.json();

    // Check if validation job exists
    const existingJob = await db.validationJob.findUnique({
      where: { id },
    });

    if (!existingJob) {
      return NextResponse.json(
        { success: false, error: 'Validation job not found' },
        { status: 404 }
      );
    }

    // Validate type if provided
    const validTypes = ['eval', 'infer'];
    if (body.type && !validTypes.includes(body.type)) {
      return NextResponse.json(
        { success: false, error: `Invalid type. Must be one of: ${validTypes.join(', ')}` },
        { status: 400 }
      );
    }

    // Validate status if provided
    const validStatuses = ['pending', 'running', 'completed', 'failed'];
    if (body.status && !validStatuses.includes(body.status)) {
      return NextResponse.json(
        { success: false, error: `Invalid status. Must be one of: ${validStatuses.join(', ')}` },
        { status: 400 }
      );
    }

    // Build update data
    const updateData: Record<string, unknown> = {};

    if (body.name !== undefined) updateData.name = body.name;
    if (body.type !== undefined) updateData.type = body.type;
    if (body.configPath !== undefined) updateData.configPath = body.configPath || null;
    if (body.weightsPath !== undefined) updateData.weightsPath = body.weightsPath || null;
    if (body.datasetPath !== undefined) updateData.datasetPath = body.datasetPath || null;
    if (body.inferInputPath !== undefined) updateData.inferInputPath = body.inferInputPath || null;
    if (body.inferOutputPath !== undefined) updateData.inferOutputPath = body.inferOutputPath || null;
    if (body.status !== undefined) updateData.status = body.status;
    if (body.command !== undefined) updateData.command = body.command || null;
    if (body.resultJson !== undefined) updateData.resultJson = body.resultJson || null;
    if (body.resultPath !== undefined) updateData.resultPath = body.resultPath || null;
    if (body.outputLog !== undefined) updateData.outputLog = body.outputLog || null;
    if (body.startedAt !== undefined) {
      updateData.startedAt = body.startedAt ? new Date(body.startedAt) : null;
    }
    if (body.completedAt !== undefined) {
      updateData.completedAt = body.completedAt ? new Date(body.completedAt) : null;
    }

    const updatedJob = await db.validationJob.update({
      where: { id },
      data: updateData,
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
      data: updatedJob,
    });
  } catch (error) {
    console.error('Error updating validation job:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to update validation job' },
      { status: 500 }
    );
  }
}

// DELETE /api/validation-jobs/[id] - Delete a validation job
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    // Check if validation job exists
    const existingJob = await db.validationJob.findUnique({
      where: { id },
    });

    if (!existingJob) {
      return NextResponse.json(
        { success: false, error: 'Validation job not found' },
        { status: 404 }
      );
    }

    await db.validationJob.delete({
      where: { id },
    });

    return NextResponse.json({
      success: true,
      message: 'Validation job deleted successfully',
    });
  } catch (error) {
    console.error('Error deleting validation job:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to delete validation job' },
      { status: 500 }
    );
  }
}
