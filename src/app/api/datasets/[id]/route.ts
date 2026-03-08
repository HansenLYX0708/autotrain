import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';

// GET /api/datasets/[id] - Get a single dataset by ID
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

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
    const body = await request.json();

    // Check if dataset exists
    const existingDataset = await db.dataset.findUnique({
      where: { id },
    });

    if (!existingDataset) {
      return NextResponse.json(
        {
          success: false,
          error: 'Dataset not found',
          message: `Dataset with id ${id} does not exist`,
        },
        { status: 404 }
      );
    }

    // If projectId is being changed, verify the new project exists
    if (body.projectId && body.projectId !== existingDataset.projectId) {
      const project = await db.project.findUnique({
        where: { id: body.projectId },
      });

      if (!project) {
        return NextResponse.json(
          {
            success: false,
            error: 'Project not found',
            message: `Project with id ${body.projectId} does not exist`,
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

    // Check if dataset exists
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

    if (!existingDataset) {
      return NextResponse.json(
        {
          success: false,
          error: 'Dataset not found',
          message: `Dataset with id ${id} does not exist`,
        },
        { status: 404 }
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
    const body = await request.json();

    // Check if dataset exists
    const existingDataset = await db.dataset.findUnique({
      where: { id },
    });

    if (!existingDataset) {
      return NextResponse.json(
        {
          success: false,
          error: 'Dataset not found',
          message: `Dataset with id ${id} does not exist`,
        },
        { status: 404 }
      );
    }

    const updatedDataset = await db.dataset.update({
      where: { id },
      data: body,
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
