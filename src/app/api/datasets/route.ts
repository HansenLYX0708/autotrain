import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';

// GET /api/datasets - Get all datasets with project info
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const projectId = searchParams.get('projectId');

    const where = projectId ? { projectId } : {};

    const datasets = await db.dataset.findMany({
      where,
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
      orderBy: {
        createdAt: 'desc',
      },
    });

    return NextResponse.json({
      success: true,
      data: datasets,
    });
  } catch (error) {
    console.error('Error fetching datasets:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to fetch datasets',
        message: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}

// POST /api/datasets - Create a new dataset
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Validate required fields
    if (!body.name || !body.projectId) {
      return NextResponse.json(
        {
          success: false,
          error: 'Missing required fields',
          message: 'name and projectId are required',
        },
        { status: 400 }
      );
    }

    // Check if project exists
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

    const dataset = await db.dataset.create({
      data: {
        name: body.name,
        description: body.description || null,
        projectId: body.projectId,
        format: body.format || 'COCO',
        trainImagePath: body.trainImagePath || null,
        trainAnnoPath: body.trainAnnoPath || null,
        evalImagePath: body.evalImagePath || null,
        evalAnnoPath: body.evalAnnoPath || null,
        datasetDir: body.datasetDir || null,
        numClasses: body.numClasses || 0,
        numAnnotations: body.numAnnotations || 0,
        numTrainImages: body.numTrainImages || 0,
        numEvalImages: body.numEvalImages || 0,
        classStats: body.classStats || null,
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

    return NextResponse.json(
      {
        success: true,
        data: dataset,
        message: 'Dataset created successfully',
      },
      { status: 201 }
    );
  } catch (error) {
    console.error('Error creating dataset:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to create dataset',
        message: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}
