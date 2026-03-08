import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';

// GET /api/validation-jobs - Get all validation jobs with project relation
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const projectId = searchParams.get('projectId');
    const status = searchParams.get('status');
    const type = searchParams.get('type');

    const where: {
      projectId?: string;
      status?: string;
      type?: string;
    } = {};

    if (projectId) {
      where.projectId = projectId;
    }
    if (status) {
      where.status = status;
    }
    if (type) {
      where.type = type;
    }

    const validationJobs = await db.validationJob.findMany({
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
      data: validationJobs,
    });
  } catch (error) {
    console.error('Error fetching validation jobs:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to fetch validation jobs',
      },
      { status: 500 }
    );
  }
}

// POST /api/validation-jobs - Create a new validation job
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Validate required fields
    if (!body.projectId) {
      return NextResponse.json(
        {
          success: false,
          error: 'projectId is required',
        },
        { status: 400 }
      );
    }

    if (!body.name) {
      return NextResponse.json(
        {
          success: false,
          error: 'name is required',
        },
        { status: 400 }
      );
    }

    // Validate type
    const validTypes = ['eval', 'infer_single', 'infer_batch', 'export_trt'];
    if (body.type && !validTypes.includes(body.type)) {
      return NextResponse.json(
        {
          success: false,
          error: `Invalid type. Must be one of: ${validTypes.join(', ')}`,
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
        },
        { status: 404 }
      );
    }

    const validationJob = await db.validationJob.create({
      data: {
        projectId: body.projectId,
        name: body.name,
        type: body.type || 'eval',
        datasetPath: body.datasetPath || null,
        inferPath: body.inferPath || null,
        weightsPath: body.weightsPath || null,
        outputDir: body.outputDir || null,
        status: body.status || 'pending',
        command: body.command || null,
        mapResult: body.mapResult ? parseFloat(body.mapResult) : null,
        resultPath: body.resultPath || null,
        startedAt: body.startedAt ? new Date(body.startedAt) : null,
        completedAt: body.completedAt ? new Date(body.completedAt) : null,
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
        data: validationJob,
      },
      { status: 201 }
    );
  } catch (error) {
    console.error('Error creating validation job:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to create validation job',
      },
      { status: 500 }
    );
  }
}
