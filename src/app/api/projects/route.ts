import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { requireAuth, buildUserFilter } from '@/lib/auth';

// GET /api/projects - Get all projects with counts (filtered by user for non-admins)
export async function GET(request: NextRequest) {
  try {
    // Check authentication
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;

    const { userId, role } = auth;
    const whereClause = buildUserFilter(userId, role, 'userId');

    const projects = await db.project.findMany({
      where: whereClause,
      orderBy: {
        createdAt: 'desc',
      },
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
      data: projects,
    });
  } catch (error) {
    console.error('Error fetching projects:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to fetch projects',
      },
      { status: 500 }
    );
  }
}

// POST /api/projects - Create a new project
export async function POST(request: NextRequest) {
  try {
    // Check authentication
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;

    const { userId } = auth;

    const body = await request.json();
    const { name, description, framework } = body;

    // Validate required fields
    if (!name || typeof name !== 'string' || name.trim() === '') {
      return NextResponse.json(
        {
          success: false,
          error: 'Project name is required',
        },
        { status: 400 }
      );
    }

    // Check if project with same name already exists
    const existingProject = await db.project.findFirst({
      where: {
        name: name.trim(),
      },
    });

    if (existingProject) {
      return NextResponse.json(
        {
          success: false,
          error: 'A project with this name already exists',
        },
        { status: 409 }
      );
    }

    // Create the project with userId
    const project = await db.project.create({
      data: {
        name: name.trim(),
        description: description?.trim() || null,
        framework: framework || 'PaddleDetection',
        status: 'active',
        userId: userId,
      },
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

    return NextResponse.json(
      {
        success: true,
        data: project,
        message: 'Project created successfully',
      },
      { status: 201 }
    );
  } catch (error) {
    console.error('Error creating project:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to create project',
      },
      { status: 500 }
    );
  }
}
