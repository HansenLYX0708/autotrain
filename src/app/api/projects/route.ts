import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { requireAuth, buildUserFilter } from '@/lib/auth';
import { logActivity } from '@/lib/activity-log';
import { normalizeFramework, resolveTask } from '@/lib/frameworks';

// GET /api/projects - Get all projects with counts (filtered by user for non-admins)
export async function GET(request: NextRequest) {
  try {
    // Check authentication
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;

    const { userId, role } = auth;
    const whereClause = buildUserFilter(userId, role, 'userId');

    // Paginated/filtered listing for the Projects page. When none of these
    // params is present the response shape stays { success, data } exactly as
    // before — other pages fetch this route for project dropdowns.
    const params = request.nextUrl.searchParams;
    const query = params.get('q')?.trim() || '';
    const limitParam = params.get('limit');
    const pageParam = params.get('page');
    const pageSizeParam = params.get('pageSize');
    if (query || limitParam !== null || pageParam !== null || pageSizeParam !== null) {
      const where = query
        ? { AND: [whereClause, { OR: [
            { name: { contains: query } },
            { description: { contains: query } },
          ] }] }
        : whereClause;
      const include = { _count: { select: { datasets: true, models: true, trainingJobs: true } } };
      const orderBy = { updatedAt: 'desc' as const };
      let take: number | undefined;
      let skip: number | undefined;
      let page = 1;
      let pageSize = 0;
      if (limitParam !== null) {
        take = Math.max(1, Math.min(100, parseInt(limitParam, 10) || 5));
      } else {
        pageSize = Math.max(1, Math.min(100, parseInt(pageSizeParam ?? '', 10) || 12));
        page = Math.max(1, parseInt(pageParam ?? '', 10) || 1);
        take = pageSize;
        skip = (page - 1) * pageSize;
      }
      const [projects, total] = await db.$transaction([
        db.project.findMany({ where, orderBy, take, skip, include }),
        db.project.count({ where }),
      ]);
      return NextResponse.json({ success: true, data: projects, total, page, pageSize });
    }

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
    const { name, description, framework, task } = body;

    // Resolve framework and task from the framework registry: segmentation
    // frameworks always use semantic_segmentation, classification uses
    // classification, and instance_segmentation is only meaningful for the
    // detection frameworks. Unknown values normalise to PaddleDetection.
    const allowedTasks = ['detection', 'instance_segmentation'];
    const resolvedFramework = normalizeFramework(framework);
    const finalTask = resolveTask(
      resolvedFramework,
      task && allowedTasks.includes(task) ? task : undefined,
    );

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
        framework: resolvedFramework,
        task: finalTask,
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

    // Log activity
    await logActivity(userId, {
      action: 'create_project',
      entityType: 'project',
      entityId: project.id,
      entityName: project.name,
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
