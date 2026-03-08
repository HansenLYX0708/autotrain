import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";

// GET /api/training-jobs - Get all training jobs with relations
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const page = parseInt(searchParams.get("page") || "1");
    const limit = parseInt(searchParams.get("limit") || "20");
    const skip = (page - 1) * limit;
    const status = searchParams.get("status");
    const projectId = searchParams.get("projectId");

    // Build filter conditions
    const where: Record<string, unknown> = {};
    if (status) {
      where.status = status;
    }
    if (projectId) {
      where.projectId = projectId;
    }

    const [jobs, total] = await Promise.all([
      db.trainingJob.findMany({
        where,
        skip,
        take: limit,
        orderBy: { createdAt: "desc" },
        include: {
          project: {
            select: {
              id: true,
              name: true,
              framework: true,
            },
          },
          dataset: {
            select: {
              id: true,
              name: true,
              format: true,
            },
          },
          model: {
            select: {
              id: true,
              name: true,
              architecture: true,
            },
          },
          config: {
            select: {
              id: true,
              name: true,
            },
          },
          _count: {
            select: { logs: true },
          },
        },
      }),
      db.trainingJob.count({ where }),
    ]);

    return NextResponse.json({
      data: jobs,
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
      },
    });
  } catch (error) {
    console.error("Error fetching training jobs:", error);
    return NextResponse.json(
      { error: "Failed to fetch training jobs" },
      { status: 500 }
    );
  }
}

// POST /api/training-jobs - Create a new training job
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Validate required relations exist
    const [project, dataset, model] = await Promise.all([
      db.project.findUnique({ where: { id: body.projectId } }),
      db.dataset.findUnique({ where: { id: body.datasetId } }),
      db.model.findUnique({ where: { id: body.modelId } }),
    ]);

    if (!project) {
      return NextResponse.json(
        { error: "Project not found" },
        { status: 400 }
      );
    }
    if (!dataset) {
      return NextResponse.json(
        { error: "Dataset not found" },
        { status: 400 }
      );
    }
    if (!model) {
      return NextResponse.json(
        { error: "Model not found" },
        { status: 400 }
      );
    }

    // Validate config if provided
    if (body.configId) {
      const config = await db.trainingConfig.findUnique({
        where: { id: body.configId },
      });
      if (!config) {
        return NextResponse.json(
          { error: "Training config not found" },
          { status: 400 }
        );
      }
    }

    const job = await db.trainingJob.create({
      data: {
        projectId: body.projectId,
        datasetId: body.datasetId,
        modelId: body.modelId,
        configId: body.configId,
        name: body.name,
        status: body.status ?? "pending",
        command: body.command,
        // Progress
        currentEpoch: body.currentEpoch ?? 0,
        totalEpochs: body.totalEpochs ?? 100,
        currentLoss: body.currentLoss,
        currentLr: body.currentLr,
        // Paths
        outputDir: body.outputDir,
        weightsPath: body.weightsPath,
        vdlLogDir: body.vdlLogDir,
        // Training params and YAML config
        trainingParams: body.trainingParams ? JSON.stringify(body.trainingParams) : null,
        yamlConfig: body.yamlConfig || null,
        // Timing
        startedAt: body.startedAt ? new Date(body.startedAt) : undefined,
        completedAt: body.completedAt ? new Date(body.completedAt) : undefined,
      },
      include: {
        project: {
          select: {
            id: true,
            name: true,
            framework: true,
          },
        },
        dataset: {
          select: {
            id: true,
            name: true,
            format: true,
          },
        },
        model: {
          select: {
            id: true,
            name: true,
            architecture: true,
          },
        },
        config: {
          select: {
            id: true,
            name: true,
          },
        },
      },
    });

    return NextResponse.json(job, { status: 201 });
  } catch (error) {
    console.error("Error creating training job:", error);
    return NextResponse.json(
      { error: "Failed to create training job" },
      { status: 500 }
    );
  }
}
