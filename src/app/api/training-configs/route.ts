import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";

// GET /api/training-configs - Get all training configs
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const page = parseInt(searchParams.get("page") || "1");
    const limit = parseInt(searchParams.get("limit") || "20");
    const skip = (page - 1) * limit;
    const projectId = searchParams.get("projectId");

    const where = projectId ? { projectId } : {};

    const [configs, total] = await Promise.all([
      db.trainingConfig.findMany({
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
          _count: {
            select: { trainingJobs: true },
          },
        },
      }),
      db.trainingConfig.count({ where }),
    ]);

    return NextResponse.json({
      data: configs,
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
      },
    });
  } catch (error) {
    console.error("Error fetching training configs:", error);
    return NextResponse.json(
      { error: "Failed to fetch training configs" },
      { status: 500 }
    );
  }
}

// POST /api/training-configs - Create a new training config
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Validate required fields
    if (!body.name || !body.projectId) {
      return NextResponse.json(
        { error: "Missing required fields: name and projectId are required" },
        { status: 400 }
      );
    }

    const config = await db.trainingConfig.create({
      data: {
        name: body.name,
        projectId: body.projectId,
        // Training parameters
        epoch: body.epoch ?? 100,
        batchSize: body.batchSize ?? 8,
        baseLr: body.baseLr ?? 0.001,
        momentum: body.momentum ?? 0.9,
        weightDecay: body.weightDecay ?? 0.0005,
        // Scheduler
        scheduler: body.scheduler ?? "CosineDecay",
        warmupEpochs: body.warmupEpochs ?? 5,
        maxEpochs: body.maxEpochs ?? 100,
        // Reader settings
        workerNum: body.workerNum ?? 4,
        evalHeight: body.evalHeight ?? 640,
        evalWidth: body.evalWidth ?? 640,
        // Runtime
        useGpu: body.useGpu ?? true,
        logIter: body.logIter ?? 20,
        // Output
        saveDir: body.saveDir,
        snapshotEpoch: body.snapshotEpoch ?? 1,
        outputDir: body.outputDir,
        weights: body.weights,
        pretrainWeights: body.pretrainWeights,
        // YAML config
        yamlConfig: body.yamlConfig,
      },
    });

    return NextResponse.json(config, { status: 201 });
  } catch (error) {
    console.error("Error creating training config:", error);
    return NextResponse.json(
      { error: "Failed to create training config" },
      { status: 500 }
    );
  }
}
