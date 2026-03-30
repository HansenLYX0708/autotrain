import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";

// GET /api/models - Get all models with project info
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const projectId = searchParams.get("projectId");

    const where = projectId ? { projectId } : {};

    const models = await db.model.findMany({
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
        createdAt: "desc",
      },
    });

    return NextResponse.json({
      success: true,
      data: models,
    });
  } catch (error) {
    console.error("Error fetching models:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to fetch models",
      },
      { status: 500 }
    );
  }
}

// POST /api/models - Create a new model
export async function POST(request: Request) {
  try {
    const body = await request.json();

    const {
      name,
      description,
      projectId,
      architecture = "YOLOv3",
      backbone = "CSPResNet",
      neck = "CustomCSPPAN",
      head = "PPYOLOEHead",
      numClasses = 1,
      normType = "sync_bn",
      useEma = true,
      emaDecay = 0.9998,
      depthMult = 0.33,
      widthMult = 0.5,
      pretrainWeights,
      yamlConfig,
    } = body;

    // Validate required fields
    if (!name || !projectId) {
      return NextResponse.json(
        {
          success: false,
          error: "Missing required fields: name and projectId are required",
        },
        { status: 400 }
      );
    }

    // Check if project exists
    const project = await db.project.findUnique({
      where: { id: projectId },
    });

    if (!project) {
      return NextResponse.json(
        {
          success: false,
          error: "Project not found",
        },
        { status: 404 }
      );
    }

    const model = await db.model.create({
      data: {
        name,
        description,
        projectId,
        architecture,
        backbone,
        neck,
        head,
        numClasses,
        normType,
        useEma,
        emaDecay,
        depthMult,
        widthMult,
        pretrainWeights,
        yamlConfig,
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
        data: model,
      },
      { status: 201 }
    );
  } catch (error) {
    console.error("Error creating model:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to create model",
      },
      { status: 500 }
    );
  }
}
