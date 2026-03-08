import { NextResponse } from "next/server";
import { db } from "@/lib/db";

// GET /api/models/[id] - Get a single model by ID
export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    const model = await db.model.findUnique({
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
            createdAt: "desc",
          },
          take: 10,
        },
      },
    });

    if (!model) {
      return NextResponse.json(
        {
          success: false,
          error: "Model not found",
        },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      data: model,
    });
  } catch (error) {
    console.error("Error fetching model:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to fetch model",
      },
      { status: 500 }
    );
  }
}

// PUT /api/models/[id] - Update a model
export async function PUT(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const body = await request.json();

    // Check if model exists
    const existingModel = await db.model.findUnique({
      where: { id },
    });

    if (!existingModel) {
      return NextResponse.json(
        {
          success: false,
          error: "Model not found",
        },
        { status: 404 }
      );
    }

    const {
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
    } = body;

    // If projectId is being changed, verify the new project exists
    if (projectId && projectId !== existingModel.projectId) {
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
    }

    const updatedModel = await db.model.update({
      where: { id },
      data: {
        ...(name !== undefined && { name }),
        ...(description !== undefined && { description }),
        ...(projectId !== undefined && { projectId }),
        ...(architecture !== undefined && { architecture }),
        ...(backbone !== undefined && { backbone }),
        ...(neck !== undefined && { neck }),
        ...(head !== undefined && { head }),
        ...(numClasses !== undefined && { numClasses }),
        ...(normType !== undefined && { normType }),
        ...(useEma !== undefined && { useEma }),
        ...(emaDecay !== undefined && { emaDecay }),
        ...(depthMult !== undefined && { depthMult }),
        ...(widthMult !== undefined && { widthMult }),
        ...(pretrainWeights !== undefined && { pretrainWeights }),
        ...(yamlConfig !== undefined && { yamlConfig }),
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
      data: updatedModel,
    });
  } catch (error) {
    console.error("Error updating model:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to update model",
      },
      { status: 500 }
    );
  }
}

// DELETE /api/models/[id] - Delete a model
export async function DELETE(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    // Check if model exists
    const existingModel = await db.model.findUnique({
      where: { id },
      include: {
        _count: {
          select: {
            trainingJobs: true,
          },
        },
      },
    });

    if (!existingModel) {
      return NextResponse.json(
        {
          success: false,
          error: "Model not found",
        },
        { status: 404 }
      );
    }

    // Delete the model (cascade will handle trainingJobs relation)
    await db.model.delete({
      where: { id },
    });

    return NextResponse.json({
      success: true,
      message: "Model deleted successfully",
      data: {
        id,
        deletedTrainingJobsCount: existingModel._count.trainingJobs,
      },
    });
  } catch (error) {
    console.error("Error deleting model:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to delete model",
      },
      { status: 500 }
    );
  }
}
