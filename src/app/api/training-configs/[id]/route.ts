import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";

// GET /api/training-configs/[id] - Get a single training config
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    const config = await db.trainingConfig.findUnique({
      where: { id },
      include: {
        trainingJobs: {
          select: {
            id: true,
            name: true,
            status: true,
            createdAt: true,
          },
          orderBy: { createdAt: "desc" },
          take: 10,
        },
        _count: {
          select: { trainingJobs: true },
        },
      },
    });

    if (!config) {
      return NextResponse.json(
        { error: "Training config not found" },
        { status: 404 }
      );
    }

    return NextResponse.json(config);
  } catch (error) {
    console.error("Error fetching training config:", error);
    return NextResponse.json(
      { error: "Failed to fetch training config" },
      { status: 500 }
    );
  }
}

// PUT /api/training-configs/[id] - Update a training config
export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const body = await request.json();

    // Check if config exists
    const existingConfig = await db.trainingConfig.findUnique({
      where: { id },
    });

    if (!existingConfig) {
      return NextResponse.json(
        { error: "Training config not found" },
        { status: 404 }
      );
    }

    const config = await db.trainingConfig.update({
      where: { id },
      data: {
        name: body.name,
        // Training parameters
        epoch: body.epoch,
        batchSize: body.batchSize,
        baseLr: body.baseLr,
        momentum: body.momentum,
        weightDecay: body.weightDecay,
        // Scheduler
        scheduler: body.scheduler,
        warmupEpochs: body.warmupEpochs,
        maxEpochs: body.maxEpochs,
        // Reader settings
        workerNum: body.workerNum,
        evalHeight: body.evalHeight,
        evalWidth: body.evalWidth,
        // Output
        saveDir: body.saveDir,
        snapshotEpoch: body.snapshotEpoch,
      },
    });

    return NextResponse.json(config);
  } catch (error) {
    console.error("Error updating training config:", error);
    return NextResponse.json(
      { error: "Failed to update training config" },
      { status: 500 }
    );
  }
}

// DELETE /api/training-configs/[id] - Delete a training config
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    // Check if config exists
    const existingConfig = await db.trainingConfig.findUnique({
      where: { id },
      include: {
        _count: {
          select: { trainingJobs: true },
        },
      },
    });

    if (!existingConfig) {
      return NextResponse.json(
        { error: "Training config not found" },
        { status: 404 }
      );
    }

    // Check if config is being used by any training jobs
    if (existingConfig._count.trainingJobs > 0) {
      return NextResponse.json(
        { error: "Cannot delete training config that is being used by training jobs" },
        { status: 400 }
      );
    }

    await db.trainingConfig.delete({
      where: { id },
    });

    return NextResponse.json({ message: "Training config deleted successfully" });
  } catch (error) {
    console.error("Error deleting training config:", error);
    return NextResponse.json(
      { error: "Failed to delete training config" },
      { status: 500 }
    );
  }
}
