import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { notFoundOrDenied, requireOwnedScope } from "@/lib/auth";

/**
 * Authenticate and confirm the caller may act on this job's logs.
 *
 * All three handlers here previously ran unauthenticated, which let anyone
 * read another user's training metrics, inject fabricated log rows (POST also
 * rewrites the parent job's progress), or wipe a job's history.
 */
async function requireJobAccess(request: NextRequest, id: string) {
  const scope = await requireOwnedScope(request, id);
  if (scope instanceof NextResponse) return { error: scope };

  const job = await db.trainingJob.findFirst({
    where: scope.where,
    select: { id: true, name: true, status: true },
  });
  if (!job) return { error: notFoundOrDenied("Training job") };

  return { job };
}

// GET /api/training-jobs/[id]/logs - Get logs for a training job
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const { searchParams } = new URL(request.url);

    const access = await requireJobAccess(request, id);
    if (access.error) return access.error;
    const job = access.job;

    // Pagination. Clamp so a hostile or accidental `limit=1000000` cannot pull
    // the entire log table into memory.
    const page = Math.max(1, parseInt(searchParams.get("page") || "1") || 1);
    const limit = Math.min(1000, Math.max(1, parseInt(searchParams.get("limit") || "100") || 100));
    const skip = (page - 1) * limit;

    // Filters
    const epoch = searchParams.get("epoch");
    const minLoss = searchParams.get("minLoss");
    const maxLoss = searchParams.get("maxLoss");

    // Sort order
    const sortOrder = searchParams.get("sort") === "asc" ? "asc" : "desc";

    // Build filter conditions
    const where: Record<string, unknown> = { jobId: id };
    if (epoch) {
      where.epoch = parseInt(epoch);
    }
    if (minLoss || maxLoss) {
      where.loss = {};
      if (minLoss) (where.loss as Record<string, unknown>).gte = parseFloat(minLoss);
      if (maxLoss) (where.loss as Record<string, unknown>).lte = parseFloat(maxLoss);
    }

    const [logs, total] = await Promise.all([
      db.trainingLog.findMany({
        where,
        skip,
        take: limit,
        orderBy: { timestamp: sortOrder },
      }),
      db.trainingLog.count({ where }),
    ]);

    // Calculate summary statistics
    const stats = await db.trainingLog.aggregate({
      where: { jobId: id },
      _count: true,
      _min: {
        loss: true,
        learningRate: true,
      },
      _max: {
        loss: true,
        learningRate: true,
      },
      _avg: {
        loss: true,
        lossCls: true,
        lossIou: true,
        lossDfl: true,
        lossL1: true,
        learningRate: true,
        batchCost: true,
        ips: true,
      },
    });

    // Get unique epochs
    const epochs = await db.trainingLog.findMany({
      where: { jobId: id },
      select: { epoch: true },
      distinct: ["epoch"],
      orderBy: { epoch: "asc" },
    });

    return NextResponse.json({
      job: {
        id: job.id,
        name: job.name,
        status: job.status,
      },
      data: logs,
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
      },
      stats: {
        totalLogs: stats._count,
        minLoss: stats._min.loss,
        maxLoss: stats._max.loss,
        avgLoss: stats._avg.loss,
        avgLearningRate: stats._avg.learningRate,
        avgBatchCost: stats._avg.batchCost,
        avgIps: stats._avg.ips,
      },
      epochs: epochs.map((e) => e.epoch),
    });
  } catch (error) {
    console.error("Error fetching training logs:", error);
    return NextResponse.json(
      { error: "Failed to fetch training logs" },
      { status: 500 }
    );
  }
}

// POST /api/training-jobs/[id]/logs - Add a new log entry
export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    const access = await requireJobAccess(request, id);
    if (access.error) return access.error;

    const body = await request.json();

    const log = await db.trainingLog.create({
      data: {
        jobId: id,
        epoch: body.epoch,
        iteration: body.iteration,
        totalIter: body.totalIter,
        // Loss metrics
        loss: body.loss,
        lossCls: body.lossCls,
        lossIou: body.lossIou,
        lossDfl: body.lossDfl,
        lossL1: body.lossL1,
        learningRate: body.learningRate,
        // Performance metrics
        eta: body.eta,
        batchCost: body.batchCost,
        dataCost: body.dataCost,
        ips: body.ips,
        memReserved: body.memReserved,
        memAllocated: body.memAllocated,
        // Raw log
        rawLog: body.rawLog,
      },
    });

    // Update the training job's current progress
    await db.trainingJob.update({
      where: { id },
      data: {
        currentEpoch: body.epoch,
        currentLoss: body.loss,
        currentLr: body.learningRate,
      },
    });

    return NextResponse.json(log, { status: 201 });
  } catch (error) {
    console.error("Error creating training log:", error);
    return NextResponse.json(
      { error: "Failed to create training log" },
      { status: 500 }
    );
  }
}

// DELETE /api/training-jobs/[id]/logs - Delete all logs for a job
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    const access = await requireJobAccess(request, id);
    if (access.error) return access.error;

    // Delete all logs for this job
    const result = await db.trainingLog.deleteMany({
      where: { jobId: id },
    });

    return NextResponse.json({
      success: true,
      message: "Training logs deleted successfully",
      deletedCount: result.count,
    });
  } catch (error) {
    console.error("Error deleting training logs:", error);
    return NextResponse.json(
      { error: "Failed to delete training logs" },
      { status: 500 }
    );
  }
}
