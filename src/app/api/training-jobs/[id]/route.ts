import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { spawn } from "child_process";

// Store running processes
const runningProcesses = new Map<string, ReturnType<typeof spawn>>();

// GET /api/training-jobs/[id] - Get a single training job
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    const job = await db.trainingJob.findUnique({
      where: { id },
      include: {
        project: {
          select: {
            id: true,
            name: true,
            framework: true,
            status: true,
          },
        },
        dataset: {
          select: {
            id: true,
            name: true,
            format: true,
            trainImagePath: true,
            trainAnnoPath: true,
            evalImagePath: true,
            evalAnnoPath: true,
            numClasses: true,
            numTrainImages: true,
            numEvalImages: true,
          },
        },
        model: {
          select: {
            id: true,
            name: true,
            architecture: true,
            backbone: true,
            neck: true,
            head: true,
            numClasses: true,
            pretrainWeights: true,
          },
        },
        config: {
          select: {
            id: true,
            name: true,
            epoch: true,
            batchSize: true,
            baseLr: true,
            momentum: true,
            weightDecay: true,
            scheduler: true,
            warmupEpochs: true,
            maxEpochs: true,
            workerNum: true,
            evalHeight: true,
            evalWidth: true,
            saveDir: true,
            snapshotEpoch: true,
          },
        },
        logs: {
          orderBy: { timestamp: "desc" },
          take: 100,
          select: {
            id: true,
            epoch: true,
            iteration: true,
            totalIter: true,
            loss: true,
            lossCls: true,
            lossIou: true,
            lossDfl: true,
            lossL1: true,
            learningRate: true,
            eta: true,
            batchCost: true,
            dataCost: true,
            ips: true,
            memReserved: true,
            memAllocated: true,
            timestamp: true,
          },
        },
        _count: {
          select: { logs: true },
        },
      },
    });

    if (!job) {
      return NextResponse.json(
        { error: "Training job not found" },
        { status: 404 }
      );
    }

    return NextResponse.json(job);
  } catch (error) {
    console.error("Error fetching training job:", error);
    return NextResponse.json(
      { error: "Failed to fetch training job" },
      { status: 500 }
    );
  }
}

// PUT /api/training-jobs/[id] - Update a training job
export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const body = await request.json();

    // Check if job exists
    const existingJob = await db.trainingJob.findUnique({
      where: { id },
      include: {
        project: { select: { framework: true } },
      },
    });

    if (!existingJob) {
      return NextResponse.json(
        { error: "Training job not found" },
        { status: 404 }
      );
    }

    // Build update data object with only provided fields
    const updateData: Record<string, unknown> = {};

    // Basic fields
    if (body.name !== undefined) updateData.name = body.name;
    if (body.status !== undefined) updateData.status = body.status;
    if (body.command !== undefined) updateData.command = body.command;

    // Progress fields
    if (body.currentEpoch !== undefined) updateData.currentEpoch = body.currentEpoch;
    if (body.totalEpochs !== undefined) updateData.totalEpochs = body.totalEpochs;
    if (body.currentLoss !== undefined) updateData.currentLoss = body.currentLoss;
    if (body.currentLr !== undefined) updateData.currentLr = body.currentLr;

    // Path fields
    if (body.outputDir !== undefined) updateData.outputDir = body.outputDir;
    if (body.weightsPath !== undefined) updateData.weightsPath = body.weightsPath;
    if (body.vdlLogDir !== undefined) updateData.vdlLogDir = body.vdlLogDir;

    // Timing fields
    if (body.startedAt !== undefined) {
      updateData.startedAt = body.startedAt ? new Date(body.startedAt) : null;
    }
    if (body.completedAt !== undefined) {
      updateData.completedAt = body.completedAt ? new Date(body.completedAt) : null;
    }

    // Handle status change to running - start actual training
    if (body.status === "running" && existingJob.status !== "running") {
      updateData.startedAt = new Date();
      
      // Get system config for paths
      const systemConfig = await db.systemConfig.findFirst();
      
      if (existingJob.command && systemConfig) {
        const framework = existingJob.project?.framework || "PaddleDetection";
        const workDir = framework === "PaddleClas" 
          ? systemConfig.paddleClasPath 
          : systemConfig.paddleDetectionPath;

        if (workDir) {
          // Start training process
          startTrainingProcess(id, existingJob.command, workDir, systemConfig.pythonPath || "python");
        }
      }
    }

    // Handle status change to stopped - kill running process
    if (body.status === "stopped" && existingJob.status === "running") {
      const process = runningProcesses.get(id);
      if (process) {
        process.kill();
        runningProcesses.delete(id);
      }
    }

    const job = await db.trainingJob.update({
      where: { id },
      data: updateData,
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

    return NextResponse.json(job);
  } catch (error) {
    console.error("Error updating training job:", error);
    return NextResponse.json(
      { error: "Failed to update training job" },
      { status: 500 }
    );
  }
}

// DELETE /api/training-jobs/[id] - Delete a training job
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    // Check if job exists
    const existingJob = await db.trainingJob.findUnique({
      where: { id },
    });

    if (!existingJob) {
      return NextResponse.json(
        { error: "Training job not found" },
        { status: 404 }
      );
    }

    // Kill running process if exists
    const process = runningProcesses.get(id);
    if (process) {
      process.kill();
      runningProcesses.delete(id);
    }

    // Delete the job (logs will be cascade deleted)
    await db.trainingJob.delete({
      where: { id },
    });

    return NextResponse.json({ message: "Training job deleted successfully" });
  } catch (error) {
    console.error("Error deleting training job:", error);
    return NextResponse.json(
      { error: "Failed to delete training job" },
      { status: 500 }
    );
  }
}

// Start training process
function startTrainingProcess(jobId: string, command: string, workDir: string, pythonPath: string) {
  // Parse command - replace 'python' with configured python path
  const fullCommand = command.replace(/^python\b/, pythonPath);
  const parts = fullCommand.split(" ").filter(Boolean);
  
  const childProcess = spawn(parts[0], parts.slice(1), {
    cwd: workDir,
    shell: true,
    env: {
      ...process.env,
      PYTHONUNBUFFERED: "1",
    },
  });

  runningProcesses.set(jobId, childProcess);

  childProcess.stdout?.on("data", async (data: Buffer) => {
    const output = data.toString();
    console.log(`[Job ${jobId}] ${output}`);
    
    // Parse training progress from output
    await parseAndUpdateProgress(jobId, output);
  });

  childProcess.stderr?.on("data", (data: Buffer) => {
    console.error(`[Job ${jobId} ERROR] ${data.toString()}`);
  });

  childProcess.on("close", async (code) => {
    runningProcesses.delete(jobId);
    
    // Update job status
    const status = code === 0 ? "completed" : "failed";
    try {
      await db.trainingJob.update({
        where: { id: jobId },
        data: {
          status,
          completedAt: new Date(),
        },
      });
    } catch (error) {
      console.error("Failed to update job status:", error);
    }
  });

  childProcess.on("error", (error) => {
    console.error(`[Job ${jobId} PROCESS ERROR]`, error);
    runningProcesses.delete(jobId);
  });
}

// Parse training output and update progress
async function parseAndUpdateProgress(jobId: string, output: string) {
  try {
    // Common patterns in PaddleDetection/PaddleClas training output
    // Example: "epoch: 1, iter: 100, loss: 0.5, lr: 0.001"
    const epochMatch = output.match(/epoch[:\s]+(\d+)/i);
    const iterMatch = output.match(/iter[:\s]+(\d+)/i);
    const lossMatch = output.match(/loss[:\s]+([\d.]+)/i);
    const lrMatch = output.match(/lr[:\s]+([\d.e-]+)/i);

    const updateData: Record<string, unknown> = {};
    
    if (epochMatch) {
      updateData.currentEpoch = parseInt(epochMatch[1]);
    }
    if (lossMatch) {
      updateData.currentLoss = parseFloat(lossMatch[1]);
    }
    if (lrMatch) {
      updateData.currentLr = parseFloat(lrMatch[1]);
    }

    if (Object.keys(updateData).length > 0) {
      await db.trainingJob.update({
        where: { id: jobId },
        data: updateData,
      });

      // Also save log entry
      await db.trainingLog.create({
        data: {
          jobId,
          epoch: epochMatch ? parseInt(epochMatch[1]) : 0,
          iteration: iterMatch ? parseInt(iterMatch[1]) : 0,
          totalIter: 0,
          loss: lossMatch ? parseFloat(lossMatch[1]) : null,
          learningRate: lrMatch ? parseFloat(lrMatch[1]) : null,
          rawLog: output.slice(0, 1000), // Limit log size
        },
      });
    }
  } catch (error) {
    // Silently ignore parsing errors
  }
}
