import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { spawn } from "child_process";
import { exec } from "child_process";
import { logActivity } from "@/lib/activity-log";
import { getCurrentUser } from "@/lib/auth";

// Store running processes
const runningProcesses = new Map<string, ReturnType<typeof spawn>>();

// Helper function to get Python path for a job
async function getPythonPathForJob(job: any): Promise<string> {
  const systemConfig = await db.systemConfig.findFirst();
  
  // Parse training params for GPU info
  let trainingParams: Record<string, unknown> = {};
  try {
    trainingParams = job?.trainingParams 
      ? JSON.parse(job.trainingParams as string) 
      : {};
  } catch {
    // Ignore parse errors
  }
  
  const gpuIdsStr = (trainingParams.gpuIds as string) || '0';
  const gpuIds = gpuIdsStr.split(',').map(id => parseInt(id.trim(), 10)).filter(id => !isNaN(id));
  const primaryGpuId = gpuIds[0] || 0;
  
  let pythonPath = 'python';
  
  if (systemConfig?.gpuPythonMappings) {
    try {
      const gpuMappings = JSON.parse(systemConfig.gpuPythonMappings) as Record<string, string>;
      const gpuSpecificPath = gpuMappings[primaryGpuId.toString()];
      if (gpuSpecificPath && gpuSpecificPath.trim()) {
        pythonPath = gpuSpecificPath.trim();
      }
    } catch (e) {
      console.error(`Failed to parse GPU Python mappings:`, e);
    }
  }
  
  return pythonPath;
}

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

    // Get Python path for this job
    const pythonPath = await getPythonPathForJob(job);

    return NextResponse.json({ ...job, pythonPath });
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
    
    // Parse training params for GPU info
    let trainingParams: Record<string, unknown> = {};
    try {
      trainingParams = existingJob?.trainingParams 
        ? JSON.parse(existingJob.trainingParams as string) 
        : {};
    } catch {
      // Ignore parse errors
    }

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
    
    // Error message
    if (body.errorMessage !== undefined) updateData.errorMessage = body.errorMessage;

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
      
      // Log activity
      const user = await getCurrentUser(request);
      if (user) {
        await logActivity(user.userId, {
          action: 'start_training',
          entityType: 'job',
          entityId: existingJob.id,
          entityName: existingJob.name,
        });
      }
      
      // Get system config for paths
      const systemConfig = await db.systemConfig.findFirst();
      
      // Parse GPU IDs from training params
      const gpuIdsStr = (trainingParams.gpuIds as string) || '0';
      const gpuIds = gpuIdsStr.split(',').map(id => parseInt(id.trim(), 10)).filter(id => !isNaN(id));
      const primaryGpuId = gpuIds[0] || 0;
      
      // Determine Python path based on GPU mapping - MUST be configured
      let pythonPath: string | null = null;
      
      if (systemConfig?.gpuPythonMappings) {
        try {
          const gpuMappings = JSON.parse(systemConfig.gpuPythonMappings) as Record<string, string>;
          const gpuSpecificPath = gpuMappings[primaryGpuId.toString()];
          if (gpuSpecificPath && gpuSpecificPath.trim()) {
            pythonPath = gpuSpecificPath.trim();
            console.log(`[Job ${id}] Using GPU ${primaryGpuId} specific Python path: ${pythonPath}`);
          }
        } catch (e) {
          console.error(`[Job ${id}] Failed to parse GPU Python mappings:`, e);
        }
      }
      
      console.log(`[Job ${id}] Starting training...`);
      console.log(`[Job ${id}] Command: ${existingJob.command}`);
      console.log(`[Job ${id}] System config found: ${!!systemConfig}`);
      console.log(`[Job ${id}] Selected GPUs: ${gpuIdsStr}, Primary GPU: ${primaryGpuId}`);
      
      if (!existingJob.command) {
        console.error(`[Job ${id}] No command found for job`);
        updateData.status = "failed";
        updateData.errorMessage = "No training command configured for this job";
        updateData.completedAt = new Date();
      } else if (!systemConfig) {
        console.error(`[Job ${id}] System config not found`);
        updateData.status = "failed";
        updateData.errorMessage = "System configuration not found. Please configure paths in Settings.";
        updateData.completedAt = new Date();
      } else if (!pythonPath) {
        console.error(`[Job ${id}] No Python path configured for GPU ${primaryGpuId}`);
        updateData.status = "failed";
        updateData.errorMessage = `No Python environment configured for GPU ${primaryGpuId}. Please configure GPU Python mapping in Settings.`;
        updateData.completedAt = new Date();
      } else {
        const framework = existingJob.project?.framework || "PaddleDetection";
        const workDir = framework === "PaddleClas" 
          ? systemConfig.paddleClasPath 
          : systemConfig.paddleDetectionPath;

        console.log(`[Job ${id}] Framework: ${framework}`);
        console.log(`[Job ${id}] Work directory: ${workDir}`);
        console.log(`[Job ${id}] Python path: ${pythonPath}`);
        console.log(`[Job ${id}] Conda env: ${systemConfig.condaEnv || "not set"}`);

        if (!workDir) {
          console.error(`[Job ${id}] Work directory not configured`);
          updateData.status = "failed";
          updateData.errorMessage = `${framework} path not configured in Settings. Please configure the path first.`;
          updateData.completedAt = new Date();
        } else {
          // Start training process
          try {
            startTrainingProcess(
              id, 
              existingJob.command, 
              workDir, 
              pythonPath, 
              gpuIdsStr,
              systemConfig.condaEnv || null,
              systemConfig.condaPath || null
            );
            console.log(`[Job ${id}] Training process started successfully`);
          } catch (error) {
            console.error(`[Job ${id}] Failed to start training process:`, error);
            updateData.status = "failed";
            updateData.errorMessage = `Failed to start training: ${error instanceof Error ? error.message : "Unknown error"}`;
            updateData.completedAt = new Date();
          }
        }
      }
    }

    // Handle status change to stopped - kill running process
    if (body.status === "stopped" && existingJob.status === "running") {
      // Log activity
      const user = await getCurrentUser(request);
      if (user) {
        await logActivity(user.userId, {
          action: 'stop_training',
          entityType: 'job',
          entityId: existingJob.id,
          entityName: existingJob.name,
        });
      }
      
      const process = runningProcesses.get(id);
      if (process) {
        if (process?.pid) {
          killProcessTree(process.pid);
        }
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

    // Log activity
    const user = await getCurrentUser(request);
    if (user) {
      await logActivity(user.userId, {
        action: 'delete_job',
        entityType: 'job',
        entityId: existingJob.id,
        entityName: existingJob.name,
      });
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

function killProcessTree(pid: number) {
  exec(`taskkill /PID ${pid} /T /F`, (err) => {
    if (err) console.error(err);
  });
}

// Start training process
function startTrainingProcess(
  jobId: string, 
  command: string, 
  workDir: string, 
  pythonPath: string, 
  gpuIds: string = '0',
  condaEnv: string | null = null,
  condaPath: string | null = null
) {
  console.log(`\n========== TRAINING PROCESS START ==========`);
  console.log(`[Job ${jobId}] GPU(s): ${gpuIds}`);
  console.log(`[Job ${jobId}] Python path: "${pythonPath}"`);
  console.log(`[Job ${jobId}] Conda env (from config): "${condaEnv || 'not set'}"`);
  console.log(`[Job ${jobId}] Conda path (from config): "${condaPath || 'not set'}"`);
  console.log(`[Job ${jobId}] Original command: ${command}`);
  console.log(`[Job ${jobId}] Working directory: ${workDir}`);
  
  // Collect stderr for error reporting
  let stderrCollector: string[] = [];
  
  // Build environment with CUDA_VISIBLE_DEVICES
  const env: Record<string, string> = {
    ...process.env,
    PYTHONUNBUFFERED: "1",
    CUDA_VISIBLE_DEVICES: gpuIds,
  };
  
  // Detect if python path is in a conda environment
  let detectedCondaEnv: string | null = condaEnv;
  let detectedCondaPath: string | null = condaPath;
  
  // Auto-detect conda environment from python path
  if (!detectedCondaEnv && pythonPath) {
    console.log(`[Job ${jobId}] Attempting to auto-detect conda environment...`);
    
    // Check for conda envs pattern - multiple regex patterns for different installations
    const patterns = [
      // Standard anaconda/miniconda with envs: /envs/envname/
      /[\/\\](?:anaconda3|miniconda3|anaconda|miniconda)[\/\\]envs[\/\\]([^\/\\]+)/i,
      // condax envs pattern
      /[\/\\]\.condax[\/\\]([^\/\\]+)/i,
      // Direct env path (some custom setups)
      /[\/\\]envs[\/\\]([^\/\\]+)/i,
    ];
    
    for (const pattern of patterns) {
      const match = pythonPath.match(pattern);
      if (match) {
        detectedCondaEnv = match[1];
        console.log(`[Job ${jobId}] ✓ Conda environment detected via pattern: ${detectedCondaEnv}`);
        break;
      }
    }
    
    if (!detectedCondaEnv) {
      console.log(`[Job ${jobId}] ✗ No conda environment pattern matched in python path`);
    }
    
    // Try to detect conda executable path
    if (detectedCondaEnv && !detectedCondaPath) {
      const pathParts = pythonPath.split(/[\/\\]/);
      const envsIndex = pathParts.findIndex(p => p === 'envs');
      if (envsIndex > 0) {
        const condaRoot = pathParts.slice(0, envsIndex).join('/');
        const isWindows = pythonPath.includes('\\') || pythonPath.match(/^[A-Z]:\\/i);
        if (isWindows) {
          // Windows: conda.exe is in Scripts folder
          detectedCondaPath = `${condaRoot}\\Scripts\\conda.exe`;
        } else {
          detectedCondaPath = `${condaRoot}/bin/conda`;
        }
        console.log(`[Job ${jobId}] Detected conda path: ${detectedCondaPath}`);
      }
    }
  }
  
  // Build the final command
  let fullCommand = command;
  
  if (detectedCondaEnv) {
    const condaExec = detectedCondaPath || 'conda';
    console.log(`[Job ${jobId}] Using conda executable: "${condaExec}"`);
    console.log(`[Job ${jobId}] Using conda environment: "${detectedCondaEnv}"`);
    
    // Use conda run with --no-capture-output to see real-time output
    // Wrap the command properly for Windows
    const isWindows = workDir.includes('\\') || workDir.match(/^[A-Z]:\\/i);
    if (isWindows) {
      // On Windows, wrap in quotes properly
      fullCommand = `"${condaExec}" run -n ${detectedCondaEnv} --no-capture-output ${fullCommand}`;
    } else {
      fullCommand = `${condaExec} run -n ${detectedCondaEnv} --no-capture-output ${fullCommand}`;
    }
    console.log(`[Job ${jobId}] Conda command built: ${fullCommand}`);
  } else {
    console.log(`[Job ${jobId}] No conda environment - using direct python execution`);
    let pythonExec = pythonPath || 'python';
    if (pythonExec.includes(' ')) {
      pythonExec = `"${pythonExec}"`;
    }
    fullCommand = fullCommand.replace(/^python\b/, pythonExec);
  }
  
  console.log(`\n[Job ${jobId}] ===== FINAL COMMAND =====`);
  console.log(`[Job ${jobId}] ${fullCommand}`);
  console.log(`[Job ${jobId}] ============================\n`);
  
  const childProcess = spawn(fullCommand, [], {
    cwd: workDir,
    shell: true,
    env,
  });

  runningProcesses.set(jobId, childProcess);

  childProcess.stdout?.on("data", async (data: Buffer) => {
    const output = data.toString();
    console.log(`[Job ${jobId}] ${output}`);
    
    // Parse training progress from output
    await parseAndUpdateProgress(jobId, output);
  });

  childProcess.stderr?.on("data", (data: Buffer) => {
    const stderrOutput = data.toString();
    console.error(`[Job ${jobId} ERROR] ${stderrOutput}`);
    // Collect stderr for error message
    stderrCollector.push(stderrOutput);
    // Keep only last 50 lines to avoid memory issues
    if (stderrCollector.length > 50) {
      stderrCollector = stderrCollector.slice(-50);
    }
  });

  childProcess.on("close", async (code) => {
    runningProcesses.delete(jobId);
    
    // Update job status
    const status = code === 0 ? "completed" : "failed";
    try {
      const updateData: Record<string, unknown> = {
        status,
        completedAt: new Date(),
      };
      
      // If failed, capture stderr and exit code for error message
      if (status === "failed") {
        const stderrSummary = stderrCollector.slice(-10).join('\n').trim();
        if (stderrSummary) {
          updateData.errorMessage = `Training failed with exit code ${code}:\n${stderrSummary}`;
        } else {
          updateData.errorMessage = `Training process exited with code ${code}. Check logs for details.`;
        }
      }
      
      await db.trainingJob.update({
        where: { id: jobId },
        data: updateData,
      });
    } catch (error) {
      console.error("Failed to update job status:", error);
    }
  });

  childProcess.on("error", async (error) => {
    console.error(`[Job ${jobId} PROCESS ERROR]`, error);
    runningProcesses.delete(jobId);
    
    // Update job with error
    try {
      await db.trainingJob.update({
        where: { id: jobId },
        data: {
          status: "failed",
          errorMessage: error.message || "Failed to start training process",
          completedAt: new Date(),
        },
      });
    } catch (dbError) {
      console.error("Failed to update job with error:", dbError);
    }
  });
}

// Parse training output and update progress
async function parseAndUpdateProgress(jobId: string, output: string) {
  try {
    // PaddleDetection log format:
    // Epoch: [8] [60/79] learning_rate: 0.000996 loss: 4.193813 loss_cls: 1.671748 ...
    
    // Match epoch: Epoch: [8]
    const epochMatch = output.match(/Epoch:\s*\[(\d+)\]/i);
    
    // Match iteration: [iter/total] - find all patterns, use the last one (after epoch)
    const iterPatterns = output.matchAll(/\[(\d+)\/(\d+)\]/g);
    const iterMatches = Array.from(iterPatterns);
    let iteration = 0;
    let totalIter = 0;
    if (iterMatches.length > 0) {
      // Use the last [x/y] pattern (which is the iteration, after epoch)
      const lastMatch = iterMatches[iterMatches.length - 1];
      iteration = parseInt(lastMatch[1], 10);
      totalIter = parseInt(lastMatch[2], 10);
    }

    // Extract metrics - support both "learning_rate" and "lr"
    const lrMatch = output.match(/learning_rate:\s*([\d.e-]+)/i) || output.match(/lr[:\s]+([\d.e-]+)/i);
    // Match "loss:" but NOT "loss_cls:", "loss_iou:", etc. (use negative lookahead for underscore)
    const lossMatch = output.match(/(?:^|\s)loss:\s*([\d.]+)(?!\w)/i);
    const lossClsMatch = output.match(/loss_cls:\s*([\d.]+)/i);
    const lossIouMatch = output.match(/loss_iou:\s*([\d.]+)/i);
    const lossDflMatch = output.match(/loss_dfl:\s*([\d.]+)/i);
    const lossL1Match = output.match(/loss_l1:\s*([\d.]+)/i);
    
    // Extract ETA: eta: 0:02:24
    const etaMatch = output.match(/eta:\s*(\d+:\d{2}:\d{2})/i);
    
    // Extract costs
    const batchCostMatch = output.match(/batch_cost:\s*([\d.]+)/i);
    const dataCostMatch = output.match(/data_cost:\s*([\d.]+)/i);
    
    // Extract IPS
    const ipsMatch = output.match(/ips:\s*([\d.]+)/i);
    
    // Extract memory info (MB)
    const memReservedMatch = output.match(/max_mem_reserved:\s*(\d+)/i);
    const memAllocatedMatch = output.match(/max_mem_allocated:\s*(\d+)/i);

    // Skip if no training metrics found
    if (!epochMatch && !lossMatch && !lrMatch) {
      return;
    }

    // Update job progress
    const updateData: Record<string, unknown> = {};
    
    if (epochMatch) {
      updateData.currentEpoch = parseInt(epochMatch[1], 10);
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
    }

    // Save detailed log entry
    await db.trainingLog.create({
      data: {
        jobId,
        epoch: epochMatch ? parseInt(epochMatch[1], 10) : 0,
        iteration,
        totalIter,
        loss: lossMatch ? parseFloat(lossMatch[1]) : null,
        lossCls: lossClsMatch ? parseFloat(lossClsMatch[1]) : null,
        lossIou: lossIouMatch ? parseFloat(lossIouMatch[1]) : null,
        lossDfl: lossDflMatch ? parseFloat(lossDflMatch[1]) : null,
        lossL1: lossL1Match ? parseFloat(lossL1Match[1]) : null,
        learningRate: lrMatch ? parseFloat(lrMatch[1]) : null,
        eta: etaMatch ? etaMatch[1] : null,
        batchCost: batchCostMatch ? parseFloat(batchCostMatch[1]) : null,
        dataCost: dataCostMatch ? parseFloat(dataCostMatch[1]) : null,
        ips: ipsMatch ? parseFloat(ipsMatch[1]) : null,
        memReserved: memReservedMatch ? parseInt(memReservedMatch[1], 10) : null,
        memAllocated: memAllocatedMatch ? parseInt(memAllocatedMatch[1], 10) : null,
        rawLog: output.slice(0, 2000),
      },
    });
  } catch (error) {
    // Silently ignore parsing errors
  }
}
