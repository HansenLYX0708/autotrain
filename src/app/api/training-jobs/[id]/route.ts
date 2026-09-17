import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { spawn } from "child_process";
import { exec } from "child_process";
import { logActivity } from "@/lib/activity-log";
import { getCurrentUser, notFoundOrDenied, requireOwnedScope } from "@/lib/auth";
import { frameworkMeta, getWorkDir, isTorch, resolvePythonPath, tracksIterations } from "@/lib/frameworks";
import * as path from "path";
import { existsSync } from "fs";
import {
  createParserState,
  feed as feedParser,
  flush as flushParser,
  disposeParserState,
  type JobParserState,
  type ParsedTrainLog,
} from "@/lib/log-parsers";
import { promisify } from "util";
import { StringDecoder } from "string_decoder";

const execAsync = promisify(exec);

/**
 * Verify the Python interpreter selected for a job actually has the required
 * framework package installed. Prevents jobs from failing deep inside
 * `tools/train.py` with a cryptic `ModuleNotFoundError: No module named
 * 'paddleseg'` — instead we fail-fast with an actionable message.
 *
 * This matters most for the torch frameworks: a Paddle environment has no
 * `torch` and vice versa, so pointing a TorchSeg job at the default per-GPU
 * (Paddle) interpreter is an easy mistake with an unhelpful failure mode.
 *
 * Returns `null` when the env is ready, or a user-facing error string.
 */
async function checkFrameworkModuleAvailable(
  pythonPath: string,
  framework: string,
): Promise<string | null> {
  const meta = frameworkMeta(framework);
  const moduleName = meta.pythonModule;
  try {
    // `find_spec` locates the package without importing it (fast + side-effect
    // free). If it returns None, argparse-level failure is guaranteed.
    const script = `import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('${moduleName}') else 2)`;
    await execAsync(`"${pythonPath}" -c "${script}"`, { timeout: 15000 });
    return null;
  } catch (err: any) {
    // Exit code 2 = module not found; anything else = interpreter couldn't run.
    if (err?.code === 2) {
      return (
        `Python environment at "${pythonPath}" does not have the ${framework} ` +
        `package installed (import "${moduleName}" failed). ${meta.installHint}` +
        (meta.family === "torch"
          ? `  Tip: set a per-framework interpreter under Settings → "Framework Python environments" ` +
            `so ${framework} jobs do not use the PaddlePaddle environment.`
          : "")
      );
    }
    return (
      `Failed to probe Python env at "${pythonPath}" for ${framework} support: ` +
      (err instanceof Error ? err.message : String(err))
    );
  }
}

// Store running processes
const runningProcesses = new Map<string, ReturnType<typeof spawn>>();

/**
 * Jobs the user asked to stop, so the `close` handler can tell "killed on
 * purpose" from "crashed".
 *
 * Without this the stop path was self-defeating: `PUT {status:"stopped"}` wrote
 * `stopped`, then `taskkill` made the child exit non-zero, and the `close`
 * handler unconditionally overwrote the row with
 * `failed` + "Training failed with exit code 1". Every manually stopped job
 * ended up looking like a crash.
 */
const stoppingJobs = new Set<string>();

// Per-job log parser state. Mirrors the lifetime of `runningProcesses`
// entries (created when a process is spawned, deleted on close/error).
// Owning line-buffering + framework-aware dispatch here means the stdout
// listener can be a thin adapter that just forwards ParsedTrainLog records
// into the DB, keeping the routing logic in one place.
const jobParserStates = new Map<string, JobParserState>();

/** Primary GPU index a job was configured with. */
function primaryGpuOf(job: any): { gpuIdsStr: string; primaryGpuId: number } {
  let trainingParams: Record<string, unknown> = {};
  try {
    trainingParams = job?.trainingParams ? JSON.parse(job.trainingParams as string) : {};
  } catch {
    // A malformed params blob should not stop a job from starting on GPU 0.
  }
  const gpuIdsStr = (trainingParams.gpuIds as string) || '0';
  const gpuIds = gpuIdsStr.split(',').map((id) => parseInt(id.trim(), 10)).filter((id) => !isNaN(id));
  return { gpuIdsStr, primaryGpuId: gpuIds[0] || 0 };
}

/**
 * Python interpreter for a job, for display purposes (GET responses).
 *
 * Falls back to a bare `python` so the UI has something to show; the start path
 * below refuses to run without an explicit mapping instead.
 */
async function getPythonPathForJob(job: any): Promise<string> {
  const systemConfig = await db.systemConfig.findFirst();
  const { primaryGpuId } = primaryGpuOf(job);
  const framework = job?.project?.framework;
  const { pythonPath } = resolvePythonPath(framework, primaryGpuId, systemConfig);
  return pythonPath || 'python';
}

// GET /api/training-jobs/[id] - Get a single training job
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    const scope = await requireOwnedScope(request, id);
    if (scope instanceof NextResponse) return scope;

    const job = await db.trainingJob.findFirst({
      where: scope.where,
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

    if (!job) return notFoundOrDenied("Training job");

    // Get Python path for this job
    const pythonPath = await getPythonPathForJob(job);

    // Resolve the stored (possibly relative) config path against
    // `userConfigsPath`, the same way the list endpoint does. Without this a
    // caller working from a single job has only a relative path, and handing it
    // to `val.py`/`predict.py` (which run with cwd = the framework repo) fails
    // with "Config file not found".
    const systemConfig = await db.systemConfig.findFirst();
    const userConfigsPath = (systemConfig as any)?.userConfigsPath;
    const absoluteConfigPath =
      userConfigsPath && job.configPath && !path.isAbsolute(job.configPath)
        ? path.join(userConfigsPath, job.configPath)
        : job.configPath;

    return NextResponse.json({ ...job, pythonPath, absoluteConfigPath });
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

    const scope = await requireOwnedScope(request, id);
    if (scope instanceof NextResponse) return scope;

    const body = await request.json();

    // Check if job exists and belongs to the caller
    const existingJob = await db.trainingJob.findFirst({
      where: scope.where,
      include: {
        project: { select: { framework: true } },
      },
    });

    if (!existingJob) return notFoundOrDenied("Training job");

    // Parse training params for GPU info
    let trainingParams: Record<string, unknown> = {};
    try {
      trainingParams = existingJob.trainingParams
        ? JSON.parse(existingJob.trainingParams as string)
        : {};
    } catch {
      // Ignore parse errors
    }

    // Build update data object with only provided fields
    const updateData: Record<string, unknown> = {};

    // Basic fields.
    //
    // `command` is deliberately NOT accepted from the request body. It is
    // executed further down via `spawn(..., { shell: true })`, so honouring a
    // caller-supplied value turned this endpoint into arbitrary command
    // execution. The command is generated server-side at job creation and is
    // the only thing we will ever run.
    if (typeof body.name === "string" && body.name.trim()) {
      updateData.name = body.name.trim();
    }
    if (body.status !== undefined) {
      const allowedStatuses = ["pending", "running", "completed", "failed", "stopped"];
      if (!allowedStatuses.includes(body.status)) {
        return NextResponse.json(
          { error: `Invalid status. Expected one of: ${allowedStatuses.join(", ")}` },
          { status: 400 }
        );
      }
      updateData.status = body.status;
    }

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
      // Clear the previous run's failure. Without this a job that failed once
      // kept its old `errorMessage` forever, so the UI showed
      // "Training failed with exit code 1" next to a job that had since
      // completed successfully.
      updateData.errorMessage = null;
      updateData.completedAt = null;
      stoppingJobs.delete(id);
      
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

      const jobFramework = existingJob.project?.framework || "PaddleDetection";
      const gpuIdsStr = (trainingParams.gpuIds as string) || '0';
      const gpuIds = gpuIdsStr.split(',').map(id => parseInt(id.trim(), 10)).filter(id => !isNaN(id));
      const primaryGpuId = gpuIds[0] || 0;

      // Resolve the interpreter framework-first, then per-GPU. A torch framework
      // cannot run in a PaddlePaddle env (and vice versa), so the per-framework
      // mapping has to win over the historical per-GPU one.
      const { pythonPath, source: pythonSource } = resolvePythonPath(
        jobFramework,
        primaryGpuId,
        systemConfig,
      );
      if (pythonPath) {
        console.log(`[Job ${id}] Python from ${pythonSource}: ${pythonPath}`);
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
        console.error(`[Job ${id}] No Python path configured for ${jobFramework} / GPU ${primaryGpuId}`);
        updateData.status = "failed";
        updateData.errorMessage = isTorch(jobFramework)
          ? `No Python environment configured for ${jobFramework}. Add a "Framework Python environments" ` +
            `entry in Settings pointing at an environment that has PyTorch installed ` +
            `(the per-GPU mapping normally points at a PaddlePaddle environment, which cannot run torch jobs).`
          : `No Python environment configured for GPU ${primaryGpuId}. Please configure GPU Python mapping in Settings.`;
        updateData.completedAt = new Date();
      } else {
        const framework = jobFramework;
        // Route to the correct framework working directory. Previously this
        // ternary only handled PaddleClas vs PaddleDetection and silently fell
        // through to `paddleDetectionPath` for PaddleSeg jobs, which meant the
        // PaddleSeg command (`tools/train.py --config … --do_eval --save_dir …`)
        // was executed against PaddleDetection's train.py — that script does
        // not accept `--do_eval`/`--save_dir` and fails with
        //   "train.py: error: unrecognized arguments: --do_eval --save_dir …".
        // `getWorkDir` is the shared source of truth (see @/lib/frameworks).
        const workDir = getWorkDir(framework, systemConfig);

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
          // Preflight: does the selected Python env actually have the
          // framework package installed? Fails fast with an actionable
          // message instead of letting `tools/train.py` blow up with
          // `ModuleNotFoundError` deep in the child process.
          const moduleError = await checkFrameworkModuleAvailable(pythonPath, framework);
          if (moduleError) {
            console.error(`[Job ${id}] Preflight failed: ${moduleError}`);
            updateData.status = "failed";
            updateData.errorMessage = moduleError;
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
                systemConfig.condaPath || null,
                framework,
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
      
      // Record the intent *before* killing, so the `close` handler that fires
      // moments later reports "stopped" instead of "failed".
      stoppingJobs.add(id);

      const process = runningProcesses.get(id);
      if (process) {
        if (process?.pid) {
          killProcessTree(process.pid);
        }
        runningProcesses.delete(id);
      } else {
        // No handle: this server process did not spawn the job (it was
        // restarted since). The row can still be marked stopped, but the OS
        // process — if any — is outside our reach, so say so rather than
        // implying we killed it.
        console.warn(
          `[Job ${id}] Stop requested but no child process is tracked in this ` +
            `server instance. Marking the job stopped; if a training process is ` +
            `still running it must be killed manually.`,
        );
        updateData.errorMessage =
          "Marked as stopped, but this server instance had no handle on the training " +
          "process (the server was restarted since the job started). Verify with " +
          "nvidia-smi that no orphaned process is still using the GPU.";
      }
      const stopParser = jobParserStates.get(id);
      if (stopParser) {
        disposeParserState(stopParser);
        jobParserStates.delete(id);
      }
      updateData.completedAt = new Date();
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

    const scope = await requireOwnedScope(request, id);
    if (scope instanceof NextResponse) return scope;

    // Check if job exists and belongs to the caller
    const existingJob = await db.trainingJob.findFirst({
      where: scope.where,
    });

    if (!existingJob) return notFoundOrDenied("Training job");

    // Kill running process if exists.
    // `process.kill()` only signals the launcher; `paddle.distributed.launch`
    // forks worker processes that survive it and keep holding the GPU. Use the
    // same whole-tree kill the stop path uses.
    const child = runningProcesses.get(id);
    if (child) {
      if (child.pid) killProcessTree(child.pid);
      else child.kill();
      runningProcesses.delete(id);
    }
    const staleParser = jobParserStates.get(id);
    if (staleParser) {
      disposeParserState(staleParser);
      jobParserStates.delete(id);
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

/**
 * Kill a process and all of its descendants.
 *
 * `paddle.distributed.launch` forks one worker per GPU, so signalling only the
 * launcher leaves workers running and holding GPU memory. On Windows `taskkill
 * /T` walks the tree; on POSIX we signal the process *group*, which the child
 * gets because it is spawned with `detached: true`.
 */
function killProcessTree(pid: number) {
  if (process.platform === "win32") {
    exec(`taskkill /PID ${pid} /T /F`, (err) => {
      if (err) console.error(`[killProcessTree] taskkill failed for ${pid}:`, err);
    });
    return;
  }
  try {
    // Negative pid targets the whole process group.
    process.kill(-pid, "SIGTERM");
    setTimeout(() => {
      try {
        process.kill(-pid, "SIGKILL");
      } catch {
        // Already gone.
      }
    }, 5000);
  } catch (err) {
    console.error(`[killProcessTree] failed to signal group ${pid}:`, err);
    try {
      process.kill(pid, "SIGKILL");
    } catch {
      // Already gone.
    }
  }
}

// Start training process
function startTrainingProcess(
  jobId: string, 
  command: string, 
  workDir: string, 
  pythonPath: string, 
  gpuIds: string = '0',
  condaEnv: string | null = null,
  condaPath: string | null = null,
  framework: string = 'PaddleDetection',
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
  
  // Torch uses the first visible GPU; Paddle's launcher receives physical IDs via --gpus.
  const env: NodeJS.ProcessEnv = {
    ...process.env,
    PYTHONUNBUFFERED: "1",
    // Force UTF-8 on the child's stdio. When Python's stdout is a pipe it falls
    // back to the locale encoding, which on a Chinese Windows install is GBK —
    // and we decode these bytes as UTF-8 below. The result was mojibake in
    // `TrainingLog.rawLog` and, worse, in `TrainingJob.errorMessage`: the one
    // thing a user reads when a run dies came out as "ѵ����ʼ".
    PYTHONIOENCODING: "utf-8",
  };
  if (isTorch(framework)) {
    env.CUDA_VISIBLE_DEVICES = gpuIds;
  } else {
    delete env.CUDA_VISIBLE_DEVICES;
  }
  
  // Detect if python path is in a conda environment
  const condaWasConfigured = Boolean(condaEnv);
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

    if (!condaWasConfigured && detectedCondaEnv && (!detectedCondaPath || !existsSync(detectedCondaPath))) {
      console.log(`[Job ${jobId}] Ignoring conda-like env path because conda executable was not found: ${detectedCondaPath || 'not detected'}`);
      detectedCondaEnv = null;
      detectedCondaPath = null;
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
    // POSIX: put the child in its own process group so `killProcessTree` can
    // signal the whole group (paddle.distributed.launch forks GPU workers).
    // No effect on Windows, where taskkill /T handles the tree.
    detached: process.platform !== "win32",
  });

  runningProcesses.set(jobId, childProcess);

  // Framework-aware parser: owns line buffering + (Seg-only) multi-line EVAL
  // accumulation. See `@/lib/log-parsers` for the dispatcher design.
  const parserState = createParserState(framework);
  jobParserStates.set(jobId, parserState);

  // `Buffer.toString()` on a chunk that ends mid-character replaces the split
  // bytes with U+FFFD. A StringDecoder holds the incomplete sequence back until
  // the next chunk completes it, which matters as soon as the log contains any
  // non-ASCII text (Paddle prints Chinese warnings on a zh-CN install).
  const stdoutDecoder = new StringDecoder("utf8");
  const stderrDecoder = new StringDecoder("utf8");

  childProcess.stdout?.on("data", async (data: Buffer) => {
    const output = stdoutDecoder.write(data);
    if (!output) return;
    console.log(`[Job ${jobId}] ${output}`);

    // Split chunk into complete lines and dispatch each to the appropriate
    // per-framework parser. Zero, one, or many rows may come back per chunk.
    const rows = feedParser(parserState, output);
    for (const row of rows) {
      await writeParsedLog(jobId, row, framework);
    }
  });

  childProcess.stderr?.on("data", (data: Buffer) => {
    const stderrOutput = stderrDecoder.write(data);
    if (!stderrOutput) return;
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

    // Drain any trailing line the parser was still buffering, then release
    // the state map entry. Failure here is non-fatal to job closure.
    const tailState = jobParserStates.get(jobId);
    if (tailState) {
      try {
        const tail = flushParser(tailState);
        for (const row of tail) await writeParsedLog(jobId, row, framework);
      } catch (e) {
        console.error(`[Job ${jobId}] parser flush failed:`, e);
      }
      disposeParserState(tailState);
      jobParserStates.delete(jobId);
    }

    // Update job status.
    //
    // A job the user stopped exits non-zero because we killed it, so exit code
    // alone cannot distinguish "stopped" from "failed" — `stoppingJobs` carries
    // that intent. Reporting a deliberate stop as a crash (with a stderr dump
    // as the "error") is misleading and used to happen every single time.
    const wasStopped = stoppingJobs.delete(jobId);
    const status = wasStopped ? "stopped" : code === 0 ? "completed" : "failed";
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
      } else {
        // A completed (or cleanly stopped) run must not keep a stale failure
        // from an earlier attempt.
        updateData.errorMessage = null;
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
    const errState = jobParserStates.get(jobId);
    if (errState) {
      disposeParserState(errState);
      jobParserStates.delete(jobId);
    }
    
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

/**
 * Persist one framework-parsed training log record into the DB, updating the
 * `TrainingJob` progress snapshot in the same call.
 *
 * All framework specifics have already been resolved by the parser
 * dispatcher in `@/lib/log-parsers`, so this function is a thin, uniform
 * writer that never inspects log content itself.
 *
 * Failures are swallowed to keep the stdout listener resilient — an unwritten
 * log row is worth much less than a training run that dies because a DB
 * connection blipped.
 */
async function writeParsedLog(
  jobId: string,
  log: ParsedTrainLog,
  framework?: string | null,
): Promise<void> {
  try {
    // 1. Roll the TrainingJob progress snapshot forward. Only touch columns
    //    the parser actually produced so an EVAL row (which has no loss/lr)
    //    doesn't wipe the last known progress.
    const jobUpdate: Record<string, unknown> = {};
    // `currentEpoch` must be in the same unit as `totalEpochs`, which holds
    // *iterations* for every framework whose `stepUnit` is `iter` (segmentation
    // and anomaly — see `totalStepsFor`). The Seg log line carries both
    // (`epoch: 2278, iter: 20500/160000`), and storing the epoch there made the
    // progress bar read 2278/160000 instead of 20500/160000 — i.e. a run that
    // was 13% done displayed as 1%.
    if (tracksIterations(framework)) {
      if (log.iteration) jobUpdate.currentEpoch = log.iteration;
    } else if (log.epoch) {
      jobUpdate.currentEpoch = log.epoch;
    }
    if (log.loss !== null && log.loss !== undefined) jobUpdate.currentLoss = log.loss;
    if (log.learningRate !== null && log.learningRate !== undefined) {
      jobUpdate.currentLr = log.learningRate;
    }
    if (log.bestIter !== null && log.bestIter !== undefined) {
      jobUpdate.bestIter = log.bestIter;
    }
    if (log.bestMetric !== null && log.bestMetric !== undefined) {
      jobUpdate.bestMetric = log.bestMetric;
    }
    if (Object.keys(jobUpdate).length > 0) {
      await db.trainingJob.update({ where: { id: jobId }, data: jobUpdate });
    }

    // 2. Insert the detailed TrainingLog row. Per-class arrays are folded
    //    into a JSON blob so num_classes doesn't leak into the schema. The same
    //    column carries any anomaly metric that has no dedicated column (see
    //    `ParsedTrainLog.extraMetrics`), which is how a newly added anomalib
    //    metric survives instead of being silently dropped.
    const classMetrics =
      log.classIoU || log.classPrecision || log.classRecall
        ? JSON.stringify({
            iou: log.classIoU ?? null,
            precision: log.classPrecision ?? null,
            recall: log.classRecall ?? null,
          })
        : log.extraMetrics
          ? JSON.stringify({ metrics: log.extraMetrics })
          : null;

    await db.trainingLog.create({
      data: {
        jobId,
        epoch: log.epoch,
        iteration: log.iteration,
        totalIter: log.totalIter,
        loss: log.loss,
        lossCls: log.lossCls ?? null,
        lossIou: log.lossIou ?? null,
        lossDfl: log.lossDfl ?? null,
        lossL1: log.lossL1 ?? null,
        learningRate: log.learningRate,
        mIoU: log.mIoU ?? null,
        acc: log.acc ?? null,
        kappa: log.kappa ?? null,
        dice: log.dice ?? null,
        mAP: log.mAP ?? null,
        mAP50: log.mAP50 ?? null,
        imageAuroc: log.imageAuroc ?? null,
        imageF1: log.imageF1 ?? null,
        pixelAuroc: log.pixelAuroc ?? null,
        pixelF1: log.pixelF1 ?? null,
        threshold: log.threshold ?? null,
        eta: log.eta,
        batchCost: log.batchCost,
        dataCost: log.dataCost,
        readerCost: log.readerCost,
        ips: log.ips,
        memReserved: log.memReserved,
        memAllocated: log.memAllocated,
        classMetrics,
        rawLog: log.rawLog.slice(0, 2000),
      },
    });
  } catch (error) {
    // Non-fatal — one lost log line shouldn't kill the training run.
    console.error(`[Job ${jobId}] writeParsedLog failed:`, error);
  }
}
