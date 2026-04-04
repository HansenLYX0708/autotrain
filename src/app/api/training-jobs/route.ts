import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { requireAuth, buildUserFilter } from "@/lib/auth";
import * as fs from "fs";
import * as path from "path";

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

    // Get system config for PaddleDetection path and userConfigsPath
    const systemConfig = await db.systemConfig.findFirst();
    const workDir = systemConfig?.paddleDetectionPath;
    const userConfigsPath = (systemConfig as any)?.userConfigsPath;
    const userDatabasePath = (systemConfig as any)?.userDatabasePath;

    // Get current user info for username
    const currentUser = await db.user.findUnique({
      where: { id: userId },
      select: { id: true, username: true },
    });

    if (!currentUser) {
      return NextResponse.json({ error: "User not found" }, { status: 404 });
    }

    if (!workDir) {
      return NextResponse.json(
        { error: "PaddleDetection path not configured in Settings" },
        { status: 400 }
      );
    }

    // Generate job name and file name
    const jobName = body.name.replace(/\s+/g, '_').replace(/[^a-zA-Z0-9_.-]/g, '').toLowerCase();
    const configFileName = `${jobName}.yml`;

    // Concatenate YAMLs in order: dataset -> training config -> model config
    const datasetYaml = dataset.yamlConfig || '';
    const modelYaml = model.yamlConfig || '';
    const trainingYaml = trainingConfig?.yamlConfig || '';
    
    // Merge YAMLs - order matters: dataset first (defines data), then training config, then model config
    const yamlParts: string[] = [];
    
    if (datasetYaml) {
      yamlParts.push(`# Dataset Configuration\n${datasetYaml}`);
    }
    if (trainingYaml) {
      yamlParts.push(`# Training Configuration\n${trainingYaml}`);
    }
    if (modelYaml) {
      yamlParts.push(`# Model Configuration\n${modelYaml}`);
    }
    
    let mergedYaml = yamlParts.join('\n\n');

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
    // Note: CUDA_VISIBLE_DEVICES is set via environment variable during execution for cross-platform compatibility
    const gpuIds = body.gpuIds || '0';
    const useAmp = body.useAmp || false;
    const useVdl = body.useVdl || false;

    let command = '';
    const quotedConfigPath = `"${configFilePath}"`;
    if (gpuIds.includes(',')) {
      // Multi-GPU: use distributed launch
      command = `python -m paddle.distributed.launch --gpus ${gpuIds} tools/train.py -c ${quotedConfigPath}`;
    } else {
      // Single GPU: CUDA_VISIBLE_DEVICES will be set via environment variable
      command = `python tools/train.py -c ${quotedConfigPath}`;
    }
    if (useAmp) command += ' --amp';
    if (useVdl) {
      command += ` --use_vdl=true --vdl_log_dir=output/${project.name}/${jobName}/vdl`;
    }

    // Generate eval command using absolute path
    const evalCommand = `python tools/eval.py -c ${quotedConfigPath} -o weights=output/${project.name}/${jobName}/model_final.pdparams`;

    // Generate infer command (for single image inference) using absolute path
    const inferCommand = `python tools/infer.py -c ${quotedConfigPath} -o weights=output/${project.name}/${jobName}/model_final.pdparams`;

    // Create job in database with userId
    const job = await db.trainingJob.create({
      data: {
        projectId: body.projectId,
        datasetId: body.datasetId,
        modelId: body.modelId,
        configId: body.configId || null,
        userId: userId,
        name: body.name,
        status: 'pending',
        command: command,
        evalCommand: evalCommand,
        inferCommand: inferCommand,
        configPath: configPath,
        totalEpochs: trainingConfig?.epoch || 100,
        outputDir: `output/${project.name}/${jobName}`,
        vdlLogDir: useVdl ? `output/${project.name}/${jobName}/vdl` : null,
        trainingParams: JSON.stringify({
          gpuIds,
          useAmp,
          useVdl,
          epochs: trainingConfig?.epoch,
          batchSize: trainingConfig?.batchSize,
          baseLr: trainingConfig?.baseLr,
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

    return NextResponse.json({ ...job, configPath }, { status: 201 });
  } catch (error) {
    console.error("Error creating training job:", error);
    return NextResponse.json(
      { error: "Failed to create training job", message: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
