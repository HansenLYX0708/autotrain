import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import * as fs from "fs";
import * as path from "path";

// GET /api/training-jobs - Get all training jobs with relations
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const page = parseInt(searchParams.get("page") || "1");
    const limit = parseInt(searchParams.get("limit") || "5"); // Default to 5 records
    const skip = (page - 1) * limit;
    const status = searchParams.get("status");
    const projectId = searchParams.get("projectId");
    const datasetId = searchParams.get("datasetId");
    const modelId = searchParams.get("modelId");
    const configId = searchParams.get("configId");

    const where: Record<string, unknown> = {};
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
    const body = await request.json();

    // Validate required relations
    const [project, dataset, model, trainingConfig] = await Promise.all([
      db.project.findUnique({ where: { id: body.projectId } }),
      db.dataset.findUnique({ where: { id: body.datasetId } }),
      db.model.findUnique({ where: { id: body.modelId } }),
      body.configId ? db.trainingConfig.findUnique({ 
        where: { id: body.configId },
        select: {
          id: true,
          name: true,
          epoch: true,
          batchSize: true,
          baseLr: true,
          yamlConfig: true,
        }
      }) : null,
    ]);

    if (!project) {
      return NextResponse.json({ error: "Project not found" }, { status: 400 });
    }
    if (!dataset) {
      return NextResponse.json({ error: "Dataset not found" }, { status: 400 });
    }
    if (!model) {
      return NextResponse.json({ error: "Model not found" }, { status: 400 });
    }

    // Get system config for PaddleDetection path
    const systemConfig = await db.systemConfig.findFirst();
    const workDir = systemConfig?.paddleDetectionPath;

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
    
    const mergedYaml = yamlParts.join('\n\n');

    // Save to configs/autotrain/jobs folder
    const jobsConfigDir = path.join(workDir, 'configs', 'autotrain', 'jobs');
    if (!fs.existsSync(jobsConfigDir)) {
      fs.mkdirSync(jobsConfigDir, { recursive: true });
    }

    const configFilePath = path.join(jobsConfigDir, configFileName);
    fs.writeFileSync(configFilePath, mergedYaml, 'utf-8');

    // Relative path for command
    const configPath = `configs/autotrain/jobs/${configFileName}`;

    // Generate training command
    // Note: CUDA_VISIBLE_DEVICES is set via environment variable during execution for cross-platform compatibility
    const gpuIds = body.gpuIds || '0';
    const useAmp = body.useAmp || false;
    const useVdl = body.useVdl || false;

    let command = '';
    if (gpuIds.includes(',')) {
      // Multi-GPU: use distributed launch
      command = `python -m paddle.distributed.launch --gpus ${gpuIds} tools/train.py -c ${configPath}`;
    } else {
      // Single GPU: CUDA_VISIBLE_DEVICES will be set via environment variable
      command = `python tools/train.py -c ${configPath}`;
    }
    if (useAmp) command += ' --amp';
    if (useVdl) {
      command += ` --use_vdl=true --vdl_log_dir=output/${project.name}/${jobName}/vdl`;
    }

    // Generate eval command
    const evalCommand = `python tools/eval.py -c ${configPath} -o weights=output/${project.name}/${jobName}/model_final.pdparams`;

    // Generate infer command (for single image inference)
    const inferCommand = `python tools/infer.py -c ${configPath} -o weights=output/${project.name}/${jobName}/model_final.pdparams`;

    // Create job in database
    const job = await db.trainingJob.create({
      data: {
        projectId: body.projectId,
        datasetId: body.datasetId,
        modelId: body.modelId,
        configId: body.configId || null,
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
