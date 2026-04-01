import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { requireAuth } from "@/lib/auth";
import * as fs from "fs";
import * as path from "path";
import * as yaml from "yaml";

// Helper function to parse YAML and extract training parameters
function parseTrainingYaml(yamlContent: string) {
  try {
    const parsed = yaml.parse(yamlContent);
    const params: Record<string, unknown> = {};

    // Basic training params
    if (parsed.epoch !== undefined) params.epochs = parseInt(parsed.epoch);

    // Learning rate
    if (parsed.LearningRate?.base_lr !== undefined) {
      params.baseLr = parseFloat(parsed.LearningRate.base_lr);
    }

    // Scheduler params
    if (parsed.LearningRate?.schedulers && Array.isArray(parsed.LearningRate.schedulers)) {
      for (const scheduler of parsed.LearningRate.schedulers) {
        if (scheduler.name) {
          params.scheduler = scheduler.name;
        }
        if (scheduler.max_epochs !== undefined) {
          params.maxEpochs = parseInt(scheduler.max_epochs);
        }
        if (scheduler.epochs !== undefined && scheduler.name === 'LinearWarmup') {
          params.warmupEpochs = parseInt(scheduler.epochs);
        }
      }
    }

    // Optimizer params
    if (parsed.OptimizerBuilder?.optimizer) {
      if (parsed.OptimizerBuilder.optimizer.momentum !== undefined) {
        params.momentum = parseFloat(parsed.OptimizerBuilder.optimizer.momentum);
      }
    }
    if (parsed.OptimizerBuilder?.regularizer?.factor !== undefined) {
      params.weightDecay = parseFloat(parsed.OptimizerBuilder.regularizer.factor);
    }

    // Reader settings
    if (parsed.worker_num !== undefined) params.workerNum = parseInt(parsed.worker_num);
    if (parsed.eval_height !== undefined) params.imageHeight = parseInt(parsed.eval_height);
    if (parsed.eval_width !== undefined) params.imageWidth = parseInt(parsed.eval_width);

    // TrainReader batch_size
    if (parsed.TrainReader?.batch_size !== undefined) {
      params.batchSize = parseInt(parsed.TrainReader.batch_size);
    }

    // Runtime settings
    if (parsed.use_gpu !== undefined) params.useGpu = parsed.use_gpu === true || parsed.use_gpu === 'true';
    if (parsed.log_iter !== undefined) params.logIter = parseInt(parsed.log_iter);
    if (parsed.save_dir !== undefined) params.saveDir = parsed.save_dir;
    if (parsed.snapshot_epoch !== undefined) params.snapshotEpoch = parseInt(parsed.snapshot_epoch);

    // Output paths
    if (parsed.output_dir !== undefined) params.outputDir = parsed.output_dir;
    if (parsed.weights !== undefined) params.weights = parsed.weights;

    // Pretrained weights
    if (parsed.pretrain_weights !== undefined) params.pretrainWeights = parsed.pretrain_weights;

    return params;
  } catch (error) {
    console.error('Error parsing YAML:', error);
    return {};
  }
}

// GET /api/training-configs/import - List available training configs (filtered by user access)
export async function GET(request: NextRequest) {
  try {
    // Check authentication
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;

    const { userId } = auth;

    const { searchParams } = new URL(request.url);
    const projectId = searchParams.get("projectId");

    if (!projectId) {
      return NextResponse.json(
        { error: "Project ID is required" },
        { status: 400 }
      );
    }

    // Get project and verify user access
    const project = await db.project.findFirst({
      where: { 
        id: projectId,
        userId: userId,
      },
    });

    if (!project) {
      return NextResponse.json(
        { error: "Project not found or access denied" },
        { status: 404 }
      );
    }

    // Get system config for paths
    const systemConfig = await db.systemConfig.findFirst();
    const framework = project.framework || "PaddleDetection";
    const workDir = framework === "PaddleClas"
      ? systemConfig?.paddleClasPath
      : systemConfig?.paddleDetectionPath;

    if (!workDir) {
      return NextResponse.json(
        { error: `Please configure ${framework} path in Settings` },
        { status: 400 }
      );
    }

    // List configs from autotrain/training/default folder
    const defaultConfigDir = path.join(workDir, "configs", "autotrain", "training", "default");
    const defaultConfigs: Array<{ name: string; path: string; content: string }> = [];

    if (fs.existsSync(defaultConfigDir)) {
      const files = fs.readdirSync(defaultConfigDir).filter(f => f.endsWith(".yml") || f.endsWith(".yaml"));
      
      for (const file of files) {
        const filePath = path.join(defaultConfigDir, file);
        const content = fs.readFileSync(filePath, "utf-8");
        defaultConfigs.push({
          name: file.replace(/\.(yml|yaml)$/, ""),
          path: `configs/autotrain/training/default/${file}`,
          content: content,
        });
      }
    }

    // Also list user configs from autotrain/training/user folder
    const userConfigDir = path.join(workDir, "configs", "autotrain", "training", "user");
    const userConfigs: Array<{ name: string; path: string; content: string }> = [];

    if (fs.existsSync(userConfigDir)) {
      const files = fs.readdirSync(userConfigDir).filter(f => f.endsWith(".yml") || f.endsWith(".yaml"));
      
      for (const file of files) {
        const filePath = path.join(userConfigDir, file);
        const content = fs.readFileSync(filePath, "utf-8");
        userConfigs.push({
          name: file.replace(/\.(yml|yaml)$/, ""),
          path: `configs/autotrain/training/user/${file}`,
          content: content,
        });
      }
    }

    return NextResponse.json({
      success: true,
      data: {
        defaultConfigs,
        userConfigs,
        workDir,
      },
    });
  } catch (error) {
    console.error("Error listing training configs:", error);
    return NextResponse.json(
      { error: "Failed to list training configs", message: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

// POST /api/training-configs/import - Import and save training config
export async function POST(request: NextRequest) {
  try {
    // Check authentication
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;

    const { userId } = auth;

    const body = await request.json();
    const { projectId, name, description, yamlContent, isDefault, configPath, trainingParams } = body;

    if (!projectId || !name) {
      return NextResponse.json(
        { error: "Project ID and config name are required" },
        { status: 400 }
      );
    }

    // Get project and verify user access
    const project = await db.project.findFirst({
      where: { 
        id: projectId,
        userId: userId,
      },
    });

    if (!project) {
      return NextResponse.json(
        { error: "Project not found or access denied" },
        { status: 404 }
      );
    }

    // Get system config for paths
    const systemConfig = await db.systemConfig.findFirst();
    const framework = project.framework || "PaddleDetection";
    const workDir = framework === "PaddleClas"
      ? systemConfig?.paddleClasPath
      : systemConfig?.paddleDetectionPath;

    if (!workDir) {
      return NextResponse.json(
        { error: `Please configure ${framework} path in Settings` },
        { status: 400 }
      );
    }

    let savedConfigPath = configPath;
    let finalYamlContent = yamlContent;

    // If importing default config, read content from file if not provided
    if (isDefault && configPath && !yamlContent) {
      const fullPath = path.join(workDir, configPath);
      if (fs.existsSync(fullPath)) {
        finalYamlContent = fs.readFileSync(fullPath, "utf-8");
      }
    }

    // Parse YAML content to extract parameters
    const parsedParams = finalYamlContent ? parseTrainingYaml(finalYamlContent) : {};
    const finalParams = { ...parsedParams, ...trainingParams };

    // If importing default config, use existing path but still create db record
    if (isDefault) {
      // Create training config in database with userId (no file save for default configs)
      const config = await db.trainingConfig.create({
        data: {
          projectId: projectId,
          userId: userId,
          name: name,
          epoch: finalParams.epochs || 100,
          batchSize: finalParams.batchSize || 8,
          baseLr: finalParams.baseLr || 0.001,
          momentum: finalParams.momentum || 0.9,
          weightDecay: finalParams.weightDecay || 0.0005,
          scheduler: finalParams.scheduler || 'CosineDecay',
          warmupEpochs: finalParams.warmupEpochs || 5,
          maxEpochs: finalParams.maxEpochs || 100,
          workerNum: finalParams.workerNum || 4,
          evalHeight: finalParams.imageHeight || 640,
          evalWidth: finalParams.imageWidth || 640,
          useGpu: finalParams.useGpu !== undefined ? finalParams.useGpu : true,
          logIter: finalParams.logIter || 20,
          snapshotEpoch: finalParams.snapshotEpoch || 1,
          saveDir: finalParams.saveDir || null,
          outputDir: finalParams.outputDir || null,
          weights: finalParams.weights || null,
          pretrainWeights: finalParams.pretrainWeights || null,
          yamlConfig: finalYamlContent || null,
        },
      });

      return NextResponse.json({
        success: true,
        data: {
          config,
          configPath: savedConfigPath,
        },
      });
    }

    // For user configs or new configs, save to autotrain/training/user folder
    if (yamlContent) {
      const userConfigDir = path.join(workDir, "configs", "autotrain", "training", "user");
      
      // Create directory if not exists
      if (!fs.existsSync(userConfigDir)) {
        fs.mkdirSync(userConfigDir, { recursive: true });
      }

      // Generate filename
      const fileName = `${name.replace(/\s+/g, '_').replace(/[^a-zA-Z0-9_.-]/g, '').toLowerCase()}.yml`;
      const filePath = path.join(userConfigDir, fileName);
      
      // Save YAML file
      fs.writeFileSync(filePath, yamlContent, "utf-8");
      savedConfigPath = `configs/autotrain/training/user/${fileName}`;
    }

    // Create training config in database with userId
    const config = await db.trainingConfig.create({
      data: {
        projectId: projectId,
        userId: userId,
        name: name,
        epoch: finalParams.epochs || 100,
        batchSize: finalParams.batchSize || 8,
        baseLr: finalParams.baseLr || 0.001,
        momentum: finalParams.momentum || 0.9,
        weightDecay: finalParams.weightDecay || 0.0005,
        scheduler: finalParams.scheduler || 'CosineDecay',
        warmupEpochs: finalParams.warmupEpochs || 5,
        maxEpochs: finalParams.maxEpochs || 100,
        workerNum: finalParams.workerNum || 4,
        evalHeight: finalParams.imageHeight || 640,
        evalWidth: finalParams.imageWidth || 640,
        useGpu: finalParams.useGpu !== undefined ? finalParams.useGpu : true,
        logIter: finalParams.logIter || 20,
        snapshotEpoch: finalParams.snapshotEpoch || 1,
        saveDir: finalParams.saveDir || null,
        outputDir: finalParams.outputDir || null,
        weights: finalParams.weights || null,
        pretrainWeights: finalParams.pretrainWeights || null,
        yamlConfig: finalYamlContent || null,
      },
    });

    return NextResponse.json({
      success: true,
      data: {
        config,
        configPath: savedConfigPath,
      },
    });
  } catch (error) {
    console.error("Error importing training config:", error);
    return NextResponse.json(
      { error: "Failed to import training config", message: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
