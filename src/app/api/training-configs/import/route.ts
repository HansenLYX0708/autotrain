import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import * as fs from "fs";
import * as path from "path";

// GET /api/training-configs/import - List available training configs
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const projectId = searchParams.get("projectId");

    if (!projectId) {
      return NextResponse.json(
        { error: "Project ID is required" },
        { status: 400 }
      );
    }

    // Get project to determine framework
    const project = await db.project.findUnique({
      where: { id: projectId },
    });

    if (!project) {
      return NextResponse.json(
        { error: "Project not found" },
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
    const body = await request.json();
    const { projectId, name, description, yamlContent, isDefault, configPath, trainingParams } = body;

    if (!projectId || !name) {
      return NextResponse.json(
        { error: "Project ID and config name are required" },
        { status: 400 }
      );
    }

    // Get project to determine framework
    const project = await db.project.findUnique({
      where: { id: projectId },
    });

    if (!project) {
      return NextResponse.json(
        { error: "Project not found" },
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

    // If importing default config, use existing path but still create db record
    if (isDefault) {
      // Create training config in database (no file save for default configs)
      const config = await db.trainingConfig.create({
        data: {
          name: name,
          epoch: trainingParams?.epochs || 100,
          batchSize: trainingParams?.batchSize || 8,
          baseLr: trainingParams?.baseLr || 0.001,
          momentum: trainingParams?.momentum || 0.9,
          weightDecay: trainingParams?.weightDecay || 0.0005,
          scheduler: trainingParams?.scheduler || 'CosineDecay',
          warmupEpochs: trainingParams?.warmupEpochs || 5,
          maxEpochs: trainingParams?.maxEpochs || 100,
          workerNum: trainingParams?.workerNum || 4,
          evalHeight: trainingParams?.imageHeight || 640,
          evalWidth: trainingParams?.imageWidth || 640,
          snapshotEpoch: trainingParams?.snapshotEpoch || 1,
          saveDir: trainingParams?.saveDir || null,
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

    // Create training config in database with provided params
    const config = await db.trainingConfig.create({
      data: {
        name: name,
        epoch: trainingParams?.epochs || 100,
        batchSize: trainingParams?.batchSize || 8,
        baseLr: trainingParams?.baseLr || 0.001,
        momentum: trainingParams?.momentum || 0.9,
        weightDecay: trainingParams?.weightDecay || 0.0005,
        scheduler: trainingParams?.scheduler || 'CosineDecay',
        warmupEpochs: trainingParams?.warmupEpochs || 5,
        maxEpochs: trainingParams?.maxEpochs || 100,
        workerNum: trainingParams?.workerNum || 4,
        evalHeight: trainingParams?.imageHeight || 640,
        evalWidth: trainingParams?.imageWidth || 640,
        snapshotEpoch: trainingParams?.snapshotEpoch || 1,
        saveDir: trainingParams?.saveDir || null,
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
