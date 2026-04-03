import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { requireAuth } from "@/lib/auth";
import * as fs from "fs";
import * as path from "path";

// GET /api/models/import - List available model configs (filtered by user access)
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

    // Get current user info including username
    const currentUser = await db.user.findUnique({
      where: { id: userId },
      select: { id: true, username: true, role: true },
    });

    if (!currentUser) {
      return NextResponse.json(
        { error: "User not found" },
        { status: 404 }
      );
    }

    // Get system config for paths
    const systemConfig = await db.systemConfig.findFirst();
    const framework = project.framework || "PaddleDetection";
    const workDir = framework === "PaddleClas"
      ? systemConfig?.paddleClasPath
      : systemConfig?.paddleDetectionPath;
    const userConfigsPath = (systemConfig as any)?.userConfigsPath;

    // List configs from userConfigsPath/default/models folder
    let defaultConfigDir = "";
    if (userConfigsPath) {
      defaultConfigDir = path.join(userConfigsPath, "default", "models");
    } else if (workDir) {
      // Fallback to old path if userConfigsPath not set
      defaultConfigDir = path.join(workDir, "configs", "autotrain", "models", "default");
    }
    
    const configs: Array<{ name: string; path: string; content: string }> = [];

    if (defaultConfigDir && fs.existsSync(defaultConfigDir)) {
      const files = fs.readdirSync(defaultConfigDir).filter(f => f.endsWith(".yml") || f.endsWith(".yaml"));
      
      for (const file of files) {
        const filePath = path.join(defaultConfigDir, file);
        const content = fs.readFileSync(filePath, "utf-8");
        configs.push({
          name: file.replace(/\.(yml|yaml)$/, ""),
          path: userConfigsPath 
            ? path.join("default", "models", file)
            : `configs/autotrain/models/default/${file}`,
          content: content,
        });
      }
    }

    // Also list user configs from userConfigsPath/{username}/models folder or fallback
    let userConfigDir = "";
    if (userConfigsPath && currentUser.username) {
      userConfigDir = path.join(userConfigsPath, currentUser.username, "models");
    } else if (workDir) {
      userConfigDir = path.join(workDir, "configs", "autotrain", "models", "user");
    }
    
    const userConfigs: Array<{ name: string; path: string; content: string }> = [];

    if (userConfigDir && fs.existsSync(userConfigDir)) {
      const files = fs.readdirSync(userConfigDir).filter(f => f.endsWith(".yml") || f.endsWith(".yaml"));
      
      for (const file of files) {
        const filePath = path.join(userConfigDir, file);
        const content = fs.readFileSync(filePath, "utf-8");
        userConfigs.push({
          name: file.replace(/\.(yml|yaml)$/, ""),
          path: userConfigsPath && currentUser.username
            ? path.join(currentUser.username, "models", file)
            : `configs/autotrain/models/user/${file}`,
          content: content,
        });
      }
    }

    return NextResponse.json({
      success: true,
      data: {
        defaultConfigs: configs,
        userConfigs: userConfigs,
        workDir: workDir,
        userConfigsPath: userConfigsPath,
      },
    });
  } catch (error) {
    console.error("Error listing model configs:", error);
    return NextResponse.json(
      { error: "Failed to list model configs", message: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

// POST /api/models/import - Import and save model config
export async function POST(request: NextRequest) {
  try {
    // Check authentication
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;

    const { userId } = auth;

    const body = await request.json();
    const { projectId, name, description, yamlContent, isDefault, configPath } = body;

    if (!projectId || !name) {
      return NextResponse.json(
        { error: "Project ID and model name are required" },
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

    // If not using default config, save to autotrain/models/user folder
    if (!isDefault && yamlContent) {
      const userConfigDir = path.join(workDir, "configs", "autotrain", "models", "user");
      
      // Create directory if not exists
      if (!fs.existsSync(userConfigDir)) {
        fs.mkdirSync(userConfigDir, { recursive: true });
      }

      // Generate filename
      const fileName = `${name.replace(/\s+/g, '_').replace(/[^a-zA-Z0-9_.-]/g, '').toLowerCase()}.yml`;
      const filePath = path.join(userConfigDir, fileName);
      
      // Save YAML file
      fs.writeFileSync(filePath, yamlContent, "utf-8");
      savedConfigPath = `configs/autotrain/models/user/${fileName}`;
    }

    // Parse YAML content to extract model info
    const parsedInfo = parseYamlConfig(yamlContent || "");

    // Create model in database with userId
    const model = await db.model.create({
      data: {
        name: name,
        description: description || null,
        projectId: projectId,
        userId: userId,
        architecture: parsedInfo.architecture || "YOLOv3",
        backbone: parsedInfo.backbone || "CSPResNet",
        neck: parsedInfo.neck || "CustomCSPPAN",
        head: parsedInfo.head || "PPYOLOEHead",
        numClasses: parsedInfo.numClasses || 1,
        normType: parsedInfo.normType || "sync_bn",
        useEma: parsedInfo.useEma ?? true,
        emaDecay: parsedInfo.emaDecay || 0.9998,
        depthMult: parsedInfo.depthMult || 0.33,
        widthMult: parsedInfo.widthMult || 0.50,
        pretrainWeights: parsedInfo.pretrainWeights || null,
        yamlConfig: yamlContent || null,
      },
    });

    return NextResponse.json({
      success: true,
      data: {
        model: model,
        configPath: savedConfigPath,
      },
    });
  } catch (error) {
    console.error("Error importing model config:", error);
    return NextResponse.json(
      { error: "Failed to import model config", message: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

// Parse YAML config to extract model info
function parseYamlConfig(yamlContent: string): {
  architecture?: string;
  backbone?: string;
  neck?: string;
  head?: string;
  numClasses?: number;
  normType?: string;
  useEma?: boolean;
  emaDecay?: number;
  depthMult?: number;
  widthMult?: number;
  pretrainWeights?: string;
} {
  const result: {
    architecture?: string;
    backbone?: string;
    neck?: string;
    head?: string;
    numClasses?: number;
    normType?: string;
    useEma?: boolean;
    emaDecay?: number;
    depthMult?: number;
    widthMult?: number;
    pretrainWeights?: string;
  } = {};
  
  try {
    // Simple YAML parsing for key fields
    const lines = yamlContent.split('\n');
    
    for (const line of lines) {
      const trimmed = line.trim();
      
      // Skip comments and empty lines
      if (trimmed.startsWith('#') || !trimmed) continue;
      
      // Extract key-value pairs
      const match = trimmed.match(/^(\w+):\s*(.+)$/);
      if (match) {
        const [, key, value] = match;
        
        // Parse different fields
        if (key === 'architecture') result.architecture = value;
        if (key === 'norm_type') result.normType = value;
        if (key === 'use_ema') result.useEma = value.toLowerCase() === 'true';
        if (key === 'ema_decay') result.emaDecay = parseFloat(value);
        if (key === 'depth_mult') result.depthMult = parseFloat(value);
        if (key === 'width_mult') result.widthMult = parseFloat(value);
        if (key === 'num_classes') result.numClasses = parseInt(value);
        if (key === 'pretrain_weights') result.pretrainWeights = value;
      }
      
      // Extract nested values for backbone, neck, head
      const archMatch = trimmed.match(/^(\w+):\s*$/);
      if (archMatch) {
        const archKey = archMatch[1];
        // Look at next lines for backbone, neck, head
        const idx = lines.indexOf(line);
        for (let i = idx + 1; i < lines.length && lines[i].startsWith('  '); i++) {
          const nestedMatch = lines[i].trim().match(/^(\w+):\s*(.+)$/);
          if (nestedMatch) {
            const [, nestedKey, nestedValue] = nestedMatch;
            if (archKey === result.architecture || archKey === 'YOLOv3' || archKey === 'PPYOLOE') {
              if (nestedKey === 'backbone') result.backbone = nestedValue;
              if (nestedKey === 'neck') result.neck = nestedValue;
              if (nestedKey === 'yolo_head') result.head = nestedValue;
            }
          }
        }
      }
    }
  } catch (e) {
    console.error("Error parsing YAML:", e);
  }
  
  return result;
}
