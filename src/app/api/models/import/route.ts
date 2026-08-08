import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { requireAuth } from "@/lib/auth";
import { getWorkDir } from "@/lib/frameworks";
import { ensureDefaultConfigs, getDefaultFolderName } from "@/lib/default-configs";
import { asConfigFramework } from "@/lib/training-yaml";
import { defaultModelParams, modelParamsToColumns, parseModelParams } from "@/lib/model-yaml";
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
    const workDir = getWorkDir(framework, systemConfig);
    const userConfigsPath = (systemConfig as any)?.userConfigsPath;

    // Per-framework default folder: `default` for detection, `defaultSeg` for
    // PaddleSeg. First access seeds the folder with built-in starter configs
    // so the dropdown is never empty on a fresh install.
    const defaultFolder = getDefaultFolderName(framework);
    let defaultConfigDir = "";
    if (userConfigsPath) {
      defaultConfigDir = ensureDefaultConfigs(userConfigsPath, framework, "models");
    } else if (workDir) {
      // Fallback to old path if userConfigsPath not set.
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
            ? path.join(defaultFolder, "models", file)
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
    const workDir = getWorkDir(framework, systemConfig);
    const userConfigsPath = (systemConfig as any)?.userConfigsPath;

    // Look up current user's username so we can save into their personal folder.
    const currentUser = await db.user.findUnique({
      where: { id: userId },
      select: { username: true },
    });

    // Resolve the YAML we're going to persist. For default-config imports the
    // request often ships without `yamlContent` (only `configPath`), so read
    // it from disk relative to `userConfigsPath` (new layout) or `workDir`
    // (legacy fallback).
    let finalYamlContent: string | null = yamlContent || null;
    if (!finalYamlContent && configPath) {
      const candidates = [
        userConfigsPath ? path.join(userConfigsPath, configPath) : null,
        workDir ? path.join(workDir, configPath) : null,
      ].filter(Boolean) as string[];
      for (const c of candidates) {
        if (fs.existsSync(c)) {
          finalYamlContent = fs.readFileSync(c, "utf-8");
          break;
        }
      }
    }

    let savedConfigPath = configPath;

    // Always persist a per-user copy under
    //   {userConfigsPath}/{username}/models/{name}.yml
    // so the imported model has its own YAML file (editable, versioned by the
    // user). This applies to both default-config imports and hand-authored
    // configs. Falls back to the legacy `workDir/configs/autotrain/models/user`
    // path only when `userConfigsPath` is not configured.
    if (finalYamlContent) {
      const fileName = `${name.replace(/\s+/g, "_").replace(/[^a-zA-Z0-9_.-]/g, "").toLowerCase()}.yml`;

      let userConfigDir = "";
      let relPath = "";
      if (userConfigsPath && currentUser?.username) {
        userConfigDir = path.join(userConfigsPath, currentUser.username, "models");
        relPath = path.join(currentUser.username, "models", fileName);
      } else if (workDir) {
        userConfigDir = path.join(workDir, "configs", "autotrain", "models", "user");
        relPath = `configs/autotrain/models/user/${fileName}`;
      }

      if (userConfigDir) {
        if (!fs.existsSync(userConfigDir)) fs.mkdirSync(userConfigDir, { recursive: true });
        fs.writeFileSync(path.join(userConfigDir, fileName), finalYamlContent, "utf-8");
        savedConfigPath = relPath;
      }
    }

    // Derive the display columns from the YAML using the framework-aware
    // parser. The previous hand-rolled line scanner only recognised
    // PaddleDetection's flat keys, so every imported PaddleSeg model was
    // recorded as "YOLOv3 / CSPResNet" regardless of its real architecture.
    const configFramework = asConfigFramework(framework);
    const resolvedParams = {
      ...defaultModelParams(configFramework),
      ...parseModelParams(configFramework, finalYamlContent),
    };

    // Create model in database with userId.
    // NOTE: persist `finalYamlContent`, not `yamlContent`. Preset imports send
    // only `configPath` and let the server read the file, so keying off the
    // request field stored NULL and left the model with no config at all,
    // which then produced a training job with an empty model section.
    const model = await db.model.create({
      data: {
        name: name,
        description: description || null,
        projectId: projectId,
        userId: userId,
        ...modelParamsToColumns(resolvedParams),
        yamlConfig: finalYamlContent || null,
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
