import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { requireAuth } from "@/lib/auth";
import { getWorkDir } from "@/lib/frameworks";
import { ensureDefaultConfigs, getDefaultFolderName } from "@/lib/default-configs";
import {
  asConfigFramework,
  defaultTrainingParams,
  parseTrainingParams,
  trainingParamsToColumns,
} from "@/lib/training-yaml";
import * as fs from "fs";
import * as path from "path";

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
      defaultConfigDir = ensureDefaultConfigs(userConfigsPath, framework, "training");
    } else if (workDir) {
      // Fallback to old path if userConfigsPath not set.
      defaultConfigDir = path.join(workDir, "configs", "autotrain", "training", "default");
    }

    const defaultConfigs: Array<{ name: string; path: string; content: string }> = [];

    if (defaultConfigDir && fs.existsSync(defaultConfigDir)) {
      const files = fs.readdirSync(defaultConfigDir).filter(f => f.endsWith(".yml") || f.endsWith(".yaml"));
      
      for (const file of files) {
        const filePath = path.join(defaultConfigDir, file);
        const content = fs.readFileSync(filePath, "utf-8");
        defaultConfigs.push({
          name: file.replace(/\.(yml|yaml)$/, ""),
          path: userConfigsPath
            ? path.join(defaultFolder, "training", file)
            : `configs/autotrain/training/default/${file}`,
          content: content,
        });
      }
    }

    // Also list user configs from userConfigsPath/{username}/training folder or fallback
    let userConfigDir = "";
    if (userConfigsPath && currentUser.username) {
      userConfigDir = path.join(userConfigsPath, currentUser.username, "training");
    } else if (workDir) {
      userConfigDir = path.join(workDir, "configs", "autotrain", "training", "user");
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
            ? path.join(currentUser.username, "training", file)
            : `configs/autotrain/training/user/${file}`,
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
        userConfigsPath,
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
    const workDir = getWorkDir(framework, systemConfig);
    const userConfigsPath = (systemConfig as any)?.userConfigsPath;

    // Look up current user's username so we can save into their personal folder.
    const currentUser = await db.user.findUnique({
      where: { id: userId },
      select: { username: true },
    });

    let savedConfigPath = configPath;
    let finalYamlContent: string | null = yamlContent || null;

    // If importing a default config the request may ship without yamlContent;
    // resolve it from disk under the new `userConfigsPath` layout first, then
    // fall back to the legacy `workDir` layout.
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

    // Derive the display columns from the YAML using the framework-aware
    // parser. The previous implementation only understood PaddleDetection keys,
    // so every PaddleSeg config was persisted with placeholder values (100
    // epochs / lr 0.001 / batch 8) no matter what the YAML actually said.
    // `yamlConfig` is authoritative; these columns exist for the list view.
    const configFramework = asConfigFramework(framework);
    const resolvedParams = {
      ...defaultTrainingParams(configFramework),
      ...parseTrainingParams(configFramework, finalYamlContent),
      // An explicit `trainingParams` payload (from the create dialog) wins over
      // whatever we could recover from the text.
      ...(trainingParams ?? {}),
    };
    const columns = trainingParamsToColumns(configFramework, resolvedParams);

    // Always persist a per-user copy under
    //   {userConfigsPath}/{username}/training/{name}.yml
    // Applies to default-config imports and hand-authored configs alike, so
    // the user always ends up with their own editable YAML on disk. Falls back
    // to the legacy `workDir/configs/autotrain/training/user` path only when
    // `userConfigsPath` is not configured.
    if (finalYamlContent) {
      const fileName = `${name.replace(/\s+/g, "_").replace(/[^a-zA-Z0-9_.-]/g, "").toLowerCase()}.yml`;

      let userConfigDir = "";
      let relPath = "";
      if (userConfigsPath && currentUser?.username) {
        userConfigDir = path.join(userConfigsPath, currentUser.username, "training");
        relPath = path.join(currentUser.username, "training", fileName);
      } else if (workDir) {
        userConfigDir = path.join(workDir, "configs", "autotrain", "training", "user");
        relPath = `configs/autotrain/training/user/${fileName}`;
      }

      if (userConfigDir) {
        if (!fs.existsSync(userConfigDir)) fs.mkdirSync(userConfigDir, { recursive: true });
        fs.writeFileSync(path.join(userConfigDir, fileName), finalYamlContent, "utf-8");
        savedConfigPath = relPath;
      }
    }

    // Suppress unused-var warnings from the now-simplified flow.
    void isDefault;

    // Create training config in database with userId
    const config = await db.trainingConfig.create({
      data: {
        projectId: projectId,
        userId: userId,
        name: name,
        ...columns,
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
