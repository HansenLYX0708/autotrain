import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { requireAdmin } from "@/lib/auth";
import * as fs from "fs";
import * as path from "path";

/**
 * Path of the PyTorch trainer bundled with this project.
 *
 * Unlike the Paddle repositories, `torchtrain/` ships inside the app, so we can
 * offer a working default instead of making the user hunt for it. Only used when
 * the folder actually exists next to the running server.
 */
function bundledTorchPath(): string {
  const candidate = path.join(process.cwd(), "torchtrain");
  return fs.existsSync(path.join(candidate, "tools", "train.py")) ? candidate : "";
}

// Default configuration values
const DEFAULT_CONFIG = {
  condaEnv: "",  // Conda environment name
  condaPath: "",  // Path to conda executable
  pythonEnvsBasePath: "",  // Base path for multiple Python environments
  gpuPythonMappings: "",  // JSON string for GPU to Python path mappings
  frameworkPythonMappings: "",  // JSON string for framework to Python path mappings
  userConfigsPath: "",  // Base path for user-specific training configs
  userDatabasePath: "",  // Base path for user database storage
  paddleDetectionPath: "",
  paddleClasPath: "",
  paddleSegPath: "",
  torchPath: "",
  defaultFramework: "PaddleDetection",
};

/**
 * GET /api/settings
 * Returns the system configuration. Creates a default one if none exists.
 * Only admins can access this endpoint.
 */
export async function GET(request: NextRequest) {
  // Check admin permission
  const auth = await requireAdmin(request);
  if (auth instanceof NextResponse) {
    return auth;
  }

  try {
    let config = await db.systemConfig.findFirst();

    // Create default config if none exists
    if (!config) {
      config = await db.systemConfig.create({
        data: { ...DEFAULT_CONFIG, torchPath: bundledTorchPath() },
      });
    } else if (!config.torchPath && bundledTorchPath()) {
      // Backfill for configs created before the torch frameworks existed, so a
      // user upgrading does not have to discover and type this path.
      config = await db.systemConfig.update({
        where: { id: config.id },
        data: { torchPath: bundledTorchPath() },
      });
    }

    return NextResponse.json({
      success: true,
      data: config,
    });
  } catch (error) {
    console.error("Error fetching system config:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to fetch system configuration",
      },
      { status: 500 }
    );
  }
}

/**
 * PUT /api/settings
 * Updates the system configuration.
 * Only admins can access this endpoint.
 */
export async function PUT(request: NextRequest) {
  // Check admin permission
  const auth = await requireAdmin(request);
  if (auth instanceof NextResponse) {
    return auth;
  }

  try {
    const body = await request.json();

    // Get existing config or create default
    let config = await db.systemConfig.findFirst();

    if (!config) {
      // Create new config with provided values
      config = await db.systemConfig.create({
        data: {
          condaEnv: body.condaEnv ?? DEFAULT_CONFIG.condaEnv,
          condaPath: body.condaPath ?? DEFAULT_CONFIG.condaPath,
          pythonEnvsBasePath: body.pythonEnvsBasePath ?? DEFAULT_CONFIG.pythonEnvsBasePath,
          gpuPythonMappings: body.gpuPythonMappings ?? DEFAULT_CONFIG.gpuPythonMappings,
          frameworkPythonMappings: body.frameworkPythonMappings ?? DEFAULT_CONFIG.frameworkPythonMappings,
          userConfigsPath: body.userConfigsPath ?? DEFAULT_CONFIG.userConfigsPath,
          userDatabasePath: body.userDatabasePath ?? DEFAULT_CONFIG.userDatabasePath,
          paddleDetectionPath: body.paddleDetectionPath ?? DEFAULT_CONFIG.paddleDetectionPath,
          paddleClasPath: body.paddleClasPath ?? DEFAULT_CONFIG.paddleClasPath,
          paddleSegPath: body.paddleSegPath ?? DEFAULT_CONFIG.paddleSegPath,
          torchPath: body.torchPath ?? bundledTorchPath(),
          defaultFramework: body.defaultFramework ?? DEFAULT_CONFIG.defaultFramework,
        },
      });
    } else {
      // Update existing config
      config = await db.systemConfig.update({
        where: { id: config.id },
        data: {
          condaEnv: body.condaEnv,
          condaPath: body.condaPath,
          pythonEnvsBasePath: body.pythonEnvsBasePath,
          gpuPythonMappings: body.gpuPythonMappings,
          frameworkPythonMappings: body.frameworkPythonMappings,
          userConfigsPath: body.userConfigsPath,
          userDatabasePath: body.userDatabasePath,
          paddleDetectionPath: body.paddleDetectionPath,
          paddleClasPath: body.paddleClasPath,
          paddleSegPath: body.paddleSegPath,
          torchPath: body.torchPath,
          defaultFramework: body.defaultFramework,
        },
      });
    }

    return NextResponse.json({
      success: true,
      data: config,
      message: "Settings updated successfully",
    });
  } catch (error) {
    console.error("Error updating system config:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to update system configuration",
      },
      { status: 500 }
    );
  }
}
