import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";

// Default configuration values
const DEFAULT_CONFIG = {
  pythonPath: "python",  // Use 'python' command, works on Windows
  paddleDetectionPath: "",
  paddleClasPath: "",
  defaultGpu: 0,
  defaultFramework: "PaddleDetection",
};

/**
 * GET /api/settings
 * Returns the system configuration. Creates a default one if none exists.
 */
export async function GET() {
  try {
    let config = await db.systemConfig.findFirst();

    // Create default config if none exists
    if (!config) {
      config = await db.systemConfig.create({
        data: DEFAULT_CONFIG,
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
 */
export async function PUT(request: NextRequest) {
  try {
    const body = await request.json();

    // Get existing config or create default
    let config = await db.systemConfig.findFirst();

    if (!config) {
      // Create new config with provided values
      config = await db.systemConfig.create({
        data: {
          pythonPath: body.pythonPath ?? DEFAULT_CONFIG.pythonPath,
          paddleDetectionPath: body.paddleDetectionPath ?? DEFAULT_CONFIG.paddleDetectionPath,
          paddleClasPath: body.paddleClasPath ?? DEFAULT_CONFIG.paddleClasPath,
          defaultGpu: body.defaultGpu ?? DEFAULT_CONFIG.defaultGpu,
          defaultFramework: body.defaultFramework ?? DEFAULT_CONFIG.defaultFramework,
        },
      });
    } else {
      // Update existing config
      config = await db.systemConfig.update({
        where: { id: config.id },
        data: {
          pythonPath: body.pythonPath,
          paddleDetectionPath: body.paddleDetectionPath,
          paddleClasPath: body.paddleClasPath,
          defaultGpu: body.defaultGpu,
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
