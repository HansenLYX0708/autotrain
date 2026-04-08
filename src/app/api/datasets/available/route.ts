import { NextRequest, NextResponse } from "next/server";
import { readdir, rmdir, unlink } from "fs/promises";
import { existsSync } from "fs";
import { join } from "path";
import { sessions } from "../../auth/route";
import { db } from "@/lib/db";

// Recursively delete a directory and all its contents
async function deleteDirectory(dirPath: string): Promise<void> {
  if (!existsSync(dirPath)) {
    return;
  }
  
  const entries = await readdir(dirPath, { withFileTypes: true });
  
  for (const entry of entries) {
    const fullPath = join(dirPath, entry.name);
    if (entry.isDirectory()) {
      await deleteDirectory(fullPath);
    } else {
      await unlink(fullPath);
    }
  }
  
  await rmdir(dirPath);
}

// Scan a directory for JSON annotation files
async function scanAnnotations(annotationsPath: string): Promise<string[]> {
  const jsonFiles: string[] = [];
  try {
    if (!existsSync(annotationsPath)) {
      return jsonFiles;
    }
    const entries = await readdir(annotationsPath, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.isFile() && entry.name.endsWith('.json')) {
        jsonFiles.push(entry.name);
      }
    }
  } catch (error) {
    console.error('Error scanning annotations:', error);
  }
  return jsonFiles;
}

export async function GET(request: NextRequest) {
  try {
    // Get current user from session
    const token = request.cookies.get("auth-token")?.value;
    if (!token || !sessions.has(token)) {
      return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
    }

    const session = sessions.get(token)!;
    const userId = session.userId;

    // Get user info
    const user = await db.user.findUnique({
      where: { id: userId },
      select: { id: true, username: true, role: true },
    });

    if (!user) {
      return NextResponse.json({ error: "User not found" }, { status: 404 });
    }

    // Get system config for userDatabasePath
    const systemConfig = await db.systemConfig.findFirst();
    const userDatabasePath = (systemConfig as any)?.userDatabasePath;

    if (!userDatabasePath) {
      return NextResponse.json(
        { error: "User database path not configured" },
        { status: 500 }
      );
    }

    // Build the COCO path: {userDatabasePath}/{username}/COCO
    const cocoPath = join(userDatabasePath, user.username, "COCO");

    // Check if COCO directory exists
    if (!existsSync(cocoPath)) {
      return NextResponse.json({
        success: true,
        datasets: [],
        message: "COCO directory does not exist"
      });
    }

    // Read all dataset directories in COCO folder
    // Each dataset folder should contain a "data" subfolder with train/val/annotations
    const datasets: Array<{
      name: string;
      path: string;
      hasTrain: boolean;
      hasVal: boolean;
      hasAnnotations: boolean;
      trainAnnotations: string[];
      valAnnotations: string[];
    }> = [];

    const entries = await readdir(cocoPath, { withFileTypes: true });

    for (const entry of entries) {
      if (entry.isDirectory()) {
        const datasetPath = join(cocoPath, entry.name);
        const dataPath = join(datasetPath, "data");
        
        // Check if data subfolder exists
        if (!existsSync(dataPath)) {
          continue; // Skip folders without data subfolder
        }
        
        const trainPath = join(dataPath, "train");
        const valPath = join(dataPath, "val");
        const annotationsPath = join(dataPath, "annotations");

        const hasTrain = existsSync(trainPath);
        const hasVal = existsSync(valPath);
        const hasAnnotations = existsSync(annotationsPath);

        // Scan for annotation files
        const trainAnnotations: string[] = [];
        const valAnnotations: string[] = [];

        if (hasAnnotations) {
          const annoFiles = await scanAnnotations(annotationsPath);
          for (const file of annoFiles) {
            if (file.toLowerCase().includes('train')) {
              trainAnnotations.push(`data/annotations/${file}`);
            } else if (file.toLowerCase().includes('val') || file.toLowerCase().includes('valid')) {
              valAnnotations.push(`data/annotations/${file}`);
            }
          }
        }

        datasets.push({
          name: entry.name,
          path: datasetPath,
          hasTrain,
          hasVal,
          hasAnnotations,
          trainAnnotations,
          valAnnotations,
        });
      }
    }

    return NextResponse.json({
      success: true,
      datasets,
      cocoPath,
    });

  } catch (error) {
    console.error("Error checking available data:", error);
    return NextResponse.json(
      { error: "Failed to check data", message: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

// DELETE /api/datasets/available - Delete a dataset folder from file system
export async function DELETE(request: NextRequest) {
  try {
    // Get current user from session
    const token = request.cookies.get("auth-token")?.value;
    if (!token || !sessions.has(token)) {
      return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
    }

    const session = sessions.get(token)!;
    const userId = session.userId;

    // Get user info
    const user = await db.user.findUnique({
      where: { id: userId },
      select: { id: true, username: true, role: true },
    });

    if (!user) {
      return NextResponse.json({ error: "User not found" }, { status: 404 });
    }

    // Get dataset name from query params
    const { searchParams } = new URL(request.url);
    const datasetName = searchParams.get("name");

    if (!datasetName) {
      return NextResponse.json(
        { error: "Dataset name is required" },
        { status: 400 }
      );
    }

    // Get system config for userDatabasePath
    const systemConfig = await db.systemConfig.findFirst();
    const userDatabasePath = (systemConfig as any)?.userDatabasePath;

    if (!userDatabasePath) {
      return NextResponse.json(
        { error: "User database path not configured" },
        { status: 500 }
      );
    }

    // Build the dataset path: {userDatabasePath}/{username}/COCO/{datasetName}
    const datasetPath = join(userDatabasePath, user.username, "COCO", datasetName);

    // Check if dataset directory exists
    if (!existsSync(datasetPath)) {
      return NextResponse.json(
        { error: "Dataset not found", path: datasetPath },
        { status: 404 }
      );
    }

    // Delete the dataset directory recursively
    await deleteDirectory(datasetPath);

    console.log(`[Dataset Delete] Deleted dataset folder: ${datasetPath}`);

    return NextResponse.json({
      success: true,
      message: `Dataset "${datasetName}" deleted successfully`,
      path: datasetPath,
    });

  } catch (error) {
    console.error("Error deleting dataset:", error);
    return NextResponse.json(
      { error: "Failed to delete dataset", message: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
