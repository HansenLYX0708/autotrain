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

// Scan a directory for JSON files in labelme format
async function scanLabelmeJsons(jsonsPath: string): Promise<string[]> {
  const jsonFiles: string[] = [];
  try {
    if (!existsSync(jsonsPath)) {
      return jsonFiles;
    }
    const entries = await readdir(jsonsPath, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.isFile() && entry.name.endsWith('.json')) {
        jsonFiles.push(entry.name);
      }
    }
  } catch (error) {
    console.error('Error scanning labelme jsons:', error);
  }
  return jsonFiles;
}

// Scan a directory for image files
async function scanImages(imgsPath: string): Promise<string[]> {
  const imageFiles: string[] = [];
  const imageExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'];
  try {
    if (!existsSync(imgsPath)) {
      return imageFiles;
    }
    const entries = await readdir(imgsPath, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.isFile()) {
        const ext = entry.name.slice(entry.name.lastIndexOf('.')).toLowerCase();
        if (imageExtensions.includes(ext)) {
          imageFiles.push(entry.name);
        }
      }
    }
  } catch (error) {
    console.error('Error scanning images:', error);
  }
  return imageFiles;
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

    // Build paths for both COCO and labelme formats
    const cocoPath = join(userDatabasePath, user.username, "COCO");
    const labelmePath = join(userDatabasePath, user.username, "labelme");

    // Read all dataset directories from both COCO and labelme folders
    const datasets: Array<{
      name: string;
      path: string;
      format: 'COCO' | 'labelme';
      hasTrain: boolean;
      hasVal: boolean;
      hasAnnotations: boolean;
      hasImgs: boolean;
      hasJsons: boolean;
      trainAnnotations: string[];
      valAnnotations: string[];
      images: string[];
      jsons: string[];
    }> = [];

    // Scan COCO format datasets
    if (existsSync(cocoPath)) {
      const cocoEntries = await readdir(cocoPath, { withFileTypes: true });

      for (const entry of cocoEntries) {
        if (entry.isDirectory()) {
          const datasetPath = join(cocoPath, entry.name);
          const dataPath = join(datasetPath, "data");

          // Check if data subfolder exists
          if (!existsSync(dataPath)) {
            continue;
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
            format: 'COCO',
            hasTrain,
            hasVal,
            hasAnnotations,
            hasImgs: false,
            hasJsons: false,
            trainAnnotations,
            valAnnotations,
            images: [],
            jsons: [],
          });
        }
      }
    }

    // Scan labelme format datasets
    if (existsSync(labelmePath)) {
      const labelmeEntries = await readdir(labelmePath, { withFileTypes: true });

      for (const entry of labelmeEntries) {
        if (entry.isDirectory()) {
          const datasetPath = join(labelmePath, entry.name);
          const dataPath = join(datasetPath, "data");

          // Check if data subfolder exists
          if (!existsSync(dataPath)) {
            continue;
          }

          const imgsPath = join(dataPath, "imgs");
          const jsonsPath = join(dataPath, "jsons");

          const hasImgs = existsSync(imgsPath);
          const hasJsons = existsSync(jsonsPath);

          // Scan for images and json files
          const images: string[] = hasImgs ? await scanImages(imgsPath) : [];
          const jsons: string[] = hasJsons ? await scanLabelmeJsons(jsonsPath) : [];

          datasets.push({
            name: entry.name,
            path: datasetPath,
            format: 'labelme',
            hasTrain: false,
            hasVal: false,
            hasAnnotations: false,
            hasImgs,
            hasJsons,
            trainAnnotations: [],
            valAnnotations: [],
            images,
            jsons,
          });
        }
      }
    }

    return NextResponse.json({
      success: true,
      datasets,
      cocoPath: existsSync(cocoPath) ? cocoPath : null,
      labelmePath: existsSync(labelmePath) ? labelmePath : null,
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

    // Try to find dataset in both COCO and labelme paths
    const cocoDatasetPath = join(userDatabasePath, user.username, "COCO", datasetName);
    const labelmeDatasetPath = join(userDatabasePath, user.username, "labelme", datasetName);

    let datasetPath: string | null = null;
    let format: 'COCO' | 'labelme' | null = null;

    if (existsSync(cocoDatasetPath)) {
      datasetPath = cocoDatasetPath;
      format = 'COCO';
    } else if (existsSync(labelmeDatasetPath)) {
      datasetPath = labelmeDatasetPath;
      format = 'labelme';
    }

    // Check if dataset directory exists in either location
    if (!datasetPath) {
      return NextResponse.json(
        { error: "Dataset not found", searchedPaths: [cocoDatasetPath, labelmeDatasetPath] },
        { status: 404 }
      );
    }

    // Delete the dataset directory recursively
    await deleteDirectory(datasetPath);

    console.log(`[Dataset Delete] Deleted ${format} dataset folder: ${datasetPath}`);

    return NextResponse.json({
      success: true,
      message: `Dataset "${datasetName}" (${format} format) deleted successfully`,
      path: datasetPath,
      format,
    });

  } catch (error) {
    console.error("Error deleting dataset:", error);
    return NextResponse.json(
      { error: "Failed to delete dataset", message: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
