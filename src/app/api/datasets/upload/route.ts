import { NextRequest, NextResponse } from "next/server";
import { writeFile, mkdir } from "fs/promises";
import { existsSync } from "fs";
import * as path from "path";
import { sessions } from "../../auth/route";
import { join } from "path";
import { db } from "@/lib/db";

export async function POST(request: NextRequest) {
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

    // Parse form data
    const formData = await request.formData();
    const files = formData.getAll("files") as File[];
    const format = formData.get("format") as string; // "COCO" or "labelme"
    const datasetName = formData.get("datasetName") as string;

    if (!files || files.length === 0) {
      return NextResponse.json(
        { error: "No files provided" },
        { status: 400 }
      );
    }

    if (!format || !datasetName) {
      return NextResponse.json(
        { error: "Format and dataset name are required" },
        { status: 400 }
      );
    }

    // Validate format
    if (format !== "COCO" && format !== "labelme") {
      return NextResponse.json(
        { error: "Invalid format. Must be COCO or labelme" },
        { status: 400 }
      );
    }

    // Validate dataset name (no special characters)
    if (!/^[a-zA-Z0-9_-]+$/.test(datasetName)) {
      return NextResponse.json(
        { error: "Dataset name can only contain letters, numbers, underscores and hyphens" },
        { status: 400 }
      );
    }

    // Build the target path: {userDatabasePath}/{username}/{format}/{datasetName}
    // User uploads a folder named "data", so files will be saved as {datasetName}/data/...
    const targetDir = join(userDatabasePath, user.username, format, datasetName);
    
    // Check if dataset directory already exists
    if (existsSync(targetDir)) {
      return NextResponse.json(
        { error: `Dataset "${datasetName}" already exists. Please choose a different name or delete it first.` },
        { status: 400 }
      );
    }

    // Create dataset directory
    await mkdir(targetDir, { recursive: true });

    // Save files - handle both flat files and folder structure
    const savedFiles: string[] = [];
    const errors: string[] = [];

    for (const file of files) {
      try {
        const bytes = await file.arrayBuffer();
        const buffer = Buffer.from(bytes);

        // Get the relative path from webkitRelativePath (for folder uploads)
        // or just use file.name (for single file uploads)
        const relativePath = (file as any).webkitRelativePath || file.name;
        const filePath = join(targetDir, relativePath);

        // Create subdirectories if needed
        const fileDir = path.dirname(filePath);
        if (!existsSync(fileDir)) {
          await mkdir(fileDir, { recursive: true });
        }

        await writeFile(filePath, buffer);
        savedFiles.push(relativePath);
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : String(err);
        errors.push(`Failed to save ${(file as any).webkitRelativePath || file.name}: ${errorMsg}`);
      }
    }

    // If all files failed, clean up and return error
    if (savedFiles.length === 0 && errors.length > 0) {
      // Clean up the created directory
      try {
        const { rmSync } = await import("fs");
        rmSync(targetDir, { recursive: true, force: true });
      } catch (cleanupErr) {
        console.error("Failed to cleanup directory:", cleanupErr);
      }

      return NextResponse.json(
        {
          error: "All files failed to upload",
          details: errors,
        },
        { status: 500 }
      );
    }

    // Return success - only files uploaded, no database record created
    return NextResponse.json({
      success: true,
      message: `Uploaded ${savedFiles.length} files`,
      data: {
        datasetName: datasetName,
        format: format,
        savedPath: targetDir,
        files: savedFiles,
        errors: errors.length > 0 ? errors : undefined,
      },
    });

  } catch (error) {
    console.error("Error uploading dataset:", error);
    return NextResponse.json(
      { error: "Failed to upload dataset", message: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
