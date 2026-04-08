import { NextRequest, NextResponse } from "next/server";
import { mkdir, readdir } from "fs/promises";
import { existsSync } from "fs";
import { join } from "path";
import { sessions } from "../../../auth/route";
import { db } from "@/lib/db";
import { randomUUID } from "crypto";

// POST /api/datasets/upload/init - Initialize chunked upload
export async function POST(request: NextRequest) {
  try {
    const token = request.cookies.get("auth-token")?.value;
    if (!token || !sessions.has(token)) {
      return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
    }

    const session = sessions.get(token)!;
    const userId = session.userId;

    const user = await db.user.findUnique({
      where: { id: userId },
      select: { id: true, username: true },
    });

    if (!user) {
      return NextResponse.json({ error: "User not found" }, { status: 404 });
    }

    const systemConfig = await db.systemConfig.findFirst();
    const userDatabasePath = (systemConfig as any)?.userDatabasePath;

    if (!userDatabasePath) {
      return NextResponse.json(
        { error: "User database path not configured" },
        { status: 500 }
      );
    }

    const body = await request.json();
    const { datasetName, format, totalFiles, totalSize, files } = body;

    if (!datasetName || !format) {
      return NextResponse.json(
        { error: "Dataset name and format are required" },
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

    // Validate dataset name
    if (!/^[a-zA-Z0-9_-]+$/.test(datasetName)) {
      return NextResponse.json(
        { error: "Dataset name can only contain letters, numbers, underscores and hyphens" },
        { status: 400 }
      );
    }

    // Generate upload session ID
    const uploadId = randomUUID();
    
    // Target directory
    const targetDir = join(userDatabasePath, user.username, format, datasetName);
    
    // Temp directory for chunks
    const tempDir = join(userDatabasePath, user.username, ".uploads", uploadId);

    // Check if dataset already exists
    if (existsSync(targetDir)) {
      return NextResponse.json(
        { error: `Dataset "${datasetName}" already exists` },
        { status: 400 }
      );
    }

    // Create temp directory
    await mkdir(tempDir, { recursive: true });

    // Check for existing uploaded chunks (in case of resume)
    let uploadedChunks: Record<string, number[]> = {};
    try {
      if (existsSync(tempDir)) {
        const entries = await readdir(tempDir, { withFileTypes: true });
        for (const entry of entries) {
          if (entry.isDirectory()) {
            const fileTempDir = join(tempDir, entry.name);
            const chunks = await readdir(fileTempDir);
            uploadedChunks[entry.name] = chunks
              .filter(c => c.endsWith(".chunk"))
              .map(c => parseInt(c.replace(".chunk", "")));
          }
        }
      }
    } catch {
      // Ignore errors
    }

    // Calculate progress for each file
    const filesInfo = files.map((f: any) => {
      const fileChunks = uploadedChunks[f.relativePath] || [];
      const totalChunks = Math.ceil(f.size / (10 * 1024 * 1024)); // 10MB chunks
      return {
        relativePath: f.relativePath,
        size: f.size,
        totalChunks,
        uploadedChunks: fileChunks, // 返回已上传的分片索引数组
        progress: Math.round((fileChunks.length / totalChunks) * 100),
      };
    });

    const totalUploadedChunks = Object.values(uploadedChunks).flat().length;
    const totalExpectedChunks = filesInfo.reduce((acc: number, f: any) => acc + f.totalChunks, 0);
    const overallProgress = totalExpectedChunks > 0 
      ? Math.round((totalUploadedChunks / totalExpectedChunks) * 100) 
      : 0;

    return NextResponse.json({
      success: true,
      uploadId,
      targetDir,
      tempDir,
      files: filesInfo,
      overallProgress,
      chunkSize: 10 * 1024 * 1024, // 10MB
    });

  } catch (error) {
    console.error("Error initializing upload:", error);
    return NextResponse.json(
      { error: "Failed to initialize upload", message: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
