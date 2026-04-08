import { NextRequest, NextResponse } from "next/server";
import { writeFile, mkdir, readdir, readFile, unlink, rmdir } from "fs/promises";
import { existsSync } from "fs";
import { join, dirname } from "path";
import { sessions } from "../../../auth/route";
import { db } from "@/lib/db";

// POST /api/datasets/upload/complete - Complete upload and merge chunks
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
    const { uploadId, targetDir, files, format, datasetName } = body;

    if (!uploadId || !targetDir || !files) {
      return NextResponse.json(
        { error: "Missing required fields" },
        { status: 400 }
      );
    }

    // Security: validate targetDir is within user's directory
    const allowedBase = join(userDatabasePath, user.username);
    if (!targetDir.startsWith(allowedBase)) {
      return NextResponse.json({ error: "Invalid target directory" }, { status: 403 });
    }

    // Create target directory
    await mkdir(targetDir, { recursive: true });

    const tempDir = join(userDatabasePath, user.username, ".uploads", uploadId);
    const errors: string[] = [];
    const savedFiles: string[] = [];

    // Merge chunks for each file
    for (const fileInfo of files) {
      const { relativePath, totalChunks } = fileInfo;
      const fileTempDir = join(tempDir, relativePath.replace(/[/\\]/g, "___"));
      
      try {
        // Read all chunks and merge
        const chunkBuffers: Buffer[] = [];
        let missingChunks: number[] = [];

        for (let i = 0; i < totalChunks; i++) {
          const chunkPath = join(fileTempDir, `${i}.chunk`);
          if (existsSync(chunkPath)) {
            const chunkData = await readFile(chunkPath);
            chunkBuffers.push(chunkData);
          } else {
            missingChunks.push(i);
          }
        }

        if (missingChunks.length > 0) {
          errors.push(`Missing chunks for ${relativePath}: ${missingChunks.join(", ")}`);
          continue;
        }

        // Write merged file
        const filePath = join(targetDir, relativePath);
        const fileDir = dirname(filePath);
        
        if (!existsSync(fileDir)) {
          await mkdir(fileDir, { recursive: true });
        }

        const mergedBuffer = Buffer.concat(chunkBuffers);
        await writeFile(filePath, mergedBuffer);
        savedFiles.push(relativePath);

        // Clean up chunks for this file
        try {
          for (let i = 0; i < totalChunks; i++) {
            const chunkPath = join(fileTempDir, `${i}.chunk`);
            if (existsSync(chunkPath)) {
              await unlink(chunkPath);
            }
          }
          await rmdir(fileTempDir);
        } catch {
          // Ignore cleanup errors
        }

      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : String(err);
        errors.push(`Failed to merge ${relativePath}: ${errorMsg}`);
      }
    }

    // Clean up temp directory
    try {
      if (existsSync(tempDir)) {
        await rmdir(tempDir);
      }
    } catch {
      // Ignore
    }

    // If all files failed, clean up target directory
    if (savedFiles.length === 0 && errors.length > 0) {
      try {
        const { rmSync } = await import("fs");
        rmSync(targetDir, { recursive: true, force: true });
      } catch {
        // Ignore
      }

      return NextResponse.json(
        {
          error: "All files failed to upload",
          details: errors,
        },
        { status: 500 }
      );
    }

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
    console.error("Error completing upload:", error);
    return NextResponse.json(
      { error: "Failed to complete upload", message: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
