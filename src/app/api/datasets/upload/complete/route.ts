import { NextRequest, NextResponse } from "next/server";
import { existsSync } from "fs";
import { copyFile, mkdir, readdir, rename, rm } from "fs/promises";
import { createReadStream, createWriteStream } from "fs";
import { dirname, join } from "path";
import { pipeline } from "stream/promises";
import { requireAuth } from "@/lib/auth";
import { db } from "@/lib/db";
import { isInside } from "@/lib/safe-path";
import { fileTempDirName, safeRelativePath, uploadTempDir } from "@/lib/upload-tasks";

const UPLOAD_ID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

// POST /api/datasets/upload/complete - Assemble all chunks into the final dataset
export async function POST(request: NextRequest) {
  let taskId: string | null = null;
  try {
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;
    const { userId } = auth;

    const user = await db.user.findUnique({
      where: { id: userId },
      select: { id: true, username: true },
    });
    if (!user) return NextResponse.json({ error: "User not found" }, { status: 404 });

    const systemConfig = await db.systemConfig.findFirst();
    const userDatabasePath = (systemConfig as any)?.userDatabasePath;
    if (!userDatabasePath) {
      return NextResponse.json({ error: "User database path not configured" }, { status: 500 });
    }

    const body = await request.json();
    const { uploadId } = body;
    if (!uploadId || !UPLOAD_ID_RE.test(uploadId)) {
      return NextResponse.json({ error: "Invalid upload ID" }, { status: 400 });
    }

    const task = await db.uploadTask.findFirst({ where: { id: uploadId, userId, source: "browser" } });
    if (!task) return NextResponse.json({ error: "Upload task not found" }, { status: 404 });
    if (task.status === "completed") {
      return NextResponse.json({ success: true, alreadyCompleted: true });
    }
    if (task.status === "cancelled") {
      return NextResponse.json({ error: "Task was cancelled" }, { status: 409 });
    }
    taskId = task.id;

    // targetDir comes from the DB record, never from the client body.
    const targetDir = task.targetDir;
    const tempDir = uploadTempDir(userDatabasePath, user.username, uploadId);
    const userRoot = join(userDatabasePath, user.username);
    if (!isInside(userRoot, targetDir)) {
      return NextResponse.json({ error: "Invalid target directory" }, { status: 400 });
    }
    if (!existsSync(tempDir)) {
      return NextResponse.json({ error: "Upload temp directory not found" }, { status: 400 });
    }

    await db.uploadTask.update({ where: { id: task.id }, data: { status: "finalizing" } });

    // For labelme format with overwrite, clean up existing files first
    if (task.overwrite && task.format === "labelme" && existsSync(targetDir)) {
      try {
        for (const file of await readdir(targetDir)) {
          await rm(join(targetDir, file), { recursive: true, force: true });
        }
      } catch (err) {
        console.error("Error cleaning up existing directory:", err);
      }
    }
    await mkdir(targetDir, { recursive: true });

    const files: { relativePath: string; totalChunks: number; size: number }[] = Array.isArray(body.files)
      ? body.files
      : [];
    const errors: string[] = [];
    const savedFiles: string[] = [];

    for (const fileInfo of files) {
      const relativePath = safeRelativePath(fileInfo.relativePath);
      if (!relativePath) {
        errors.push(`Unsafe path in upload: ${JSON.stringify(fileInfo.relativePath)}`);
        continue;
      }
      const targetPath = join(targetDir, relativePath);
      if (!isInside(targetDir, targetPath)) {
        errors.push(`Path escapes target directory: ${relativePath}`);
        continue;
      }
      const fileTempDir = join(tempDir, fileTempDirName(relativePath));
      const totalChunks = Math.max(1, Number(fileInfo.totalChunks) || 1);

      // Check all chunks are on disk before starting.
      const missing: number[] = [];
      for (let i = 0; i < totalChunks; i++) {
        if (!existsSync(join(fileTempDir, `${i}.chunk`))) missing.push(i);
      }
      if (missing.length > 0) {
        errors.push(`Missing chunks for ${relativePath}: ${missing.join(", ")}`);
        continue;
      }

      try {
        await mkdir(dirname(targetPath), { recursive: true });
        if (totalChunks === 1) {
          // Single-chunk files are renamed into place (copy fallback for the
          // EXDEV case where temp and target sit on different volumes).
          const chunkPath = join(fileTempDir, "0.chunk");
          try {
            await rename(chunkPath, targetPath);
          } catch {
            await copyFile(chunkPath, targetPath);
          }
        } else {
          // Stream chunks — never Buffer.concat the whole file into memory.
          const output = createWriteStream(targetPath);
          try {
            for (let i = 0; i < totalChunks; i++) {
              await pipeline(createReadStream(join(fileTempDir, `${i}.chunk`)), output, { end: false });
            }
          } finally {
            output.end();
            await new Promise<void>((resolve) => output.once("close", resolve));
          }
        }
        savedFiles.push(relativePath);
      } catch (err) {
        errors.push(`Failed to assemble ${relativePath}: ${err instanceof Error ? err.message : String(err)}`);
      }
    }

    // If every file failed, do not leave a partial/emptied dataset behind.
    if (savedFiles.length === 0 && files.length > 0) {
      await rm(targetDir, { recursive: true, force: true }).catch(() => {});
    }
    await rm(tempDir, { recursive: true, force: true }).catch(() => {});

    if (savedFiles.length === 0 && files.length > 0) {
      const message = `All files failed to assemble: ${errors[0] || "unknown error"}`;
      await db.uploadTask.update({ where: { id: task.id }, data: { status: "failed", error: message } });
      return NextResponse.json({ error: message, errors }, { status: 500 });
    }

    await db.uploadTask.update({
      where: { id: task.id },
      data: {
        status: "completed",
        completedAt: new Date(),
        doneFiles: savedFiles.length,
        error: errors.length > 0 ? `${errors.length} file(s) skipped: ${errors[0]}` : null,
      },
    });

    return NextResponse.json({
      success: true,
      uploadId: task.id,
      savedFiles: savedFiles.length,
      skippedFiles: errors.length,
      errors: errors.slice(0, 20),
    });
  } catch (error) {
    console.error("Error completing upload:", error);
    if (taskId) {
      await db.uploadTask.update({
        where: { id: taskId },
        data: { status: "failed", error: error instanceof Error ? error.message : "Unknown error" },
      }).catch(() => {});
    }
    return NextResponse.json(
      { error: "Failed to complete upload", message: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
