import { NextRequest, NextResponse } from "next/server";
import { mkdir, readdir } from "fs/promises";
import { existsSync } from "fs";
import { join } from "path";
import { requireAuth } from "@/lib/auth";
import { db } from "@/lib/db";
import { randomUUID } from "crypto";
import { fileTempDirName, serializeTask, uploadTempDir, validateUploadRequest } from "@/lib/upload-tasks";

const CHUNK_SIZE = 10 * 1024 * 1024; // 10MB

// POST /api/datasets/upload/init - Initialize chunked upload
export async function POST(request: NextRequest) {
  try {
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;
    const { userId } = auth;

    const user = await db.user.findUnique({
      where: { id: userId },
      select: { id: true, username: true, maxStorageQuota: true },
    });
    if (!user) return NextResponse.json({ error: "User not found" }, { status: 404 });

    const systemConfig = await db.systemConfig.findFirst();
    const userDatabasePath = (systemConfig as any)?.userDatabasePath;
    if (!userDatabasePath) {
      return NextResponse.json({ error: "User database path not configured" }, { status: 500 });
    }

    const body = await request.json();
    const { datasetName, format, totalFiles, totalSize, files, confirmedOverwrite, resumeUploadId } = body;

    // Resume path: an existing own task tells the client which chunks are
    // already on disk. The task's name/format/targetDir come from the DB —
    // the client cannot retarget an old session.
    if (resumeUploadId) {
      if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(resumeUploadId)) {
        return NextResponse.json({ error: "Invalid upload ID" }, { status: 400 });
      }
      const task = await db.uploadTask.findFirst({ where: { id: resumeUploadId, userId, source: "browser" } });
      if (!task) return NextResponse.json({ error: "Upload task not found" }, { status: 404 });
      if (task.status === "completed" || task.status === "cancelled") {
        return NextResponse.json({ error: `Task is ${task.status} and cannot be resumed` }, { status: 400 });
      }
      const tempDir = uploadTempDir(userDatabasePath, user.username, task.id);
      const uploadedChunks = await readUploadedChunks(tempDir);
      const filesInfo = (Array.isArray(files) ? files : []).map((f: any) => {
        // Chunk dirs are stored under the encoded name (`/` → `___`).
        const fileChunks = uploadedChunks[fileTempDirName(f.relativePath)] || [];
        const totalChunks = Math.max(1, Math.ceil(f.size / CHUNK_SIZE));
        return {
          relativePath: f.relativePath,
          size: f.size,
          totalChunks,
          uploadedChunks: fileChunks,
          progress: Math.round((fileChunks.length / totalChunks) * 100),
        };
      });
      await db.uploadTask.update({ where: { id: task.id }, data: { status: "uploading", error: null } });
      return NextResponse.json({
        success: true,
        uploadId: task.id,
        datasetName: task.datasetName,
        format: task.format,
        targetDir: task.targetDir,
        tempDir,
        files: filesInfo,
        overallProgress: 0,
        chunkSize: CHUNK_SIZE,
        resume: true,
      });
    }

    // Same name/format/quota/overwrite rules as the server-import path.
    const validation = await validateUploadRequest(user, { userDatabasePath }, {
      datasetName, format, totalSize, confirmedOverwrite,
    });
    if (!validation.ok) {
      const { status, ...payload } = validation.response;
      return NextResponse.json(payload, { status });
    }
    const { targetDir, overwrite } = validation;

    // Generate upload session ID (= UploadTask.id)
    const uploadId = randomUUID();
    const tempDir = uploadTempDir(userDatabasePath, user.username, uploadId);
    await mkdir(tempDir, { recursive: true });

    const filesInfo = (Array.isArray(files) ? files : []).map((f: any) => {
      const totalChunks = Math.max(1, Math.ceil(f.size / CHUNK_SIZE));
      return {
        relativePath: f.relativePath,
        size: f.size,
        totalChunks,
        uploadedChunks: [] as number[],
        progress: 0,
      };
    });

    const task = await db.uploadTask.create({
      data: {
        id: uploadId,
        userId,
        source: "browser",
        datasetName,
        format,
        targetDir,
        status: "uploading",
        totalFiles: filesInfo.length,
        totalBytes: totalSize || 0,
        overwrite,
      },
    });

    return NextResponse.json({
      success: true,
      uploadId,
      targetDir,
      tempDir,
      files: filesInfo,
      overallProgress: 0,
      chunkSize: CHUNK_SIZE,
      task: serializeTask(task),
    });
  } catch (error) {
    console.error("Error initializing upload:", error);
    return NextResponse.json(
      { error: "Failed to initialize upload", message: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

async function readUploadedChunks(tempDir: string): Promise<Record<string, number[]>> {
  const uploadedChunks: Record<string, number[]> = {};
  try {
    if (existsSync(tempDir)) {
      const entries = await readdir(tempDir, { withFileTypes: true });
      for (const entry of entries) {
        if (entry.isDirectory()) {
          const chunks = await readdir(join(tempDir, entry.name));
          uploadedChunks[entry.name] = chunks
            .filter(c => c.endsWith(".chunk"))
            .map(c => parseInt(c.replace(".chunk", ""), 10))
            .filter(n => !isNaN(n));
        }
      }
    }
  } catch {
    // Ignore errors — resume just starts from nothing.
  }
  return uploadedChunks;
}
