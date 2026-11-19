import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import { requireAuth } from "@/lib/auth";
import { db } from "@/lib/db";
import { folderSize, serializeTask, validateUploadRequest } from "@/lib/upload-tasks";
import { startServerImport } from "@/lib/upload-server-runner";
import { randomUUID } from "crypto";

export const runtime = "nodejs";

/**
 * POST /api/upload-tasks/server-import — copy a server-local folder into the
 * user's dataset storage as a background UploadTask. Same access policy as
 * the server-path browser (/api/fs/browse): any authenticated user.
 */
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

    const { sourcePath, datasetName, format, confirmedOverwrite } = await request.json();
    if (!sourcePath || typeof sourcePath !== "string") {
      return NextResponse.json({ error: "sourcePath is required" }, { status: 400 });
    }
    const stat = await fs.stat(sourcePath).catch(() => null);
    if (!stat) return NextResponse.json({ error: `Source path does not exist: ${sourcePath}` }, { status: 400 });
    if (!stat.isDirectory()) return NextResponse.json({ error: `Not a directory: ${sourcePath}` }, { status: 400 });

    // Same name/format/quota/exists rules as the browser upload; the quota is
    // charged against the source folder's total size.
    const sourceSize = await folderSize(sourcePath);
    const validation = await validateUploadRequest(user, { userDatabasePath }, {
      datasetName, format, totalSize: sourceSize, confirmedOverwrite,
    });
    if (!validation.ok) {
      const { status, ...payload } = validation.response;
      return NextResponse.json(payload, { status });
    }
    const { targetDir, overwrite } = validation;

    const task = await db.uploadTask.create({
      data: {
        id: randomUUID(),
        userId,
        source: "server",
        datasetName,
        format,
        sourcePath,
        targetDir,
        status: "uploading",
        totalBytes: sourceSize,
        overwrite,
      },
    });

    startServerImport(task.id);
    return NextResponse.json({ success: true, data: serializeTask(task) });
  } catch (error) {
    console.error("Error starting server import:", error);
    return NextResponse.json(
      { error: "Failed to start server import", message: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
