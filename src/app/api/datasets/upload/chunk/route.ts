import { NextRequest, NextResponse } from "next/server";
import { writeFile, mkdir } from "fs/promises";
import { join } from "path";
import { requireAuth } from "@/lib/auth";
import { db } from "@/lib/db";
import { fileTempDirName, safeRelativePath, uploadTempDir } from "@/lib/upload-tasks";

const UPLOAD_ID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

type ManifestEntry = { relativePath: string; chunkIndex: number; totalChunks: number };

/**
 * POST /api/datasets/upload/chunk — upload one chunk or a batch of chunks.
 *
 * Batched form (new client): fields `uploadId`, `manifest` (JSON array of
 * `{relativePath, chunkIndex, totalChunks}`) and blobs `file0..fileN` in the
 * same order. One DB update per request.
 *
 * Legacy form: `uploadId`, `relativePath`, `chunkIndex`, `totalChunks`, `chunk`.
 */
export async function POST(request: NextRequest) {
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

    const formData = await request.formData();
    const uploadId = formData.get("uploadId") as string;
    if (!uploadId || !UPLOAD_ID_RE.test(uploadId)) {
      return NextResponse.json({ error: "Invalid upload ID" }, { status: 400 });
    }

    // Task must exist, belong to this user, and still accept chunks.
    const task = await db.uploadTask.findFirst({ where: { id: uploadId, userId, source: "browser" } });
    if (!task) return NextResponse.json({ error: "Upload task not found" }, { status: 404 });
    if (task.status !== "uploading" && task.status !== "pending") {
      return NextResponse.json({ error: `Task is ${task.status}; chunks are no longer accepted` }, { status: 409 });
    }

    // Normalise into a list of {entry, blob} pairs.
    const manifestRaw = formData.get("manifest");
    const items: { entry: ManifestEntry; blob: Blob }[] = [];
    if (manifestRaw) {
      let manifest: ManifestEntry[];
      try {
        manifest = JSON.parse(manifestRaw as string);
      } catch {
        return NextResponse.json({ error: "Invalid manifest JSON" }, { status: 400 });
      }
      if (!Array.isArray(manifest) || manifest.length === 0 || manifest.length > 200) {
        return NextResponse.json({ error: "Manifest must list 1-200 chunks" }, { status: 400 });
      }
      for (const [index, entry] of manifest.entries()) {
        const blob = formData.get(`file${index}`) as Blob | null;
        if (!blob) return NextResponse.json({ error: `Missing blob file${index}` }, { status: 400 });
        items.push({ entry, blob });
      }
    } else {
      const entry: ManifestEntry = {
        relativePath: formData.get("relativePath") as string,
        chunkIndex: parseInt(formData.get("chunkIndex") as string, 10),
        totalChunks: parseInt(formData.get("totalChunks") as string, 10),
      };
      const blob = formData.get("chunk") as Blob | null;
      if (!blob) return NextResponse.json({ error: "Missing required fields" }, { status: 400 });
      items.push({ entry, blob });
    }

    const tempDir = uploadTempDir(userDatabasePath, user.username, uploadId);
    let bytes = 0;
    let completedFiles = 0;
    for (const { entry, blob } of items) {
      const relativePath = safeRelativePath(entry.relativePath);
      const chunkIndex = Number(entry.chunkIndex);
      const totalChunks = Number(entry.totalChunks);
      if (
        !relativePath ||
        !Number.isInteger(chunkIndex) || chunkIndex < 0 || chunkIndex > 10000 ||
        !Number.isInteger(totalChunks) || totalChunks < 1 || totalChunks > 10000 ||
        chunkIndex >= totalChunks
      ) {
        return NextResponse.json({ error: `Invalid chunk descriptor: ${JSON.stringify(entry.relativePath)}` }, { status: 400 });
      }
      const fileTempDir = join(tempDir, fileTempDirName(relativePath));
      await mkdir(fileTempDir, { recursive: true });
      const buffer = Buffer.from(await blob.arrayBuffer());
      await writeFile(join(fileTempDir, `${chunkIndex}.chunk`), buffer);
      bytes += buffer.length;
      if (chunkIndex === totalChunks - 1) completedFiles += 1;
    }

    await db.uploadTask.update({
      where: { id: task.id },
      data: { doneFiles: { increment: completedFiles }, doneBytes: { increment: bytes } },
    });

    return NextResponse.json({ success: true, uploaded: items.length });
  } catch (error) {
    console.error("Error uploading chunk:", error);
    return NextResponse.json(
      { error: "Failed to upload chunk", message: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
