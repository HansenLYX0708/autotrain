import { NextRequest, NextResponse } from "next/server";
import { rm } from "fs/promises";
import { requireAuth } from "@/lib/auth";
import { db } from "@/lib/db";
import { serializeTask, uploadTempDir } from "@/lib/upload-tasks";
import { cancelServerImport } from "@/lib/upload-server-runner";

export const runtime = "nodejs";

const ACTIVE_STATUSES = ["pending", "uploading", "finalizing"];

// POST /api/upload-tasks/[id]/cancel
export async function POST(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  try {
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;
    const { userId } = auth;
    const { id } = await params;

    const task = await db.uploadTask.findFirst({ where: { id, userId } });
    if (!task) return NextResponse.json({ error: "Upload task not found" }, { status: 404 });
    if (!ACTIVE_STATUSES.includes(task.status)) {
      return NextResponse.json({ success: true, data: serializeTask(task) });
    }

    if (task.source === "server") {
      // The live runner observes the flag, removes the partial targetDir and
      // writes the cancelled status itself. No runner → mark directly.
      if (!cancelServerImport(id)) {
        await db.uploadTask.update({ where: { id }, data: { status: "cancelled" } });
      }
    } else {
      // Browser upload: mark cancelled so chunk requests start rejecting.
      await db.uploadTask.update({ where: { id }, data: { status: "cancelled" } });
      const systemConfig = await db.systemConfig.findFirst();
      const userDatabasePath = (systemConfig as any)?.userDatabasePath;
      const username = (await db.user.findUnique({ where: { id: userId }, select: { username: true } }))?.username;
      if (userDatabasePath && username) {
        await rm(uploadTempDir(userDatabasePath, username, id), { recursive: true, force: true }).catch(() => {});
      }
    }

    const updated = await db.uploadTask.findUnique({ where: { id } });
    return NextResponse.json({ success: true, data: updated ? serializeTask(updated) : null });
  } catch (error) {
    console.error("Error cancelling upload task:", error);
    return NextResponse.json({ error: "Failed to cancel upload task" }, { status: 500 });
  }
}
