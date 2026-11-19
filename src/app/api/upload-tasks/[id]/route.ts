import { NextRequest, NextResponse } from "next/server";
import { requireAuth } from "@/lib/auth";
import { db } from "@/lib/db";
import { serializeTask } from "@/lib/upload-tasks";
import { markInterruptedServerTasks } from "@/lib/upload-server-runner";

export const runtime = "nodejs";

const ACTIVE_STATUSES = ["pending", "uploading", "finalizing"];

// GET /api/upload-tasks/[id]
export async function GET(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  try {
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;
    const { userId } = auth;
    const { id } = await params;

    await markInterruptedServerTasks([id]);

    const task = await db.uploadTask.findFirst({ where: { id, userId } });
    if (!task) return NextResponse.json({ error: "Upload task not found" }, { status: 404 });
    return NextResponse.json({ success: true, data: serializeTask(task) });
  } catch (error) {
    console.error("Error getting upload task:", error);
    return NextResponse.json({ error: "Failed to get upload task" }, { status: 500 });
  }
}

// DELETE /api/upload-tasks/[id] - remove the record (finished tasks only)
export async function DELETE(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  try {
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;
    const { userId } = auth;
    const { id } = await params;

    const task = await db.uploadTask.findFirst({ where: { id, userId } });
    if (!task) return NextResponse.json({ error: "Upload task not found" }, { status: 404 });
    if (ACTIVE_STATUSES.includes(task.status)) {
      return NextResponse.json({ error: `Task is ${task.status}; cancel it first` }, { status: 409 });
    }
    await db.uploadTask.delete({ where: { id } });
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Error deleting upload task:", error);
    return NextResponse.json({ error: "Failed to delete upload task" }, { status: 500 });
  }
}
