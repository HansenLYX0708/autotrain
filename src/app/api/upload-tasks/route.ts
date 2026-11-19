import { NextRequest, NextResponse } from "next/server";
import { requireAuth } from "@/lib/auth";
import { db } from "@/lib/db";
import { serializeTask } from "@/lib/upload-tasks";
import { markInterruptedServerTasks } from "@/lib/upload-server-runner";

export const runtime = "nodejs";

// GET /api/upload-tasks - the caller's tasks, newest first (max 100)
export async function GET(request: NextRequest) {
  try {
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;
    const { userId } = auth;

    // Reconcile tasks orphaned by a server restart before listing.
    await markInterruptedServerTasks();

    const tasks = await db.uploadTask.findMany({
      where: { userId },
      orderBy: { createdAt: "desc" },
      take: 100,
    });
    return NextResponse.json({ success: true, data: tasks.map(serializeTask) });
  } catch (error) {
    console.error("Error listing upload tasks:", error);
    return NextResponse.json({ error: "Failed to list upload tasks" }, { status: 500 });
  }
}
