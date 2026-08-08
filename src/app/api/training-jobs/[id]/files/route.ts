import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { notFoundOrDenied, requireOwnedScope } from "@/lib/auth";
import { isInside, toSafeSlug } from "@/lib/safe-path";
import { readdir, rmdir, unlink } from "fs/promises";
import { existsSync } from "fs";
import { join } from "path";

// Recursively delete a directory and all its contents
async function deleteDirectory(dirPath: string): Promise<void> {
  if (!existsSync(dirPath)) {
    return;
  }

  const entries = await readdir(dirPath, { withFileTypes: true });

  for (const entry of entries) {
    const fullPath = join(dirPath, entry.name);
    if (entry.isDirectory()) {
      await deleteDirectory(fullPath);
    } else {
      await unlink(fullPath);
    }
  }

  await rmdir(dirPath);
}

/**
 * Resolve the on-disk folder for a job's outputs.
 *
 * The folder is named after the *slug* of the job name, which is what
 * `POST /api/training-jobs` uses when it creates the job. Reading the raw
 * `job.name` here (as this route used to) meant any job whose name contained a
 * space or capital letter never matched its own folder, so "delete files"
 * silently did nothing.
 */
function jobFolderName(job: { name: string; trainingParams: string | null }): string {
  try {
    if (job.trainingParams) {
      const parsed = JSON.parse(job.trainingParams) as { jobSlug?: unknown };
      if (typeof parsed.jobSlug === "string" && parsed.jobSlug) {
        return toSafeSlug(parsed.jobSlug, "job");
      }
    }
  } catch {
    // Fall through to deriving it from the name.
  }
  return toSafeSlug(job.name, "job");
}

// DELETE /api/training-jobs/[id]/files - Delete all checkpoint and export model files for a job
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    // Authenticate and scope the lookup. Previously this route had no auth at
    // all: any anonymous caller could recursively delete another user's
    // training outputs just by guessing a job id.
    const scope = await requireOwnedScope(request, id);
    if (scope instanceof NextResponse) return scope;

    const job = await db.trainingJob.findFirst({
      where: scope.where,
      select: {
        id: true,
        name: true,
        status: true,
        trainingParams: true,
        user: { select: { username: true } },
        project: { select: { user: { select: { username: true } } } },
      },
    });

    if (!job) return notFoundOrDenied("Training job");

    if (job.status === "running") {
      return NextResponse.json(
        { success: false, error: "Cannot delete files while the job is running. Stop it first." },
        { status: 409 }
      );
    }

    const systemConfig = await db.systemConfig.findFirst();
    const userDatabasePath = systemConfig?.userDatabasePath;

    if (!userDatabasePath) {
      return NextResponse.json(
        { success: false, error: "User database path not configured" },
        { status: 500 }
      );
    }

    // Prefer the job's own owner; fall back to the project owner for legacy
    // rows created before jobs carried a userId.
    const username = job.user?.username ?? job.project?.user?.username;
    if (!username) {
      return NextResponse.json(
        { success: false, error: "Job owner not found" },
        { status: 500 }
      );
    }

    const jobsRoot = join(userDatabasePath, toSafeSlug(username, "user"), "jobs");
    const jobFolderPath = join(jobsRoot, jobFolderName(job));

    // Defence in depth: even with a slugified name, never delete anything that
    // is not strictly beneath this user's jobs directory.
    if (!isInside(jobsRoot, jobFolderPath) || jobFolderPath === jobsRoot) {
      console.error(`[Job Files Delete] Refusing unsafe path: ${jobFolderPath}`);
      return NextResponse.json(
        { success: false, error: "Refusing to delete an unsafe path" },
        { status: 400 }
      );
    }

    if (!existsSync(jobFolderPath)) {
      return NextResponse.json({
        success: true,
        message: `Job folder does not exist: ${jobFolderPath}`,
        path: jobFolderPath,
      });
    }

    await deleteDirectory(jobFolderPath);

    console.log(`[Job Files Delete] Deleted job folder: ${jobFolderPath}`);

    return NextResponse.json({
      success: true,
      message: `All checkpoint and export model files for job "${job.name}" deleted successfully`,
      path: jobFolderPath,
    });
  } catch (error) {
    console.error("Error deleting job files:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to delete job files",
        message: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
