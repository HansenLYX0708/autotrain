import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
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

// DELETE /api/training-jobs/[id]/files - Delete all checkpoint and export model files for a job
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    // Get the job details
    const job = await db.trainingJob.findUnique({
      where: { id },
      include: {
        project: {
          select: {
            user: {
              select: {
                username: true,
              },
            },
          },
        },
      },
    });

    if (!job) {
      return NextResponse.json(
        { error: "Training job not found" },
        { status: 404 }
      );
    }

    // Get system config for userDatabasePath
    const systemConfig = await db.systemConfig.findFirst();
    const userDatabasePath = (systemConfig as any)?.userDatabasePath;

    if (!userDatabasePath) {
      return NextResponse.json(
        { error: "User database path not configured" },
        { status: 500 }
      );
    }

    const username = job.project?.user?.username;
    if (!username) {
      return NextResponse.json(
        { error: "Job owner not found" },
        { status: 500 }
      );
    }

    // Build the job folder path: {userDatabasePath}/{username}/jobs/{jobName}
    const jobFolderPath = join(userDatabasePath, username, "jobs", job.name);

    // Check if job directory exists
    if (!existsSync(jobFolderPath)) {
      return NextResponse.json({
        success: true,
        message: `Job folder does not exist: ${jobFolderPath}`,
        path: jobFolderPath,
      });
    }

    // Delete the job directory recursively
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
      { error: "Failed to delete job files", message: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
