import { NextResponse } from "next/server";
import { db } from "@/lib/db";

interface GpuUsageInfo {
  gpuId: number;
  jobIds: string[];
  jobNames: string[];
}

/**
 * GET /api/system/gpu-usage
 * Returns GPU usage information from running training jobs
 */
export async function GET() {
  try {
    // Get all running training jobs
    const runningJobs = await db.trainingJob.findMany({
      where: { status: "running" },
      select: {
        id: true,
        name: true,
        trainingParams: true,
      },
    });

    // Extract GPU IDs from running jobs
    const gpuUsageMap = new Map<number, GpuUsageInfo>();

    for (const job of runningJobs) {
      if (!job.trainingParams) continue;

      try {
        const params = JSON.parse(job.trainingParams as string);
        const gpuIdsStr = params.gpuIds as string | undefined;
        
        if (gpuIdsStr) {
          // Parse GPU IDs (can be single "0" or multiple "0,1")
          const gpuIds = gpuIdsStr.split(',').map((id: string) => parseInt(id.trim(), 10)).filter((id: number) => !isNaN(id));
          
          for (const gpuId of gpuIds) {
            if (!gpuUsageMap.has(gpuId)) {
              gpuUsageMap.set(gpuId, {
                gpuId,
                jobIds: [],
                jobNames: [],
              });
            }
            const usage = gpuUsageMap.get(gpuId)!;
            usage.jobIds.push(job.id);
            usage.jobNames.push(job.name);
          }
        }
      } catch {
        // Ignore parse errors
        console.error(`Failed to parse trainingParams for job ${job.id}`);
      }
    }

    const gpuUsage = Array.from(gpuUsageMap.values());

    return NextResponse.json({
      success: true,
      data: {
        runningJobsCount: runningJobs.length,
        gpuUsage,
        occupiedGpuIds: gpuUsage.map((g) => g.gpuId),
      },
    });
  } catch (error) {
    console.error("Error fetching GPU usage:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to fetch GPU usage information",
        data: {
          runningJobsCount: 0,
          gpuUsage: [],
          occupiedGpuIds: [],
        },
      },
      { status: 500 }
    );
  }
}
