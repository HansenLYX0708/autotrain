import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { requireAuth, buildUserFilter } from "@/lib/auth";

interface DashboardStats {
  totalProjects: number;
  totalDatasets: number;
  totalModels: number;
  runningTrainingJobs: number;
  completedTrainingJobs: number;
  pendingTrainingJobs: number;
  failedTrainingJobs: number;
  gpuMetrics: GpuMetricInfo[];
}

interface GpuMetricInfo {
  gpuId: number;
  utilization: number;
  memoryUsed: number;
  memoryTotal: number;
  temperature: number;
  timestamp: Date;
}

/**
 * GET /api/dashboard/stats
 * Returns aggregated dashboard statistics filtered by user access.
 */
export async function GET(request: NextRequest) {
  try {
    // Check authentication
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;

    const { userId, role } = auth;

    // Build user filter for regular users (admin sees all)
    const userFilter = role === 'admin' ? {} : { userId: userId };
    const [
      projectsCount,
      datasetsCount,
      modelsCount,
      runningJobsCount,
      completedJobsCount,
      pendingJobsCount,
      failedJobsCount,
      latestGpuMetrics,
    ] = await Promise.all([
      // Total projects count (filtered by user)
      db.project.count({
        where: { 
          status: "active",
          ...userFilter,
        },
      }),
      // Total datasets count (filtered by user)
      db.dataset.count({
        where: userFilter,
      }),
      // Total models count (filtered by user)
      db.model.count({
        where: userFilter,
      }),
      // Running training jobs count (filtered by user)
      db.trainingJob.count({
        where: { 
          status: "running",
          ...userFilter,
        },
      }),
      // Completed training jobs count (filtered by user)
      db.trainingJob.count({
        where: { 
          status: "completed",
          ...userFilter,
        },
      }),
      // Pending training jobs count (filtered by user)
      db.trainingJob.count({
        where: { 
          status: "pending",
          ...userFilter,
        },
      }),
      // Failed training jobs count (filtered by user)
      db.trainingJob.count({
        where: { 
          status: "failed",
          ...userFilter,
        },
      }),
      // Get latest GPU metrics for each GPU
      db.gpuMetric.groupBy({
        by: ["gpuId"],
        _max: {
          timestamp: true,
        },
      }),
    ]);

    // Fetch the actual latest GPU metric data
    let gpuMetrics: GpuMetricInfo[] = [];

    if (latestGpuMetrics.length > 0) {
      const gpuMetricPromises = latestGpuMetrics.map(async (metric) => {
        const latestMetric = await db.gpuMetric.findFirst({
          where: {
            gpuId: metric.gpuId,
            timestamp: metric._max.timestamp,
          },
          orderBy: {
            timestamp: "desc",
          },
        });
        return latestMetric;
      });

      const gpuMetricResults = await Promise.all(gpuMetricPromises);
      gpuMetrics = gpuMetricResults
        .filter((m): m is NonNullable<typeof m> => m !== null)
        .map((m) => ({
          gpuId: m.gpuId,
          utilization: m.utilization,
          memoryUsed: m.memoryUsed,
          memoryTotal: m.memoryTotal,
          temperature: m.temperature,
          timestamp: m.timestamp,
        }));
    }

    const stats: DashboardStats = {
      totalProjects: projectsCount,
      totalDatasets: datasetsCount,
      totalModels: modelsCount,
      runningTrainingJobs: runningJobsCount,
      completedTrainingJobs: completedJobsCount,
      pendingTrainingJobs: pendingJobsCount,
      failedTrainingJobs: failedJobsCount,
      gpuMetrics,
    };

    return NextResponse.json({
      success: true,
      data: stats,
    });
  } catch (error) {
    console.error("Error fetching dashboard stats:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to fetch dashboard statistics",
      },
      { status: 500 }
    );
  }
}
