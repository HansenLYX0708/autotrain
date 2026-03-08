import { NextResponse } from "next/server";
import { db } from "@/lib/db";

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
 * Returns aggregated dashboard statistics.
 */
export async function GET() {
  try {
    // Execute all count queries in parallel for better performance
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
      // Total projects count
      db.project.count({
        where: { status: "active" },
      }),
      // Total datasets count
      db.dataset.count(),
      // Total models count
      db.model.count(),
      // Running training jobs count
      db.trainingJob.count({
        where: { status: "running" },
      }),
      // Completed training jobs count
      db.trainingJob.count({
        where: { status: "completed" },
      }),
      // Pending training jobs count
      db.trainingJob.count({
        where: { status: "pending" },
      }),
      // Failed training jobs count
      db.trainingJob.count({
        where: { status: "failed" },
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
