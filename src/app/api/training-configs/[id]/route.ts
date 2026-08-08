import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { requireAuth } from "@/lib/auth";
import { getWorkDir } from "@/lib/frameworks";
import {
  asConfigFramework,
  defaultTrainingParams,
  parseTrainingParams,
  trainingParamsToColumns,
} from "@/lib/training-yaml";
import * as fs from "fs";
import * as path from "path";

/**
 * Load a training config the caller is allowed to touch.
 *
 * Every handler in this file previously ran with no authentication at all, so
 * any unauthenticated client could read, rewrite, or delete another user's
 * configs by id. Admins keep global access; everyone else is scoped to rows
 * they own (or legacy rows with no owner).
 */
async function loadOwnedConfig(request: NextRequest, id: string) {
  const auth = await requireAuth(request);
  if (auth instanceof NextResponse) return { error: auth };

  const { userId, role } = auth;
  const config = await db.trainingConfig.findFirst({
    where:
      role === "admin"
        ? { id }
        : { id, OR: [{ userId }, { userId: null }] },
    include: {
      project: { select: { id: true, name: true, framework: true } },
    },
  });

  if (!config) {
    return {
      error: NextResponse.json(
        { error: "Training config not found or access denied" },
        { status: 404 }
      ),
    };
  }

  return { auth, config };
}

/** Mirror of the naming used by the import route, so edits overwrite in place. */
function configFileName(name: string): string {
  return `${name
    .replace(/\s+/g, "_")
    .replace(/[^a-zA-Z0-9_.-]/g, "")
    .toLowerCase()}.yml`;
}

/**
 * Write the config YAML back to the user's on-disk config folder so the file
 * and the DB row do not drift after an edit.
 * Best-effort: a filesystem problem must not fail the update.
 */
async function persistYamlToDisk(
  userId: string,
  framework: string,
  name: string,
  yamlConfig: string
): Promise<string | null> {
  try {
    const systemConfig = await db.systemConfig.findFirst();
    const userConfigsPath = systemConfig?.userConfigsPath;
    const workDir = getWorkDir(framework, systemConfig);
    const user = await db.user.findUnique({
      where: { id: userId },
      select: { username: true },
    });

    let dir = "";
    let relPath = "";
    const fileName = configFileName(name);
    if (userConfigsPath && user?.username) {
      dir = path.join(userConfigsPath, user.username, "training");
      relPath = path.join(user.username, "training", fileName);
    } else if (workDir) {
      dir = path.join(workDir, "configs", "autotrain", "training", "user");
      relPath = `configs/autotrain/training/user/${fileName}`;
    } else {
      return null;
    }

    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, fileName), yamlConfig, "utf-8");
    return relPath;
  } catch (error) {
    console.warn("[training-configs] Failed to write config to disk:", error);
    return null;
  }
}

// GET /api/training-configs/[id] - Get a single training config
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const result = await loadOwnedConfig(request, id);
    if (result.error) return result.error;

    const config = await db.trainingConfig.findUnique({
      where: { id },
      include: {
        project: { select: { id: true, name: true, framework: true } },
        trainingJobs: {
          select: { id: true, name: true, status: true, createdAt: true },
          orderBy: { createdAt: "desc" },
          take: 10,
        },
        _count: { select: { trainingJobs: true } },
      },
    });

    return NextResponse.json({ success: true, data: config });
  } catch (error) {
    console.error("Error fetching training config:", error);
    return NextResponse.json(
      { error: "Failed to fetch training config" },
      { status: 500 }
    );
  }
}

// PUT /api/training-configs/[id] - Update a training config
export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const result = await loadOwnedConfig(request, id);
    if (result.error) return result.error;
    const { auth, config: existingConfig } = result;

    const body = await request.json();
    const framework = asConfigFramework(existingConfig.project?.framework);

    const updateData: Record<string, unknown> = {};
    if (typeof body.name === "string" && body.name.trim()) {
      updateData.name = body.name.trim();
    }

    // `yamlConfig` is the source of truth. When it is supplied we re-derive the
    // whole set of display columns from it rather than trusting individually
    // posted scalars, which is what used to let the row drift away from the
    // YAML that actually gets trained on.
    if (typeof body.yamlConfig === "string") {
      const yamlConfig = body.yamlConfig;
      const resolved = {
        ...defaultTrainingParams(framework),
        ...parseTrainingParams(framework, yamlConfig),
      };
      Object.assign(updateData, trainingParamsToColumns(framework, resolved));
      updateData.yamlConfig = yamlConfig;

      await persistYamlToDisk(
        auth.userId,
        existingConfig.project?.framework ?? "PaddleDetection",
        (updateData.name as string) ?? existingConfig.name,
        yamlConfig
      );
    } else {
      // Legacy scalar-only update path, kept for callers that tweak one field.
      const scalarFields = [
        "epoch", "batchSize", "baseLr", "momentum", "weightDecay",
        "scheduler", "warmupEpochs", "maxEpochs", "iters", "saveInterval",
        "workerNum", "evalHeight", "evalWidth", "useGpu", "logIter",
        "saveDir", "snapshotEpoch", "outputDir", "weights", "pretrainWeights",
      ] as const;
      for (const field of scalarFields) {
        if (body[field] !== undefined) updateData[field] = body[field];
      }
    }

    if (Object.keys(updateData).length === 0) {
      return NextResponse.json({ error: "Nothing to update" }, { status: 400 });
    }

    const config = await db.trainingConfig.update({
      where: { id },
      data: updateData,
      include: {
        project: { select: { id: true, name: true, framework: true } },
      },
    });

    return NextResponse.json({ success: true, data: config });
  } catch (error) {
    console.error("Error updating training config:", error);
    return NextResponse.json(
      { error: "Failed to update training config" },
      { status: 500 }
    );
  }
}

// DELETE /api/training-configs/[id] - Delete a training config
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const result = await loadOwnedConfig(request, id);
    if (result.error) return result.error;

    const usage = await db.trainingJob.count({ where: { configId: id } });
    if (usage > 0) {
      return NextResponse.json(
        {
          error: `Cannot delete: this config is used by ${usage} training job(s). Delete those jobs first.`,
        },
        { status: 400 }
      );
    }

    await db.trainingConfig.delete({ where: { id } });

    return NextResponse.json({
      success: true,
      message: "Training config deleted successfully",
    });
  } catch (error) {
    console.error("Error deleting training config:", error);
    return NextResponse.json(
      { error: "Failed to delete training config" },
      { status: 500 }
    );
  }
}
