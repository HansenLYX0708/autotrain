import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { requireAuth } from "@/lib/auth";
import { getWorkDir } from "@/lib/frameworks";
import { asConfigFramework } from "@/lib/training-yaml";
import { defaultModelParams, modelParamsToColumns, parseModelParams } from "@/lib/model-yaml";
import * as fs from "fs";
import * as path from "path";

/**
 * Load a model the caller is allowed to touch.
 *
 * These handlers previously ran with no authentication, so any unauthenticated
 * client could read, rewrite, or delete another user's models by id.
 */
async function loadOwnedModel(request: NextRequest, id: string) {
  const auth = await requireAuth(request);
  if (auth instanceof NextResponse) return { error: auth };

  const { userId, role } = auth;
  const model = await db.model.findFirst({
    where: role === "admin" ? { id } : { id, OR: [{ userId }, { userId: null }] },
    include: {
      project: { select: { id: true, name: true, framework: true, status: true } },
    },
  });

  if (!model) {
    return {
      error: NextResponse.json(
        { success: false, error: "Model not found or access denied" },
        { status: 404 }
      ),
    };
  }

  return { auth, model };
}

/** Mirror of the naming used by the import route, so edits overwrite in place. */
function modelFileName(name: string): string {
  return `${name
    .replace(/\s+/g, "_")
    .replace(/[^a-zA-Z0-9_.-]/g, "")
    .toLowerCase()}.yml`;
}

/**
 * Write the model YAML back to the user's on-disk config folder so the file and
 * the DB row do not drift after an edit. Best-effort.
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
    const fileName = modelFileName(name);
    if (userConfigsPath && user?.username) {
      dir = path.join(userConfigsPath, user.username, "models");
      relPath = path.join(user.username, "models", fileName);
    } else if (workDir) {
      dir = path.join(workDir, "configs", "autotrain", "models", "user");
      relPath = `configs/autotrain/models/user/${fileName}`;
    } else {
      return null;
    }

    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, fileName), yamlConfig, "utf-8");
    return relPath;
  } catch (error) {
    console.warn("[models] Failed to write model config to disk:", error);
    return null;
  }
}

// GET /api/models/[id] - Get a single model by ID
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const result = await loadOwnedModel(request, id);
    if (result.error) return result.error;

    const model = await db.model.findUnique({
      where: { id },
      include: {
        project: {
          select: { id: true, name: true, framework: true, status: true, description: true },
        },
        trainingJobs: {
          select: { id: true, name: true, status: true, createdAt: true },
          orderBy: { createdAt: "desc" },
          take: 10,
        },
      },
    });

    return NextResponse.json({ success: true, data: model });
  } catch (error) {
    console.error("Error fetching model:", error);
    return NextResponse.json(
      { success: false, error: "Failed to fetch model" },
      { status: 500 }
    );
  }
}

// PUT /api/models/[id] - Update a model
export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const result = await loadOwnedModel(request, id);
    if (result.error) return result.error;
    const { auth, model: existingModel } = result;

    const body = await request.json();
    const framework = asConfigFramework(existingModel.project?.framework);

    const updateData: Record<string, unknown> = {};
    if (typeof body.name === "string" && body.name.trim()) updateData.name = body.name.trim();
    if (body.description !== undefined) updateData.description = body.description;

    if (typeof body.yamlConfig === "string") {
      // `yamlConfig` is the source of truth; derive the display columns from it.
      // The old editor did the opposite — it regenerated YAML from the form's
      // scalar fields on every save, which silently destroyed any imported or
      // hand-written config the moment a user opened the dialog.
      const resolved = {
        ...defaultModelParams(framework),
        ...parseModelParams(framework, body.yamlConfig),
      };
      Object.assign(updateData, modelParamsToColumns(resolved));
      updateData.yamlConfig = body.yamlConfig;

      await persistYamlToDisk(
        auth.userId,
        existingModel.project?.framework ?? "PaddleDetection",
        (updateData.name as string) ?? existingModel.name,
        body.yamlConfig
      );
    } else {
      const scalarFields = [
        "architecture", "backbone", "neck", "head", "numClasses",
        "normType", "useEma", "emaDecay", "depthMult", "widthMult", "pretrainWeights",
      ] as const;
      for (const field of scalarFields) {
        if (body[field] !== undefined) updateData[field] = body[field];
      }
    }

    if (Object.keys(updateData).length === 0) {
      return NextResponse.json(
        { success: false, error: "Nothing to update" },
        { status: 400 }
      );
    }

    const updatedModel = await db.model.update({
      where: { id },
      data: updateData,
      include: {
        project: { select: { id: true, name: true, framework: true, status: true } },
      },
    });

    return NextResponse.json({ success: true, data: updatedModel });
  } catch (error) {
    console.error("Error updating model:", error);
    return NextResponse.json(
      { success: false, error: "Failed to update model" },
      { status: 500 }
    );
  }
}

// DELETE /api/models/[id] - Delete a model
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const result = await loadOwnedModel(request, id);
    if (result.error) return result.error;

    // Deleting a model cascades to its training jobs (see schema), so make the
    // blast radius explicit instead of silently dropping training history.
    const jobCount = await db.trainingJob.count({ where: { modelId: id } });
    if (jobCount > 0) {
      return NextResponse.json(
        {
          success: false,
          error: `Cannot delete: this model is used by ${jobCount} training job(s). Delete those jobs first.`,
        },
        { status: 400 }
      );
    }

    await db.model.delete({ where: { id } });

    return NextResponse.json({
      success: true,
      message: "Model deleted successfully",
      data: { id },
    });
  } catch (error) {
    console.error("Error deleting model:", error);
    return NextResponse.json(
      { success: false, error: "Failed to delete model" },
      { status: 500 }
    );
  }
}
