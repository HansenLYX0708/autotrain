import { NextRequest, NextResponse } from "next/server";
import { writeFile, mkdir } from "fs/promises";
import { existsSync } from "fs";
import { join } from "path";
import { sessions } from "../../../auth/route";
import { db } from "@/lib/db";

// POST /api/datasets/upload/chunk - Upload a single chunk
export async function POST(request: NextRequest) {
  try {
    const token = request.cookies.get("auth-token")?.value;
    if (!token || !sessions.has(token)) {
      return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
    }

    const session = sessions.get(token)!;
    const userId = session.userId;

    const user = await db.user.findUnique({
      where: { id: userId },
      select: { id: true, username: true },
    });

    if (!user) {
      return NextResponse.json({ error: "User not found" }, { status: 404 });
    }

    const systemConfig = await db.systemConfig.findFirst();
    const userDatabasePath = (systemConfig as any)?.userDatabasePath;

    if (!userDatabasePath) {
      return NextResponse.json(
        { error: "User database path not configured" },
        { status: 500 }
      );
    }

    // Parse multipart form data
    const formData = await request.formData();
    const uploadId = formData.get("uploadId") as string;
    const relativePath = formData.get("relativePath") as string;
    const chunkIndex = parseInt(formData.get("chunkIndex") as string);
    const totalChunks = parseInt(formData.get("totalChunks") as string);
    const chunk = formData.get("chunk") as Blob;

    if (!uploadId || !relativePath || isNaN(chunkIndex) || !chunk) {
      return NextResponse.json(
        { error: "Missing required fields" },
        { status: 400 }
      );
    }

    // Security: ensure uploadId is safe (UUID format)
    if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(uploadId)) {
      return NextResponse.json({ error: "Invalid upload ID" }, { status: 400 });
    }

    // Temp directory for this file
    const fileTempDir = join(userDatabasePath, user.username, ".uploads", uploadId, relativePath.replace(/[/\\]/g, "___"));
    
    if (!existsSync(fileTempDir)) {
      await mkdir(fileTempDir, { recursive: true });
    }

    // Save chunk
    const chunkPath = join(fileTempDir, `${chunkIndex}.chunk`);
    const bytes = await chunk.arrayBuffer();
    await writeFile(chunkPath, Buffer.from(bytes));

    return NextResponse.json({
      success: true,
      chunkIndex,
      relativePath,
      uploaded: true,
    });

  } catch (error) {
    console.error("Error uploading chunk:", error);
    return NextResponse.json(
      { error: "Failed to upload chunk", message: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}
