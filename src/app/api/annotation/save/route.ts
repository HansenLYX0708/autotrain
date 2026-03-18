import { NextRequest, NextResponse } from "next/server";
import * as fs from "fs";
import * as path from "path";

// POST /api/annotation/save - Save annotation JSON file
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { fileName, annotation } = body;

    if (!fileName || !annotation) {
      return NextResponse.json(
        { error: "File name and annotation are required" },
        { status: 400 }
      );
    }

    // Save to the annotations folder in the project
    const annotationsDir = path.join(process.cwd(), "annotations");

    // Create directory if not exists
    if (!fs.existsSync(annotationsDir)) {
      fs.mkdirSync(annotationsDir, { recursive: true });
    }

    // Sanitize file name
    const safeFileName = fileName.replace(/[^a-zA-Z0-9_.-]/g, "_");
    const filePath = path.join(annotationsDir, safeFileName);

    // Write JSON file
    fs.writeFileSync(filePath, JSON.stringify(annotation, null, 2), "utf-8");

    return NextResponse.json({
      success: true,
      path: `annotations/${safeFileName}`,
    });
  } catch (error) {
    console.error("Error saving annotation:", error);
    return NextResponse.json(
      { error: "Failed to save annotation", message: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 }
    );
  }
}

// GET /api/annotation/save - List saved annotations
export async function GET() {
  try {
    const annotationsDir = path.join(process.cwd(), "annotations");

    if (!fs.existsSync(annotationsDir)) {
      return NextResponse.json({ files: [] });
    }

    const files = fs.readdirSync(annotationsDir)
      .filter(f => f.endsWith(".json"))
      .map(f => ({
        name: f,
        path: `annotations/${f}`,
        modified: fs.statSync(path.join(annotationsDir, f)).mtime,
      }));

    return NextResponse.json({ files });
  } catch (error) {
    console.error("Error listing annotations:", error);
    return NextResponse.json(
      { error: "Failed to list annotations" },
      { status: 500 }
    );
  }
}
