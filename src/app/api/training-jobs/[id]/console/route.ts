import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { notFoundOrDenied, requireOwnedScope } from "@/lib/auth";
import { getWorkDir } from "@/lib/frameworks";
import * as fs from "fs";
import { Readable } from "stream";
import {
  jobConsoleLogPath,
  readAmpSummary,
  readBefore,
  readFrom,
  readTail,
  searchFile,
} from "@/lib/job-console";

const DEFAULT_MAX_BYTES = 256 * 1024;
const MIN_MAX_BYTES = 4096;
const MAX_MAX_BYTES = 1024 * 1024;

/**
 * GET /api/training-jobs/[id]/console - read the job's autotrain-console.log.
 *
 * The console file is the verbatim stdout+stderr capture written next to the
 * job's checkpoints (see `src/lib/job-console.ts`). It exists only for runs
 * started after this feature landed; older jobs answer `available: false`
 * (still 200) so the UI can fall back to the parsed TrainingLog rows.
 *
 * Modes (query):
 *   download=1        stream the whole file as an attachment
 *   q=<text>          case-insensitive line search over the whole file
 *   offset=<bytes>    forward read for live-tailing a running job
 *   before=<bytes>    backwards read for "Load earlier"
 *   (default)         tail of the file
 *   maxBytes          read window, 4 KB .. 1 MB (default 256 KB)
 *
 * Range responses never split a line: the trailing incomplete line is held
 * back unless the job has finished and the read reached EOF.
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    const scope = await requireOwnedScope(request, id);
    if (scope instanceof NextResponse) return scope;

    const job = await db.trainingJob.findFirst({
      where: scope.where,
      select: {
        id: true,
        name: true,
        status: true,
        outputDir: true,
        project: { select: { framework: true } },
      },
    });
    if (!job) return notFoundOrDenied("Training job");

    const { searchParams } = new URL(request.url);

    // Parse numeric params up front so a malformed value is a 400 regardless
    // of which mode consumes it.
    const maxBytesRaw = searchParams.get("maxBytes");
    let maxBytes = DEFAULT_MAX_BYTES;
    if (maxBytesRaw !== null) {
      const n = Number(maxBytesRaw);
      if (!Number.isFinite(n)) {
        return NextResponse.json({ error: "Invalid maxBytes" }, { status: 400 });
      }
      maxBytes = Math.min(MAX_MAX_BYTES, Math.max(MIN_MAX_BYTES, Math.floor(n)));
    }
    const parseOffset = (name: string): number | null | NextResponse => {
      const raw = searchParams.get(name);
      if (raw === null) return null;
      const n = Number(raw);
      if (!Number.isFinite(n) || n < 0 || !Number.isInteger(n)) {
        return NextResponse.json({ error: `Invalid ${name}` }, { status: 400 });
      }
      return n;
    };
    const offset = parseOffset("offset");
    if (offset instanceof NextResponse) return offset;
    const before = parseOffset("before");
    if (before instanceof NextResponse) return before;

    const systemConfig = await db.systemConfig.findFirst();
    const workDir = getWorkDir(job.project?.framework, systemConfig) || "";
    const file = jobConsoleLogPath(job.outputDir, workDir);

    const unavailable = () =>
      NextResponse.json({ available: false, size: 0, amp: null });
    if (!file || !fs.existsSync(file)) return unavailable();

    // Whole-file download as a .log attachment.
    if (searchParams.get("download") === "1") {
      const slug =
        job.name
          .toLowerCase()
          .replace(/[^a-z0-9-_]+/g, "-")
          .replace(/^-+|-+$/g, "") || "job";
      const stream = Readable.toWeb(fs.createReadStream(file)) as ReadableStream;
      return new Response(stream, {
        headers: {
          "Content-Type": "text/plain; charset=utf-8",
          "Content-Disposition": `attachment; filename="${slug}-console.log"`,
        },
      });
    }

    const amp = await readAmpSummary(file);
    const size = fs.statSync(file).size;

    // Full-file search.
    const q = searchParams.get("q");
    if (q !== null) {
      const result = await searchFile(file, q);
      return NextResponse.json({
        available: true,
        path: file,
        size,
        ...result,
        amp,
      });
    }

    // Range reads.
    const range =
      offset !== null
        ? await readFrom(file, offset, maxBytes, job.status !== "running")
        : before !== null
          ? await readBefore(file, before, maxBytes)
          : await readTail(file, maxBytes);

    return NextResponse.json({
      available: true,
      path: file,
      ...range,
      amp,
    });
  } catch (error) {
    console.error("Error reading console log:", error);
    return NextResponse.json(
      { error: "Failed to read console log" },
      { status: 500 }
    );
  }
}
