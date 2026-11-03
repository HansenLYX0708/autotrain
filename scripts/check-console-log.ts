/**
 * Console-log helper checks.
 *
 *   npx tsx scripts/check-console-log.ts
 *
 * Covers `src/lib/job-console.ts` (AMP fold, incremental summary, range
 * readers, search) and `src/lib/chart-decimate.ts` (min/max decimation) —
 * the pieces the console API route is a thin wrapper over.
 */
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import {
  emptyAmp,
  foldAmpLine,
  readAmpSummary,
  readBefore,
  readFrom,
  readTail,
  runHeader,
  searchFile,
} from "../src/lib/job-console";
import { decimateMinMax } from "../src/lib/chart-decimate";

let failures = 0;
function check(name: string, ok: boolean, detail?: unknown) {
  if (ok) {
    console.log(`  PASS  ${name}`);
  } else {
    failures += 1;
    console.log(`  FAIL  ${name}`, detail ?? "");
  }
}
function eq(name: string, got: unknown, want: unknown) {
  check(name, JSON.stringify(got) === JSON.stringify(want), { got, want });
}

const dir = fs.mkdtempSync(path.join(os.tmpdir(), "console-check-"));
const file = path.join(dir, "autotrain-console.log");

async function main() {
  console.log("[1] foldAmpLine");
  {
    let s = emptyAmp();
    s = foldAmpLine(s, "noise line");
    eq("non-matching line ignored", s, emptyAmp());

    s = foldAmpLine(s, "Found inf or nan, current scale is: 1024.0, decrease to: 1024.0*0.5");
    eq("first event", s, { count: 1, initialScale: 1024, currentScale: 512, minScale: 512, lastLine: "Found inf or nan, current scale is: 1024.0, decrease to: 1024.0*0.5" });

    s = foldAmpLine(s, "Found inf or nan, current scale is: 512.0, decrease to: 512.0*0.5");
    eq("second event", { ...s, lastLine: null }, { count: 2, initialScale: 1024, currentScale: 256, minScale: 256, lastLine: null });

    s = foldAmpLine(s, "Found inf or nan, current scale is: 128");
    eq("no multiplier -> ratio 0.5", { ...s, lastLine: null }, { count: 3, initialScale: 1024, currentScale: 64, minScale: 64, lastLine: null });

    const beforeReset = s;
    s = foldAmpLine(s, runHeader(new Date()).trim());
    eq("run header resets", s, emptyAmp());
    check("reset dropped previous", beforeReset.count === 3);

    s = foldAmpLine(s, "Found inf or nan, current scale is: banana");
    eq("non-finite scale ignored", s, emptyAmp());
  }

  console.log("[2] readAmpSummary incremental, split line, truncation");
  {
    fs.writeFileSync(file, runHeader(new Date("2026-10-29T00:00:00Z")));
    fs.appendFileSync(file, "Found inf or nan, current scale is: 1024.0, decrease to: 1024.0*0.5\n");
    let s = await readAmpSummary(file);
    eq("first scan", { ...s, lastLine: null }, { count: 1, initialScale: 1024, currentScale: 512, minScale: 512, lastLine: null });

    // Split the next AMP line across two appends.
    fs.appendFileSync(file, "Found inf or nan, current scale is: 512.");
    s = await readAmpSummary(file);
    eq("partial line held back", s.count, 1);
    fs.appendFileSync(file, "0, decrease to: 512.0*0.5\nunrelated\n");
    s = await readAmpSummary(file);
    eq("split line folded once complete", { ...s, lastLine: null }, { count: 2, initialScale: 1024, currentScale: 256, minScale: 256, lastLine: null });

    // Second run resets the summary.
    fs.appendFileSync(file, runHeader(new Date("2026-10-29T01:00:00Z")));
    fs.appendFileSync(file, "Found inf or nan, current scale is: 8.0, decrease to: 8.0*0.5\n");
    s = await readAmpSummary(file);
    eq("second run only", { ...s, lastLine: null }, { count: 1, initialScale: 8, currentScale: 4, minScale: 4, lastLine: null });

    // Truncation resets the scan.
    fs.writeFileSync(file, "Found inf or nan, current scale is: 2.0\n");
    s = await readAmpSummary(file);
    eq("truncated file rescanned", { ...s, lastLine: null }, { count: 1, initialScale: 2, currentScale: 1, minScale: 1, lastLine: null });
  }

  console.log("[3] range readers with multi-byte UTF-8");
  {
    // Lines mixing ASCII and multi-byte characters; byte offsets must stay exact.
    const lines = [
      "ascii line one",
      "训练已经开始，中文占三字节",
      "Found inf or nan, current scale is: 64.0",
      "short",
      "last line no newline",
    ];
    const content = lines.slice(0, 4).join("\n") + "\n" + lines[4];
    fs.writeFileSync(file, content, "utf-8");
    const totalBytes = Buffer.byteLength(content, "utf-8");

    const tail = await readTail(file, 40);
    check("tail size", tail.size === totalBytes);
    check("tail hasMore", tail.hasMore === true);
    check("tail ends at EOF", tail.end === totalBytes);
    check("tail starts on a line boundary", tail.text.split("\n")[0] === "short" || tail.text.split("\n")[0] === "last line no newline", tail.text);
    check("tail text complete", tail.text.endsWith("last line no newline"));

    const full = await readTail(file, totalBytes + 10);
    eq("tail of whole file", { ...full, size: 0 }, { size: 0, start: 0, end: totalBytes, text: content, hasMore: false });

    // readFrom from the start of the Chinese line: the boundary is a byte
    // offset, so compute it in bytes, not characters.
    const l0 = lines.slice(0, 1).join("\n") + "\n";
    const chineseStart = Buffer.byteLength(l0, "utf-8");
    const r1 = await readFrom(file, chineseStart, 1024, false);
    check(
      "readFrom stops before unterminated EOF line",
      r1.end === totalBytes - Buffer.byteLength(lines[4], "utf-8"),
      r1.end,
    );
    check("readFrom drops unterminated last line", r1.text === lines[1] + "\n" + lines[2] + "\n" + lines[3] + "\n", r1.text);
    check("readFrom hasMore at EOF-partial", r1.hasMore === true);

    const r2 = await readFrom(file, chineseStart, 1024, true);
    check("includePartial keeps EOF partial", r2.text === lines[1] + "\n" + lines[2] + "\n" + lines[3] + "\n" + lines[4]);
    check("includePartial end = EOF", r2.end === totalBytes);
    check("includePartial no hasMore", r2.hasMore === false);

    // Bounded read: end must be a byte count at a newline boundary.
    const r3 = await readFrom(file, 0, 30, true);
    check("bounded readFrom hasMore", r3.hasMore === true);
    check("bounded end at newline", r3.end === l0.length || r3.end === 0, r3.end);
    check("bounded text is whole lines", r3.text === "" || r3.text.endsWith("\n"));

    // readBefore ending right after line 3.
    const beforeAt = Buffer.byteLength(lines.slice(0, 3).join("\n") + "\n", "utf-8");
    const rb = await readBefore(file, beforeAt, 60);
    check("readBefore ends at before", rb.end === beforeAt);
    check("readBefore line boundary", rb.text.endsWith("\n") && !rb.text.startsWith("ascii"), rb.text);
    check("readBefore hasMore", rb.hasMore === (rb.start > 0));
    const rbAll = await readBefore(file, beforeAt, beforeAt + 100);
    check("readBefore whole head", rbAll.text === lines.slice(0, 3).join("\n") + "\n");
    check("readBefore start 0 -> no hasMore", rbAll.start === 0 && rbAll.hasMore === false);
  }

  console.log("[4] searchFile");
  {
    const res = await searchFile(file, "found inf");
    check("search case-insensitive", res.totalMatches === 1);
    check("match line number", res.matches[0]?.line === 3, res.matches);
    check("not truncated", res.truncated === false);
    const capped = await searchFile(file, "l", 2);
    check("cap respected", capped.matches.length <= 2 && capped.truncated === (capped.totalMatches > 2));
    const none = await searchFile(file, "does-not-exist");
    check("no matches", none.totalMatches === 0 && none.matches.length === 0 && none.truncated === false);
  }

  console.log("[5] decimateMinMax");
  {
    const small = [1, 2, 3].map((v) => ({ v }));
    check("no-op under cap", decimateMinMax(small, (r) => r.v, 10) === small);

    // One spike among 10000 flat rows must survive.
    const flat = Array.from({ length: 10000 }, () => ({ v: 1 }));
    flat[4321] = { v: 999 };
    const out = decimateMinMax(flat, (r) => r.v, 4000);
    check("bounded length", out.length <= 4000 + 2, out.length);
    check("first kept", out[0] === flat[0]);
    check("last kept", out[out.length - 1] === flat[flat.length - 1]);
    check("spike kept", out.some((r) => r.v === 999));
    check("order preserved", out.every((r, i) => i === 0 || flat.indexOf(r) > flat.indexOf(out[i - 1])));

    // Nulls do not poison min/max.
    const withNulls = Array.from({ length: 5000 }, (_, i) => ({ v: i % 7 === 0 ? null : i }));
    const out2 = decimateMinMax(withNulls, (r) => r.v, 100);
    check("nulls tolerated", out2.length <= 102 && out2[0] === withNulls[0]);
  }

  fs.rmSync(dir, { recursive: true, force: true });
  if (failures > 0) {
    console.log(`\n${failures} FAILURE(S)`);
    process.exit(1);
  }
  console.log("\nALL PASS");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
