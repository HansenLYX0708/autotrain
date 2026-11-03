import * as fs from "fs";
import { promises as fsp } from "fs";
import * as path from "path";
import * as readline from "readline";

/**
 * Per-job console capture (`autotrain-console.log`).
 *
 * The TrainingLog table only stores lines the framework parsers recognise —
 * AMP overflow notices, warnings, stderr and anything else a training run
 * prints are lost. This module backs a verbatim copy of stdout+stderr written
 * next to the job's checkpoints, plus the readers the console API uses to
 * tail/seek/search it without loading the whole file.
 *
 * Everything except the `fs` reads is pure so `scripts/check-console-log.ts`
 * can exercise it directly.
 */

export const CONSOLE_LOG_NAME = "autotrain-console.log";

export const RUN_HEADER_RE = /^===== AutoTrain run started (\S+) =====$/;

const NL = 0x0a; // '\n'

/**
 * Absolute path of the console log for a job, resolved the same way the
 * checkpoints route resolves `saveDir`: an absolute `outputDir` wins,
 * otherwise it is relative to the framework working directory.
 */
export function jobConsoleLogPath(
  outputDir: string | null | undefined,
  workDir: string | null | undefined,
): string | null {
  if (!outputDir) return null;
  const dir = path.isAbsolute(outputDir)
    ? outputDir
    : path.join(workDir ?? "", outputDir);
  return path.join(dir, CONSOLE_LOG_NAME);
}

export function runHeader(date: Date): string {
  return `===== AutoTrain run started ${date.toISOString()} =====\n`;
}

export function runFooter(status: string, code: number | null): string {
  return `===== AutoTrain run finished: ${status} (exit code ${code}) =====\n`;
}

/** AMP loss-scaler overflow summary for the *latest* run in the file. */
export interface AmpSummary {
  count: number;
  initialScale: number | null;
  currentScale: number | null;
  minScale: number | null;
  lastLine: string | null;
}

export function emptyAmp(): AmpSummary {
  return { count: 0, initialScale: null, currentScale: null, minScale: null, lastLine: null };
}

// paddle/amp/grad_scaler.py:
//   "Found inf or nan, current scale is: 512.0, decrease to: 512.0*0.5"
// The `decrease to` multiplier is optional so lines from other AMP versions
// still count; it defaults to Paddle's 0.5 factor.
const AMP_LINE_RE =
  /Found inf or nan, current scale is:\s*([0-9.eE+-]+)(?:,\s*decrease to:\s*[0-9.eE+-]+\s*\*\s*([0-9.eE+-]+))?/i;

export function foldAmpLine(s: AmpSummary, line: string): AmpSummary {
  const trimmed = line.trim();
  // A new run resets the summary — the alert describes the current run only.
  if (RUN_HEADER_RE.test(trimmed)) return emptyAmp();
  const m = AMP_LINE_RE.exec(trimmed);
  if (!m) return s;
  const scale = parseFloat(m[1]);
  const ratio = m[2] !== undefined ? parseFloat(m[2]) : 0.5;
  if (!Number.isFinite(scale) || !Number.isFinite(ratio)) return s;
  const currentScale = scale * ratio;
  return {
    count: s.count + 1,
    initialScale: s.initialScale ?? scale,
    currentScale,
    minScale: s.minScale === null ? currentScale : Math.min(s.minScale, currentScale),
    lastLine: trimmed,
  };
}

interface AmpCacheEntry {
  scannedTo: number;
  partial: string;
  summary: AmpSummary;
}

const ampCache = new Map<string, AmpCacheEntry>();

/**
 * Incrementally fold AMP overflow lines into a summary. Only bytes appended
 * since the previous call are read, and a line split across the read boundary
 * is held back until its '\n' arrives. A truncated/rewritten file (size <
 * scannedTo) restarts the scan.
 */
export async function readAmpSummary(file: string): Promise<AmpSummary> {
  const size = (await fsp.stat(file)).size;
  let entry = ampCache.get(file);
  if (!entry || size < entry.scannedTo) {
    entry = { scannedTo: 0, partial: "", summary: emptyAmp() };
    ampCache.set(file, entry);
  }
  if (size > entry.scannedTo) {
    const handle = await fsp.open(file, "r");
    try {
      const length = size - entry.scannedTo;
      const buf = Buffer.allocUnsafe(length);
      await handle.read(buf, 0, length, entry.scannedTo);
      entry.scannedTo = size;
      const text = entry.partial + buf.toString("utf-8");
      const lines = text.split("\n");
      entry.partial = lines.pop() ?? "";
      for (const line of lines) {
        entry.summary = foldAmpLine(entry.summary, line);
      }
    } finally {
      await handle.close();
    }
  }
  return entry.summary;
}

export interface ConsoleRange {
  size: number;
  start: number;
  end: number;
  text: string;
  hasMore: boolean;
}

/** Byte offset just past the next '\n', or `from` when none exists. */
function afterFirstNewline(buf: Buffer, from: number): number {
  const idx = buf.indexOf(NL, from);
  return idx === -1 ? from : idx + 1;
}

/** The last `maxBytes` of the file, starting on a line boundary. */
export async function readTail(file: string, maxBytes: number): Promise<ConsoleRange> {
  const size = (await fsp.stat(file)).size;
  const windowStart = Math.max(0, size - maxBytes);
  const handle = await fsp.open(file, "r");
  try {
    const buf = Buffer.allocUnsafe(size - windowStart);
    await handle.read(buf, 0, buf.length, windowStart);
    const start =
      windowStart > 0 ? windowStart + afterFirstNewline(buf, 0) : windowStart;
    const text = buf.subarray(start - windowStart).toString("utf-8");
    return { size, start, end: size, text, hasMore: start > 0 };
  } finally {
    await handle.close();
  }
}

/**
 * Forward read from `offset`. A trailing line that has not been fully written
 * yet is held back unless `includePartial` and the read reached EOF — this is
 * what keeps a live tail from ever showing half a log line. `end` is a byte
 * offset, so multi-byte characters cannot skew the resume position.
 */
export async function readFrom(
  file: string,
  offset: number,
  maxBytes: number,
  includePartial: boolean,
): Promise<ConsoleRange> {
  const size = (await fsp.stat(file)).size;
  if (offset >= size) {
    return { size, start: offset, end: offset, text: "", hasMore: false };
  }
  const readLen = Math.min(maxBytes, size - offset);
  const reachedEof = offset + readLen >= size;
  const handle = await fsp.open(file, "r");
  try {
    const buf = Buffer.allocUnsafe(readLen);
    await handle.read(buf, 0, readLen, offset);
    let end = readLen;
    if (!reachedEof || !includePartial) {
      const lastNl = buf.lastIndexOf(NL);
      if (lastNl >= 0) {
        end = lastNl + 1;
      } else if (reachedEof) {
        // EOF without a newline and partial lines are not allowed: nothing to return.
        end = 0;
      }
      // No newline and more data pending: the line is longer than maxBytes;
      // returning the chunk is the only way to make progress.
    }
    const text = buf.subarray(0, end).toString("utf-8");
    const endOffset = offset + end;
    return { size, start: offset, end: endOffset, text, hasMore: endOffset < size };
  } finally {
    await handle.close();
  }
}

/**
 * Up to `maxBytes` ending at byte offset `before` (exclusive), starting on a
 * line boundary. Used by "Load earlier" to page backwards through the file.
 */
export async function readBefore(
  file: string,
  before: number,
  maxBytes: number,
): Promise<ConsoleRange> {
  const size = (await fsp.stat(file)).size;
  const end = Math.min(Math.max(0, before), size);
  const windowStart = Math.max(0, end - maxBytes);
  if (end === 0) {
    return { size, start: 0, end: 0, text: "", hasMore: false };
  }
  const handle = await fsp.open(file, "r");
  try {
    const buf = Buffer.allocUnsafe(end - windowStart);
    await handle.read(buf, 0, buf.length, windowStart);
    const start =
      windowStart > 0 ? windowStart + afterFirstNewline(buf, 0) : windowStart;
    const text = buf.subarray(start - windowStart).toString("utf-8");
    return { size, start, end, text, hasMore: start > 0 };
  } finally {
    await handle.close();
  }
}

export interface SearchResult {
  matches: { line: number; text: string }[];
  totalMatches: number;
  truncated: boolean;
}

/**
 * Case-insensitive substring search over the whole file, streamed line by
 * line so arbitrarily large logs don't load into memory. Line numbers are
 * 1-based. At most `cap` matches are collected; counting continues so the
 * caller can report how many it did not show.
 */
export async function searchFile(
  file: string,
  query: string,
  cap = 1000,
): Promise<SearchResult> {
  const needle = query.toLowerCase();
  const matches: { line: number; text: string }[] = [];
  let totalMatches = 0;
  let lineNo = 0;
  const rl = readline.createInterface({
    input: fs.createReadStream(file, { encoding: "utf-8" }),
    crlfDelay: Infinity,
  });
  try {
    for await (const line of rl) {
      lineNo += 1;
      if (needle && line.toLowerCase().includes(needle)) {
        totalMatches += 1;
        if (matches.length < cap) matches.push({ line: lineNo, text: line });
      }
    }
  } finally {
    rl.close();
  }
  return { matches, totalMatches, truncated: totalMatches > cap };
}
