import { promises as fsp } from "fs";
import * as path from "path";

/**
 * Labelme JSON → image resolution.
 *
 * `imagePath` inside a Labelme JSON records where the image sat when the
 * annotation was created (e.g. `..\LDS\foo.jpg`), which is almost never where
 * it sits after upload: images get renamed to the JSON stem, land flat in
 * `imgs/`, or keep their upload subfolders (`imgs/MC/foo.jpg`). Every Labelme
 * entry point (merge, COCO conversion, PaddleSeg conversion) resolves through
 * this module so the rules live — and improve — in one place.
 */

export const IMAGE_EXTENSIONS = [
  ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif",
];

export interface ImageIndex {
  root: string;
  /** lower-cased file name -> absolute paths (recursive scan of root). */
  byName: Map<string, string[]>;
  /** lower-cased file stem (name minus extension) -> absolute paths. */
  byStem: Map<string, string[]>;
}

/** Recursive index of an images directory; dot-directories are skipped. */
export async function buildImageIndex(imgsDir: string): Promise<ImageIndex> {
  const index: ImageIndex = { root: imgsDir, byName: new Map(), byStem: new Map() };
  const add = (map: Map<string, string[]>, key: string, file: string) => {
    const list = map.get(key);
    if (list) list.push(file);
    else map.set(key, [file]);
  };
  const walk = async (dir: string) => {
    let entries;
    try {
      entries = await fsp.readdir(dir, { withFileTypes: true });
    } catch {
      return; // missing/unreadable directory: empty index, resolver reports "not found"
    }
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (!entry.name.startsWith(".")) await walk(full);
        continue;
      }
      const ext = path.extname(entry.name).toLowerCase();
      if (!IMAGE_EXTENSIONS.includes(ext)) continue;
      add(index.byName, entry.name.toLowerCase(), full);
      add(index.byStem, path.basename(entry.name, ext).toLowerCase(), full);
    }
  };
  await walk(imgsDir);
  return index;
}

/** JSON.parse that tolerates a leading UTF-8 BOM (U+FEFF). */
export function parseLabelmeJson(raw: string): unknown {
  return JSON.parse(raw.replace(/^\uFEFF/, ""));
}

export type ImageResolution =
  | { path: string; via: "imagePath" | "jsonName" }
  | { error: string };

// Path segments of an imagePath value, with `.`/`..`/empties removed — a
// Labelme `..\MC\a.JPG` contributes ["mc", "a.jpg"] for suffix matching.
const rawSegments = (value: string) =>
  value
    .split(/[\\/]/)
    .filter((s) => s && s !== "." && s !== "..")
    .map((s) => s.toLowerCase());

/** Relative path inside the index root, '/'-separated, lower-cased. */
function relKey(index: ImageIndex, absolute: string): string {
  return path.relative(index.root, absolute).split(path.sep).join("/").toLowerCase();
}

/**
 * Resolution order:
 *  1. The file name recorded in `imagePath` (last path segment). Ambiguous
 *     base names are disambiguated by the imagePath's own trailing segments.
 *  2. The JSON file's stem (`5A3B829B2A.json` -> `5A3B829B2A.<ext>`), used
 *     when the image was renamed to match the JSON. Extension-ambiguous stems
 *     prefer the extension `imagePath` recorded.
 */
export function resolveLabelmeImage(
  index: ImageIndex,
  jsonFileName: string,
  imagePath: unknown,
): ImageResolution {
  const raw = imagePath == null ? "" : String(imagePath);
  const segments = rawSegments(raw);
  const name = segments.length ? segments[segments.length - 1] : "";
  const jsonStem = jsonFileName.replace(/\.json$/i, "");

  // 1. imagePath basename, case-insensitive, anywhere under root.
  if (name) {
    const candidates = index.byName.get(name.toLowerCase()) ?? [];
    if (candidates.length === 1) return { path: candidates[0], via: "imagePath" };
    if (candidates.length > 1) {
      // Keep candidates whose relative path ends with the imagePath's own
      // trailing segments ("MC/a.jpg" prefers imgs/MC/a.jpg over imgs/NA/a.jpg).
      const wanted = segments;
      const scored = candidates.filter((candidate) => {
        const rel = relKey(index, candidate).split("/");
        const take = Math.min(wanted.length, rel.length);
        return rel.slice(-take).join("/") === wanted.slice(-take).join("/");
      });
      if (scored.length === 1) return { path: scored[0], via: "imagePath" };
      const listed = candidates.slice(0, 3).map((c) => relKey(index, c)).join(", ");
      return { error: `ambiguous image "${name}": ${candidates.length} files under imgs (${listed}${candidates.length > 3 ? ", …" : ""})` };
    }
  }

  // 2. JSON-stem fallback (image renamed to the JSON's file name).
  if (jsonStem) {
    const candidates = index.byStem.get(jsonStem.toLowerCase()) ?? [];
    if (candidates.length === 1) return { path: candidates[0], via: "jsonName" };
    if (candidates.length > 1) {
      const wantedExt = path.extname(name).toLowerCase();
      const sameExt = candidates.filter(
        (c) => path.extname(c).toLowerCase() === wantedExt,
      );
      if (sameExt.length === 1) return { path: sameExt[0], via: "jsonName" };
      const listed = candidates.slice(0, 3).map((c) => relKey(index, c)).join(", ");
      return { error: `ambiguous image "${jsonStem}.*": ${candidates.length} files under imgs (${listed}${candidates.length > 3 ? ", …" : ""})` };
    }
  }

  const imageExt = path.extname(name);
  return {
    error:
      `image not found: imagePath "${raw}"; looked for ` +
      (name ? `"${name}" and ` : "") +
      `"${jsonStem}${imageExt || ".*"}" under imgs (including subfolders)`,
  };
}
