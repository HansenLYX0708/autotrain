import { promises as fsp } from "fs";
import * as path from "path";
import { resolveIndexedLabelmeImage, type ImageResolution } from "./labelme-image-match";
export type { ImageResolution } from "./labelme-image-match";

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
  return resolveIndexedLabelmeImage(index, jsonFileName, imagePath,
    file => path.relative(index.root, file).split(path.sep).join("/").toLowerCase());
}
