import { readFile, rename, writeFile } from "fs/promises";
import { existsSync } from "fs";
import * as path from "path";

/**
 * Recover the validation paths emitted by our LabelMe converter when an older
 * import omitted them. Explicit selections always win, including broken paths:
 * callers must report those errors rather than silently selecting another split.
 */
export function resolveCocoValidationPaths(
  root: string,
  dataset: { evalAnnoPath?: string | null; evalImagePath?: string | null },
): { evalAnnoPath: string | null; evalImagePath: string | null } {
  let evalAnnoPath = dataset.evalAnnoPath || null;
  let evalImagePath = dataset.evalImagePath || null;
  if (!evalAnnoPath) {
    evalAnnoPath = ["data/annotations/instance_val.json", "annotations/instance_val.json"]
      .find(candidate => existsSync(path.join(root, candidate))) || null;
  }
  if (!evalImagePath && evalAnnoPath) {
    const relativeAnno = path.isAbsolute(evalAnnoPath) ? path.relative(root, evalAnnoPath) : evalAnnoPath;
    const imageDir = relativeAnno.replace(/\\/g, "/").startsWith("data/annotations/") ? "data/val" : "val";
    if (existsSync(path.join(root, imageDir))) evalImagePath = imageDir;
  }
  return { evalAnnoPath, evalImagePath };
}

export interface CocoAreaRepairResult {
  annotations: number;
  repaired: number;
}

export async function repairCocoAnnotationAreas(filePath: string): Promise<CocoAreaRepairResult> {
  const source = await readFile(filePath, "utf-8");
  const document = JSON.parse(source) as { annotations?: Array<Record<string, unknown>> };
  const annotations = Array.isArray(document.annotations) ? document.annotations : [];
  let repaired = 0;

  for (const annotation of annotations) {
    const bbox = annotation.bbox;
    if (!Array.isArray(bbox) || bbox.length < 4) continue;
    const width = Number(bbox[2]);
    const height = Number(bbox[3]);
    const area = Number(annotation.area);
    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 1e-5 || height <= 1e-5) continue;
    if (Number.isFinite(area) && area > 0) continue;
    annotation.area = width * height;
    repaired++;
  }

  if (repaired > 0) {
    const temporaryPath = `${filePath}.${process.pid}.${Date.now()}.tmp`;
    await writeFile(temporaryPath, JSON.stringify(document, null, 2), "utf-8");
    await rename(temporaryPath, filePath);
  }

  return { annotations: annotations.length, repaired };
}
