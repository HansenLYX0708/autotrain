import { readFile, rename, writeFile } from "fs/promises";

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
