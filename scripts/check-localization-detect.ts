/**
 * COCO detection export checks.
 *
 *   npx tsx scripts/check-localization-detect.ts
 *
 * Covers the task:'detect' branch of `selectLocalizationData`
 * (src/lib/localization-import.ts): the OBB→AABB conversion
 * (aabbFromObject), split counting, the rotatedConverted report field, and
 * that the same validation rules as OBB apply (pose checks stay off).
 */
import assert from "node:assert/strict";
import {
  aabbFromObject, projectSchema, type AnnotationImage, type AnnotationObject,
} from "../src/lib/localization-annotations";
import { selectLocalizationData } from "../src/lib/localization-import";

let failures = 0;
function check(name: string, fn: () => void) {
  try {
    fn();
    console.log(`  PASS  ${name}`);
  } catch (error) {
    failures += 1;
    console.error(`  FAIL  ${name}: ${error instanceof Error ? error.message : error}`);
  }
}

const rect = (label = "part"): AnnotationObject => ({
  id: `r-${label}`, label, kind: "rectangle", cx: 50, cy: 50, width: 40, height: 20, angle: 0, keypoints: {},
});
const obb = (angle: number, label = "part"): AnnotationObject => ({
  id: `o-${label}-${angle}`, label, kind: "obb", cx: 50, cy: 50, width: 40, height: 20, angle, keypoints: {},
});
const image = (name: string, split: AnnotationImage["split"], objects: AnnotationObject[]): AnnotationImage => ({
  name, width: 100, height: 100, objects, reviewed: true, split, group: `g-${split}-${name}`,
});

const project = (images: AnnotationImage[]) => projectSchema.parse({
  format: "autotrain-localization", version: 1, id: "6cc338da-dfeb-44c8-a345-1043910a392f",
  name: "detect-check", labels: ["part"], keypointNames: [], images,
});

const base = { task: "detect" as const, splitMode: "existing" as const, valRatio: 0.2, testRatio: 0.2, seed: 42 };

check("aabb of an axis-aligned rectangle is the rectangle itself", () => {
  const box = aabbFromObject(rect(), { width: 100, height: 100 });
  assert.deepEqual(box, { x: 30, y: 40, width: 40, height: 20 });
});

check("aabb of a 90°-rotated OBB swaps width and height", () => {
  const box = aabbFromObject(obb(90), { width: 100, height: 100 });
  assert.ok(box);
  assert.equal(box!.x, 40);
  assert.equal(box!.y, 30);
  assert.ok(Math.abs(box!.width - 20) < 1e-6);
  assert.ok(Math.abs(box!.height - 40) < 1e-6);
});

check("aabb of a 45° OBB is the enclosing upright box", () => {
  const box = aabbFromObject(obb(45), { width: 100, height: 100 });
  assert.ok(box);
  const half = (40 * Math.SQRT1_2 + 20 * Math.SQRT1_2) / 2;
  assert.ok(Math.abs(box!.width - half * 2) < 1e-4 && Math.abs(box!.height - half * 2) < 1e-4);
  assert.ok(Math.abs(box!.x - (50 - half)) < 1e-4 && Math.abs(box!.y - (50 - half)) < 1e-4);
});

check("aabb is clipped to the image bounds", () => {
  const box = aabbFromObject({ ...rect(), cx: 95 }, { width: 100, height: 100 });
  assert.deepEqual(box, { x: 75, y: 40, width: 25, height: 20 });
});

check("a fully clipped box returns null", () => {
  assert.equal(aabbFromObject({ ...rect(), cx: 130 }, { width: 100, height: 100 }), null);
});

check("detect split counts and rotatedConverted", () => {
  const source = project([
    image("a.png", "train", [rect()]),
    image("b.png", "train", [obb(30)]),
    image("c.png", "val", [obb(0), rect()]),
    image("d.png", "test", [obb(-15)]),
  ]);
  const { report } = selectLocalizationData(source, base);
  assert.equal(report.errors.length, 0);
  assert.equal(report.includedImages, 4);
  assert.equal(report.instances, 5);
  // Two non-axis-aligned OBBs (30° and -15°); the 0° OBB is already upright.
  assert.equal(report.rotatedConverted, 2);
  assert.equal(report.splits.train.images, 2);
  assert.equal(report.splits.train.instances, 2);
  assert.equal(report.splits.val.images, 1);
  assert.equal(report.splits.val.instances, 2);
  assert.equal(report.splits.test.images, 1);
});

check("unreviewed images are excluded before counting", () => {
  const source = project([
    image("a.png", "train", [rect()]),
    image("b.png", "val", [rect()]),
    { ...image("c.png", "train", [rect()]), reviewed: false },
  ]);
  const { report } = selectLocalizationData(source, base);
  assert.equal(report.includedImages, 2);
  assert.deepEqual(report.unreviewedImages, ["c.png"]);
});

check("missing val images are reported as errors (same as OBB)", () => {
  const source = project([
    image("a.png", "train", [rect()]),
    image("b.png", "train", [rect()]),
  ]);
  const { report } = selectLocalizationData(source, base);
  assert.ok(report.errors.some(issue => issue.includes("Missing val images")));
});

check("keypoints are not required for detect", () => {
  // projectIssues(project, false): no keypoint checks even with an empty template.
  const source = project([
    image("a.png", "train", [rect()]),
    image("b.png", "val", [rect()]),
  ]);
  const { report } = selectLocalizationData(source, base);
  assert.equal(report.errors.length, 0);
});

// --- PaddleDetection pre-labeling: 'detect' task maps to axis-aligned rectangles
import { predictionToImage } from "../src/lib/localization-predictions";

check("predictionToImage detect task produces axis-aligned rectangles", () => {
  const source = image("a.png", "train", []);
  const result = {
    jobId: "job1", task: "detect" as const, labels: ["obj"], keypointNames: [],
    images: [{
      source: "a.png", width: source.width, height: source.height,
      instances: [{ label: "obj", confidence: 0.9, bbox_xyxy_px: [10, 20, 60, 80] as [number, number, number, number] }],
    }],
  };
  const out = predictionToImage(source, result, result.images[0]);
  assert.equal(out.objects.length, 1);
  assert.equal(out.objects[0].kind, "rectangle");
  assert.equal(out.objects[0].angle, 0);
  assert.equal(out.objects[0].cx, 35);
  assert.equal(out.objects[0].cy, 50);
  assert.equal(out.objects[0].width, 50);
  assert.equal(out.objects[0].height, 60);
  assert.equal(out.objects[0].label, "obj");
  assert.deepEqual(out.objects[0].keypoints, {});
  assert.equal(out.prediction?.task, "detect");
});

check("predictionToImage detect without bbox errors", () => {
  const source = image("a.png", "train", []);
  const result = {
    jobId: "job1", task: "detect" as const, labels: [], keypointNames: [],
    images: [{ source: "a.png", width: source.width, height: source.height, instances: [{ label: "obj", confidence: 0.9 }] }],
  };
  assert.throws(() => predictionToImage(source, result, result.images[0]));
});

// --- paddleDetectionWeights ordering
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { paddleDetectionWeights } from "../src/lib/paddledet-weights";

const finish = (async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pdw-"));
  const touch = (n: string) => fs.writeFileSync(path.join(dir, n), "x");
  const checkAsync = async (name: string, expected: string | null) => {
    try {
      assert.equal(await paddleDetectionWeights(dir), expected);
      console.log(`  PASS  ${name}`);
    } catch (error) {
      failures += 1;
      console.error(`  FAIL  ${name}: ${error instanceof Error ? error.message : error}`);
    }
  };
  await checkAsync("paddleDetectionWeights: null on empty dir", null);
  touch("3.pdparams"); touch("10.pdparams"); touch("9.pdparams"); touch("readme.txt");
  await checkAsync("paddleDetectionWeights: highest numbered checkpoint", path.join(dir, "10.pdparams"));
  touch("model_final.pdparams");
  await checkAsync("paddleDetectionWeights: model_final beats numbered", path.join(dir, "model_final.pdparams"));
  touch("best_model.pdparams");
  await checkAsync("paddleDetectionWeights: best_model wins", path.join(dir, "best_model.pdparams"));
  fs.rmSync(dir, { recursive: true, force: true });
})();

finish.then(() => {
  if (failures) {
    console.error(`\n${failures} check(s) failed`);
    process.exit(1);
  }
  console.log("\nAll detection export checks passed");
});
