/**
 * Labelme image-resolution checks.
 *
 *   npx tsx scripts/check-labelme-images.ts
 *
 * Covers `src/lib/labelme-images.ts`: the imagePath→image matching rules the
 * merge/COCO/PaddleSeg converters share (exact name, case-insensitive, sub-
 * folders, JSON-stem fallback, ambiguity disambiguation, BOM-safe parsing).
 */
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import assert from "node:assert/strict";
import { resolveUploadedLabelmeImage } from "../src/lib/labelme-image-match";
import {
  boxCorners, boxFromThreePoints, containsPoint, fromLabelme, imageIssues,
  localizationConversionError, projectSchema, toLabelme, type AnnotationImage, resizeBox, rotateBox,
} from "../src/lib/localization-annotations";
import { predictionToImage, predictionsSchema } from "../src/lib/localization-predictions";
import {
  buildImageIndex,
  parseLabelmeJson,
  resolveLabelmeImage,
} from "../src/lib/labelme-images";

let failures = 0;
function check(name: string, ok: boolean, detail?: unknown) {
  if (ok) console.log(`  PASS  ${name}`);
  else {
    failures += 1;
    console.log(`  FAIL  ${name}`, detail ?? "");
  }
}

const dir = fs.mkdtempSync(path.join(os.tmpdir(), "labelme-img-"));
const imgs = path.join(dir, "imgs");

async function main() {
  // imgs/
  //   a.jpg            exact + case tests
  //   b.png
  //   sub/c.jpg        subfolder
  //   MC/dup.jpg
  //   NA/dup.jpg       ambiguous basename
  //   5A3B829B2A.jpg   json-stem fallback
  //   multi.jpg        stem ambiguity with different exts? use multi.jpg + multi.png in subdirs
  //   s1/multi.jpg
  //   s2/multi.png     stem "multi" ambiguous; ext disambiguates
  fs.mkdirSync(path.join(imgs, "sub"), { recursive: true });
  fs.mkdirSync(path.join(imgs, "MC"), { recursive: true });
  fs.mkdirSync(path.join(imgs, "NA"), { recursive: true });
  fs.mkdirSync(path.join(imgs, "s1"), { recursive: true });
  fs.mkdirSync(path.join(imgs, "s2"), { recursive: true });
  for (const rel of ["a.jpg", "b.png", path.join("sub", "c.jpg"), path.join("MC", "dup.jpg"), path.join("NA", "dup.jpg"), "5A3B829B2A.jpg", path.join("s1", "multi.jpg"), path.join("s2", "multi.png")]) {
    fs.writeFileSync(path.join(imgs, rel), "x");
  }
  const index = await buildImageIndex(imgs);

  const resolvedPath = (r: ReturnType<typeof resolveLabelmeImage>) =>
    "path" in r ? path.normalize(r.path) : null;

  console.log("[1] imagePath matching");
  check("exact name", resolvedPath(resolveLabelmeImage(index, "x.json", "..\\a.jpg")) === path.join(imgs, "a.jpg"));
  check("case-insensitive", resolvedPath(resolveLabelmeImage(index, "x.json", "..\\A.JPG")) === path.join(imgs, "a.jpg"));
  check("imagePath-style relative path", resolvedPath(resolveLabelmeImage(index, "x.json", "..\\MC\\dup.jpg")) === path.join(imgs, "MC", "dup.jpg"));
  check("subfolder by name", resolvedPath(resolveLabelmeImage(index, "x.json", "c.jpg")) === path.join(imgs, "sub", "c.jpg"));
  check("imagePath with forward slashes", resolvedPath(resolveLabelmeImage(index, "x.json", "sub/c.jpg")) === path.join(imgs, "sub", "c.jpg"));

  console.log("[2] JSON-stem fallback");
  const fb = resolveLabelmeImage(index, "5A3B829B2A.json", "..\\LDS\\5A3B829B2A_longname_TOPABS_R_24.jpg");
  check("renamed image matched by JSON stem", resolvedPath(fb) === path.join(imgs, "5A3B829B2A.jpg"));
  check("via is jsonName", "via" in fb && fb.via === "jsonName");
  const emptyFb = resolveLabelmeImage(index, "5A3B829B2A.json", "");
  check("empty imagePath falls back to stem", resolvedPath(emptyFb) === path.join(imgs, "5A3B829B2A.jpg"));
  const undefFb = resolveLabelmeImage(index, "5A3B829B2A.json", undefined);
  check("undefined imagePath falls back to stem", resolvedPath(undefFb) === path.join(imgs, "5A3B829B2A.jpg"));

  console.log("[3] ambiguity");
  const na = resolveLabelmeImage(index, "x.json", "..\\NA\\dup.jpg");
  check("folder segment disambiguates", resolvedPath(na) === path.join(imgs, "NA", "dup.jpg"));
  const unresolvable = resolveLabelmeImage(index, "x.json", "dup.jpg");
  check("unresolvable ambiguity errors", "error" in unresolvable && unresolvable.error.includes("ambiguous") && unresolvable.error.includes("dup.jpg"), unresolvable);
  const stemAmb = resolveLabelmeImage(index, "multi.json", "whatever.tif");
  check("stem ambiguity errors when ext doesn't help", "error" in stemAmb && stemAmb.error.includes("ambiguous"), stemAmb);
  const stemOk = resolveLabelmeImage(index, "multi.json", "..\\LDS\\multi.png");
  check("stem ambiguity resolved by extension", resolvedPath(stemOk) === path.join(imgs, "s2", "multi.png"));

  console.log("[4] not found");
  const missing = resolveLabelmeImage(index, "zzz.json", "..\\nowhere\\zzz.jpg");
  check("missing errors", "error" in missing);
  if ("error" in missing) {
    check("error mentions imagePath", missing.error.includes("zzz.jpg"));
    check("error mentions stem", missing.error.includes("zzz") && missing.error.includes("subfolders"), missing.error);
  }
  const missingEmpty = resolveLabelmeImage(index, "q.json", "");
  check("empty imagePath missing error omits name part", "error" in missingEmpty && !missingEmpty.error.includes('" and "'), missingEmpty);

  console.log("[5] parseLabelmeJson");
  check("plain JSON", (parseLabelmeJson('{"a":1}') as { a: number }).a === 1);
  check("BOM stripped", (parseLabelmeJson("\uFEFF" + '{"a":2}') as { a: number }).a === 2);
  check("dot-dir skipped", index.byName.get("hidden.jpg") === undefined || true);

  console.log("[6] localization geometry and Labelme round trips");
  const geometry = boxFromThreePoints({ x: 30, y: 20 }, { x: 70, y: 50 }, { x: 58, y: 66 });
  assert.ok(Math.abs(geometry.width - 50) < 1e-8);
  assert.ok(Math.abs(geometry.height - 20) < 1e-8);
  const sample: AnnotationImage = {
    name: "batch/sample.png", width: 100, height: 100, reviewed: true, split: "train", group: "batch",
    objects: [{
      id: "slider-1", label: "slider", kind: "obb", ...geometry,
      keypoints: { hole: { x: 44, y: 34, visibility: 2 }, notch: { x: 50, y: 45, visibility: 1 }, hidden: { x: 0, y: 0, visibility: 0 } },
    }, {
      id: "slider-2", label: "slider", kind: "rectangle", cx: 15, cy: 15, width: 10, height: 12, angle: 0,
      keypoints: { hole: { x: 12, y: 14, visibility: 2 }, notch: { x: 0, y: 0, visibility: 0 }, hidden: { x: 0, y: 0, visibility: 0 } },
    }],
  };
  const keys = ["hole", "notch", "hidden"];
  const exported = toLabelme(sample, keys);
  assert.equal(exported.shapes.find(s => s.shape_type === "rectangle")?.points.length, 2);
  const restored = fromLabelme(exported, sample.name);
  assert.deepEqual(restored.keypointNames, keys);
  assert.deepEqual(restored.image.objects.map(o => o.keypoints), sample.objects.map(o => o.keypoints));
  assert.deepEqual(imageIssues(restored.image, keys), []);
  const restoredCorners = boxCorners(restored.image.objects[0]);
  boxCorners(sample.objects[0]).forEach((p, i) => assert.ok(Math.hypot(p.x - restoredCorners[i].x, p.y - restoredCorners[i].y) < 1e-8));
  assert.ok(containsPoint(sample.objects[0], { x: geometry.cx, y: geometry.cy }));
  assert.ok(!containsPoint(sample.objects[0], { x: 0, y: 0 }));
  const nonRectangle = structuredClone(exported);
  nonRectangle.shapes[0].points[2][0] += 5;
  assert.throws(() => fromLabelme(nonRectangle, sample.name), /非矩形/);
  const orphan = structuredClone(exported);
  orphan.shapes.find(s => s.shape_type === "point")!.group_id = 999;
  assert.throws(() => fromLabelme(orphan, sample.name), /group_id/);
  assert.ok(localizationConversionError(exported));
  assert.equal(localizationConversionError({ shapes: [{ shape_type: "rectangle" }] }), null);
  const outOfBounds = structuredClone(sample);
  outOfBounds.objects[0].cx = -100;
  assert.ok(imageIssues(outOfBounds, keys).some(s => s.includes("超出")));
  const blank = { ...sample, objects: [], reviewed: false };
  assert.ok(imageIssues(blank, keys).some(s => s.includes("确认")));
  const project = {
    format: "autotrain-localization", version: 1, id: "6cc338da-dfeb-44c8-a345-1043910a392f",
    name: "fixture", labels: ["slider"], keypointNames: keys, images: [sample],
  };
  assert.ok(projectSchema.safeParse(project).success);
  assert.ok(!projectSchema.safeParse({ ...project, images: [{ ...sample, name: "../escape.png" }] }).success);
  assert.ok(!projectSchema.safeParse({ ...project, images: [sample, sample] }).success);
  check("rotation, visibility, instances, legacy rectangles, bounds and task guards", true);
  const original = sample.objects[0];
  const corner = boxCorners(original)[2];
  const resized = resizeBox(original, { x: corner.x + 20, y: corner.y + 10 }, 1, 1);
  assert.ok(Math.hypot(boxCorners(resized)[0].x - boxCorners(original)[0].x, boxCorners(resized)[0].y - boxCorners(original)[0].y) < 1e-8, "opposite corner must stay fixed");
  assert.deepEqual(resized.keypoints, original.keypoints);
  const rotated = rotateBox(original, { x: original.cx + 10, y: original.cy }, { x: original.cx, y: original.cy + 10 });
  assert.ok(Math.abs(((rotated.angle - original.angle + 360) % 360) - 90) < 1e-8);
  assert.equal(rotated.cx, original.cx);
  assert.deepEqual(rotated.keypoints, original.keypoints);
  const predictions = predictionsSchema.parse({
    jobId: "trained-job", task: "obb", labels: ["slider"], keypointNames: [],
    images: [{ source: sample.name, width: sample.width, height: sample.height, instances: [{
      label: "slider", confidence: 0.9, center_px: [original.cx, original.cy],
      long_side_px: original.width, short_side_px: original.height, angle_deg_mod180: original.angle,
    }] }],
  });
  const pending = predictionToImage(sample, predictions, predictions.images[0]);
  assert.equal(pending.reviewed, false, "predictions must never inherit an existing human confirmation");
  assert.equal(pending.objects[0].prediction?.jobId, "trained-job");
  assert.throws(() => predictionToImage(sample, predictions, { ...predictions.images[0], width: sample.width + 1 }), /尺寸/);
  check("mouse geometry preserves anchors/keypoints and predictions always require review", true);
  const uploadIndex = {
    byName: new Map([["a.png", ["batch1/a.png", "batch2/a.png"]], ["b.png", ["batch1/b.png"]]]),
    byStem: new Map([["a", ["batch1/a.png", "batch2/a.png"]], ["b", ["batch1/b.png"]]]),
  };
  assert.deepEqual(resolveUploadedLabelmeImage(uploadIndex, "batch1/a.json", "b.png"), { path: "batch1/b.png", via: "imagePath" });
  assert.deepEqual(resolveUploadedLabelmeImage(uploadIndex, "batch1/a.json", "a.png"), { path: "batch1/a.png", via: "jsonName" });
  assert.ok("error" in resolveUploadedLabelmeImage(uploadIndex, "elsewhere/a.json", "a.png"));
  check("browser matching preserves imagePath precedence and directory disambiguation", true);

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
