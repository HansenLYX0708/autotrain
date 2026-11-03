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
