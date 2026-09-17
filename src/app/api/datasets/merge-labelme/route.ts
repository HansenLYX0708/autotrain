import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { requireAuth } from "@/lib/auth";
import { resolveWithin, toSafeSlug } from "@/lib/safe-path";
import { copyFile, readFile, readdir, rename, stat, writeFile } from "fs/promises";
import { existsSync } from "fs";
import * as path from "path";
import { randomInt } from "crypto";
import { assessLabelmeMerge, convertLabelmeShapes } from "@/lib/labelme-coco-merge";

interface CocoCategory { id: number; name: string; supercategory?: string }
interface CocoImage { id: number; file_name: string; width: number; height: number }
interface CocoAnnotation { id: number; image_id: number; category_id: number; bbox: number[]; area: number; iscrowd: number; segmentation?: number[][] }
interface CocoDocument { images: CocoImage[]; annotations: CocoAnnotation[]; categories: CocoCategory[]; [key: string]: unknown }
interface LabelmeShape { label: string; points: number[][]; shape_type?: string }
interface LabelmeDocument { imagePath: string; imageWidth: number; imageHeight: number; shapes: LabelmeShape[] }
interface SourceSample { jsonPath: string; imagePath: string; imageName: string; data: LabelmeDocument }

function ratio(value: unknown): number | null {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= 0 && parsed <= 1 ? parsed : null;
}

function shuffle<T>(items: T[]): T[] {
  const result = [...items];
  for (let i = result.length - 1; i > 0; i--) {
    const j = randomInt(i + 1);
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

function nextId(items: Array<{ id: number }>): number {
  return items.reduce((maximum, item) => Math.max(maximum, Number(item.id) || 0), 0) + 1;
}

function classStats(document: CocoDocument) {
  const counts = new Map<number, number>();
  const imageIds = new Map<number, Set<number>>();
  for (const annotation of document.annotations) {
    counts.set(annotation.category_id, (counts.get(annotation.category_id) || 0) + 1);
    if (!imageIds.has(annotation.category_id)) imageIds.set(annotation.category_id, new Set());
    imageIds.get(annotation.category_id)!.add(annotation.image_id);
  }
  return document.categories.map((category) => ({
    id: category.id,
    name: category.name,
    count: counts.get(category.id) || 0,
    imageCount: imageIds.get(category.id)?.size || 0,
  }));
}

async function readCoco(filePath: string): Promise<CocoDocument> {
  const parsed = JSON.parse(await readFile(filePath, "utf-8"));
  if (!Array.isArray(parsed.images) || !Array.isArray(parsed.annotations) || !Array.isArray(parsed.categories)) {
    throw new Error(`Invalid COCO annotation file: ${filePath}`);
  }
  return parsed;
}

async function annotationFiles(datasetRoot: string) {
  const directory = path.join(datasetRoot, "data", "annotations");
  const files = await readdir(directory);
  const train = files.find((file) => file.toLowerCase().endsWith(".json") && file.toLowerCase().includes("train"));
  const val = files.find((file) => file.toLowerCase().endsWith(".json") && (file.toLowerCase().includes("val") || file.toLowerCase().includes("valid")));
  if (!train || !val) throw new Error("The target COCO dataset must have both train and val annotation JSON files.");
  return { train: path.join(directory, train), val: path.join(directory, val) };
}

async function sourceSamples(datasetRoot: string) {
  const imageDirectory = path.join(datasetRoot, "data", "imgs");
  const jsonDirectory = path.join(datasetRoot, "data", "jsons");
  const files = (await readdir(jsonDirectory)).filter((file) => file.toLowerCase().endsWith(".json"));
  const samples: SourceSample[] = [];
  const categories = new Set<string>();
  const invalid: string[] = [];
  for (const file of files) {
    try {
      const jsonPath = path.join(jsonDirectory, file);
      const data = JSON.parse(await readFile(jsonPath, "utf-8")) as LabelmeDocument;
      const imageName = path.basename(data.imagePath || "");
      const imagePath = resolveWithin(imageDirectory, imageName);
      if (!imageName || !imagePath || !existsSync(imagePath)) {
        invalid.push(`${file}: image not found`);
        continue;
      }
      for (const shape of data.shapes || []) if (shape.label?.trim()) categories.add(shape.label.trim());
      samples.push({ jsonPath, imagePath, imageName, data });
    } catch (error) {
      invalid.push(`${file}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  return { samples, categories: [...categories].sort(), invalid, totalJson: files.length };
}

function uniqueImageName(original: string, sourceName: string, used: Set<string>): string {
  if (!used.has(original.toLowerCase())) {
    used.add(original.toLowerCase());
    return original;
  }
  const extension = path.extname(original);
  const stem = path.basename(original, extension);
  const prefix = toSafeSlug(sourceName, "labelme");
  let index = 1;
  let candidate = `${prefix}_${stem}${extension}`;
  while (used.has(candidate.toLowerCase())) candidate = `${prefix}_${stem}_${index++}${extension}`;
  used.add(candidate.toLowerCase());
  return candidate;
}

async function folderSize(directory: string): Promise<number> {
  let total = 0;
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const fullPath = path.join(directory, entry.name);
    total += entry.isDirectory() ? await folderSize(fullPath) : (await stat(fullPath)).size;
  }
  return total;
}

async function atomicWrite(filePath: string, document: CocoDocument) {
  const temporary = `${filePath}.${process.pid}.${Date.now()}.tmp`;
  await writeFile(temporary, JSON.stringify(document, null, 2), "utf-8");
  await rename(temporary, filePath);
}

export async function POST(request: NextRequest) {
  try {
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;
    const body = await request.json();
    const action = body.action === "merge" ? "merge" : "verify";
    const trainRatio = ratio(body.trainRatio);
    const valRatio = ratio(body.valRatio);
    if (!body.labelmeDataset || !body.cocoDataset || trainRatio === null || valRatio === null) {
      return NextResponse.json({ success: false, error: "Dataset selections and ratios between 0 and 1 are required." }, { status: 400 });
    }
    if (trainRatio + valRatio > 1 + 1e-9) {
      return NextResponse.json({ success: false, error: "Train ratio + val ratio must be less than or equal to 1." }, { status: 400 });
    }

    const user = await db.user.findUnique({ where: { id: auth.userId }, select: { id: true, username: true, maxStorageQuota: true } });
    const config = await db.systemConfig.findFirst();
    if (!user || !config?.userDatabasePath) return NextResponse.json({ success: false, error: "User storage is not configured." }, { status: 400 });
    const userRoot = path.join(config.userDatabasePath, user.username);
    const labelmeRoot = resolveWithin(path.join(userRoot, "labelme"), String(body.labelmeDataset));
    const cocoRoot = resolveWithin(path.join(userRoot, "COCO"), String(body.cocoDataset));
    if (!labelmeRoot || !cocoRoot || !existsSync(labelmeRoot) || !existsSync(cocoRoot)) {
      return NextResponse.json({ success: false, error: "Selected dataset folder was not found." }, { status: 404 });
    }

    const files = await annotationFiles(cocoRoot);
    const [source, trainDocument, valDocument] = await Promise.all([
      sourceSamples(labelmeRoot),
      readCoco(files.train),
      readCoco(files.val),
    ]);
    const trainCategories = trainDocument.categories.map((category) => category.name);
    const valCategories = valDocument.categories.map((category) => category.name);
    const before = { trainImages: trainDocument.images.length, valImages: valDocument.images.length, trainAnnotations: trainDocument.annotations.length, valAnnotations: valDocument.annotations.length };
    const assessment = assessLabelmeMerge(source.categories, trainCategories, valCategories, source.samples.length, trainRatio, valRatio, before);
    const preview = {
      canMerge: assessment.canMerge,
      labelme: { name: body.labelmeDataset, categories: source.categories, totalJson: source.totalJson, validSamples: source.samples.length, invalidSamples: source.invalid },
      coco: { name: body.cocoDataset, categories: assessment.cocoCategories, trainCategories, valCategories },
      differences: assessment.differences,
      before,
      planned: assessment.planned,
    };
    if (action === "verify") return NextResponse.json({ success: true, data: preview });
    if (!assessment.canMerge) return NextResponse.json({ success: false, error: "Category verification failed. Resolve category differences before merging.", data: preview }, { status: 400 });
    if (assessment.planned.trainAdd + assessment.planned.valAdd === 0) return NextResponse.json({ success: false, error: "The selected ratios add zero samples." }, { status: 400 });

    const selected = shuffle(source.samples);
    const trainSamples = selected.slice(0, assessment.planned.trainAdd);
    const valSamples = selected.slice(assessment.planned.trainAdd, assessment.planned.trainAdd + assessment.planned.valAdd);
    const additionalSize = (await Promise.all([...trainSamples, ...valSamples].map(async (sample) => (await stat(sample.imagePath)).size))).reduce((sum, size) => sum + size, 0);
    const usedStorage = await folderSize(userRoot);
    if (usedStorage + additionalSize > Number(user.maxStorageQuota)) {
      return NextResponse.json({ success: false, error: "Storage quota exceeded by this merge." }, { status: 403 });
    }
    const append = async (document: CocoDocument, samples: SourceSample[], imageDirectory: string) => {
      const categoryIds = new Map(document.categories.map((category) => [category.name.trim(), category.id]));
      const usedNames = new Set(document.images.map((image) => image.file_name.toLowerCase()));
      let imageId = nextId(document.images);
      let annotationId = nextId(document.annotations);
      for (const sample of samples) {
        const fileName = uniqueImageName(sample.imageName, String(body.labelmeDataset), usedNames);
        const annotations = convertLabelmeShapes(sample.data.shapes, categoryIds, imageId, annotationId);
        await copyFile(sample.imagePath, path.join(imageDirectory, fileName));
        document.images.push({ id: imageId, file_name: fileName, width: Number(sample.data.imageWidth) || 0, height: Number(sample.data.imageHeight) || 0 });
        document.annotations.push(...annotations);
        imageId++;
        annotationId += annotations.length;
      }
    };
    await append(trainDocument, trainSamples, path.join(cocoRoot, "data", "train"));
    await append(valDocument, valSamples, path.join(cocoRoot, "data", "val"));
    await atomicWrite(files.train, trainDocument);
    await atomicWrite(files.val, valDocument);

    const datasetRows = await db.dataset.findMany({ where: { userId: user.id, format: "COCO" }, select: { id: true, datasetDir: true } });
    const matchingIds = datasetRows.filter((dataset) => {
      if (!dataset.datasetDir) return false;
      const root = path.isAbsolute(dataset.datasetDir) ? dataset.datasetDir : path.join(userRoot, dataset.datasetDir);
      return path.resolve(root) === path.resolve(cocoRoot);
    }).map((dataset) => dataset.id);
    if (matchingIds.length > 0) {
      await db.dataset.updateMany({
        where: { id: { in: matchingIds } },
        data: {
          numClasses: trainDocument.categories.length,
          numAnnotations: trainDocument.annotations.length,
          numTrainImages: trainDocument.images.length,
          numEvalImages: valDocument.images.length,
          classStats: JSON.stringify({ train: classStats(trainDocument), eval: classStats(valDocument) }),
        },
      });
    }

    return NextResponse.json({
      success: true,
      data: {
        ...preview,
        added: { train: trainSamples.length, val: valSamples.length, skipped: source.samples.length - trainSamples.length - valSamples.length },
        after: { trainImages: trainDocument.images.length, valImages: valDocument.images.length, trainAnnotations: trainDocument.annotations.length, valAnnotations: valDocument.annotations.length },
        updatedDatasetRecords: matchingIds.length,
      },
    });
  } catch (error) {
    console.error("Labelme merge failed:", error);
    return NextResponse.json({ success: false, error: error instanceof Error ? error.message : "Failed to merge datasets." }, { status: 500 });
  }
}
