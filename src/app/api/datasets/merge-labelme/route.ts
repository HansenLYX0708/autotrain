import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { requireAuth } from "@/lib/auth";
import { resolveWithin, toSafeSlug } from "@/lib/safe-path";
import { copyFile, readFile, readdir, rename, writeFile } from "fs/promises";
import { existsSync } from "fs";
import * as path from "path";
import { randomInt } from "crypto";

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

function names(categories: CocoCategory[]): string[] {
  return [...new Set(categories.map((category) => category.name.trim()).filter(Boolean))].sort();
}

function difference(left: string[], right: string[]): string[] {
  const rightSet = new Set(right);
  return left.filter((name) => !rightSet.has(name));
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

function convertShapes(shapes: LabelmeShape[], categoryIds: Map<string, number>, imageId: number, firstAnnotationId: number) {
  const annotations: CocoAnnotation[] = [];
  let annotationId = firstAnnotationId;
  for (const shape of shapes || []) {
    const categoryId = categoryIds.get(shape.label?.trim());
    const points = shape.points;
    if (!categoryId || !Array.isArray(points) || points.length < 2) continue;
    const xs = points.map((point) => Number(point[0]));
    const ys = points.map((point) => Number(point[1]));
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const width = maxX - minX;
    const height = maxY - minY;
    if (![minX, maxX, minY, maxY, width, height].every(Number.isFinite) || width <= 1e-5 || height <= 1e-5) continue;
    let area = 0;
    for (let i = 0; i < points.length; i++) {
      const j = (i + 1) % points.length;
      area += Number(points[i][0]) * Number(points[j][1]);
      area -= Number(points[j][0]) * Number(points[i][1]);
    }
    area = Math.abs(area) / 2 || width * height;
    const segmentation = shape.shape_type === "polygon" && points.length >= 3
      ? points.flat().map(Number)
      : [minX, minY, maxX, minY, maxX, maxY, minX, maxY];
    annotations.push({ id: annotationId++, image_id: imageId, category_id: categoryId, bbox: [minX, minY, width, height], area, iscrowd: 0, segmentation: [segmentation] });
  }
  return annotations;
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

    const user = await db.user.findUnique({ where: { id: auth.userId }, select: { id: true, username: true } });
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
    const trainCategories = names(trainDocument.categories);
    const valCategories = names(valDocument.categories);
    const cocoCategories = [...new Set([...trainCategories, ...valCategories])].sort();
    const onlyInLabelme = difference(source.categories, cocoCategories);
    const missingFromTrain = difference(source.categories, trainCategories);
    const missingFromVal = difference(source.categories, valCategories);
    const canMerge = onlyInLabelme.length === 0 && missingFromTrain.length === 0 && missingFromVal.length === 0 && source.samples.length > 0;
    const trainAdd = Math.floor(source.samples.length * trainRatio);
    const valAdd = Math.floor(source.samples.length * valRatio);
    const preview = {
      canMerge,
      labelme: { name: body.labelmeDataset, categories: source.categories, totalJson: source.totalJson, validSamples: source.samples.length, invalidSamples: source.invalid },
      coco: { name: body.cocoDataset, categories: cocoCategories, trainCategories, valCategories },
      differences: { matched: source.categories.filter((name) => cocoCategories.includes(name)), onlyInLabelme, onlyInCoco: difference(cocoCategories, source.categories), missingFromTrain, missingFromVal },
      before: { trainImages: trainDocument.images.length, valImages: valDocument.images.length, trainAnnotations: trainDocument.annotations.length, valAnnotations: valDocument.annotations.length },
      planned: { trainAdd, valAdd, skipped: source.samples.length - trainAdd - valAdd, trainImages: trainDocument.images.length + trainAdd, valImages: valDocument.images.length + valAdd },
    };
    if (action === "verify") return NextResponse.json({ success: true, data: preview });
    if (!canMerge) return NextResponse.json({ success: false, error: "Category verification failed. Resolve category differences before merging.", data: preview }, { status: 400 });
    if (trainAdd + valAdd === 0) return NextResponse.json({ success: false, error: "The selected ratios add zero samples." }, { status: 400 });

    const selected = shuffle(source.samples);
    const trainSamples = selected.slice(0, trainAdd);
    const valSamples = selected.slice(trainAdd, trainAdd + valAdd);
    const append = async (document: CocoDocument, samples: SourceSample[], imageDirectory: string) => {
      const categoryIds = new Map(document.categories.map((category) => [category.name.trim(), category.id]));
      const usedNames = new Set(document.images.map((image) => image.file_name.toLowerCase()));
      let imageId = nextId(document.images);
      let annotationId = nextId(document.annotations);
      for (const sample of samples) {
        const fileName = uniqueImageName(sample.imageName, String(body.labelmeDataset), usedNames);
        const annotations = convertShapes(sample.data.shapes, categoryIds, imageId, annotationId);
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
