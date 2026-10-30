import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { requireAuth } from '@/lib/auth';
import * as fs from 'fs';
import * as path from 'path';
import { randomInt } from 'crypto';

// Helper to get folder size recursively
function getFolderSize(folderPath: string): number {
  let totalSize = 0;

  if (!fs.existsSync(folderPath)) {
    return 0;
  }

  const stats = fs.statSync(folderPath);

  if (stats.isFile()) {
    return stats.size;
  }

  const files = fs.readdirSync(folderPath);
  for (const file of files) {
    const filePath = path.join(folderPath, file);
    const fileStats = fs.statSync(filePath);

    if (fileStats.isDirectory()) {
      totalSize += getFolderSize(filePath);
    } else {
      totalSize += fileStats.size;
    }
  }

  return totalSize;
}

interface LabelmeShape {
  label: string;
  points: number[][];
  shape_type: string;
  group_id: number | null;
}

interface LabelmeData {
  version: string;
  flags: Record<string, unknown>;
  shapes: LabelmeShape[];
  imagePath: string;
  imageData: string | null;
  imageHeight: number;
  imageWidth: number;
}

interface CocoAnnotation {
  id: number;
  image_id: number;
  category_id: number;
  bbox: number[];
  area: number;
  iscrowd: number;
  // COCO polygon segmentation: array of polygons, each as a flat
  // [x1, y1, x2, y2, ...] list. Required for instance-segmentation training
  // (Mask R-CNN, SOLOv2, etc.); harmless for detection-only models.
  segmentation?: number[][];
}

interface CocoImage {
  id: number;
  file_name: string;
  height: number;
  width: number;
}

interface CocoCategory {
  id: number;
  name: string;
  supercategory: string;
}

interface CocoDataset {
  info: {
    description: string;
    version: string;
    year: number;
    contributor: string;
    date_created: string;
  };
  images: CocoImage[];
  annotations: CocoAnnotation[];
  categories: CocoCategory[];
}

// POST /api/datasets/labelme-to-coco - Convert Labelme format to COCO format
export async function POST(request: NextRequest) {
  try {
    // Check authentication
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;

    const { userId, role } = auth;
    
    const body = await request.json();
    const {
      projectId,
      name,
      description,
      labelmeImagesPath,
      labelmeAnnotationsPath,
      outputDatasetDir,
      trainRatio,
      valRatio,
      testRatio,
    } = body;

    // Validate required fields
    if (!name || !labelmeImagesPath || !labelmeAnnotationsPath) {
      return NextResponse.json(
        { error: 'Missing required fields: name, labelmeImagesPath, labelmeAnnotationsPath' },
        { status: 400 }
      );
    }

    // Validate ratios
    const train = parseFloat(trainRatio) || 0;
    const val = parseFloat(valRatio) || 0;
    const test = parseFloat(testRatio) || 0;
    const total = train + val + test;

    if (![train, val, test].every(value => Number.isFinite(value) && value >= 0 && value <= 1) || Math.abs(total - 1.0) > 0.001) {
      return NextResponse.json(
        { error: `Train/Val/Test ratios must sum to 1.0, current sum: ${total.toFixed(3)}` },
        { status: 400 }
      );
    }

    if (train <= 0 || val <= 0) {
      return NextResponse.json(
        { error: 'Train and Val ratios must be positive' },
        { status: 400 }
      );
    }

    if (train + val > 1.0) {
      return NextResponse.json(
        { error: 'Train + Val ratios must not exceed 1.0' },
        { status: 400 }
      );
    }

    // Get system config for PaddleDetection path
    const systemConfig = await db.systemConfig.findFirst();
    const workDir = systemConfig?.paddleDetectionPath;

    if (!workDir) {
      return NextResponse.json(
        { error: 'PaddleDetection path not configured. Please configure it in Settings.' },
        { status: 400 }
      );
    }

    // Resolve absolute paths
    const absoluteImagesPath = path.isAbsolute(labelmeImagesPath) 
      ? labelmeImagesPath 
      : path.join(workDir, labelmeImagesPath);
    
    const absoluteAnnotationsPath = path.isAbsolute(labelmeAnnotationsPath)
      ? labelmeAnnotationsPath
      : path.join(workDir, labelmeAnnotationsPath);

    // Validate paths exist
    if (!fs.existsSync(absoluteImagesPath)) {
      return NextResponse.json(
        { error: `Images path does not exist: ${absoluteImagesPath}` },
        { status: 400 }
      );
    }

    if (!fs.existsSync(absoluteAnnotationsPath)) {
      return NextResponse.json(
        { error: `Annotations path does not exist: ${absoluteAnnotationsPath}` },
        { status: 400 }
      );
    }



    // Get user info for storage quota check
    const user = await db.user.findUnique({
      where: { id: userId },
      select: { id: true, username: true, maxStorageQuota: true },
    });

    if (!user) {
      return NextResponse.json({ error: "User not found" }, { status: 404 });
    }

    // Check storage quota
    const userDatabasePath = (systemConfig as any)?.userDatabasePath;
    if (userDatabasePath) {
      const userFolderPath = path.join(userDatabasePath, user.username);
      const usedStorage = getFolderSize(userFolderPath);
      const maxQuota = Number(user.maxStorageQuota);

      // Calculate required space (labelme images will be copied to COCO format)
      const requiredSize = getFolderSize(absoluteImagesPath);

      if (usedStorage + requiredSize > maxQuota) {
        return NextResponse.json(
          {
            error: "存储空间不足",
            message: `您已使用 ${(usedStorage / 1024 / 1024 / 1024).toFixed(2)} GB，配额为 ${(maxQuota / 1024 / 1024 / 1024).toFixed(2)} GB。本次转换需要 ${(requiredSize / 1024 / 1024).toFixed(2)} MB 空间来存储COCO格式数据。请联系管理员扩容或删除不需要的数据。`,
            usedStorage,
            maxStorageQuota: maxQuota,
            requiredSpace: requiredSize
          },
          { status: 403 }
        );
      }
    }

    // Generate output paths
    const datasetDir = outputDatasetDir || `dataset/COCO/${name}`;
    const absoluteDatasetDir = path.isAbsolute(datasetDir)
      ? datasetDir
      : path.join(workDir, datasetDir);

    // Check the directory the conversion actually writes to. This used to check
    // `{paddleDetectionPath}/dataset/COCO/{name}`, which is not where the output
    // goes, so converting twice silently mixed two splits into one folder.
    if (fs.existsSync(absoluteDatasetDir) && fs.readdirSync(absoluteDatasetDir).length > 0) {
      return NextResponse.json(
        { error: `Dataset "${name}" already exists at ${absoluteDatasetDir}. Please choose a different name.` },
        { status: 400 }
      );
    }

    // Create output directories
    const trainImagesDir = path.join(absoluteDatasetDir, 'data', 'train');
    const valImagesDir = path.join(absoluteDatasetDir, 'data', 'val');
    const testImagesDir = path.join(absoluteDatasetDir, 'data', 'test');
    const annotationsDir = path.join(absoluteDatasetDir, 'data', 'annotations');

    [trainImagesDir, valImagesDir, testImagesDir, annotationsDir].forEach(dir => {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
    });

    // Read all labelme annotation files
    const labelmeFiles = fs.readdirSync(absoluteAnnotationsPath)
      .filter(file => file.toLowerCase().endsWith('.json'));

    if (labelmeFiles.length === 0) {
      return NextResponse.json(
        { error: 'No labelme annotation files found (.json files)' },
        { status: 400 }
      );
    }

    // Shuffle files for random split
    const shuffledFiles = [...labelmeFiles];
    for (let i = shuffledFiles.length - 1; i > 0; i--) {
      const j = randomInt(i + 1);
      [shuffledFiles[i], shuffledFiles[j]] = [shuffledFiles[j], shuffledFiles[i]];
    }
    
    // Calculate split indices
    const totalFiles = shuffledFiles.length;
    const trainCount = Math.floor(totalFiles * train);
    const valCount = test === 0 ? totalFiles - trainCount : Math.floor(totalFiles * val);
    const testCount = totalFiles - trainCount - valCount;

    const trainFiles = shuffledFiles.slice(0, trainCount);
    const valFiles = shuffledFiles.slice(trainCount, trainCount + valCount);
    const testFiles = shuffledFiles.slice(trainCount + valCount);

    // Convert each split
    const categories = new Map<string, number>();
    let categoryId = 1;
    let annotationId = 1;

    const skipped: string[] = [];
    const usedNames = new Set<string>();

    const convertSplit = (files: string[], splitName: string): { images: CocoImage[]; annotations: CocoAnnotation[]; imageCount: number; annotationCount: number } => {
      const images: CocoImage[] = [];
      const annotations: CocoAnnotation[] = [];
      let imageId = 1;
      let imageCount = 0;
      let annotationCount = 0;

      for (const file of files) {
        const labelmePath = path.join(absoluteAnnotationsPath, file);
        let labelmeData: LabelmeData;
        try {
          labelmeData = JSON.parse(fs.readFileSync(labelmePath, 'utf-8'));
        } catch (error) {
          skipped.push(`${file}: invalid JSON (${error instanceof Error ? error.message : String(error)})`);
          continue;
        }

        // Labelme stores `imagePath` relative to where the JSON was *created*,
        // e.g. "..\MC\a.JPG". After upload every image sits flat in imgs/, so
        // resolve by base name (either separator) — joining the raw value made
        // every such sample "not found" and produced an empty dataset.
        const imageFileName = String(labelmeData.imagePath || '').split(/[\\/]/).pop() || '';
        const sourceImagePath = imageFileName ? path.join(absoluteImagesPath, imageFileName) : '';

        if (!imageFileName || !fs.existsSync(sourceImagePath)) {
          skipped.push(`${file}: image not found (${labelmeData.imagePath})`);
          continue;
        }
        if (usedNames.has(imageFileName.toLowerCase())) {
          skipped.push(`${file}: duplicate image ${imageFileName}`);
          continue;
        }
        usedNames.add(imageFileName.toLowerCase());

        // Copy image to destination
        const destImagePath = path.join(
          splitName === 'train' ? trainImagesDir : splitName === 'val' ? valImagesDir : testImagesDir,
          imageFileName
        );
        fs.copyFileSync(sourceImagePath, destImagePath);

        // Add image to COCO
        const image: CocoImage = {
          id: imageId,
          file_name: imageFileName,
          height: labelmeData.imageHeight,
          width: labelmeData.imageWidth,
        };
        images.push(image);
        imageCount++;

        // Convert shapes to annotations
        for (const shape of labelmeData.shapes || []) {
          // Calculate bbox from polygon points
          const points = shape.points;
          if (!Array.isArray(points) || points.length < 2) continue;
          const xs = points.map(p => p[0]);
          const ys = points.map(p => p[1]);
          const minX = Math.min(...xs);
          const maxX = Math.max(...xs);
          const minY = Math.min(...ys);
          const maxY = Math.max(...ys);
          const width = maxX - minX;
          const height = maxY - minY;
          if (![minX, maxX, minY, maxY, width, height].every(Number.isFinite) || width <= 1e-5 || height <= 1e-5) {
            console.warn(`Invalid bbox in ${file} for label "${shape.label}", skipping...`);
            continue;
          }

          // Get or create category
          const label = String(shape.label ?? '').trim();
          if (!label) continue;
          if (!categories.has(label)) {
            categories.set(label, categoryId++);
          }
          const catId = categories.get(label)!;

          // Calculate polygon area using shoelace formula
          let area = 0;
          for (let i = 0; i < points.length; i++) {
            const j = (i + 1) % points.length;
            area += points[i][0] * points[j][1];
            area -= points[j][0] * points[i][1];
          }
          area = Math.abs(area) / 2 || width * height;

          // Build polygon segmentation for shapes that actually describe a
          // closed region (polygon / linestrip with >= 3 points). For pure
          // rectangle shapes (only two corners) we synthesize a 4-point
          // polygon from the bbox so the output is still usable for
          // instance-segmentation training.
          let segPoints: number[] = [];
          if (shape.shape_type === 'polygon' && points.length >= 3) {
            segPoints = points.flat();
          } else {
            // Rectangle / unsupported: derive a 4-corner polygon from bbox
            segPoints = [
              minX, minY,
              maxX, minY,
              maxX, maxY,
              minX, maxY,
            ];
          }

          const annotation: CocoAnnotation = {
            id: annotationId++,
            image_id: imageId,
            category_id: catId,
            bbox: [minX, minY, width, height],
            area: area,
            iscrowd: 0,
            segmentation: [segPoints],
          };
          annotations.push(annotation);
          annotationCount++;
        }

        imageId++;
      }

      return { images, annotations, imageCount, annotationCount };
    };

    // Convert all splits
    const trainData = convertSplit(trainFiles, 'train');
    const valData = convertSplit(valFiles, 'val');
    const testData = convertSplit(testFiles, 'test');

    if (skipped.length > 0) {
      console.warn(`[labelme-to-coco] Skipped ${skipped.length} sample(s):`, skipped.slice(0, 20));
    }

    // An empty split used to be reported as a successful conversion; the import
    // then showed an empty (or train-only) statistics panel with no explanation.
    if (trainData.imageCount === 0 || valData.imageCount === 0) {
      fs.rmSync(absoluteDatasetDir, { recursive: true, force: true });
      return NextResponse.json(
        {
          error: `Conversion produced ${trainData.imageCount} train and ${valData.imageCount} val images; both splits need at least one image.`,
          message: skipped.length > 0
            ? `${skipped.length} of ${labelmeFiles.length} samples were skipped, e.g. ${skipped.slice(0, 3).join('; ')}`
            : 'Add more samples or adjust the split ratios.',
          skipped: skipped.slice(0, 50),
        },
        { status: 400 }
      );
    }

    // Create COCO datasets
    const createCocoDataset = (images: CocoImage[], annotations: CocoAnnotation[]): CocoDataset => ({
      info: {
        description: `COCO Format Dataset: ${name}`,
        version: '1.0',
        year: new Date().getFullYear(),
        contributor: 'Auto Training Platform',
        date_created: new Date().toISOString(),
      },
      images,
      annotations,
      categories: Array.from(categories.entries()).map(([name, id]) => ({
        id,
        name,
        supercategory: 'object',
      })),
    });

    // Save annotation files
    fs.writeFileSync(
      path.join(annotationsDir, 'instance_train.json'),
      JSON.stringify(createCocoDataset(trainData.images, trainData.annotations), null, 2)
    );
    fs.writeFileSync(
      path.join(annotationsDir, 'instance_val.json'),
      JSON.stringify(createCocoDataset(valData.images, valData.annotations), null, 2)
    );
    if (testData.images.length > 0) {
      fs.writeFileSync(
        path.join(annotationsDir, 'instance_test.json'),
        JSON.stringify(createCocoDataset(testData.images, testData.annotations), null, 2)
      );
    }

    // Calculate statistics
    const totalAnnotations = trainData.annotationCount + valData.annotationCount + testData.annotationCount;
    const classDistribution: Record<string, number> = {};
    const classImageDistribution: Record<string, Set<number>> = {};
    
    [trainData, valData, testData].forEach((splitData, splitIndex) => {
      const imageIdOffset = splitIndex * 1000000; // Offset to distinguish splits
      splitData.annotations.forEach((anno, idx) => {
        const catName = Array.from(categories.entries()).find(([, id]) => id === anno.category_id)?.[0];
        if (catName) {
          classDistribution[catName] = (classDistribution[catName] || 0) + 1;
          if (!classImageDistribution[catName]) {
            classImageDistribution[catName] = new Set();
          }
          classImageDistribution[catName].add(anno.image_id + imageIdOffset);
        }
      });
    });

    const classStats = Array.from(categories.entries()).map(([name, id]) => ({
      name,
      id,
      count: classDistribution[name] || 0,
      imageCount: classImageDistribution[name]?.size || 0,
    }));

    // Return success without creating database record
    // User can import the converted COCO dataset using Import COCO Dataset feature
    return NextResponse.json({
      success: true,
      message: `Successfully converted ${labelmeFiles.length} Labelme files to COCO format`,
      outputPath: absoluteDatasetDir,
      stats: {
        totalFiles: labelmeFiles.length,
        trainCount: trainData.imageCount,
        valCount: valData.imageCount,
        testCount: testData.imageCount,
        categories: categories.size,
        totalAnnotations,
        classStats,
        skippedCount: skipped.length,
        skipped: skipped.slice(0, 50),
      },
    });

  } catch (error) {
    console.error('Error converting Labelme to COCO:', error);
    return NextResponse.json(
      { 
        error: 'Failed to convert Labelme to COCO', 
        message: error instanceof Error ? error.message : 'Unknown error' 
      },
      { status: 500 }
    );
  }
}
