import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { requireAuth } from '@/lib/auth';
import * as fs from 'fs';
import * as path from 'path';

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
    if (!projectId || !name || !labelmeImagesPath || !labelmeAnnotationsPath) {
      return NextResponse.json(
        { error: 'Missing required fields: projectId, name, labelmeImagesPath, labelmeAnnotationsPath' },
        { status: 400 }
      );
    }

    // Validate ratios
    const train = parseFloat(trainRatio) || 0;
    const val = parseFloat(valRatio) || 0;
    const test = parseFloat(testRatio) || 0;
    const total = train + val + test;

    if (Math.abs(total - 1.0) > 0.001) {
      return NextResponse.json(
        { error: `Train/Val/Test ratios must sum to 1.0, current sum: ${total.toFixed(3)}` },
        { status: 400 }
      );
    }

    if (train <= 0 || val <= 0 || train + val >= 1.0) {
      return NextResponse.json(
        { error: 'Train and Val ratios must be positive and less than 1.0' },
        { status: 400 }
      );
    }

    // Get project and verify user access
    const project = await db.project.findFirst({
      where: role === 'admin' 
        ? { id: projectId }
        : { id: projectId, userId },
    });

    if (!project) {
      return NextResponse.json(
        { error: 'Project not found or access denied' },
        { status: 404 }
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

    // Generate output paths
    const datasetDir = outputDatasetDir || `dataset/${name.toLowerCase().replace(/\s+/g, '_')}`;
    const absoluteDatasetDir = path.isAbsolute(datasetDir)
      ? datasetDir
      : path.join(workDir, datasetDir);

    // Create output directories
    const trainImagesDir = path.join(absoluteDatasetDir, 'images', 'train');
    const valImagesDir = path.join(absoluteDatasetDir, 'images', 'val');
    const testImagesDir = path.join(absoluteDatasetDir, 'images', 'test');
    const annotationsDir = path.join(absoluteDatasetDir, 'annotations');

    [trainImagesDir, valImagesDir, testImagesDir, annotationsDir].forEach(dir => {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
    });

    // Read all labelme annotation files
    const labelmeFiles = fs.readdirSync(absoluteAnnotationsPath)
      .filter(file => file.endsWith('.json'));

    if (labelmeFiles.length === 0) {
      return NextResponse.json(
        { error: 'No labelme annotation files found (.json files)' },
        { status: 400 }
      );
    }

    // Shuffle files for random split
    const shuffledFiles = labelmeFiles.sort(() => Math.random() - 0.5);
    
    // Calculate split indices
    const totalFiles = shuffledFiles.length;
    const trainCount = Math.floor(totalFiles * train);
    const valCount = Math.floor(totalFiles * val);
    const testCount = totalFiles - trainCount - valCount;

    const trainFiles = shuffledFiles.slice(0, trainCount);
    const valFiles = shuffledFiles.slice(trainCount, trainCount + valCount);
    const testFiles = shuffledFiles.slice(trainCount + valCount);

    // Convert each split
    const categories = new Map<string, number>();
    let categoryId = 1;
    let annotationId = 1;

    const convertSplit = (files: string[], splitName: string): { images: CocoImage[]; annotations: CocoAnnotation[]; imageCount: number; annotationCount: number } => {
      const images: CocoImage[] = [];
      const annotations: CocoAnnotation[] = [];
      let imageId = 1;
      let imageCount = 0;
      let annotationCount = 0;

      for (const file of files) {
        const labelmePath = path.join(absoluteAnnotationsPath, file);
        const labelmeData: LabelmeData = JSON.parse(fs.readFileSync(labelmePath, 'utf-8'));

        // Find corresponding image
        const imageFileName = labelmeData.imagePath;
        const sourceImagePath = path.join(absoluteImagesPath, imageFileName);
        
        if (!fs.existsSync(sourceImagePath)) {
          console.warn(`Image not found: ${sourceImagePath}, skipping...`);
          continue;
        }

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
        for (const shape of labelmeData.shapes) {
          // Get or create category
          if (!categories.has(shape.label)) {
            categories.set(shape.label, categoryId++);
          }
          const catId = categories.get(shape.label)!;

          // Calculate bbox from polygon points
          const points = shape.points;
          const xs = points.map(p => p[0]);
          const ys = points.map(p => p[1]);
          const minX = Math.min(...xs);
          const maxX = Math.max(...xs);
          const minY = Math.min(...ys);
          const maxY = Math.max(...ys);
          const width = maxX - minX;
          const height = maxY - minY;

          // Calculate polygon area using shoelace formula
          let area = 0;
          for (let i = 0; i < points.length; i++) {
            const j = (i + 1) % points.length;
            area += points[i][0] * points[j][1];
            area -= points[j][0] * points[i][1];
          }
          area = Math.abs(area) / 2;

          const annotation: CocoAnnotation = {
            id: annotationId++,
            image_id: imageId,
            category_id: catId,
            bbox: [minX, minY, width, height],
            area: area,
            iscrowd: 0,
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
      path.join(annotationsDir, 'train.json'),
      JSON.stringify(createCocoDataset(trainData.images, trainData.annotations), null, 2)
    );
    fs.writeFileSync(
      path.join(annotationsDir, 'val.json'),
      JSON.stringify(createCocoDataset(valData.images, valData.annotations), null, 2)
    );
    if (testData.images.length > 0) {
      fs.writeFileSync(
        path.join(annotationsDir, 'test.json'),
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

    // Create dataset record in database
    const dataset = await db.dataset.create({
      data: {
        name,
        description: description || `Converted from Labelme format. Train: ${trainCount}, Val: ${valCount}, Test: ${testCount}`,
        format: 'COCO',
        projectId,
        userId,
        trainImagePath: path.join(datasetDir, 'images', 'train'),
        trainAnnoPath: path.join(datasetDir, 'annotations', 'train.json'),
        evalImagePath: path.join(datasetDir, 'images', 'val'),
        evalAnnoPath: path.join(datasetDir, 'annotations', 'val.json'),
        datasetDir,
        numClasses: categories.size,
        numAnnotations: totalAnnotations,
        numTrainImages: trainData.imageCount,
        numEvalImages: valData.imageCount,
        classStats: JSON.stringify(classStats),
      },
    });

    return NextResponse.json({
      success: true,
      data: dataset,
      message: `Successfully converted ${labelmeFiles.length} Labelme files to COCO format`,
      stats: {
        totalFiles: labelmeFiles.length,
        trainCount: trainData.imageCount,
        valCount: valData.imageCount,
        testCount: testData.imageCount,
        categories: categories.size,
        totalAnnotations,
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
