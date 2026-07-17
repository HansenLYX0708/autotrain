import { NextRequest, NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import { db } from '@/lib/db';
import * as fs from 'fs';
import * as path from 'path';
import { sessions } from '../../auth/route';
import sharp from 'sharp';
import { buildColorIndex } from '@/lib/seg-colors';
import { parseListFile } from '@/lib/paddleseg-list';

const execAsync = promisify(exec);

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
  width: number;
  height: number;
}

interface CocoCategory {
  id: number;
  name: string;
  supercategory?: string;
}

interface CocoDataset {
  images: CocoImage[];
  annotations: CocoAnnotation[];
  categories: CocoCategory[];
}

// Parse COCO JSON file
async function parseCocoFile(filePath: string): Promise<CocoDataset | null> {
  try {
    // Check if file exists
    if (!fs.existsSync(filePath)) {
      console.error('File not found:', filePath);
      return null;
    }

    const content = fs.readFileSync(filePath, 'utf-8');
    const data = JSON.parse(content);
    
    return {
      images: data.images || [],
      annotations: data.annotations || [],
      categories: data.categories || [],
    };
  } catch (error) {
    console.error('Error parsing COCO file:', error);
    return null;
  }
}

// Calculate statistics from COCO dataset
function calculateStats(cocoData: CocoDataset) {
  const categoryCount: Record<number, number> = {};
  const categoryImageCount: Record<number, Set<number>> = {};

  // Count annotations per category and track images per category
  for (const ann of cocoData.annotations) {
    categoryCount[ann.category_id] = (categoryCount[ann.category_id] || 0) + 1;
    
    if (!categoryImageCount[ann.category_id]) {
      categoryImageCount[ann.category_id] = new Set();
    }
    categoryImageCount[ann.category_id].add(ann.image_id);
  }

  // Build category statistics
  const categoryMap = new Map<number, CocoCategory>();
  for (const cat of cocoData.categories) {
    categoryMap.set(cat.id, cat);
  }

  const classStats = cocoData.categories.map(cat => ({
    id: cat.id,
    name: cat.name,
    count: categoryCount[cat.id] || 0,
    imageCount: categoryImageCount[cat.id]?.size || 0,
  }));

  return {
    numClasses: cocoData.categories.length,
    numAnnotations: cocoData.annotations.length,
    numImages: cocoData.images.length,
    classStats,
  };
}

// POST /api/datasets/parse - Parse COCO dataset and update statistics
export async function POST(request: NextRequest) {
  try {
    // Get current user from session
    const token = request.cookies.get("auth-token")?.value;
    if (!token || !sessions.has(token)) {
      return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
    }

    const session = sessions.get(token)!;
    const userId = session.userId;

    // Get user info
    const user = await db.user.findUnique({
      where: { id: userId },
      select: { id: true, username: true, role: true },
    });

    if (!user) {
      return NextResponse.json({ error: "User not found" }, { status: 404 });
    }

    // Get system config for userDatabasePath
    const systemConfig = await db.systemConfig.findFirst();
    const userDatabasePath = (systemConfig as any)?.userDatabasePath;

    if (!userDatabasePath) {
      return NextResponse.json(
        { error: "User database path not configured" },
        { status: 500 }
      );
    }

    const body = await request.json();
    const { datasetId, annotationPath, imageDir } = body;

    if (!datasetId) {
      return NextResponse.json(
        { success: false, error: 'Dataset ID is required' },
        { status: 400 }
      );
    }

    // Get dataset from database
    const dataset = await db.dataset.findUnique({
      where: { id: datasetId },
      include: { project: { select: { framework: true } } },
    });

    if (!dataset) {
      return NextResponse.json(
        { success: false, error: 'Dataset not found' },
        { status: 404 }
      );
    }

    // PaddleSeg datasets: compute counts from list files and a sampled per-class
    // pixel/image distribution by decoding pseudo-color masks.
    if ((dataset as any).project?.framework === 'PaddleSeg') {
      const root = dataset.datasetDir && path.isAbsolute(dataset.datasetDir)
        ? dataset.datasetDir
        : path.join(userDatabasePath, user.username, dataset.datasetDir || '');

      // Robust parser that handles filenames containing spaces (see
      // `@/lib/paddleseg-list`) — a naive whitespace split would truncate
      // paths like "RMF 18(36,45)_1300kx.tif".
      const readList = (rel: string | null) => {
        if (!rel) return [] as ReturnType<typeof parseListFile>;
        const p = path.isAbsolute(rel) ? rel : path.join(root, rel);
        if (!fs.existsSync(p)) return [] as ReturnType<typeof parseListFile>;
        return parseListFile(fs.readFileSync(p, 'utf-8'));
      };

      const trainEntries = readList(dataset.trainAnnoPath || 'train.txt');
      const valEntries = readList(dataset.evalAnnoPath || 'val.txt');

      let classNames: string[] = [];
      for (const f of ['class_names.txt', 'labels.txt']) {
        const cnPath = path.join(root, f);
        if (fs.existsSync(cnPath)) {
          classNames = fs.readFileSync(cnPath, 'utf-8').split(/\r?\n/).map(s => s.trim()).filter(Boolean);
          break;
        }
      }
      if (classNames.length === 0 && dataset.numClasses) {
        classNames = Array.from({ length: dataset.numClasses }, (_, i) => `class_${i}`);
      }
      const numClasses = classNames.length || dataset.numClasses || 1;

      // Sample masks evenly across the train set to estimate class distribution.
      const SAMPLE = 40;
      const colorIndex = buildColorIndex(numClasses);
      const pixelCount = new Array(numClasses).fill(0);
      const imageCount = new Array(numClasses).fill(0);
      let sampledMasks = 0;
      const step = Math.max(1, Math.floor(trainEntries.length / SAMPLE));
      for (let i = 0; i < trainEntries.length && sampledMasks < SAMPLE; i += step) {
        const maskRel = trainEntries[i].maskRel;
        if (!maskRel) continue;
        const maskPath = path.isAbsolute(maskRel) ? maskRel : path.join(root, maskRel);
        if (!fs.existsSync(maskPath)) continue;
        try {
          const { data, info } = await sharp(maskPath)
            .resize({ width: 512, height: 512, fit: 'inside', kernel: sharp.kernel.nearest, withoutEnlargement: true })
            .raw()
            .toBuffer({ resolveWithObject: true });
          const ch = info.channels;
          const present = new Set<number>();
          if (ch === 1) {
            // Grayscale label map: pixel value is the class id directly
            // (labelme-to-paddleseg output format).
            for (let p = 0; p < data.length; p++) {
              const cls = data[p];
              if (cls < numClasses) {
                pixelCount[cls] += 1;
                present.add(cls);
              }
            }
          } else {
            // Pseudo-color RGB mask: look each pixel up in the VOC palette.
            for (let p = 0; p < data.length; p += ch) {
              const cls = colorIndex.get(`${data[p]},${data[p + 1]},${data[p + 2]}`);
              if (cls !== undefined) {
                pixelCount[cls] += 1;
                present.add(cls);
              }
            }
          }
          present.forEach(c => { imageCount[c] += 1; });
          sampledMasks += 1;
        } catch (e) {
          console.error('[Parse Seg] Failed to decode mask', maskPath, e);
        }
      }

      const segClassStats = classNames.map((name, id) => ({
        id,
        name,
        count: pixelCount[id] || 0,
        imageCount: imageCount[id] || 0,
      }));

      const updatedSegDataset = await db.dataset.update({
        where: { id: datasetId },
        data: {
          numClasses,
          numAnnotations: 0,
          numTrainImages: trainEntries.length,
          numEvalImages: valEntries.length,
          classStats: JSON.stringify({ train: segClassStats, eval: [], sampledMasks }),
        },
        include: { project: { select: { id: true, name: true } } },
      });

      return NextResponse.json({
        success: true,
        data: updatedSegDataset,
        stats: {
          numClasses,
          numTrainImages: trainEntries.length,
          numEvalImages: valEntries.length,
          classStats: segClassStats,
          sampledMasks,
        },
        message: `PaddleSeg statistics computed from ${sampledMasks} sampled mask(s)`,
      });
    }

    // Determine annotation file path
    let annoPath: string | null = null;
    
    if (annotationPath) {
      // If absolute path provided, use it; otherwise build absolute path
      annoPath = path.isAbsolute(annotationPath) 
        ? annotationPath 
        : path.join(userDatabasePath, user.username, annotationPath);
    } else if (dataset.datasetDir && dataset.trainAnnoPath) {
      // Build absolute path: {userDatabasePath}/{username}/{datasetDir}/{trainAnnoPath}
      annoPath = path.join(userDatabasePath, user.username, dataset.datasetDir, dataset.trainAnnoPath);
    }

    if (!annoPath) {
      return NextResponse.json(
        { success: false, error: 'Annotation path is required' },
        { status: 400 }
      );
    }

    // Parse COCO file
    const cocoData = await parseCocoFile(annoPath);
    
    if (!cocoData) {
      return NextResponse.json(
        { success: false, error: `Failed to parse COCO file at ${annoPath}. Check if the file exists and is valid JSON.` },
        { status: 400 }
      );
    }

    // Calculate statistics
    const stats = calculateStats(cocoData);

    // Parse eval annotation and calculate stats
    let numEvalImages = 0;
    let evalClassStats: Array<{ id: number; name: string; count: number; imageCount: number }> = [];
    
    if (dataset.datasetDir && dataset.evalAnnoPath) {
      // Build absolute path for eval annotation
      const evalPath = path.join(userDatabasePath, user.username, dataset.datasetDir, dataset.evalAnnoPath);
      const evalData = await parseCocoFile(evalPath);
      if (evalData) {
        numEvalImages = evalData.images.length;
        const evalStats = calculateStats(evalData);
        evalClassStats = evalStats.classStats;
      }
    }

    // Update dataset with statistics
    const updatedDataset = await db.dataset.update({
      where: { id: datasetId },
      data: {
        numClasses: stats.numClasses,
        numAnnotations: stats.numAnnotations,
        numTrainImages: stats.numImages,
        numEvalImages: numEvalImages,
        classStats: JSON.stringify({ train: stats.classStats, eval: evalClassStats }),
      },
      include: {
        project: {
          select: { id: true, name: true },
        },
      },
    });

    return NextResponse.json({
      success: true,
      data: updatedDataset,
      stats: {
        numClasses: stats.numClasses,
        numAnnotations: stats.numAnnotations,
        numTrainImages: stats.numImages,
        numEvalImages,
        classStats: stats.classStats,
        evalClassStats,
      },
      message: 'Dataset parsed successfully',
    });
  } catch (error) {
    console.error('Error parsing dataset:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to parse dataset',
        message: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}

// GET /api/datasets/parse - Parse COCO dataset file (without updating database)
export async function GET(request: NextRequest) {
  try {
    // Get current user from session
    const token = request.cookies.get("auth-token")?.value;
    if (!token || !sessions.has(token)) {
      return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
    }

    const session = sessions.get(token)!;
    const userId = session.userId;

    // Get user info
    const user = await db.user.findUnique({
      where: { id: userId },
      select: { id: true, username: true, role: true },
    });

    if (!user) {
      return NextResponse.json({ error: "User not found" }, { status: 404 });
    }

    // Get system config for userDatabasePath
    const systemConfig = await db.systemConfig.findFirst();
    const userDatabasePath = (systemConfig as any)?.userDatabasePath;

    if (!userDatabasePath) {
      return NextResponse.json(
        { error: "User database path not configured" },
        { status: 500 }
      );
    }

    const { searchParams } = new URL(request.url);
    const filePath = searchParams.get('path');
    const datasetDir = searchParams.get('datasetDir');

    if (!filePath) {
      return NextResponse.json(
        { success: false, error: 'File path is required' },
        { status: 400 }
      );
    }

    // Build absolute path
    const absolutePath = path.isAbsolute(filePath)
      ? filePath
      : datasetDir
        ? path.join(userDatabasePath, user.username, datasetDir, filePath)
        : path.join(userDatabasePath, user.username, filePath);

    // PaddleSeg detection: a `.txt` list file lives next to `class_names.txt`
    // (labels.txt as fallback). Return `numClasses` = number of class entries
    // so the Import dialog can auto-fill it before the dataset row exists.
    const lowerPath = absolutePath.toLowerCase();
    if (lowerPath.endsWith('.txt')) {
      const dir = path.dirname(absolutePath);
      for (const f of ['class_names.txt', 'labels.txt']) {
        const cnPath = path.join(dir, f);
        if (fs.existsSync(cnPath)) {
          const names = fs.readFileSync(cnPath, 'utf-8')
            .split(/\r?\n/)
            .map(s => s.trim())
            .filter(Boolean);
          if (names.length > 0) {
            return NextResponse.json({
              success: true,
              data: {
                numClasses: names.length,
                numAnnotations: 0,
                numImages: fs.existsSync(absolutePath)
                  ? fs.readFileSync(absolutePath, 'utf-8').split(/\r?\n/).filter(l => l.trim()).length
                  : 0,
                classStats: names.map((name, id) => ({ id, name, count: 0 })),
                categories: names.map((name, id) => ({ id, name })),
                format: 'PaddleSeg',
              },
            });
          }
        }
      }
    }

    const cocoData = await parseCocoFile(absolutePath);
    
    if (!cocoData) {
      return NextResponse.json(
        { success: false, error: `Failed to parse COCO file at ${absolutePath}` },
        { status: 400 }
      );
    }

    const stats = calculateStats(cocoData);

    return NextResponse.json({
      success: true,
      data: {
        numClasses: stats.numClasses,
        numAnnotations: stats.numAnnotations,
        numImages: stats.numImages,
        classStats: stats.classStats,
        categories: cocoData.categories,
      },
    });
  } catch (error) {
    console.error('Error parsing dataset:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to parse dataset',
        message: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}
