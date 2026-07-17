import { NextRequest, NextResponse } from 'next/server';
import * as fs from 'fs';
import * as path from 'path';
import { db } from '@/lib/db';
import { sessions } from '../../auth/route';
import { getSegColorMap } from '@/lib/seg-colors';
import { parseListFile } from '@/lib/paddleseg-list';

interface CocoAnnotation {
  id: number;
  image_id: number;
  category_id: number;
  bbox: number[];
  area: number;
  iscrowd: number;
  segmentation?: number[] | number[][];
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

interface SampleImage {
  id: number;
  fileName: string;
  width: number;
  height: number;
  imagePath: string;
  annotations: {
    id: number;
    categoryId: number;
    categoryName: string;
    bbox: number[];
    area: number;
  }[];
}

// Parse COCO JSON file
async function parseCocoFile(filePath: string): Promise<CocoDataset | null> {
  try {
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

// GET /api/datasets/samples?datasetId=xxx&categoryId=xxx&limit=20
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
    const datasetId = searchParams.get('datasetId');
    const categoryId = searchParams.get('categoryId');
    const limit = parseInt(searchParams.get('limit') || '20', 10);

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

    // PaddleSeg datasets: list files (train.txt/val.txt) of "image mask" pairs.
    if ((dataset as any).project?.framework === 'PaddleSeg') {
      const root = dataset.datasetDir && path.isAbsolute(dataset.datasetDir)
        ? dataset.datasetDir
        : path.join(userDatabasePath, user.username, dataset.datasetDir || '');

      const listRel = dataset.trainAnnoPath || 'train.txt';
      const listPath = path.isAbsolute(listRel) ? listRel : path.join(root, listRel);
      if (!fs.existsSync(listPath)) {
        return NextResponse.json(
          { success: false, error: `List file not found at ${listPath}` },
          { status: 404 }
        );
      }

      // Anchor-based parser handles filenames that contain spaces (e.g.
      // "RMF 18(36,45)_1300kx.tif"), which naive whitespace splitting would
      // truncate at the first space.
      const entries = parseListFile(fs.readFileSync(listPath, 'utf-8'));

      // Resolve class names: prefer class_names.txt/labels.txt, else classStats.
      let classNames: string[] = [];
      for (const f of ['class_names.txt', 'labels.txt']) {
        const cnPath = path.join(root, f);
        if (fs.existsSync(cnPath)) {
          classNames = fs.readFileSync(cnPath, 'utf-8').split(/\r?\n/).map(s => s.trim()).filter(Boolean);
          break;
        }
      }
      if (classNames.length === 0) {
        try {
          const cs = JSON.parse(dataset.classStats || '{}');
          const arr = Array.isArray(cs) ? cs : (cs.train || []);
          classNames = arr.map((c: any) => c.name).filter(Boolean);
        } catch { /* ignore */ }
      }
      if (classNames.length === 0 && dataset.numClasses) {
        classNames = Array.from({ length: dataset.numClasses }, (_, i) => `class_${i}`);
      }

      const colorMap = getSegColorMap(Math.max(classNames.length, 1));
      const categories = classNames.map((name, id) => ({
        id,
        name,
        color: colorMap[id] || [0, 0, 0],
      }));

      const segSamples = entries.slice(0, limit).map(({ imageRel, maskRel }, idx) => ({
        id: idx,
        fileName: path.basename(imageRel),
        width: 0,
        height: 0,
        imagePath: path.isAbsolute(imageRel) ? imageRel : path.join(root, imageRel),
        maskPath: maskRel ? (path.isAbsolute(maskRel) ? maskRel : path.join(root, maskRel)) : '',
        annotations: [],
      }));

      return NextResponse.json({
        success: true,
        data: {
          type: 'segmentation',
          samples: segSamples,
          categories,
          totalImages: entries.length,
          totalAnnotations: 0,
        },
      });
    }

    // Build absolute annotation file path
    let annoPath: string | null = null;
    if (dataset.datasetDir && dataset.trainAnnoPath) {
      annoPath = path.join(userDatabasePath, user.username, dataset.datasetDir, dataset.trainAnnoPath);
    }

    if (!annoPath || !fs.existsSync(annoPath)) {
      return NextResponse.json(
        { success: false, error: `Annotation file not found at ${annoPath}` },
        { status: 404 }
      );
    }

    // Parse COCO file
    const cocoData = await parseCocoFile(annoPath);
    
    if (!cocoData) {
      return NextResponse.json(
        { success: false, error: 'Failed to parse COCO file' },
        { status: 400 }
      );
    }

    // Build category map
    const categoryMap = new Map<number, CocoCategory>();
    for (const cat of cocoData.categories) {
      categoryMap.set(cat.id, cat);
    }

    // Filter annotations by category if specified
    let filteredAnnotations = cocoData.annotations;
    if (categoryId) {
      const catId = parseInt(categoryId, 10);
      filteredAnnotations = cocoData.annotations.filter(ann => ann.category_id === catId);
    }

    // Get unique image IDs from filtered annotations
    const imageIdSet = new Set<number>();
    for (const ann of filteredAnnotations) {
      imageIdSet.add(ann.image_id);
    }

    // Convert to array and limit
    const imageIds = Array.from(imageIdSet).slice(0, limit);

    // Build image map
    const imageMap = new Map<number, CocoImage>();
    for (const img of cocoData.images) {
      imageMap.set(img.id, img);
    }

    // Build samples with absolute image paths
    const samples: SampleImage[] = [];
    let imageDir: string | null = null;
    if (dataset.datasetDir && dataset.trainImagePath) {
      imageDir = path.join(userDatabasePath, user.username, dataset.datasetDir, dataset.trainImagePath);
    }

    for (const imageId of imageIds) {
      const image = imageMap.get(imageId);
      if (!image) continue;

      // Get annotations for this image
      const imageAnnotations = filteredAnnotations.filter(ann => ann.image_id === imageId);
      
      const sample: SampleImage = {
        id: image.id,
        fileName: image.file_name,
        width: image.width,
        height: image.height,
        imagePath: imageDir ? path.join(imageDir, image.file_name) : image.file_name,
        annotations: imageAnnotations.map(ann => ({
          id: ann.id,
          categoryId: ann.category_id,
          categoryName: categoryMap.get(ann.category_id)?.name || 'Unknown',
          bbox: ann.bbox,
          area: ann.area,
        })),
      };

      samples.push(sample);
    }

    return NextResponse.json({
      success: true,
      data: {
        samples,
        categories: cocoData.categories,
        totalImages: cocoData.images.length,
        totalAnnotations: cocoData.annotations.length,
      },
    });
  } catch (error) {
    console.error('Error fetching samples:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to fetch samples',
        message: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}
