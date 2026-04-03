import { NextRequest, NextResponse } from 'next/server';
import * as fs from 'fs';
import * as path from 'path';
import { db } from '@/lib/db';
import { sessions } from '../../auth/route';

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
    });

    if (!dataset) {
      return NextResponse.json(
        { success: false, error: 'Dataset not found' },
        { status: 404 }
      );
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
