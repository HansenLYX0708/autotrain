import { NextRequest, NextResponse } from 'next/server';
import * as fs from 'fs';
import * as path from 'path';
import sharp from 'sharp';
import { getSegColorMap } from '@/lib/seg-colors';

// GET /api/datasets/image?path=xxx
//   Serves a raw dataset image/mask file.
//
// Query params:
//   path      absolute path to the file (required)
//   colorize  "1" to render segmentation masks as pseudo-color PNGs
//   classes   number of classes (used with colorize to pick palette length)
//
// Behavior:
// - TIFF/TIF sources are always transcoded to PNG so browsers can display them.
// - When `colorize=1` is set AND the source decodes as a single-channel image
//   (grayscale mask where pixel value = class id), the endpoint maps each
//   pixel through the shared VOC palette (`@/lib/seg-colors`) and returns a
//   pseudo-color PNG. Multi-channel inputs are served untouched so pre-existing
//   pseudo-color PaddleSeg masks keep working.
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const filePath = searchParams.get('path');
    const colorize = searchParams.get('colorize') === '1';
    const classesParam = parseInt(searchParams.get('classes') || '0', 10);

    if (!filePath) {
      return NextResponse.json(
        { success: false, error: 'File path is required' },
        { status: 400 },
      );
    }

    if (filePath.includes('..')) {
      return NextResponse.json(
        { success: false, error: 'Invalid file path' },
        { status: 403 },
      );
    }

    if (!fs.existsSync(filePath)) {
      return NextResponse.json(
        { success: false, error: 'File not found' },
        { status: 404 },
      );
    }

    const stat = fs.statSync(filePath);
    if (!stat.isFile()) {
      return NextResponse.json(
        { success: false, error: 'Not a file' },
        { status: 400 },
      );
    }

    const ext = path.extname(filePath).toLowerCase();
    const isTiff = ext === '.tif' || ext === '.tiff';

    // Colorize path: decode raw, look up class-id -> RGB via VOC palette.
    if (colorize && classesParam > 0) {
      const meta = await sharp(filePath).metadata();
      // Only remap when the source is genuinely a single-channel label map;
      // otherwise fall through and serve the file as-is (already pseudo-color).
      if (meta.channels === 1) {
        const { data, info } = await sharp(filePath).raw().toBuffer({ resolveWithObject: true });
        const w = info.width;
        const h = info.height;
        const palette = getSegColorMap(classesParam);
        const rgb = Buffer.alloc(w * h * 3);
        for (let i = 0, j = 0; i < data.length; i++, j += 3) {
          const cls = data[i];
          const color = palette[cls] || [0, 0, 0];
          rgb[j] = color[0];
          rgb[j + 1] = color[1];
          rgb[j + 2] = color[2];
        }
        const png = await sharp(rgb, { raw: { width: w, height: h, channels: 3 } })
          .png({ compressionLevel: 9 })
          .toBuffer();
        return new NextResponse(png as unknown as BodyInit, {
          headers: {
            'Content-Type': 'image/png',
            'Cache-Control': 'public, max-age=86400',
          },
        });
      }
      // Multi-channel input: fall through to normal serving below.
    }

    // Non-colorize path: transcode TIFF to PNG (browsers can't display TIFF),
    // otherwise serve raw bytes with an appropriate content type.
    if (isTiff) {
      const png = await sharp(filePath).png({ compressionLevel: 9 }).toBuffer();
      return new NextResponse(png as unknown as BodyInit, {
        headers: {
          'Content-Type': 'image/png',
          'Cache-Control': 'public, max-age=86400',
        },
      });
    }

    const fileBuffer = fs.readFileSync(filePath);
    const contentTypeMap: Record<string, string> = {
      '.jpg': 'image/jpeg',
      '.jpeg': 'image/jpeg',
      '.png': 'image/png',
      '.gif': 'image/gif',
      '.bmp': 'image/bmp',
      '.webp': 'image/webp',
    };
    const contentType = contentTypeMap[ext] || 'application/octet-stream';

    return new NextResponse(fileBuffer, {
      headers: {
        'Content-Type': contentType,
        'Cache-Control': 'public, max-age=86400',
      },
    });
  } catch (error) {
    console.error('Error serving image:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to serve image',
        message: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 },
    );
  }
}
