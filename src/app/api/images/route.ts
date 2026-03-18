import { NextRequest, NextResponse } from 'next/server';
import { readFile, existsSync, readdirSync, statSync } from 'fs';
import { join, extname, basename } from 'path';
import { promisify } from 'util';

const readFileAsync = promisify(readFile);

// Supported image extensions
const IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'];

// GET /api/images - Serve an image or list images in a directory
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const path = searchParams.get('path');
    const dir = searchParams.get('dir');
    const action = searchParams.get('action') || 'serve'; // 'serve' or 'list'

    // List images in a directory
    if (action === 'list' && dir) {
      return listImages(dir);
    }

    // Serve a single image
    if (!path) {
      return NextResponse.json(
        { success: false, error: 'path parameter is required' },
        { status: 400 }
      );
    }

    // Check if file exists
    if (!existsSync(path)) {
      return NextResponse.json(
        { success: false, error: 'Image file not found' },
        { status: 404 }
      );
    }

    // Read the image file
    const imageBuffer = await readFileAsync(path);
    const ext = extname(path).toLowerCase();
    
    // Determine content type
    const contentTypes: Record<string, string> = {
      '.jpg': 'image/jpeg',
      '.jpeg': 'image/jpeg',
      '.png': 'image/png',
      '.bmp': 'image/bmp',
      '.gif': 'image/gif',
      '.tiff': 'image/tiff',
      '.webp': 'image/webp',
    };
    
    const contentType = contentTypes[ext] || 'application/octet-stream';
    
    return new NextResponse(imageBuffer, {
      headers: {
        'Content-Type': contentType,
        'Cache-Control': 'public, max-age=3600',
      },
    });
  } catch (error) {
    console.error('Error serving image:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to serve image' },
      { status: 500 }
    );
  }
}

// List images in a directory
async function listImages(dir: string) {
  try {
    if (!existsSync(dir)) {
      return NextResponse.json({
        success: true,
        data: {
          dir,
          images: [],
          message: 'Directory does not exist'
        }
      });
    }

    const files = readdirSync(dir);
    const images: Array<{
      name: string;
      path: string;
      size: number;
      mtime: string;
    }> = [];

    for (const file of files) {
      const filePath = join(dir, file);
      try {
        const stat = statSync(filePath);
        
        if (stat.isFile()) {
          const ext = extname(file).toLowerCase();
          if (IMAGE_EXTENSIONS.includes(ext)) {
            images.push({
              name: file,
              path: filePath,
              size: stat.size,
              mtime: stat.mtime.toISOString(),
            });
          }
        }
      } catch {
        // Skip files that can't be accessed
      }
    }

    // Sort by modification time (newest first)
    images.sort((a, b) => new Date(b.mtime).getTime() - new Date(a.mtime).getTime());

    return NextResponse.json({
      success: true,
      data: {
        dir,
        images,
        imageCount: images.length,
      }
    });
  } catch (error) {
    console.error('Error listing images:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to list images' },
      { status: 500 }
    );
  }
}
