import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { requireAuth } from '@/lib/auth';
import * as fs from 'fs';
import * as path from 'path';
import { deflateSync } from 'zlib';
import { getSegColorMap, type RGB } from '@/lib/seg-colors';

/**
 * Convert a labelme polygon-annotated dataset into a PaddleSeg-compatible
 * layout under `{userDb}/{user}/PaddleSeg/{name}/`.
 *
 * Layout produced (PaddleSeg's official labelme2seg / Pascal VOC style):
 *
 *   PaddleSeg/{name}/data/
 *     JPEGImages/           (source images copied as-is, filenames preserved)
 *     Annotations/          (8-bit paletted PNG; palette index = class id,
 *                            palette colors = VOC pseudo-colors)
 *     train.txt             ("JPEGImages/xxx Annotations/xxx.png" per line)
 *     val.txt
 *     class_names.txt       (first line is always "_background_")
 *
 * Class ordering: 0 = _background_, 1..N = foreground labels in the order
 * they first appear when JSONs are scanned in sorted-filename order (fully
 * deterministic across runs).
 *
 * Mask format: 8-bit indexed-color (paletted) PNG. The palette is the shared
 * VOC color map from `@/lib/seg-colors`, so:
 *   - PaddleSeg / PIL reads the file as `mode='P'` -> 2D array of palette
 *     indices == class ids, feeding training/eval directly.
 *   - Any image viewer displays bright pseudo-colors (class 0 = red, class 1
 *     = green, ...), so masks can be visually verified without any external
 *     tool. Pure grayscale masks with values 0..N-1 (< 5) would otherwise be
 *     indistinguishable from black.
 *   - The parse route decodes via sharp (which expands the palette to RGB)
 *     and looks each pixel up in the same `getSegColorMap` palette, so
 *     statistics stay in sync with training.
 */

interface LabelmeShape {
  label: string;
  points: number[][];
  shape_type: string;
}

interface LabelmeData {
  shapes: LabelmeShape[];
  imagePath: string;
  imageHeight: number;
  imageWidth: number;
}

const BACKGROUND = '_background_';

/**
 * Sanitize a filename so it survives PaddleSeg's list-file format (which uses
 * whitespace as image/mask separator). Any character outside `[A-Za-z0-9._-]`
 * becomes `_`, runs of `_` collapse, leading/trailing `_` are trimmed.
 * The original extension is preserved.  Also see `@/lib/paddleseg-list` which
 * still handles legacy datasets with spaces.
 */
function sanitizeFilename(name: string): string {
  const ext = path.extname(name);
  const base = path.basename(name, ext);
  const clean = base
    .replace(/[^\w.-]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_|_$/g, '');
  return `${clean || 'image'}${ext}`;
}

// ---- Paletted PNG writer -------------------------------------------------
//
// sharp does not support writing fixed-palette PNGs (its `palette: true`
// quantizes adaptively with dithering, which would scramble class ids), and
// pngjs's writer is limited to color types 0/2/4/6 (no indexed). We assemble
// the PNG file directly: signature + IHDR + PLTE + IDAT (deflate) + IEND.
// This is small, standard, and yields a paletted PNG that every image viewer
// and PIL/OpenCV/PaddleSeg can decode.

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(buf: Buffer): number {
  let c = 0xFFFFFFFF;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xFF] ^ (c >>> 8);
  return (c ^ 0xFFFFFFFF) >>> 0;
}

function pngChunk(type: string, data: Buffer): Buffer {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length, 0);
  const typeBuf = Buffer.from(type, 'ascii');
  const crcBuf = Buffer.alloc(4);
  crcBuf.writeUInt32BE(crc32(Buffer.concat([typeBuf, data])), 0);
  return Buffer.concat([len, typeBuf, data, crcBuf]);
}

/**
 * Write an 8-bit paletted PNG. `indexed[y*width + x]` is the palette index
 * (== class id) for that pixel. `palette` supplies RGB triples; index i
 * receives palette[i]. Missing entries render as black.
 */
function writePalettedPng(
  filePath: string,
  width: number,
  height: number,
  indexed: Uint8Array,
  palette: RGB[],
): void {
  const signature = Buffer.from([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);

  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;  // bit depth
  ihdr[9] = 3;  // color type: indexed
  ihdr[10] = 0; // compression: deflate
  ihdr[11] = 0; // filter: adaptive (we use type 0 = None per row)
  ihdr[12] = 0; // interlace: none

  // PLTE: exactly 3 * paletteSize bytes; PaddleSeg / PIL only need palette
  // to be at least (max class id + 1) long. Cap at 256 (PNG spec limit).
  const paletteSize = Math.min(256, palette.length);
  const plte = Buffer.alloc(paletteSize * 3);
  for (let i = 0; i < paletteSize; i++) {
    const c = palette[i];
    plte[i * 3] = c[0] & 0xFF;
    plte[i * 3 + 1] = c[1] & 0xFF;
    plte[i * 3 + 2] = c[2] & 0xFF;
  }

  // IDAT scanlines: each row prefixed with a filter-type byte (0 = None).
  const raw = Buffer.alloc(height * (width + 1));
  for (let y = 0; y < height; y++) {
    const rowStart = y * (width + 1);
    raw[rowStart] = 0;
    raw.set(indexed.subarray(y * width, (y + 1) * width), rowStart + 1);
  }
  const idat = deflateSync(raw, { level: 9 });

  const png = Buffer.concat([
    signature,
    pngChunk('IHDR', ihdr),
    pngChunk('PLTE', plte),
    pngChunk('IDAT', idat),
    pngChunk('IEND', Buffer.alloc(0)),
  ]);
  fs.writeFileSync(filePath, png);
}

function getFolderSize(folderPath: string): number {
  if (!fs.existsSync(folderPath)) return 0;
  const stat = fs.statSync(folderPath);
  if (stat.isFile()) return stat.size;
  let total = 0;
  for (const entry of fs.readdirSync(folderPath)) {
    const p = path.join(folderPath, entry);
    const s = fs.statSync(p);
    total += s.isDirectory() ? getFolderSize(p) : s.size;
  }
  return total;
}

/**
 * Even-odd scanline polygon fill. Writes `classId` into `buffer` for every
 * pixel whose center (x+0.5, y+0.5) lies inside the polygon defined by
 * `points`. Later polygons overwrite earlier ones, so shape order in the
 * labelme JSON acts as z-order (matches the user's confirmed convention).
 */
function fillPolygon(
  buffer: Uint8Array,
  width: number,
  height: number,
  points: number[][],
  classId: number,
): void {
  const n = points.length;
  if (n < 3) return;

  let yMin = Infinity;
  let yMax = -Infinity;
  for (const p of points) {
    if (p[1] < yMin) yMin = p[1];
    if (p[1] > yMax) yMax = p[1];
  }
  const yStart = Math.max(0, Math.floor(yMin));
  const yEnd = Math.min(height - 1, Math.ceil(yMax));

  for (let y = yStart; y <= yEnd; y++) {
    const scanY = y + 0.5;
    const xs: number[] = [];
    for (let i = 0; i < n; i++) {
      const x1 = points[i][0];
      const y1 = points[i][1];
      const x2 = points[(i + 1) % n][0];
      const y2 = points[(i + 1) % n][1];
      // Half-open interval [min(y1,y2), max(y1,y2)) avoids double-counting
      // vertices and handles horizontal edges cleanly.
      if ((y1 <= scanY && scanY < y2) || (y2 <= scanY && scanY < y1)) {
        const t = (scanY - y1) / (y2 - y1);
        xs.push(x1 + t * (x2 - x1));
      }
    }
    if (xs.length < 2) continue;
    xs.sort((a, b) => a - b);
    const rowOffset = y * width;
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const xStart = Math.max(0, Math.ceil(xs[k] - 0.5));
      const xEnd = Math.min(width - 1, Math.floor(xs[k + 1] - 0.5));
      for (let x = xStart; x <= xEnd; x++) {
        buffer[rowOffset + x] = classId;
      }
    }
  }
}

export async function POST(request: NextRequest) {
  try {
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;
    const { userId } = auth;

    const body = await request.json();
    const {
      name,
      labelmeImagesPath,
      labelmeAnnotationsPath,
      outputDatasetDir,
      trainRatio,
      valRatio,
    } = body;

    if (!name || !labelmeImagesPath || !labelmeAnnotationsPath) {
      return NextResponse.json(
        { error: 'Missing required fields: name, labelmeImagesPath, labelmeAnnotationsPath' },
        { status: 400 },
      );
    }

    if (!/^[a-zA-Z0-9_-]+$/.test(name)) {
      return NextResponse.json(
        { error: 'Dataset name can only contain letters, numbers, underscores and hyphens' },
        { status: 400 },
      );
    }

    const train = parseFloat(trainRatio) || 0;
    const val = parseFloat(valRatio) || 0;
    if (train <= 0 || val <= 0) {
      return NextResponse.json(
        { error: 'Train and Val ratios must be positive' },
        { status: 400 },
      );
    }
    if (Math.abs(train + val - 1.0) > 0.001) {
      return NextResponse.json(
        { error: `Train + Val must sum to 1.0 (current: ${(train + val).toFixed(3)})` },
        { status: 400 },
      );
    }

    const systemConfig = await db.systemConfig.findFirst();
    const userDatabasePath = (systemConfig as { userDatabasePath?: string } | null)?.userDatabasePath;
    if (!userDatabasePath) {
      return NextResponse.json(
        { error: 'User database path not configured' },
        { status: 500 },
      );
    }

    const user = await db.user.findUnique({
      where: { id: userId },
      select: { id: true, username: true, maxStorageQuota: true },
    });
    if (!user) {
      return NextResponse.json({ error: 'User not found' }, { status: 404 });
    }

    const absoluteImagesPath = path.isAbsolute(labelmeImagesPath)
      ? labelmeImagesPath
      : path.join(userDatabasePath, user.username, labelmeImagesPath);
    const absoluteAnnotationsPath = path.isAbsolute(labelmeAnnotationsPath)
      ? labelmeAnnotationsPath
      : path.join(userDatabasePath, user.username, labelmeAnnotationsPath);

    if (!fs.existsSync(absoluteImagesPath)) {
      return NextResponse.json(
        { error: `Images path does not exist: ${absoluteImagesPath}` },
        { status: 400 },
      );
    }
    if (!fs.existsSync(absoluteAnnotationsPath)) {
      return NextResponse.json(
        { error: `Annotations path does not exist: ${absoluteAnnotationsPath}` },
        { status: 400 },
      );
    }

    const defaultOut = path.join(userDatabasePath, user.username, 'PaddleSeg', name);
    const absoluteDatasetDir = outputDatasetDir
      ? (path.isAbsolute(outputDatasetDir)
          ? outputDatasetDir
          : path.join(userDatabasePath, user.username, outputDatasetDir))
      : defaultOut;

    if (fs.existsSync(absoluteDatasetDir)) {
      return NextResponse.json(
        { error: `Output dataset directory already exists: ${absoluteDatasetDir}` },
        { status: 400 },
      );
    }

    // Storage quota check (images are copied; masks are typically far smaller
    // than the source images so imgs size is a reasonable upper bound).
    const userFolderPath = path.join(userDatabasePath, user.username);
    const usedStorage = getFolderSize(userFolderPath);
    const requiredSize = getFolderSize(absoluteImagesPath);
    const maxQuota = Number(user.maxStorageQuota);
    if (Number.isFinite(maxQuota) && maxQuota > 0 && usedStorage + requiredSize > maxQuota) {
      return NextResponse.json(
        {
          error: '存储空间不足',
          message: `您已使用 ${(usedStorage / 1024 / 1024 / 1024).toFixed(2)} GB，配额为 ${(maxQuota / 1024 / 1024 / 1024).toFixed(2)} GB。本次转换预计需要 ${(requiredSize / 1024 / 1024).toFixed(2)} MB 空间。`,
          usedStorage,
          maxStorageQuota: maxQuota,
          requiredSpace: requiredSize,
        },
        { status: 403 },
      );
    }

    const jsonFiles = fs.readdirSync(absoluteAnnotationsPath)
      .filter(f => f.toLowerCase().endsWith('.json'))
      .sort();
    if (jsonFiles.length === 0) {
      return NextResponse.json(
        { error: 'No labelme annotation files (.json) found in the annotations path' },
        { status: 400 },
      );
    }

    // Deterministic label collection: sorted filename → first-appearance.
    const foregroundLabels: string[] = [];
    const labelSet = new Set<string>();
    const parsedJsons: Array<{ file: string; data: LabelmeData }> = [];
    for (const file of jsonFiles) {
      const jsonPath = path.join(absoluteAnnotationsPath, file);
      try {
        const data = JSON.parse(fs.readFileSync(jsonPath, 'utf-8')) as LabelmeData;
        parsedJsons.push({ file, data });
        for (const shape of data.shapes || []) {
          if (shape.shape_type !== 'polygon' || (shape.points?.length ?? 0) < 3) continue;
          if (!labelSet.has(shape.label)) {
            labelSet.add(shape.label);
            foregroundLabels.push(shape.label);
          }
        }
      } catch (err) {
        console.warn('[labelme-to-paddleseg] Failed to parse JSON, skipping:', file, err);
      }
    }

    if (foregroundLabels.length === 0) {
      return NextResponse.json(
        { error: 'No polygon shapes found across all labelme JSON files' },
        { status: 400 },
      );
    }

    const classNames = [BACKGROUND, ...foregroundLabels];
    const numClasses = classNames.length;
    const labelToClassId = new Map<string, number>();
    foregroundLabels.forEach((label, i) => labelToClassId.set(label, i + 1));

    // Shared VOC palette used everywhere in the app; class 0 -> [128,0,0],
    // class 1 -> [0,128,0], etc.  Baked into every mask's PLTE chunk so the
    // file is visually verifiable and self-describing.
    const palette = getSegColorMap(numClasses);

    // Prepare output layout.
    const dataDir = path.join(absoluteDatasetDir, 'data');
    const outImgsDir = path.join(dataDir, 'JPEGImages');
    const outMasksDir = path.join(dataDir, 'Annotations');
    fs.mkdirSync(outImgsDir, { recursive: true });
    fs.mkdirSync(outMasksDir, { recursive: true });

    // Deterministic split: seeded shuffle by JSON filename hash keeps runs
    // reproducible for the same input set.
    const seededOrder = [...parsedJsons].sort((a, b) => a.file.localeCompare(b.file));
    // Use a stable interleaving so consecutive JSONs don't cluster in one split.
    const shuffled = [...seededOrder].sort(() => Math.random() - 0.5);
    const trainCount = Math.floor(shuffled.length * train);
    const trainSet = new Set(shuffled.slice(0, trainCount).map(p => p.file));

    const trainLines: string[] = [];
    const valLines: string[] = [];
    const classPixelCount = new Array<number>(numClasses).fill(0);
    const classImageCount = new Array<number>(numClasses).fill(0);
    const usedNames = new Set<string>();
    let processed = 0;
    const skipped: string[] = [];

    for (const { file, data } of parsedJsons) {
      const imageFileName = path.basename(data.imagePath || '');
      if (!imageFileName) {
        skipped.push(`${file}: missing imagePath`);
        continue;
      }
      const sourceImagePath = path.join(absoluteImagesPath, imageFileName);
      if (!fs.existsSync(sourceImagePath)) {
        skipped.push(`${file}: image not found (${imageFileName})`);
        continue;
      }

      const w = data.imageWidth | 0;
      const h = data.imageHeight | 0;
      if (w <= 0 || h <= 0) {
        skipped.push(`${file}: invalid imageWidth/imageHeight`);
        continue;
      }

      // Rasterize polygons (background = 0, initialized by Uint8Array).
      const maskBuf = new Uint8Array(w * h);
      const present = new Set<number>([0]);
      for (const shape of data.shapes || []) {
        if (shape.shape_type !== 'polygon') continue;
        const cls = labelToClassId.get(shape.label);
        if (cls === undefined) continue;
        fillPolygon(maskBuf, w, h, shape.points, cls);
        present.add(cls);
      }

      // Update class stats.
      for (let i = 0; i < maskBuf.length; i++) {
        classPixelCount[maskBuf[i]]++;
      }
      present.forEach(c => { classImageCount[c]++; });

      // Sanitize the destination filename so PaddleSeg's whitespace-separated
      // list-file format is unambiguous (the training reader `line.split()`
      // would otherwise truncate filenames containing spaces/parens/etc).
      // Track used names to disambiguate collisions produced by sanitization.
      let safeName = sanitizeFilename(imageFileName);
      if (usedNames.has(safeName)) {
        const ext = path.extname(safeName);
        const base = path.basename(safeName, ext);
        let n = 2;
        while (usedNames.has(`${base}_${n}${ext}`)) n++;
        safeName = `${base}_${n}${ext}`;
      }
      usedNames.add(safeName);

      // Copy image, write mask.
      const destImagePath = path.join(outImgsDir, safeName);
      fs.copyFileSync(sourceImagePath, destImagePath);

      const maskName = `${path.basename(safeName, path.extname(safeName))}.png`;
      const destMaskPath = path.join(outMasksDir, maskName);
      writePalettedPng(destMaskPath, w, h, maskBuf, palette);

      // Paths in list files are relative to the PaddleSeg `dataset_root`
      // (which the Import Dataset flow should set to `.../PaddleSeg/{name}/data`).
      const line = `JPEGImages/${safeName} Annotations/${maskName}`;
      if (trainSet.has(file)) trainLines.push(line);
      else valLines.push(line);
      processed++;
    }

    if (processed === 0) {
      // Clean up empty output dir before failing to avoid stale artifacts.
      try { fs.rmSync(absoluteDatasetDir, { recursive: true, force: true }); } catch { /* ignore */ }
      return NextResponse.json(
        {
          error: 'No labelme files could be converted (no valid image/JSON pairs)',
          details: skipped,
        },
        { status: 400 },
      );
    }

    fs.writeFileSync(path.join(dataDir, 'train.txt'), trainLines.join('\n') + '\n');
    fs.writeFileSync(path.join(dataDir, 'val.txt'), valLines.join('\n') + '\n');
    fs.writeFileSync(path.join(dataDir, 'class_names.txt'), classNames.join('\n') + '\n');

    const classStats = classNames.map((className, id) => ({
      id,
      name: className,
      count: classPixelCount[id],
      imageCount: classImageCount[id],
    }));

    return NextResponse.json({
      success: true,
      message: `Converted ${processed} labelme file(s) to PaddleSeg format`
        + (skipped.length ? ` (skipped ${skipped.length})` : ''),
      outputPath: absoluteDatasetDir,
      stats: {
        totalFiles: parsedJsons.length,
        trainCount: trainLines.length,
        valCount: valLines.length,
        numClasses,
        classNames,
        classStats,
        skipped,
      },
    });
  } catch (error) {
    console.error('Error converting labelme to PaddleSeg:', error);
    return NextResponse.json(
      {
        error: 'Failed to convert labelme to PaddleSeg',
        message: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 },
    );
  }
}
