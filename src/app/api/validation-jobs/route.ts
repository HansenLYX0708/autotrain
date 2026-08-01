import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { requireAuth, buildUserFilter } from '@/lib/auth';
import { getWorkDir } from '@/lib/frameworks';
import { spawn, exec } from 'child_process';
import { existsSync, readdirSync, statSync, mkdirSync } from 'fs';
import { join, basename, extname, dirname } from 'path';
import { promisify } from 'util';

const execFileAsync = promisify(exec);

/**
 * PaddleSeg's `paddleseg/utils/utils.py:get_image_list()` hard-codes its
 * accepted image extensions to JPEG/JPG/BMP/PNG only. When the user hands
 * `predict.py --image_path` a `.tif` file it falls through to the "treat as
 * list file" branch and `open(path, 'r')` blows up on binary bytes
 * (`UnicodeDecodeError: 'gbk' codec can't decode ...` on Windows). Same for
 * directories — TIFFs are silently skipped.
 *
 * Fix: before spawning predict.py, transparently mirror any TIFF input into a
 * staging folder with `.png` copies (via Pillow in the user's Python env) and
 * feed the staging path to PaddleSeg instead. Non-TIFF inputs are returned
 * unchanged so we don't add latency on happy-path cases.
 *
 * Returns the effective inferInputPath to hand to predict.py.
 */
async function stagePaddleSegTiffInput(
  inferInputPath: string,
  pythonPath: string,
  inferOutputPath: string | undefined,
): Promise<string> {
  if (!inferInputPath || !existsSync(inferInputPath)) return inferInputPath;

  const isTiff = (p: string) => /\.tiff?$/i.test(p);
  const stat = statSync(inferInputPath);

  // Detect whether staging is needed at all
  let needsStaging = false;
  if (stat.isFile()) {
    needsStaging = isTiff(inferInputPath);
  } else if (stat.isDirectory()) {
    needsStaging = readdirSync(inferInputPath).some((f) => isTiff(f));
  }
  if (!needsStaging) return inferInputPath;

  // Staging dir: next to the output dir when we have one (so cleanup is easy
  // and everything for a run lives together); otherwise fall back to a folder
  // beside the input.
  const stagingRoot = inferOutputPath
    ? join(inferOutputPath, '_input_staging')
    : join(stat.isDirectory() ? inferInputPath : dirname(inferInputPath), '_input_staging');
  mkdirSync(stagingRoot, { recursive: true });

  // Python payload does the actual conversion. Written to a temp .py file
  // rather than passed via `-c` to keep quoting sane on Windows.
  const scriptPath = join(stagingRoot, '_convert.py');
  const script = `import os, sys, shutil
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # allow large TEM images
src = sys.argv[1]
dst_dir = sys.argv[2]
os.makedirs(dst_dir, exist_ok=True)
def convert_one(path, out_dir):
    stem, ext = os.path.splitext(os.path.basename(path))
    out = os.path.join(out_dir, stem + '.png')
    Image.open(path).save(out)
    print(out)
if os.path.isfile(src):
    convert_one(src, dst_dir)
else:
    for name in sorted(os.listdir(src)):
        full = os.path.join(src, name)
        if not os.path.isfile(full):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in ('.tif', '.tiff'):
            convert_one(full, dst_dir)
        elif ext in ('.jpg', '.jpeg', '.png', '.bmp'):
            # PaddleSeg already accepts these; just place them alongside the
            # converted TIFFs so predict.py sees the whole set in one folder.
            shutil.copy2(full, os.path.join(dst_dir, name))
`;
  const fs = await import('fs/promises');
  await fs.writeFile(scriptPath, script, 'utf8');

  const cmd = `"${pythonPath}" "${scriptPath}" "${inferInputPath}" "${stagingRoot}"`;
  console.log(`[TIFF staging] ${cmd}`);
  const { stdout, stderr } = await execFileAsync(cmd, { timeout: 5 * 60 * 1000 });
  if (stderr) console.log(`[TIFF staging] stderr: ${stderr}`);
  if (stdout) console.log(`[TIFF staging] converted:\n${stdout}`);

  // For a single-file input we point predict.py at the produced PNG directly.
  if (stat.isFile()) {
    const stem = basename(inferInputPath, extname(inferInputPath));
    return join(stagingRoot, stem + '.png');
  }
  return stagingRoot;
}

// Store running processes
const runningProcesses = new Map<string, ReturnType<typeof spawn>>();

// Helper function to get Python path for a job based on GPU configuration
async function getPythonPathForJob(trainingJobId: string): Promise<string> {
  let pythonPath = 'python';
  
  if (!trainingJobId) return pythonPath;
  
  try {
    // Get training job to find GPU info
    const job = await db.trainingJob.findUnique({
      where: { id: trainingJobId },
      select: { trainingParams: true },
    });
    
    if (!job?.trainingParams) return pythonPath;
    
    // Parse training params for GPU info
    let trainingParams: Record<string, unknown> = {};
    try {
      trainingParams = JSON.parse(job.trainingParams as string);
    } catch {
      return pythonPath;
    }
    
    const gpuIdsStr = (trainingParams.gpuIds as string) || '0';
    const gpuIds = gpuIdsStr.split(',').map(id => parseInt(id.trim(), 10)).filter(id => !isNaN(id));
    const primaryGpuId = gpuIds[0] || 0;
    
    // Get system config for GPU Python mappings
    const systemConfig = await db.systemConfig.findFirst();
    if (systemConfig?.gpuPythonMappings) {
      try {
        const gpuMappings = JSON.parse(systemConfig.gpuPythonMappings) as Record<string, { pythonPath: string }>;
        const mapping = gpuMappings[primaryGpuId.toString()];
        if (mapping?.pythonPath) {
          pythonPath = mapping.pythonPath;
        }
      } catch (e) {
        console.error('Failed to parse GPU Python mappings:', e);
      }
    }
  } catch (error) {
    console.error('Error getting Python path for job:', error);
  }
  
  return pythonPath;
}

// Check if path is likely a file (has image extension)
function isImageFile(path: string): boolean {
  const imageExtensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'];
  const ext = extname(path).toLowerCase();
  return imageExtensions.includes(ext);
}

// Find inference result images in output directory
function findInferenceImages(outputDir: string): string[] {
  const images: string[] = [];
  
  try {
    if (!existsSync(outputDir)) {
      console.log(`Output directory does not exist: ${outputDir}`);
      return images;
    }
    
    const files = readdirSync(outputDir);
    const imageExtensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'];
    
    for (const file of files) {
      const filePath = join(outputDir, file);
      const stat = statSync(filePath);
      
      if (stat.isFile()) {
        const ext = extname(file).toLowerCase();
        if (imageExtensions.includes(ext)) {
          images.push(filePath);
        }
      }
    }
    
    // Sort by modification time (newest first)
    images.sort((a, b) => {
      const statA = statSync(a);
      const statB = statSync(b);
      return statB.mtimeMs - statA.mtimeMs;
    });
    
  } catch (error) {
    console.error('Error finding inference images:', error);
  }
  
  return images;
}

// GET /api/validation-jobs - Get all validation jobs with project relation (filtered by user for non-admins)
export async function GET(request: NextRequest) {
  try {
    // Check authentication
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;

    const { userId, role } = auth;

    const { searchParams } = new URL(request.url);
    const projectId = searchParams.get('projectId');
    const status = searchParams.get('status');
    const type = searchParams.get('type');

    // Build where clause with user filter
    const userFilter = buildUserFilter(userId, role, 'userId');
    const where: Record<string, unknown> = { ...userFilter };

    if (projectId) {
      where.projectId = projectId;
    }
    if (status) {
      where.status = status;
    }
    if (type) {
      where.type = type;
    }

    const validationJobs = await db.validationJob.findMany({
      where,
      include: {
        project: {
          select: {
            id: true,
            name: true,
            framework: true,
            status: true,
          },
        },
      },
      orderBy: {
        createdAt: 'desc',
      },
    });

    return NextResponse.json({
      success: true,
      data: validationJobs,
    });
  } catch (error) {
    console.error('Error fetching validation jobs:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to fetch validation jobs' },
      { status: 500 }
    );
  }
}

// POST /api/validation-jobs - Create and optionally run a validation job
export async function POST(request: NextRequest) {
  try {
    // Check authentication
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;

    const { userId, role } = auth;
    
    const body = await request.json();

    // Validate required fields
    if (!body.projectId) {
      return NextResponse.json(
        { success: false, error: 'projectId is required' },
        { status: 400 }
      );
    }

    if (!body.name) {
      return NextResponse.json(
        { success: false, error: 'name is required' },
        { status: 400 }
      );
    }

    // Validate type
    const validTypes = ['eval', 'infer'];
    if (body.type && !validTypes.includes(body.type)) {
      return NextResponse.json(
        { success: false, error: `Invalid type. Must be one of: ${validTypes.join(', ')}` },
        { status: 400 }
      );
    }

    // Check if project exists and user has access
    // Admin can access any project, regular user can only access their own
    const project = role === 'admin'
      ? await db.project.findUnique({ where: { id: body.projectId } })
      : await db.project.findFirst({
          where: { 
            id: body.projectId,
            userId: userId,
          },
        });

    if (!project) {
      return NextResponse.json(
        { success: false, error: 'Project not found or access denied' },
        { status: 404 }
      );
    }

    // Get system config
    const systemConfig = await db.systemConfig.findFirst();
    const framework = project.framework || 'PaddleDetection';
    const workDir = getWorkDir(framework, systemConfig);

    // Get Python path based on training job's GPU configuration
    const pythonPath = body.trainingJobId 
      ? await getPythonPathForJob(body.trainingJobId)
      : 'python';

    // Build command based on type - use customCommand if provided
    let command = body.customCommand || '';
    
    if (!command) {
      const configPath = body.configPath || '';
      const weightsPath = body.weightsPath || '';
      if (framework === 'PaddleSeg') {
        // PaddleSeg uses val.py (eval) and predict.py (infer) with --config/--model_path
        if (body.type === 'eval') {
          command = `${pythonPath} tools/val.py --config ${configPath} --model_path ${weightsPath}`;
        } else if (body.type === 'infer') {
          const inputPath = body.inferInputPath || '';
          const outputPath = body.inferOutputPath || 'output/predict_results';
          command = `${pythonPath} tools/predict.py --config ${configPath} --model_path ${weightsPath} --image_path ${inputPath} --save_dir ${outputPath}`;
        }
      } else if (body.type === 'eval') {
        command = `${pythonPath} tools/eval.py -c ${configPath} -o weights=${weightsPath}`;
      } else if (body.type === 'infer') {
        const inputPath = body.inferInputPath || '';
        const outputPath = body.inferOutputPath || 'output/infer_results';
        
        // Determine if input is a file or directory
        const inputParam = isImageFile(inputPath) ? '--infer_img' : '--infer_dir';
        
        command = `${pythonPath} tools/infer.py -c ${configPath} -o weights=${weightsPath} ${inputParam}=${inputPath} --output_dir=${outputPath}`;
      }
    }

    // Create validation job with userId
    const validationJob = await db.validationJob.create({
      data: {
        projectId: body.projectId,
        trainingJobId: body.trainingJobId || null,
        userId: userId,
        name: body.name,
        type: body.type || 'eval',
        configPath: body.configPath || null,
        weightsPath: body.weightsPath || null,
        datasetPath: body.datasetPath || null,
        inferInputPath: body.inferInputPath || null,
        inferOutputPath: body.inferOutputPath || null,
        status: 'pending',
        command: command,
      },
      include: {
        project: {
          select: {
            id: true,
            name: true,
            framework: true,
            status: true,
          },
        },
      },
    });

    // If runImmediately is true, start the validation
    if (body.runImmediately && workDir && command) {
      // PaddleSeg + infer + TIFF input: transparently convert to PNGs into a
      // staging folder so predict.py's `get_image_list()` accepts them. The
      // conversion is a no-op when no TIFFs are present.
      let effectiveInferInput = body.inferInputPath;
      if (framework === 'PaddleSeg' && body.type === 'infer' && body.inferInputPath) {
        try {
          effectiveInferInput = await stagePaddleSegTiffInput(
            body.inferInputPath,
            pythonPath,
            body.inferOutputPath,
          );
          if (effectiveInferInput !== body.inferInputPath) {
            console.log(`[Validation ${validationJob.id}] TIFF staging: "${body.inferInputPath}" -> "${effectiveInferInput}"`);
          }
        } catch (err) {
          console.error(`[Validation ${validationJob.id}] TIFF staging failed:`, err);
          const msg = err instanceof Error ? err.message : String(err);
          await db.validationJob.update({
            where: { id: validationJob.id },
            data: {
              status: 'failed',
              // ValidationJob has no dedicated errorMessage column; surface
              // the failure through outputLog which the UI already renders.
              outputLog: `TIFF pre-conversion failed: ${msg}`,
              completedAt: new Date(),
            },
          });
          return NextResponse.json(
            { success: false, error: `TIFF pre-conversion failed: ${msg}` },
            { status: 500 }
          );
        }
      }

      startValidationProcess(
        validationJob.id,
        command,
        workDir,
        pythonPath,
        systemConfig?.condaEnv || null,
        systemConfig?.condaPath || null,
        body.type || 'eval',
        body.configPath,
        body.weightsPath,
        effectiveInferInput,
        body.inferOutputPath,
        framework
      );
    }

    return NextResponse.json(
      { success: true, data: validationJob },
      { status: 201 }
    );
  } catch (error) {
    console.error('Error creating validation job:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to create validation job' },
      { status: 500 }
    );
  }
}

// Start validation process
function startValidationProcess(
  jobId: string,
  command: string,
  workDir: string,
  pythonPath: string,
  condaEnv: string | null,
  condaPath: string | null,
  type: string,
  configPath?: string,
  weightsPath?: string,
  inferInputPath?: string,
  inferOutputPath?: string,
  framework: string = 'PaddleDetection'
) {
  console.log(`\n========== VALIDATION PROCESS START ==========`);
  console.log(`[Validation ${jobId}] Type: ${type}`);
  console.log(`[Validation ${jobId}] Command: ${command}`);
  console.log(`[Validation ${jobId}] Working directory: ${workDir}`);
  console.log(`[Validation ${jobId}] Python path: ${pythonPath}`);

  // Collect output
  let outputCollector: string[] = [];

  // Build environment
  const env: NodeJS.ProcessEnv = {
    ...process.env,
    PYTHONUNBUFFERED: '1',
    PYTHONPATH: workDir,
  };

  // Build args array properly to handle spaces in paths
  // Use the provided pythonPath directly (should be absolute path to python.exe)
  let args: string[] = [];

  if (framework === 'PaddleSeg') {
    if (type === 'eval') {
      // eval: python tools/val.py --config configPath --model_path weightsPath
      args = [
        'tools/val.py',
        '--config', configPath || '',
        '--model_path', weightsPath || '',
      ];
    } else if (type === 'infer') {
      // infer: python tools/predict.py --config configPath --model_path weightsPath --image_path input --save_dir output
      args = [
        'tools/predict.py',
        '--config', configPath || '',
        '--model_path', weightsPath || '',
        '--image_path', inferInputPath || '',
        '--save_dir', inferOutputPath || 'output/predict_results',
      ];
    }
  } else if (type === 'eval') {
    // eval: python tools/eval.py -c configPath -o weights=weightsPath
    args = [
      'tools/eval.py',
      '-c', configPath || '',
      '-o', `weights=${weightsPath || ''}`,
    ];
  } else if (type === 'infer') {
    // infer: python tools/infer.py -c configPath -o weights=weightsPath --infer_img/inputPath --output_dir=outputPath
    const inputParam = inferInputPath && isImageFile(inferInputPath) ? '--infer_img' : '--infer_dir';
    args = [
      'tools/infer.py',
      '-c', configPath || '',
      '-o', `weights=${weightsPath || ''}`,
      `${inputParam}=${inferInputPath || ''}`,
      `--output_dir=${inferOutputPath || 'output/infer_results'}`,
    ];
  }

  console.log(`[Validation ${jobId}] Python: ${pythonPath}`);
  console.log(`[Validation ${jobId}] Args: ${JSON.stringify(args)}`);

  // Update job status to running
  db.validationJob.update({
    where: { id: jobId },
    data: { status: 'running', startedAt: new Date() },
  }).catch(console.error);

  // Use spawn with args array (Windows compatible)
  const childProcess = spawn(pythonPath, args, {
    cwd: workDir,
    env,
  });

  runningProcesses.set(jobId, childProcess);

  childProcess.stdout?.on('data', (data: Buffer) => {
    const output = data.toString();
    console.log(`[Validation ${jobId}] ${output}`);
    outputCollector.push(output);
    if (outputCollector.length > 500) {
      outputCollector = outputCollector.slice(-500);
    }
  });

  childProcess.stderr?.on('data', (data: Buffer) => {
    const output = data.toString();
    console.error(`[Validation ${jobId} ERROR] ${output}`);
    outputCollector.push(output);
    if (outputCollector.length > 500) {
      outputCollector = outputCollector.slice(-500);
    }
  });

  childProcess.on('close', async (code) => {
    runningProcesses.delete(jobId);

    const fullOutput = outputCollector.join('\n');
    const status = code === 0 ? 'completed' : 'failed';

    // Parse results based on type
    let resultJson: string | null = null;
    let resultPath: string | null = null;

    if (type === 'eval' && status === 'completed' && framework === 'PaddleSeg') {
      // PaddleSeg val.py prints: [EVAL] #Images: N mIoU: .. Acc: .. Kappa: .. Dice: ..
      const segF = (re: RegExp): number | null => {
        const m = fullOutput.match(re);
        return m ? parseFloat(m[1]) : null;
      };
      resultJson = JSON.stringify({
        mIoU: segF(/mIoU:\s*([\d.]+)/i),
        acc: segF(/Acc:\s*([\d.]+)/i),
        kappa: segF(/Kappa:\s*([\d.]+)/i),
        dice: segF(/Dice:\s*([\d.]+)/i),
      });
    } else if (type === 'eval' && status === 'completed') {
      // Parse sample count from eval output
      const samplesMatch = fullOutput.match(/Load\s*\[(\d+)\s+samples\s+valid/);
      const samplesCount = samplesMatch ? parseInt(samplesMatch[1], 10) : null;

      // PaddleDetection prints COCO eval blocks delimited by
      //   "Evaluate annotation type *bbox*"
      //   "Evaluate annotation type *segm*"
      // For instance-segmentation models both blocks are present, so we
      // parse them separately. For detection-only output there is only one
      // block (or no marker at all) and we treat the whole output as bbox.
      const extractBlock = (marker: RegExp): string | null => {
        const startMatch = fullOutput.match(marker);
        if (!startMatch || startMatch.index === undefined) return null;
        const start = startMatch.index + startMatch[0].length;
        // Stop at the next "Evaluate annotation type" marker, if any
        const rest = fullOutput.slice(start);
        const nextMatch = rest.match(/Evaluate annotation type/i);
        return nextMatch && nextMatch.index !== undefined
          ? rest.slice(0, nextMatch.index)
          : rest;
      };

      const bboxBlock = extractBlock(/Evaluate annotation type \*bbox\*/i) ?? fullOutput;
      const segmBlock = extractBlock(/Evaluate annotation type \*segm\*/i);

      const parseMetricFrom = (block: string, pattern: string): number | null => {
        const match = block.match(
          new RegExp(pattern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\s*=\\s*([\\d.]+)', 'i')
        );
        return match ? parseFloat(match[1]) : null;
      };

      const parseCocoBlock = (block: string) => ({
        mAP: parseMetricFrom(block, 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]'),
        mAP50: parseMetricFrom(block, 'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]'),
        mAP75: parseMetricFrom(block, 'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]'),
        mAP_small: parseMetricFrom(block, 'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]'),
        mAP_medium: parseMetricFrom(block, 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]'),
        mAP_large: parseMetricFrom(block, 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=large | maxDets=100 ]'),
        AR_1: parseMetricFrom(block, 'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]'),
        AR_10: parseMetricFrom(block, 'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]'),
        AR_100: parseMetricFrom(block, 'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]'),
        AR_small: parseMetricFrom(block, 'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]'),
        AR_medium: parseMetricFrom(block, 'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]'),
        AR_large: parseMetricFrom(block, 'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]'),
      });

      const bboxMetrics = parseCocoBlock(bboxBlock);
      const segmMetrics = segmBlock ? parseCocoBlock(segmBlock) : null;

      // Store all metrics. Bbox metrics keep their original (unsuffixed)
      // names for backward compatibility with the existing UI; segm metrics
      // are emitted with a "_segm" suffix and only when present.
      const resultPayload: Record<string, unknown> = {
        samplesCount,
        // bbox metrics (detection or mask-model bbox head)
        ...bboxMetrics,
      };

      if (segmMetrics) {
        resultPayload.hasSegm = true;
        resultPayload.mAP_segm = segmMetrics.mAP;
        resultPayload.mAP50_segm = segmMetrics.mAP50;
        resultPayload.mAP75_segm = segmMetrics.mAP75;
        resultPayload.mAP_small_segm = segmMetrics.mAP_small;
        resultPayload.mAP_medium_segm = segmMetrics.mAP_medium;
        resultPayload.mAP_large_segm = segmMetrics.mAP_large;
        resultPayload.AR_1_segm = segmMetrics.AR_1;
        resultPayload.AR_10_segm = segmMetrics.AR_10;
        resultPayload.AR_100_segm = segmMetrics.AR_100;
        resultPayload.AR_small_segm = segmMetrics.AR_small;
        resultPayload.AR_medium_segm = segmMetrics.AR_medium;
        resultPayload.AR_large_segm = segmMetrics.AR_large;
      }

      resultJson = JSON.stringify(resultPayload);
    } else if (type === 'infer' && status === 'completed') {
      // Get the validation job to find output path
      try {
        const vJob = await db.validationJob.findUnique({ where: { id: jobId } });
        const outputDir = vJob?.inferOutputPath || 'output/infer_results';
        
        // Find inference result images
        const inferImages = findInferenceImages(outputDir);
        
        // Parse detection results from log if available
        const boxMatch = fullOutput.match(/(\d+)\s*(?:bbox|bounding box|detections?)/gi);
        
        if (inferImages.length > 0) {
          resultPath = outputDir;
          resultJson = JSON.stringify({
            outputDir,
            images: inferImages,
            imageCount: inferImages.length,
            detectionSummary: boxMatch ? boxMatch[0] : null
          });
        } else {
          // Try to extract from log
          const outputMatch = fullOutput.match(/(?:output|save)[_-]?(?:dir|to)?[:\s]+([^\n]+)/i);
          if (outputMatch) {
            resultPath = outputMatch[1].trim();
            const fallbackImages = findInferenceImages(resultPath);
            if (fallbackImages.length > 0) {
              resultJson = JSON.stringify({
                outputDir: resultPath,
                images: fallbackImages,
                imageCount: fallbackImages.length
              });
            }
          } else {
            resultPath = outputDir;
          }
        }
      } catch (e) {
        console.error('Error processing inference results:', e);
        resultPath = 'output/infer_results';
      }
    }

    try {
      await db.validationJob.update({
        where: { id: jobId },
        data: {
          status,
          completedAt: new Date(),
          outputLog: fullOutput.slice(-10000),
          resultJson,
          resultPath,
        },
      });
    } catch (error) {
      console.error('Failed to update validation job status:', error);
    }
  });

  childProcess.on('error', async (error) => {
    console.error(`[Validation ${jobId} PROCESS ERROR]`, error);
    runningProcesses.delete(jobId);

    try {
      await db.validationJob.update({
        where: { id: jobId },
        data: {
          status: 'failed',
          completedAt: new Date(),
          outputLog: error.message,
        },
      });
    } catch (dbError) {
      console.error('Failed to update validation job with error:', dbError);
    }
  });
}
