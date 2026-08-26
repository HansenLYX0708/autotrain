import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { requireAuth, buildUserFilter } from '@/lib/auth';
import { defaultInferOutputDir, frameworkMeta, getWorkDir, isAnomaly, isSegmentation, resolvePythonPath } from '@/lib/frameworks';
import { buildEvalCommand, buildInferCommand } from '@/lib/job-commands';
import { spawn, exec } from 'child_process';
import { existsSync, readdirSync, readFileSync, statSync, mkdirSync } from 'fs';
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

/**
 * Python interpreter for a validation run, resolved framework-first.
 *
 * Two bugs this replaces: the previous implementation read
 * `gpuPythonMappings[id].pythonPath`, but the stored shape is
 * `{"0": "C:/.../python.exe"}` — a plain string — so it *always* fell back to a
 * bare `python`; and it had no notion of framework, so a torch job would have
 * been handed a PaddlePaddle interpreter.
 */
async function getPythonPathForJob(trainingJobId: string, framework?: string | null): Promise<string> {
  if (!trainingJobId) {
    const systemConfig = await db.systemConfig.findFirst();
    return resolvePythonPath(framework, 0, systemConfig).pythonPath || 'python';
  }

  try {
    const job = await db.trainingJob.findUnique({
      where: { id: trainingJobId },
      select: { trainingParams: true, project: { select: { framework: true } } },
    });

    let trainingParams: Record<string, unknown> = {};
    try {
      trainingParams = job?.trainingParams ? JSON.parse(job.trainingParams as string) : {};
    } catch {
      // Fall through to GPU 0.
    }

    const gpuIdsStr = (trainingParams.gpuIds as string) || '0';
    const gpuIds = gpuIdsStr.split(',').map(id => parseInt(id.trim(), 10)).filter(id => !isNaN(id));
    const primaryGpuId = gpuIds[0] || 0;

    const systemConfig = await db.systemConfig.findFirst();
    const effectiveFramework = framework || job?.project?.framework;
    const { pythonPath } = resolvePythonPath(effectiveFramework, primaryGpuId, systemConfig);
    return pythonPath || 'python';
  } catch (error) {
    console.error('Error getting Python path for job:', error);
    return 'python';
  }
}

// Check if path is likely a file (has image extension)
function isImageFile(path: string): boolean {
  const imageExtensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'];
  const ext = extname(path).toLowerCase();
  return imageExtensions.includes(ext);
}

/**
 * Find inference result images under an output directory.
 *
 * Recurses one level, which is required for segmentation: both PaddleSeg's
 * `predict.py` and torchtrain's write into `added_prediction/` and
 * `pseudo_color_prediction/` sub-folders rather than the root. A flat scan (what
 * this used to do) therefore found nothing and the validation page reported a
 * successful run with zero images. Detection writes to the root, which is still
 * covered.
 *
 * `_input_staging` is skipped: those are the PNG copies of the *input* TIFFs
 * created for PaddleSeg, not results.
 */
/**
 * Read `scores.json` written by the anomaly predictor.
 *
 * Returns null (rather than throwing) when the file is absent or malformed: a
 * missing score file must not turn a successful inference run into a failed
 * validation job, since the heatmaps on disk are still useful.
 */
function readAnomalyScores(outputDir: string): Record<string, unknown> | null {
  try {
    const scoresPath = join(outputDir, 'scores.json');
    if (!existsSync(scoresPath)) return null;
    const parsed = JSON.parse(readFileSync(scoresPath, 'utf-8'));
    if (!parsed || typeof parsed !== 'object') return null;
    const images = Array.isArray(parsed.images) ? parsed.images : [];
    return {
      threshold: typeof parsed.threshold === 'number' ? parsed.threshold : null,
      count: typeof parsed.count === 'number' ? parsed.count : images.length,
      anomalous: typeof parsed.anomalous === 'number' ? parsed.anomalous : null,
      model: typeof parsed.model === 'string' ? parsed.model : null,
      // Cap the payload: a folder of 10k images would otherwise bloat every
      // read of this row, and the UI only ever renders a page of them.
      images: images.slice(0, 500),
      truncated: images.length > 500,
    };
  } catch (error) {
    console.error(`Failed to read anomaly scores from ${outputDir}:`, error);
    return null;
  }
}

function findInferenceImages(outputDir: string, depth = 1): string[] {
  const images: string[] = [];
  const imageExtensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'];

  try {
    if (!existsSync(outputDir)) {
      console.log(`Output directory does not exist: ${outputDir}`);
      return images;
    }

    for (const file of readdirSync(outputDir)) {
      const filePath = join(outputDir, file);
      const stat = statSync(filePath);

      if (stat.isFile()) {
        if (imageExtensions.includes(extname(file).toLowerCase())) images.push(filePath);
      } else if (stat.isDirectory() && depth > 0 && file !== '_input_staging') {
        images.push(...findInferenceImages(filePath, depth - 1));
      }
    }

    // Sort by modification time (newest first)
    images.sort((a, b) => statSync(b).mtimeMs - statSync(a).mtimeMs);
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

    // Get Python path based on the training job's GPU + this project's framework
    const pythonPath = await getPythonPathForJob(body.trainingJobId, framework);

    // Build command based on type - use customCommand if provided.
    // The strings come from `@/lib/job-commands`, the same module the UI preview
    // uses, so the command a user reviews is the command that runs.
    let command = body.customCommand || '';

    if (!command) {
      const configPath = body.configPath || '';
      const weightsPath = body.weightsPath || '';
      if (body.type === 'eval') {
        command = buildEvalCommand({ framework, configPath, weightsPath, python: pythonPath });
      } else if (body.type === 'infer') {
        const inputPath = body.inferInputPath || '';
        const outputPath = body.inferOutputPath || defaultInferOutputDir(framework);
        command = buildInferCommand({
          framework,
          configPath,
          weightsPath,
          inputPath,
          outputPath,
          inputIsFile: isImageFile(inputPath),
          python: pythonPath,
        });
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
      //
      // Not needed for TorchSeg: `torchtrain`'s predictor reads TIFFs directly
      // (see `read_image` in torchtrain/torchtrain/seg/dataset.py), so staging
      // would only add latency and a folder of duplicate PNGs.
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

  // Build the argv array from the framework's declared CLI dialect. Using an
  // argv array (rather than a shell string) is what makes paths with spaces work
  // without quoting, so this mirrors `@/lib/job-commands` rather than reusing it.
  const meta = frameworkMeta(framework);
  const defaultOutput = defaultInferOutputDir(framework);

  if (meta.cliStyle === 'config-flags') {
    if (type === 'eval') {
      args = [meta.scripts.eval, '--config', configPath || '', '--model_path', weightsPath || ''];
    } else if (type === 'infer') {
      args = [
        meta.scripts.infer,
        '--config', configPath || '',
        '--model_path', weightsPath || '',
        '--image_path', inferInputPath || '',
        '--save_dir', inferOutputPath || defaultOutput,
      ];
    }
  } else if (type === 'eval') {
    args = [meta.scripts.eval, '-c', configPath || '', '-o', `weights=${weightsPath || ''}`];
  } else if (type === 'infer') {
    // PaddleDetection distinguishes a single image from a directory by flag name.
    const inputParam = inferInputPath && isImageFile(inferInputPath) ? '--infer_img' : '--infer_dir';
    args = [
      meta.scripts.infer,
      '-c', configPath || '',
      '-o', `weights=${weightsPath || ''}`,
      `${inputParam}=${inferInputPath || ''}`,
      `--output_dir=${inferOutputPath || defaultOutput}`,
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

    if (type === 'eval' && status === 'completed' && isAnomaly(framework)) {
      // Anomaly eval prints a single line, e.g.
      //   [EVAL] #Images: 40 image_auroc: 0.9812 image_f1: 0.9231 pixel_auroc: ... threshold: 12.3456
      // Read every `key: value` pair rather than a fixed list: anomalib's
      // Evaluator can be given any metric, and a hard-coded set would quietly
      // drop the new one. See torchtrain/torchtrain/ad/logger.py.
      const evalLine = fullOutput.match(/\[EVAL\][^\n]*#Images:[^\n]*/i);
      const metrics: Record<string, number> = {};
      if (evalLine) {
        // The `#` of `#Images:` is captured so the sample count is skipped
        // instead of being recorded as a metric named `images`.
        const re = /(#?)([A-Za-z][A-Za-z0-9_]*)\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)/g;
        let m: RegExpExecArray | null;
        while ((m = re.exec(evalLine[0])) !== null) {
          if (m[1] === '#') continue;
          const value = parseFloat(m[3]);
          if (Number.isFinite(value)) metrics[m[2].toLowerCase()] = value;
        }
      }
      resultJson = JSON.stringify({
        taskKind: 'anomaly',
        samplesCount: (evalLine && parseInt((evalLine[0].match(/#Images:\s*(\d+)/i) ?? [])[1] ?? '', 10)) || null,
        imageAuroc: metrics.image_auroc ?? null,
        imageF1: metrics.image_f1 ?? metrics.image_f1score ?? null,
        pixelAuroc: metrics.pixel_auroc ?? null,
        pixelF1: metrics.pixel_f1 ?? metrics.pixel_f1score ?? null,
        threshold: metrics.threshold ?? null,
        metrics,
      });
    } else if (type === 'eval' && status === 'completed' && isSegmentation(framework)) {
      // Segmentation eval prints: [EVAL] #Images: N mIoU: .. Acc: .. Kappa: .. Dice: ..
      // `torchtrain`'s val.py reproduces this line exactly (see
      // torchtrain/torchtrain/logger.py), so TorchSeg parses identically.
      const segF = (re: RegExp): number | null => {
        const m = fullOutput.match(re);
        return m ? parseFloat(m[1]) : null;
      };
      const classArrays = fullOutput.match(/\[EVAL\]\s*Class IoU:\s*\r?\n\s*\[([^\]]*)\]/i);
      resultJson = JSON.stringify({
        mIoU: segF(/mIoU:\s*([\d.]+)/i),
        acc: segF(/Acc:\s*([\d.]+)/i),
        kappa: segF(/Kappa:\s*([\d.]+)/i),
        dice: segF(/Dice:\s*([\d.]+)/i),
        samplesCount: segF(/#Images:\s*(\d+)/i),
        classIoU: classArrays
          ? classArrays[1].trim().split(/\s+/).map(Number).filter((n) => !Number.isNaN(n))
          : undefined,
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
        // pycocotools right-aligns the area label in 6 chars (`{:>6s}`), so
        // "large" is printed as " large" — the same as the AR rows below. The
        // previous `area=large` spelling never matched, so mAP_large was always
        // null for PaddleDetection too.
        mAP_large: parseMetricFrom(block, 'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]'),
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
        
        // Anomaly inference also writes per-image scores, which no image can
        // convey: the score, the threshold it was compared against, and the
        // verdict. `tools/predict.py` writes it next to the heatmaps.
        const anomalyScores = isAnomaly(framework) ? readAnomalyScores(outputDir) : null;

        if (inferImages.length > 0 || anomalyScores) {
          resultPath = outputDir;
          resultJson = JSON.stringify({
            outputDir,
            images: inferImages,
            imageCount: inferImages.length,
            detectionSummary: boxMatch ? boxMatch[0] : null,
            ...(anomalyScores ? { taskKind: 'anomaly', anomaly: anomalyScores } : {}),
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
