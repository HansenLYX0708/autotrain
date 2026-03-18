import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { spawn } from 'child_process';
import { existsSync, readdirSync, statSync } from 'fs';
import { join, basename, extname } from 'path';

// Store running processes
const runningProcesses = new Map<string, ReturnType<typeof spawn>>();

// Quote path if it contains spaces
function quotePath(path: string): string {
  if (!path) return '';
  if (path.includes(' ')) {
    return `"${path}"`;
  }
  return path;
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

// GET /api/validation-jobs - Get all validation jobs with project relation
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const projectId = searchParams.get('projectId');
    const status = searchParams.get('status');
    const type = searchParams.get('type');

    const where: {
      projectId?: string;
      status?: string;
      type?: string;
    } = {};

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

    // Check if project exists
    const project = await db.project.findUnique({
      where: { id: body.projectId },
    });

    if (!project) {
      return NextResponse.json(
        { success: false, error: 'Project not found' },
        { status: 404 }
      );
    }

    // Get system config
    const systemConfig = await db.systemConfig.findFirst();
    const workDir = project.framework === 'PaddleClas'
      ? systemConfig?.paddleClasPath
      : systemConfig?.paddleDetectionPath;

    // Build command based on type - use customCommand if provided
    let command = body.customCommand || '';
    
    if (!command) {
      if (body.type === 'eval') {
        const configPath = quotePath(body.configPath || '');
        const weightsPath = quotePath(body.weightsPath || '');
        command = `python tools/eval.py -c ${configPath} -o weights=${weightsPath}`;
      } else if (body.type === 'infer') {
        const configPath = quotePath(body.configPath || '');
        const weightsPath = quotePath(body.weightsPath || '');
        const inputPath = body.inferInputPath || '';
        const outputPath = body.inferOutputPath ? quotePath(body.inferOutputPath) : 'output/infer_results';
        
        // Determine if input is a file or directory
        // Use --infer_img for single image files, --infer_dir for directories
        const inputParam = isImageFile(inputPath) ? '--infer_img' : '--infer_dir';
        const quotedInputPath = quotePath(inputPath);
        
        command = `python tools/infer.py -c ${configPath} -o weights=${weightsPath} ${inputParam}=${quotedInputPath} --output_dir=${outputPath}`;
      }
    }

    // Create validation job
    const validationJob = await db.validationJob.create({
      data: {
        projectId: body.projectId,
        trainingJobId: body.trainingJobId || null,
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
      startValidationProcess(
        validationJob.id,
        command,
        workDir,
        systemConfig?.pythonPath || 'python',
        systemConfig?.condaEnv || null,
        systemConfig?.condaPath || null,
        body.type || 'eval'
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
  type: string
) {
  console.log(`\n========== VALIDATION PROCESS START ==========`);
  console.log(`[Validation ${jobId}] Type: ${type}`);
  console.log(`[Validation ${jobId}] Command: ${command}`);
  console.log(`[Validation ${jobId}] Working directory: ${workDir}`);

  // Collect output
  let outputCollector: string[] = [];

  // Build environment
  const env: Record<string, string> = {
    ...process.env,
    PYTHONUNBUFFERED: '1',
  };

  // Detect conda environment
  let detectedCondaEnv: string | null = condaEnv;

  if (!detectedCondaEnv && pythonPath) {
    const patterns = [
      /[\/\\](?:anaconda3|miniconda3|anaconda|miniconda)[\/\\]envs[\/\\]([^\/\\]+)/i,
      /[\/\\]\.condax[\/\\]([^\/\\]+)/i,
      /[\/\\]envs[\/\\]([^\/\\]+)/i,
    ];

    for (const pattern of patterns) {
      const match = pythonPath.match(pattern);
      if (match) {
        detectedCondaEnv = match[1];
        break;
      }
    }
  }

  // Build final command
  let fullCommand = command;

  if (detectedCondaEnv) {
    const condaExec = condaPath || 'conda';
    const isWindows = workDir.includes('\\') || workDir.match(/^[A-Z]:\\/i);
    if (isWindows) {
      fullCommand = `"${condaExec}" run -n ${detectedCondaEnv} --no-capture-output ${fullCommand}`;
    } else {
      fullCommand = `${condaExec} run -n ${detectedCondaEnv} --no-capture-output ${fullCommand}`;
    }
  } else {
    let pythonExec = pythonPath || 'python';
    if (pythonExec.includes(' ')) {
      pythonExec = `"${pythonExec}"`;
    }
    fullCommand = fullCommand.replace(/^python\b/, pythonExec);
  }

  console.log(`[Validation ${jobId}] Final command: ${fullCommand}`);

  // Update job status to running
  db.validationJob.update({
    where: { id: jobId },
    data: { status: 'running', startedAt: new Date() },
  }).catch(console.error);

  const childProcess = spawn(fullCommand, [], {
    cwd: workDir,
    shell: true,
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

    if (type === 'eval' && status === 'completed') {
      // Parse sample count from eval output
      const samplesMatch = fullOutput.match(/Load\s*\[(\d+)\s+samples\s+valid/);
      const samplesCount = samplesMatch ? parseInt(samplesMatch[1], 10) : null;

      // Parse mAP results from COCO-style eval output
      // Format: Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.006
      const parseMetric = (pattern: string): number | null => {
        const match = fullOutput.match(new RegExp(pattern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\s*=\\s*([\\d.]+)', 'i'));
        return match ? parseFloat(match[1]) : null;
      };

      // mAP (AP) metrics
      const mapAll = parseMetric('Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]');
      const map50 = parseMetric('Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]');
      const map75 = parseMetric('Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]');
      const mapSmall = parseMetric('Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]');
      const mapMedium = parseMetric('Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]');
      const mapLarge = parseMetric('Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]');

      // AR metrics
      const ar1 = parseMetric('Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]');
      const ar10 = parseMetric('Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]');
      const ar100 = parseMetric('Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]');
      const arSmall = parseMetric('Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]');
      const arMedium = parseMetric('Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]');
      const arLarge = parseMetric('Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]');

      // Store all metrics
      resultJson = JSON.stringify({
        samplesCount,
        // mAP metrics
        mAP: mapAll,
        mAP50: map50,
        mAP75: map75,
        mAP_small: mapSmall,
        mAP_medium: mapMedium,
        mAP_large: mapLarge,
        // AR metrics
        AR_1: ar1,
        AR_10: ar10,
        AR_100: ar100,
        AR_small: arSmall,
        AR_medium: arMedium,
        AR_large: arLarge,
      });
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
