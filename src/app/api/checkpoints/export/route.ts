import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { spawn, exec } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import { promisify } from 'util';
import { requireAuth } from '@/lib/auth';
import { logActivity } from '@/lib/activity-log';
import { frameworkMeta, getWorkDir, resolvePythonPath } from '@/lib/frameworks';

const execAsync = promisify(exec);

// POST /api/checkpoints/export - Export checkpoint to TensorRT format
export async function POST(request: NextRequest) {
  try {
    // Check authentication
    const authResult = await requireAuth(request);
    if (authResult instanceof NextResponse) {
      return authResult;
    }
    const { userId } = authResult;

    // Get user info
    const user = await db.user.findUnique({
      where: { id: userId },
      select: { username: true },
    });
    if (!user) {
      return NextResponse.json({ error: 'User not found' }, { status: 404 });
    }

    const body = await request.json();
    const { jobId, checkpointPath, checkpointName, configPath: providedConfigPath, outputDir: providedOutputDir } = body;

    if (!jobId || !checkpointPath) {
      return NextResponse.json(
        { error: 'Job ID and checkpoint path are required' },
        { status: 400 }
      );
    }

    // Get job details
    const job = await db.trainingJob.findUnique({
      where: { id: jobId },
      include: {
        project: { select: { id: true, name: true, framework: true } },
        config: true,
      },
    });

    if (!job) {
      return NextResponse.json({ error: 'Job not found' }, { status: 404 });
    }

    // Get system config. This route used to hardcode PaddleDetection's repo and
    // `tools/export_model.py`, so exporting a PaddleSeg or torch checkpoint ran
    // the wrong script in the wrong repository.
    const systemConfig = await db.systemConfig.findFirst();
    const framework = job.project?.framework || 'PaddleDetection';
    const meta = frameworkMeta(framework);
    const workDir = getWorkDir(framework, systemConfig);

    if (!meta.scripts.export) {
      return NextResponse.json(
        { error: `${framework} has no export entrypoint in this platform.` },
        { status: 400 }
      );
    }
    if (!workDir) {
      return NextResponse.json(
        { error: `${framework} path not configured in Settings.` },
        { status: 400 }
      );
    }

    // GPU the job ran on. `trainingParams` is a JSON *string*, so the previous
    // `(job.trainingParams as any)?.gpuIds` was always undefined and every export
    // silently used GPU 0's interpreter.
    let jobParams: Record<string, unknown> = {};
    try {
      jobParams = job.trainingParams ? JSON.parse(job.trainingParams as string) : {};
    } catch {
      // Fall through to GPU 0.
    }
    const primaryGpuId = String((jobParams.gpuIds as string) || '0').split(',')[0].trim();

    const { pythonPath: resolvedPython, source: pythonSource } = resolvePythonPath(
      framework,
      primaryGpuId,
      systemConfig,
    );
    if (!resolvedPython) {
      return NextResponse.json(
        {
          error: `No Python environment configured for ${framework}. ` +
            `Set one under Settings → Framework Python environments (or GPU ${primaryGpuId}).`,
        },
        { status: 400 }
      );
    }
    const pythonPath = resolvedPython;

    // Use provided config path, else the job's own merged config.
    let configPath: string | null = providedConfigPath || null;

    if (!configPath) {
      const userConfigsPath = (systemConfig as any)?.userConfigsPath || process.env.USER_CONFIGS_PATH;
      // `job.configPath` is the merged job config written at creation time; it is
      // the only config guaranteed to describe what was actually trained.
      if (job.configPath) {
        const candidate = path.isAbsolute(job.configPath)
          ? job.configPath
          : path.join(userConfigsPath || '', job.configPath);
        if (fs.existsSync(candidate)) configPath = candidate;
      }

      // Fallback: materialise the merged YAML stored on the job. Prefer it over
      // `job.config.yamlConfig`, which is only the *training* fragment and lacks
      // the dataset and model blocks the export script needs.
      const yaml = job.yamlConfig || job.config?.yamlConfig;
      if (!configPath && yaml) {
        const tempDir = path.join(process.cwd(), 'temp');
        if (!fs.existsSync(tempDir)) {
          fs.mkdirSync(tempDir, { recursive: true });
        }
        configPath = path.join(tempDir, `export_config_${job.id}.yml`);
        fs.writeFileSync(configPath, yaml, 'utf-8');
      }
    }

    if (!configPath) {
      return NextResponse.json(
        { error: 'Training config not found' },
        { status: 404 }
      );
    }

    // Export API: 统一使用标准路径 {userDatabasePath}/{username}/jobs/{jobName}/export_model
    const userDatabasePath = (systemConfig as any)?.userDatabasePath || process.env.DATABASE_PATH || process.cwd();
    const username = user?.username || 'default';
    const outputDir = path.join(userDatabasePath, username, 'jobs', job.name, 'export_model');

    // Ensure output directory exists
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    // Build export command. Args are passed as an argv array (never a shell
    // string), so paths containing spaces need no quoting and cannot inject.
    const absoluteConfigPath = path.resolve(configPath);
    const absoluteCheckpointPath = checkpointPath; // Use frontend-provided path directly
    const absoluteOutputDir = path.resolve(outputDir);

    const args = meta.cliStyle === 'config-flags'
      ? [
          meta.scripts.export,
          '--config', absoluteConfigPath,
          '--model_path', absoluteCheckpointPath,
          '--save_dir', absoluteOutputDir,
          // torchtrain writes TorchScript by default; PaddleSeg ignores --format.
          ...(meta.family === 'torch' ? ['--format', String(body.format || 'torchscript')] : []),
        ]
      : [
          meta.scripts.export,
          '-c', absoluteConfigPath,
          '-o', `weights=${absoluteCheckpointPath}`,
          // TensorRT-friendly export; PaddleDetection-only knob.
          'trt=True',
          '--output_dir', absoluteOutputDir,
        ];

    // Log activity
    await logActivity(userId, {
      action: 'export_model',
      entityType: 'checkpoint',
      entityId: jobId,
      entityName: checkpointName,
      details: { jobName: job.name, outputDir, pythonPath, cwd: workDir, framework },
    });

    // Debug logging
    console.log('[Export Debug] Framework:', framework);
    console.log('[Export Debug] Python Path:', pythonPath, `(from ${pythonSource})`);
    console.log('[Export Debug] Working Directory:', workDir);
    console.log('[Export Debug] Config Path:', absoluteConfigPath);
    console.log('[Export Debug] Checkpoint Path:', absoluteCheckpointPath);
    console.log('[Export Debug] Output Dir:', absoluteOutputDir);
    console.log('[Export Debug] Full Command:', `${pythonPath} ${args.join(' ')}`);

    // Execute export command
    return new Promise((resolve) => {
      const exportProcess = spawn(pythonPath, args, {
        cwd: workDir,
        env: { ...process.env, PYTHONPATH: workDir, PYTHONUNBUFFERED: '1' },
      });

      let stdout = '';
      let stderr = '';

      exportProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      exportProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      exportProcess.on('close', (code) => {
        if (code === 0) {
          // Find the exported model folders (子文件夹即为导出的模型)
          const exportedFolders: string[] = [];
          if (fs.existsSync(outputDir)) {
            const entries = fs.readdirSync(outputDir, { withFileTypes: true });
            entries.forEach((entry) => {
              if (entry.isDirectory()) {
                exportedFolders.push(path.join(outputDir, entry.name));
              }
            });
          }

          resolve(
            NextResponse.json({
              success: true,
              message: 'Export completed successfully',
              outputDir,
              exportedFiles: exportedFolders, // 返回文件夹路径
              jobId,
              checkpointName,
            })
          );
        } else {
          resolve(
            NextResponse.json(
              {
                error: 'Export failed',
                details: stderr || stdout,
                exitCode: code,
              },
              { status: 500 }
            )
          );
        }
      });

      exportProcess.on('error', (error) => {
        resolve(
          NextResponse.json(
            {
              error: 'Failed to start export process',
              details: error.message,
            },
            { status: 500 }
          )
        );
      });
    });
  } catch (error) {
    console.error('Export error:', error);
    return NextResponse.json(
      {
        error: 'Failed to export model',
        details: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}

// GET /api/checkpoints/export - Download exported model
export async function GET(request: NextRequest) {
  try {
    // Check authentication
    const authResult = await requireAuth(request);
    if (authResult instanceof NextResponse) {
      return authResult;
    }
    const { userId } = authResult;

    // Get user info
    const user = await db.user.findUnique({
      where: { id: userId },
      select: { username: true },
    });
    if (!user) {
      return NextResponse.json({ error: 'User not found' }, { status: 404 });
    }

    const { searchParams } = new URL(request.url);
    const folderPath = searchParams.get('path');
    const isFolder = searchParams.get('folder') === 'true';

    if (!folderPath) {
      return NextResponse.json({ error: 'Path required' }, { status: 400 });
    }

    // Security check - ensure file is within user's jobs directory
    const systemConfig = await db.systemConfig.findFirst();
    const userDatabasePath = (systemConfig as any)?.userDatabasePath || process.env.DATABASE_PATH || process.cwd();
    const allowedBase = path.join(userDatabasePath, user.username || 'default', 'jobs');
    const resolvedPath = path.resolve(folderPath);
    const resolvedAllowed = path.resolve(allowedBase);

    if (!resolvedPath.startsWith(resolvedAllowed)) {
      return NextResponse.json({ error: 'Access denied' }, { status: 403 });
    }

    if (!fs.existsSync(resolvedPath)) {
      return NextResponse.json({ error: 'Path not found' }, { status: 404 });
    }

    // If it's a folder, create a zip file
    if (isFolder && fs.statSync(resolvedPath).isDirectory()) {
      try {
        const files = fs.readdirSync(resolvedPath);
        if (files.length === 0) {
          return NextResponse.json({ error: 'Folder is empty' }, { status: 404 });
        }
        
        // Create a temporary zip file
        const tempDir = path.join(process.cwd(), 'temp');
        if (!fs.existsSync(tempDir)) {
          fs.mkdirSync(tempDir, { recursive: true });
        }
        
        const folderName = path.basename(resolvedPath);
        const zipFileName = `${folderName}.zip`;
        const zipFilePath = path.join(tempDir, zipFileName);
        
        // Use PowerShell on Windows to create zip (built-in)
        const isWindows = process.platform === 'win32';
        let zipCommand: string;
        
        if (isWindows) {
          // Windows PowerShell Compress-Archive
          zipCommand = `powershell.exe -Command "Compress-Archive -Path '${resolvedPath.replace(/'/g, "''")}\\*' -DestinationPath '${zipFilePath.replace(/'/g, "''")}' -Force"`;
        } else {
          // Linux/Mac use zip command
          zipCommand = `cd "${resolvedPath}" && zip -r "${zipFilePath}" .`;
        }
        
        await execAsync(zipCommand);
        
        // Read the zip file
        const zipBuffer = fs.readFileSync(zipFilePath);
        
        // Clean up temp file
        try {
          fs.unlinkSync(zipFilePath);
        } catch {
          // Ignore cleanup errors
        }
        
        return new NextResponse(zipBuffer, {
          headers: {
            'Content-Type': 'application/zip',
            'Content-Disposition': `attachment; filename="${zipFileName}"`,
          },
        });
      } catch (error) {
        console.error('Failed to create zip:', error);
        return NextResponse.json(
          { error: 'Failed to create zip archive', details: error instanceof Error ? error.message : 'Unknown error' },
          { status: 500 }
        );
      }
    }

    // Single file download
    const fileBuffer = fs.readFileSync(resolvedPath);
    const fileName = path.basename(resolvedPath);

    return new NextResponse(fileBuffer, {
      headers: {
        'Content-Type': 'application/octet-stream',
        'Content-Disposition': `attachment; filename="${fileName}"`,
      },
    });
  } catch (error) {
    console.error('Download error:', error);
    return NextResponse.json(
      { error: 'Download failed', details: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}
