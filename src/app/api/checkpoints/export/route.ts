import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { spawn, exec } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import { promisify } from 'util';
import { requireAuth, buildUserFilter, getCurrentUser } from '@/lib/auth';
import { logActivity } from '@/lib/activity-log';

const execAsync = promisify(exec);

// POST /api/checkpoints/export - Export checkpoint to TensorRT format
export async function POST(request: NextRequest) {
  try {
    // Check authentication
    const authResult = await requireAuth(request);
    if (authResult instanceof NextResponse) {
      return authResult;
    }
    const { userId, role } = authResult;

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
        project: { select: { id: true, name: true } },
        config: true,
      },
    });

    if (!job) {
      return NextResponse.json({ error: 'Job not found' }, { status: 404 });
    }

    // Get system config
    const systemConfig = await db.systemConfig.findFirst();
    if (!systemConfig?.paddleDetectionPath) {
      return NextResponse.json(
        { error: 'PaddleDetection path not configured' },
        { status: 400 }
      );
    }

    const paddleDetectionPath = systemConfig.paddleDetectionPath;

    // Get Python path based on job's GPU configuration
    let gpuMappings = (systemConfig as any).gpuPythonMappings;
    let pythonPath = 'python';

    // Parse gpuPythonMappings if it's a JSON string
    if (gpuMappings && typeof gpuMappings === 'string') {
      try {
        gpuMappings = JSON.parse(gpuMappings);
        console.log('[Export Debug] Parsed gpuPythonMappings:', gpuMappings);
      } catch (e) {
        console.error('[Export] Failed to parse gpuPythonMappings:', e);
        gpuMappings = null;
      }
    }

    // Get GPU ID from job's trainingParams (default to '0' if not found)
    const jobGpuId = (job.trainingParams as any)?.gpuIds || '0';
    const primaryGpuId = String(jobGpuId).split(',')[0].trim();

    console.log('[Export Debug] GPU Mappings type:', typeof gpuMappings);
    console.log('[Export Debug] GPU Mappings:', gpuMappings);
    console.log('[Export Debug] Primary GPU ID:', primaryGpuId);

    if (gpuMappings) {
      // Handle array format: [{"gpuId": "0", "pythonPath": "..."}, ...]
      if (Array.isArray(gpuMappings)) {
        const mapping = gpuMappings.find((m: any) => String(m.gpuId) === primaryGpuId);
        console.log('[Export Debug] Found array mapping:', mapping);
        if (mapping?.pythonPath) {
          pythonPath = mapping.pythonPath;
        }
      }
      // Handle object format: {"0": {"pythonPath": "..."}, ...}
      else if (typeof gpuMappings === 'object') {
        const mapping = gpuMappings[primaryGpuId] || 
                        gpuMappings[parseInt(primaryGpuId)] || 
                        gpuMappings['0'] || 
                        gpuMappings[0];
        console.log('[Export Debug] Found object mapping:', mapping);
        
        pythonPath = mapping;
        
      }
    }

    // Use provided config path or find the training config file path
    let configPath: string | null = providedConfigPath || null;

    // If no config path provided, try to find it
    if (!configPath) {
      const userConfigsPath = process.env.USER_CONFIGS_PATH;
      const config = job.config;

      if (userConfigsPath && user?.username && config) {
        // Try to find the config file in user's directory
        const configDir = path.join(userConfigsPath, user.username, 'training_configs');
        const configName = job.name.replace(/\s+/g, '_').replace(/[^a-zA-Z0-9_.-]/g, '').toLowerCase();
        const possibleConfigPath = path.join(configDir, `${configName}.yml`);
        
        if (fs.existsSync(possibleConfigPath)) {
          configPath = possibleConfigPath;
        }
      }

      // Fallback: create a temporary config from yamlConfig
      if (!configPath && config?.yamlConfig) {
        const tempDir = path.join(process.cwd(), 'temp');
        if (!fs.existsSync(tempDir)) {
          fs.mkdirSync(tempDir, { recursive: true });
        }
        configPath = path.join(tempDir, `export_config_${job.id}.yml`);
        fs.writeFileSync(configPath, config.yamlConfig, 'utf-8');
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

    // Build export command
    // Use the checkpointPath directly as provided by frontend (same as frontend)
    const absoluteConfigPath = path.resolve(configPath);
    const absoluteCheckpointPath = checkpointPath; // Use frontend-provided path directly
    const absoluteOutputDir = path.resolve(outputDir);
    const args = [
      'tools/export_model.py',
      '-c', absoluteConfigPath,
      '-o', `weights=${absoluteCheckpointPath}`,
      'trt=True',
      '--output_dir', absoluteOutputDir,
    ];

    // Log activity
    await logActivity(userId, {
      action: 'export_model',
      entityType: 'checkpoint',
      entityId: jobId,
      entityName: checkpointName,
      details: { jobName: job.name, outputDir, pythonPath, cwd: paddleDetectionPath },
    });

    // Debug logging
    console.log('[Export Debug] Python Path:', pythonPath);
    console.log('[Export Debug] Working Directory:', paddleDetectionPath);
    console.log('[Export Debug] Config Path:', absoluteConfigPath);
    console.log('[Export Debug] Checkpoint Path:', absoluteCheckpointPath);
    console.log('[Export Debug] Output Dir:', absoluteOutputDir);
    console.log('[Export Debug] Full Command:', `${pythonPath} ${args.join(' ')}`);

    // Execute export command
    return new Promise((resolve) => {
      const exportProcess = spawn(pythonPath, args, {
        cwd: paddleDetectionPath,
        env: { ...process.env, PYTHONPATH: paddleDetectionPath },
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
