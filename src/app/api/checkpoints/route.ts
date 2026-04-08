import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import * as fs from 'fs';
import * as path from 'path';

// Simple YAML parser for save_dir extraction (avoiding js-yaml dependency issues)
function extractSaveDir(yamlContent: string): string | null {
  // Look for save_dir in various formats
  const patterns = [
    /^save_dir:\s*["']?([^"'\n]+)["']?/m,
    /^\s+save_dir:\s*["']?([^"'\n]+)["']?/m,
    /log_dir:\s*["']?([^"'\n]+)["']?/m,
  ];
  
  for (const pattern of patterns) {
    const match = yamlContent.match(pattern);
    if (match) {
      return match[1].trim();
    }
  }
  return null;
}

// GET /api/checkpoints - Get checkpoints from a directory or training job
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const jobId = searchParams.get('jobId');
    const customDir = searchParams.get('dir');
    const checkExported = searchParams.get('checkExported') === 'true';

    let saveDir = customDir;

    // If jobId is provided, get the save_dir from the job's yaml config
    if (jobId && !customDir) {
      const job = await db.trainingJob.findUnique({
        where: { id: jobId },
        include: {
          project: { select: { name: true, user: { select: { username: true } } } },
        },
      });

      if (!job) {
        return NextResponse.json(
          { error: 'Training job not found' },
          { status: 404 }
        );
      }

      // Try to get save_dir from yamlConfig
      if (job.yamlConfig) {
        const extractedDir = extractSaveDir(job.yamlConfig);
        if (extractedDir) {
          saveDir = extractedDir;
        }
      }
      
      // Fallback to job's outputDir if no save_dir found
      if (!saveDir && job.outputDir) {
        saveDir = job.outputDir;
      }
    }

    if (!saveDir) {
      return NextResponse.json({
        saveDir: null,
        checkpoints: [],
        message: 'No save directory specified',
      });
    }

    // Get system config for work directory
    const systemConfig = await db.systemConfig.findFirst();
    const workDir = systemConfig?.paddleDetectionPath;

    if (!workDir) {
      return NextResponse.json(
        { error: 'PaddleDetection path not configured' },
        { status: 400 }
      );
    }

    // Resolve full path
    const fullPath = path.isAbsolute(saveDir) ? saveDir : path.join(workDir, saveDir);

    // Check if directory exists
    if (!fs.existsSync(fullPath)) {
      return NextResponse.json({
        saveDir,
        fullPath,
        checkpoints: [],
        message: 'Output directory does not exist yet (training may not have started)',
      });
    }

    // Find all checkpoint files
    const checkpoints: Array<{
      name: string;
      path: string;
      relativePath: string;
      size: number;
      mtime: string;
      epoch?: number;
      exportedFiles?: string[];
    }> = [];

    // Read directory contents
    const files = fs.readdirSync(fullPath);
    
    // Check for exported files - 统一检查 {userDatabasePath}/{username}/jobs/{jobName}/export_model
    let hasExportDir = false;
    let exportModelDir = '';
    
    if (checkExported && jobId) {
      // 重新获取 job 信息以确定正确的 job name
      const jobForExport = await db.trainingJob.findUnique({
        where: { id: jobId },
        select: { name: true, project: { select: { user: { select: { username: true } } } } }
      });
      
      if (jobForExport) {
        const systemConfig = await db.systemConfig.findFirst();
        const userDatabasePath = (systemConfig as any)?.userDatabasePath || process.env.DATABASE_PATH || process.cwd();
        const jobUsername = jobForExport.project?.user?.username || 'default';
        
        // 统一路径: {userDatabasePath}/{username}/jobs/{jobName}/export_model
        exportModelDir = path.join(userDatabasePath, jobUsername, 'jobs', jobForExport.name, 'export_model');
        hasExportDir = fs.existsSync(exportModelDir);
      }
    }
    
    for (const file of files) {
      if (file.endsWith('.pdparams')) {
        const filePath = path.join(fullPath, file);
        const stats = fs.statSync(filePath);
        
        // Extract epoch number if present (e.g., model_epoch_10.pdparams -> epoch 10)
        const epochMatch = file.match(/epoch[_]?(\d+)/i) || file.match(/(\d+)\.pdparams$/);
        const epoch = epochMatch ? parseInt(epochMatch[1], 10) : undefined;

        // Determine relative path for command
        const relativePath = path.isAbsolute(saveDir) 
          ? filePath 
          : path.join(saveDir, file);

        const checkpoint: typeof checkpoints[0] = {
          name: file,
          path: filePath,
          relativePath: relativePath.replace(/\\/g, '/'),
          size: stats.size,
          mtime: stats.mtime.toISOString(),
          epoch,
        };
        
        // Check for exported model folders - 扫描 export_model 下的子文件夹
        if (hasExportDir) {
          try {
            const exportEntries = fs.readdirSync(exportModelDir, { withFileTypes: true });
            // 只获取子文件夹（每个子文件夹是一个导出的模型）
            const modelFolders = exportEntries
              .filter(entry => entry.isDirectory())
              .map(entry => path.join(exportModelDir, entry.name));
            
            if (modelFolders.length > 0) {
              checkpoint.exportedFiles = modelFolders;
            }
          } catch {
            // Ignore errors reading export directory
          }
        }

        checkpoints.push(checkpoint);
      }
    }

    // Sort by modification time (newest first), but prioritize model_final.pdparams and best_model.pdparams
    checkpoints.sort((a, b) => {
      if (a.name === 'model_final.pdparams') return -1;
      if (b.name === 'model_final.pdparams') return 1;
      if (a.name === 'best_model.pdparams') return -1;
      if (b.name === 'best_model.pdparams') return 1;
      if (a.epoch !== undefined && b.epoch !== undefined) {
        return b.epoch - a.epoch;
      }
      return new Date(b.mtime).getTime() - new Date(a.mtime).getTime();
    });

    return NextResponse.json({
      saveDir,
      fullPath,
      checkpoints,
      count: checkpoints.length,
      exportModelDir: hasExportDir ? exportModelDir : undefined,
    });
  } catch (error) {
    console.error('Error getting checkpoints:', error);
    return NextResponse.json(
      { error: 'Failed to get checkpoints', message: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}
