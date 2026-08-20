import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import * as fs from 'fs';
import * as path from 'path';
import { frameworkMeta, getWorkDir } from '@/lib/frameworks';

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

// PaddleSeg, TorchSeg and TorchDet keep save_dir out of the YAML and pass it via
// CLI:
//   ... tools/train.py --config "..." --save_dir "H:\...\jobs\<name>" ...
// This extractor recovers it from a stored command string so historical jobs
// (whose `outputDir` still holds only the relative fallback) can still resolve
// their checkpoint folder.
function extractSaveDirFromCommand(cmd: string | null | undefined): string | null {
  if (!cmd) return null;
  const patterns = [
    /--save_dir\s+"([^"]+)"/,
    /--save_dir\s+'([^']+)'/,
    /--save_dir[=\s]+(\S+)/,
  ];
  for (const p of patterns) {
    const m = cmd.match(p);
    if (m) return m[1].trim();
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
    let framework: string = 'PaddleDetection';

    // If jobId is provided, get the save_dir from the job's yaml config
    if (jobId && !customDir) {
      const job = await db.trainingJob.findUnique({
        where: { id: jobId },
        select: {
          yamlConfig: true,
          outputDir: true,
          command: true,
          project: { select: { name: true, framework: true, user: { select: { username: true } } } },
        },
      });

      if (!job) {
        return NextResponse.json(
          { error: 'Training job not found' },
          { status: 404 }
        );
      }

      framework = job.project?.framework || 'PaddleDetection';

      // For frameworks that take save_dir on the CLI (`--save_dir "..."`) rather
      // than in the YAML, check the stored command first so freshly-trained jobs
      // resolve without users having to manually type a path.
      if (frameworkMeta(framework).saveDirOnCli && !saveDir) {
        const fromCmd = extractSaveDirFromCommand(job.command);
        if (fromCmd) saveDir = fromCmd;
      }

      // Try to get save_dir from yamlConfig
      if (!saveDir && job.yamlConfig) {
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

    // Get system config, resolve the framework-specific work directory. When
    // save_dir is already absolute (the common case in this platform) workDir
    // is only used as a relative-path fallback, so a missing framework path is
    // not fatal here.
    const systemConfig = await db.systemConfig.findFirst();
    const workDir = getWorkDir(framework, systemConfig) || systemConfig?.paddleDetectionPath || '';

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

    const meta = frameworkMeta(framework);

    if (meta.checkpointLayout === 'nested') {
      // Nested layout (PaddleSeg, TorchSeg, TorchDet):
      //   {save_dir}/best_model/<weightFile>
      //   {save_dir}/iter_{N}/<weightFile>     (segmentation)
      //   {save_dir}/epoch_{N}/<weightFile>    (TorchDet)
      //   {save_dir}/model_final/<weightFile>  (TorchDet, last epoch)
      // We surface one entry per subfolder, using the subfolder name as the
      // checkpoint's display name and its weight file as the target path (this
      // is what `val.py --model_path` / `predict.py --model_path` expect).
      const entries = fs.readdirSync(fullPath, { withFileTypes: true });
      for (const entry of entries) {
        if (!entry.isDirectory()) continue;
        const weightsFile = path.join(fullPath, entry.name, meta.weightFile);
        if (!fs.existsSync(weightsFile)) continue;

        const stats = fs.statSync(weightsFile);
        // `iter_N` for segmentation, `epoch_N` for TorchDet.
        const stepMatch = entry.name.match(/(?:iter|epoch)[_-]?(\d+)/i);
        const step = stepMatch ? parseInt(stepMatch[1], 10) : undefined;

        // Preserve absolute vs relative semantics used by the flat branch: when
        // save_dir was absolute, keep the full path; otherwise join under
        // save_dir so downstream commands stay portable.
        const relativePath = path.isAbsolute(saveDir)
          ? weightsFile
          : path.join(saveDir, entry.name, meta.weightFile);

        checkpoints.push({
          name: entry.name, // e.g. "best_model", "iter_20000", "epoch_11"
          path: weightsFile,
          relativePath: relativePath.replace(/\\/g, '/'),
          size: stats.size,
          mtime: stats.mtime.toISOString(),
          epoch: step,
        });
      }

      // Sort: best_model, then model_final, then step folders desc, then mtime.
      const rank = (name: string) => (name === 'best_model' ? 0 : name === 'model_final' ? 1 : 2);
      checkpoints.sort((a, b) => {
        const byRank = rank(a.name) - rank(b.name);
        if (byRank !== 0) return byRank;
        if (a.epoch !== undefined && b.epoch !== undefined) return b.epoch - a.epoch;
        return new Date(b.mtime).getTime() - new Date(a.mtime).getTime();
      });

      return NextResponse.json({
        saveDir,
        fullPath,
        checkpoints,
        count: checkpoints.length,
      });
    }

    // PaddleDetection / PaddleClas layout: flat *.pdparams files at save_dir root.
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
