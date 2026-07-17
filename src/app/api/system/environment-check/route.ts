import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { exec } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs';
import * as path from 'path';

const execAsync = promisify(exec);

interface FrameworkSupport {
  /** `import ppdet` succeeded. */
  paddleDetection: boolean;
  /** `import ppcls` succeeded. */
  paddleClas: boolean;
  /** `import paddleseg` succeeded. */
  paddleSeg: boolean;
}

interface GpuEnvironmentCheck {
  gpuId: number;
  pythonPath: string;
  exists: boolean;
  version: string | null;
  isValid: boolean;
  error?: string;
  /**
   * Which Python framework packages are installed in this env. Independent of
   * the repository path checks — a user can have the PaddleSeg *repo* on disk
   * but not have `pip install paddleseg`ed the package into this GPU's env.
   */
  frameworks?: FrameworkSupport;
}

interface VersionCheckResult {
  exists: boolean;
  version: string | null;
  isValid: boolean;
  error?: string;
}

interface EnvironmentCheck {
  paddleDetection: VersionCheckResult;
  paddleClas: VersionCheckResult;
  paddleSeg: VersionCheckResult;
  gpuEnvironments: GpuEnvironmentCheck[];
  totalGpus: number;
  configuredGpus: number;
  validGpus: number;
}

/**
 * Probe a Python env for which framework packages are installed. One
 * short-lived Python process attempts three imports and prints a compact
 * JSON payload; a total failure (e.g. non-existent interpreter) falls back to
 * all-false so the UI degrades gracefully.
 *
 * We intentionally do NOT install anything — the UI merely reports the state.
 */
async function checkFrameworkModules(pythonPath: string): Promise<FrameworkSupport> {
  const script =
    'import json,importlib.util;' +
    'r={};' +
    "[r.__setitem__(k, (True if importlib.util.find_spec(m) else False)) for k,m in " +
    "[('paddleDetection','ppdet'),('paddleClas','ppcls'),('paddleSeg','paddleseg')]];" +
    'print(json.dumps(r))';
  try {
    const { stdout } = await execAsync(`"${pythonPath}" -c "${script}"`, { timeout: 15000 });
    const parsed = JSON.parse(stdout.trim().split(/\r?\n/).pop() || '{}');
    return {
      paddleDetection: !!parsed.paddleDetection,
      paddleClas: !!parsed.paddleClas,
      paddleSeg: !!parsed.paddleSeg,
    };
  } catch {
    return { paddleDetection: false, paddleClas: false, paddleSeg: false };
  }
}

// Check Python version
async function checkPythonVersion(pythonPath: string): Promise<Omit<GpuEnvironmentCheck, 'gpuId' | 'pythonPath' | 'frameworks'>> {
  if (!pythonPath || !fs.existsSync(pythonPath)) {
    return {
      exists: false,
      version: null,
      isValid: false,
      error: 'Python executable not found',
    };
  }

  try {
    const { stdout } = await execAsync(`"${pythonPath}" --version`);
    const versionMatch = stdout.match(/Python\s+(\d+)\.(\d+)\.(\d+)/);
    
    if (!versionMatch) {
      return {
        exists: true,
        version: stdout.trim(),
        isValid: false,
        error: 'Could not parse Python version',
      };
    }

    const major = parseInt(versionMatch[1]);
    const minor = parseInt(versionMatch[2]);
    const version = `${major}.${minor}.${versionMatch[3]}`;
    
    // Check if version is between 3.7 and 3.10
    const isValid = major === 3 && minor >= 7 && minor <= 10;

    return {
      exists: true,
      version,
      isValid,
      error: isValid ? undefined : `Python ${version} is not supported. Required: 3.7 - 3.10`,
    };
  } catch (error) {
    return {
      exists: true,
      version: null,
      isValid: false,
      error: 'Failed to execute Python --version',
    };
  }
}

// Required marker files per framework (relative to the framework root)
const FRAMEWORK_REQUIRED_FILES: Record<string, string[]> = {
  PaddleDetection: ['ppdet', 'tools/train.py', 'tools/eval.py'],
  PaddleClas: ['ppcls', 'tools/train.py', 'tools/eval.py'],
  PaddleSeg: ['paddleseg', 'tools/train.py', 'tools/val.py'],
};

// Check a framework installation by verifying its key files exist
async function checkFrameworkInstall(
  frameworkPath: string,
  framework: string
): Promise<VersionCheckResult> {
  if (!frameworkPath) {
    return {
      exists: false,
      version: null,
      isValid: false,
      error: `${framework} path not configured`,
    };
  }

  if (!fs.existsSync(frameworkPath)) {
    return {
      exists: false,
      version: null,
      isValid: false,
      error: `${framework} directory not found`,
    };
  }

  // Check if it's a valid installation by checking for key files
  const required = FRAMEWORK_REQUIRED_FILES[framework] || [];
  const requiredFiles = required.map(rel => path.join(frameworkPath, ...rel.split('/')));

  const allFilesExist = requiredFiles.every(file => fs.existsSync(file));

  if (!allFilesExist) {
    return {
      exists: true,
      version: null,
      isValid: false,
      error: `Invalid ${framework} installation. Required files not found.`,
    };
  }

  return {
    exists: true,
    version: null,
    isValid: true,
  };
}

// GET /api/system/environment-check - Check Python environments for each GPU and PaddleDetection
export async function GET(request: NextRequest) {
  try {
    // Get system config
    const systemConfig = await db.systemConfig.findFirst();
    
    if (!systemConfig) {
      return NextResponse.json({
        success: false,
        error: 'System configuration not found',
        data: {
          paddleDetection: { exists: false, version: null, isValid: false, error: 'Not configured' },
          paddleClas: { exists: false, version: null, isValid: false, error: 'Not configured' },
          paddleSeg: { exists: false, version: null, isValid: false, error: 'Not configured' },
          gpuEnvironments: [],
          totalGpus: 0,
          configuredGpus: 0,
          validGpus: 0,
        },
      });
    }

    // Check each configured framework installation
    const paddleDetection = await checkFrameworkInstall(systemConfig.paddleDetectionPath || '', 'PaddleDetection');
    const paddleClas = await checkFrameworkInstall((systemConfig as any).paddleClasPath || '', 'PaddleClas');
    const paddleSeg = await checkFrameworkInstall((systemConfig as any).paddleSegPath || '', 'PaddleSeg');

    // Parse GPU Python mappings
    let gpuPythonMappings: Record<string, string> = {};
    const gpuMappingsConfig = (systemConfig as any).gpuPythonMappings;
    if (gpuMappingsConfig) {
      try {
        gpuPythonMappings = JSON.parse(gpuMappingsConfig);
      } catch (e) {
        console.error('Failed to parse gpuPythonMappings:', e);
      }
    }

    // Check each GPU's Python environment
    const gpuEnvironments: GpuEnvironmentCheck[] = [];
    const gpuEntries = Object.entries(gpuPythonMappings);
    
    for (const [gpuIdStr, pythonPath] of gpuEntries) {
      const gpuId = parseInt(gpuIdStr, 10);
      if (isNaN(gpuId)) continue;

      const check = await checkPythonVersion(pythonPath);
      // Only probe framework packages when the interpreter is at least
      // reachable — saves a slow spawn for missing binaries.
      const frameworks = check.exists
        ? await checkFrameworkModules(pythonPath)
        : undefined;
      gpuEnvironments.push({
        gpuId,
        pythonPath,
        ...check,
        frameworks,
      });
    }

    const totalGpus = gpuEnvironments.length;
    const configuredGpus = gpuEnvironments.filter(g => g.exists).length;
    const validGpus = gpuEnvironments.filter(g => g.isValid).length;

    return NextResponse.json({
      success: true,
      data: {
        paddleDetection,
        paddleClas,
        paddleSeg,
        gpuEnvironments,
        totalGpus,
        configuredGpus,
        validGpus,
      },
    });
  } catch (error) {
    console.error('Error checking environment:', error);
    return NextResponse.json({
      success: false,
      error: 'Failed to check environment',
      message: error instanceof Error ? error.message : 'Unknown error',
      data: {
        paddleDetection: { exists: false, version: null, isValid: false, error: 'Check failed' },
        paddleClas: { exists: false, version: null, isValid: false, error: 'Check failed' },
        paddleSeg: { exists: false, version: null, isValid: false, error: 'Check failed' },
        gpuEnvironments: [],
        totalGpus: 0,
        configuredGpus: 0,
        validGpus: 0,
      },
    }, { status: 500 });
  }
}
