import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { exec } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs';
import * as path from 'path';

const execAsync = promisify(exec);

interface GpuEnvironmentCheck {
  gpuId: number;
  pythonPath: string;
  exists: boolean;
  version: string | null;
  isValid: boolean;
  error?: string;
}

interface VersionCheckResult {
  exists: boolean;
  version: string | null;
  isValid: boolean;
  error?: string;
}

interface EnvironmentCheck {
  paddleDetection: VersionCheckResult;
  gpuEnvironments: GpuEnvironmentCheck[];
  totalGpus: number;
  configuredGpus: number;
  validGpus: number;
}

// Check Python version
async function checkPythonVersion(pythonPath: string): Promise<Omit<GpuEnvironmentCheck, 'gpuId' | 'pythonPath'>> {
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

// Check PaddleDetection path exists
async function checkPaddleDetectionVersion(paddlePath: string): Promise<VersionCheckResult> {
  if (!paddlePath) {
    return {
      exists: false,
      version: null,
      isValid: false,
      error: 'PaddleDetection path not configured',
    };
  }

  if (!fs.existsSync(paddlePath)) {
    return {
      exists: false,
      version: null,
      isValid: false,
      error: 'PaddleDetection directory not found',
    };
  }

  // Check if it's a valid PaddleDetection installation by checking for key files
  const requiredFiles = [
    path.join(paddlePath, 'ppdet'),
    path.join(paddlePath, 'tools', 'train.py'),
    path.join(paddlePath, 'tools', 'eval.py'),
  ];

  const allFilesExist = requiredFiles.every(file => fs.existsSync(file));

  if (!allFilesExist) {
    return {
      exists: true,
      version: null,
      isValid: false,
      error: 'Invalid PaddleDetection installation. Required files not found.',
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
          gpuEnvironments: [],
          totalGpus: 0,
          configuredGpus: 0,
          validGpus: 0,
        },
      });
    }

    // Check PaddleDetection
    const paddleDetection = await checkPaddleDetectionVersion(systemConfig.paddleDetectionPath || '');

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
      gpuEnvironments.push({
        gpuId,
        pythonPath,
        ...check,
      });
    }

    const totalGpus = gpuEnvironments.length;
    const configuredGpus = gpuEnvironments.filter(g => g.exists).length;
    const validGpus = gpuEnvironments.filter(g => g.isValid).length;

    return NextResponse.json({
      success: true,
      data: {
        paddleDetection,
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
        gpuEnvironments: [],
        totalGpus: 0,
        configuredGpus: 0,
        validGpus: 0,
      },
    }, { status: 500 });
  }
}
