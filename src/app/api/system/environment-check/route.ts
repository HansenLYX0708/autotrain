import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import { exec } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs';
import * as path from 'path';
import { FRAMEWORK_LIST, FRAMEWORK_META, type Framework } from '@/lib/frameworks';

const execAsync = promisify(exec);

interface FrameworkSupport {
  /** `import ppdet` succeeded. */
  paddleDetection: boolean;
  /** `import ppcls` succeeded. */
  paddleClas: boolean;
  /** `import paddleseg` succeeded. */
  paddleSeg: boolean;
  /** `import torch` succeeded (serves both TorchDet and TorchSeg). */
  torch: boolean;
  /** `import torchvision` succeeded; required by every torchtrain model. */
  torchvision: boolean;
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

/** A per-framework interpreter from `frameworkPythonMappings`. */
interface FrameworkEnvironmentCheck extends Omit<GpuEnvironmentCheck, 'gpuId'> {
  /** Mapping key: a framework name, or `"<framework>:<gpuId>"`. */
  key: string;
  framework: string;
  /** Whether this env has the module the framework needs. */
  hasModule: boolean;
}

interface VersionCheckResult {
  exists: boolean;
  version: string | null;
  isValid: boolean;
  error?: string;
}

const MODULE_PROBES: Array<[keyof FrameworkSupport, string]> = [
  ['paddleDetection', 'ppdet'],
  ['paddleClas', 'ppcls'],
  ['paddleSeg', 'paddleseg'],
  ['torch', 'torch'],
  ['torchvision', 'torchvision'],
];

const NO_MODULES: FrameworkSupport = {
  paddleDetection: false,
  paddleClas: false,
  paddleSeg: false,
  torch: false,
  torchvision: false,
};

/**
 * Probe a Python env for which framework packages are installed. One
 * short-lived Python process attempts the imports and prints a compact JSON
 * payload; a total failure (e.g. non-existent interpreter) falls back to
 * all-false so the UI degrades gracefully.
 *
 * We intentionally do NOT install anything — the UI merely reports the state.
 */
async function checkFrameworkModules(pythonPath: string): Promise<FrameworkSupport> {
  const pairs = MODULE_PROBES.map(([key, mod]) => `('${key}','${mod}')`).join(',');
  const script =
    'import json,importlib.util;' +
    'r={};' +
    `[r.__setitem__(k, (True if importlib.util.find_spec(m) else False)) for k,m in [${pairs}]];` +
    'print(json.dumps(r))';
  try {
    const { stdout } = await execAsync(`"${pythonPath}" -c "${script}"`, { timeout: 20000 });
    const parsed = JSON.parse(stdout.trim().split(/\r?\n/).pop() || '{}');
    const out = { ...NO_MODULES };
    for (const [key] of MODULE_PROBES) out[key] = !!parsed[key];
    return out;
  } catch {
    return { ...NO_MODULES };
  }
}

/**
 * Check a Python interpreter's version.
 *
 * The accepted range is deliberately wider than PaddlePaddle's: torch supports
 * 3.8–3.12, and rejecting a perfectly good torch env as "not supported" (which
 * the old 3.7–3.10 bound did) is worse than not flagging an unusual Paddle env.
 * Paddle-specific compatibility is surfaced by the module probe instead, which
 * reports whether `ppdet`/`paddleseg` actually imported.
 */
async function checkPythonVersion(
  pythonPath: string,
): Promise<Omit<GpuEnvironmentCheck, 'gpuId' | 'pythonPath' | 'frameworks'>> {
  if (!pythonPath || !fs.existsSync(pythonPath)) {
    return { exists: false, version: null, isValid: false, error: 'Python executable not found' };
  }

  try {
    const { stdout } = await execAsync(`"${pythonPath}" --version`);
    const versionMatch = stdout.match(/Python\s+(\d+)\.(\d+)\.(\d+)/);

    if (!versionMatch) {
      return { exists: true, version: stdout.trim(), isValid: false, error: 'Could not parse Python version' };
    }

    const major = parseInt(versionMatch[1]);
    const minor = parseInt(versionMatch[2]);
    const version = `${major}.${minor}.${versionMatch[3]}`;
    const isValid = major === 3 && minor >= 7 && minor <= 12;

    return {
      exists: true,
      version,
      isValid,
      error: isValid ? undefined : `Python ${version} is not supported. Required: 3.7 - 3.12`,
    };
  } catch {
    return { exists: true, version: null, isValid: false, error: 'Failed to execute Python --version' };
  }
}

/**
 * Check a framework installation by verifying its marker files exist.
 * The marker list lives in `FRAMEWORK_META[...].requiredFiles`.
 */
async function checkFrameworkInstall(
  frameworkPath: string,
  framework: Framework
): Promise<VersionCheckResult> {
  if (!frameworkPath) {
    return { exists: false, version: null, isValid: false, error: `${framework} path not configured` };
  }

  if (!fs.existsSync(frameworkPath)) {
    return { exists: false, version: null, isValid: false, error: `${framework} directory not found` };
  }

  const required = FRAMEWORK_META[framework].requiredFiles;
  const missing = required.filter(
    (rel) => !fs.existsSync(path.join(frameworkPath, ...rel.split('/')))
  );

  if (missing.length > 0) {
    return {
      exists: true,
      version: null,
      isValid: false,
      error: `Invalid ${framework} installation. Missing: ${missing.join(', ')}`,
    };
  }

  return { exists: true, version: null, isValid: true };
}

function parseMapping(raw: unknown): Record<string, string> {
  if (typeof raw !== 'string' || !raw.trim()) return {};
  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const out: Record<string, string> = {};
    for (const [key, value] of Object.entries(parsed ?? {})) {
      if (typeof value === 'string' && value.trim()) out[key] = value.trim();
    }
    return out;
  } catch (e) {
    console.error('Failed to parse python mappings:', e);
    return {};
  }
}

const NOT_CONFIGURED: VersionCheckResult = {
  exists: false,
  version: null,
  isValid: false,
  error: 'Not configured',
};

function emptyPayload(reason: string) {
  const frameworks: Record<string, VersionCheckResult> = {};
  for (const name of FRAMEWORK_LIST) frameworks[name] = { ...NOT_CONFIGURED, error: reason };
  return {
    // Legacy top-level keys, kept so existing dashboard code keeps working.
    paddleDetection: frameworks.PaddleDetection,
    paddleClas: frameworks.PaddleClas,
    paddleSeg: frameworks.PaddleSeg,
    torch: frameworks.TorchSeg,
    frameworks,
    gpuEnvironments: [],
    frameworkEnvironments: [],
    totalGpus: 0,
    configuredGpus: 0,
    validGpus: 0,
  };
}

// GET /api/system/environment-check - Check framework repos and Python environments
export async function GET(_request: NextRequest) {
  try {
    const systemConfig = await db.systemConfig.findFirst();

    if (!systemConfig) {
      return NextResponse.json({
        success: false,
        error: 'System configuration not found',
        data: emptyPayload('Not configured'),
      });
    }

    // Check every registered framework's repository path. TorchDet and TorchSeg
    // share `torchPath`, so both entries report on the same folder.
    const frameworks: Record<string, VersionCheckResult> = {};
    for (const name of FRAMEWORK_LIST) {
      const configured = (systemConfig as unknown as Record<string, string | null>)[
        FRAMEWORK_META[name].pathField
      ];
      frameworks[name] = await checkFrameworkInstall(configured || '', name);
    }

    // Per-GPU interpreters (the historical mapping; typically PaddlePaddle envs).
    const gpuPythonMappings = parseMapping((systemConfig as any).gpuPythonMappings);
    const gpuEnvironments: GpuEnvironmentCheck[] = [];
    for (const [gpuIdStr, pythonPath] of Object.entries(gpuPythonMappings)) {
      const gpuId = parseInt(gpuIdStr, 10);
      if (isNaN(gpuId)) continue;

      const check = await checkPythonVersion(pythonPath);
      // Only probe framework packages when the interpreter is at least
      // reachable — saves a slow spawn for missing binaries.
      const modules = check.exists ? await checkFrameworkModules(pythonPath) : undefined;
      gpuEnvironments.push({ gpuId, pythonPath, ...check, frameworks: modules });
    }

    // Per-framework interpreters. These exist because a PaddlePaddle env has no
    // `torch` and vice versa, so the per-GPU map above cannot serve both.
    const frameworkPythonMappings = parseMapping((systemConfig as any).frameworkPythonMappings);
    const frameworkEnvironments: FrameworkEnvironmentCheck[] = [];
    for (const [key, pythonPath] of Object.entries(frameworkPythonMappings)) {
      const frameworkName = key.split(':')[0];
      const meta = FRAMEWORK_META[frameworkName as Framework];
      const check = await checkPythonVersion(pythonPath);
      const modules = check.exists ? await checkFrameworkModules(pythonPath) : undefined;
      const moduleKey = meta?.pythonModule;
      const hasModule = !!(
        modules &&
        moduleKey &&
        MODULE_PROBES.some(([probeKey, mod]) => mod === moduleKey && modules[probeKey])
      );
      frameworkEnvironments.push({
        key,
        framework: frameworkName,
        pythonPath,
        ...check,
        frameworks: modules,
        hasModule,
        error: check.error ?? (hasModule ? undefined : `\`import ${moduleKey}\` failed in this environment`),
        isValid: check.isValid && hasModule,
      });
    }

    const totalGpus = gpuEnvironments.length;
    const configuredGpus = gpuEnvironments.filter(g => g.exists).length;
    const validGpus = gpuEnvironments.filter(g => g.isValid).length;

    return NextResponse.json({
      success: true,
      data: {
        // Legacy top-level keys, kept so existing dashboard code keeps working.
        paddleDetection: frameworks.PaddleDetection,
        paddleClas: frameworks.PaddleClas,
        paddleSeg: frameworks.PaddleSeg,
        /** The bundled `torchtrain/` repository (shared by TorchDet + TorchSeg). */
        torch: frameworks.TorchSeg,
        frameworks,
        gpuEnvironments,
        frameworkEnvironments,
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
      data: emptyPayload('Check failed'),
    }, { status: 500 });
  }
}
