/**
 * Framework helpers shared across API routes and pages.
 *
 * The platform supports three PaddlePaddle frameworks. Each project picks one
 * via `project.framework`. Work directories are resolved from `SystemConfig`
 * paths. Keep this the single source of truth so new frameworks only need to be
 * wired here plus the framework-specific YAML/command generators.
 */

export const FRAMEWORKS = {
  PaddleDetection: "PaddleDetection",
  PaddleClas: "PaddleClas",
  PaddleSeg: "PaddleSeg",
} as const;

export type Framework = (typeof FRAMEWORKS)[keyof typeof FRAMEWORKS];

export const FRAMEWORK_LIST: Framework[] = [
  "PaddleDetection",
  "PaddleClas",
  "PaddleSeg",
];

/** Minimal shape needed to resolve a work directory. */
export interface FrameworkPaths {
  paddleDetectionPath?: string | null;
  paddleClasPath?: string | null;
  paddleSegPath?: string | null;
}

/**
 * Resolve the framework working directory (repository root) from system config.
 * Falls back to the PaddleDetection path for unknown/legacy framework values.
 */
export function getWorkDir(
  framework: string | null | undefined,
  systemConfig: FrameworkPaths | null | undefined
): string | null | undefined {
  if (!systemConfig) return undefined;
  switch (framework) {
    case "PaddleClas":
      return systemConfig.paddleClasPath;
    case "PaddleSeg":
      return systemConfig.paddleSegPath;
    case "PaddleDetection":
    default:
      return systemConfig.paddleDetectionPath;
  }
}

/** Normalize an arbitrary value to a known framework, defaulting to PaddleDetection. */
export function normalizeFramework(framework: string | null | undefined): Framework {
  if (framework === "PaddleClas" || framework === "PaddleSeg") return framework;
  return "PaddleDetection";
}

export function isSegmentation(framework: string | null | undefined): boolean {
  return framework === "PaddleSeg";
}

export function isClassification(framework: string | null | undefined): boolean {
  return framework === "PaddleClas";
}

export function isDetection(framework: string | null | undefined): boolean {
  return normalizeFramework(framework) === "PaddleDetection";
}
