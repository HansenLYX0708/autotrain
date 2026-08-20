/**
 * Framework registry shared across API routes and pages.
 *
 * A project picks one framework via `project.framework`. Everything that varies
 * per framework - repository path, Python module to preflight, CLI script names,
 * checkpoint layout, task kind - is declared once in `FRAMEWORK_META` below, so
 * adding a framework is a data change here plus its YAML generators in
 * `@/lib/training-yaml` and `@/lib/model-yaml`.
 *
 * Two families are supported:
 *
 *   PaddlePaddle : PaddleDetection, PaddleClas, PaddleSeg
 *                  External repos; paths configured per framework in Settings.
 *   PyTorch      : TorchDet, TorchSeg
 *                  Both served by the `torchtrain/` repo bundled with this
 *                  project, so they share a single `torchPath`. It deliberately
 *                  mirrors the Paddle repo shape (`tools/train.py` + an
 *                  importable package) and emits Paddle-identical log lines, so
 *                  the log parsers, monitoring charts and progress tracking are
 *                  reused rather than duplicated.
 */

export const FRAMEWORKS = {
  PaddleDetection: "PaddleDetection",
  PaddleClas: "PaddleClas",
  PaddleSeg: "PaddleSeg",
  TorchDet: "TorchDet",
  TorchSeg: "TorchSeg",
} as const;

export type Framework = (typeof FRAMEWORKS)[keyof typeof FRAMEWORKS];

export const FRAMEWORK_LIST: Framework[] = [
  "PaddleDetection",
  "PaddleClas",
  "PaddleSeg",
  "TorchDet",
  "TorchSeg",
];

/** Which deep-learning runtime a framework belongs to. */
export type FrameworkFamily = "paddle" | "torch";

/** The kind of problem a framework solves. Drives dataset format + metrics. */
export type FrameworkTaskKind = "detection" | "classification" | "segmentation";

export interface FrameworkMeta {
  /** Human label for dropdowns and status cards. */
  label: string;
  family: FrameworkFamily;
  taskKind: FrameworkTaskKind;
  /** Default value for `project.task`. */
  defaultTask: string;
  /** Python module that must be importable for a job to run. */
  pythonModule: string;
  /** Hint shown when `pythonModule` is missing. */
  installHint: string;
  /** Which `SystemConfig` column holds this framework's repository path. */
  pathField: keyof FrameworkPaths;
  /** Files/folders that must exist inside the repository path. */
  requiredFiles: string[];
  /** CLI scripts, relative to the repository root. */
  scripts: {
    train: string;
    eval: string;
    infer: string;
    export?: string;
  };
  /**
   * CLI flag dialect.
   *   'dash-c'       `-c <cfg>` + `-o weights=<w>` (PaddleDetection, PaddleClas)
   *   'config-flags' `--config <cfg>` + `--model_path <w>` (PaddleSeg, torchtrain)
   */
  cliStyle: "dash-c" | "config-flags";
  /** Whether `save_dir` is a CLI argument rather than a YAML key. */
  saveDirOnCli: boolean;
  /**
   * Checkpoint layout.
   *   'nested' `<save_dir>/<name>/<weightFile>` (PaddleSeg, TorchSeg, TorchDet)
   *   'flat'   `<save_dir>/*.<ext>`             (PaddleDetection, PaddleClas)
   */
  checkpointLayout: "flat" | "nested";
  /** Weight file name (nested) or extension (flat). */
  weightFile: string;
  /** Dataset format this framework consumes. */
  datasetFormat: "COCO" | "PaddleSeg";
  /** Training length unit; segmentation frameworks count iterations. */
  stepUnit: "epoch" | "iter";
}

/** Minimal shape needed to resolve a work directory. */
export interface FrameworkPaths {
  paddleDetectionPath?: string | null;
  paddleClasPath?: string | null;
  paddleSegPath?: string | null;
  torchPath?: string | null;
}

/** Weight file written by `torchtrain` (see `torchtrain/torchtrain/utils.py`). */
export const TORCH_WEIGHT_FILE = "model.pt";

export const FRAMEWORK_META: Record<Framework, FrameworkMeta> = {
  PaddleDetection: {
    label: "PaddleDetection",
    family: "paddle",
    taskKind: "detection",
    defaultTask: "detection",
    pythonModule: "ppdet",
    installHint: "pip install ppdet  (or `pip install -e .` from the PaddleDetection repo root)",
    pathField: "paddleDetectionPath",
    requiredFiles: ["ppdet", "tools/train.py", "tools/eval.py"],
    scripts: { train: "tools/train.py", eval: "tools/eval.py", infer: "tools/infer.py", export: "tools/export_model.py" },
    cliStyle: "dash-c",
    saveDirOnCli: false,
    checkpointLayout: "flat",
    weightFile: ".pdparams",
    datasetFormat: "COCO",
    stepUnit: "epoch",
  },
  PaddleClas: {
    label: "PaddleClas",
    family: "paddle",
    taskKind: "classification",
    defaultTask: "classification",
    pythonModule: "ppcls",
    installHint: "pip install ppcls  (or `pip install -e .` from the PaddleClas repo root)",
    pathField: "paddleClasPath",
    requiredFiles: ["ppcls", "tools/train.py", "tools/eval.py"],
    scripts: { train: "tools/train.py", eval: "tools/eval.py", infer: "tools/infer.py" },
    cliStyle: "dash-c",
    saveDirOnCli: false,
    checkpointLayout: "flat",
    weightFile: ".pdparams",
    datasetFormat: "COCO",
    stepUnit: "epoch",
  },
  PaddleSeg: {
    label: "PaddleSeg",
    family: "paddle",
    taskKind: "segmentation",
    defaultTask: "semantic_segmentation",
    pythonModule: "paddleseg",
    installHint: "pip install paddleseg  (or `pip install -e .` from the PaddleSeg repo root)",
    pathField: "paddleSegPath",
    requiredFiles: ["paddleseg", "tools/train.py", "tools/val.py"],
    scripts: { train: "tools/train.py", eval: "tools/val.py", infer: "tools/predict.py", export: "tools/export.py" },
    cliStyle: "config-flags",
    saveDirOnCli: true,
    checkpointLayout: "nested",
    weightFile: "model.pdparams",
    datasetFormat: "PaddleSeg",
    stepUnit: "iter",
  },
  TorchDet: {
    label: "PyTorch Detection",
    family: "torch",
    taskKind: "detection",
    defaultTask: "detection",
    pythonModule: "torch",
    installHint:
      "Install PyTorch in the environment mapped to this framework, e.g. " +
      "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118",
    pathField: "torchPath",
    requiredFiles: ["torchtrain", "tools/train.py", "tools/val.py"],
    scripts: { train: "tools/train.py", eval: "tools/val.py", infer: "tools/predict.py", export: "tools/export.py" },
    cliStyle: "config-flags",
    saveDirOnCli: true,
    checkpointLayout: "nested",
    weightFile: TORCH_WEIGHT_FILE,
    datasetFormat: "COCO",
    stepUnit: "epoch",
  },
  TorchSeg: {
    label: "PyTorch Segmentation",
    family: "torch",
    taskKind: "segmentation",
    defaultTask: "semantic_segmentation",
    pythonModule: "torch",
    installHint:
      "Install PyTorch in the environment mapped to this framework, e.g. " +
      "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118",
    pathField: "torchPath",
    requiredFiles: ["torchtrain", "tools/train.py", "tools/val.py"],
    scripts: { train: "tools/train.py", eval: "tools/val.py", infer: "tools/predict.py", export: "tools/export.py" },
    cliStyle: "config-flags",
    saveDirOnCli: true,
    checkpointLayout: "nested",
    weightFile: TORCH_WEIGHT_FILE,
    datasetFormat: "PaddleSeg",
    stepUnit: "iter",
  },
};

/** Normalize an arbitrary value to a known framework, defaulting to PaddleDetection. */
export function normalizeFramework(framework: string | null | undefined): Framework {
  return framework && framework in FRAMEWORK_META ? (framework as Framework) : "PaddleDetection";
}

export function frameworkMeta(framework: string | null | undefined): FrameworkMeta {
  return FRAMEWORK_META[normalizeFramework(framework)];
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
  return systemConfig[frameworkMeta(framework).pathField];
}

export function isSegmentation(framework: string | null | undefined): boolean {
  return frameworkMeta(framework).taskKind === "segmentation";
}

export function isClassification(framework: string | null | undefined): boolean {
  return frameworkMeta(framework).taskKind === "classification";
}

export function isDetection(framework: string | null | undefined): boolean {
  return frameworkMeta(framework).taskKind === "detection";
}

export function isTorch(framework: string | null | undefined): boolean {
  return frameworkMeta(framework).family === "torch";
}

export function isPaddle(framework: string | null | undefined): boolean {
  return frameworkMeta(framework).family === "paddle";
}

/** `project.task` implied by a framework, honouring an existing compatible task. */
export function resolveTask(
  framework: string | null | undefined,
  requestedTask?: string | null
): string {
  const meta = frameworkMeta(framework);
  // Detection frameworks additionally support instance segmentation; every other
  // framework has exactly one task, so a stale request must not survive a
  // framework switch.
  if (meta.taskKind === "detection" && requestedTask === "instance_segmentation") {
    return "instance_segmentation";
  }
  return meta.defaultTask;
}

/**
 * Resolve the Python interpreter for a job.
 *
 * Two layers, because the two runtimes cannot share an environment: a Paddle env
 * has no `torch` and vice versa, while the original per-GPU mapping only ever
 * pointed at Paddle envs.
 *
 *   1. `frameworkPythonMappings`: `{"TorchSeg": "D:/pythonEnvs/torchEnv/Scripts/python.exe"}`.
 *      An entry may be keyed by framework name or by `"<framework>:<gpuId>"` when
 *      a specific GPU needs its own build.
 *   2. `gpuPythonMappings`: the pre-existing `{"0": "...python.exe"}` map.
 *
 * Returns `null` when nothing is configured, which callers surface as an
 * actionable error rather than silently running the wrong `python`.
 */
export function resolvePythonPath(
  framework: string | null | undefined,
  gpuId: number | string,
  systemConfig: { frameworkPythonMappings?: string | null; gpuPythonMappings?: string | null } | null | undefined
): { pythonPath: string | null; source: string } {
  if (!systemConfig) return { pythonPath: null, source: "none" };
  const name = normalizeFramework(framework);
  const gpu = String(gpuId ?? "0");

  const frameworkMap = parseMapping(systemConfig.frameworkPythonMappings);
  const perGpuKey = `${name}:${gpu}`;
  if (frameworkMap[perGpuKey]) {
    return { pythonPath: frameworkMap[perGpuKey], source: `frameworkPythonMappings[${perGpuKey}]` };
  }
  if (frameworkMap[name]) {
    return { pythonPath: frameworkMap[name], source: `frameworkPythonMappings[${name}]` };
  }

  const gpuMap = parseMapping(systemConfig.gpuPythonMappings);
  if (gpuMap[gpu]) {
    return { pythonPath: gpuMap[gpu], source: `gpuPythonMappings[${gpu}]` };
  }
  return { pythonPath: null, source: "none" };
}

function parseMapping(raw: string | null | undefined): Record<string, string> {
  if (!raw || !raw.trim()) return {};
  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const out: Record<string, string> = {};
    for (const [key, value] of Object.entries(parsed ?? {})) {
      if (typeof value === "string" && value.trim()) out[key] = value.trim();
    }
    return out;
  } catch {
    // A hand-edited malformed mapping must not take the whole app down; the
    // caller's "no interpreter configured" error is the right outcome.
    return {};
  }
}

/**
 * Distributed-launch prefix for a training command.
 *
 * Paddle jobs go through `paddle.distributed.launch --gpus`, which is how this
 * platform has always invoked them. torchtrain is single-process and reads the
 * GPU from `CUDA_VISIBLE_DEVICES` (set by the runner), so it needs no launcher;
 * wrapping it in one would fail, since `paddle` is not installed in a torch env.
 */
export function launchPrefix(framework: string | null | undefined, gpuIds: string): string {
  return isTorch(framework) ? "python" : `python -m paddle.distributed.launch --gpus ${gpuIds}`;
}
