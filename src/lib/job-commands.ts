/**
 * Single source of truth for the shell commands the platform runs.
 *
 * Previously the command strings were built in three places — the job creation
 * route, the job dialog's preview, and the validation page's preview — and had
 * already drifted (the preview showed flags the runner did not pass). Everything
 * now derives from `FRAMEWORK_META`, so a framework's CLI dialect is declared
 * once and the string a user sees is the string that runs.
 *
 * Deliberately filesystem-free (no `node:path`) so client components can import
 * it. Callers pass absolute paths they have already resolved; the small amount of
 * path building done here uses forward slashes, which Python accepts on Windows.
 */

import { frameworkMeta, launchPrefix, type Framework } from "./frameworks";

/** Quote a path for a shell that is invoked with `shell: true`. */
function q(value: string): string {
  return `"${String(value).replace(/"/g, "")}"`;
}

/** Join path segments with forward slashes, collapsing duplicates. */
export function joinPath(...segments: Array<string | null | undefined>): string {
  return segments
    .filter((s): s is string => typeof s === "string" && s.length > 0)
    .map((s, i) => (i === 0 ? s.replace(/[\\/]+$/, "") : s.replace(/^[\\/]+|[\\/]+$/g, "")))
    .join("/");
}

export interface TrainCommandOptions {
  framework: string | null | undefined;
  /** Absolute path to the merged job config. */
  configPath: string;
  /** Absolute output directory (`--save_dir` for config-flags frameworks). */
  saveDir: string;
  /** Relative output dir used by the `dash-c` frameworks' YAML/vdl paths. */
  outputDir?: string;
  gpuIds: string;
  useAmp?: boolean;
  useVdl?: boolean;
  doEval?: boolean;
}

/**
 * Training command.
 *
 * Note the launcher difference: Paddle jobs go through
 * `paddle.distributed.launch --gpus`, while torchtrain is single-process and
 * reads its GPU from `CUDA_VISIBLE_DEVICES` (which the runner sets). Wrapping a
 * torch job in the Paddle launcher would fail immediately, since `paddle` is not
 * installed in a torch environment.
 */
export function buildTrainCommand(options: TrainCommandOptions): string {
  const meta = frameworkMeta(options.framework);
  const parts = [launchPrefix(options.framework, options.gpuIds), meta.scripts.train];

  if (meta.cliStyle === "config-flags") {
    parts.push("--config", q(options.configPath));
    // PaddleSeg only evaluates when asked; torchtrain follows the same flag.
    if (options.doEval !== false) parts.push("--do_eval");
    if (meta.saveDirOnCli) parts.push("--save_dir", q(options.saveDir));
    if (options.useAmp && meta.family === "torch") parts.push("--amp");
    if (options.useVdl) parts.push("--use_vdl");
  } else {
    parts.push("-c", q(options.configPath));
    if (options.useAmp) parts.push("--amp");
    if (options.useVdl) {
      parts.push(`--use_vdl=true`, `--vdl_log_dir=${options.outputDir ?? options.saveDir}/vdl`);
    }
  }
  return parts.join(" ");
}

export interface EvalCommandOptions {
  framework: string | null | undefined;
  configPath: string;
  weightsPath: string;
  /** Prefix to use instead of bare `python` (e.g. an absolute interpreter). */
  python?: string;
}

export function buildEvalCommand(options: EvalCommandOptions): string {
  const meta = frameworkMeta(options.framework);
  const python = options.python ?? "python";
  if (meta.cliStyle === "config-flags") {
    return `${python} ${meta.scripts.eval} --config ${q(options.configPath)} --model_path ${q(options.weightsPath)}`;
  }
  return `${python} ${meta.scripts.eval} -c ${q(options.configPath)} -o weights=${options.weightsPath}`;
}

export interface InferCommandOptions extends EvalCommandOptions {
  /** Image file or directory. */
  inputPath: string;
  outputPath: string;
  /** True when `inputPath` is a single image (changes PaddleDetection's flag). */
  inputIsFile?: boolean;
}

export function buildInferCommand(options: InferCommandOptions): string {
  const meta = frameworkMeta(options.framework);
  const python = options.python ?? "python";
  if (meta.cliStyle === "config-flags") {
    return (
      `${python} ${meta.scripts.infer} --config ${q(options.configPath)} ` +
      `--model_path ${q(options.weightsPath)} --image_path ${q(options.inputPath)} ` +
      `--save_dir ${q(options.outputPath)}`
    );
  }
  // PaddleDetection distinguishes a single image from a directory by flag name.
  const inputFlag = options.inputIsFile ? "--infer_img" : "--infer_dir";
  return (
    `${python} ${meta.scripts.infer} -c ${q(options.configPath)} -o weights=${options.weightsPath} ` +
    `${inputFlag}=${options.inputPath} --output_dir=${options.outputPath}`
  );
}

export interface ExportCommandOptions extends EvalCommandOptions {
  outputDir: string;
  /** torchtrain only: `torchscript` (default) or `onnx`. */
  format?: string;
}

/** Export command, or `null` when the framework has no export entrypoint. */
export function buildExportCommand(options: ExportCommandOptions): string | null {
  const meta = frameworkMeta(options.framework);
  if (!meta.scripts.export) return null;
  const python = options.python ?? "python";
  if (meta.cliStyle === "config-flags") {
    const format = meta.family === "torch" && options.format ? ` --format ${options.format}` : "";
    return (
      `${python} ${meta.scripts.export} --config ${q(options.configPath)} ` +
      `--model_path ${q(options.weightsPath)} --save_dir ${q(options.outputDir)}${format}`
    );
  }
  return (
    `${python} ${meta.scripts.export} -c ${q(options.configPath)} ` +
    `-o weights=${options.weightsPath} --output_dir ${q(options.outputDir)}`
  );
}

/**
 * Path of the "best" checkpoint a finished job produced.
 *
 * `nested` frameworks (PaddleSeg, TorchSeg, TorchDet) write
 * `<save_dir>/best_model/<weightFile>`; `flat` ones (PaddleDetection,
 * PaddleClas) write `<save_dir>/model_final.pdparams`.
 */
export function bestWeightsPath(framework: string | null | undefined, saveDir: string): string {
  const meta = frameworkMeta(framework);
  if (meta.checkpointLayout === "nested") {
    return joinPath(saveDir, "best_model", meta.weightFile);
  }
  return joinPath(saveDir, `model_final${meta.weightFile}`);
}

/** Human-readable framework label, for dropdowns and status text. */
export function frameworkLabel(framework: string | null | undefined): string {
  return frameworkMeta(framework).label;
}

export type { Framework };
