/**
 * End-to-end check for the TorchAnomaly config chain.
 *
 *   npx tsx scripts/check-anomaly-config-chain.ts
 *
 * Verifies, without needing anomalib or a GPU, that:
 *
 *   1. the training and model generators emit parseable YAML and round-trip
 *      through their parsers (so editing a saved config does not silently reset
 *      fields to defaults);
 *   2. the three configs deep-merge into a valid anomalib document, with the
 *      dataset's `data:` block and the training config's refinements of it both
 *      surviving, and `autotrain:` carrying contributions from two files;
 *   3. the generated commands match the checkpoint path the platform resolves;
 *   4. the log parser reads the exact lines `torchtrain/torchtrain/ad/logger.py`
 *      prints, including a metric it has no column for.
 *
 * It writes the merged config to the OS temp directory and prints the path, so
 * the Python half can be checked too:
 *
 *   python -c "import sys; sys.path.insert(0,'.'); \
 *     from torchtrain import config as c; from torchtrain.ad import config as a; \
 *     cfg=c.load_config(r'<path>'); print(c.detect_task(cfg)); \
 *     a.validate(a.split_platform_block(cfg)[0])"
 */
import { parseDocument } from 'yaml';
import { mergeYamlConfigs } from '../src/lib/yaml-merge';
import {
  defaultTrainingParams,
  generateTrainingYaml,
  parseTrainingParams,
  totalStepsFor,
  trainingParamsToColumns,
  countsIterations,
} from '../src/lib/training-yaml';
import {
  defaultModelParams,
  generateModelYaml,
  parseModelParams,
  validateModelParams,
  ANOMALY_PRESETS,
} from '../src/lib/model-yaml';
import { bestWeightsPath, buildTrainCommand, buildEvalCommand, buildInferCommand } from '../src/lib/job-commands';
import { frameworkMeta, tracksIterations } from '../src/lib/frameworks';
import { createParserState, feed, flush } from '../src/lib/log-parsers';
import fs from 'fs';
import os from 'os';
import path from 'path';

const failures: string[] = [];
const check = (name: string, got: unknown, want: unknown) => {
  const a = JSON.stringify(got);
  const b = JSON.stringify(want);
  if (a !== b) failures.push(`${name}: got ${a}, want ${b}`);
};
const ok = (name: string, cond: boolean) => {
  if (!cond) failures.push(name);
};

// --- 1. training config ----------------------------------------------------
const tParams = defaultTrainingParams('TorchAnomaly');
tParams.adTileEnabled = true;
tParams.adTileSize = 512;
tParams.adTileStride = 256;
tParams.iters = 4000;
tParams.adValInterval = 250;
const trainingYaml = generateTrainingYaml('TorchAnomaly', tParams, 'slider quick');
ok('training yaml parses', parseDocument(trainingYaml).errors.length === 0);

const reparsed = parseTrainingParams('TorchAnomaly', trainingYaml);
check('round-trip iters', reparsed.iters, 4000);
check('round-trip val interval', reparsed.adValInterval, 250);
check('round-trip tiling', reparsed.adTileEnabled, true);
check('round-trip tile size', reparsed.adTileSize, 512);
check('round-trip stride', reparsed.adTileStride, 256);
check('round-trip batch', reparsed.trainBatchSize, tParams.trainBatchSize);
check('counts iterations', countsIterations('TorchAnomaly'), true);
check('tracks iterations', tracksIterations('TorchAnomaly'), true);
check('total steps', totalStepsFor('TorchAnomaly', tParams), 4000);
check('columns.epoch = max_steps', trainingParamsToColumns('TorchAnomaly', tParams).epoch, 4000);
check('columns.snapshotEpoch = val interval', trainingParamsToColumns('TorchAnomaly', tParams).snapshotEpoch, 250);

// --- 2. model config -------------------------------------------------------
const mParams = defaultModelParams('TorchAnomaly');
check('default algorithm', mParams.architecture, 'Patchcore');
const modelYaml = generateModelYaml('TorchAnomaly', mParams, 'patchcore 256');
ok('model yaml parses', parseDocument(modelYaml).errors.length === 0);
const mReparsed = parseModelParams('TorchAnomaly', modelYaml);
check('round-trip architecture', mReparsed.architecture, 'Patchcore');
check('round-trip backbone', mReparsed.backbone, 'wide_resnet50_2');
check('round-trip image size', [mReparsed.adImageWidth, mReparsed.adImageHeight], [256, 256]);
ok('no validation errors', validateModelParams('TorchAnomaly', mParams).every((i) => i.level !== 'error'));

// EfficientAD must warn about tiling, batch size and the download.
const eff = { ...defaultModelParams('TorchAnomaly'), architecture: 'EfficientAd', backbone: '' };
const effIssues = validateModelParams('TorchAnomaly', eff).map((i) => i.message).join(' | ');
ok('EfficientAd warns about batch size', /train_batch_size = 1/.test(effIssues));
ok('EfficientAd warns about tiling', /does not support input tiling/.test(effIssues));
ok('EfficientAd warns about download', /First run downloads/.test(effIssues));
ok('EfficientAd class path', ANOMALY_PRESETS.EfficientAd.classPath === 'anomalib.models.EfficientAd');

// Centre crop larger than the input is a hard error.
const badCrop = { ...defaultModelParams('TorchAnomaly'), adCenterCrop: 512 };
ok('crop > input is an error', validateModelParams('TorchAnomaly', badCrop).some((i) => i.level === 'error'));

// --- 3. dataset config (same shape the API route emits) --------------------
const datasetYaml = `# Dataset configuration for slider_v1
data:
  class_path: anomalib.data.Folder
  init_args:
    name: slider_v1
    root: ${process.cwd().replace(/\\/g, '/')}
    normal_dir: train_good
    normal_test_dir: test_good
    abnormal_dir: test_ng
    mask_dir: test_ng_mask
    extensions: [".png",".jpg",".jpeg",".bmp",".tif",".tiff"]
    test_split_mode: from_dir
    val_split_mode: from_test
    val_split_ratio: 0.5
    seed: 42
`;

// --- 4. merge --------------------------------------------------------------
const merged = mergeYamlConfigs([
  { label: 'dataset', content: datasetYaml },
  { label: 'training', content: trainingYaml },
  { label: 'model', content: modelYaml },
]);
check('deep merge succeeded', merged.merged, true);
check('no merge warnings', merged.warnings, []);

const doc = parseDocument(merged.yaml);
ok('merged parses', doc.errors.length === 0);
const js = doc.toJS() as any;
check('model.class_path', js.model.class_path, 'anomalib.models.Patchcore');
check('data.class_path', js.data.class_path, 'anomalib.data.Folder');
// The training config refines the datamodule the dataset config declared: both
// must survive the merge.
check('data root survived', typeof js.data.init_args.root, 'string');
check('data batch from training', js.data.init_args.train_batch_size, tParams.trainBatchSize);
check('trainer.max_steps', js.trainer.max_steps, 4000);
check('progress bar disabled', js.trainer.enable_progress_bar, false);
check('tiler callback present', js.trainer.callbacks.length, 1);
ok('tiler class path', String(js.trainer.callbacks[0].class_path).includes('TilerConfigurationCallback'));
// autotrain must carry BOTH the training and model contributions after merging.
check('autotrain.val_interval', js.autotrain.val_interval, 250);
check('autotrain.image_size', js.autotrain.image_size, [256, 256]);
check('autotrain.best_metric', js.autotrain.best_metric, 'image_AUROC');

// --- 5. commands -----------------------------------------------------------
const cmd = buildTrainCommand({
  framework: 'TorchAnomaly',
  configPath: 'D:/jobs/j1/merged.yml',
  saveDir: 'D:/jobs/j1',
  gpuIds: '0',
});
ok('train command has no paddle launcher', !cmd.includes('paddle.distributed.launch'));
ok('train command passes save_dir', cmd.includes('--save_dir "D:/jobs/j1"'));
ok('train command uses tools/train.py', cmd.includes('tools/train.py'));
check('best weights path', bestWeightsPath('TorchAnomaly', 'D:/jobs/j1'), 'D:/jobs/j1/best_model/model.ckpt');
ok(
  'eval command',
  buildEvalCommand({ framework: 'TorchAnomaly', configPath: 'c.yml', weightsPath: 'w.ckpt' }).includes('tools/val.py'),
);
ok(
  'infer command',
  buildInferCommand({
    framework: 'TorchAnomaly', configPath: 'c.yml', weightsPath: 'w.ckpt',
    inputPath: 'in', outputPath: 'out',
  }).includes('--image_path'),
);
check('weight file', frameworkMeta('TorchAnomaly').weightFile, 'model.ckpt');
check('dataset format', frameworkMeta('TorchAnomaly').datasetFormat, 'AnomalyFolder');

// --- 6. log parsing (the adapter's exact output) ---------------------------
const state = createParserState('TorchAnomaly');
const stdout = [
  '[2026/08/20 10:21:03] INFO: [TRAIN] epoch: 3, iter: 600/4000, loss: 0.1837, lr: 0.000100, batch_cost: 0.0921, reader_cost: 0.01130, ips: 86.8000 samples/sec, max_mem_reserved: 2048 MB, max_mem_allocated: 1902 MB | ETA 00:11:22',
  '[2026/08/20 10:22:31] INFO: [EVAL] #Images: 40 image_auroc: 0.9812 image_f1: 0.9231 pixel_auroc: 0.9633 pixel_f1: 0.5412 threshold: 12.3456 aupro: 0.8800',
  '[2026/08/20 10:22:31] INFO: [EVAL] The model with the best validation image_auroc (0.9812) was saved at iter 600.',
  '',
].join('\n');
const rows = [...feed(state, stdout), ...flush(state)];
check('rows parsed', rows.length, 2);
const [train, evalRow] = rows;
check('train iteration', train.iteration, 600);
check('train total', train.totalIter, 4000);
check('train loss', train.loss, 0.1837);
check('train lr', train.learningRate, 0.0001);
check('train ips', train.ips, 86.8);
check('eval kind', evalRow.kind, 'eval');
check('eval imageAuroc', evalRow.imageAuroc, 0.9812);
check('eval imageF1', evalRow.imageF1, 0.9231);
check('eval pixelAuroc', evalRow.pixelAuroc, 0.9633);
check('eval threshold', evalRow.threshold, 12.3456);
// An unanticipated metric must survive rather than be dropped.
check('eval extra metric', evalRow.extraMetrics, { aupro: 0.88 });
check('eval bestMetric', evalRow.bestMetric, 0.9812);
check('eval bestIter', evalRow.bestIter, 600);
check('eval bestMetricName', evalRow.bestMetricName, 'image_auroc');
// The eval row must inherit the train x-axis so the chart lines up.
check('eval iteration', evalRow.iteration, 600);

// A PatchCore-style flat loss line must still produce a progress row.
const state2 = createParserState('TorchAnomaly');
const patchcoreRows = feed(
  state2,
  '[2026/08/20 10:00:00] INFO: [TRAIN] epoch: 1, iter: 5/25, loss: 0.0000, lr: 0.000000, batch_cost: 0.5000, reader_cost: 0.01000, ips: 16.0000 samples/sec, max_mem_reserved: 900 MB, max_mem_allocated: 800 MB | ETA 00:00:10\n',
);
check('patchcore rows', patchcoreRows.length, 1);
check('patchcore progress', [patchcoreRows[0].iteration, patchcoreRows[0].totalIter], [5, 25]);

// --- 7. hand the merged config to the python adapter ----------------------
// Written outside the repo so a run never leaves anything behind to commit.
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'autotrain-anomaly-'));
const mergedPath = path.join(tmpDir, 'merged.yml');
fs.writeFileSync(mergedPath, merged.yaml, 'utf-8');
console.log(failures.length ? 'FAILURES:' : 'ALL CHECKS PASSED');
for (const f of failures) console.log('  -', f);
console.log('merged config written to: ' + mergedPath);
process.exit(failures.length ? 1 : 0);
