/** Run with: npx tsx scripts/check-dataset-config-chain.ts */
import assert from 'node:assert/strict';
import { parseDocument } from 'yaml';
import { applyDatasetClasses } from '../src/lib/dataset-classes';
import { defaultModelParams, generateModelYaml } from '../src/lib/model-yaml';
import { mergeYamlConfigs } from '../src/lib/yaml-merge';
import { appendCategories, assessLabelmeMerge, convertLabelmeShapes } from '../src/lib/labelme-coco-merge';
import type { ConfigFramework } from '../src/lib/training-yaml';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { resolveCocoValidationPaths } from '../src/lib/coco-annotations';
import { buildTrainCommand } from '../src/lib/job-commands';
import { validateTrainingValidation } from '../src/lib/training-validation';

for (const framework of ['PaddleDetection', 'TorchDet', 'PaddleSeg', 'TorchSeg', 'PaddleClas'] as ConfigFramework[]) {
  const seg = framework.endsWith('Seg');
  const dataYaml = seg ? 'train_dataset: {num_classes: 7}\nval_dataset: {num_classes: 7}\n' : 'num_classes: 7\n';
  const generated = generateModelYaml(framework, defaultModelParams(framework));
  assert(!/^\s*(num_classes|class_num):/m.test(generated), `${framework}: model must defer classes to dataset`);
  const modelYaml = seg
    ? 'model: {type: UNet, num_classes: 999}\n'
    : framework === 'PaddleClas' ? 'Arch: {name: ResNet50, class_num: 999}\n' : 'PPYOLOEHead: {num_classes: 999}\n';
  const merged = mergeYamlConfigs([
    { label: 'data', content: dataYaml },
    { label: 'training', content: 'LearningRate:\n  schedulers: [!CosineDecay {max_epochs: 80}]\nsize: &size [640, 640]\nresize: *size\n' },
    { label: 'model', content: modelYaml },
  ]).yaml;
  // A stale column must never beat a number explicitly declared in the dataset YAML.
  const bound = applyDatasetClasses(merged, framework, { yamlConfig: dataYaml, numClasses: 1 });
  const doc = parseDocument(bound, { logLevel: 'silent' });
  assert.equal(doc.errors.length, 0);
  assert(bound.includes('!CosineDecay'));
  assert.deepEqual(doc.toJS().resize, [640, 640]);
  assert.equal(doc.getIn(seg ? ['model', 'num_classes'] : framework === 'PaddleClas' ? ['Arch', 'class_num'] : ['num_classes']), 7);
  if (!seg && framework !== 'PaddleClas') assert.equal(doc.getIn(['PPYOLOEHead', 'num_classes']), 7);
}
const anomaly = 'model: {class_path: anomalib.models.Patchcore}\n';
assert.equal(applyDatasetClasses(anomaly, 'TorchAnomaly', { yamlConfig: null, numClasses: 2 }), anomaly);
assert.throws(() => applyDatasetClasses('model: {}', 'PaddleSeg', {
  yamlConfig: 'train_dataset: {num_classes: 2}\nval_dataset: {num_classes: 3}', numClasses: 2,
}));

const before = { trainImages: 10, valImages: 4, trainAnnotations: 12, valAnnotations: 5 };
const selected = assessLabelmeMerge(['old', 'newA', 'newB'], ['old'], ['old'], 10, 0.7, 0.2, before, ['newA']);
assert.equal(selected.canMerge, true);
assert.deepEqual(selected.planned.addedCategories, ['newA']);
assert.deepEqual(selected.planned.droppedCategories, ['newB']);
assert.equal(selected.planned.trainAdd, 7);
assert.equal(selected.planned.valAdd, 2);
assert.equal(selected.planned.skipped, 1);
assert.equal(assessLabelmeMerge(['newA'], ['old'], ['old'], 3, 1, 0, before).canMerge, true);
assert.equal(assessLabelmeMerge(['old'], ['old'], [], 3, 1, 0, before).canMerge, false);
const docs = [{ categories: [{ id: 7, name: 'old' }] }, { categories: [{ id: 7, name: 'old' }] }];
appendCategories(docs, selected.planned.addedCategories);
assert.deepEqual(docs[0].categories, docs[1].categories);
assert.equal(docs[0].categories[1].id, 8);
const shapes = ['old', 'newA', 'newB'].map(label => ({ label, shape_type: 'rectangle', points: [[0, 0], [10, 20]] }));
const annotations = convertLabelmeShapes(shapes, new Map(docs[0].categories.map(c => [c.name, c.id])), 100, 200);
assert.equal(annotations.length, 2);
assert.deepEqual(annotations.map(a => a.category_id), [7, 8]);
assert(annotations.every(a => a.area === 200 && a.image_id === 100));
console.log('PASS: dataset class binding for all frameworks, YAML tags/aliases, and selective Labelme category merge.');

const root = mkdtempSync(join(tmpdir(), 'autotrain-val-'));
try {
  mkdirSync(join(root, 'data', 'annotations'), { recursive: true });
  mkdirSync(join(root, 'data', 'val'));
  const val = join(root, 'data', 'annotations', 'instance_val.json');
  writeFileSync(val, JSON.stringify({ images: [{ id: 1 }], annotations: [], categories: [{ id: 1, name: 'defect' }] }));
  assert.deepEqual(resolveCocoValidationPaths(root, {}), {
    evalAnnoPath: 'data/annotations/instance_val.json', evalImagePath: 'data/val',
  });
  assert.equal(resolveCocoValidationPaths(root, { evalAnnoPath: 'missing.json' }).evalAnnoPath, 'missing.json');
  assert.equal(resolveCocoValidationPaths(root, { evalAnnoPath: val }).evalAnnoPath, val);
  const yaml = `EvalDataset: !COCODataSet\n  dataset_dir: ${JSON.stringify(root)}\n  anno_path: data/annotations/instance_val.json\n`;
  validateTrainingValidation(yaml, 'PaddleDetection', root);
  validateTrainingValidation(yaml, 'TorchDet', root);
  writeFileSync(val, JSON.stringify({ images: [], annotations: [], categories: [] }));
  assert.throws(() => validateTrainingValidation(yaml, 'PaddleDetection', root), /empty/);
  writeFileSync(val, '{}');
  assert.throws(() => validateTrainingValidation(yaml, 'PaddleDetection', root), /Invalid COCO/);
  rmSync(val);
  assert.throws(() => validateTrainingValidation(yaml, 'PaddleDetection', root), /Cannot read/);
  assert.throws(() => validateTrainingValidation('TrainDataset: {}', 'TorchDet', root), /EvalDataset/);
  assert.equal(resolveCocoValidationPaths(root, {}).evalAnnoPath, null);
  writeFileSync(join(root, 'data', 'val.txt'), 'image.png mask.png\n');
  validateTrainingValidation(`val_dataset: {dataset_root: data, val_path: val.txt}`, 'TorchSeg', root);
  validateTrainingValidation(`val_dataset: {val_path: data/val.txt}`, 'PaddleSeg', root);
  for (const framework of ['PaddleDetection', 'TorchDet', 'PaddleSeg', 'TorchSeg', 'PaddleClas']) {
    const options = { framework, configPath: 'C:/path with spaces/config.yml', saveDir: 'out', gpuIds: '0' };
    const enabled = buildTrainCommand({ ...options, doEval: true });
    const disabled = buildTrainCommand({ ...options, doEval: false });
    if (framework === 'PaddleClas') {
      assert(enabled.includes('Global.eval_during_train=true'));
      assert(disabled.includes('Global.eval_during_train=false'));
    } else {
      const flag = framework === 'PaddleDetection' ? '--eval' : '--do_eval';
      assert(enabled.split(' ').includes(flag), enabled);
      assert(!disabled.split(' ').includes(flag), disabled);
    }
    assert(enabled.includes('"C:/path with spaces/config.yml"'));
  }
  console.log('PASS: missing val-path recovery, explicit paths, empty/corrupt/missing validation data, and validation command toggles.');
} finally {
  rmSync(root, { recursive: true, force: true });
}
