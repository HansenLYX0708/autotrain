import * as fs from "fs";
import * as path from "path";
import { parseDocument } from "yaml";
import { isAnomaly, isClassification, isSegmentation, isTorch, isLocalization, frameworkMeta } from "./frameworks";

/** Check the merged YAML that will run, never the cached dataset image count. */
export function validateTrainingValidation(yaml: string, framework: string, workDir: string): void {
  if (isAnomaly(framework)) return; // anomalib constructs its own validation split
  const doc = parseDocument(yaml, { logLevel: "silent" });
  if (doc.errors.length) throw new Error("Cannot validate the validation set: job YAML is invalid.");
  if (isLocalization(framework)) {
    if (doc.getIn(['localization', 'task']) !== frameworkMeta(framework).localizationTask) {
      throw new Error("Dataset/model task differs from the project's OBB/Pose task.");
    }
    const dataPath = doc.getIn(['localization', 'data']);
    if (typeof dataPath !== 'string' || !path.isAbsolute(dataPath)) throw new Error('localization.data must be an absolute data.yaml path.');
    const data = parseDocument(fs.readFileSync(dataPath, 'utf8'), { logLevel: 'silent' });
    if (data.errors.length) throw new Error('Invalid localization data.yaml.');
    const val = data.get('val');
    const root = data.get('path');
    if (typeof val !== 'string') throw new Error('Localization requires a validation image directory.');
    const folder = path.resolve(path.dirname(dataPath), typeof root === 'string' ? root : '', val);
    if (!fs.readdirSync(folder).some(name => /\.(jpg|jpeg|png|bmp|tif|tiff|webp)$/i.test(name))) throw new Error('Localization validation split is empty.');
    return;
  }
  if (isClassification(framework)) {
    if (!doc.hasIn(["DataLoader", "Eval"])) throw new Error("Training validation requires DataLoader.Eval in the job YAML.");
    return;
  }
  const seg = isSegmentation(framework);
  const block = seg ? "val_dataset" : "EvalDataset";
  const anno = doc.getIn([block, seg ? "val_path" : "anno_path"]);
  if (typeof anno !== "string" || !anno.trim()) {
    throw new Error(`Training validation requires ${block}.${seg ? "val_path" : "anno_path"}. Import/parse the validation set or turn off validation.`);
  }
  const root = doc.getIn([block, seg ? "dataset_root" : "dataset_dir"]);
  // PaddleSeg opens val_path directly; TorchSeg and COCO resolve it under root.
  const file = seg && !isTorch(framework)
    ? path.resolve(workDir, anno)
    : path.resolve(workDir, typeof root === "string" ? root : "", anno);
  let source: string;
  try {
    source = fs.readFileSync(file, "utf8");
  } catch {
    throw new Error(`Cannot read validation annotations: ${file}. Check the dataset path or turn off validation.`);
  }
  const count = seg
    ? source.split(/\r?\n/).filter(line => line.trim() && !line.trim().startsWith("#")).length
    : (() => {
        try {
          const coco = JSON.parse(source);
          if (!Array.isArray(coco.images) || !Array.isArray(coco.annotations) || !Array.isArray(coco.categories)) throw new Error();
          return coco.images.length;
        } catch {
          throw new Error(`Invalid COCO validation annotations: ${file}`);
        }
      })();
  if (!count) throw new Error(`Validation set is empty: ${file}. Add validation images or turn off validation.`);
}
