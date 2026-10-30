import { isMap, isScalar, parseDocument, visit } from 'yaml';
import { isAnomaly, isClassification, isSegmentation } from '@/lib/frameworks';

/**
 * Bind model output dimensions at job assembly time. Templates may have been
 * authored against another dataset; class counts always come from this dataset.
 * Work on the AST to preserve Paddle custom tags and YAML aliases.
 */
export function applyDatasetClasses(
  yaml: string,
  framework: string,
  dataset: { yamlConfig: string | null; numClasses: number },
): string {
  if (isAnomaly(framework)) return yaml;
  const source = parseDocument(dataset.yamlConfig || '', { logLevel: 'silent' });
  if (source.errors.length) throw new Error(`Invalid dataset YAML: ${source.errors[0].message}`);
  const declared = isSegmentation(framework)
    ? [source.getIn(['train_dataset', 'num_classes']), source.getIn(['val_dataset', 'num_classes'])]
    : [source.get('num_classes'), ...(isClassification(framework) ? [source.getIn(['Arch', 'class_num'])] : [])];
  const values = declared.filter(value => value !== undefined);
  const numClasses = Number(values[0] ?? dataset.numClasses);
  if (!Number.isInteger(numClasses) || numClasses < 1 || values.some(value => Number(value) !== numClasses)) {
    throw new Error('Dataset class counts must be positive integers and agree between train and val.');
  }
  const doc = parseDocument(yaml, { logLevel: 'silent' });
  if (doc.errors.length || !isMap(doc.contents)) throw new Error('Cannot set dataset classes in invalid job YAML.');
  visit(doc, {
    Pair(_key, pair) {
      if (isScalar(pair.key) && ['num_classes', 'class_num'].includes(String(pair.key.value))) {
        pair.value = doc.createNode(numClasses);
      }
    },
  });
  if (isSegmentation(framework)) doc.setIn(['model', 'num_classes'], numClasses);
  else if (isClassification(framework)) doc.setIn(['Arch', 'class_num'], numClasses);
  else doc.set('num_classes', numClasses);
  return doc.toString({ lineWidth: 0 });
}
