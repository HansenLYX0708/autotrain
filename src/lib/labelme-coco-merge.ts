export interface MergeCounts {
  trainImages: number;
  valImages: number;
  trainAnnotations: number;
  valAnnotations: number;
}

export interface MergeAssessment {
  canMerge: boolean;
  cocoCategories: string[];
  differences: {
    matched: string[];
    onlyInLabelme: string[];
    onlyInCoco: string[];
    missingFromTrain: string[];
    missingFromVal: string[];
  };
  planned: {
    trainAdd: number;
    valAdd: number;
    skipped: number;
    trainImages: number;
    valImages: number;
    /** New (Labelme-only) categories that will be appended to the COCO dataset. */
    addedCategories: string[];
    /** New categories the user chose not to import; their shapes are dropped. */
    droppedCategories: string[];
    /** Source shapes (over all valid samples) that belong to dropped categories. */
    droppedAnnotations: number;
  };
}

export interface MergeShape {
  label: string;
  points: number[][];
  shape_type?: string;
}

export interface MergeAnnotation {
  id: number;
  image_id: number;
  category_id: number;
  bbox: number[];
  area: number;
  iscrowd: number;
  segmentation: number[][];
}

function unique(values: string[]): string[] {
  return [...new Set(values.map((value) => value.trim()).filter(Boolean))].sort();
}

function difference(left: string[], right: string[]): string[] {
  const rightSet = new Set(right);
  return left.filter((value) => !rightSet.has(value));
}

/**
 * Decide whether a Labelme folder can be merged into a COCO dataset.
 *
 * Categories that exist only in the Labelme data are no longer a hard stop: each
 * one is either listed in `addCategories` (appended to both COCO splits) or
 * dropped (its shapes are skipped). What still blocks is a category the COCO
 * dataset knows but that is missing from one of its splits — the train and val
 * files would then disagree on the class list, which is a structural problem in
 * the target rather than a choice about the new data.
 */
export function assessLabelmeMerge(
  sourceCategoryNames: string[],
  trainCategoryNames: string[],
  valCategoryNames: string[],
  validSamples: number,
  trainRatio: number,
  valRatio: number,
  before: MergeCounts,
  addCategories: string[] = [],
  sourceShapeCounts: Record<string, number> = {},
): MergeAssessment {
  const sourceCategories = unique(sourceCategoryNames);
  const trainCategories = unique(trainCategoryNames);
  const valCategories = unique(valCategoryNames);
  const cocoCategories = unique([...trainCategories, ...valCategories]);
  const onlyInLabelme = difference(sourceCategories, cocoCategories);
  const matched = sourceCategories.filter((name) => cocoCategories.includes(name));
  const missingFromTrain = difference(matched, trainCategories);
  const missingFromVal = difference(matched, valCategories);
  const requested = new Set(addCategories.map((name) => name.trim()));
  const addedCategories = onlyInLabelme.filter((name) => requested.has(name));
  const droppedCategories = onlyInLabelme.filter((name) => !requested.has(name));
  const trainAdd = Math.floor(validSamples * trainRatio);
  const valAdd = Math.floor(validSamples * valRatio);
  return {
    canMerge: validSamples > 0 && missingFromTrain.length === 0 && missingFromVal.length === 0,
    cocoCategories,
    differences: {
      matched,
      onlyInLabelme,
      onlyInCoco: difference(cocoCategories, sourceCategories),
      missingFromTrain,
      missingFromVal,
    },
    planned: {
      trainAdd,
      valAdd,
      skipped: validSamples - trainAdd - valAdd,
      trainImages: before.trainImages + trainAdd,
      valImages: before.valImages + valAdd,
      addedCategories,
      droppedCategories,
      droppedAnnotations: droppedCategories.reduce((sum, name) => sum + (sourceShapeCounts[name] || 0), 0),
    },
  };
}

/**
 * Append `names` as new categories to every document, with the same id in each
 * so train and val stay aligned. Ids continue after the largest id used by any
 * document. Returns the assigned ids. Mutates the documents.
 */
export function appendCategories(
  documents: Array<{ categories: Array<{ id: number; name: string; supercategory?: string }> }>,
  names: string[],
): Map<string, number> {
  const assigned = new Map<string, number>();
  let nextId = documents
    .flatMap((document) => document.categories.map((category) => Number(category.id) || 0))
    .reduce((maximum, id) => Math.max(maximum, id), 0) + 1;
  for (const name of unique(names)) {
    const id = nextId++;
    assigned.set(name, id);
    for (const document of documents) {
      if (!document.categories.some((category) => category.name.trim() === name)) {
        document.categories.push({ id, name, supercategory: 'object' });
      }
    }
  }
  return assigned;
}

export function convertLabelmeShapes(
  shapes: MergeShape[],
  categoryIds: Map<string, number>,
  imageId: number,
  firstAnnotationId: number,
): MergeAnnotation[] {
  const annotations: MergeAnnotation[] = [];
  let annotationId = firstAnnotationId;
  for (const shape of shapes || []) {
    const categoryId = categoryIds.get(shape.label?.trim());
    const points = shape.points;
    if (categoryId === undefined || !Array.isArray(points) || points.length < 2) continue;
    const xs = points.map((point) => Number(point[0]));
    const ys = points.map((point) => Number(point[1]));
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const width = maxX - minX;
    const height = maxY - minY;
    if (![minX, maxX, minY, maxY, width, height].every(Number.isFinite) || width <= 1e-5 || height <= 1e-5) continue;
    let area = 0;
    for (let i = 0; i < points.length; i++) {
      const j = (i + 1) % points.length;
      area += Number(points[i][0]) * Number(points[j][1]);
      area -= Number(points[j][0]) * Number(points[i][1]);
    }
    area = Math.abs(area) / 2 || width * height;
    const segmentation = shape.shape_type === "polygon" && points.length >= 3
      ? points.flat().map(Number)
      : [minX, minY, maxX, minY, maxX, maxY, minX, maxY];
    annotations.push({ id: annotationId++, image_id: imageId, category_id: categoryId, bbox: [minX, minY, width, height], area, iscrowd: 0, segmentation: [segmentation] });
  }
  return annotations;
}
