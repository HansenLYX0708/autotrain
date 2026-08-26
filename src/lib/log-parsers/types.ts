/**
 * Shared types for framework-specific training log parsers.
 *
 * All parsers emit `ParsedTrainLog` records. Framework-specific fields are
 * optional; consumers should treat absent fields as "not applicable to this
 * framework" rather than "missing data". Downstream DB writer (`route.ts`)
 * maps this shape onto the Prisma `TrainingLog` model.
 */

export type ParsedLogKind = 'train' | 'eval'

export interface ParsedTrainLog {
  /** 'train' for per-iteration lines, 'eval' for validation summary rows. */
  kind: ParsedLogKind

  /** Progress fields (zero when the framework doesn't emit them). */
  epoch: number
  iteration: number
  totalIter: number

  /** Common metrics across frameworks. */
  loss: number | null
  learningRate: number | null
  eta: string | null
  batchCost: number | null
  dataCost: number | null
  readerCost: number | null
  ips: number | null
  memReserved: number | null
  memAllocated: number | null

  // ---- Detection-specific loss decomposition ---------------------------
  lossCls?: number | null
  lossIou?: number | null
  lossDfl?: number | null
  lossL1?: number | null

  // ---- Detection eval metrics (COCO) -----------------------------------
  /** AP @[IoU=0.50:0.95 | area=all | maxDets=100]. */
  mAP?: number | null
  /** AP @[IoU=0.50 | area=all | maxDets=100]. */
  mAP50?: number | null

  // ---- Segmentation eval metrics ---------------------------------------
  mIoU?: number | null
  acc?: number | null
  kappa?: number | null
  dice?: number | null
  /** Per-class arrays, arity = num_classes (incl. background). */
  classIoU?: number[]
  classPrecision?: number[]
  classRecall?: number[]
  // ---- Anomaly-detection eval metrics (TorchAnomaly) -------------------
  /** Image-level AUROC: can the model tell a defective part from a good one. */
  imageAuroc?: number | null
  imageF1?: number | null
  /** Pixel-level AUROC/F1: only present when the eval set carries masks. */
  pixelAuroc?: number | null
  pixelF1?: number | null
  /** Anomaly-score cut-off chosen by anomalib's adaptive thresholding. */
  threshold?: number | null
  /**
   * Any further `key: value` pairs the trainer printed on the EVAL line.
   *
   * anomalib names its metrics dynamically (an `Evaluator` can be given any
   * `AnomalibMetric`), so the parser must not silently drop a metric it has no
   * column for. These are persisted as JSON alongside the row instead.
   */
  extraMetrics?: Record<string, number>

  /** From "best mIoU (X) was saved at iter N". */
  bestIter?: number | null
  bestMetric?: number | null
  /** Which metric the "best" line reported, e.g. `mIoU` or `image_auroc`. */
  bestMetricName?: string | null

  /** Original line(s) that produced this record. */
  rawLog: string
}
