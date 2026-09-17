/**
 * Log-parser regression checks.
 *
 *   npx tsx scripts/check-log-parsers.ts
 *
 * Covers the three stdout dialects the platform reads (`src/lib/log-parsers/`)
 * against the exact byte shapes their emitters produce — PaddleSeg's
 * `paddleseg/core/val.py`, PaddleDetection + pycocotools, and
 * `torchtrain/torchtrain/{logger,ad/logger}.py`.
 *
 * Cases worth keeping in mind when editing these parsers:
 *
 *   [2]  PaddleSeg prints per-class arrays with `str(np.round(...))`, and numpy
 *        wraps at `linewidth=75`. From ~10 classes up the array therefore spans
 *        several lines. Treating the first fragment as "not an array" used to
 *        close the EVAL block early, which silently dropped every per-class
 *        metric *and* the trailing best-checkpoint line (so `bestIter` /
 *        `bestMetric` stayed null for exactly the multi-class jobs that need
 *        them).
 *   [2b] ...while still not swallowing an unrelated log line that follows an
 *        unterminated array.
 *   [3]  Node hands stdout over in arbitrary chunks, so every parser must
 *        survive a split at any byte offset.
 *   [5b] Instance-segmentation runs print a `*segm*` COCO block after `*bbox*`;
 *        only the bbox one may feed the chart.
 *   [6]  For iteration-counting frameworks the progress column is `iteration`,
 *        not `epoch` (see `tracksIterations`).
 */
import { createParserState, feed, flush, type ParsedTrainLog } from "../src/lib/log-parsers";

let failures = 0;
function check(name: string, ok: boolean, detail?: unknown) {
  if (ok) {
    console.log(`  PASS  ${name}`);
  } else {
    failures++;
    console.log(`  FAIL  ${name}`);
    if (detail !== undefined) console.log(`        ${JSON.stringify(detail)}`);
  }
}

const TS = "[2026/07/20 11:08:59] INFO: ";

// --------------------------------------------------------------------------
console.log("\n[1] PaddleSeg EVAL block, 3 classes (array fits on one line)");
{
  const st = createParserState("PaddleSeg");
  const rows = feed(
    st,
    [
      `${TS}[TRAIN] epoch: 2278, iter: 20500/160000, loss: 0.5440, lr: 0.004445, batch_cost: 0.1866, reader_cost: 0.06624, ips: 21.4352 samples/sec, max_mem_reserved: 3988 MB, max_mem_allocated: 3542 MB | ETA 07:13:51`,
      `${TS}[EVAL] #Images: 10 mIoU: 0.5696 Acc: 0.9900 Kappa: 0.4986 Dice: 0.6021`,
      `${TS}[EVAL] Class IoU: `,
      `[0.9952 0.3447 0.6496]`,
      `${TS}[EVAL] Class Precision: `,
      `[0.9976 0.4916 0.764 ]`,
      `${TS}[EVAL] Class Recall: `,
      `[0.9976 0.5358 0.8128]`,
      `${TS}[EVAL] The model with the best validation mIoU (0.6675) was saved at iter 5000.`,
      ``,
    ].join("\n"),
  );
  const evalRow = rows.find((r) => r.kind === "eval");
  check("train row parsed", rows.some((r) => r.kind === "train" && r.iteration === 20500));
  check("eval row emitted", !!evalRow);
  check("eval carries mIoU", evalRow?.mIoU === 0.5696, evalRow?.mIoU);
  check("eval carries classIoU (3)", evalRow?.classIoU?.length === 3, evalRow?.classIoU);
  check("eval carries bestMetric", evalRow?.bestMetric === 0.6675, evalRow?.bestMetric);
  check("eval carries bestIter", evalRow?.bestIter === 5000, evalRow?.bestIter);
  check("eval x-axis = last train iter", evalRow?.iteration === 20500, evalRow?.iteration);
}

// --------------------------------------------------------------------------
console.log("\n[2] PaddleSeg EVAL block, 12 classes (numpy WRAPS the array)");
{
  // This is verbatim what `str(np.round(class_iou, 4))` prints for 12 classes:
  // numpy's default linewidth is 75, so the array spans two physical lines.
  const st = createParserState("PaddleSeg");
  const rows = feed(
    st,
    [
      `${TS}[TRAIN] epoch: 100, iter: 5000/160000, loss: 0.5440, lr: 0.004445, batch_cost: 0.1866, reader_cost: 0.06624, ips: 21.4352 samples/sec | ETA 07:13:51`,
      `${TS}[EVAL] #Images: 10 mIoU: 0.5696 Acc: 0.9900 Kappa: 0.4986 Dice: 0.6021`,
      `${TS}[EVAL] Class IoU: `,
      `[0.9952 0.3447 0.6496 0.2887 0.1234 0.5678 0.9012 0.3456 0.789  0.1234`,
      ` 0.4567 0.6789]`,
      `${TS}[EVAL] Class Precision: `,
      `[0.9976 0.4916 0.764  0.4805 0.2345 0.6789 0.9123 0.4567 0.8901 0.2345`,
      ` 0.5678 0.789 ]`,
      `${TS}[EVAL] Class Recall: `,
      `[0.9976 0.5358 0.8128 0.4198 0.3456 0.789  0.9234 0.5678 0.9012 0.3456`,
      ` 0.6789 0.8901]`,
      `${TS}[EVAL] The model with the best validation mIoU (0.6675) was saved at iter 5000.`,
      ``,
    ].join("\n"),
  );
  const evalRows = rows.filter((r) => r.kind === "eval");
  console.log(`        -> ${evalRows.length} eval row(s) emitted`);
  check("exactly one eval row", evalRows.length === 1, evalRows.length);
  const e = evalRows[0];
  check("eval carries mIoU", e?.mIoU === 0.5696, e?.mIoU);
  check("eval carries classIoU (12)", e?.classIoU?.length === 12, e?.classIoU);
  check("eval carries classPrecision (12)", e?.classPrecision?.length === 12, e?.classPrecision);
  check("eval carries classRecall (12)", e?.classRecall?.length === 12, e?.classRecall);
  check("eval carries bestMetric", e?.bestMetric === 0.6675, e?.bestMetric);
  check("eval carries bestIter", e?.bestIter === 5000, e?.bestIter);
}

// --------------------------------------------------------------------------
console.log("\n[2b] Wrapped array must not swallow a real log line");
{
  // If the closing `]` never arrives (truncated log), the parser must still
  // close the block and must NOT eat the following INFO line.
  const st = createParserState("PaddleSeg");
  const rows = feed(
    st,
    [
      `${TS}[TRAIN] epoch: 1, iter: 100/1000, loss: 0.5, lr: 0.001 | ETA 00:13:51`,
      `${TS}[EVAL] #Images: 10 mIoU: 0.5696 Acc: 0.99 Kappa: 0.49 Dice: 0.60`,
      `${TS}[EVAL] Class IoU: `,
      `[0.9952 0.3447 0.6496 0.2887 0.1234 0.5678 0.9012 0.3456 0.789  0.1234`,
      `${TS}Saving model to output/best_model`,
      `${TS}[TRAIN] epoch: 1, iter: 200/1000, loss: 0.4, lr: 0.001 | ETA 00:12:00`,
      ``,
    ].join("\n"),
  );
  const evals = rows.filter((r) => r.kind === "eval");
  const trains = rows.filter((r) => r.kind === "train");
  check("block still closed exactly once", evals.length === 1, evals.length);
  check("eval keeps its mIoU", evals[0]?.mIoU === 0.5696, evals[0]?.mIoU);
  check("following TRAIN line not swallowed", trains.length === 2, trains.map((t) => t.iteration));
}

// --------------------------------------------------------------------------
console.log("\n[2c] 3-way wrap (30 classes, array spans 3 lines)");
{
  const st = createParserState("PaddleSeg");
  const vals = Array.from({ length: 30 }, (_, i) => (0.5 + i / 100).toFixed(4));
  const wrap = (v: string[]) =>
    [`[${v.slice(0, 10).join(" ")}`, ` ${v.slice(10, 20).join(" ")}`, ` ${v.slice(20).join(" ")}]`];
  const rows = feed(
    st,
    [
      `${TS}[TRAIN] epoch: 1, iter: 100/1000, loss: 0.5, lr: 0.001 | ETA 00:13:51`,
      `${TS}[EVAL] #Images: 10 mIoU: 0.5696 Acc: 0.99 Kappa: 0.49 Dice: 0.60`,
      `${TS}[EVAL] Class IoU: `,
      ...wrap(vals),
      `${TS}[EVAL] Class Precision: `,
      ...wrap(vals),
      `${TS}[EVAL] Class Recall: `,
      ...wrap(vals),
      `${TS}[EVAL] The model with the best validation mIoU (0.6675) was saved at iter 5000.`,
      ``,
    ].join("\n"),
  );
  const e = rows.find((r) => r.kind === "eval");
  check("classIoU has 30 entries", e?.classIoU?.length === 30, e?.classIoU?.length);
  check("classPrecision has 30 entries", e?.classPrecision?.length === 30, e?.classPrecision?.length);
  check("classRecall has 30 entries", e?.classRecall?.length === 30, e?.classRecall?.length);
  check("first/last values intact", e?.classIoU?.[0] === 0.5 && e?.classIoU?.[29] === 0.79, [
    e?.classIoU?.[0],
    e?.classIoU?.[29],
  ]);
  check("bestIter still parsed", e?.bestIter === 5000, e?.bestIter);
}

// --------------------------------------------------------------------------
console.log("\n[3] Chunk-boundary robustness: same 3-class block, 1 char at a time");
{
  const full = [
    `${TS}[TRAIN] epoch: 1, iter: 100/1000, loss: 0.5, lr: 0.001, batch_cost: 0.1, reader_cost: 0.01, ips: 21.4 samples/sec | ETA 00:13:51`,
    `${TS}[EVAL] #Images: 10 mIoU: 0.5696 Acc: 0.9900 Kappa: 0.4986 Dice: 0.6021`,
    `${TS}[EVAL] Class IoU: `,
    `[0.9952 0.3447 0.6496]`,
    `${TS}[EVAL] Class Precision: `,
    `[0.9976 0.4916 0.764 ]`,
    `${TS}[EVAL] Class Recall: `,
    `[0.9976 0.5358 0.8128]`,
    `${TS}[EVAL] The model with the best validation mIoU (0.6675) was saved at iter 5000.`,
    ``,
  ].join("\n");
  const st = createParserState("PaddleSeg");
  const rows: ParsedTrainLog[] = [];
  for (const ch of full) rows.push(...feed(st, ch));
  rows.push(...flush(st));
  const e = rows.find((r) => r.kind === "eval");
  check("eval row survives byte-wise chunking", !!e && e.classIoU?.length === 3 && e.bestIter === 5000, {
    classIoU: e?.classIoU,
    bestIter: e?.bestIter,
  });
}

// --------------------------------------------------------------------------
console.log("\n[4] PaddleSeg: eval block closed by flush() at process exit");
{
  const st = createParserState("PaddleSeg");
  // No trailing newline on the last line, and no closing "best mIoU" line:
  // exactly what an aborted run looks like.
  feed(
    st,
    [
      `${TS}[TRAIN] epoch: 1, iter: 100/1000, loss: 0.5, lr: 0.001 | ETA 00:13:51`,
      `${TS}[EVAL] #Images: 10 mIoU: 0.7777 Acc: 0.99 Kappa: 0.5 Dice: 0.6`,
      ``,
    ].join("\n"),
  );
  const tail = flush(st);
  check("flush emits the dangling eval row", tail.some((r) => r.mIoU === 0.7777), tail);
}

// --------------------------------------------------------------------------
console.log("\n[5] PaddleDetection TRAIN + full 12-row COCO eval block");
{
  const st = createParserState("PaddleDetection");
  const cocoBlock = [
    `Evaluate annotation type *bbox*`,
    ` Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.512`,
    ` Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.803`,
    ` Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.561`,
    ` Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.201`,
    ` Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.433`,
    ` Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.655`,
    ` Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.400`,
    ` Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.600`,
    ` Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.620`,
    ` Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.300`,
    ` Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.550`,
    ` Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.750`,
  ];
  const rows = feed(
    st,
    [
      `Epoch: [8] [60/79] learning_rate: 0.000996 loss: 4.193813 loss_cls: 1.671748 loss_iou: 0.591968 loss_dfl: 0.899393 loss_l1: 1.030704 eta: 0:02:24 batch_cost: 0.3400 data_cost: 0.0100 ips: 23.5294 images/s`,
      ...cocoBlock,
      `Best test bbox ap is 0.512.`,
      ``,
    ].join("\n"),
  );
  check("detection train row", rows.some((r) => r.loss === 4.193813 && r.lossCls === 1.671748));
  const m = rows.find((r) => r.mAP != null);
  check("detection mAP parsed", m?.mAP === 0.512, m?.mAP);
  check("detection mAP50 parsed", m?.mAP50 === 0.803, m?.mAP50);
  check("detection eta parsed", rows.find((r) => r.kind === "train")?.eta === "0:02:24");
  const best = rows.find((r) => r.bestMetric != null);
  check("detection bestMetric parsed", best?.bestMetric === 0.512, best?.bestMetric);
}

// --------------------------------------------------------------------------
console.log("\n[5b] Detection: instance-seg *segm* block must not overwrite bbox mAP");
{
  const st = createParserState("PaddleDetection");
  const rows = feed(
    st,
    [
      `Epoch: [1] [10/79] learning_rate: 0.001 loss: 2.0 eta: 0:02:24`,
      `Evaluate annotation type *bbox*`,
      ...Array.from({ length: 12 }, (_, i) =>
        i === 0
          ? ` Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.500`
          : ` Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=${i} ] = 0.4`,
      ),
      `Evaluate annotation type *segm*`,
      ` Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.111`,
      ``,
    ].join("\n"),
  );
  const evals = rows.filter((r) => r.mAP != null);
  check("exactly one mAP row (bbox only)", evals.length === 1, evals.map((e) => e.mAP));
  check("mAP is the bbox one", evals[0]?.mAP === 0.5, evals[0]?.mAP);
}

// --------------------------------------------------------------------------
console.log("\n[5c] Detection: COCO -1 (no GT in bucket) must not be charted");
{
  const st = createParserState("PaddleDetection");
  const rows = feed(
    st,
    [
      `Evaluate annotation type *bbox*`,
      ` Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = -1.000`,
      ` Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = -1.000`,
      ...Array.from({ length: 10 }, () => ` Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000`),
      ``,
    ].join("\n"),
  );
  const e = rows.find((r) => r.kind === "eval");
  check("-1 mAP stored as null", e != null && e.mAP == null && e.mAP50 == null, { mAP: e?.mAP, mAP50: e?.mAP50 });
}

// --------------------------------------------------------------------------
console.log("\n[5d] TorchAnomaly EVAL block");
{
  // Exactly what `PaddleStyleLogger.emit_eval` + the best-model line print
  // (torchtrain/torchtrain/ad/logger.py).
  const st = createParserState("TorchAnomaly");
  const rows = feed(
    st,
    [
      `${TS}[TRAIN] epoch: 3, iter: 600/8000, loss: 0.1837, lr: 0.000100, batch_cost: 0.0921, reader_cost: 0.01130, ips: 86.8000 samples/sec, max_mem_reserved: 2048 MB, max_mem_allocated: 1902 MB | ETA 00:11:22`,
      `${TS}[EVAL] #Images: 40 image_auroc: 0.9812 image_f1: 0.9231 pixel_auroc: 0.9633 pixel_f1: 0.5412 threshold: 12.3456 aupro: 0.8800`,
      `${TS}[EVAL] The model with the best validation image_auroc (0.9812) was saved at iter 3000.`,
      ``,
    ].join("\n"),
  );
  const e = rows.find((r) => r.kind === "eval");
  check("anomaly train row iter=600", rows[0]?.iteration === 600, rows[0]?.iteration);
  check("anomaly eval row produced", !!e);
  check("image AUROC parsed", e?.imageAuroc === 0.9812, e?.imageAuroc);
  check("pixel AUROC parsed", e?.pixelAuroc === 0.9633, e?.pixelAuroc);
  check("threshold parsed", e?.threshold === 12.3456, e?.threshold);
  check("bestMetric parsed", e?.bestMetric === 0.9812, e?.bestMetric);
  check("bestIter parsed", e?.bestIter === 3000, e?.bestIter);
  check("unknown metric kept in extraMetrics", e?.extraMetrics?.aupro === 0.88, e?.extraMetrics);
  check("eval x-axis inherits train iter", e?.iteration === 600, e?.iteration);
}

// --------------------------------------------------------------------------
console.log("\n[6] tracksIterations semantics for currentEpoch");
{
  // TorchSeg emits PaddleSeg-shaped lines; make sure iteration is what the
  // progress bar would read.
  const st = createParserState("TorchSeg");
  const rows = feed(
    st,
    `${TS}[TRAIN] epoch: 2278, iter: 20500/160000, loss: 0.544, lr: 0.004 | ETA 07:13:51\n`,
  );
  check("TorchSeg train row iteration=20500", rows[0]?.iteration === 20500, rows[0]?.iteration);
  check("TorchSeg train row totalIter=160000", rows[0]?.totalIter === 160000, rows[0]?.totalIter);
}

console.log(`\n${failures === 0 ? "ALL PASS" : `${failures} CHECK(S) FAILED`}\n`);
process.exit(failures === 0 ? 0 : 1);
